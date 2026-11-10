#!/usr/bin/env python3
"""Build the byte-compatible Selene MemmapGenome array with bounded RAM.

Selene stores a float32 C-order array shaped (4, sum(chromosome lengths)).
Chromosomes are concatenated in lexicographic order, channels are A/C/G/T,
and every unrecognized base is represented by 0.25 in all four channels.
Progress is committed after each chromosome so a 30-minute job can resume.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def read_fai(path: Path):
    records = {}
    with path.open("rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                raise ValueError(f"Malformed FAI line {line_number}: {line!r}")
            name = fields[0]
            if name in records:
                raise ValueError(f"Duplicate FAI sequence: {name}")
            records[name] = tuple(map(int, fields[1:5]))
    if not records:
        raise ValueError("FAI is empty")
    return records


def read_fasta_range(handle, record, start: int, length: int) -> bytes:
    seq_len, seq_offset, line_bases, line_width = record
    if start < 0 or length < 0 or start + length > seq_len:
        raise ValueError((start, length, seq_len))
    parts = []
    position = start
    remaining = length
    while remaining:
        line_index, within = divmod(position, line_bases)
        take = min(remaining, line_bases - within)
        handle.seek(seq_offset + line_index * line_width + within)
        block = handle.read(take)
        if len(block) != take:
            raise EOFError(f"FASTA ended at base {position}")
        parts.append(block)
        position += take
        remaining -= take
    return b"".join(parts)


def encoding_table() -> np.ndarray:
    table = np.full((256, 4), 0.25, dtype=np.float32)
    for index, chars in enumerate((b"Aa", b"Cc", b"Gg", b"Tt")):
        for char in chars:
            table[char] = 0.0
            table[char, index] = 1.0
    return table


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--fai", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-bases", type=int, default=4_000_000)
    args = parser.parse_args()
    if args.chunk_bases <= 0:
        raise ValueError("--chunk-bases must be positive")
    fai = args.fai or Path(str(args.fasta) + ".fai")
    records = read_fai(fai)
    chromosomes = sorted(records)
    total_bases = sum(records[name][0] for name in chromosomes)
    expected_bytes = 4 * total_bases * np.dtype(np.float32).itemsize
    partial = Path(str(args.output) + ".partial")
    progress_path = Path(str(args.output) + ".progress.json")
    metadata = {
        "fasta": str(args.fasta.resolve()),
        "fasta_size": args.fasta.stat().st_size,
        "fai": str(fai.resolve()),
        "chromosomes": chromosomes,
        "lengths": {name: records[name][0] for name in chromosomes},
        "total_bases": total_bases,
        "shape": [4, total_bases],
        "dtype": "float32",
        "base_order": ["A", "C", "G", "T"],
    }
    complete = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["metadata"] != metadata:
            raise RuntimeError("Existing progress metadata does not match current FASTA/FAI")
        complete = progress["complete"]
    elif partial.exists():
        raise RuntimeError(f"{partial} exists without {progress_path}; refusing to guess its state")
    if args.output.exists():
        if args.output.stat().st_size != expected_bytes:
            raise RuntimeError("Existing final memmap has the wrong size")
        print(f"memmap_already_complete={args.output}")
        return
    mode = "r+" if partial.exists() else "w+"
    mmap = np.memmap(partial, dtype=np.float32, mode=mode, shape=(4, total_bases))
    if not progress_path.exists():
        atomic_json(progress_path, {"metadata": metadata, "complete": []})
    table = encoding_table()
    global_offset = 0
    with args.fasta.open("rb") as fasta_handle:
        for name in chromosomes:
            chromosome_length = records[name][0]
            if name in complete:
                global_offset += chromosome_length
                continue
            print(f"chromosome_start={name} length={chromosome_length} offset={global_offset}", flush=True)
            for start in range(0, chromosome_length, args.chunk_bases):
                count = min(args.chunk_bases, chromosome_length - start)
                bases = read_fasta_range(fasta_handle, records[name], start, count)
                encoded = table[np.frombuffer(bases, dtype=np.uint8)]
                mmap[:, global_offset + start:global_offset + start + count] = encoded.T
            mmap.flush()
            # Verify deterministic boundary samples against the source FASTA.
            sample_starts = sorted({0, max(0, chromosome_length // 2 - 128), max(0, chromosome_length - 256)})
            for start in sample_starts:
                count = min(256, chromosome_length - start)
                source = read_fasta_range(fasta_handle, records[name], start, count)
                expected = table[np.frombuffer(source, dtype=np.uint8)].T
                observed = mmap[:, global_offset + start:global_offset + start + count]
                if not np.array_equal(observed, expected):
                    raise RuntimeError(f"Verification failed for {name}:{start}-{start + count}")
            complete.append(name)
            atomic_json(progress_path, {"metadata": metadata, "complete": complete})
            global_offset += chromosome_length
            print(f"chromosome_complete={name}", flush=True)
    mmap.flush()
    del mmap
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError("Completed partial memmap has the wrong size")
    os.replace(partial, args.output)
    final_metadata = Path(str(args.output) + ".metadata.json")
    atomic_json(final_metadata, metadata)
    progress_path.unlink()
    print(f"memmap_complete={args.output}")
    print(f"bytes={expected_bytes} total_bases={total_bases}")


if __name__ == "__main__":
    main()
