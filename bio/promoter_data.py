from __future__ import annotations

import csv
import gzip
from bisect import bisect_left
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_CHROMS = {
    "valid": {"chr10"},
    "test": {"chr8", "chr9"},
}


def read_fai(path: Path):
    records = {}
    with path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            records[fields[0]] = tuple(map(int, fields[1:5]))
    return records


def load_tss_rows(path: Path, split: str, first_n: int = 100_000):
    with path.open("rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))[:first_n]
    held_out = SPLIT_CHROMS["valid"] | SPLIT_CHROMS["test"]
    if split == "train":
        rows = [row for row in rows if row["chr"] not in held_out]
    elif split in SPLIT_CHROMS:
        rows = [row for row in rows if row["chr"] in SPLIT_CHROMS[split]]
    else:
        raise ValueError(f"Unknown split: {split}")
    return rows


class PromoterDataset(Dataset):
    """FANTOM/GRCh38 promoter windows with the upstream chromosome split.

    The common smoke benchmark deterministically selects windows containing only
    unambiguous A/C/G/T positions. This avoids silently converting N to A for the
    categorical methods. Full paper reproduction uses the complete upstream loader.
    """

    def __init__(self, root: str | Path, split: str, max_samples: int, seed: int = 42,
                 sequence_length: int = 1024):
        self.root = Path(root)
        self.split = split
        self.sequence_length = sequence_length
        fasta = self.root / "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
        self.fai = read_fai(Path(str(fasta) + ".fai"))
        self.chromosomes = sorted(self.fai)
        self.chrom_offsets = {}
        offset = 0
        for chrom in self.chromosomes:
            self.chrom_offsets[chrom] = offset
            offset += self.fai[chrom][0]
        self.total_bases = offset
        self.genome = np.memmap(
            str(fasta) + ".mmap", dtype=np.float32, mode="r", shape=(4, self.total_bases)
        )
        tss_path = self.root / "FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv"
        all_rows = load_tss_rows(tss_path, split)
        self.total_split_rows = len(all_rows)
        order = np.random.default_rng(seed).permutation(len(all_rows))
        selected = []
        ambiguous = 0
        half = sequence_length // 2
        for row_index in order:
            row = all_rows[int(row_index)]
            chrom = row["chr"]
            position = int(row["TSS"])
            strand = row["strand"]
            strand_offset = 1 if strand == "-" else 0
            start = position - half + strand_offset
            end = position + half + strand_offset
            if chrom not in self.chrom_offsets or start < 0 or end > self.fai[chrom][0]:
                continue
            seq = self._sequence(chrom, start, end, strand)
            if not np.all(seq.max(axis=1) == 1.0):
                ambiguous += 1
                continue
            selected.append((row, start, end))
            if len(selected) >= max_samples:
                break
        if len(selected) < max_samples:
            raise RuntimeError(
                f"Only {len(selected)} unambiguous {split} windows found, requested {max_samples}"
            )
        self.rows = selected
        self.ambiguous_windows_skipped = ambiguous
        self._bigwig = None
        self._blacklists = None

    def _sequence(self, chrom: str, start: int, end: int, strand: str):
        global_start = self.chrom_offsets[chrom] + start
        seq = np.asarray(self.genome[:, global_start:global_start + (end - start)]).T.copy()
        if strand == "-":
            seq = seq[::-1, ::-1].copy()
        return seq

    def _open_signal(self):
        if self._bigwig is not None:
            return
        import pyBigWig
        self._bigwig = [
            pyBigWig.open(str(self.root / "agg.plus.bw.bedgraph.bw")),
            pyBigWig.open(str(self.root / "agg.minus.bw.bedgraph.bw")),
        ]
        intervals = {}
        for filename in ("fantom.blacklist8.plus.bed.gz", "fantom.blacklist8.minus.bed.gz"):
            with gzip.open(self.root / filename, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line or line.startswith("#"):
                        continue
                    chrom, start, end, *_ = line.rstrip("\n").split("\t")
                    intervals.setdefault(chrom, []).append((int(start), int(end)))
        self._blacklists = {}
        for chrom, values in intervals.items():
            merged = []
            for start, end in sorted(values):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            self._blacklists[chrom] = (
                [start for start, _ in merged], [end for _, end in merged]
            )

    def _signal(self, chrom: str, start: int, end: int, strand: str):
        self._open_signal()
        signal = np.stack([
            handle.values(chrom, start, end, numpy=True) for handle in self._bigwig
        ]).astype(np.float32)
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        starts, ends = self._blacklists.get(chrom, ([], []))
        index = bisect_left(ends, start)
        while index < len(starts) and starts[index] < end:
            left = max(starts[index] - start, 0)
            right = min(ends[index] - start, end - start)
            signal[:, left:right] = 0
            index += 1
        if strand == "-":
            signal = signal[::-1, ::-1].copy()
        # The upstream models condition on channel 0 after strand orientation.
        return signal[0, :, None]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row, start, end = self.rows[index]
        seq = self._sequence(row["chr"], start, end, row["strand"])
        signal = self._signal(row["chr"], start, end, row["strand"])
        tokens = seq.argmax(axis=-1).astype(np.int64)
        return {
            "tokens": torch.from_numpy(tokens),
            "one_hot": torch.from_numpy(seq.astype(np.float32)),
            "signal": torch.from_numpy(signal),
            "chrom": row["chr"],
            "strand": row["strand"],
            "tss": int(row["TSS"]),
        }

    def audit(self):
        return {
            "split": self.split,
            "total_split_rows_first_100000": self.total_split_rows,
            "selected_unambiguous_rows": len(self.rows),
            "ambiguous_windows_skipped_during_selection": self.ambiguous_windows_skipped,
            "sequence_length": self.sequence_length,
            "chromosomes": sorted({row[0]["chr"] for row in self.rows}),
        }
