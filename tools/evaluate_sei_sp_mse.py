#!/usr/bin/env python3
"""Evaluate promoter FASTA files with the paper's strand-invariant Sei SP-MSE."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from bio.promoter_data import PromoterDataset


BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


class NonStrandSpecific(nn.Module):
    """Selene-compatible mean of forward and reverse-complement predictions."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, values):
        reverse_complement = torch.flip(values, dims=(1, 2))
        return (self.model(values) + self.model(reverse_complement)) / 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ddsm-upstream", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fasta(path):
    sequences = []
    current = []
    with path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip().upper()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current))
    if not sequences:
        raise RuntimeError(f"No sequences found in {path}")
    lengths = {len(sequence) for sequence in sequences}
    invalid = sorted({base for sequence in sequences for base in sequence} - set(BASE_TO_INDEX))
    if len(lengths) != 1 or invalid:
        raise RuntimeError(f"Invalid FASTA {path}: lengths={lengths}, invalid_bases={invalid}")
    return np.asarray([
        [BASE_TO_INDEX[base] for base in sequence] for sequence in sequences
    ], dtype=np.int64)


def load_sei_class(upstream):
    module_path = upstream / "external" / "sei.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Official Sei implementation not found: {module_path}")
    specification = importlib.util.spec_from_file_location("ddsm_official_sei", module_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.Sei


def load_feature_mask(path):
    labels = []
    with path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("|")
            if len(fields) < 2:
                raise RuntimeError(f"Malformed Sei feature row: {line!r}")
            labels.append(fields[1].strip())
    mask = np.asarray([label == "H3K4me3" for label in labels], dtype=bool)
    if len(labels) != 21_907 or not mask.any():
        raise RuntimeError(
            f"Unexpected feature manifest: total={len(labels)}, H3K4me3={int(mask.sum())}"
        )
    return torch.from_numpy(mask), int(mask.sum())


def load_sei_model(args, device):
    Sei = load_sei_class(args.ddsm_upstream)
    model = NonStrandSpecific(Sei(4096, 21_907))
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    state = {
        (name[len("module."):] if name.startswith("module.") else name): value
        for name, value in state.items()
    }
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Incompatible Sei checkpoint: {incompatible}")
    model.eval().to(device)
    return model


def sei_profiles(model, tokens, feature_mask, batch_size, device):
    profiles = []
    feature_mask = feature_mask.to(device)
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            batch = torch.from_numpy(tokens[start:start + batch_size]).to(device)
            one_hot = F.one_hot(batch, num_classes=4).float().transpose(1, 2)
            flank = torch.full(
                (len(batch), 4, 1536), 0.25, dtype=one_hot.dtype, device=device
            )
            predictions = model(torch.cat([flank, one_hot, flank], dim=2))
            profiles.append(predictions[:, feature_mask].mean(dim=1).cpu())
            print(f"sei_profiles={min(start + batch_size, len(tokens))}/{len(tokens)}", flush=True)
    return torch.cat(profiles).numpy().astype(np.float64)


def target_tokens(data_dir, sample_count, valid_samples, seed, sequence_length):
    dataset = PromoterDataset(
        data_dir, "valid", valid_samples, seed + 1, sequence_length
    )
    rows = []
    for row, start, end in dataset.rows[:sample_count]:
        sequence = dataset._sequence(row["chr"], start, end, row["strand"])
        rows.append(sequence.argmax(axis=-1))
    return np.stack(rows).astype(np.int64)


def confidence_interval(values, replicates, seed=137):
    if replicates <= 0:
        return [None, None]
    generator = np.random.default_rng(seed)
    sample_indices = generator.integers(0, len(values), size=(replicates, len(values)))
    bootstrap_means = values[sample_indices].mean(axis=1)
    return np.quantile(bootstrap_means, [0.025, 0.975]).tolist()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    run_directories = sorted({path.parent for path in args.run_root.glob("**/samples.fasta")})
    if not run_directories:
        raise RuntimeError(f"No samples.fasta below {args.run_root}")

    feature_mask, feature_count = load_feature_mask(args.features)
    print(f"H3K4me3_features={feature_count}", flush=True)
    model = load_sei_model(args, device)
    weights_hash = sha256(args.weights)
    features_hash = sha256(args.features)
    target_cache = {}
    rows = []

    for run_directory in run_directories:
        summary_path = run_directory / "summary.json"
        checkpoint_path = run_directory / "last_checkpoint.pt"
        if not summary_path.is_file() or not checkpoint_path.is_file():
            raise RuntimeError(f"Incomplete run directory: {run_directory}")
        summary = json.loads(summary_path.read_text())
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        training_args = checkpoint["args"]
        generated = load_fasta(run_directory / "samples.fasta")
        cache_key = (
            int(training_args["seed"]), int(training_args["valid_samples"]),
            int(training_args["sequence_length"]), len(generated),
        )
        if cache_key not in target_cache:
            targets = target_tokens(
                args.data_dir, len(generated), int(training_args["valid_samples"]),
                int(training_args["seed"]), int(training_args["sequence_length"]),
            )
            target_cache[cache_key] = (
                targets,
                sei_profiles(model, targets, feature_mask, args.batch_size, device),
            )
        targets, target_profiles = target_cache[cache_key]
        if generated.shape != targets.shape:
            raise RuntimeError(
                f"Generated/target shape mismatch for {run_directory}: "
                f"{generated.shape} vs {targets.shape}"
            )
        print(f"evaluating={run_directory}", flush=True)
        generated_profiles = sei_profiles(
            model, generated, feature_mask, args.batch_size, device
        )
        squared_errors = (target_profiles - generated_profiles) ** 2
        result = {
            "method": summary["method"],
            "seed": int(training_args["seed"]),
            "sample_count": len(generated),
            "sequence_length": int(generated.shape[1]),
            "sp_mse": float(squared_errors.mean()),
            "sp_mse_95pct_bootstrap_ci": confidence_interval(
                squared_errors, args.bootstrap_replicates
            ),
            "sp_squared_error_std": float(squared_errors.std(ddof=1)),
            "sp_mae": float(np.abs(target_profiles - generated_profiles).mean()),
            "target_profile_mean": float(target_profiles.mean()),
            "generated_profile_mean": float(generated_profiles.mean()),
            "h3k4me3_feature_count": feature_count,
            "sei_weights_sha256": weights_hash,
            "sei_features_sha256": features_hash,
            "status": "PASS",
        }
        atomic_json(run_directory / "sei_metrics.json", result)
        rows.append(result | {"run_directory": str(run_directory)})
        print(json.dumps(result, sort_keys=True), flush=True)

    atomic_json(args.run_root / "sei_summary.json", rows)
    table_path = args.run_root / "sei_summary.tsv"
    with table_path.open("wt", encoding="utf-8") as handle:
        handle.write("method\tseed\tsamples\tSP_MSE\tCI_low\tCI_high\tSP_MAE\n")
        for row in rows:
            low, high = row["sp_mse_95pct_bootstrap_ci"]
            handle.write(
                f'{row["method"]}\t{row["seed"]}\t{row["sample_count"]}\t'
                f'{row["sp_mse"]:.8g}\t{low:.8g}\t{high:.8g}\t{row["sp_mae"]:.8g}\n'
            )
    print(f"saved={args.run_root / 'sei_summary.json'}", flush=True)
    print(table_path.read_text(), flush=True)


if __name__ == "__main__":
    main()
