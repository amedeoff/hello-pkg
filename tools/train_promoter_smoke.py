#!/usr/bin/env python3
"""Short, common-backbone DNA smoke benchmark for four generative processes.

This validates real promoter data, conditioning, optimization, sampling and
artifacts. It is deliberately small and is not a replacement for paper-scale
training or the official Sei evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bio.model import DDSMScoreAdapter, PromoterCNN
from bio.processes import (
    d3pm_training_batch,
    dirichlet_fm_training_batch,
    discrete_fm_training_batch,
    sample_d3pm,
    sample_dirichlet_fm,
    sample_discrete_fm,
)
from bio.promoter_data import PromoterDataset


BASES = np.asarray(list("ACGT"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["d3pm", "discrete_fm", "ddsm", "dirichlet_fm"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ddsm-upstream", type=Path)
    parser.add_argument("--ddsm-noise", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--valid-samples", type=int, default=128)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--process-steps", type=int, default=32)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--alpha-max", type=float, default=8.0)
    parser.add_argument("--alpha-scale", type=float, default=2.0)
    parser.add_argument("--prior-pseudocount", type=float, default=2.0)
    parser.add_argument("--validation-batches", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--training-wall-seconds", type=float, default=0,
        help="Stop training cleanly after this many seconds; 0 disables the limit.",
    )
    parser.add_argument(
        "--scope", default="short common-backbone promoter smoke test; not paper-scale evaluation"
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def condition_transform(signal):
    # Shared numerical transform for the common smoke benchmark only.
    return torch.log1p(torch.clamp(signal, min=0, max=1e6))


def load_ddsm(args):
    if args.ddsm_upstream is None or args.ddsm_noise is None:
        raise ValueError("DDSM requires --ddsm-upstream and --ddsm-noise")
    sys.path.insert(0, str(args.ddsm_upstream))
    import ddsm
    cache = torch.load(args.ddsm_noise, map_location="cpu", weights_only=False)
    v_one, _, v_one_loggrad, _, timepoints = cache
    return ddsm, v_one.float(), v_one_loggrad.float(), timepoints.float()


def make_model(args, device):
    state_channels = 8 if args.method == "dirichlet_fm" else 4
    model = PromoterCNN(
        state_channels=state_channels,
        channels=args.channels,
        layers=args.layers,
        center_output=args.method == "ddsm",
    ).to(device)
    return model


def loss_for_batch(args, model, batch, device, ddsm_state=None):
    tokens = batch["tokens"].to(device)
    signal = condition_transform(batch["signal"].to(device))
    if args.method == "d3pm":
        state, time_value = d3pm_training_batch(tokens, args.process_steps)
        logits = model(state, signal, time_value)
        return F.cross_entropy(logits.transpose(1, 2), tokens), signal
    if args.method == "discrete_fm":
        state, time_value = discrete_fm_training_batch(tokens)
        logits = model(state, signal, time_value)
        return F.cross_entropy(logits.transpose(1, 2), tokens), signal
    if args.method == "dirichlet_fm":
        state, time_value = dirichlet_fm_training_batch(
            tokens, args.alpha_scale, args.alpha_max, args.prior_pseudocount
        )
        logits = model(state, signal, time_value)
        return F.cross_entropy(logits.transpose(1, 2), tokens), signal
    ddsm, v_one, v_one_loggrad, timepoints = ddsm_state
    clean = batch["one_hot"]
    time_index = torch.randint(0, len(timepoints), (clean.shape[0],))
    perturbed, target_score = ddsm.diffusion_fast_flatdirichlet(
        clean, time_index, v_one, v_one_loggrad
    )
    perturbed = perturbed.to(device)
    target_score = target_score.to(device)
    continuous_time = timepoints[time_index].to(device)
    score = model(perturbed, signal, continuous_time / 4.0)
    stick = ddsm.UnitStickBreakingTransform()._inverse(perturbed, prevent_nan=True).detach()
    speed = 2 / (
        torch.ones(3, device=device) + torch.arange(3, 0, -1, device=device).float()
    )
    predicted_stick_score = ddsm.gx_to_gv(score, perturbed, create_graph=True)
    target_stick_score = ddsm.gx_to_gv(target_score, perturbed)
    loss = (
        speed[None, None, :] * stick * (1 - stick)
        * (predicted_stick_score - target_stick_score).square()
    ).mean()
    return loss, signal


@torch.no_grad()
def generate(args, model, signal, device, ddsm_state=None):
    diagnostics = {}
    if args.method == "d3pm":
        return sample_d3pm(model, signal, args.process_steps), diagnostics
    if args.method == "discrete_fm":
        return sample_discrete_fm(model, signal, args.sample_steps), diagnostics
    if args.method == "dirichlet_fm":
        return sample_dirichlet_fm(
            model, signal, args.sample_steps, args.alpha_max, args.prior_pseudocount
        )
    ddsm = ddsm_state[0]
    adapter = DDSMScoreAdapter(model)
    probabilities = ddsm.Euler_Maruyama_sampler(
        adapter,
        (signal.shape[1], 4),
        batch_size=signal.shape[0],
        max_time=4,
        min_time=4 / args.process_steps,
        time_dilation=1,
        num_steps=args.sample_steps,
        eps=1e-5,
        speed_balanced=True,
        device=str(device),
        concat_input=signal,
    )
    return probabilities.argmax(-1), diagnostics


def base_frequencies(tokens):
    counts = np.bincount(tokens.reshape(-1), minlength=4).astype(np.float64)
    return (counts / counts.sum()).tolist()


def kmer_distribution(tokens, k=3):
    size = 4 ** k
    counts = np.ones(size, dtype=np.float64) * 1e-8
    for sequence in tokens:
        code = np.zeros(len(sequence) - k + 1, dtype=np.int64)
        for offset in range(k):
            code = code * 4 + sequence[offset:offset + len(code)]
        counts += np.bincount(code, minlength=size)
    return counts / counts.sum()


def js_divergence(first, second):
    middle = (first + second) / 2
    return float(0.5 * np.sum(first * np.log(first / middle)) + 0.5 * np.sum(second * np.log(second / middle)))


def evaluate_samples(generated, targets, signal, diagnostics):
    generated = generated.detach().cpu().numpy().astype(np.int64)
    targets = targets.detach().cpu().numpy().astype(np.int64)
    generated_strings = ["".join(BASES[row].tolist()) for row in generated]
    target_kmers = kmer_distribution(targets)
    generated_kmers = kmer_distribution(generated)
    result = {
        "sample_count": len(generated),
        "sequence_length": generated.shape[1],
        "base_frequencies_acgt": base_frequencies(generated),
        "target_base_frequencies_acgt": base_frequencies(targets),
        "gc_fraction": float(np.isin(generated, [1, 2]).mean()),
        "target_gc_fraction": float(np.isin(targets, [1, 2]).mean()),
        "unique_fraction": len(set(generated_strings)) / len(generated_strings),
        "conditioned_target_identity": float((generated == targets).mean()),
        "three_mer_js_nats": js_divergence(generated_kmers, target_kmers),
        "signal_mean": float(signal.mean()),
        "signal_std": float(signal.std()),
    }
    result.update(diagnostics)
    return result, generated_strings


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_checkpoint(path, model, optimizer, args, completed_steps):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "completed_steps": completed_steps,
    }, temporary)
    os.replace(temporary, path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the cluster smoke benchmark")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_data = PromoterDataset(
        args.data_dir, "train", args.train_samples, args.seed, args.sequence_length
    )
    valid_data = PromoterDataset(
        args.data_dir, "valid", args.valid_samples, args.seed + 1, args.sequence_length
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.device == "cuda",
    )
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    audit = {"train": train_data.audit(), "valid": valid_data.audit()}
    atomic_json(args.output_dir / "dataset_audit.json", audit)
    model = make_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    ddsm_state = load_ddsm(args) if args.method == "ddsm" else None
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    accelerator = torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"
    print(f"method={args.method} accelerator={accelerator} parameters={parameter_count}")
    print(json.dumps(audit, sort_keys=True))
    metrics_path = args.output_dir / "metrics.jsonl"
    checkpoint_path = args.output_dir / "last_checkpoint.pt"
    model.train()
    iterator = iter(train_loader)
    started = time.time()
    completed_steps = 0
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.max_steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_for_batch(args, model, batch, device, ddsm_state)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            record = {
                "step": step, "train_loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm), "elapsed_seconds": time.time() - started,
            }
            metrics_file.write(json.dumps(record) + "\n")
            completed_steps = step
            if step == 1 or step % args.log_every == 0 or step == args.max_steps:
                metrics_file.flush()
                print(json.dumps(record), flush=True)
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                atomic_checkpoint(checkpoint_path, model, optimizer, args, completed_steps)
                print(f"checkpoint={checkpoint_path} step={step}", flush=True)
            if args.training_wall_seconds and time.time() - started >= args.training_wall_seconds:
                print(
                    f"training_wall_limit_reached={args.training_wall_seconds} "
                    f"completed_steps={completed_steps}",
                    flush=True,
                )
                break
    atomic_checkpoint(checkpoint_path, model, optimizer, args, completed_steps)
    model.eval()
    validation_losses = []
    for batch_index, validation_batch in enumerate(valid_loader):
        if batch_index >= args.validation_batches:
            break
        with torch.set_grad_enabled(args.method == "ddsm"):
            validation_loss, _ = loss_for_batch(args, model, validation_batch, device, ddsm_state)
        validation_losses.append(float(validation_loss.detach()))
    if not validation_losses:
        raise RuntimeError("Validation loader produced no batches")

    generated_parts = []
    target_parts = []
    signal_parts = []
    diagnostics = {}
    remaining = min(args.sample_count, len(valid_data))
    for validation_batch in valid_loader:
        if remaining <= 0:
            break
        count = min(remaining, validation_batch["tokens"].shape[0])
        signal = condition_transform(validation_batch["signal"][:count].to(device))
        target = validation_batch["tokens"][:count].to(device)
        generated, batch_diagnostics = generate(args, model, signal, device, ddsm_state)
        generated_parts.append(generated)
        target_parts.append(target)
        signal_parts.append(signal)
        for key, value in batch_diagnostics.items():
            diagnostics[key] = diagnostics.get(key, 0) + value
        remaining -= count
        print(f"generated_samples={args.sample_count - remaining}/{args.sample_count}", flush=True)
    generated = torch.cat(generated_parts)
    target = torch.cat(target_parts)
    signal = torch.cat(signal_parts)
    sample_metrics, strings = evaluate_samples(generated, target, signal, diagnostics)
    summary = {
        "method": args.method,
        "seed": args.seed,
        "parameter_count": parameter_count,
        "train_steps": completed_steps,
        "requested_train_steps": args.max_steps,
        "validation_batches": len(validation_losses),
        "validation_loss": float(np.mean(validation_losses)),
        "wall_seconds": time.time() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
        "sample_metrics": sample_metrics,
        "status": "PASS",
        "scope": args.scope,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_checkpoint(checkpoint_path, model, optimizer, args, completed_steps)
    with (args.output_dir / "samples.fasta").open("w", encoding="utf-8") as handle:
        for index, sequence in enumerate(strings):
            handle.write(f">{args.method}_seed{args.seed}_{index}\n{sequence}\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    print(f"saved={args.output_dir}")


if __name__ == "__main__":
    main()
