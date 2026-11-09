#!/usr/bin/env python3
"""Train DDSM on static binarized MNIST.

This is a command-line version of ``toy_example_bin_mnist.ipynb``.  It keeps
the data representation and DDSM loss from the official repository, while
adding deterministic seeds, checkpoints, JSONL metrics, and small smoke-test
settings suitable for validating a new machine before a full GPU run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from binary_mnist.eval_bin_mnist import (  # noqa: E402
    BinMNIST,
    ScoreNet,
    binary_to_onehot,
    load_mnist_binarized,
)
from ddsm import (  # noqa: E402
    Euler_Maruyama_sampler,
    UnitStickBreakingTransform,
    diffusion_factory,
    gx_to_gv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "bin_mnist")
    parser.add_argument("--noise-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "ddsm_bin_mnist")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--weight-estimation-batches", type=int, default=200)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=100)
    # The upstream sampler multiplies the number of integration steps by this
    # value and passes the result to torch.linspace, which requires an integer.
    parser.add_argument("--time-dilation", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--channel-multipliers", type=int, nargs="+", default=(1, 2, 2, 2))
    parser.add_argument("--attention-levels", type=int, nargs="+", default=(1,))
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def load_noise(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Noise cache not found: {path}. Generate it with presample_noise.py first."
        )
    try:
        values = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument
        values = torch.load(path, map_location="cpu")
    if len(values) != 5:
        raise ValueError(f"Expected five tensors in {path}, got {len(values)}")
    return values


def maybe_subset(dataset, limit: int | None):
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def to_onehot(batch: torch.Tensor) -> torch.Tensor:
    return binary_to_onehot(batch.squeeze(1)).float()


def estimate_time_weights(
    loader: DataLoader,
    n_time_steps: int,
    noise,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    device: torch.device,
    max_batches: int,
) -> torch.Tensor:
    """Estimate the importance-sampling weights used by the official notebook."""
    v_one, v_zero, v_one_loggrad, v_zero_loggrad, _ = noise
    sb = UnitStickBreakingTransform()
    sums = torch.zeros(n_time_steps, device=device)
    counts = torch.zeros(n_time_steps, device=device)

    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        x = to_onehot(batch)
        batch_size = x.shape[0]
        # Stratification guarantees coverage in short smoke tests.
        random_t = (torch.arange(batch_size) + batch_index * batch_size) % n_time_steps
        perturbed_x, target_score = diffusion_factory(
            x,
            random_t,
            v_one,
            v_zero,
            v_one_loggrad,
            v_zero_loggrad,
            alpha,
            beta,
            device=str(device),
        )
        perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()
        score_v = gx_to_gv(
            target_score,
            perturbed_x,
            compute_gradlogdet=False,
        )
        per_item = (
            perturbed_v * (1.0 - perturbed_v) * score_v.square()
        ).flatten(1).mean(1)
        random_t_device = random_t.to(device)
        sums.index_add_(0, random_t_device, per_item.detach())
        counts.index_add_(0, random_t_device, torch.ones_like(per_item))

    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0).flatten().tolist()
        raise RuntimeError(
            "Time-weight estimation did not cover every time bin. "
            f"Missing {len(missing)} bins; increase --weight-estimation-batches."
        )
    weights = sums / counts
    weights = weights / weights.mean()
    if not torch.isfinite(weights).all():
        raise RuntimeError("Non-finite time-dependent weights")
    return weights


def score_loss(
    prediction: torch.Tensor,
    target_score: torch.Tensor,
    perturbed_x: torch.Tensor,
    importance_weight: torch.Tensor,
) -> torch.Tensor:
    sb = UnitStickBreakingTransform()
    perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()
    predicted_v = gx_to_gv(
        prediction,
        perturbed_x,
        create_graph=True,
        compute_gradlogdet=False,
    )
    target_v = gx_to_gv(target_score, perturbed_x, compute_gradlogdet=False)
    broadcast_weight = importance_weight[(...,) + (None,) * (perturbed_x.ndim - 1)]
    return (
        perturbed_v
        * (1.0 - perturbed_v)
        * (predicted_v - target_v).square()
        / broadcast_weight
    ).mean()


@torch.no_grad()
def validate(
    model,
    loader,
    noise,
    timepoints,
    time_weights,
    alpha,
    beta,
    device,
) -> float:
    model.eval()
    losses = []
    probabilities = torch.sqrt(time_weights)
    probabilities = probabilities / probabilities.sum()
    # Validation only reports a score-matching diagnostic; paper NLL is computed
    # by binary_mnist/eval_bin_mnist.py from the best checkpoint.
    with torch.enable_grad():
        for batch in loader:
            x = to_onehot(batch)
            random_t = torch.multinomial(probabilities, x.shape[0], replacement=True)
            perturbed_x, target_score = diffusion_factory(
                x,
                random_t.cpu(),
                *noise[:4],
                alpha,
                beta,
                device=str(device),
            )
            prediction = model(perturbed_x, timepoints[random_t])
            loss = score_loss(
                prediction,
                target_score,
                perturbed_x,
                torch.sqrt(time_weights)[random_t],
            )
            losses.append(loss.detach().item())
    model.train()
    return float(np.mean(losses))


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    noise = load_noise(args.noise_path)
    *_, timepoints = noise
    timepoints = timepoints.float().to(device)
    n_time_steps = len(timepoints)
    alpha = torch.tensor([1.0])
    beta = torch.tensor([1.0])

    x_train, x_valid, _ = load_mnist_binarized(args.data_dir)
    train_set = BinMNIST(x_train)
    valid_set = BinMNIST(x_valid)
    train_set = maybe_subset(train_set, args.max_train_samples)
    valid_set = maybe_subset(valid_set, args.max_valid_samples)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_set, shuffle=False, **loader_kwargs)

    print(f"device={device} train={len(train_set)} valid={len(valid_set)} time_bins={n_time_steps}")
    time_weights = estimate_time_weights(
        train_loader,
        n_time_steps,
        noise,
        alpha,
        beta,
        device,
        args.weight_estimation_batches,
    )
    torch.save(time_weights.cpu(), args.output_dir / "time_weights.pt")

    time_step = float(timepoints[0].item())
    model = ScoreNet(
        ch=args.base_channels,
        ch_mult=list(args.channel_multipliers),
        attn=list(args.attention_levels),
        num_res_blocks=args.res_blocks,
        dropout=args.dropout,
        time_dependent_weights=time_weights,
        time_step=time_step,
        max_time=float(timepoints[-1].item()),
    ).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    probabilities = torch.sqrt(time_weights)
    probabilities = probabilities / probabilities.sum()
    metrics_path = args.output_dir / "metrics.jsonl"
    best_valid = math.inf
    global_step = 0
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            x = to_onehot(batch)
            random_t = torch.multinomial(probabilities, x.shape[0], replacement=True)
            perturbed_x, target_score = diffusion_factory(
                x,
                random_t.cpu(),
                *noise[:4],
                alpha,
                beta,
                device=str(device),
            )
            prediction = model(perturbed_x, timepoints[random_t])
            loss = score_loss(
                prediction,
                target_score,
                perturbed_x,
                torch.sqrt(time_weights)[random_t],
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {global_step}: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            global_step += 1
            if args.max_steps is not None and global_step >= args.max_steps:
                break

        train_loss = float(np.mean(epoch_losses))
        record = {
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": train_loss,
            "elapsed_seconds": time.time() - started,
        }
        should_validate = (epoch + 1) % args.validate_every == 0 or epoch == 0
        if should_validate:
            valid_loss = validate(
                model,
                valid_loader,
                noise,
                timepoints,
                time_weights,
                alpha,
                beta,
                device,
            )
            record["valid_loss"] = valid_loss
            if valid_loss < best_valid:
                best_valid = valid_loss
                torch.save(model.state_dict(), args.output_dir / "best_model.pt")
        append_jsonl(metrics_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "epoch": epoch + 1,
            "global_step": global_step,
            "time_weights": time_weights.cpu(),
        },
        args.output_dir / "last_checkpoint.pt",
    )

    model.eval()
    samples = Euler_Maruyama_sampler(
        model,
        (28, 28, 2),
        batch_size=args.sample_count,
        max_time=float(timepoints[-1].item()),
        min_time=float(timepoints[0].item()),
        num_steps=args.sample_steps,
        time_dilation=args.time_dilation,
        time_dilation_start_time=1.0 if args.time_dilation > 1 else None,
        eps=1e-5,
        device=str(device),
    )
    images = samples[..., 0].clamp(0.0, 1.0).unsqueeze(1).cpu()
    save_image(images, args.output_dir / "samples.png", nrow=max(1, int(math.sqrt(args.sample_count))))
    print(f"saved={args.output_dir}")


if __name__ == "__main__":
    main()
