#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("run_root", type=Path)
args = parser.parse_args()

rows = []
for summary_path in sorted(args.run_root.glob("**/summary.json")):
    summary = json.loads(summary_path.read_text())
    sample = summary["sample_metrics"]
    sei_path = summary_path.parent / "sei_metrics.json"
    sei = json.loads(sei_path.read_text()) if sei_path.is_file() else None
    rows.append({
        "method": summary["method"],
        "seed": summary["seed"],
        "steps": summary["train_steps"],
        "validation_loss": summary["validation_loss"],
        "gc": sample["gc_fraction"],
        "target_gc": sample["target_gc_fraction"],
        "three_mer_js": sample["three_mer_js_nats"],
        "sp_mse": None if sei is None else sei["sp_mse"],
        "seconds": summary["wall_seconds"],
    })

if not rows:
    raise SystemExit(f"No summary.json files below {args.run_root}")

print("method\tseed\tsteps\tval_loss\tGC\ttarget_GC\t3mer_JS\tSP_MSE\tseconds")
for row in rows:
    sp_mse = "-" if row["sp_mse"] is None else f'{row["sp_mse"]:.6g}'
    print(
        f'{row["method"]}\t{row["seed"]}\t{row["steps"]}\t'
        f'{row["validation_loss"]:.6g}\t{row["gc"]:.4f}\t'
        f'{row["target_gc"]:.4f}\t{row["three_mer_js"]:.5f}\t'
        f'{sp_mse}\t{row["seconds"]:.1f}'
    )

grouped = defaultdict(list)
for row in rows:
    grouped[row["method"]].append(row)

if any(len(values) > 1 for values in grouped.values()):
    print("\naggregate_mean_plus_minus_sample_std")
    print("method\tseeds\tGC\t3mer_JS\tSP_MSE")
    for method in ("d3pm", "discrete_fm", "ddsm", "dirichlet_fm"):
        values = grouped.get(method, [])
        if not values:
            continue
        def mean_std(key):
            numbers = np.asarray([row[key] for row in values if row[key] is not None])
            if not len(numbers):
                return "-"
            deviation = numbers.std(ddof=1) if len(numbers) > 1 else 0.0
            return f"{numbers.mean():.6g} +/- {deviation:.3g}"
        print(
            f'{method}\t{len(values)}\t{mean_std("gc")}\t'
            f'{mean_std("three_mer_js")}\t{mean_std("sp_mse")}'
        )
