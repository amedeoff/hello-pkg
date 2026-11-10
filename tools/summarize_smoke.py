#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("run_root", type=Path)
args = parser.parse_args()
methods = ["d3pm", "discrete_fm", "ddsm", "dirichlet_fm"]
rows = []
for method in methods:
    matches = sorted(args.run_root.glob(f"{method}-task*/summary.json"))
    if not matches:
        rows.append((method, "MISSING", "-", "-", "-", "-"))
        continue
    summary = json.loads(matches[-1].read_text())
    metrics = summary["sample_metrics"]
    rows.append((
        method, summary["status"], f'{summary["validation_loss"]:.5g}',
        f'{metrics["gc_fraction"]:.4f}', f'{metrics["three_mer_js_nats"]:.4f}',
        f'{summary["wall_seconds"]:.1f}',
    ))
print("method\tstatus\tvalidation_loss\tGC\t3mer_JS_nats\tseconds")
for row in rows:
    print("\t".join(row))
