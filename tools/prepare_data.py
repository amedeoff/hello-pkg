#!/usr/bin/env python3
"""Download the official static binarized-MNIST split on a login node."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np


URLS = {
    "train": "https://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat",
    "valid": "https://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat",
    "test": "https://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    cache_dir = args.data_dir / "bin-mnist"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / "mnist.pkl.gz"
    if output.is_file():
        print(f"already_exists={output}")
        return

    splits = {}
    for split, url in URLS.items():
        destination = cache_dir / f"binarized_mnist_{split}.amat"
        print(f"downloading={url}", flush=True)
        urlretrieve(url, destination)
        splits[split] = np.loadtxt(destination)
        destination.unlink()

    joblib.dump(
        [splits["train"], splits["valid"], splits["test"]],
        output,
        compress=3,
    )
    print(f"saved={output}")


if __name__ == "__main__":
    main()
