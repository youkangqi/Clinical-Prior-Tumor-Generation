#!/usr/bin/env python
import argparse
import json

import torch
from torch_fidelity import calculate_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID/KID with torch-fidelity.")
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dir", type=str, required=True)
    parser.add_argument("--kid_subset_size", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    metrics = calculate_metrics(
        input1=args.real_dir,
        input2=args.fake_dir,
        fid=True,
        kid=True,
        kid_subset_size=args.kid_subset_size,
        cuda=device == "cuda",
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
