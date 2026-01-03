#!/usr/bin/env python
import argparse
import json
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np


LABELS = ["benign", "malignant"]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare BUSI dataset splits and prompts.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to Dataset_BUSI_with_GT.")
    parser.add_argument("--out_dir", type=str, default="data/busi",
                        help="Output directory for jsonl and masks.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_count", type=int, default=396)
    parser.add_argument("--val_count", type=int, default=54)
    parser.add_argument("--test_count", type=int, default=197)
    parser.add_argument("--min_mask_area", type=int, default=10)
    return parser.parse_args()


def sanitize_id(label, stem):
    safe = stem.replace(" ", "_").replace("(", "").replace(")", "")
    return f"{label}_{safe}"


def compute_features(mask):
    area = float(mask.sum())
    h, w = mask.shape
    area_ratio = area / float(h * w)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return area_ratio, 0.0, 0.0
    cnt = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4.0 * math.pi * area / (perimeter * perimeter + 1e-6)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)
    return area_ratio, circularity, solidity


def bucket(value, q1, q2, labels):
    if value <= q1:
        return labels[0]
    if value <= q2:
        return labels[1]
    return labels[2]


def build_prompt(label, size, shape, boundary):
    return (
        f"breast ultrasound image of a {size} {shape} tumor "
        f"with {boundary} boundary, {label}"
    )


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    mask_out_dir = out_dir / "masks"
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for label in LABELS:
        label_dir = data_root / label
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label dir: {label_dir}")
        for img_path in sorted(label_dir.glob("*.png")):
            if "_mask" in img_path.stem:
                continue
            mask_paths = sorted(label_dir.glob(img_path.stem + "_mask*.png"))
            if not mask_paths:
                continue
            union = None
            for mask_path in mask_paths:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                mask = (mask > 0).astype(np.uint8)
                union = mask if union is None else np.maximum(union, mask)
            if union is None or union.sum() < args.min_mask_area:
                continue

            sample_id = sanitize_id(label, img_path.stem)
            mask_out_path = mask_out_dir / f"{sample_id}.png"
            cv2.imwrite(str(mask_out_path), (union * 255).astype(np.uint8))

            area_ratio, circularity, solidity = compute_features(union)
            samples.append({
                "id": sample_id,
                "label": label,
                "image_path": str(img_path.resolve()),
                "mask_path": str(mask_out_path.resolve()),
                "area_ratio": area_ratio,
                "circularity": circularity,
                "solidity": solidity,
            })

    if len(samples) < (args.train_count + args.val_count + args.test_count):
        raise ValueError("Not enough samples for requested split counts.")

    area_ratios = np.array([s["area_ratio"] for s in samples], dtype=np.float32)
    circularities = np.array([s["circularity"] for s in samples], dtype=np.float32)
    solidities = np.array([s["solidity"] for s in samples], dtype=np.float32)

    size_q1, size_q2 = np.quantile(area_ratios, [0.33, 0.66]).tolist()
    shape_q1, shape_q2 = np.quantile(circularities, [0.33, 0.66]).tolist()
    boundary_q = float(np.quantile(solidities, 0.5))

    for sample in samples:
        size = bucket(sample["area_ratio"], size_q1, size_q2, ["small", "medium", "large"])
        shape = bucket(sample["circularity"], shape_q1, shape_q2, ["irregular", "oval", "round"])
        boundary = "clear" if sample["solidity"] >= boundary_q else "unclear"
        sample["size"] = size
        sample["shape"] = shape
        sample["boundary"] = boundary
        sample["prompt"] = build_prompt(sample["label"], size, shape, boundary)

    random.Random(args.seed).shuffle(samples)

    train_end = args.train_count
    val_end = train_end + args.val_count

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:val_end + args.test_count]

    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    for name, split_samples in splits.items():
        jsonl_path = split_dir / f"{name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for sample in split_samples:
                f.write(json.dumps(sample, ensure_ascii=True) + "\n")

    meta = {
        "counts": {k: len(v) for k, v in splits.items()},
        "thresholds": {
            "size": [size_q1, size_q2],
            "shape": [shape_q1, shape_q2],
            "boundary": boundary_q,
        },
        "seed": args.seed,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    print("Prepared splits:")
    for name, split_samples in splits.items():
        print(f"  {name}: {len(split_samples)}")


if __name__ == "__main__":
    main()
