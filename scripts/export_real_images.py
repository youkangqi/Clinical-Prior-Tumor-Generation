#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Export resized real images for FID/KID.")
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/real_test")
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args()


def load_samples(jsonl_path):
    samples = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.jsonl)
    for sample in tqdm(samples, desc="exporting"):
        image = Image.open(sample["image_path"]).convert("RGB")
        image = image.resize((args.resolution, args.resolution), Image.BICUBIC)
        image.save(output_dir / f"{sample['id']}.png")

    print(f"Saved {len(samples)} images to {output_dir}")


if __name__ == "__main__":
    main()
