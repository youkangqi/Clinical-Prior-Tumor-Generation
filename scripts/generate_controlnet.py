#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate BUSI images with ControlNet.")
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/generated")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_model", type=str, default="lllyasviel/control_v11p_sd15_seg")
    parser.add_argument("--controlnet_config", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
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


def load_controlnet(model_path, local_files_only, controlnet_config, torch_dtype):
    path = Path(model_path)
    if path.is_file():
        if controlnet_config is not None and not Path(controlnet_config).exists():
            raise FileNotFoundError(f"Missing controlnet_config: {controlnet_config}")
        if hasattr(ControlNetModel, "from_single_file"):
            kwargs = {}
            if controlnet_config:
                kwargs["original_config"] = controlnet_config
            kwargs["local_files_only"] = local_files_only
            if torch_dtype is not None:
                kwargs["torch_dtype"] = torch_dtype
            return ControlNetModel.from_single_file(str(path), **kwargs)
        raise ValueError(
            "controlnet_model points to a single-file checkpoint, but this diffusers "
            "version lacks ControlNetModel.from_single_file. Convert to a diffusers "
            "directory or upgrade diffusers."
        )
    return ControlNetModel.from_pretrained(
        str(path),
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    )


def main():
    args = parse_args()
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = load_controlnet(
        args.controlnet_model,
        args.local_files_only,
        args.controlnet_config,
        torch_dtype=dtype,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        local_files_only=args.local_files_only,
    )
    pipe.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.jsonl)

    for idx, sample in enumerate(tqdm(samples, desc="generating")):
        mask = Image.open(sample["mask_path"]).convert("L")
        mask = mask.resize((args.resolution, args.resolution), Image.NEAREST)
        mask_rgb = Image.merge("RGB", [mask, mask, mask])

        prompt = sample.get("prompt", "breast ultrasound tumor")
        generator = torch.Generator(device=device).manual_seed(args.seed + idx)

        image = pipe(
            prompt,
            image=mask_rgb,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        image.save(output_dir / f"{sample['id']}.png")

    print(f"Saved {len(samples)} images to {output_dir}")


if __name__ == "__main__":
    main()
