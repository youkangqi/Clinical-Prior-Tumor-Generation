#!/usr/bin/env python
import argparse
import json
import math
import inspect
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer


class BUSIControlDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, resolution=512):
        self.samples = []
        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

        self.tokenizer = tokenizer
        self.image_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.cond_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("L")

        image = self.image_transform(image)
        cond = self.cond_transform(mask)
        cond = cond.repeat(3, 1, 1)

        prompt = sample.get("prompt", "breast ultrasound tumor")
        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "pixel_values": image,
            "conditioning_pixel_values": cond,
            "input_ids": input_ids,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Simple ControlNet training on BUSI.")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_model", type=str, default="lllyasviel/control_v11p_sd15_seg")
    parser.add_argument("--output_dir", type=str, default="outputs/controlnet_busi")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_every_steps", type=int, default=1000)
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16"], default="no")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--controlnet_config", type=str, default=None)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }


def load_controlnet(model_path, local_files_only, controlnet_config):
    path = Path(model_path)
    if path.is_file():
        if controlnet_config is not None and not Path(controlnet_config).exists():
            raise FileNotFoundError(f"Missing controlnet_config: {controlnet_config}")
        if hasattr(ControlNetModel, "from_single_file"):
            kwargs = {}
            if controlnet_config:
                sig = inspect.signature(ControlNetModel.from_single_file)
                if "original_config_file" in sig.parameters:
                    kwargs["original_config_file"] = controlnet_config
                elif "config" in sig.parameters:
                    kwargs["config"] = controlnet_config
            return ControlNetModel.from_single_file(str(path), **kwargs)
        raise ValueError(
            "controlnet_model points to a single-file checkpoint, but this diffusers "
            "version lacks ControlNetModel.from_single_file. Convert to a diffusers "
            "directory or upgrade diffusers."
        )
    return ControlNetModel.from_pretrained(str(path), local_files_only=local_files_only)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=True, indent=2)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model,
        subfolder="tokenizer",
        local_files_only=args.local_files_only,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model,
        subfolder="text_encoder",
        local_files_only=args.local_files_only,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model,
        subfolder="vae",
        local_files_only=args.local_files_only,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model,
        subfolder="unet",
        local_files_only=args.local_files_only,
    )
    controlnet = load_controlnet(args.controlnet_model, args.local_files_only, args.controlnet_config)

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    text_encoder.to(device)
    vae.to(device)
    unet.to(device)
    controlnet.to(device)

    train_dataset = BUSIControlDataset(args.train_jsonl, tokenizer, resolution=args.resolution)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps
    if max_train_steps is None:
        max_train_steps = args.num_epochs * num_update_steps_per_epoch

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )

    mixed_precision = args.mixed_precision == "fp16" and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    global_step = 0
    progress_bar = tqdm(total=max_train_steps, desc="training")

    for epoch in range(args.num_epochs):
        controlnet.train()
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                pixel_values = batch["pixel_values"].to(device)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
            conditioning = batch["conditioning_pixel_values"].to(device)

            with torch.cuda.amp.autocast(enabled=mixed_precision):
                down_samples, mid_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    conditioning,
                    return_dict=False,
                )
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                ).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                loss = loss / args.gradient_accumulation_steps

            if mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                if args.save_every_steps and global_step % args.save_every_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    controlnet.save_pretrained(ckpt_dir)

                if global_step >= max_train_steps:
                    break
        if global_step >= max_train_steps:
            break

    controlnet.save_pretrained(output_dir / "controlnet")
    print(f"Saved controlnet to {output_dir / 'controlnet'}")


if __name__ == "__main__":
    main()
