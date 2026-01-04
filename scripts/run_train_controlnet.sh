#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 python scripts/train_controlnet_simple.py \
  --train_jsonl data/busi/splits/train.jsonl \
  --output_dir outputs/controlnet_busi \
  --pretrained_model /homeB/youkangqi/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14 \
  --controlnet_model /homeB/youkangqi/.cache/huggingface/hub/models--lllyasviel--ControlNet/snapshots/e78a8c4a5052a238198043ee5c0cb44e22abb9f7/models/control_sd15_seg.pth \
  --controlnet_config /homeB/youkangqi/imagine-colorization/ControlNet/models/cldm_v15.yaml \
  --num_epochs 200 \
  --batch_size 4 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --mixed_precision fp16
