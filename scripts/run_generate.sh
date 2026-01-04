#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 python scripts/generate_controlnet.py \
  --jsonl data/busi/splits/test.jsonl \
  --output_dir outputs/generated_test \
  --base_model /homeB/youkangqi/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14 \
  --controlnet_model outputs/controlnet_busi/controlnet \
  --local_files_only
