#!/usr/bin/env bash
set -euo pipefail

python scripts/export_real_images.py \
  --jsonl data/busi/splits/test.jsonl \
  --output_dir outputs/real_test
