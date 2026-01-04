#!/usr/bin/env bash
set -euo pipefail

python scripts/eval_fid_kid.py \
  --real_dir outputs/real_test \
  --fake_dir outputs/generated_test
