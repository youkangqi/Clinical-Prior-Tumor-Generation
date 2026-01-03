# Simple BUSI Reproduction (Generation + FID/KID)

This is a lightweight reproduction setup based on the paper settings. It builds a BUSI prompt+mask dataset, fine-tunes a ControlNet, generates images, and computes FID/KID.

## 1) Setup

Install deps (GPU recommended):

```bash
pip install -r requirements.txt
```

## 2) Prepare BUSI splits and prompts

```bash
python scripts/prepare_busi.py \
  --data_root Dataset_BUSI/Dataset_BUSI_with_GT \
  --out_dir data/busi
```

Outputs:
- `data/busi/splits/train.jsonl`
- `data/busi/splits/val.jsonl`
- `data/busi/splits/test.jsonl`
- `data/busi/masks/*.png`

Prompts are derived from mask size/shape/boundary heuristics.

## 3) Train ControlNet (simple)

```bash
python scripts/train_controlnet_simple.py \
  --train_jsonl data/busi/splits/train.jsonl \
  --output_dir outputs/controlnet_busi \
  --num_epochs 1 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --mixed_precision fp16
```

Notes:
- Increase `--num_epochs` for better quality (paper uses 200 epochs).
- Default base model: SD v1.5. Default ControlNet: `control_v11p_sd15_seg`.

## 4) Generate BUSI images

```bash
python scripts/generate_controlnet.py \
  --jsonl data/busi/splits/test.jsonl \
  --output_dir outputs/generated_test \
  --controlnet_model outputs/controlnet_busi/controlnet
```

## 5) Export real images (for FID/KID)

```bash
python scripts/export_real_images.py \
  --jsonl data/busi/splits/test.jsonl \
  --output_dir outputs/real_test
```

## 6) Compute FID/KID

```bash
python scripts/eval_fid_kid.py \
  --real_dir outputs/real_test \
  --fake_dir outputs/generated_test
```

The command prints a JSON with FID and KID scores.
