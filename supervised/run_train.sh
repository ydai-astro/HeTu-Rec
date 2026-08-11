#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

python experiment.py \
  --data-root Data \
  --dinov3-repo ../external/dinov3 \
  --weights ../weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --out-dir runs/full_d4_seed42 \
  --seed 42 \
  --image-size 224 \
  --train-batch-size 4 \
  --eval-batch-size 16 \
  --gradient-accumulation 8 \
  --max-steps 30000 \
  --warmup-steps 1000 \
  --eval-every 500 \
  --early-stopping-patience 12 \
  --backbone-lr 3e-6 \
  --head-lr 5e-4 \
  --weight-decay 0.01 \
  --label-smoothing 0.01 \
  --sampling moderate_positive \
  --positive-weight 1.0 \
  --workers 4 \
  --gpu-id 0
