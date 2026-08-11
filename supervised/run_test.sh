#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-runs/full_d4_seed42}"
cd "$PROJECT_DIR"

python evaluate.py \
  --checkpoint "$RUN_DIR/best_checkpoint.pt" \
  --split val \
  --data-root Data \
  --dinov3-repo ../external/dinov3 \
  --weights ../weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --output-dir "$RUN_DIR/evaluation" \
  --calibrate-threshold \
  --tta \
  --gpu-id 0

python evaluate.py \
  --checkpoint "$RUN_DIR/best_checkpoint.pt" \
  --split test \
  --data-root Data \
  --dinov3-repo ../external/dinov3 \
  --weights ../weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --output-dir "$RUN_DIR/evaluation" \
  --threshold-file "$RUN_DIR/evaluation/threshold.json" \
  --tta \
  --gpu-id 0

python plot.py --run-dir "$RUN_DIR"
