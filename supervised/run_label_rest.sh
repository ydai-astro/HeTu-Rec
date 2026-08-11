#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

THRESHOLD_FILE="runs/full_d4_seed42/evaluation/threshold.json"
if [[ ! -f "$THRESHOLD_FILE" ]]; then
  echo "Missing $THRESHOLD_FILE. Run bash run_test.sh runs/full_d4_seed42 first."
  exit 1
fi

python label_unlabeled.py \
  --input-dir Data/rest \
  --output-dir Data/not_label \
  --checkpoint runs/full_d4_seed42/best_checkpoint.pt \
  --dinov3-repo ../external/dinov3 \
  --weights ../weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --threshold-file "$THRESHOLD_FILE" \
  --batch-size 16 \
  --workers 4 \
  --gpu-id 0
