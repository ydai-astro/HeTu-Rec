#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python experiment.py \
  --data-root Data --dinov3-repo ../external/dinov3 \
  --weights ../weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --out-dir runs/ablation_weighted_ce_seed42 \
  --sampling natural --positive-weight 4.0 --seed 42 --gpu-id 0
