#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHT_FILE="dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
WEIGHT_URL="https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

mkdir -p "$PROJECT_DIR/weights"
curl -L --fail --continue-at - "$WEIGHT_URL" -o "$PROJECT_DIR/weights/$WEIGHT_FILE"
echo "Downloaded: $PROJECT_DIR/weights/$WEIGHT_FILE"
