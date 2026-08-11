#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ ! -f external/dinov3/hubconf.py ]]; then
  git clone https://github.com/facebookresearch/dinov3.git external/dinov3
fi

python scripts/check_environment.py
