#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
  --config configs/train_512_ema.yaml
