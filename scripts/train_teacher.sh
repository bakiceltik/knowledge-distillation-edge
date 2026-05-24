#!/usr/bin/env bash
set -euo pipefail

echo "=== Training ResNet50 teacher on the Potato subset ==="
python -m src.train_baseline --config configs/teacher_resnet50_potato.yaml
