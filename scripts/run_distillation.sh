#!/usr/bin/env bash
set -euo pipefail

echo "=== Training MobileNetV3-Small with ResNet50 distillation ==="
python -m src.train_distillation --config configs/distillation_resnet50_potato.yaml
