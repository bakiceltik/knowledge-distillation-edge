#!/usr/bin/env bash
set -euo pipefail

echo "=== Checking Cassava dataset ==="
python -m src.data --config configs/distillation_resnet50_cassava.yaml

echo ""
echo "=== Distilling MobileNetV3-Small from ResNet50 on Cassava (single split) ==="
python -m src.train_distillation --config configs/distillation_resnet50_cassava.yaml
