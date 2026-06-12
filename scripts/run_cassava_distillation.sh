#!/usr/bin/env bash
set -euo pipefail

echo "=== Checking full PlantVillage dataset ==="
python -m src.data --config configs/distillation_resnet50_full.yaml

echo ""
echo "=== Distilling MobileNetV3-Small from ResNet50 on full PlantVillage ==="
python -m src.train_distillation --config configs/distillation_resnet50_full.yaml
