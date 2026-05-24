#!/usr/bin/env bash
set -euo pipefail

echo "=== Checking full PlantVillage dataset ==="
python -m src.data --config configs/teacher_resnet50_full.yaml

echo ""
echo "=== Training ResNet50 teacher on full PlantVillage ==="
python -m src.train_baseline --config configs/teacher_resnet50_full.yaml
