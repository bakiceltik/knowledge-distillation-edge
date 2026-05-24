#!/usr/bin/env bash
set -euo pipefail

echo "=== Checking full PlantVillage dataset ==="
python -m src.data --config configs/student_mobilenetv3_full.yaml

echo ""
echo "=== Training MobileNetV3-Small baseline on full PlantVillage ==="
python -m src.train_baseline --config configs/student_mobilenetv3_full.yaml
