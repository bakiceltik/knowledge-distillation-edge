#!/usr/bin/env bash
# Run the MobileNetV3-Small student baseline experiment on the Potato subset.
set -euo pipefail

echo "=== Checking dataset ==="
python -m src.data --config configs/student_mobilenetv3_potato.yaml

echo ""
echo "=== Training MobileNetV3-Small student baseline ==="
python -m src.train_baseline --config configs/student_mobilenetv3_potato.yaml
