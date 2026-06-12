#!/usr/bin/env bash
set -euo pipefail

echo "=== Checking Cassava dataset ==="
python -m src.data --config configs/student_mobilenetv3_cassava.yaml

echo ""
echo "=== Training MobileNetV3-Small baseline on Cassava (single split) ==="
python -m src.train_baseline --config configs/student_mobilenetv3_cassava.yaml
