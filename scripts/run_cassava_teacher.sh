#!/usr/bin/env bash
set -euo pipefail

echo "=== Checking Cassava dataset ==="
python -m src.data --config configs/teacher_resnet50_cassava.yaml

echo ""
echo "=== Training ResNet50 teacher on Cassava (single split) ==="
python -m src.train_baseline --config configs/teacher_resnet50_cassava.yaml
