#!/usr/bin/env bash
# Run the ResNet18 baseline experiment on the Potato subset.
set -euo pipefail

echo "=== Checking dataset ==="
python -m src.data --config configs/baseline_resnet18_potato.yaml

echo ""
echo "=== Training ResNet18 baseline ==="
python -m src.train_baseline --config configs/baseline_resnet18_potato.yaml
