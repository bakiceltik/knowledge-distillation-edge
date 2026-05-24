#!/usr/bin/env bash
set -euo pipefail

echo "=== Post-training quantization: MobileNetV3-Small student ==="
python -m src.quantize --config configs/quantization.yaml
