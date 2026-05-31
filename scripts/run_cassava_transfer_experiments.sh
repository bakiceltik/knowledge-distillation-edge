#!/usr/bin/env bash

set -euo pipefail

echo "=== Cassava: transfer from PlantVillage distilled student ==="
./.venv/bin/python -m src.train_baseline --config configs/student_mobilenetv3_cassava_from_plantvillage.yaml

echo "=== Cassava: scratch MobileNetV3-Small ==="
./.venv/bin/python -m src.train_baseline --config configs/student_mobilenetv3_cassava_scratch.yaml
