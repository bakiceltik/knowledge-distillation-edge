#!/usr/bin/env bash

set -euo pipefail

if [ -x "./.venv/bin/python" ]; then
  PYTHON_BIN="./.venv/bin/python"
elif [ -x "./.venv/Scripts/python.exe" ]; then
  PYTHON_BIN="./.venv/Scripts/python.exe"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Python executable not found. Activate the venv or install Python first."
  exit 1
fi

DATASET_DIR="data/raw/cassava"
TRANSFER_CONFIG="configs/student_mobilenetv3_cassava_from_plantvillage.yaml"
SCRATCH_CONFIG="configs/student_mobilenetv3_cassava_scratch.yaml"
TRANSFER_CKPT="outputs/mobilenetv3_full/best_model.pth"

if [ ! -d "$DATASET_DIR" ]; then
  echo "Cassava dataset not found at $DATASET_DIR"
  echo "Place the dataset there or update the Cassava config paths first."
  exit 1
fi

if [ -f "$TRANSFER_CKPT" ]; then
  echo "=== Cassava: transfer from PlantVillage student ==="
  "$PYTHON_BIN" -m src.train_baseline --config "$TRANSFER_CONFIG"
else
  echo "=== Cassava: transfer run skipped ==="
  echo "Missing checkpoint: $TRANSFER_CKPT"
  echo "Copy the PlantVillage student checkpoint to that path"
  echo "or edit $TRANSFER_CONFIG and set init_checkpoint to a checkpoint that exists."
fi

echo "=== Cassava: scratch MobileNetV3-Small ==="
"$PYTHON_BIN" -m src.train_baseline --config "$SCRATCH_CONFIG"
