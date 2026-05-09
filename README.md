# Knowledge Distillation for Edge Deployment — PlantVillage

A university machine learning term project on knowledge distillation for
plant disease classification. The goal is to transfer knowledge from a strong
teacher network (ResNet50) to a lightweight student model (MobileNetV3) that
is suitable for edge deployment.

## Problem Statement

Deep image classifiers achieve strong accuracy but are expensive to run on
edge devices due to their memory footprint and latency. This project
investigates whether knowledge distillation can compress a strong PlantVillage
classifier into a smaller student network that maintains competitive accuracy
while being faster and leaner.

## Dataset

**PlantVillage** — 38-class plant disease image classification dataset.

- Source: https://www.kaggle.com/datasets/mohitsingh1804/plantvillage
- Not committed to this repository (see [data/README.md](data/README.md))
- The initial checkpoint uses the **Potato subset** (3 classes) for fast
  iteration. Full 38-class training is planned for the next phase.

## Current Checkpoint Status

This repository contains a **runnable progress checkpoint**:

- Dataset loading with subset filtering, train/val/test splits, and
  ImageNet-normalized transforms.
- Model factory supporting ResNet18, ResNet50, MobileNetV3-Small,
  EfficientNet-B0.
- Supervised training script with per-epoch metrics (loss, accuracy, macro-F1).
- Evaluation helpers: classification report, confusion matrix, inference
  latency measurement.
- Distillation loss placeholder ready for the next phase.
- Reproducible YAML-driven experiments with automatic device selection
  (CUDA > MPS > CPU).

## Setup

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

## Dataset Download

1. Download from Kaggle:
   https://www.kaggle.com/datasets/mohitsingh1804/plantvillage
2. Unzip and place at:

```
data/raw/PlantVillage/
├── Apple___Apple_scab/
├── Potato___Early_blight/
├── Potato___Late_blight/
├── Potato___healthy/
└── ...
```

The default `data_dir` in all configs points to `data/raw/PlantVillage`.
Edit the YAML if you place the data elsewhere.

## How to Run

### 1. Check dataset

```bash
python -m src.data --config configs/baseline_resnet18.yaml
```

Expected output:
```
Dataset path : .../data/raw/PlantVillage
Subset prefix: Potato
Classes (3): ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
Train samples: 1500
Val   samples: 272
Test  samples: 272
```

### 2. Train ResNet18 baseline (Potato subset)

```bash
python -m src.train_baseline --config configs/baseline_resnet18.yaml
```

Or use the shell script:

```bash
bash scripts/run_baseline.sh
```

### 3. Train MobileNetV3-Small student baseline (Potato subset)

```bash
python -m src.train_baseline --config configs/student_mobilenetv3.yaml
```

Or:

```bash
bash scripts/run_student.sh
```

## Device Selection

Device is selected automatically — no config change needed:

| Available hardware | Device used |
|--------------------|-------------|
| NVIDIA GPU (CUDA)  | `cuda`      |
| Apple Silicon MPS  | `mps`       |
| CPU only           | `cpu`       |

## Running on Mac (Apple Silicon M4 Pro)

```bash
# From repo root, after dataset is in place:
source .venv/bin/activate
bash scripts/run_baseline.sh
bash scripts/run_student.sh
```

MPS will be selected automatically. Training 3 epochs on the Potato subset
takes a few minutes on M4 Pro.

## Running on Windows/Linux (RTX 4070)

```bash
# From repo root, after dataset is in place:
.venv\Scripts\activate          # Windows PowerShell
bash scripts/run_baseline.sh    # or Git Bash / WSL
bash scripts/run_student.sh
```

CUDA will be selected automatically. Training is faster than on MPS.

## Expected Outputs

After a training run, the following files appear under `outputs/<experiment_name>/`:

```
outputs/resnet18_potato_subset/
├── metrics.json            # test accuracy, F1, latency, parameter count
├── results.csv             # single-row summary for comparison tables
├── training_history.csv    # per-epoch train_loss, val_loss, val_accuracy, val_f1
├── confusion_matrix.png    # heatmap over test set
└── model_summary.json      # model name, class count, parameter count
```

## Repository Structure

```
knowledge-distillation-edge/
├── configs/
│   ├── baseline_resnet18.yaml      # ResNet18 / Potato subset (active)
│   ├── student_mobilenetv3.yaml    # MobileNetV3 / Potato subset (active)
│   └── distillation.yaml          # Next-phase placeholder
├── data/
│   └── README.md                  # Dataset download instructions
├── outputs/
│   └── .gitkeep
├── scripts/
│   ├── run_baseline.sh
│   └── run_student.sh
├── src/
│   ├── __init__.py
│   ├── data.py                    # Dataset loading and DataLoaders
│   ├── models.py                  # Model factory
│   ├── train_baseline.py          # Supervised training entry point
│   ├── evaluate.py                # Evaluation helpers
│   ├── utils.py                   # Config, seed, device, I/O utilities
│   └── distillation.py            # Distillation loss (next phase)
├── requirements.txt
└── README.md
```

## Planned Next Phase

- Train ResNet50 teacher on the full 38-class PlantVillage dataset.
- Train MobileNetV3-Small student using knowledge distillation
  (soft targets + hard-label CE, temperature scaling).
- Compare teacher vs. distilled student on accuracy, F1, parameter count,
  model size, and inference latency.
- Investigate post-training quantization as an additional edge optimization.
