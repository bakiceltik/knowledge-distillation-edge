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

This repository contains a **runnable progress checkpoint** with completed
Potato-subset experiments:

- Dataset loading with subset filtering, train/val/test splits, and
  ImageNet-normalized transforms.
- Model factory supporting ResNet18, ResNet50, MobileNetV3-Small,
  EfficientNet-B0.
- Supervised training script with per-epoch metrics (loss, accuracy, macro-F1).
- Active knowledge distillation training from a saved teacher checkpoint to a
  MobileNetV3-Small student.
- Evaluation helpers: classification report, confusion matrix, inference
  latency measurement.
- Reproducible YAML-driven experiments with automatic device selection
  (CUDA > MPS > CPU).

Completed Potato-subset runs currently include ResNet18 baseline,
MobileNetV3-Small baseline, ResNet50 teacher, historical ResNet18-to-MobileNetV3
distillation, and ResNet50-to-MobileNetV3 distillation. Quantization and full
38-class PlantVillage experiments are still pending.

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

The active Potato-subset configs point to `data/raw/PlantVillage/train`.
Edit the YAML if you place the data elsewhere.

## How to Run

### 1. Check dataset

```bash
python -m src.data --config configs/baseline_resnet18.yaml
```

Expected output:
```
Dataset path : .../data/raw/PlantVillage/train
Subset prefix: Potato
Classes (3): ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
Train samples: 1205
Val   samples: 258
Test  samples: 258
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

### 4. Train ResNet50 teacher (Potato subset)

```bash
python -m src.train_baseline --config configs/teacher_resnet50_potato.yaml
```

Or:

```bash
bash scripts/train_teacher.sh
```

### 5. Distill MobileNetV3-Small from ResNet50 teacher (Potato subset)

```bash
python -m src.train_distillation --config configs/distillation_resnet50_potato.yaml
```

Or:

```bash
bash scripts/run_distillation.sh
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
bash scripts/train_teacher.sh
bash scripts/run_distillation.sh
```

MPS will be selected automatically. Training 3 epochs on the Potato subset
takes a few minutes on M4 Pro.

## Running on Windows/Linux (RTX 4070)

```bash
# From repo root, after dataset is in place:
.venv\Scripts\activate          # Windows PowerShell
bash scripts/run_baseline.sh    # or Git Bash / WSL
bash scripts/run_student.sh
bash scripts/train_teacher.sh
bash scripts/run_distillation.sh
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

## Potato-Subset Results

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | Params | Latency |
|------------|----------|----------|-------------|--------|---------|
| ResNet18 baseline | 98.06% | 95.90% | 98.00% | 11,178,051 | 1.78 ms |
| MobileNetV3-Small baseline | 94.57% | 92.63% | 94.57% | 1,520,931 | 3.43 ms |
| ResNet50 teacher | 99.22% | 98.22% | 99.22% | 23,514,179 | 4.65 ms |
| MobileNetV3-Small distilled from ResNet50 | 93.41% | 88.29% | 93.19% | 1,520,931 | 3.54 ms |

The current ResNet50 distillation run validates the active KD pipeline and
artifact generation, but it does not yet improve over the supervised
MobileNetV3-Small baseline under the default 3-epoch setting.

## Repository Structure

```
knowledge-distillation-edge/
├── configs/
│   ├── baseline_resnet18.yaml      # ResNet18 / Potato subset (active)
│   ├── student_mobilenetv3.yaml    # MobileNetV3 / Potato subset (active)
│   ├── teacher_resnet50_potato.yaml
│   ├── distillation_potato.yaml    # ResNet18 teacher KD history
│   └── distillation_resnet50_potato.yaml
├── data/
│   └── README.md                  # Dataset download instructions
├── outputs/
│   └── .gitkeep
├── scripts/
│   ├── run_baseline.sh
│   ├── run_student.sh
│   ├── train_teacher.sh
│   └── run_distillation.sh
├── src/
│   ├── __init__.py
│   ├── data.py                    # Dataset loading and DataLoaders
│   ├── models.py                  # Model factory
│   ├── train_baseline.py          # Supervised training entry point
│   ├── train_distillation.py      # Active KD training entry point
│   ├── evaluate.py                # Evaluation helpers
│   ├── utils.py                   # Config, seed, device, I/O utilities
│   └── distillation.py            # Reusable distillation loss helper
├── requirements.txt
└── README.md
```

## Planned Next Phase

- Train ResNet50 teacher on the full 38-class PlantVillage dataset.
- Repeat supervised and distillation comparisons on the full 38-class
  PlantVillage dataset.
- Tune distillation hyperparameters (temperature, alpha, epochs) to test
  whether KD can outperform the supervised MobileNetV3-Small baseline.
- Investigate post-training quantization as an additional edge optimization.
