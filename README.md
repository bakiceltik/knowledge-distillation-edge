# Knowledge Distillation for Edge Architecture

This repository contains the initial scaffold for a university machine learning term project on knowledge distillation for edge-friendly plant disease classification. The central goal is to transfer knowledge from a stronger teacher network to a lightweight student model that is more suitable for deployment on resource-constrained devices.

The planned application domain is image classification on the PlantVillage dataset. The long-term objective is to study whether a distilled MobileNetV3 student can preserve useful predictive performance while reducing model size and inference cost compared with heavier teacher models such as ResNet50 or EfficientNet variants.

## Problem Statement

Deep image classification models can achieve strong predictive performance, but many of them are expensive to deploy on edge devices because of their memory footprint and latency. This project studies how knowledge distillation can be used to compress a strong plant disease classifier into a smaller student network without training the student only from hard labels.

The motivating question is:

Can we train an edge-friendly student model for PlantVillage classification that maintains competitive accuracy while improving deployment characteristics such as parameter count, model size, and inference latency?

## Planned Methodology

The project is organized around the following stages:

1. Establish supervised baselines for a teacher model and a lightweight student model.
2. Train or fine-tune a stronger teacher classifier on PlantVillage.
3. Train a MobileNetV3 student using knowledge distillation with a combination of soft targets and standard cross-entropy loss.
4. Evaluate classification quality and deployment-oriented metrics.
5. Explore post-training quantization or related edge deployment optimizations as a future extension.

## Teacher and Student Design

- Teacher candidates: `ResNet50`, with room for future comparison to `EfficientNet-B0` or `EfficientNet-B2`
- Student candidate: `MobileNetV3`
- Distillation concept: blend hard-label supervision with teacher-guided soft targets using temperature scaling and an `alpha` weighting factor

## Planned Baselines

The following comparisons are planned for later experimental phases:

- Teacher-only supervised training
- Student-only supervised training
- Distilled student training
- Distilled student with future quantization experiments

## Planned Evaluation Metrics

The current plan is to report:

- Accuracy
- Macro-F1
- Model size
- Parameter count
- Inference latency

These metrics are intended to balance predictive quality with edge deployment practicality.

## Repository Structure

```text
knowledge-distillation-edge/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
├── data/
├── notebooks/
├── reports/
├── scripts/
└── src/
```

Key directories:

- `configs/`: YAML experiment configurations for baseline, distillation, and quantization workflows
- `reports/`: planning documents and, later, academic report material
- `scripts/`: lightweight shell entry points for common experiment commands
- `src/`: Python source code for data loading, models, training, evaluation, and deployment utilities

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Example entry points for the future training workflow:

```bash
python -m src.train --config configs/teacher_resnet50.yaml
python -m src.train --config configs/student_mobilenetv3.yaml
python -m src.distill --config configs/distillation.yaml
python -m src.evaluate --config configs/base.yaml
```

## Reproducibility Notes

This scaffold is structured with reproducibility in mind:

- central YAML configuration files for experiments
- explicit seed utilities
- separated training, evaluation, distillation, and inference entry points
- modular code layout for clean experimental comparison

Future iterations should add:

- checkpoint versioning conventions
- experiment logging
- deterministic settings review
- environment capture for report-ready reproduction

## Project Status

This repository is currently an initial scaffold. It intentionally does not contain completed experiments, trained checkpoints, reported metrics, or finalized conclusions. The present goal is to establish a clean and realistic project foundation for implementing the full machine learning pipeline in later commits.
