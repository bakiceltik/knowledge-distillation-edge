"""Evaluation helpers: metrics, confusion matrix, and inference latency."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    """Run inference on *dataloader* and return a dict of evaluation metrics.

    Returns:
        accuracy          float
        macro_f1          float
        weighted_f1       float
        classification_report  dict  (per-class precision/recall/f1)
        confusion_matrix  np.ndarray (shape: [n_classes, n_classes])
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    """Save a confusion-matrix heatmap using matplotlib only."""
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def measure_inference_latency(
    model: nn.Module,
    device: torch.device,
    image_size: int = 224,
    batch_size: int = 1,
    warmup: int = 10,
    runs: int = 50,
) -> float:
    """Measure average single-batch inference latency in milliseconds.

    Safe for CUDA, MPS, and CPU:
    - CUDA: uses torch.cuda.synchronize() for accurate wall-clock timing.
    - MPS: relies on wall-clock time only (no MPS-specific sync API needed).
    - CPU: plain wall-clock time.

    Returns:
        Average latency in milliseconds over *runs* timed iterations.
    """
    model.eval()
    dummy = torch.randn(batch_size, 3, image_size, image_size, device=device)

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            _sync()

    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(runs):
            _sync()
            t0 = time.perf_counter()
            _ = model(dummy)
            _sync()
            latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(latencies))
