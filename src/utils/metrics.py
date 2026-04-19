"""Metric helpers for classification experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Compute classification accuracy."""
    return float(accuracy_score(list(y_true), list(y_pred)))


def compute_macro_f1(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Compute macro-averaged F1 score."""
    return float(f1_score(list(y_true), list(y_pred), average="macro"))


def count_parameters(model) -> int:
    """Return the number of trainable parameters in a model."""
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def estimate_model_size_mb(model) -> float:
    """Estimate model size in megabytes from parameters and buffers."""
    total_bytes = 0
    for tensor in model.state_dict().values():
        total_bytes += tensor.nelement() * tensor.element_size()
    return float(total_bytes / (1024**2))


def summarize_predictions(y_true: Iterable[int], y_pred: Iterable[int]) -> dict:
    """Return a compact metric summary for evaluation scripts."""
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    return {
        "accuracy": compute_accuracy(y_true_arr, y_pred_arr),
        "macro_f1": compute_macro_f1(y_true_arr, y_pred_arr),
    }
