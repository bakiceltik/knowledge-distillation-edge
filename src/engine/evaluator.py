"""Evaluation engine utilities."""

from __future__ import annotations

from typing import Any

import torch

from src.utils.metrics import summarize_predictions


class Evaluator:
    """Evaluate classification models on a labeled dataset."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, dataloader) -> dict[str, Any]:
        """Run a simple evaluation pass and return metric placeholders."""
        model.eval()
        all_targets: list[int] = []
        all_predictions: list[int] = []

        for images, targets in dataloader:
            images = images.to(self.device)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1).cpu().tolist()

            all_predictions.extend(predictions)
            all_targets.extend(targets.tolist())

        metrics = summarize_predictions(all_targets, all_predictions)

        # TODO: Add parameter count, checkpoint size, and latency benchmarking.
        return metrics
