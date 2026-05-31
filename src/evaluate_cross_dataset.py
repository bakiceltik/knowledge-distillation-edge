"""Cross-dataset evaluation for direct zero-shot behavior analysis.

This script is meant for cases where the source model's class space and the
target dataset's class space do not match. Instead of reporting standard
accuracy, it summarizes which source labels the model predicts on the target
dataset, along with confidence and entropy statistics.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import get_dataloaders
from src.models import build_model
from src.utils import ensure_dir, get_device, load_yaml_config, save_json, set_seed


def _load_class_names(metrics_path: str | Path) -> tuple[str, list[str]]:
    metrics = load_yaml_config(metrics_path)
    model_name = metrics.get("model_name") or metrics.get("student_model_name")
    class_names = metrics.get("class_names")
    if not model_name:
        raise ValueError(f"Could not find model name in metrics file: {metrics_path}")
    if not class_names:
        raise ValueError(f"Could not find class_names in metrics file: {metrics_path}")
    return str(model_name), list(class_names)


def _select_loader(
    split_name: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
) -> torch.utils.data.DataLoader:
    if split_name == "val":
        return val_loader
    if split_name == "test":
        return test_loader
    raise ValueError(f"Unsupported target_split '{split_name}'. Use val or test.")


def _extract_ordered_samples(dataset: Any) -> list[tuple[str, int]]:
    """Return (path, remapped_label) tuples in DataLoader iteration order."""
    if not hasattr(dataset, "indices") or not hasattr(dataset, "dataset") or not hasattr(dataset, "label_map"):
        raise TypeError("Expected a remapped subset dataset with indices, dataset, and label_map.")

    ordered_samples: list[tuple[str, int]] = []
    for original_index in dataset.indices:
        image_path, original_label = dataset.dataset.samples[original_index]
        remapped_label = dataset.label_map[original_label]
        ordered_samples.append((image_path, remapped_label))
    return ordered_samples


def _plot_target_source_heatmap(
    matrix: np.ndarray,
    source_class_names: list[str],
    target_class_names: list[str],
    output_path: Path,
) -> None:
    fig_width = max(10, len(source_class_names) * 0.35)
    fig_height = max(4, len(target_class_names) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(source_class_names)))
    ax.set_xticklabels(source_class_names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(target_class_names)))
    ax.set_yticklabels(target_class_names, fontsize=8)
    ax.set_xlabel("Predicted PlantVillage label")
    ax.set_ylabel("True Cassava label")
    ax.set_title("Cross-Dataset Prediction Counts")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset zero-shot evaluation.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    set_seed(cfg.get("seed", 42))
    device = get_device()
    output_dir = ensure_dir(cfg.get("output_dir", "outputs/cross_dataset_eval"))

    metrics_model_name, source_class_names = _load_class_names(cfg["source_metrics"])
    source_model_name = cfg.get("source_model_name", metrics_model_name)
    if source_model_name != metrics_model_name:
        raise ValueError(
            "source_model_name does not match the architecture recorded in "
            f"{cfg['source_metrics']}: {source_model_name} != {metrics_model_name}"
        )
    source_checkpoint = Path(cfg["source_checkpoint"])
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_checkpoint}")

    train_loader, val_loader, test_loader, target_class_names = get_dataloaders(
        data_dir=cfg["data_dir"],
        image_size=cfg.get("image_size", 224),
        batch_size=cfg.get("batch_size", 64),
        val_ratio=cfg.get("val_ratio", 0.15),
        test_ratio=cfg.get("test_ratio", 0.15),
        seed=cfg.get("seed", 42),
        num_workers=cfg.get("num_workers", 2),
        subset_prefix=cfg.get("subset_prefix") or None,
    )
    target_split = cfg.get("target_split", "test")
    eval_loader = _select_loader(target_split, train_loader, val_loader, test_loader)
    ordered_samples = _extract_ordered_samples(eval_loader.dataset)

    model = build_model(
        model_name=source_model_name,
        num_classes=len(source_class_names),
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(source_checkpoint, map_location=device))
    model.eval()

    print(f"\n{'='*60}")
    print(f"Experiment   : {cfg.get('experiment_name', 'unnamed')}")
    print(f"Device       : {device}")
    print(f"Source model : {source_model_name}")
    print(f"Source ckpt  : {source_checkpoint}")
    print(f"Target split : {target_split}")
    print(f"Output       : {output_dir}")
    print(f"{'='*60}")

    sample_rows: list[dict[str, Any]] = []
    overall_predictions = Counter()
    per_target_predictions: dict[str, Counter[str]] = defaultdict(Counter)
    target_source_matrix = np.zeros((len(target_class_names), len(source_class_names)), dtype=np.int64)
    max_confidences: list[float] = []
    entropies: list[float] = []

    cursor = 0
    with torch.no_grad():
        for images, labels in eval_loader:
            batch_size = labels.size(0)
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1)

            for batch_idx in range(batch_size):
                source_index = int(preds[batch_idx].item())
                target_index = int(labels[batch_idx].item())
                confidence = float(probs[batch_idx, source_index].item())
                entropy = float(
                    -(probs[batch_idx] * torch.log(probs[batch_idx].clamp_min(1e-12))).sum().item()
                )
                image_path, ordered_target_index = ordered_samples[cursor]
                if ordered_target_index != target_index:
                    raise RuntimeError("Sample ordering mismatch while exporting predictions.")
                cursor += 1

                source_label = source_class_names[source_index]
                target_label = target_class_names[target_index]
                overall_predictions[source_label] += 1
                per_target_predictions[target_label][source_label] += 1
                target_source_matrix[target_index, source_index] += 1
                max_confidences.append(confidence)
                entropies.append(entropy)

                sample_rows.append(
                    {
                        "image_path": image_path,
                        "true_target_label": target_label,
                        "predicted_source_label": source_label,
                        "predicted_source_index": source_index,
                        "max_confidence": confidence,
                        "entropy": entropy,
                    }
                )

    normalized_entropy = [value / math.log(len(source_class_names)) for value in entropies]
    overall_top_predictions = [
        {
            "source_label": label,
            "count": count,
            "share": count / len(sample_rows),
        }
        for label, count in overall_predictions.most_common(10)
    ]
    per_target_top_predictions = {
        target_label: [
            {
                "source_label": source_label,
                "count": count,
                "share_within_target": count / sum(counter.values()),
            }
            for source_label, count in counter.most_common(5)
        ]
        for target_label, counter in per_target_predictions.items()
    }

    summary = {
        "experiment_name": cfg.get("experiment_name"),
        "device": str(device),
        "source_model_name": source_model_name,
        "source_checkpoint": str(source_checkpoint),
        "source_num_classes": len(source_class_names),
        "source_class_names": source_class_names,
        "target_split": target_split,
        "target_num_classes": len(target_class_names),
        "target_class_names": target_class_names,
        "num_samples": len(sample_rows),
        "avg_max_confidence": float(np.mean(max_confidences)),
        "median_max_confidence": float(np.median(max_confidences)),
        "avg_entropy": float(np.mean(entropies)),
        "avg_normalized_entropy": float(np.mean(normalized_entropy)),
        "top_source_predictions": overall_top_predictions,
        "per_target_top_predictions": per_target_top_predictions,
        "note": (
            "Standard accuracy is not reported because the source PlantVillage labels "
            "and target Cassava labels are different class spaces."
        ),
    }
    save_json(summary, output_dir / "summary.json")

    with (output_dir / "sample_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sample_rows[0].keys())
        writer.writeheader()
        writer.writerows(sample_rows)

    with (output_dir / "target_to_source_counts.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["true_target_label", "predicted_source_label", "count", "share_within_target"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for target_label, counter in per_target_predictions.items():
            total = sum(counter.values())
            for source_label, count in counter.most_common():
                writer.writerow(
                    {
                        "true_target_label": target_label,
                        "predicted_source_label": source_label,
                        "count": count,
                        "share_within_target": count / total,
                    }
                )

    _plot_target_source_heatmap(
        target_source_matrix,
        source_class_names=source_class_names,
        target_class_names=target_class_names,
        output_path=output_dir / "target_vs_source_heatmap.png",
    )

    print("\nCross-dataset evaluation complete.")
    print(f"Samples                 : {len(sample_rows)}")
    print(f"Average max confidence  : {summary['avg_max_confidence']:.4f}")
    print(f"Average norm. entropy   : {summary['avg_normalized_entropy']:.4f}")
    print("Top source predictions  :")
    for row in overall_top_predictions[:5]:
        print(f"  {row['source_label']}: {row['count']} ({row['share']:.2%})")
    print(f"Outputs saved           : {output_dir}")


if __name__ == "__main__":
    main()
