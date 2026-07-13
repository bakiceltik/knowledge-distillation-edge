"""Knowledge distillation training script.

Usage:
    python -m src.train_distillation --config configs/distillation_potato.yaml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.data import get_dataloaders
from src.evaluate import evaluate_model, measure_inference_latency, plot_confusion_matrix
from src.models import build_model
from src.utils import (
    count_parameters,
    ensure_dir,
    get_device,
    load_yaml_config,
    save_json,
    set_seed,
)


def _distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    ce_loss = F.cross_entropy(student_logits, labels)
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return alpha * kd_loss + (1.0 - alpha) * ce_loss


def _train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
    temperature: float,
    scaler: torch.cuda.amp.GradScaler | None,
) -> float:
    student.train()
    teacher.eval()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)
            loss = _distillation_loss(student_logits, teacher_logits, labels, alpha, temperature)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def _eval_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  val  ", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    from sklearn.metrics import accuracy_score, f1_score

    return {
        "val_loss": total_loss / len(loader.dataset),
        "val_accuracy": float(accuracy_score(all_labels, all_preds)),
        "val_macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge distillation training.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    set_seed(cfg.get("seed", 42))
    device = get_device()
    output_dir = ensure_dir(cfg.get("output_dir", "outputs/experiment"))

    print(f"\n{'='*60}")
    print(f"Experiment  : {cfg.get('experiment_name', 'unnamed')}")
    print(f"Device      : {device}")
    print(f"Output      : {output_dir}")
    print(f"{'='*60}")

    # ---- data ----
    subset_prefix = cfg.get("subset_prefix") or None
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=cfg["data_dir"],
        image_size=cfg.get("image_size", 224),
        batch_size=cfg.get("batch_size", 32),
        val_ratio=cfg.get("val_ratio", 0.15),
        test_ratio=cfg.get("test_ratio", 0.15),
        seed=cfg.get("seed", 42),
        num_workers=cfg.get("num_workers", 2),
        subset_prefix=subset_prefix,
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    num_classes = len(class_names)

    print(f"\nClasses ({num_classes}): {class_names}")
    print(f"Train / Val / Test : {n_train} / {n_val} / {n_test}")

    # Reseed *after* the split is built so `train_seed` varies student init and
    # batch order while leaving the data partition fixed by `seed`. The teacher
    # was trained on the `seed` split, so it stays valid across replicate runs.
    train_seed = int(cfg.get("train_seed", cfg.get("seed", 42)))
    set_seed(train_seed)
    print(f"Train seed         : {train_seed} (split seed: {cfg.get('seed', 42)})")

    # ---- teacher ----
    teacher_ckpt = Path(cfg["teacher_checkpoint"])
    if not teacher_ckpt.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {teacher_ckpt}\n"
            "Run the ResNet-18 baseline first:\n"
            "  python -m src.train_baseline --config configs/baseline_resnet18_potato.yaml"
        )

    teacher = build_model(
        model_name=cfg.get("teacher_model_name", "resnet18"),
        num_classes=num_classes,
        pretrained=False,
    ).to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"\nTeacher loaded from {teacher_ckpt}")

    # ---- student ----
    student = build_model(
        model_name=cfg.get("student_model_name", "mobilenet_v3_small"),
        num_classes=num_classes,
        pretrained=cfg.get("pretrained", True),
    ).to(device)
    n_params = count_parameters(student)
    print(f"Student parameters : {n_params:,}")

    # ---- training ----
    alpha = float(cfg.get("alpha", 0.5))
    temperature = float(cfg.get("temperature", 4.0))
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=cfg.get("learning_rate", 0.001),
        weight_decay=cfg.get("weight_decay", 0.0001),
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    if scaler is not None:
        print("AMP      : enabled (fp16 mixed precision)")

    epochs = cfg.get("epochs", 3)
    history: list[dict[str, Any]] = []
    best_val_acc = -1.0
    best_ckpt_path = output_dir / "best_model.pth"

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = _train_one_epoch(
            student, teacher, train_loader, optimizer, device, alpha, temperature, scaler
        )
        val_metrics = _eval_one_epoch(student, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        print(
            f"  train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['val_loss']:.4f}  "
            f"val_acc={val_metrics['val_accuracy']:.4f}  "
            f"val_f1={val_metrics['val_macro_f1']:.4f}"
        )
        if val_metrics["val_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["val_accuracy"]
            torch.save(student.state_dict(), best_ckpt_path)
            print(f"  [checkpoint] best_model.pth saved (val_acc={best_val_acc:.4f})")

    # ---- test evaluation (load best checkpoint) ----
    print("\nLoading best checkpoint for test evaluation...")
    student.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    print("Evaluating on test set...")
    test_results = evaluate_model(student, test_loader, device, class_names)
    cm = test_results.pop("confusion_matrix")

    print(f"  Test accuracy   : {test_results['accuracy']:.4f}")
    print(f"  Test macro-F1   : {test_results['macro_f1']:.4f}")
    print(f"  Test weighted-F1: {test_results['weighted_f1']:.4f}")

    # ---- latency ----
    print("\nMeasuring inference latency (batch=1)...")
    latency_ms = measure_inference_latency(
        student, device, image_size=cfg.get("image_size", 224), batch_size=1
    )
    print(f"  Latency : {latency_ms:.2f} ms")

    # ---- save outputs ----
    metrics: dict[str, Any] = {
        "experiment_name": cfg.get("experiment_name"),
        "device": str(device),
        "teacher_model_name": cfg.get("teacher_model_name"),
        "student_model_name": cfg.get("student_model_name"),
        "temperature": temperature,
        "alpha": alpha,
        "num_classes": num_classes,
        "class_names": class_names,
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": n_test,
        **test_results,
        "latency_ms": latency_ms,
    }
    save_json(metrics, output_dir / "metrics.json")

    with (output_dir / "training_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    results_row = {
        "experiment_name": cfg.get("experiment_name"),
        "student_model_name": cfg.get("student_model_name"),
        "teacher_model_name": cfg.get("teacher_model_name"),
        "num_classes": num_classes,
        "test_accuracy": test_results["accuracy"],
        "test_macro_f1": test_results["macro_f1"],
        "test_weighted_f1": test_results["weighted_f1"],
        "latency_ms": latency_ms,
        "parameters": n_params,
        "temperature": temperature,
        "alpha": alpha,
    }
    with (output_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_row.keys())
        writer.writeheader()
        writer.writerow(results_row)

    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    model_summary = {
        "model_name": cfg.get("student_model_name"),
        "num_classes": num_classes,
        "trainable_parameters": n_params,
        "pretrained": cfg.get("pretrained", True),
        "teacher_model_name": cfg.get("teacher_model_name"),
        "temperature": temperature,
        "alpha": alpha,
    }
    save_json(model_summary, output_dir / "model_summary.json")

    print(f"\n{'='*60}")
    print("DISTILLATION COMPLETE")
    print(f"  Device         : {device}")
    print(f"  Classes        : {class_names}")
    print(f"  Train/Val/Test : {n_train}/{n_val}/{n_test}")
    print(f"  Test accuracy  : {test_results['accuracy']:.4f}")
    print(f"  Test macro-F1  : {test_results['macro_f1']:.4f}")
    print(f"  Parameters     : {n_params:,}")
    print(f"  Latency (ms)   : {latency_ms:.2f}")
    print(f"  Outputs saved  : {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
