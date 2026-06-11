"""Stratified k-fold cross-validation runner for baseline and distillation.

One script drives both experiment types; the mode is inferred from the config:

  - If ``teacher_model_name`` is present  -> distillation CV. For every fold the
    teacher is **retrained from scratch on that fold's training split** before
    distilling the student, so the held-out test fold never leaks into the
    teacher. This is the formally correct (if more expensive) protocol.
  - Otherwise                              -> baseline CV: a single model
    (``model_name``) is trained supervised on each fold.

For each fold the test metrics (accuracy, macro-F1, weighted-F1) are recorded,
and the script reports the mean +/- standard deviation across folds, the
standard way to present cross-validated results.

Usage:
    python -m src.cross_validate --config configs/cv_distillation_resnet50_cassava.yaml
    python -m src.cross_validate --config configs/cv_teacher_resnet50_cassava.yaml
    python -m src.cross_validate --config configs/cv_distillation_vit_b16_cassava.yaml --folds 5
"""

from __future__ import annotations

import argparse
import csv
import statistics
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.data import get_cv_dataloaders
from src.evaluate import evaluate_model, measure_inference_latency
from src.models import build_model
from src.utils import (
    count_parameters,
    ensure_dir,
    get_device,
    load_yaml_config,
    save_json,
    set_seed,
)


# Metrics aggregated across folds (order = display order in the summary/CSV).
METRIC_KEYS = [
    "accuracy",
    "macro_precision", "macro_recall", "macro_f1",
    "weighted_precision", "weighted_recall", "weighted_f1",
]


# ---------------------------------------------------------------------------
# Training / evaluation primitives (shared by both CV modes)
# ---------------------------------------------------------------------------

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


def _val_accuracy(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def _train_supervised(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    scaler: torch.cuda.amp.GradScaler | None,
    desc: str,
) -> nn.Module:
    """Train *model* supervised, keeping the best-val-accuracy weights in memory."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"  {desc} e{epoch}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                loss = criterion(model(images), labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        acc = _val_accuracy(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def _train_distillation(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    alpha: float,
    temperature: float,
    scaler: torch.cuda.amp.GradScaler | None,
) -> nn.Module:
    """Distill *student* from a frozen *teacher*, keeping best-val-accuracy weights."""
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    teacher.eval()
    best_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    for epoch in range(1, epochs + 1):
        student.train()
        for images, labels in tqdm(train_loader, desc=f"  distill e{epoch}/{epochs}", leave=False):
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
        acc = _val_accuracy(student, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    student.load_state_dict(best_state)
    return student


# ---------------------------------------------------------------------------
# Per-fold drivers
# ---------------------------------------------------------------------------

def _run_baseline_fold(cfg, loaders, num_classes, device, scaler) -> nn.Module:
    train_loader, val_loader, _ = loaders
    model = build_model(
        model_name=cfg["model_name"],
        num_classes=num_classes,
        pretrained=cfg.get("pretrained", True),
    ).to(device)
    return _train_supervised(
        model, train_loader, val_loader, device,
        epochs=cfg.get("epochs", 10),
        lr=cfg.get("learning_rate", 0.001),
        weight_decay=cfg.get("weight_decay", 0.0001),
        scaler=scaler,
        desc=cfg["model_name"],
    )


def _run_distillation_fold(cfg, loaders, num_classes, device, scaler) -> nn.Module:
    train_loader, val_loader, _ = loaders

    # 1) Train the teacher on THIS fold's training split (no test leakage).
    teacher = build_model(
        model_name=cfg["teacher_model_name"],
        num_classes=num_classes,
        pretrained=cfg.get("pretrained", True),
    ).to(device)
    teacher = _train_supervised(
        teacher, train_loader, val_loader, device,
        epochs=cfg.get("teacher_epochs", cfg.get("epochs", 10)),
        lr=cfg.get("teacher_learning_rate", cfg.get("learning_rate", 0.0003)),
        weight_decay=cfg.get("weight_decay", 0.0001),
        scaler=scaler,
        desc=f"teacher:{cfg['teacher_model_name']}",
    )
    for p in teacher.parameters():
        p.requires_grad_(False)

    # 2) Distill the student from the fold-specific teacher.
    student = build_model(
        model_name=cfg["student_model_name"],
        num_classes=num_classes,
        pretrained=cfg.get("pretrained", True),
    ).to(device)
    return _train_distillation(
        student, teacher, train_loader, val_loader, device,
        epochs=cfg.get("epochs", 10),
        lr=cfg.get("learning_rate", 0.0005),
        weight_decay=cfg.get("weight_decay", 0.0001),
        alpha=float(cfg.get("alpha", 0.5)),
        temperature=float(cfg.get("temperature", 4.0)),
        scaler=scaler,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _configure_cuda_backend() -> None:
    """Enable safe throughput optimizations for fixed-size 224x224 inputs.

    - cudnn.benchmark: autotune the fastest conv algorithm once and reuse it
      (valid because the input shape is constant across batches).
    - TF32: faster matmul/conv on Ampere+ GPUs at negligible accuracy cost,
      which especially helps the ViT teacher's attention matmuls.
    """
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified k-fold cross-validation runner.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--folds", type=int, default=None, help="Override n_folds from the config.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    n_folds = args.folds or cfg.get("n_folds", 10)
    is_distillation = "teacher_model_name" in cfg

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device()
    _configure_cuda_backend()
    output_dir = ensure_dir(cfg.get("output_dir", "outputs/cv_experiment"))

    mode = "DISTILLATION" if is_distillation else "BASELINE"
    print(f"\n{'='*64}")
    print(f"{n_folds}-FOLD CROSS-VALIDATION  ({mode})")
    print(f"Experiment : {cfg.get('experiment_name', 'unnamed')}")
    print(f"Device     : {device}")
    print(f"Output     : {output_dir}")
    if is_distillation:
        print(f"Teacher    : {cfg['teacher_model_name']}  (retrained per fold)")
        print(f"Student    : {cfg['student_model_name']}")
    else:
        print(f"Model      : {cfg['model_name']}")
    print(f"{'='*64}")

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    subset_prefix = cfg.get("subset_prefix") or None
    image_size = cfg.get("image_size", 224)

    fold_rows: list[dict[str, Any]] = []
    n_params = 0

    for fold in range(n_folds):
        # Re-seed per fold so model init is reproducible while splits stay fixed.
        set_seed(seed + fold)
        train_loader, val_loader, test_loader, class_names = get_cv_dataloaders(
            data_dir=cfg["data_dir"],
            n_folds=n_folds,
            fold=fold,
            image_size=image_size,
            batch_size=cfg.get("batch_size", 32),
            val_ratio=cfg.get("val_ratio", 0.15),
            seed=seed,
            num_workers=cfg.get("num_workers", 2),
            subset_prefix=subset_prefix,
        )
        num_classes = len(class_names)
        loaders = (train_loader, val_loader, test_loader)

        print(
            f"\n--- Fold {fold + 1}/{n_folds} "
            f"(train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
            f"test={len(test_loader.dataset)}) ---"
        )

        if is_distillation:
            model = _run_distillation_fold(cfg, loaders, num_classes, device, scaler)
        else:
            model = _run_baseline_fold(cfg, loaders, num_classes, device, scaler)

        n_params = count_parameters(model)
        results = evaluate_model(model, test_loader, device, class_names)
        results.pop("confusion_matrix", None)
        results.pop("classification_report", None)

        print(
            f"  fold acc={results['accuracy']:.4f}  "
            f"P={results['macro_precision']:.4f}  "
            f"R={results['macro_recall']:.4f}  "
            f"macro_f1={results['macro_f1']:.4f}"
        )
        fold_rows.append({"fold": fold + 1, **{k: results[k] for k in METRIC_KEYS}})

    # ---- aggregate ----
    summary = {k: _summarize([r[k] for r in fold_rows]) for k in METRIC_KEYS}

    latency_ms = measure_inference_latency(model, device, image_size=image_size, batch_size=1)

    print(f"\n{'='*64}")
    print(f"CROSS-VALIDATION SUMMARY  ({n_folds} folds)")
    for k in METRIC_KEYS:
        s = summary[k]
        print(f"  {k:18s}: {s['mean']*100:.2f}% +/- {s['std']*100:.2f}%  "
              f"(min {s['min']*100:.2f}, max {s['max']*100:.2f})")
    print(f"  {'parameters':18s}: {n_params:,}")
    print(f"  {'latency_ms':18s}: {latency_ms:.2f}")
    print(f"{'='*64}\n")

    # ---- persist ----
    cv_results = {
        "experiment_name": cfg.get("experiment_name"),
        "mode": mode.lower(),
        "n_folds": n_folds,
        "data_dir": str(cfg.get("data_dir", "")),
        "device": str(device),
        "model_name": cfg.get("model_name") or cfg.get("student_model_name"),
        "teacher_model_name": cfg.get("teacher_model_name"),
        "parameters": n_params,
        "latency_ms": latency_ms,
        "per_fold": fold_rows,
        "summary": summary,
    }
    if is_distillation:
        cv_results["temperature"] = float(cfg.get("temperature", 4.0))
        cv_results["alpha"] = float(cfg.get("alpha", 0.5))
    save_json(cv_results, output_dir / "cv_results.json")

    with (output_dir / "cv_fold_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", *METRIC_KEYS])
        writer.writeheader()
        writer.writerows(fold_rows)
        writer.writerow({"fold": "mean", **{k: summary[k]["mean"] for k in METRIC_KEYS}})
        writer.writerow({"fold": "std", **{k: summary[k]["std"] for k in METRIC_KEYS}})

    print(f"Saved: {output_dir / 'cv_results.json'}")
    print(f"Saved: {output_dir / 'cv_fold_metrics.csv'}")


if __name__ == "__main__":
    main()
