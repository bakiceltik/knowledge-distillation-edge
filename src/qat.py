"""Quantization-aware training for the distilled MobileNetV3-Small student.

The paper's post-training-quantization study shows that int8 PTQ collapses this
architecture (84% -> ~31%) regardless of calibration size, weight granularity, or
which modules are exempted, and that the cause is the architecture rather than
distillation. From that we asserted that quantization-aware training would be a
*prerequisite* for an int8 student. This script tests that assertion instead of
leaving it as one.

Method: fine-tune the already-distilled fp32 student with fake-quantization
inserted (PT2E `prepare_qat_pt2e`), then convert. The quantizer, the split, and
the evaluation are identical to those in `src.replicate_quant`, so the resulting
int8 accuracy is directly comparable to the PTQ numbers.

Note QAT needs no calibration set: the observers learn their ranges during
training. The calibration-draw noise that makes PTQ figures irreproducible on
this model therefore does not apply here, which we verify by reporting results
over several training seeds rather than several calibration seeds.

Usage:
    python -m src.qat --config configs/qat_cassava.yaml
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data import get_dataloaders
from src.models import build_model
from src.utils import ensure_dir, load_yaml_config, save_json, set_seed


def _evaluate(model: torch.nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu())
            ys.append(y)
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    return accuracy_score(y, p) * 100, f1_score(y, p, average="macro") * 100


def _kd_loss(student_logits, teacher_logits, labels, alpha: float, temperature: float):
    ce = F.cross_entropy(student_logits, labels)
    kd = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return alpha * kd + (1.0 - alpha) * ce


def run_qat(cfg: dict, train_seed: int, loaders, device: torch.device) -> dict:
    from torch.ao.quantization import allow_exported_model_train_eval
    from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torch.export import export_for_training

    train_loader, val_loader, test_loader, class_names = loaders
    set_seed(train_seed)

    # ---- start from the already-distilled fp32 student -----------------------
    student = build_model(
        model_name=cfg["student_model_name"],
        num_classes=len(class_names),
        pretrained=False,
    )
    ckpt = Path(cfg["checkpoint"].format(train_seed=train_seed))
    student.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    student.eval().cpu()

    fp32_acc, fp32_f1 = _evaluate(student, test_loader, torch.device("cpu"))

    # ---- insert fake quantization -------------------------------------------
    exported = export_for_training(
        copy.deepcopy(student).eval(), (torch.randn(1, 3, 224, 224),)
    ).module()
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_qat=True, is_per_channel=True)
    )
    qat_model = prepare_qat_pt2e(exported, quantizer)
    allow_exported_model_train_eval(qat_model)
    qat_model.to(device)

    # ---- optional teacher, so QAT keeps the distillation signal --------------
    teacher = None
    if cfg.get("teacher_checkpoint"):
        teacher = build_model(
            model_name=cfg["teacher_model_name"],
            num_classes=len(class_names),
            pretrained=False,
        )
        teacher.load_state_dict(
            torch.load(cfg["teacher_checkpoint"], map_location="cpu", weights_only=True)
        )
        teacher.eval().to(device)
        for p in teacher.parameters():
            p.requires_grad_(False)

    epochs = int(cfg.get("epochs", 5))
    optimizer = torch.optim.Adam(
        qat_model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-5)),   # small: we are fine-tuning
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    # Cosine decay to ~0. A flat LR left the int8 accuracy still climbing when the
    # budget ran out, so the reported number was a budget artifact rather than a
    # converged one; annealing lets the model actually settle.
    scheduler = None
    if cfg.get("cosine_schedule", True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    alpha = float(cfg.get("alpha", 0.7))
    temperature = float(cfg.get("temperature", 4.0))
    # Converting + evaluating on CPU costs about as much as a training epoch, so
    # only do it periodically (and always on the last epoch).
    val_every = int(cfg.get("val_every", 1))

    best_acc, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        qat_model.train()
        for images, labels in tqdm(train_loader, desc=f"  qat s{train_seed} e{epoch}/{epochs}",
                                   leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = qat_model(images)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(images)
                loss = _kd_loss(logits, t_logits, labels, alpha, temperature)
            else:
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if epoch % val_every and epoch != epochs:
            continue

        # Select on the *quantized* model, not the fake-quant one: that is what
        # actually ships, and the two can disagree.
        qat_model.eval()
        converted = convert_pt2e(copy.deepcopy(qat_model).cpu())
        allow_exported_model_train_eval(converted)
        acc, _ = _evaluate(converted, val_loader, torch.device("cpu"))
        history.append({"epoch": epoch, "int8_val_accuracy": acc})
        print(f"    epoch {epoch:3d}: int8 val acc = {acc:.2f}%  "
              f"(lr {optimizer.param_groups[0]['lr']:.2e})", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(qat_model).cpu().state_dict()
        qat_model.to(device)

    # ---- final int8 model ----------------------------------------------------
    qat_model.cpu()
    if best_state is not None:
        qat_model.load_state_dict(best_state)
    qat_model.eval()
    final = convert_pt2e(qat_model)
    allow_exported_model_train_eval(final)
    int8_acc, int8_f1 = _evaluate(final, test_loader, torch.device("cpu"))

    print(f"  seed {train_seed}: fp32 {fp32_acc:.2f}%  ->  QAT int8 {int8_acc:.2f}%", flush=True)
    return {
        "train_seed": train_seed,
        "fp32_accuracy": fp32_acc, "fp32_macro_f1": fp32_f1,
        "int8_accuracy": int8_acc, "int8_macro_f1": int8_f1,
        "recovered_pp": int8_acc - fp32_acc,
        # Kept so the reader can see whether the run converged or merely ran out
        # of budget -- the distinction that made our first QAT number misleading.
        "val_history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(cfg["output_dir"])

    loaders = get_dataloaders(
        data_dir=cfg["data_dir"],
        image_size=cfg.get("image_size", 224),
        batch_size=cfg.get("batch_size", 64),
        val_ratio=cfg.get("val_ratio", 0.15),
        test_ratio=cfg.get("test_ratio", 0.15),
        seed=cfg.get("seed", 42),
        num_workers=cfg.get("num_workers", 8),
    )

    print(f"\n{'='*66}\nQUANTIZATION-AWARE TRAINING  (device={device})\n{'='*66}")
    results = [run_qat(cfg, s, loaders, device) for s in cfg["train_seeds"]]

    fp = [r["fp32_accuracy"] for r in results]
    q8 = [r["int8_accuracy"] for r in results]
    summary = {
        "train_seeds": cfg["train_seeds"],
        "per_seed": results,
        "fp32_mean": mean(fp), "fp32_sd": stdev(fp) if len(fp) > 1 else 0.0,
        "int8_mean": mean(q8), "int8_sd": stdev(q8) if len(q8) > 1 else 0.0,
        "mean_gap_pp": mean(q8) - mean(fp),
    }
    print(f"\n{'='*66}")
    print(f"  fp32      : {summary['fp32_mean']:.2f} +/- {summary['fp32_sd']:.2f}")
    print(f"  QAT int8  : {summary['int8_mean']:.2f} +/- {summary['int8_sd']:.2f}")
    print(f"  gap       : {summary['mean_gap_pp']:+.2f} pp")
    print(f"  (PTQ int8 for the same students was 33.31 +/- 14.62)")
    save_json(summary, Path(output_dir) / "qat_results.json")
    print(f"\nSaved -> {Path(output_dir)/'qat_results.json'}")


if __name__ == "__main__":
    main()
