"""Post-training quantization for the distilled MobileNetV3-Small student.

Supports two modes configured via YAML:
  post_training_static  — FX-graph-mode PTQ; best accuracy/latency tradeoff.
  post_training_dynamic — quantizes weights only; no calibration data needed.

Usage:
    python -m src.quantize --config configs/quantization_cassava.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import get_dataloaders
from src.evaluate import evaluate_model, measure_inference_latency, plot_confusion_matrix
from src.models import build_model
from src.utils import count_parameters, ensure_dir, load_yaml_config, save_json, set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_size_mb(model: nn.Module) -> float:
    """Return serialized state-dict size in MB (accurate for quantized models)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


def _calibrate(model: nn.Module, loader, num_batches: int) -> None:
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(loader, desc="  calibrate", leave=False)):
            if i >= num_batches:
                break
            model(images)  # calibration always on CPU


# ---------------------------------------------------------------------------
# Quantization strategies
# ---------------------------------------------------------------------------

def quantize_static(
    model: nn.Module,
    calibration_loader,
    backend: str,
    num_batches: int,
) -> nn.Module:
    """FX-graph-mode post-training static quantization."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    torch.backends.quantized.engine = backend
    qconfig_mapping = get_default_qconfig_mapping(backend)
    example_inputs = (torch.randn(1, 3, 224, 224),)

    float_copy = copy.deepcopy(model).eval().cpu()
    prepared = prepare_fx(float_copy, qconfig_mapping, example_inputs)
    _calibrate(prepared, calibration_loader, num_batches)
    return convert_fx(prepared)


def quantize_dynamic(model: nn.Module, backend: str) -> nn.Module:
    """Dynamic quantization — weights only, no calibration needed."""
    torch.backends.quantized.engine = backend
    return torch.quantization.quantize_dynamic(
        copy.deepcopy(model).eval().cpu(),
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training quantization.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(cfg.get("seed", 42))

    output_dir = ensure_dir(cfg.get("output_dir", "outputs/student_quantized"))
    quant_cfg = cfg.get("quantization", {})
    mode = quant_cfg.get("mode", "post_training_static")
    backend = quant_cfg.get("backend", "fbgemm")
    calibration_batches = quant_cfg.get("calibration_batches", 8)

    print(f"\n{'='*60}")
    print(f"Experiment : {cfg.get('experiment_name', 'quantization')}")
    print(f"Mode       : {mode}  backend={backend}")
    print(f"Output     : {output_dir}")
    print(f"{'='*60}")

    # ---- data ----------------------------------------------------------------
    train_loader, _val, test_loader, class_names = get_dataloaders(
        data_dir=cfg["data_dir"],
        image_size=cfg.get("image_size", 224),
        batch_size=cfg.get("batch_size", 32),
        val_ratio=cfg.get("val_ratio", 0.15),
        test_ratio=cfg.get("test_ratio", 0.15),
        seed=cfg.get("seed", 42),
        num_workers=cfg.get("num_workers", 2),
        subset_prefix=cfg.get("subset_prefix") or None,
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    # ---- load float32 checkpoint --------------------------------------------
    ckpt_path = Path(cfg["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Re-run the corresponding training script to regenerate it."
        )

    model = build_model(
        model_name=cfg.get("model_name", "mobilenet_v3_small"),
        num_classes=len(class_names),
        pretrained=False,
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True)
    )
    model.eval().cpu()

    cpu = torch.device("cpu")

    # ---- float32 baseline (CPU) ----------------------------------------------
    print("\nEvaluating float32 model on CPU...")
    float_results = evaluate_model(model, test_loader, cpu, class_names)
    float_cm = float_results.pop("confusion_matrix")
    float_latency = measure_inference_latency(model, cpu)
    float_size = _model_size_mb(model)
    float_params = count_parameters(model)

    print(f"  Accuracy : {float_results['accuracy']:.4f}")
    print(f"  Macro-F1 : {float_results['macro_f1']:.4f}")
    print(f"  Latency  : {float_latency:.2f} ms")
    print(f"  Size     : {float_size:.2f} MB")

    # ---- quantize ------------------------------------------------------------
    print(f"\nQuantizing ({mode}, backend={backend})...")
    if mode == "post_training_static":
        quant_model = quantize_static(model, train_loader, backend, calibration_batches)
    elif mode == "post_training_dynamic":
        quant_model = quantize_dynamic(model, backend)
    else:
        raise ValueError(f"Unknown quantization mode: '{mode}'")

    # ---- quantized evaluation (CPU) -----------------------------------------
    print("\nEvaluating quantized model on CPU...")
    quant_results = evaluate_model(quant_model, test_loader, cpu, class_names)
    quant_cm = quant_results.pop("confusion_matrix")
    quant_latency = measure_inference_latency(quant_model, cpu)
    quant_size = _model_size_mb(quant_model)

    print(f"  Accuracy : {quant_results['accuracy']:.4f}")
    print(f"  Macro-F1 : {quant_results['macro_f1']:.4f}")
    print(f"  Latency  : {quant_latency:.2f} ms")
    print(f"  Size     : {quant_size:.2f} MB")

    # ---- save quantized model -----------------------------------------------
    quant_path = output_dir / "quantized_model.pth"
    torch.save(quant_model.state_dict(), quant_path)

    # ---- compute deltas -----------------------------------------------------
    acc_delta = quant_results["accuracy"] - float_results["accuracy"]
    latency_speedup = float_latency / quant_latency if quant_latency > 0 else None
    size_reduction = float_size / quant_size if quant_size > 0 else None

    # ---- save artifacts -------------------------------------------------------
    metrics: dict[str, Any] = {
        "experiment_name": cfg.get("experiment_name"),
        "mode": mode,
        "backend": backend,
        "model_name": cfg.get("model_name"),
        "num_classes": len(class_names),
        "class_names": class_names,
        "parameters": float_params,
        "float32": {
            **float_results,
            "latency_ms": float_latency,
            "size_mb": float_size,
        },
        "quantized": {
            **quant_results,
            "latency_ms": quant_latency,
            "size_mb": quant_size,
        },
        "accuracy_delta": acc_delta,
        "latency_speedup": latency_speedup,
        "size_reduction": size_reduction,
    }
    save_json(metrics, output_dir / "metrics.json")

    plot_confusion_matrix(float_cm, class_names, output_dir / "confusion_matrix_float32.png")
    plot_confusion_matrix(quant_cm, class_names, output_dir / "confusion_matrix_quantized.png")

    results_row = {
        "experiment_name": cfg.get("experiment_name"),
        "model_name": cfg.get("model_name"),
        "mode": mode,
        "backend": backend,
        "num_classes": len(class_names),
        "float32_accuracy": float_results["accuracy"],
        "quant_accuracy": quant_results["accuracy"],
        "accuracy_delta": acc_delta,
        "float32_macro_f1": float_results["macro_f1"],
        "quant_macro_f1": quant_results["macro_f1"],
        "float32_latency_ms": float_latency,
        "quant_latency_ms": quant_latency,
        "latency_speedup": latency_speedup,
        "float32_size_mb": float_size,
        "quant_size_mb": quant_size,
        "size_reduction": size_reduction,
        "parameters": float_params,
    }
    with (output_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_row.keys())
        writer.writeheader()
        writer.writerow(results_row)

    # ---- final summary -------------------------------------------------------
    sign = "+" if acc_delta >= 0 else ""
    print(f"\n{'='*60}")
    print("QUANTIZATION COMPLETE")
    print(f"  Mode          : {mode} ({backend})")
    print(f"  Float32   acc={float_results['accuracy']:.4f}  "
          f"lat={float_latency:.2f}ms  size={float_size:.2f}MB")
    print(f"  Quantized acc={quant_results['accuracy']:.4f}  "
          f"lat={quant_latency:.2f}ms  size={quant_size:.2f}MB")
    print(f"  Accuracy delta : {sign}{acc_delta:.4f}")
    if latency_speedup:
        print(f"  Latency speedup: {latency_speedup:.2f}x")
    if size_reduction:
        print(f"  Size reduction : {size_reduction:.2f}x")
    print(f"  Outputs saved  : {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
