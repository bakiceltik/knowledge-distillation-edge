"""Post-training quantization for the distilled MobileNetV3-Small student.

Modes (set via YAML):
  post_training_static        — PT2E (torch.export) PTQ; weights and activations.
  post_training_static_mixed  — as above, but sensitive modules held in float32.
  post_training_dynamic       — eager-mode dynamic PTQ; Linear layers only.

Static PTQ uses the PT2E API rather than the legacy FX graph-mode API. On
torch 2.6, ``quantize_fx.convert_fx`` corrupts this model: converting with
``qconfig=None`` everywhere -- i.e. quantizing nothing -- still collapses test
accuracy from 84.2% to ~15%, so any number produced through that path measures
the bug, not quantization.

Note that eager-mode dynamic quantization supports only Linear/RNN layers;
passing ``nn.Conv2d`` in the module set is silently ignored. On a convolutional
network it therefore quantizes the classifier head alone and leaves the
backbone in float32, which the reported ``quantized_modules`` counts make
explicit.

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
    # PT2E exported graph modules reject .eval(); they are already in eval mode.
    try:
        model.eval()
    except NotImplementedError:
        pass
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(loader, desc="  calibrate", leave=False)):
            if i >= num_batches:
                break
            model(images)  # calibration always on CPU


# ---------------------------------------------------------------------------
# Quantization strategies
# ---------------------------------------------------------------------------

def quantize_static_pt2e(
    model: nn.Module,
    calibration_loader,
    num_batches: int,
    per_channel: bool = True,
    skip_se: bool = False,
    calib_seed: int = 1234,
) -> tuple[nn.Module, list[str]]:
    """PT2E post-training static quantization (weights + activations).

    With ``skip_se`` the squeeze-and-excitation blocks are held in float32 while
    the convolutions are still quantized, which isolates them as a candidate
    cause of the accuracy collapse.

    ``calib_seed`` is reseeded immediately before calibration rather than relying
    on the global seed set at start-up. The calibration loader shuffles and
    augments, so the draw depends on the RNG state at this exact point: any change
    to the code upstream would silently shift it, and on this model a different
    draw moves int8 accuracy by several points. Seeding here pins the reported
    numbers to the calibration set and nothing else.

    Returns the converted model and the list of modules held in float32.
    """
    from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torch.export import export_for_training
    from torchvision.ops.misc import SqueezeExcitation

    set_seed(calib_seed)

    example_inputs = (torch.randn(1, 3, 224, 224),)
    exported = export_for_training(
        copy.deepcopy(model).eval().cpu(), example_inputs
    ).module()

    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=per_channel)
    )

    held_in_float: list[str] = []
    if skip_se:
        quantizer = quantizer.set_module_type(SqueezeExcitation, None)
        held_in_float.append("SqueezeExcitation (all blocks)")

    prepared = prepare_pt2e(exported, quantizer)
    _calibrate(prepared, calibration_loader, num_batches)
    converted = convert_pt2e(prepared)

    # Exported graph modules raise on .train()/.eval(); shared evaluation helpers
    # call .eval(), so allow it explicitly.
    from torch.ao.quantization import allow_exported_model_train_eval

    allow_exported_model_train_eval(converted)
    return converted, held_in_float


def quantize_dynamic(model: nn.Module, backend: str) -> tuple[nn.Module, dict[str, int]]:
    """Eager-mode dynamic quantization.

    PyTorch supports dynamic quantization for Linear/RNN layers only, so on this
    convolutional student it quantizes the classifier head and leaves all conv
    layers in float32. The returned counts record exactly what was converted.
    """
    torch.backends.quantized.engine = backend
    quant_model = torch.quantization.quantize_dynamic(
        copy.deepcopy(model).eval().cpu(),
        {nn.Linear},
        dtype=torch.qint8,
    )
    counts = {
        "quantized_linear": sum(
            1 for m in quant_model.modules()
            if isinstance(m, torch.ao.nn.quantized.dynamic.Linear)
        ),
        "float_conv2d_remaining": sum(
            1 for m in quant_model.modules() if type(m) is nn.Conv2d
        ),
    }
    return quant_model, counts


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
    print(f"  Precision: {float_results['macro_precision']:.4f}")
    print(f"  Recall   : {float_results['macro_recall']:.4f}")
    print(f"  Macro-F1 : {float_results['macro_f1']:.4f}")
    print(f"  Latency  : {float_latency:.2f} ms")
    print(f"  Size     : {float_size:.2f} MB")

    # ---- quantize ------------------------------------------------------------
    print(f"\nQuantizing ({mode}, backend={backend})...")
    held_in_float: list[str] = []
    quantized_modules: dict[str, int] = {}
    per_channel = quant_cfg.get("per_channel", True)
    calib_seed = int(quant_cfg.get("calibration_seed", 1234))

    if mode in ("post_training_static", "post_training_static_mixed"):
        quant_model, held_in_float = quantize_static_pt2e(
            model,
            train_loader,
            calibration_batches,
            per_channel=per_channel,
            skip_se=(mode == "post_training_static_mixed"),
            calib_seed=calib_seed,
        )
        print(f"  PT2E static, per_channel={per_channel}, "
              f"calibration={calibration_batches * cfg.get('batch_size', 32)} images")
        for name in held_in_float:
            print(f"  Held in float32: {name}")
    elif mode == "post_training_dynamic":
        quant_model, quantized_modules = quantize_dynamic(model, backend)
        print(f"  Quantized Linear modules   : {quantized_modules['quantized_linear']}")
        print(f"  Conv2d left in float32     : {quantized_modules['float_conv2d_remaining']}")
    else:
        raise ValueError(f"Unknown quantization mode: '{mode}'")

    # ---- quantized evaluation (CPU) -----------------------------------------
    print("\nEvaluating quantized model on CPU...")
    quant_results = evaluate_model(quant_model, test_loader, cpu, class_names)
    quant_cm = quant_results.pop("confusion_matrix")
    quant_latency = measure_inference_latency(quant_model, cpu)
    quant_size = _model_size_mb(quant_model)

    print(f"  Accuracy : {quant_results['accuracy']:.4f}")
    print(f"  Precision: {quant_results['macro_precision']:.4f}")
    print(f"  Recall   : {quant_results['macro_recall']:.4f}")
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
        "per_channel": per_channel if mode.startswith("post_training_static") else None,
        "calibration_seed": calib_seed if mode.startswith("post_training_static") else None,
        "calibration_images": (
            calibration_batches * cfg.get("batch_size", 32)
            if mode.startswith("post_training_static") else None
        ),
        "modules_held_in_float": held_in_float,
        "quantized_modules": quantized_modules,
        "float32": {
            **float_results,
            "latency_ms": float_latency,
            "size_mb": float_size,
        },
        "quantized": {
            **quant_results,
            "latency_ms": quant_latency,
            "size_mb": quant_size,
            # PT2E emits a reference-quantized graph that simulates int8 with
            # explicit quantize/dequantize ops in float. Accuracy is faithful;
            # latency and serialized size are NOT deployment measurements.
            "simulated": mode.startswith("post_training_static"),
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
        "float32_macro_precision": float_results["macro_precision"],
        "quant_macro_precision": quant_results["macro_precision"],
        "float32_macro_recall": float_results["macro_recall"],
        "quant_macro_recall": quant_results["macro_recall"],
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
