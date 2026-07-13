"""Seed-replicate study: is the int8 collapse caused by distillation, or the architecture?

Trains supervised and distilled MobileNetV3-Small students under N training seeds,
quantizes each through the PT2E path, and runs a paired t-test on the per-seed
differences.

Design notes (these are what make the comparison valid):

  * The data split is held FIXED (``seed``). Only ``train_seed`` varies, changing
    weight initialization and batch order. This keeps the teacher --- trained on
    the fixed split --- valid for every replicate, with no test leakage.

  * Both arms use IDENTICAL training budgets, so the only difference between them
    is the loss (CE vs KD+CE). Without this, a quantizability gap could be an
    artifact of epochs or batch size rather than distillation.

Reproduces Table `tab:quant-arch` of the report.

Usage:
    python -m src.replicate_quant --config configs/replicate_quant_cassava.yaml
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

import torch
import yaml
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

from src.data import get_dataloaders
from src.models import build_model
from src.utils import ensure_dir, load_yaml_config, save_json, set_seed

ARMS = ("supervised", "distilled")
_MODULE = {"supervised": "src.train_baseline", "distilled": "src.train_distillation"}


def _arm_config(cfg: dict, arm: str, train_seed: int, cfg_dir: Path) -> Path:
    """Materialize a per-arm, per-seed training config."""
    d = {
        "experiment_name": f"{arm}_seed{train_seed}",
        "output_dir": f"{cfg['output_dir']}/{arm}_seed{train_seed}",
        "data_dir": cfg["data_dir"],
        "subset_prefix": None,
        "pretrained": True,
        "image_size": cfg.get("image_size", 224),
        # Matched across arms -- see module docstring.
        "batch_size": cfg["batch_size"],
        "epochs": cfg["epochs"],
        "learning_rate": cfg["learning_rate"],
        "weight_decay": cfg.get("weight_decay", 1e-4),
        "val_ratio": cfg.get("val_ratio", 0.15),
        "test_ratio": cfg.get("test_ratio", 0.15),
        "seed": cfg.get("seed", 42),          # split seed -- fixed
        "train_seed": train_seed,             # varies
        "num_workers": cfg.get("num_workers", 8),
    }
    if arm == "supervised":
        d["model_name"] = cfg["student_model_name"]
    else:
        d["student_model_name"] = cfg["student_model_name"]
        d["teacher_model_name"] = cfg["teacher_model_name"]
        d["teacher_checkpoint"] = cfg["teacher_checkpoint"]
        d["temperature"] = cfg["temperature"]
        d["alpha"] = cfg["alpha"]

    path = cfg_dir / f"{arm}_seed{train_seed}.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False))
    return path


def _evaluate(model: torch.nn.Module, loader) -> tuple[float, float]:
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x).argmax(1))
            ys.append(y)
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    return accuracy_score(y, p) * 100, f1_score(y, p, average="macro") * 100


def _quantize(
    model: torch.nn.Module, calib_loader, num_batches: int, calib_seed: int
) -> torch.nn.Module:
    """PT2E static quantization.

    ``calib_seed`` is essential, not cosmetic: the calibration loader shuffles and
    augments, so an unseeded run draws a different calibration set each time and
    derives different quantization parameters. On this model that alone moves int8
    accuracy by tens of points, which would make the reported figures
    irreproducible.
    """
    from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torch.export import export_for_training

    set_seed(calib_seed)

    exported = export_for_training(
        copy.deepcopy(model).eval().cpu(), (torch.randn(1, 3, 224, 224),)
    ).module()
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True)
    )
    prepared = prepare_pt2e(exported, quantizer)
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            if i >= num_batches:
                break
            prepared(x)
    return convert_pt2e(prepared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    output_dir = ensure_dir(cfg["output_dir"])
    cfg_dir = ensure_dir(Path(output_dir) / "configs")
    seeds = list(cfg["train_seeds"])
    calib_batches = int(cfg.get("calibration_batches", 32))
    calib_seed = int(cfg.get("calibration_seed", 1234))

    root = Path(__file__).resolve().parent.parent

    # ---- train every replicate (skipping any already done) --------------------
    for seed in seeds:
        for arm in ARMS:
            ckpt = Path(cfg["output_dir"]) / f"{arm}_seed{seed}" / "best_model.pth"
            if ckpt.exists():
                print(f"[skip] {arm} seed={seed} already trained")
                continue
            path = _arm_config(cfg, arm, seed, cfg_dir)
            print(f"\n===== TRAIN {arm} seed={seed} =====", flush=True)
            r = subprocess.run(
                [sys.executable, "-m", _MODULE[arm], "--config", str(path)], cwd=root
            )
            if r.returncode != 0:
                raise SystemExit(f"training failed: {arm} seed={seed}")

    # ---- quantize and evaluate every replicate --------------------------------
    train_loader, _val, test_loader, class_names = get_dataloaders(
        data_dir=cfg["data_dir"],
        image_size=cfg.get("image_size", 224),
        batch_size=32,
        val_ratio=cfg.get("val_ratio", 0.15),
        test_ratio=cfg.get("test_ratio", 0.15),
        seed=cfg.get("seed", 42),
        num_workers=0,
    )

    rows: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for seed in seeds:
            ckpt = Path(cfg["output_dir"]) / f"{arm}_seed{seed}" / "best_model.pth"
            model = build_model(
                model_name=cfg["student_model_name"],
                num_classes=len(class_names),
                pretrained=False,
            )
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            model.eval().cpu()

            fp32_acc, fp32_f1 = _evaluate(model, test_loader)
            int8_acc, int8_f1 = _evaluate(
                _quantize(model, train_loader, calib_batches, calib_seed), test_loader
            )
            rows[arm].append({
                "train_seed": seed,
                "fp32_accuracy": fp32_acc, "fp32_macro_f1": fp32_f1,
                "int8_accuracy": int8_acc, "int8_macro_f1": int8_f1,
                "drop": fp32_acc - int8_acc,
            })
            print(f"{arm:11s} seed{seed}  fp32={fp32_acc:6.2f}  int8={int8_acc:6.2f}  "
                  f"drop={fp32_acc - int8_acc:6.2f}", flush=True)

    # ---- paired significance ---------------------------------------------------
    summary: dict[str, object] = {
        "train_seeds": seeds,
        "calibration_seed": calib_seed,
        "per_seed": rows,
        "arms": {},
    }
    for arm in ARMS:
        fp = [r["fp32_accuracy"] for r in rows[arm]]
        q8 = [r["int8_accuracy"] for r in rows[arm]]
        summary["arms"][arm] = {
            "fp32_mean": mean(fp), "fp32_sd": stdev(fp),
            "int8_mean": mean(q8), "int8_sd": stdev(q8),
        }

    summary["paired"] = {}
    for key, field in (("float32", "fp32_accuracy"), ("int8", "int8_accuracy")):
        diffs = [d[field] - s[field] for d, s in zip(rows["distilled"], rows["supervised"])]
        t, p = stats.ttest_rel(diffs, [0.0] * len(diffs))
        summary["paired"][key] = {
            "mean_diff_pp": mean(diffs), "sd_pp": stdev(diffs),
            "t": float(t), "p": float(p), "significant": bool(p < 0.05),
        }

    print("\n" + "=" * 68)
    for arm in ARMS:
        a = summary["arms"][arm]
        print(f"{arm:11s}  fp32 {a['fp32_mean']:.2f} +/- {a['fp32_sd']:.2f}   "
              f"int8 {a['int8_mean']:.2f} +/- {a['int8_sd']:.2f}")
    for key in ("float32", "int8"):
        s = summary["paired"][key]
        print(f"paired diff (distilled - supervised), {key:7s}: "
              f"{s['mean_diff_pp']:+.2f} +/- {s['sd_pp']:.2f} pp  "
              f"t={s['t']:.3f}  p={s['p']:.4f}  "
              f"{'significant' if s['significant'] else 'NOT significant'}")

    save_json(summary, Path(output_dir) / "replicate_quant_results.json")
    print(f"\nSaved -> {Path(output_dir) / 'replicate_quant_results.json'}")


if __name__ == "__main__":
    main()
