"""Control experiments behind the int8-collapse claim.

Every number the paper uses to argue that the collapse is caused by the
*architecture* rather than by distillation is produced here, so that a reader who
clones the repo can regenerate it:

  teacher_control    ResNet50 quantized through the identical PT2E path, over
                     several calibration draws. This is the load-bearing control:
                     if the teacher survives and the student does not, the
                     pipeline is sound and the failure is architecture-specific.

  calibration_size   Does more calibration data rescue the student? Sweeps the
                     number of calibration images.

  activation_range   Peak absolute activation entering each convolution. Tests
                     the obvious hypothesis that MobileNetV3 fails because its
                     activations have a wider dynamic range than ResNet50's.

  layer_sensitivity  Quantizes the student one block-group at a time, everything
                     else left in float32. Distinguishes "the failure is localized
                     in a few layers" from "it is distributed across the network".

Usage:
    python -m src.quant_analysis --config configs/quant_analysis_cassava.yaml
"""

from __future__ import annotations

import argparse
import copy
import statistics as st
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.ao.quantization.quantizer.quantizer import Quantizer as _Quantizer

from src.data import get_dataloaders
from src.evaluate import evaluate_model
from src.models import build_model
from src.quantize import quantize_static_pt2e
from src.utils import ensure_dir, load_yaml_config, save_json, set_seed


def _load(model_name: str, ckpt: str, num_classes: int) -> nn.Module:
    m = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return m.eval().cpu()


def _acc(model: nn.Module, loader, class_names) -> tuple[float, float]:
    r = evaluate_model(model, loader, torch.device("cpu"), class_names)
    return r["accuracy"] * 100, r["macro_f1"] * 100


def _summary(vals: list[float]) -> dict[str, float]:
    return {
        "mean": st.mean(vals),
        "sd": st.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


# ---------------------------------------------------------------------------
# 1. Teacher control -- the load-bearing experiment
# ---------------------------------------------------------------------------

def teacher_control(cfg, train_loader, test_loader, class_names) -> dict[str, Any]:
    e = cfg["teacher_control"]
    seeds = [int(s) for s in e["calibration_seeds"]]
    batches = int(e.get("calibration_batches", 32))

    model = _load(e["model_name"], e["checkpoint"], len(class_names))
    fp32_acc, fp32_f1 = _acc(model, test_loader, class_names)
    print(f"\n[teacher_control] {e['model_name']}  float32 acc={fp32_acc:.2f}  f1={fp32_f1:.2f}")

    draws = []
    for s in seeds:
        qm, _ = quantize_static_pt2e(model, train_loader, batches, per_channel=True, calib_seed=s)
        a, f = _acc(qm, test_loader, class_names)
        draws.append({"calibration_seed": s, "accuracy": a, "macro_f1": f})
        print(f"  calib_seed={s}  int8 acc={a:.2f}  (drop {a - fp32_acc:+.2f})")

    accs = [d["accuracy"] for d in draws]
    out = {
        "model_name": e["model_name"],
        "float32": {"accuracy": fp32_acc, "macro_f1": fp32_f1},
        "int8_per_draw": draws,
        "int8": _summary(accs),
        "drop_mean": st.mean(accs) - fp32_acc,
    }
    print(f"  -> int8 {out['int8']['mean']:.2f} +/- {out['int8']['sd']:.2f}  "
          f"(drop {out['drop_mean']:+.2f} pts)")
    return out


# ---------------------------------------------------------------------------
# 2. Calibration-size sweep
# ---------------------------------------------------------------------------

def calibration_size(cfg, train_loader, test_loader, class_names, batch_size) -> dict[str, Any]:
    """Does more calibration data rescue the student?

    Each size must be measured over SEVERAL draws. A single draw per size is
    uninterpretable here: the draw-to-draw noise on this model (SD ~5 points) is
    larger than any plausible size effect, so a one-draw-per-size sweep will
    report a spread that is pure noise.
    """
    e = cfg["calibration_size"]
    model = _load(e["model_name"], e["checkpoint"], len(class_names))
    seeds = [int(s) for s in e.get("calibration_seeds", [e.get("calibration_seed", 1234)])]

    rows = []
    for nb in e["calibration_batches"]:
        accs = []
        for s in seeds:
            qm, _ = quantize_static_pt2e(model, train_loader, int(nb), per_channel=True, calib_seed=s)
            a, _f = _acc(qm, test_loader, class_names)
            accs.append(a)
        stats = _summary(accs)
        rows.append({"calibration_batches": int(nb),
                     "calibration_images": int(nb) * batch_size,
                     "accuracy": stats, "per_draw": accs})
        print(f"[calibration_size] {int(nb) * batch_size:5d} images  "
              f"int8 acc = {stats['mean']:.2f} +/- {stats['sd']:.2f}  "
              f"(min {stats['min']:.2f}, max {stats['max']:.2f})")

    means = [r["accuracy"]["mean"] for r in rows]
    span = max(means) - min(means)
    within = st.mean([r["accuracy"]["sd"] for r in rows])
    print(f"  -> spread of MEANS across sizes: {span:.2f} pts; "
          f"typical within-size SD: {within:.2f} pts")
    return {"rows": rows, "spread_of_means_pts": span, "mean_within_size_sd_pts": within,
            "calibration_seeds": seeds}


# ---------------------------------------------------------------------------
# 3. Activation-range comparison
# ---------------------------------------------------------------------------

def activation_range(cfg, train_loader, class_names) -> dict[str, Any]:
    e = cfg["activation_range"]
    batches = int(e.get("batches", 4))
    out = {}
    for m_cfg in e["models"]:
        model = _load(m_cfg["model_name"], m_cfg["checkpoint"], len(class_names))
        peaks: list[float] = []
        hooks = []

        def hook(_m, inp, _out):
            if isinstance(inp, tuple) and inp and torch.is_tensor(inp[0]):
                peaks.append(inp[0].abs().max().item())

        for mod in model.modules():
            if isinstance(mod, nn.Conv2d):
                hooks.append(mod.register_forward_hook(hook))
        with torch.no_grad():
            for i, (x, _) in enumerate(train_loader):
                if i >= batches:
                    break
                model(x)
        for h in hooks:
            h.remove()

        peak, med = max(peaks), st.median(peaks)
        out[m_cfg["model_name"]] = {
            "peak_abs_activation": peak,
            "median_abs_activation": med,
            # NOTE: this is one of several defensible "outlier-dominance" measures,
            # and they do not agree (see `_outlier_metrics` below and the mechanism
            # discussion in the paper). It is the ratio of the single largest
            # per-convolution peak to the median per-convolution peak.
            "outlier_ratio_peak_over_median": peak / med,
            "n_conv_inputs_observed": len(peaks),
            # Three competing definitions of the same intuition, recorded together
            # precisely because they disagree on direction.
            "alt_outlier_metrics": _outlier_metrics(model, train_loader, batches),
        }
        print(f"[activation_range] {m_cfg['model_name']:20s} "
              f"peak={peak:7.1f}  median={med:6.2f}  peak/median={peak / med:5.1f}x")
    return out


def _outlier_metrics(model: nn.Module, train_loader, batches: int) -> dict[str, float]:
    """Two activation-outlier measures that DISAGREE with the aggregate metric above.

    The aggregate `outlier_ratio_peak_over_median` (largest per-conv peak over the
    median per-conv peak) ranks the student as more outlier-heavy than the teacher.
    Both measures here instead look WITHIN each activation tensor, and both reverse
    that ranking:

      (b) peak-to-median within each conv-input tensor, pooled by median;
      (c) peak to 99.9th percentile within each tensor, pooled by median (robust to
          a single spike).

    Reported precisely because they contradict the aggregate: no explanation of the
    collapse can safely rest on "activation range" when reasonable definitions of it
    disagree on which network is worse.
    """
    within_ratios: list[float] = []
    p999_ratios: list[float] = []

    def hook(_m, inp, _out):
        if not (isinstance(inp, tuple) and inp and torch.is_tensor(inp[0])):
            return
        x = inp[0].detach().abs().flatten().float()
        med = x.median()
        if med > 1e-6:                      # skip mostly-zero tensors (median 0)
            within_ratios.append((x.max() / med).item())
        # quantile() caps tensor size; subsample large activation maps.
        xs = x[torch.randperm(x.numel())[:200_000]] if x.numel() > 200_000 else x
        q = torch.quantile(xs, 0.999)
        if q > 1e-6:
            p999_ratios.append((x.max() / q).item())

    hooks = [mod.register_forward_hook(hook)
             for mod in model.modules() if isinstance(mod, nn.Conv2d)]
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            if i >= batches:
                break
            model(x)
    for h in hooks:
        h.remove()

    return {
        "within_tensor_peak_over_median_median": st.median(within_ratios),
        "peak_over_p999_median": st.median(p999_ratios),
    }


# ---------------------------------------------------------------------------
# 4. Per-layer sensitivity -- is the failure localized, or distributed?
# ---------------------------------------------------------------------------

class _GroupOnlyQuantizer(_Quantizer):
    """Quantize exactly one block-group; leave the rest of the network in float32.

    The obvious implementation --- ``XNNPACKQuantizer().set_global(None)`` plus
    ``set_module_name(group, cfg)`` --- is silently wrong on this model. For
    ``features.0``, ``features.2`` and ``features.12`` it annotates *nothing*: the
    converted graph comes back with zero quantize nodes, i.e. bit-identical to
    float32, and the group is then misreported as costing 0.00 accuracy points.
    Those three blocks are quantized normally when the whole model is quantized
    (all 52 convolutions are), so the failure is in the module-name filter, not in
    the quantizer's coverage. Disabling by name is not an option either:
    ``set_module_name(name, None)`` raises ``NotImplementedError`` in XNNPACK.

    We therefore annotate globally --- the path that provably works and the one the
    full-model result uses --- and then strip the annotations from every node
    outside the target group. What remains is exactly the target group, quantized
    by the same code that quantizes the full network.
    """

    def __init__(self, prefixes: list[str]) -> None:
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer, get_symmetric_quantization_config,
        )
        super().__init__()
        self._inner = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True)
        )
        self._prefixes = prefixes

    def _in_group(self, node) -> bool:
        for path, _cls in node.meta.get("nn_module_stack", {}).values():
            p = str(path).replace("L['self'].", "")
            if any(p == pre or p.startswith(pre + ".") for pre in self._prefixes):
                return True
        return False

    def annotate(self, model):
        model = self._inner.annotate(model)
        for node in model.graph.nodes:
            if "quantization_annotation" in node.meta and not self._in_group(node):
                del node.meta["quantization_annotation"]
        return model

    def validate(self, model) -> None:  # required by the Quantizer protocol
        return None

    def transform_for_annotation(self, model):
        return self._inner.transform_for_annotation(model)


def layer_sensitivity(cfg, train_loader, test_loader, class_names) -> dict[str, Any]:
    """Quantize ONE block-group at a time; everything else stays float32.

    If a single group reproduces most of the collapse, the failure is localized
    and could be fixed by exempting that group. If no single group does much
    damage on its own while quantizing all of them is catastrophic, the failure
    is genuinely distributed -- which is the claim the paper needs to earn.
    """
    from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
    from torch.ao.quantization import allow_exported_model_train_eval
    from torch.export import export_for_training

    e = cfg["layer_sensitivity"]
    seeds = [int(s) for s in e.get("calibration_seeds", [e.get("calibration_seed", 1234)])]
    batches = int(e.get("calibration_batches", 32))
    model = _load(e["model_name"], e["checkpoint"], len(class_names))
    fp32_acc, _ = _acc(model, test_loader, class_names)

    # MobileNetV3-Small: features.0 .. features.12, then the classifier head.
    groups = {f"features.{i}": [f"features.{i}"] for i in range(13)}
    groups["classifier"] = ["classifier"]

    rows = []
    for gname, prefixes in groups.items():
        accs = []
        for seed in seeds:
            set_seed(seed)
            exported = export_for_training(
                copy.deepcopy(model).eval().cpu(), (torch.randn(1, 3, 224, 224),)
            ).module()

            prep = prepare_pt2e(exported, _GroupOnlyQuantizer(prefixes))
            with torch.no_grad():
                for i, (x, _) in enumerate(train_loader):
                    if i >= batches:
                        break
                    prep(x)
            qm = convert_pt2e(prep)

            # A group that ends up with no quantize nodes was never quantized, and
            # would be silently misreported as costing 0.0 accuracy. Fail loudly
            # instead: this is exactly how the first version of this experiment
            # under-tested features.0, features.2 and features.12.
            n_q = sum(1 for n in qm.graph.nodes if "quantize" in str(n.target))
            if n_q == 0:
                raise RuntimeError(
                    f"group '{gname}' produced 0 quantize nodes -- it was not "
                    f"quantized, so its accuracy drop would be meaningless."
                )

            allow_exported_model_train_eval(qm)
            a, _f = _acc(qm, test_loader, class_names)
            accs.append(a)

        stats = _summary(accs)
        rows.append({
            "group": gname,
            "accuracy": stats,
            "per_draw": accs,
            "drop_mean": fp32_acc - stats["mean"],
            "drop_sd": stats["sd"],
        })
        print(f"[layer_sensitivity] quantize only {gname:14s} -> "
              f"acc={stats['mean']:6.2f} +/- {stats['sd']:4.2f}  "
              f"(drop {fp32_acc - stats['mean']:+6.2f})")

    worst = max(rows, key=lambda r: r["drop_mean"])
    print(f"  -> worst single group: {worst['group']} "
          f"(drop {worst['drop_mean']:.2f} +/- {worst['drop_sd']:.2f})")
    return {
        "float32_accuracy": fp32_acc,
        "calibration_seeds": seeds,
        "rows": rows,
        "worst_single_group": worst,
        "max_single_group_drop": worst["drop_mean"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", nargs="*", help="subset of analyses to run")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(cfg.get("seed", 42))
    out_dir = ensure_dir(cfg.get("output_dir", "outputs/quant_analysis"))
    batch_size = int(cfg.get("batch_size", 32))

    train_loader, _v, test_loader, class_names = get_dataloaders(
        data_dir=cfg["data_dir"],
        image_size=cfg.get("image_size", 224),
        batch_size=batch_size,
        val_ratio=cfg.get("val_ratio", 0.15),
        test_ratio=cfg.get("test_ratio", 0.15),
        seed=cfg.get("seed", 42),
        num_workers=cfg.get("num_workers", 0),
    )

    todo = args.only or ["teacher_control", "calibration_size",
                         "activation_range", "layer_sensitivity"]

    # Merge into any existing results rather than overwriting: `--only X` must not
    # silently delete the other analyses from the artifact.
    out_path = Path(out_dir) / "quant_analysis.json"
    results: dict[str, Any] = {}
    if out_path.exists():
        import json as _json
        results = _json.loads(out_path.read_text())
        kept = [k for k in results if k not in todo]
        if kept:
            print(f"Merging into existing results; keeping: {', '.join(kept)}")

    if "teacher_control" in todo:
        results["teacher_control"] = teacher_control(cfg, train_loader, test_loader, class_names)
    if "calibration_size" in todo:
        results["calibration_size"] = calibration_size(
            cfg, train_loader, test_loader, class_names, batch_size)
    if "activation_range" in todo:
        results["activation_range"] = activation_range(cfg, train_loader, class_names)
    if "layer_sensitivity" in todo:
        results["layer_sensitivity"] = layer_sensitivity(
            cfg, train_loader, test_loader, class_names)

    save_json(results, Path(out_dir) / "quant_analysis.json")
    print(f"\nSaved -> {Path(out_dir) / 'quant_analysis.json'}")


if __name__ == "__main__":
    main()
