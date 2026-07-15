"""Mixed-precision post-training quantization: keep the first blocks in float32.

The per-block sensitivity analysis (src.quant_analysis) shows the int8 collapse is
not distributed across the network at all. It is almost entirely produced by two
blocks -- the input convolution (features.0) and the first bottleneck
(features.1), which cost 50.2 and 42.2 accuracy points respectively when quantized
alone, while all twelve remaining blocks cost at most 0.16 points each.

This script tests the mitigation that follows directly from that: quantize
everything except those blocks. If the failure really is localized, PTQ should
recover -- which would mean an int8 student is reachable *without* the
quantization-aware retraining that our earlier results implied was mandatory.

Note the ordinary XNNPACK route cannot express this. ``set_module_name(name, None)``
raises, and ``set_global(None)`` + ``set_module_name(group, cfg)`` silently
annotates nothing for several blocks (which is what hid the culprit in the first
place). We therefore annotate globally -- the same code path the full-model
quantization uses -- and strip the annotations inside the exempted blocks.

Usage:
    python -m src.mixed_ptq --config configs/mixed_ptq_cassava.yaml
"""

from __future__ import annotations

import argparse
import copy
import statistics as st
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.ao.quantization import allow_exported_model_train_eval
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.quantizer import Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.export import export_for_training

from src.data import get_dataloaders
from src.models import build_model
from src.utils import ensure_dir, load_yaml_config, save_json, set_seed


class ExemptGroupsQuantizer(Quantizer):
    """Quantize the whole network except the named module prefixes."""

    def __init__(self, exempt: list[str]) -> None:
        super().__init__()
        self._inner = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True)
        )
        self._exempt = exempt

    def _is_exempt(self, node) -> bool:
        for path, _cls in node.meta.get("nn_module_stack", {}).values():
            p = str(path).replace("L['self'].", "")
            if any(p == e or p.startswith(e + ".") for e in self._exempt):
                return True
        return False

    def annotate(self, model):
        model = self._inner.annotate(model)
        for node in model.graph.nodes:
            if "quantization_annotation" in node.meta and self._is_exempt(node):
                del node.meta["quantization_annotation"]
        return model

    def validate(self, model) -> None:
        return None

    def transform_for_annotation(self, model):
        return self._inner.transform_for_annotation(model)


def _evaluate(model, loader) -> tuple[float, float]:
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x).argmax(1))
            ys.append(y)
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    return accuracy_score(y, p) * 100, f1_score(y, p, average="macro") * 100


def _quantize(model, exempt, train_loader, batches, calib_seed):
    set_seed(calib_seed)
    exported = export_for_training(
        copy.deepcopy(model).eval().cpu(), (torch.randn(1, 3, 224, 224),)
    ).module()
    quantizer = (
        ExemptGroupsQuantizer(exempt)
        if exempt
        else XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True)
        )
    )
    prep = prepare_pt2e(exported, quantizer)
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            if i >= batches:
                break
            prep(x)
    qm = convert_pt2e(prep)
    n_q = sum(1 for n in qm.graph.nodes if "quantize" in str(n.target))
    if n_q == 0:
        raise RuntimeError("nothing was quantized; the exemption filter is wrong")
    allow_exported_model_train_eval(qm)
    return qm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)

    output_dir = ensure_dir(cfg["output_dir"])
    calib_seeds = [int(s) for s in cfg["calibration_seeds"]]
    batches = int(cfg.get("calibration_batches", 32))
    variants: dict[str, list[str]] = {k: list(v) for k, v in cfg["variants"].items()}

    train_loader, _v, test_loader, class_names = get_dataloaders(
        data_dir=cfg["data_dir"],
        image_size=cfg.get("image_size", 224),
        batch_size=cfg.get("batch_size", 32),
        val_ratio=cfg.get("val_ratio", 0.15),
        test_ratio=cfg.get("test_ratio", 0.15),
        seed=cfg.get("seed", 42),
        num_workers=0,   # worker RNG would perturb the calibration draw
    )

    results: dict[str, Any] = {"variants": {}, "train_seeds": cfg["train_seeds"],
                               "calibration_seeds": calib_seeds}
    fp32_by_seed = {}

    for vname, exempt in variants.items():
        per_student = []
        for tseed in cfg["train_seeds"]:
            ckpt = Path(cfg["checkpoint"].format(train_seed=tseed))
            model = build_model(model_name=cfg["model_name"],
                                num_classes=len(class_names), pretrained=False)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            model.eval().cpu()
            if tseed not in fp32_by_seed:
                fp32_by_seed[tseed] = _evaluate(model, test_loader)[0]

            # Average over calibration draws: a single draw is not a measurement.
            accs = [
                _evaluate(_quantize(model, exempt, train_loader, batches, cs), test_loader)[0]
                for cs in calib_seeds
            ]
            per_student.append({
                "train_seed": tseed,
                "fp32_accuracy": fp32_by_seed[tseed],
                "int8_accuracy_mean": st.mean(accs),
                "int8_accuracy_sd": st.stdev(accs) if len(accs) > 1 else 0.0,
                "per_calibration_draw": accs,
            })
            print(f"  [{vname}] seed {tseed}: fp32 {fp32_by_seed[tseed]:.2f} -> "
                  f"int8 {st.mean(accs):.2f} +/- "
                  f"{st.stdev(accs) if len(accs) > 1 else 0.0:.2f}", flush=True)

        means = [p["int8_accuracy_mean"] for p in per_student]
        fps = [p["fp32_accuracy"] for p in per_student]
        results["variants"][vname] = {
            "exempt": exempt,
            "per_student": per_student,
            "int8_mean": st.mean(means),
            "int8_sd": st.stdev(means) if len(means) > 1 else 0.0,
            "fp32_mean": st.mean(fps),
            "gap_pp": st.mean(means) - st.mean(fps),
        }
        r = results["variants"][vname]
        print(f"== {vname}: int8 {r['int8_mean']:.2f} +/- {r['int8_sd']:.2f} "
              f"(fp32 {r['fp32_mean']:.2f}, gap {r['gap_pp']:+.2f})\n", flush=True)

    save_json(results, Path(output_dir) / "mixed_ptq.json")
    print(f"Saved -> {Path(output_dir) / 'mixed_ptq.json'}")


if __name__ == "__main__":
    main()
