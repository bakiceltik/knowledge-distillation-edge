"""Per-block quantization sensitivity, replicated across the distilled students.

The single-checkpoint version (src.quant_analysis) establishes that quantizing
features.0 or features.1 alone reproduces almost the whole int8 collapse while the
other twelve blocks are harmless. This script repeats that measurement on every one
of the five distilled replicate students, so the per-block drops carry a standard
deviation across training seeds rather than resting on one checkpoint.

For each (student, block) we quantize only that block through the same annotate-
globally-then-strip-outside path used elsewhere (the naive set_module_name filter
silently fails to quantize features.0/2/12), averaging over a few calibration
draws, and then aggregate across students.

Usage:
    python -m src.layer_sensitivity_replicate --config configs/layer_sensitivity_replicate_cassava.yaml
"""

from __future__ import annotations

import argparse
import copy
import statistics as st
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score
from torch.ao.quantization import allow_exported_model_train_eval
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.export import export_for_training

from src.data import get_dataloaders
from src.models import build_model
from src.quant_analysis import _GroupOnlyQuantizer
from src.utils import ensure_dir, load_yaml_config, save_json, set_seed


def _acc(model, loader) -> float:
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x).argmax(1))
            ys.append(y)
    return accuracy_score(torch.cat(ys).numpy(), torch.cat(ps).numpy()) * 100


def _quantize_group(model, prefixes, train_loader, batches, calib_seed):
    set_seed(calib_seed)
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
    if sum(1 for n in qm.graph.nodes if "quantize" in str(n.target)) == 0:
        raise RuntimeError(f"group {prefixes} produced no quantize nodes")
    allow_exported_model_train_eval(qm)
    return qm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)

    output_dir = ensure_dir(cfg["output_dir"])
    train_seeds = cfg["train_seeds"]
    calib_seeds = [int(s) for s in cfg["calibration_seeds"]]
    batches = int(cfg.get("calibration_batches", 32))

    train_loader, _v, test_loader, class_names = get_dataloaders(
        data_dir=cfg["data_dir"], image_size=cfg.get("image_size", 224),
        batch_size=cfg.get("batch_size", 32),
        val_ratio=cfg.get("val_ratio", 0.15), test_ratio=cfg.get("test_ratio", 0.15),
        seed=cfg.get("seed", 42), num_workers=0,
    )
    groups = [f"features.{i}" for i in range(13)] + ["classifier"]

    # per_student_drop[group] = list of one mean drop per student
    per_student_drop: dict[str, list[float]] = {g: [] for g in groups}
    fp32_by_seed: dict[int, float] = {}

    for tseed in train_seeds:
        ckpt = Path(cfg["checkpoint"].format(train_seed=tseed))
        model = build_model(model_name=cfg["model_name"],
                            num_classes=len(class_names), pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.eval().cpu()
        fp32 = _acc(model, test_loader)
        fp32_by_seed[tseed] = fp32
        print(f"\n== student seed {tseed}  (fp32 {fp32:.2f}%) ==", flush=True)

        for g in groups:
            draws = [_acc(_quantize_group(model, [g], train_loader, batches, cs), test_loader)
                     for cs in calib_seeds]
            drop = fp32 - st.mean(draws)
            per_student_drop[g].append(drop)
            print(f"   {g:12s} drop {drop:6.2f}  (int8 {st.mean(draws):.2f})", flush=True)

    rows = []
    for g in groups:
        drops = per_student_drop[g]
        rows.append({
            "group": g,
            "drop_mean": st.mean(drops),
            "drop_sd": st.stdev(drops) if len(drops) > 1 else 0.0,
            "drop_min": min(drops),
            "drop_max": max(drops),
            "per_student_drop": drops,
        })
    rows.sort(key=lambda r: r["drop_mean"], reverse=True)

    print(f"\n{'='*60}\nPER-BLOCK SENSITIVITY across {len(train_seeds)} students "
          f"(x{len(calib_seeds)} calib draws)\n{'='*60}")
    print(f"{'block':12s} {'drop mean':>10s} {'sd':>7s} {'[min,max]':>16s}")
    for r in rows:
        print(f"{r['group']:12s} {r['drop_mean']:10.2f} {r['drop_sd']:7.2f} "
              f"  [{r['drop_min']:.2f},{r['drop_max']:.2f}]")

    out = {
        "train_seeds": train_seeds, "calibration_seeds": calib_seeds,
        "fp32_by_seed": fp32_by_seed, "rows": rows,
    }
    save_json(out, Path(output_dir) / "layer_sensitivity_replicate.json")
    print(f"\nSaved -> {Path(output_dir) / 'layer_sensitivity_replicate.json'}")


if __name__ == "__main__":
    main()
