"""Batch-one inference latency benchmark.

Our first attempt at latency was unusable: two checkpoints of the *same*
architecture timed more than a factor of two apart, because measurements were
taken with training jobs resident on the GPU and summarized with a mean, which
one slow outlier is enough to wreck.

This benchmark is built so its own reliability is visible in the output:

  * median and inter-quartile range, not mean -- robust to scheduler outliers;
  * the entire measurement is repeated ``repeats`` times from scratch, and the
    spread *between* repeats is reported. If that spread is large, the number is
    not trustworthy and the table says so rather than hiding it;
  * explicit warm-up, CUDA synchronization, and a pinned thread count;
  * `torch.backends.cudnn.benchmark` enabled so autotuning happens during warm-up
    rather than polluting the timed region.

Run on an otherwise idle machine.

Usage:
    python -m src.benchmark_latency --config configs/benchmark_latency_cassava.yaml
"""

from __future__ import annotations

import argparse
import copy
import statistics as st
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.models import build_model
from src.utils import count_parameters, ensure_dir, load_yaml_config, save_json, set_seed


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_once(
    model: nn.Module, device: torch.device, image_size: int, warmup: int, iters: int
) -> list[float]:
    """One independent measurement: warm up, then time `iters` batch-one passes."""
    x = torch.randn(1, 3, image_size, image_size, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        _sync(device)

        samples: list[float] = []
        for _ in range(iters):
            _sync(device)
            t0 = time.perf_counter()
            model(x)
            _sync(device)
            samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def benchmark(
    model: nn.Module,
    device: torch.device,
    image_size: int,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, float]:
    """Repeat the whole measurement `repeats` times and expose the between-run spread."""
    medians: list[float] = []
    all_samples: list[float] = []
    for _ in range(repeats):
        s = _time_once(model, device, image_size, warmup, iters)
        medians.append(st.median(s))
        all_samples.extend(s)

    q = st.quantiles(all_samples, n=4)
    ordered = sorted(all_samples)
    return {
        "median_ms": st.median(all_samples),
        "iqr_ms": q[2] - q[0],
        "p95_ms": ordered[int(0.95 * len(ordered)) - 1],
        "min_ms": ordered[0],
        # The reliability check: how far apart are independent repeats of the
        # entire measurement? Small => the median is a real number.
        "between_run_median_spread_ms": max(medians) - min(medians),
        "repeat_medians_ms": [round(m, 3) for m in medians],
        "n_samples": len(all_samples),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(cfg.get("seed", 42))

    image_size = int(cfg.get("image_size", 224))
    warmup = int(cfg.get("warmup", 50))
    iters = int(cfg.get("iters", 300))
    repeats = int(cfg.get("repeats", 3))
    num_classes = int(cfg.get("num_classes", 5))
    threads = cfg.get("cpu_threads")
    out_dir = ensure_dir(cfg.get("output_dir", "outputs/latency_benchmark"))

    if threads:
        torch.set_num_threads(int(threads))
    torch.backends.cudnn.benchmark = True

    print(f"\n{'='*72}")
    print("BATCH-ONE LATENCY BENCHMARK  (run this on an idle machine)")
    print(f"  warmup={warmup}  iters={iters}  repeats={repeats}  "
          f"cpu_threads={torch.get_num_threads()}")
    print(f"{'='*72}")

    results: dict[str, Any] = {
        "image_size": image_size, "warmup": warmup, "iters": iters,
        "repeats": repeats, "cpu_threads": torch.get_num_threads(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models": {},
    }

    for entry in cfg["models"]:
        name = entry["name"]
        arch = entry["model_name"]
        ckpt = entry.get("checkpoint")

        model = build_model(model_name=arch, num_classes=num_classes, pretrained=False)
        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.eval()
        params = count_parameters(model)

        results["models"][name] = {"model_name": arch, "parameters": params, "devices": {}}
        print(f"\n{name}  ({arch}, {params:,} params)")

        for dev_name in entry.get("devices", ["cpu"]):
            if dev_name == "cuda" and not torch.cuda.is_available():
                continue
            device = torch.device(dev_name)
            m = copy.deepcopy(model).to(device)
            r = benchmark(m, device, image_size, warmup, iters, repeats)
            results["models"][name]["devices"][dev_name] = r
            print(f"  {dev_name:5s}  median {r['median_ms']:7.2f} ms   "
                  f"IQR {r['iqr_ms']:5.2f}   p95 {r['p95_ms']:7.2f}   "
                  f"between-run spread {r['between_run_median_spread_ms']:.2f} ms")
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Dynamic-int8 variant (CPU only; Linear layers only -- see src.quantize).
        if entry.get("dynamic_int8"):
            qm = torch.quantization.quantize_dynamic(
                copy.deepcopy(model).cpu(), {nn.Linear}, dtype=torch.qint8
            )
            r = benchmark(qm, torch.device("cpu"), image_size, warmup, iters, repeats)
            results["models"][name]["devices"]["cpu_dynamic_int8"] = r
            print(f"  {'cpu-i8':5s}  median {r['median_ms']:7.2f} ms   "
                  f"IQR {r['iqr_ms']:5.2f}   p95 {r['p95_ms']:7.2f}   "
                  f"between-run spread {r['between_run_median_spread_ms']:.2f} ms")

    save_json(results, Path(out_dir) / "latency.json")
    print(f"\nSaved -> {Path(out_dir) / 'latency.json'}\n")


if __name__ == "__main__":
    main()
