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


# Independent repeats of the whole measurement must agree to within this
# fraction of the median, or the number is not certified.
RELIABILITY_TOLERANCE = 0.10

# Maximum CPU load from OTHER processes, averaged over the run. The
# between-repeat check alone is not sufficient: it verifies self-consistency, not
# correctness. A machine under steady background load produces timings that are
# consistently slow -- and therefore consistently wrong -- while sailing through
# the spread check. Measuring the interference directly is the only thing that
# catches that.
MAX_EXTERNAL_CPU_LOAD = 10.0  # percent


def _external_cpu_load(proc, interval: float = 0.5) -> float:
    """System-wide CPU load minus our own, in percent of one machine."""
    import psutil

    n = psutil.cpu_count() or 1
    system = psutil.cpu_percent(interval=interval)      # 0..100 of all cores
    ours = proc.cpu_percent(interval=None) / n          # normalize to same scale
    return max(0.0, system - ours)


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
    median = st.median(all_samples)
    spread = max(medians) - min(medians)

    # A measurement whose independent repeats disagree by more than this fraction
    # of the median is not a measurement. On a laptop with background services
    # running, batch-one CPU timings routinely fail this -- and a failing number
    # looks perfectly authoritative unless something says otherwise. It is far
    # better to refuse to certify it than to publish it.
    reliable = spread <= RELIABILITY_TOLERANCE * median

    return {
        "median_ms": median,
        "iqr_ms": q[2] - q[0],
        "p95_ms": ordered[int(0.95 * len(ordered)) - 1],
        "min_ms": ordered[0],
        # The reliability check: how far apart are independent repeats of the
        # entire measurement? Small => the median is a real number.
        "between_run_median_spread_ms": spread,
        "spread_as_fraction_of_median": spread / median,
        "reliable": reliable,
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

    import psutil
    proc = psutil.Process()
    proc.cpu_percent(interval=None)          # prime the counter
    load_before = _external_cpu_load(proc, interval=1.0)
    print(f"\nExternal CPU load before start: {load_before:.1f}% "
          f"(must stay under {MAX_EXTERNAL_CPU_LOAD:.0f}% for CPU timings to count)")
    if load_before > MAX_EXTERNAL_CPU_LOAD:
        print("  WARNING: the machine is NOT quiet. CPU timings from this run will "
              "be flagged unreliable.")
    loads = [load_before]

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
            flag = "" if r["reliable"] else "   *** UNRELIABLE - DO NOT PUBLISH ***"
            print(f"  {dev_name:5s}  median {r['median_ms']:7.2f} ms   "
                  f"IQR {r['iqr_ms']:5.2f}   p95 {r['p95_ms']:7.2f}   "
                  f"spread {r['between_run_median_spread_ms']:6.2f} ms "
                  f"({100 * r['spread_as_fraction_of_median']:4.1f}%){flag}")
            loads.append(_external_cpu_load(proc, interval=0.3))
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
            flag = "" if r["reliable"] else "   *** UNRELIABLE - DO NOT PUBLISH ***"
            print(f"  {'cpu-i8':5s}  median {r['median_ms']:7.2f} ms   "
                  f"IQR {r['iqr_ms']:5.2f}   p95 {r['p95_ms']:7.2f}   "
                  f"spread {r['between_run_median_spread_ms']:6.2f} ms "
                  f"({100 * r['spread_as_fraction_of_median']:4.1f}%){flag}")

    mean_load = st.mean(loads)
    machine_quiet = mean_load <= MAX_EXTERNAL_CPU_LOAD
    results["external_cpu_load_percent_mean"] = mean_load
    results["external_cpu_load_percent_max"] = max(loads)
    results["machine_quiet"] = machine_quiet

    # A CPU timing is only certified if BOTH the repeats agree AND the machine was
    # quiet. GPU timings are largely immune to CPU-side contention, so they are
    # judged on the spread alone.
    for m in results["models"].values():
        for dev, r in m["devices"].items():
            if dev.startswith("cpu") and not machine_quiet:
                r["reliable"] = False
                r["unreliable_reason"] = (
                    f"external CPU load {mean_load:.1f}% > {MAX_EXTERNAL_CPU_LOAD:.0f}%"
                )

    print(f"\nExternal CPU load during run: mean {mean_load:.1f}%, "
          f"peak {max(loads):.1f}%  -> machine "
          f"{'quiet' if machine_quiet else 'NOT quiet'}")

    bad = [
        f"{name} [{dev}]"
        for name, m in results["models"].items()
        for dev, r in m["devices"].items()
        if not r["reliable"]
    ]
    results["all_reliable"] = not bad
    results["unreliable_rows"] = bad
    results["reliability_tolerance"] = RELIABILITY_TOLERANCE

    save_json(results, Path(out_dir) / "latency.json")
    print(f"\nSaved -> {Path(out_dir) / 'latency.json'}")
    if bad:
        print("\n" + "!" * 72)
        print(f"{len(bad)} row(s) FAILED the reliability check and must NOT be published:")
        for b in bad:
            print(f"  - {b}")
        if not machine_quiet:
            print(f"\nCause: other processes were using the CPU ({mean_load:.1f}%).")
            print("Close background applications and re-run.")
        else:
            # This is the harder case, and the one we actually hit: an idle machine
            # can still fail if the CPU's clock is not stable. On a thermally
            # constrained laptop, sustained inference makes the clock oscillate, and
            # neither the median (inflated by throttling) nor the minimum (a lucky
            # boost burst) estimates steady-state latency. Cooling is the fix, not
            # closing applications.
            print(f"\nThe machine WAS quiet ({mean_load:.1f}% external load), so this is")
            print("clock instability, not contention -- most likely thermal throttling")
            print("after sustained load. Let the machine cool and re-run; if it still")
            print("fails, this hardware cannot measure these models reliably and the")
            print("affected rows must be omitted rather than published.")
        print("!" * 72 + "\n")
    else:
        print("\nAll rows passed the reliability check.\n")


if __name__ == "__main__":
    main()
