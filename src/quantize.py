"""Placeholder utilities for future quantization experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    """Parse quantization CLI arguments."""
    parser = argparse.ArgumentParser(description="Quantization workflow placeholder.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def prepare_quantization_plan(config: dict) -> dict:
    """Return a lightweight summary of the planned quantization setup.

    This function exists to keep the repository structure ready for later
    post-training quantization or quantization-aware training experiments.
    """
    quant_cfg = config.get("quantization", {})
    return {
        "mode": quant_cfg.get("mode", "post_training_dynamic"),
        "backend": quant_cfg.get("backend", "qnnpack"),
        "export_path": quant_cfg.get("export_path"),
    }


def main() -> None:
    """Print the configured quantization plan.

    TODO:
    - Decide whether to use post-training quantization, static quantization,
      or quantization-aware training for the final study.
    - Implement calibration or representative data handling.
    - Add model export and benchmark routines.
    """
    args = parse_args()
    config = load_yaml(args.config)
    plan = prepare_quantization_plan(config)
    print(plan)


if __name__ == "__main__":
    main()
