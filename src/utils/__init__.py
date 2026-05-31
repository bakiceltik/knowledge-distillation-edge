"""Shared utilities exposed at the src.utils package level."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a flat YAML config file and return it as a dict."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Serialize *data* to a JSON file, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_compatible_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load only the checkpoint tensors that exist in *model* with matching shapes.

    This is useful for transfer experiments where the source and target tasks have
    different classifier sizes. Compatible backbone tensors are restored, while
    mismatched tensors such as the final classification layer are skipped.
    """
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    raw_state = torch.load(checkpoint, map_location=map_location)
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        state_dict = raw_state["state_dict"]
    else:
        state_dict = raw_state

    model_state = model.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    skipped_keys: list[str] = []

    for key, value in state_dict.items():
        if key not in model_state or model_state[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        compatible_state[key] = value

    model_state.update(compatible_state)
    model.load_state_dict(model_state)

    missing_keys = [key for key in model_state.keys() if key not in compatible_state]
    return {
        "checkpoint_path": str(checkpoint),
        "loaded_keys": sorted(compatible_state.keys()),
        "skipped_keys": sorted(skipped_keys),
        "missing_keys": missing_keys,
        "num_loaded_keys": len(compatible_state),
        "num_skipped_keys": len(skipped_keys),
    }
