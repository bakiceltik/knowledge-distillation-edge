"""I/O helpers for configuration and experiment outputs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and resolve simple local ``defaults`` inheritance."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    defaults = config.pop("defaults", [])
    if not defaults:
        return config

    merged: dict[str, Any] = {}
    for default_name in defaults:
        default_path = config_path.parent / f"{default_name}.yaml"
        base_config = load_yaml(default_path)
        merged = _deep_merge(merged, base_config)

    return _deep_merge(merged, config)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write a JSON payload to disk."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    return destination
