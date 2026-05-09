"""PlantVillage data loading: subset filtering, splits, and DataLoaders."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.utils import load_yaml_config, set_seed


# ImageNet normalization constants
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _build_train_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def _build_eval_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_dataloaders(
    data_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 2,
    subset_prefix: Optional[str] = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train/val/test DataLoaders for PlantVillage (or a subset).

    Args:
        data_dir: Path to the PlantVillage root (contains one sub-dir per class).
        image_size: Square pixel size for all images.
        batch_size: Samples per batch.
        val_ratio: Fraction of data reserved for validation.
        test_ratio: Fraction of data reserved for test.
        seed: Random seed for reproducible splits.
        num_workers: DataLoader worker processes.
        subset_prefix: If provided, only classes whose folder name starts with
            this string are included (e.g. "Potato" keeps Potato___* classes).

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Please download PlantVillage from Kaggle and place it at the "
            "expected path. See data/README.md for instructions."
        )

    set_seed(seed)

    # Load full dataset with eval transforms first to get class list
    full_dataset = ImageFolder(root=str(data_dir), transform=_build_eval_transforms(image_size))

    if subset_prefix is not None:
        # Filter class indices to those matching the prefix
        matching = [
            idx for cls, idx in full_dataset.class_to_idx.items()
            if cls.startswith(subset_prefix)
        ]
        if not matching:
            raise ValueError(
                f"No classes found with prefix '{subset_prefix}' in {data_dir}.\n"
                f"Available classes: {list(full_dataset.class_to_idx.keys())}"
            )
        matching_set = set(matching)
        indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label in matching_set]
        class_names = [cls for cls, idx in sorted(full_dataset.class_to_idx.items(), key=lambda x: x[1]) if idx in matching_set]
    else:
        indices = list(range(len(full_dataset)))
        class_names = full_dataset.classes

    n = len(indices)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    # Reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = random_split(
        indices, [n_train, n_val, n_test], generator=generator
    )
    # random_split returns Subset of the indices list; extract actual dataset indices
    train_indices = [indices[i] for i in train_idx.indices]
    val_indices = [indices[i] for i in val_idx.indices]
    test_indices = [indices[i] for i in test_idx.indices]

    # Rebuild datasets with split-appropriate transforms
    train_dataset_full = ImageFolder(root=str(data_dir), transform=_build_train_transforms(image_size))
    eval_dataset_full = ImageFolder(root=str(data_dir), transform=_build_eval_transforms(image_size))

    train_subset = Subset(train_dataset_full, train_indices)
    val_subset = Subset(eval_dataset_full, val_indices)
    test_subset = Subset(eval_dataset_full, test_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names


# ---------------------------------------------------------------------------
# CLI: python -m src.data --config configs/baseline_resnet18.yaml
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Inspect PlantVillage dataset.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    data_dir = Path(cfg["data_dir"])
    subset_prefix = cfg.get("subset_prefix") or None

    print(f"Dataset path : {data_dir.resolve()}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        image_size=cfg.get("image_size", 224),
        batch_size=cfg.get("batch_size", 32),
        val_ratio=cfg.get("val_ratio", 0.15),
        test_ratio=cfg.get("test_ratio", 0.15),
        seed=cfg.get("seed", 42),
        num_workers=cfg.get("num_workers", 2),
        subset_prefix=subset_prefix,
    )

    print(f"Subset prefix: {subset_prefix}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val   samples: {len(val_loader.dataset)}")
    print(f"Test  samples: {len(test_loader.dataset)}")


if __name__ == "__main__":
    _cli()
