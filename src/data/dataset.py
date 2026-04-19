"""Dataset wrapper for PlantVillage experiments."""

from __future__ import annotations

from pathlib import Path

from torchvision.datasets import ImageFolder


class PlantVillageDataset(ImageFolder):
    """Thin wrapper around ``ImageFolder`` for PlantVillage classification.

    Expected directory layout:

    - ``data/plantvillage/train/<class_name>/*.jpg``
    - ``data/plantvillage/val/<class_name>/*.jpg``
    - ``data/plantvillage/test/<class_name>/*.jpg``

    TODO:
    - Support split generation when only a single raw dataset directory exists.
    - Add optional CSV-based split definitions for reproducible experiments.
    - Add dataset integrity checks and class-distribution summaries.
    """

    def __init__(self, root: str | Path, split: str, transform=None) -> None:
        self.root_dir = Path(root)
        self.split = split
        split_dir = self.root_dir / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Expected split directory '{split_dir}' was not found. "
                "Populate the dataset before running training or evaluation."
            )

        super().__init__(root=str(split_dir), transform=transform)
