"""Entry point for model evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import PlantVillageDataset
from src.data.transforms import build_eval_transforms
from src.engine.evaluator import Evaluator
from src.models.student import build_student_model
from src.models.teacher import build_teacher_model
from src.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluation entry point.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main() -> None:
    """Run the evaluation scaffold for a configured model."""
    args = parse_args()
    config = load_yaml(args.config)

    dataset = PlantVillageDataset(
        root=Path(config["data"]["root"]),
        split=config["data"]["test_split"],
        transform=build_eval_transforms(config["data"]["image_size"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["project"]["num_workers"],
    )

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "mobilenet_v3_small")
    role = model_cfg.get("role", "student")
    pretrained = model_cfg.get("pretrained", False)
    num_classes = len(dataset.classes)

    if role == "teacher":
        model = build_teacher_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )
    else:
        model = build_student_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )

    device = torch.device(
        "cuda" if config["project"]["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    # TODO: Load trained checkpoints before running final experiments.
    evaluator = Evaluator(device=device)
    metrics = evaluator.evaluate(model=model, dataloader=dataloader)
    print(metrics)


if __name__ == "__main__":
    main()
