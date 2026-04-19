"""Entry point for supervised model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import PlantVillageDataset
from src.data.transforms import build_eval_transforms, build_train_transforms
from src.engine.trainer import Trainer
from src.models.student import build_student_model
from src.models.teacher import build_teacher_model
from src.utils.io import ensure_dir, load_yaml
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for supervised training."""
    parser = argparse.ArgumentParser(description="Supervised training entry point.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def build_model(config: dict, num_classes: int) -> nn.Module:
    """Instantiate a teacher or student model from config metadata."""
    role = config.get("model", {}).get("role", "student")
    model_name = config.get("model", {}).get("name", "mobilenet_v3_small")
    pretrained = config.get("model", {}).get("pretrained", True)

    if role == "teacher":
        return build_teacher_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )

    return build_student_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )


def main() -> None:
    """Run the supervised training scaffold."""
    args = parse_args()
    config = load_yaml(args.config)

    seed_everything(config["project"]["seed"])
    ensure_dir(config["logging"]["output_dir"])
    ensure_dir(config["logging"]["checkpoint_dir"])

    image_size = config["data"]["image_size"]
    batch_size = config["train"]["batch_size"]
    data_root = Path(config["data"]["root"])

    train_dataset = PlantVillageDataset(
        root=data_root,
        split=config["data"]["train_split"],
        transform=build_train_transforms(image_size),
    )
    val_dataset = PlantVillageDataset(
        root=data_root,
        split=config["data"]["val_split"],
        transform=build_eval_transforms(image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["project"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["project"]["num_workers"],
    )

    num_classes = len(train_dataset.classes)
    model = build_model(config=config, num_classes=num_classes)

    device = torch.device(
        "cuda" if config["project"]["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=Path(config["logging"]["checkpoint_dir"]),
        log_interval=config["logging"]["log_interval"],
    )

    # TODO: Add scheduler support, checkpoint resume logic, and experiment logging.
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["train"]["epochs"],
    )


if __name__ == "__main__":
    main()
