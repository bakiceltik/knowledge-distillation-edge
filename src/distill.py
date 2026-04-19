"""Entry point for future knowledge distillation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import PlantVillageDataset
from src.data.transforms import build_eval_transforms, build_train_transforms
from src.engine.trainer import DistillationTrainer
from src.models.student import build_student_model
from src.models.teacher import build_teacher_model
from src.utils.io import load_yaml
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for distillation experiments."""
    parser = argparse.ArgumentParser(description="Knowledge distillation entry point.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """Combine hard-label supervision and softened teacher targets.

    TODO:
    - Validate the exact reduction choice.
    - Compare KL divergence and cross-entropy variants.
    - Log the hard and soft loss terms separately during experiments.
    """
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(student_log_probs, soft_targets, reduction="batchmean")
    hard_loss = F.cross_entropy(student_logits, targets)
    return alpha * (temperature**2) * soft_loss + (1.0 - alpha) * hard_loss


def main() -> None:
    """Run the distillation training scaffold."""
    args = parse_args()
    config = load_yaml(args.config)
    seed_everything(config["project"]["seed"])

    data_root = Path(config["data"]["root"])
    image_size = config["data"]["image_size"]
    batch_size = config["train"]["batch_size"]

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
    teacher = build_teacher_model(
        model_name=config["teacher"]["name"],
        num_classes=num_classes,
        pretrained=config["teacher"].get("pretrained", False),
    )
    student = build_student_model(
        model_name=config["student"]["name"],
        num_classes=num_classes,
        pretrained=config["student"].get("pretrained", True),
    )

    device = torch.device(
        "cuda" if config["project"]["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    )
    teacher = teacher.to(device)
    student = student.to(device)

    # TODO: Load a trained teacher checkpoint and freeze teacher parameters.
    optimizer = torch.optim.Adam(
        params=student.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )

    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        optimizer=optimizer,
        device=device,
        temperature=config["distillation"]["temperature"],
        alpha=config["distillation"]["alpha"],
        criterion=distillation_loss,
    )
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["train"]["epochs"],
    )


if __name__ == "__main__":
    main()
