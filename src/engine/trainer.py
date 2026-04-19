"""Training engine skeletons for supervised and distillation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from tqdm import tqdm


class Trainer:
    """Minimal supervised trainer for image classification experiments."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        output_dir: Path,
        log_interval: int = 20,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = output_dir
        self.log_interval = log_interval

    def train_one_epoch(self, dataloader) -> float:
        """Run one supervised training epoch."""
        self.model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="train", leave=False)):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.item())

            if batch_idx % self.log_interval == 0:
                tqdm.write(f"batch={batch_idx} loss={loss.item():.4f}")

        return running_loss / max(len(dataloader), 1)

    def fit(self, train_loader, val_loader, epochs: int) -> None:
        """Run the training scaffold across epochs.

        TODO:
        - Track validation metrics and early stopping.
        - Save best and last checkpoints.
        - Integrate scheduler support and experiment logging.
        """
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            print(f"epoch={epoch} train_loss={train_loss:.4f}")
            if val_loader is not None:
                print("validation hook is reserved for a future commit")


class DistillationTrainer:
    """Minimal trainer for student-teacher distillation experiments."""

    def __init__(
        self,
        teacher: torch.nn.Module,
        student: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        temperature: float,
        alpha: float,
        criterion: Callable,
    ) -> None:
        self.teacher = teacher.eval()
        self.student = student
        self.optimizer = optimizer
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = criterion

    def train_one_epoch(self, dataloader) -> float:
        """Run one distillation training epoch."""
        self.student.train()
        running_loss = 0.0

        for images, targets in tqdm(dataloader, desc="distill", leave=False):
            images = images.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                teacher_logits = self.teacher(images)

            student_logits = self.student(images)
            loss = self.criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                targets=targets,
                temperature=self.temperature,
                alpha=self.alpha,
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += float(loss.item())

        return running_loss / max(len(dataloader), 1)

    def fit(self, train_loader, val_loader, epochs: int) -> None:
        """Run the distillation scaffold across epochs."""
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            print(f"epoch={epoch} distillation_loss={train_loss:.4f}")
            if val_loader is not None:
                print("validation hook is reserved for a future commit")
