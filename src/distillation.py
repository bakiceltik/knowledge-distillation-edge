"""Knowledge distillation loss — placeholder for next-phase implementation.

Full teacher-student training will be implemented after the progress
checkpoint. The distillation_loss function below is mathematically correct
and can be dropped into a training loop once the teacher checkpoint is ready.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Combine hard-label cross-entropy with soft-target KL divergence.

    Loss = alpha * KL(soft_student || soft_teacher) * T^2
           + (1 - alpha) * CE(student_logits, labels)

    Args:
        student_logits: Raw logits from the student model  [B, C].
        teacher_logits: Raw logits from the teacher model  [B, C].
        labels:         Ground-truth class indices          [B].
        temperature:    Softening temperature T > 1.
        alpha:          Weight for the soft-target KL term.

    Returns:
        Scalar loss tensor.
    """
    # Hard-label cross-entropy
    ce_loss = F.cross_entropy(student_logits, labels)

    # Soft targets
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)

    # KL divergence scaled by T^2 (restores gradient magnitude after softening)
    kl_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)

    return alpha * kl_loss + (1.0 - alpha) * ce_loss
