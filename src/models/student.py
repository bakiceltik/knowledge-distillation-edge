"""Student model factory."""

from __future__ import annotations

from torch import nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)


def build_student_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a MobileNetV3 student model."""
    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
    elif model_name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"Unsupported student model: {model_name}")

    last_channel = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(last_channel, num_classes)
    return model
