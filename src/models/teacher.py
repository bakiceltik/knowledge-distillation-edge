"""Teacher model factory."""

from __future__ import annotations

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    efficientnet_b2,
    resnet50,
)


def build_teacher_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a teacher model for classification experiments."""
    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b2":
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = efficientnet_b2(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported teacher model: {model_name}")
