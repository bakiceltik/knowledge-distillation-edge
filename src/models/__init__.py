"""Unified model factory exposed at the src.models package level."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """Instantiate a torchvision model and replace its classification head.

    Supported names: resnet18, resnet50, mobilenet_v3_small,
    mobilenet_v3_large, efficientnet_b0, efficientnet_b2, vit_b_16
    """
    weights_arg = "DEFAULT" if pretrained else None

    if model_name == "resnet18":
        model = tvm.resnet18(weights=weights_arg)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        model = tvm.resnet50(weights=weights_arg)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet_v3_small":
        model = tvm.mobilenet_v3_small(weights=weights_arg)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        model = tvm.mobilenet_v3_large(weights=weights_arg)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=weights_arg)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name == "efficientnet_b2":
        model = tvm.efficientnet_b2(weights=weights_arg)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name == "vit_b_16":
        model = tvm.vit_b_16(weights=weights_arg)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    else:
        supported = [
            "resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large",
            "efficientnet_b0", "efficientnet_b2", "vit_b_16",
        ]
        raise ValueError(f"Unknown model_name '{model_name}'. Supported: {supported}")

    return model
