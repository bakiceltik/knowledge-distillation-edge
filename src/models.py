"""Model factory for baseline and distillation experiments."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """Instantiate a torchvision model and replace its classification head.

    Supported model_name values:
        resnet18, resnet50, mobilenet_v3_small, efficientnet_b0

    The final linear layer is replaced to match *num_classes*.
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
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=weights_arg)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        supported = ["resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0"]
        raise ValueError(
            f"Unknown model_name '{model_name}'. Supported: {supported}"
        )

    return model
