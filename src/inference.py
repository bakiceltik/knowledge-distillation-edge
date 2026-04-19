"""Single-image inference scaffold for trained models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from src.data.transforms import build_eval_transforms
from src.models.student import build_student_model


def parse_args() -> argparse.Namespace:
    """Parse inference CLI arguments."""
    parser = argparse.ArgumentParser(description="Single-image inference scaffold.")
    parser.add_argument("--image", type=Path, required=True, help="Path to an input image.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="mobilenet_v3_small",
        help="Student model architecture to instantiate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path for trained weights.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="Number of output classes for the trained classifier.",
    )
    return parser.parse_args()


def predict_image(
    image_path: Path,
    model: torch.nn.Module,
    class_names: Sequence[str] | None = None,
    image_size: int = 224,
) -> tuple[int, str | None]:
    """Run inference on a single image and return the predicted class index."""
    transform = build_eval_transforms(image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        prediction = int(torch.argmax(logits, dim=1).item())

    label = class_names[prediction] if class_names is not None else None
    return prediction, label


def main() -> None:
    """Run a basic inference workflow.

    TODO:
    - Load checkpoints safely from disk.
    - Support device placement and latency benchmarking.
    - Add label mapping based on dataset metadata.
    """
    args = parse_args()
    model = build_student_model(model_name=args.model_name, num_classes=args.num_classes)

    if args.checkpoint is not None:
        # TODO: Load actual trained weights once checkpoints are available.
        print(f"Checkpoint loading is not implemented yet: {args.checkpoint}")

    prediction, label = predict_image(args.image, model=model, image_size=args.image_size)
    print({"prediction": prediction, "label": label})


if __name__ == "__main__":
    main()
