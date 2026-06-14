"""t-SNE visualization of the distilled student's penultimate embeddings.

Loads a trained MobileNetV3 student checkpoint, extracts the feature vector that
feeds the final classifier on the single-split test set, projects it to 2D with
t-SNE, and saves a scatter plot colored by true class. This is the
interpretability visualization referenced in the report's error analysis: it
shows how separable the learned representation is per class, which mirrors the
per-class precision/recall and the confusion-matrix error patterns.

Usage:
    python -m src.visualize_embeddings --config configs/quantization_cassava.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from src.data.data_utils import get_dataloaders
from src.models.student import build_student_model
from src.utils import get_device, load_yaml_config, set_seed


def extract_embeddings(model, loader, device):
    """Return (features, true_labels, predictions) over a DataLoader.

    A forward hook on the final classifier layer captures its input, i.e. the
    penultimate feature vector (1024-d for MobileNetV3-Small).
    """
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inp, _out):
        captured["x"] = inp[0].detach()

    handle = model.classifier[-1].register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            outputs = model(images)
            feats.append(captured["x"].cpu().numpy())
            preds.append(outputs.argmax(dim=1).cpu().numpy())
            labels.append(y.numpy())
    handle.remove()
    return np.concatenate(feats), np.concatenate(labels), np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="t-SNE of student embeddings.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device()

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=cfg["data_dir"],
        image_size=cfg.get("image_size", 224),
        batch_size=cfg.get("batch_size", 64),
        val_ratio=cfg.get("val_ratio", 0.15),
        test_ratio=cfg.get("test_ratio", 0.15),
        seed=seed,
        num_workers=0,  # one-off pass; avoids Windows worker spawn overhead
        subset_prefix=cfg.get("subset_prefix") or None,
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    model_name = cfg.get("model_name") or cfg.get("student_model_name", "mobilenet_v3_small")
    model = build_student_model(model_name, num_classes=len(class_names), pretrained=False)
    ckpt = cfg.get("checkpoint") or str(Path(cfg["output_dir"]) / "best_model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)
    print(f"Loaded {model_name} from {ckpt} on {device}")

    feats, labels, _ = extract_embeddings(model, loader, device)
    print(f"Extracted {feats.shape[0]} embeddings of dim {feats.shape[1]} ({args.split} split)")

    perplexity = min(30, max(5, (feats.shape[0] - 1) // 3))
    tsne = TSNE(
        n_components=2,
        init="pca",
        perplexity=perplexity,
        learning_rate="auto",
        random_state=seed,
    )
    emb2d = tsne.fit_transform(feats)

    short_names = [c.split("___")[-1].replace("_", " ") for c in class_names]
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for k, name in enumerate(short_names):
        mask = labels == k
        ax.scatter(
            emb2d[mask, 0], emb2d[mask, 1],
            s=8, alpha=0.6, color=cmap(k),
            label=f"{name} (n={int(mask.sum())})",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("t-SNE of distilled student embeddings (Cassava test set)")
    ax.legend(loc="best", fontsize=8, markerscale=2, framealpha=0.9)
    fig.tight_layout()

    out = args.output or str(Path(cfg["output_dir"]) / "tsne_embeddings.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved t-SNE plot to {out}")


if __name__ == "__main__":
    main()
