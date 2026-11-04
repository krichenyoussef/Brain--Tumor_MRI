"""Grad-CAM heatmaps: show which part of the scan drove the prediction.

Usage:
    python -m src.gradcam --checkpoint checkpoints/resnet50.pt          # grid of test examples
    python -m src.gradcam --checkpoint checkpoints/resnet50.pt --image path/to/scan.jpg
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import MANIFEST, RESULTS
from src.data import eval_transform
from src.models import build_model, gradcam_target, is_vit, vit_reshape_transform


def load_checkpoint(path: Path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["timm_name"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


class Explainer:
    """Predict a class and its Grad-CAM heatmap for one image."""

    def __init__(self, model, timm_name: str, classes: list[str], img_size: int):
        self.model, self.classes, self.img_size = model, classes, img_size
        self.transform = eval_transform(img_size)
        self.cam = GradCAM(
            model=model,
            target_layers=[gradcam_target(model, timm_name)],
            reshape_transform=vit_reshape_transform if is_vit(timm_name) else None,
        )

    def __call__(self, image: Image.Image):
        rgb = image.convert("RGB")
        x = self.transform(rgb).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].numpy()
        heat = self.cam(input_tensor=x)[0]  # explains the top predicted class
        # Show the heatmap on the same centered crop the model saw.
        view = rgb.resize((int(self.img_size * 1.15),) * 2)
        off = (view.width - self.img_size) // 2
        view = np.asarray(view.crop((off, off, off + self.img_size, off + self.img_size)), dtype=np.float32) / 255
        overlay = show_cam_on_image(view, heat, use_rgb=True)
        return {c: float(p) for c, p in zip(self.classes, probs)}, overlay


def example_grid(explainer: Explainer, manifest: Path, out: Path, n: int = 8, seed: int = 0):
    with manifest.open() as f:
        test = [r for r in csv.DictReader(f) if r["split"] == "test"]
    random.Random(seed).shuffle(test)
    rows = test[:n]
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(rows) / cols)), cols, figsize=(3.2 * cols, 3.6 * np.ceil(len(rows) / cols)))
    for ax, r in zip(np.ravel(axes), rows):
        with Image.open(r["path"]) as img:
            probs, overlay = explainer(img)
        pred = max(probs, key=probs.get)
        ax.imshow(overlay)
        ax.set_title(f"true: {r['label']}\npred: {pred} ({probs[pred]:.0%})", fontsize=9,
                     color="black" if pred == r["label"] else "crimson")
        ax.axis("off")
    for ax in np.ravel(axes)[len(rows):]:
        ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, help="explain a single image instead of a grid of test images")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)

    model, ckpt = load_checkpoint(args.checkpoint)
    explainer = Explainer(model, ckpt["timm_name"], ckpt["classes"], ckpt["img_size"])
    if args.image:
        with Image.open(args.image) as img:
            probs, overlay = explainer(img)
        out = args.out or args.image.with_name(args.image.stem + "_gradcam.png")
        Image.fromarray(overlay).save(out)
        print({k: round(v, 3) for k, v in probs.items()}, "->", out)
    else:
        out = args.out or RESULTS / ckpt["model"] / "gradcam_examples.png"
        print("saved", example_grid(explainer, args.manifest, out, args.n))


if __name__ == "__main__":
    main()
