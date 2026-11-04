"""End-to-end smoke test: dedup -> train -> evaluate -> compare -> Grad-CAM, on tiny fake data."""
import json

import pytest
import torch
from PIL import Image

from src import compare, dedup, gradcam, train
from src.models import build_model, gradcam_target, is_vit, vit_reshape_transform


@pytest.mark.parametrize("name", ["resnet18", "efficientnet_b0", "vit_tiny_patch16_224"])
def test_backbones_forward_and_gradcam_target(name):
    size = 224 if is_vit(name) else 64
    model = build_model(name, num_classes=4, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, size, size)).shape == (2, 4)
    assert gradcam_target(model, name) is not None


def test_vit_reshape_transform():
    tokens = torch.randn(2, 1 + 14 * 14, 192)
    assert vit_reshape_transform(tokens).shape == (2, 192, 14, 14)


def test_full_pipeline(raw_dataset, tmp_path):
    manifest = tmp_path / "splits.csv"
    dedup.main(["--raw", str(raw_dataset), "--out", str(manifest), "--report", str(tmp_path / "r.json")])

    results, ckpts = tmp_path / "results", tmp_path / "ckpt"
    [metrics] = train.main([
        "--model", "resnet18", "--epochs", "2", "--batch-size", "8", "--img-size", "64", "--workers", "0",
        "--no-pretrained", "--manifest", str(manifest), "--results", str(results), "--checkpoints", str(ckpts),
    ])
    assert 0 <= metrics["accuracy"] <= 1
    saved = json.loads((results / "resnet18" / "metrics.json").read_text())
    assert saved["n_test_images"] == metrics["n_test_images"] > 0
    assert (results / "resnet18" / "confusion_matrix.png").exists()

    table = compare.main(["--results", str(results)])
    assert "resnet18" in table and (results / "RESULTS.md").exists()

    grid = tmp_path / "grid.png"
    gradcam.main(["--checkpoint", str(ckpts / "resnet18.pt"), "--manifest", str(manifest), "--n", "4", "--out", str(grid)])
    assert grid.exists()

    model, ckpt = gradcam.load_checkpoint(ckpts / "resnet18.pt")
    explainer = gradcam.Explainer(model, ckpt["timm_name"], ckpt["classes"], ckpt["img_size"])
    probs, overlay = explainer(Image.new("L", (80, 80), 40))
    assert abs(sum(probs.values()) - 1) < 1e-4
    assert overlay.shape == (64, 64, 3)
