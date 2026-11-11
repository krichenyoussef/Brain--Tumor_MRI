"""
Load outputs/model.pt (ResNet50 classifier) and predict on one or more images.

usage: python src/predict.py <image_or_folder> [more paths...]
"""
import sys
from pathlib import Path
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "outputs" / "model.pt"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_model(device=None):
    """Return (model, classes, transform). Model is in eval mode on `device`."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    classes, img = ck["classes"], ck["img"]
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((img, img)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return model, classes, tf


@torch.no_grad()
def predict(model, classes, tf, path, device=None):
    """Return (label, confidence, {class: prob}) for one image path."""
    device = device or next(model.parameters()).device
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu()
    i = int(probs.argmax())
    return classes[i], float(probs[i]), {c: float(p) for c, p in zip(classes, probs)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model, classes, tf = load_model()
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = []
    for a in sys.argv[1:]:
        p = Path(a)
        paths += sorted(q for q in p.rglob("*") if q.suffix.lower() in exts) if p.is_dir() else [p]
    for p in paths:
        label, conf, probs = predict(model, classes, tf, p)
        detail = "  ".join(f"{c}={v:.3f}" for c, v in probs.items())
        print(f"{p}: {label} ({conf:.1%})   [{detail}]")
