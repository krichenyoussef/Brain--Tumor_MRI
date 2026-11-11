"""
Step 4 - Generate automatic bounding boxes with Grad-CAM.

For every tumor image (glioma / meningioma / pituitary) in Training and Testing:
  1. run the trained ResNet50, hook the last conv block (layer4),
  2. Grad-CAM for the *true* class -> heat-map of where the evidence is,
  3. upsample the map to the original image size, keep only the brain region,
  4. threshold at a fraction of the max, take the blob with the most CAM mass,
  5. its bounding rectangle is the box.
No-tumor images get an empty label file (YOLO "background" images).

Outputs:
  labels/auto/<split>/<class>/<name>.txt   YOLO format: cls cx cy w h (normalised)
  labels/auto_boxes.csv                    one row per image with pixel box + diagnostics
"""
import csv, os
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR  # noqa: E402  (Kaggle images, see paths.py)
DATA, OUT, LAB = DATA_DIR, ROOT / "outputs", ROOT / "labels" / "auto"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAM_THRESH = float(os.environ.get("CAM_THRESH", 0.45))   # fraction of CAM max
MAX_AREA_FRAC = 0.55                                       # if blob bigger than this, re-threshold higher
YOLO_ID = {"glioma": 0, "meningioma": 1, "pituitary": 2}  # notumor -> no box
BATCH = 32

ck = torch.load(OUT / "model.pt", map_location=device)
classes = ck["classes"]; IMG = ck["img"]
model = models.resnet50(weights=None); model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ck["state_dict"]); model.to(device).eval()

acts, grads = {}, {}
model.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
model.layer4.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))

tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def brain_mask(gray):
    """Largest bright connected region = the head; removes black background & stray text."""
    _, m = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.ones_like(gray, dtype=bool)
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return lab == big


def cam_to_box(cam, mask, thresh):
    """Threshold CAM, keep the blob with most CAM mass inside the brain, return (x1,y1,x2,y2) or None."""
    cam = cam * mask
    if cam.max() <= 0:
        return None
    cam = cam / cam.max()
    last = None
    for t in (thresh, 0.6, 0.75, 0.85):
        b = (cam >= t).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(b)
        if n <= 1:
            continue
        best, best_mass = None, -1
        for k in range(1, n):
            mass = cam[lab == k].sum()
            if mass > best_mass:
                best, best_mass = k, mass
        x, y, w, h, area = stats[best]
        last = (int(x), int(y), int(x + w), int(y + h))
        if area / cam.size <= MAX_AREA_FRAC:
            return last
    return last


def main():
    rows = []
    for split in ("Training", "Testing"):
        for cls in classes:
            files = sorted((DATA / split / cls).glob("*.jpg"))
            outdir = LAB / split / cls; outdir.mkdir(parents=True, exist_ok=True)
            if cls not in YOLO_ID:
                for f in files:
                    (outdir / (f.stem + ".txt")).write_text("")
                    rows.append([split, cls, f.name, "", "", "", "", "", "", "", ""])
                print(split, cls, len(files), "background images"); continue
            ci = classes.index(cls); yid = YOLO_ID[cls]
            for i in range(0, len(files), BATCH):
                chunk = files[i:i + BATCH]
                pil = [Image.open(f).convert("RGB") for f in chunk]
                x = torch.stack([tf(p) for p in pil]).to(device)
                out = model(x)
                model.zero_grad(set_to_none=True)
                out[:, ci].sum().backward()           # per-sample grads (eval-mode BN => samples independent)
                a, g = acts["a"], grads["g"]           # [B,2048,7,7]
                w = g.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((w * a).sum(1))           # [B,7,7]
                probs = F.softmax(out.detach(), 1)
                for j, f in enumerate(chunk):
                    W, H = pil[j].size
                    c = cam[j].detach().float().cpu().numpy()
                    c = cv2.resize(c, (W, H), interpolation=cv2.INTER_CUBIC)
                    c = np.clip(c, 0, None)
                    gray = np.array(pil[j].convert("L"))
                    mask = brain_mask(gray)
                    box = cam_to_box(c, mask, CAM_THRESH)
                    pred = classes[int(probs[j].argmax())]; conf = float(probs[j, ci])
                    if box is None:
                        (outdir / (f.stem + ".txt")).write_text("")
                        rows.append([split, cls, f.name, pred, f"{conf:.3f}", "", "", "", "", "", ""]); continue
                    x1, y1, x2, y2 = box
                    cx, cy, bw, bh = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H
                    (outdir / (f.stem + ".txt")).write_text(f"{yid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    rows.append([split, cls, f.name, pred, f"{conf:.3f}", x1, y1, x2, y2, W, H])
                print(f"{split} {cls} {min(i + BATCH, len(files))}/{len(files)}", flush=True)
    with open(ROOT / "labels" / "auto_boxes.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["split", "cls", "file", "pred", "conf_true_cls", "x1", "y1", "x2", "y2", "W", "H"])
        wr.writerows(rows)
    print("done", len(rows))


if __name__ == "__main__":
    main()
