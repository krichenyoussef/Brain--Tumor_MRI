"""
Step 5b - Refine Grad-CAM boxes using image intensity.

Why: Grad-CAM on ResNet50 layer4 is a 7x7 map, so CAM boxes are coarse (often 2-3x too big
or shifted). Most tumors in this dataset are contrast-enhanced T1 slices where the tumor is
a bright, compact blob. So:
  1. recompute the Grad-CAM map (true class) like gradcam_boxes.py,
  2. build a brain mask and erode it so the bright skull / scalp ring is excluded,
  3. "search region" = CAM >= SEARCH_T, dilated,
  4. inside it, keep pixels brighter than a high percentile of brain intensity,
  5. split into connected blobs, score each blob by the CAM mass it covers,
  6. if the best blob has a sensible size and real CAM support -> box = its bounding rect
     (+ small padding)  [method = "bright"],
     else keep the plain CAM box                                [method = "cam"].

Output: labels/auto_boxes_v2.csv (same columns as auto_boxes.csv + 'method').
"""
import csv, os
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEARCH_T = 0.30        # CAM level that defines the search region
BRIGHT_PCT = 93        # percentile of brain intensity counted as "bright"
MIN_AREA, MAX_AREA = 0.002, 0.15   # blob area as fraction of the image
MIN_CAM_MEAN = 0.30    # blob must sit on real CAM evidence
PAD = 0.08             # padding added to blob box, as fraction of its size
TUMOR = ("glioma", "meningioma", "pituitary")
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
    _, m = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.ones_like(gray, dtype=np.uint8)
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    m = (lab == big).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    # fill holes so dark ventricles / cysts stay inside the mask
    ff = m.copy(); h, w = ff.shape; tmp = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, tmp, (0, 0), 1)
    return m | (1 - ff)


def cam_box(cam, mask):
    cam = cam * mask
    if cam.max() <= 0:
        return None
    cam = cam / cam.max(); last = None
    for t in (0.45, 0.6, 0.75, 0.85):
        n, lab, stats, _ = cv2.connectedComponentsWithStats((cam >= t).astype(np.uint8))
        if n <= 1:
            continue
        best = max(range(1, n), key=lambda k: cam[lab == k].sum())
        x, y, w, h, area = stats[best]; last = (x, y, x + w, y + h)
        if area / cam.size <= 0.55:
            return last
    return last


def refine(gray, cam, mask):
    H, W = gray.shape
    k = max(3, int(0.06 * min(H, W)))
    inner = cv2.erode(mask, np.ones((k, k), np.uint8))           # drop skull ring
    if inner.sum() < 0.05 * H * W:
        return None
    camn = cam * mask; camn = camn / (camn.max() + 1e-8)
    search = cv2.dilate((camn >= SEARCH_T).astype(np.uint8), np.ones((k, k), np.uint8)) & inner
    vals = gray[inner.astype(bool)]
    thr = np.percentile(vals, BRIGHT_PCT)
    bright = ((gray >= thr) & search.astype(bool)).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright)
    best, best_score = None, 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA] / (H * W)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        sel = lab == i
        cm = camn[sel].mean()
        if cm < MIN_CAM_MEAN:
            continue
        score = camn[sel].sum()
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None
    x, y, w, h, _ = stats[best]
    px, py = int(PAD * w) + 2, int(PAD * h) + 2
    return max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)


def main():
    rows = []
    for split in ("Training", "Testing"):
        for cls in classes:
            files = sorted((DATA / split / cls).glob("*.jpg"))
            if cls not in TUMOR:
                rows += [[split, cls, f.name, "", "", "", "", "", "", "", "", ""] for f in files]; continue
            ci = classes.index(cls)
            for i in range(0, len(files), BATCH):
                chunk = files[i:i + BATCH]
                pil = [Image.open(f).convert("RGB") for f in chunk]
                out = model(torch.stack([tf(p) for p in pil]).to(device))
                model.zero_grad(set_to_none=True); out[:, ci].sum().backward()
                a, g = acts["a"], grads["g"]
                cams = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1)).detach().float().cpu().numpy()
                probs = F.softmax(out.detach(), 1)
                for j, f in enumerate(chunk):
                    W, H = pil[j].size
                    gray = np.array(pil[j].convert("L"))
                    cam = np.clip(cv2.resize(cams[j], (W, H), interpolation=cv2.INTER_CUBIC), 0, None)
                    mask = brain_mask(gray)
                    box, method = refine(gray, cam, mask), "bright"
                    if box is None:
                        box, method = cam_box(cam, mask), "cam"
                    pred = classes[int(probs[j].argmax())]; conf = f"{float(probs[j, ci]):.3f}"
                    if box is None:
                        rows.append([split, cls, f.name, pred, conf, "", "", "", "", W, H, "none"]); continue
                    rows.append([split, cls, f.name, pred, conf, *map(int, box), W, H, method])
                print(f"{split} {cls} {min(i + BATCH, len(files))}/{len(files)}", flush=True)
    with open(ROOT / "labels" / "auto_boxes_v2.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "cls", "file", "pred", "conf_true_cls", "x1", "y1", "x2", "y2", "W", "H", "method"])
        w.writerows(rows)
    print("done", len(rows))


if __name__ == "__main__":
    main()
