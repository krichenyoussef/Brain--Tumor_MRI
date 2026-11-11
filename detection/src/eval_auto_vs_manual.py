"""
Step 7b - Score the automatic boxes against the manually reviewed ("gold") boxes.

Gold box per reviewed image:  verdict ok  -> the v2 auto box (accepted as correct)
                              verdict fix -> the reviewer's box
                              verdict drop-> excluded (no visible tumor)
For v1 (CAM only) and v2 (CAM + brightness) auto boxes we report, per class:
  mean IoU with gold, and the share of images with IoU >= 0.5 (the usual "correct detection" bar).
Also writes outputs/annot/gold_check_*.png sheets showing gold boxes (cyan) for spot-checking.
"""
import csv
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR  # noqa: E402  (Kaggle images, see paths.py)
LAB = ROOT / "labels"


def load(name):
    return {(r["split"], r["cls"], r["file"]): r for r in csv.DictReader(open(LAB / name))}


def box(r):
    return tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")) if r and r["x1"] else None


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


v1, v2 = load("auto_boxes.csv"), load("auto_boxes_v2.csv")
review = list(csv.DictReader(open(LAB / "manual_review.csv")))
gold = {}
for r in review:
    k = (r["split"], r["cls"], r["file"])
    if r["verdict"] == "drop":
        continue
    if r["verdict"] == "ok":
        gold[k] = box(v2[k])
    else:
        W, H = int(v2[k]["W"]), int(v2[k]["H"])
        gold[k] = (float(r["x1p"]) / 100 * W, float(r["y1p"]) / 100 * H, float(r["x2p"]) / 100 * W, float(r["y2p"]) / 100 * H)

lines = ["class        n   v1 meanIoU  v1 IoU>=.5   v2 meanIoU  v2 IoU>=.5"]
for cls in ("glioma", "meningioma", "pituitary", "ALL"):
    ks = [k for k in gold if cls == "ALL" or k[1] == cls]
    a = np.array([iou(box(v1[k]), gold[k]) for k in ks]); b = np.array([iou(box(v2[k]), gold[k]) for k in ks])
    lines.append(f"{cls:<11}{len(ks):>4}   {a.mean():>9.3f}  {(a >= .5).mean():>9.1%}   {b.mean():>9.3f}  {(b >= .5).mean():>9.1%}")
verd = {}
for r in review:
    verd.setdefault(r["cls"], {}).setdefault(r["verdict"], 0); verd[r["cls"]][r["verdict"]] += 1
lines.append("\nverdicts: " + str(verd))
txt = "\n".join(lines); print(txt)
(ROOT / "outputs" / "auto_vs_manual.txt").write_text(txt + "\n")

# spot-check sheets: gold box cyan, v2 auto box green (thin)
out = ROOT / "outputs" / "annot"; S = 300; tiles = []
for r in review:
    k = (r["split"], r["cls"], r["file"])
    img = cv2.imread(str(DATA_DIR / k[0] / k[1] / k[2])); H, W = img.shape[:2]
    sx, sy = S / W, S / H; img = cv2.resize(img, (S, S))
    a = box(v2[k])
    if a: cv2.rectangle(img, (int(a[0] * sx), int(a[1] * sy)), (int(a[2] * sx), int(a[3] * sy)), (0, 255, 0), 1)
    g = gold.get(k)
    if g: cv2.rectangle(img, (int(g[0] * sx), int(g[1] * sy)), (int(g[2] * sx), int(g[3] * sy)), (255, 255, 0), 2)
    cv2.putText(img, f"{r['cls'][:4]} {r['verdict']} {k[2]}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    tiles.append(img)
for i in range(0, len(tiles), 20):
    ch = tiles[i:i + 20]
    while len(ch) % 5: ch.append(np.zeros((S, S, 3), np.uint8))
    cv2.imwrite(str(out / f"gold_check_{i // 20:02d}.png"), np.vstack([np.hstack(ch[j:j + 5]) for j in range(0, len(ch), 5)]))
