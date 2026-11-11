"""
Step 6 helper - Render annotation sheets for manual review.

usage: python src/annot_sheet.py <list.csv> <out_prefix> [per_sheet=4] [csv_boxes=auto_boxes_v2.csv]
<list.csv> has columns split,cls,file. Each image is resized to 480x480 with a grid every 10%
(labels 0..100 on the edges) and its current auto box drawn in green, so a reviewer can read
corrected coordinates in percent of width/height.
"""
import csv, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR  # noqa: E402  (Kaggle images, see paths.py)
lst, prefix = sys.argv[1], sys.argv[2]
per = int(sys.argv[3]) if len(sys.argv) > 3 else 4
boxes_csv = sys.argv[4] if len(sys.argv) > 4 else "auto_boxes_v2.csv"
S, M = 480, 26          # tile size, margin for axis labels
boxes = {(r["split"], r["cls"], r["file"]): r for r in csv.DictReader(open(ROOT / "labels" / boxes_csv))}
items = list(csv.DictReader(open(lst)))
outdir = ROOT / "outputs" / "annot"; outdir.mkdir(parents=True, exist_ok=True)


def tile(idx, it):
    img = cv2.imread(str(DATA_DIR / it["split"] / it["cls"] / it["file"]))
    H, W = img.shape[:2]
    img = cv2.resize(img, (S, S))
    for k in range(1, 10):
        p = int(S * k / 10)
        col = (0, 200, 255) if k == 5 else (0, 120, 200)
        cv2.line(img, (p, 0), (p, S), col, 1); cv2.line(img, (0, p), (S, p), col, 1)
    r = boxes.get((it["split"], it["cls"], it["file"]))
    if r and r["x1"]:
        x1, y1, x2, y2 = (int(r["x1"]) * S // W, int(r["y1"]) * S // H, int(r["x2"]) * S // W, int(r["y2"]) * S // H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    canvas = np.zeros((S + 2 * M, S + 2 * M, 3), np.uint8)
    canvas[M:M + S, M:M + S] = img
    for k in range(0, 11):
        p = M + int(S * k / 10)
        cv2.putText(canvas, str(k * 10), (p - 8, M - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(canvas, str(k * 10), (2, p + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(canvas, f"#{idx} {it['cls'][:4]} {it['file']}", (M, S + 2 * M - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas


for k in range(0, len(items), per):
    ts = [tile(k + j, it) for j, it in enumerate(items[k:k + per])]
    while len(ts) % 2: ts.append(np.zeros_like(ts[0]))
    sheet = np.vstack([np.hstack(ts[i:i + 2]) for i in range(0, len(ts), 2)])
    p = outdir / f"{prefix}_{k // per:03d}.png"; cv2.imwrite(str(p), sheet)
print(len(items), "items ->", (len(items) + per - 1) // per, "sheets in", outdir)
