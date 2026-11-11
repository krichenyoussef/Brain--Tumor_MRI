"""
Step 5 - Draw auto boxes on images and tile them into review grids.

usage: python src/review_grid.py <split> <class> [per_grid] [max_images] [csv_name]
Writes outputs/review/<split>_<class>_<k>.png, each tile labelled with its index + file name,
so a reviewer can say e.g. "grid 2, tile 7 is wrong".
"""
import sys, csv
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR  # noqa: E402  (Kaggle images, see paths.py)
split, cls = sys.argv[1], sys.argv[2]
per = int(sys.argv[3]) if len(sys.argv) > 3 else 20
maxn = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
csv_name = sys.argv[5] if len(sys.argv) > 5 else "auto_boxes.csv"
cols, tile = 5, 256
outdir = ROOT / "outputs" / "review"
outdir.mkdir(parents=True, exist_ok=True)

rows = [r for r in csv.DictReader(open(ROOT / "labels" / csv_name))
        if r["split"] == split and r["cls"] == cls][:maxn]
tiles = []
for idx, r in enumerate(rows):
    img = cv2.imread(str(DATA_DIR / split / cls / r["file"]))
    if r["x1"]:
        x1, y1, x2, y2 = map(int, (r["x1"], r["y1"], r["x2"], r["y2"]))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    img = cv2.resize(img, (tile, tile))
    cv2.rectangle(img, (0, 0), (tile, 22), (0, 0, 0), -1)
    label = f"{idx} {r['file']} p={r['conf_true_cls']}"
    cv2.putText(img, label, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    tiles.append(img)
for k in range(0, len(tiles), per):
    chunk = tiles[k:k + per]
    while len(chunk) % cols:
        chunk.append(np.zeros((tile, tile, 3), np.uint8))
    grid = np.vstack([np.hstack(chunk[i:i + cols]) for i in range(0, len(chunk), cols)])
    p = outdir / f"{split}_{cls}_{k // per}.png"
    cv2.imwrite(str(p), grid)
    print(p)
