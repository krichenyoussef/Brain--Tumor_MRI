"""
Step 7 - Package images + final labels into YOLO (ultralytics) datasets.

usage: python src/build_yolo_dataset.py [full|gold]

Validation set (both modes) = the human-reviewed Testing images only (verified + corrected boxes)
plus 20 no-tumor Testing images as background. This keeps evaluation on trusted boxes.

Training set:
  full -> every Training tumor image (auto boxes, with manual corrections applied, dropped ones
          removed) + all no-tumor Training images as background images (empty label files).
  gold -> only the human-reviewed Training tumor images + 60 no-tumor background images.

Output: yolo_<mode>/images/{train,val}, yolo_<mode>/labels/{train,val}, yolo_<mode>/data.yaml
Images are copied; the original data/ folder is never modified.
"""
import csv, random, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR  # noqa: E402  (Kaggle images, see paths.py)
MODE = sys.argv[1] if len(sys.argv) > 1 else "full"
assert MODE in ("full", "gold")
DST = ROOT / f"yolo_{MODE}"
if DST.exists():
    shutil.rmtree(DST)
random.seed(0)
rows = list(csv.DictReader(open(ROOT / "labels" / "final_boxes.csv")))
GOLD = ("verified", "corrected")

train, val = [], []
bg_train = [r for r in rows if r["split"] == "Training" and r["status"] == "background"]
bg_test = [r for r in rows if r["split"] == "Testing" and r["status"] == "background"]
val += [r for r in rows if r["split"] == "Testing" and r["status"] in GOLD] + random.sample(bg_test, 20)
if MODE == "full":
    train += [r for r in rows if r["split"] == "Training" and r["status"] in ("auto",) + GOLD] + bg_train
else:
    train += [r for r in rows if r["split"] == "Training" and r["status"] in GOLD] + random.sample(bg_train, 60)

for part, items in (("train", train), ("val", val)):
    (DST / "images" / part).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / part).mkdir(parents=True, exist_ok=True)
    for r in items:
        stem = Path(r["file"]).stem
        shutil.copy2(DATA_DIR / r["split"] / r["cls"] / r["file"], DST / "images" / part / r["file"])
        shutil.copy2(ROOT / "labels" / "final" / r["split"] / r["cls"] / f"{stem}.txt", DST / "labels" / part / f"{stem}.txt")

(DST / "data.yaml").write_text(
    f"path: {DST.as_posix()}\ntrain: images/train\nval: images/val\n"
    "names:\n  0: glioma\n  1: meningioma\n  2: pituitary\n")


def summary(items):
    c = {}
    for r in items:
        c[r["cls"]] = c.get(r["cls"], 0) + 1
    return c


print(MODE, "train:", len(train), summary(train))
print(MODE, "val:  ", len(val), summary(val))
