"""Find duplicate MRI images and build a leak-free train/val/test split.

Public brain MRI datasets contain the same scan several times (re-encoded, resized
or copied between the Training and Testing folders). If a duplicate sits in both
train and test, test accuracy is inflated. This script:

1. hashes every image (MD5 for exact copies, perceptual hash for re-encoded copies);
2. groups images that share a hash;
3. drops groups whose copies carry different labels (the label is unreliable);
4. keeps one image per group and splits groups into train/val/test, stratified by class.

Usage:
    python -m src.dedup --raw data/raw --out data/splits.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import CLASSES, MANIFEST, RAW_DIR, RESULTS, SEED

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(raw_dir: Path) -> list[dict]:
    """Return every image under raw_dir/<split>/<class>/ with its original split and label."""
    items = []
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXT:
            continue
        label = path.parent.name.lower()
        if label not in CLASSES:
            continue
        items.append({"path": path, "label": label, "orig_split": path.parent.parent.name})
    return items


def image_hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    with Image.open(path) as img:
        phash = str(imagehash.phash(img.convert("L"), hash_size=8))
    return md5, phash


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def group_duplicates(items: list[dict]) -> list[list[int]]:
    """Group item indices that share an MD5 or a perceptual hash."""
    uf = _UnionFind(len(items))
    first_seen: dict[tuple[str, str], int] = {}
    for i, it in enumerate(items):
        for key in (("md5", it["md5"]), ("phash", it["phash"])):
            if key in first_seen:
                uf.union(i, first_seen[key])
            else:
                first_seen[key] = i
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[uf.find(i)].append(i)
    return list(groups.values())


def build_split(items, groups, val_size=0.15, test_size=0.15, seed=SEED):
    """Keep one image per clean group and split groups, stratified by label."""
    kept, conflicts, dup_groups, cross_split = [], 0, 0, 0
    for g in groups:
        labels = {items[i]["label"] for i in g}
        if len(g) > 1:
            dup_groups += 1
            if len({items[i]["orig_split"] for i in g}) > 1:
                cross_split += 1
        if len(labels) > 1:
            conflicts += 1
            continue
        kept.append(items[g[0]])

    labels = [it["label"] for it in kept]
    train_val, test = train_test_split(kept, test_size=test_size, stratify=labels, random_state=seed)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=[it["label"] for it in train_val], random_state=seed
    )
    rows = [(it, "train") for it in train] + [(it, "val") for it in val] + [(it, "test") for it in test]

    report = {
        "images_found": len(items),
        "duplicate_groups": dup_groups,
        "duplicates_removed": len(items) - len(groups),
        "groups_spanning_original_train_and_test": cross_split,
        "groups_with_conflicting_labels_dropped": conflicts,
        "images_kept": len(kept),
        "split_sizes": dict(Counter(s for _, s in rows)),
        "class_counts": dict(Counter(labels)),
    }
    return rows, report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=RAW_DIR)
    p.add_argument("--out", type=Path, default=MANIFEST)
    p.add_argument("--report", type=Path, default=RESULTS / "dedup_report.json")
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    args = p.parse_args(argv)

    items = list_images(args.raw)
    if not items:
        raise SystemExit(f"No images found under {args.raw}/<split>/<class>/. See the README for the dataset layout.")
    for it in items:
        it["md5"], it["phash"] = image_hashes(it["path"])

    groups = group_duplicates(items)
    rows, report = build_split(items, groups, args.val_size, args.test_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        for it, split in rows:
            w.writerow([it["path"].resolve().as_posix(), it["label"], split])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
