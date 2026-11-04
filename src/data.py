"""Dataset and image transforms."""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.config import CLASSES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


def train_transform(size: int = IMG_SIZE):
    return T.Compose([
        T.Resize(int(size * 1.15)),
        T.RandomResizedCrop(size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size: int = IMG_SIZE):
    return T.Compose([
        T.Resize(int(size * 1.15)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def read_manifest(path: Path, split: str) -> list[tuple[str, int]]:
    with Path(path).open() as f:
        return [(r["path"], CLASSES.index(r["label"])) for r in csv.DictReader(f) if r["split"] == split]


class MRIDataset(Dataset):
    def __init__(self, manifest: Path, split: str, transform):
        self.samples = read_manifest(manifest, split)
        if not self.samples:
            raise ValueError(f"No '{split}' images in {manifest}")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as img:
            x = self.transform(img.convert("RGB"))
        return x, label
