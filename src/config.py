"""Shared settings for the brain tumor MRI project."""
from pathlib import Path

# Class folder names used by the Kaggle "Brain Tumor MRI Dataset" (Masoud Nickparvar).
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NO_TUMOR = "notumor"

# Backbones compared in this project (timm model names).
BACKBONES = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "vit_small": "vit_small_patch16_224",
}

IMG_SIZE = 224
SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # Kaggle archive extracted here (Training/ and Testing/)
MANIFEST = DATA_DIR / "splits.csv"    # clean train/val/test split produced by src.dedup
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
