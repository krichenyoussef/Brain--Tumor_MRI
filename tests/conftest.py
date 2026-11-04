"""A tiny synthetic dataset in the Kaggle layout, with planted duplicates."""
import shutil

import numpy as np
import pytest
from PIL import Image

from src.config import CLASSES


def _scan(rng, cls_idx):
    """A fake 64x64 'scan': noise plus a bright blob whose position depends on the class."""
    img = rng.integers(0, 60, (64, 64), dtype=np.uint8)
    cy, cx = [(16, 16), (16, 48), (48, 16), (48, 48)][cls_idx]
    img[cy - 8:cy + 8, cx - 8:cx + 8] = 220
    img += rng.integers(0, 30, (64, 64), dtype=np.uint8)
    return Image.fromarray(img)


@pytest.fixture
def raw_dataset(tmp_path):
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    for split, n in (("Training", 10), ("Testing", 4)):
        for ci, c in enumerate(CLASSES):
            d = raw / split / c
            d.mkdir(parents=True)
            for i in range(n):
                _scan(rng, ci).save(d / f"{split[:2]}_{c}_{i}.jpg", quality=95)

    # 1) exact copy of a training image placed in the Testing folder (train/test leak)
    shutil.copy(raw / "Training/glioma/Tr_glioma_0.jpg", raw / "Testing/glioma/leak_copy.jpg")
    # 2) re-encoded, resized copy (same scan, different bytes)
    with Image.open(raw / "Training/pituitary/Tr_pituitary_1.jpg") as im:
        im.resize((128, 128)).save(raw / "Testing/pituitary/leak_resized.png")
    # 3) the same scan filed under two different labels (unreliable label)
    shutil.copy(raw / "Training/meningioma/Tr_meningioma_2.jpg", raw / "Training/notumor/conflict.jpg")
    return raw
