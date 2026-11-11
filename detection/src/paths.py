"""
Where the Kaggle images live. Shared by every detection script.

Search order (first folder that contains Training/ wins):
  1. $BRAIN_TUMOR_DATA
  2. <repo>/data/raw   (same location the main classification project uses)
  3. <repo>/data
  4. <repo>/detection/data
"""
import os
from pathlib import Path

DETECTION_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = DETECTION_DIR.parent


def _find_data():
    env = os.environ.get("BRAIN_TUMOR_DATA")
    if env:
        return Path(env)
    for cand in (REPO_DIR / "data" / "raw", REPO_DIR / "data", DETECTION_DIR / "data"):
        if (cand / "Training").is_dir():
            return cand
    return REPO_DIR / "data" / "raw"


DATA_DIR = _find_data()
