"""
Benchmark - train a YOLO11n detector on one of the exported datasets.

usage: python src/train_yolo.py <gold|full> [epochs]
Weights land in outputs/yolo/<mode>/weights/{last,best}.pt.
The benchmark uses last.pt: best.pt is chosen on the val set, which is also the benchmark set,
so using it would leak test information into model selection.
"""
import sys
from pathlib import Path
from ultralytics.utils import SETTINGS
# Turn off experiment-tracker integrations for this process only (dict.__setitem__ skips saving to the
# global ultralytics settings file). The installed mlflow crashes on the Windows default tracking path.
for k in ("mlflow", "comet", "clearml", "dvc", "neptune", "wandb", "hub", "raytune", "tensorboard"):
    if k in SETTINGS:
        dict.__setitem__(SETTINGS, k, False)
from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
mode = sys.argv[1]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else (100 if mode == "gold" else 40)

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    model.train(data=str(ROOT / f"yolo_{mode}" / "data.yaml"), imgsz=512, epochs=epochs, batch=16,
                project=str(ROOT / "outputs" / "yolo"), name=mode, exist_ok=True, seed=0,
                workers=2, patience=0, plots=True, verbose=False)
    # workers=2: each worker is a full Python+CUDA process (~1 GB of committed memory on Windows);
    # with 4 the 16 GB laptop ran out of paging file when the loaders restarted at close_mosaic.
