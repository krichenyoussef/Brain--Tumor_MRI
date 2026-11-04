"""Fine-tune one backbone, keep the best epoch on validation, then evaluate on the test split.

Usage:
    python -m src.train --model resnet50 --epochs 15
    python -m src.train --model all          # resnet50, efficientnet_b0 and vit_small, one after another
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import BACKBONES, CHECKPOINTS, CLASSES, IMG_SIZE, MANIFEST, RESULTS, SEED
from src.data import MRIDataset, eval_transform, train_transform
from src.metrics import compute_metrics, plot_confusion, plot_reliability
from src.models import build_model, count_params


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def train_one(name: str, args) -> dict:
    timm_name = BACKBONES.get(name, name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    loaders = {
        split: DataLoader(
            MRIDataset(args.manifest, split, train_transform(args.img_size) if split == "train" else eval_transform(args.img_size)),
            batch_size=args.batch_size, shuffle=(split == "train"), num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        for split in ("train", "val", "test")
    }

    model = build_model(timm_name, len(CLASSES), pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = Path(args.checkpoints) / f"{name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, waited, history = -1.0, 0, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()

        y_val, p_val = predict(model, loaders["val"], device)
        val = compute_metrics(y_val, p_val)
        train_loss = running / len(loaders["train"].dataset)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"]})
        print(f"[{name}] epoch {epoch:02d}  loss {train_loss:.4f}  val F1 {val['macro_f1']:.4f}  "
              f"val acc {val['accuracy']:.4f}  ({time.time() - t0:.0f}s)")

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, waited = val["macro_f1"], epoch, 0
            torch.save({"model": name, "timm_name": timm_name, "classes": CLASSES, "img_size": args.img_size,
                        "state_dict": model.state_dict()}, ckpt_path)
        else:
            waited += 1
            if waited >= args.patience:
                print(f"[{name}] early stop: no validation gain for {args.patience} epochs")
                break

    # Test on the best checkpoint only (never used for model selection).
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"])
    y_test, p_test = predict(model, loaders["test"], device)
    metrics = compute_metrics(y_test, p_test)
    metrics.update({"model": name, "timm_name": timm_name, "params_m": round(count_params(model), 2),
                    "best_epoch": best_epoch, "pretrained": not args.no_pretrained, "history": history})

    out = Path(args.results) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(metrics["confusion_matrix"], CLASSES, out / "confusion_matrix.png", f"{name}: test confusion matrix")
    plot_reliability(y_test, p_test, out / "reliability.png", title=f"{name}: calibration")
    print(f"[{name}] test accuracy {metrics['accuracy']:.4f}  macro F1 {metrics['macro_f1']:.4f}  "
          f"tumor recall {metrics['tumor_recall']:.4f}  missed tumors {metrics['missed_tumors']}")
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="resnet50", help=f"one of {list(BACKBONES)}, any timm model name, or 'all'")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--checkpoints", type=Path, default=CHECKPOINTS)
    p.add_argument("--results", type=Path, default=RESULTS)
    p.add_argument("--no-pretrained", action="store_true", help="train from scratch (used by the tests)")
    args = p.parse_args(argv)

    names = list(BACKBONES) if args.model == "all" else [args.model]
    return [train_one(n, args) for n in names]


if __name__ == "__main__":
    main()
