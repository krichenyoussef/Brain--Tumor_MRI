"""
Step 3 - Train a 4-class brain tumor classifier with transfer learning.

- Backbone: ResNet50 pretrained on ImageNet (torchvision weights).
- Input: 224x224 RGB, ImageNet normalisation.
- Training set: data/Training, with 10% held out as validation (stratified).
- Test set:     data/Testing, used only once at the end for the report.
- Saves: outputs/model.pt (best validation checkpoint), outputs/train_log.json,
         outputs/test_report.txt, outputs/confusion_matrix.png
"""
import json, os, sys, time, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARCH = os.environ.get("ARCH", "resnet50")
OUT = ROOT / os.environ.get("OUT_DIR", "outputs")    # default keeps the original outputs/model.pt
OUT.mkdir(parents=True, exist_ok=True)


def build_model(arch, n):
    """ImageNet-pretrained backbone with a new n-class head."""
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); m.fc = nn.Linear(m.fc.in_features, n)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    elif arch == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n)
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n)
    else:
        raise ValueError(arch)
    return m

SEED = 42
EPOCHS = int(os.environ.get("EPOCHS", 8))
BATCH = 32
LR = 3e-4
IMG = 224

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def main():
    full_train = datasets.ImageFolder(DATA / "Training", transform=train_tf)
    full_val = datasets.ImageFolder(DATA / "Training", transform=eval_tf)
    test_ds = datasets.ImageFolder(DATA / "Testing", transform=eval_tf)
    classes = full_train.classes
    print("classes:", classes, "device:", device)

    targets = [y for _, y in full_train.samples]
    tr_idx, va_idx = train_test_split(
        np.arange(len(targets)), test_size=0.10, stratify=targets, random_state=SEED)
    train_ds = Subset(full_train, tr_idx)
    val_ds = Subset(full_val, va_idx)

    kw = dict(batch_size=BATCH, num_workers=4, pin_memory=True, persistent_workers=True)
    train_dl = DataLoader(train_ds, shuffle=True, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)
    test_dl = DataLoader(test_ds, shuffle=False, **kw)

    model = build_model(ARCH, len(classes)).to(device)
    print("arch:", ARCH, "out:", OUT)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * len(train_dl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def run_eval(dl):
        model.eval(); correct = 0; n = 0; loss_sum = 0.0; preds = []; trues = []
        with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
            for x, y in dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x)
                loss_sum += criterion(out, y).item() * x.size(0)
                p = out.argmax(1)
                correct += (p == y).sum().item(); n += x.size(0)
                preds.append(p.cpu()); trues.append(y.cpu())
        return loss_sum / n, correct / n, torch.cat(preds).numpy(), torch.cat(trues).numpy()

    log = []; best_acc = 0.0; t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train(); tl = 0.0; tc = 0; tn = 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out = model(x); loss = criterion(out, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
            tl += loss.item() * x.size(0); tc += (out.argmax(1) == y).sum().item(); tn += x.size(0)
        vl, va, _, _ = run_eval(val_dl)
        rec = dict(epoch=epoch, train_loss=tl / tn, train_acc=tc / tn, val_loss=vl, val_acc=va,
                   elapsed_s=round(time.time() - t0))
        log.append(rec); print(json.dumps(rec), flush=True)
        if va >= best_acc:
            best_acc = va
            torch.save({"state_dict": model.state_dict(), "classes": classes, "img": IMG, "arch": ARCH}, OUT / "model.pt")
    json.dump(log, open(OUT / "train_log.json", "w"), indent=2)

    # final test evaluation with the best checkpoint
    ck = torch.load(OUT / "model.pt", map_location=device)
    model.load_state_dict(ck["state_dict"])
    tl, ta, preds, trues = run_eval(test_dl)
    rep = classification_report(trues, preds, target_names=classes, digits=4)
    cm = confusion_matrix(trues, preds)
    txt = f"best val acc: {best_acc:.4f}\ntest acc: {ta:.4f}\n\n{rep}\n\nconfusion matrix (rows=true, cols=pred)\n{classes}\n{cm}\n"
    print(txt); open(OUT / "test_report.txt", "w").write(txt)

    # benchmark numbers: size, speed, headline metrics
    from sklearn.metrics import f1_score, recall_score
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    x1 = torch.randn(1, 3, IMG, IMG, device=device); x32 = torch.randn(32, 3, IMG, IMG, device=device)
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for _ in range(10): model(x1); model(x32)
        if device.type == "cuda": torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(50): model(x1)
        if device.type == "cuda": torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t) / 50 * 1000
        t = time.perf_counter()
        for _ in range(10): model(x32)
        if device.type == "cuda": torch.cuda.synchronize()
        thr = 320 / (time.perf_counter() - t)
    rec = recall_score(trues, preds, average=None)
    metrics = dict(arch=ARCH, params_m=round(n_params / 1e6, 2), test_acc=round(ta, 4),
                   macro_f1=round(f1_score(trues, preds, average="macro"), 4),
                   recall={c: round(float(r), 4) for c, r in zip(classes, rec)},
                   best_val_acc=round(best_acc, 4), train_time_s=log[-1]["elapsed_s"],
                   latency_bs1_ms=round(lat_ms, 2), throughput_bs32_img_s=round(thr, 1), epochs=EPOCHS)
    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2); print(json.dumps(metrics))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"test acc {ta:.3f}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=120)


if __name__ == "__main__":
    main()
