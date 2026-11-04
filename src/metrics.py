"""Evaluation metrics, with emphasis on missed tumors and calibration."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import CLASSES, NO_TUMOR


def expected_calibration_error(y_true, probs, n_bins: int = 15) -> float:
    """Gap between confidence and accuracy, averaged over confidence bins (0 = perfectly calibrated)."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_metrics(y_true, probs, classes=CLASSES) -> dict:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    y_pred = probs.argmax(axis=1)
    labels = list(range(len(classes)))
    no_tumor = classes.index(NO_TUMOR)

    # Tumor vs no tumor: a tumor predicted as "notumor" is the costly error.
    is_tumor = y_true != no_tumor
    missed = int(((y_pred == no_tumor) & is_tumor).sum())
    tumor_recall = float(1 - missed / is_tumor.sum()) if is_tumor.any() else float("nan")

    report = classification_report(y_true, y_pred, labels=labels, target_names=classes, output_dict=True, zero_division=0)
    return {
        "n_test_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "tumor_recall": tumor_recall,
        "missed_tumors": missed,
        "ece": expected_calibration_error(y_true, probs),
        "per_class": {c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score")} for c in classes},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusion(cm, classes, path, title="Confusion matrix"):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(y_true, probs, path, n_bins: int = 10, title="Reliability diagram"):
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            centers.append((lo + hi) / 2)
            accs.append(correct[mask].mean())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.bar(centers, accs, width=1 / n_bins, edgecolor="black", alpha=0.75, label="Model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
