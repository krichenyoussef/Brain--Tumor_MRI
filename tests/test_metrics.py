import numpy as np

from src.config import CLASSES
from src.metrics import compute_metrics, expected_calibration_error, plot_confusion, plot_reliability

G, M, N, P = (CLASSES.index(c) for c in ("glioma", "meningioma", "notumor", "pituitary"))


def one_hot(preds, conf=0.9):
    probs = np.full((len(preds), 4), (1 - conf) / 3)
    probs[np.arange(len(preds)), preds] = conf
    return probs


def test_missed_tumors_and_tumor_recall():
    y_true = np.array([G, G, M, P, N, N])
    y_pred = np.array([G, N, M, P, N, G])  # one glioma missed (predicted no tumor), one false alarm
    m = compute_metrics(y_true, one_hot(y_pred))
    assert m["missed_tumors"] == 1
    assert m["tumor_recall"] == 0.75
    assert abs(m["accuracy"] - 4 / 6) < 1e-9
    assert m["confusion_matrix"][G][N] == 1


def test_ece_is_zero_when_confidence_matches_accuracy():
    y = np.array([0, 1, 2, 3])
    assert expected_calibration_error(y, one_hot(y, conf=1.0)) == 0.0
    # always 90% confident but always right -> 0.1 gap
    assert abs(expected_calibration_error(y, one_hot(y, conf=0.9)) - 0.1) < 1e-9


def test_plots_are_written(tmp_path):
    y = np.array([G, M, N, P, G])
    probs = one_hot(np.array([G, M, N, P, M]), conf=0.7)
    m = compute_metrics(y, probs)
    plot_confusion(m["confusion_matrix"], CLASSES, tmp_path / "cm.png")
    plot_reliability(y, probs, tmp_path / "rel.png")
    assert (tmp_path / "cm.png").stat().st_size > 0 and (tmp_path / "rel.png").stat().st_size > 0
