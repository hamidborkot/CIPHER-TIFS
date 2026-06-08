"""Evaluation utilities: threshold search and metric computation."""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model import ThreatNet


def best_threshold_eval(
    model: ThreatNet,
    X: np.ndarray,
    y: np.ndarray,
    thr_range: tuple[float, float, float] = (0.05, 0.95, 0.01),
) -> tuple[dict, float]:
    """
    Evaluate model over a range of classification thresholds.
    Returns the threshold that maximises F1 and its full metric dict.
    """
    model.eval()
    with torch.no_grad():
        prob = model(torch.tensor(X, dtype=torch.float32)).numpy()

    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(*thr_range):
        pred = (prob >= thr).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    pred_final = (prob >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred_final, labels=[0, 1]).ravel()

    metrics = {
        "F1":        round(float(f1_score(y, pred_final, zero_division=0)), 4),
        "AUC":       round(float(roc_auc_score(y, prob)), 4),
        "Precision": round(float(precision_score(y, pred_final, zero_division=0)), 4),
        "Recall":    round(float(recall_score(y, pred_final, zero_division=0)), 4),
        "Accuracy":  round(float(accuracy_score(y, pred_final)), 4),
        "FPR":       round(float(fp / (fp + tn + 1e-9)), 4),
        "FNR":       round(float(fn / (fn + tp + 1e-9)), 4),
        "threshold": round(float(best_thr), 2),
    }
    return metrics, best_thr
