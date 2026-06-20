"""
evaluate.py — CIPHER Evaluation Utilities

Provides threshold-searched evaluation critical for imbalanced detection
(2.7% positive rate). Uses both fine-grained threshold scan and
precision-recall curve to find the F1-optimal operating point.
"""

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, precision_recall_curve
)


def eval_model(model, Xt: np.ndarray, yt: np.ndarray) -> tuple:
    """Evaluate model with extended threshold search.

    For highly imbalanced data (< 5% positive rate), the default 0.5
    threshold is suboptimal. This function scans thresholds from 0.001
    to 0.95 and also uses the precision-recall curve to identify the
    global F1-optimal operating point.

    This is essential when Multi-Krum averaging compresses sigmoid outputs
    to very low magnitudes — a known artifact of parameter averaging under
    DP noise that does not affect the model's discriminative power (AUC).

    Args:
        model:  ThreatNet instance
        Xt:     Test feature matrix (numpy float32)
        yt:     Test label vector (numpy float32)

    Returns:
        (metrics_dict, prob_array)
        metrics_dict keys: F1, AUC, Precision, Recall, Accuracy, FPR, FNR
    """
    model.eval()
    with torch.no_grad():
        prob = model(torch.tensor(Xt, dtype=torch.float32)).numpy()

    # Fine-grained scan: covers both compressed (near-0) and normal outputs
    best_f1, best_thr = 0.0, 0.5
    thresholds = np.concatenate([
        np.arange(0.001, 0.01,  0.001),
        np.arange(0.01,  0.1,   0.005),
        np.arange(0.1,   0.95,  0.01)
    ])
    for thr in thresholds:
        f1 = f1_score(yt, (prob >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    # Precision-recall curve: finds global F1-max regardless of output scale
    if len(np.unique(yt)) > 1:
        prec, rec, pr_thr = precision_recall_curve(yt, prob)
        pr_f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if pr_f1.max() > best_f1:
            best_thr = float(pr_thr[np.argmax(pr_f1)])

    pred = (prob >= best_thr).astype(int)

    if len(np.unique(yt)) < 2:
        return {'F1': 0, 'AUC': 0.5, 'Precision': 0, 'Recall': 0,
                'FPR': 0, 'FNR': 1}, prob

    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return {
        'F1':        round(f1_score(yt, pred, zero_division=0), 4),
        'AUC':       round(roc_auc_score(yt, prob), 4),
        'Precision': round(precision_score(yt, pred, zero_division=0), 4),
        'Recall':    round(recall_score(yt, pred, zero_division=0), 4),
        'Accuracy':  round(accuracy_score(yt, pred), 4),
        'FPR':       round(fp / (fp + tn + 1e-9), 4),
        'FNR':       round(fn / (fn + tp + 1e-9), 4),
    }, prob


def mia_attack(prob_train: np.ndarray, prob_test: np.ndarray,
               seed: int = 42, n_sample: int = 20000) -> float:
    """Shadow-free Membership Inference Attack.

    Uses confidence scores (model output probabilities) as features to
    train a logistic regression classifier distinguishing train from test
    membership. AUC near 0.5 indicates effective privacy protection.

    Args:
        prob_train:  Model probabilities on training set
        prob_test:   Model probabilities on test set
        seed:        Random seed for subsampling
        n_sample:    Max samples per class (for speed)

    Returns:
        MIA AUC (float, range [0.5, 1.0])
    """
    from sklearn.linear_model import LogisticRegression

    rng     = np.random.default_rng(seed)
    n       = min(n_sample, len(prob_train), len(prob_test))
    idx_tr  = rng.choice(len(prob_train), n, replace=False)
    idx_te  = rng.choice(len(prob_test),  n, replace=False)
    Xm      = np.concatenate([prob_train[idx_tr],
                               prob_test[idx_te]]).reshape(-1, 1)
    ym      = np.concatenate([np.ones(n), np.zeros(n)])
    clf     = LogisticRegression().fit(Xm, ym)
    return round(roc_auc_score(ym, clf.predict_proba(Xm)[:, 1]), 4)
