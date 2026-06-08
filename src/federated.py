"""
federated.py — SENTINEL-EGO Federated Learning Components

Includes:
  - local_train       : DP-SGD local training with focal class weighting
  - fed_avg           : weighted FedAvg aggregation
  - multi_krum        : Multi-Krum Byzantine-robust aggregation
  - malicious_train   : Byzantine attack simulation (label flip + grad scaling)
  - run_fl            : standard FL loop (FedAvg)
  - run_fl_byzantine  : FL loop with mixed honest/Byzantine clients
  - compute_epsilon   : Rényi-DP epsilon accounting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

# ── Default hyperparameters (override via config.yaml) ──────
FL_ROUNDS    = 10
LOCAL_EPOCHS = 3
LR           = 0.001
NOISE_SCALE  = 2.0
CLIP_NORM    = 1.0
Q_SAMPLE     = 0.01
DP_DELTA     = 1e-5
N_CLIENTS    = 10
N_BYZANTINE  = 3
GRAD_SCALE   = 10.0


def compute_epsilon(q: float, sigma: float, steps: int,
                    delta: float = DP_DELTA, alpha: int = 10) -> float:
    """Rényi-DP to (epsilon, delta)-DP conversion.

    Args:
        q:      Poisson subsampling rate per step
        sigma:  Gaussian noise multiplier
        steps:  Total number of gradient steps (rounds * local_epochs)
        delta:  DP failure probability
        alpha:  Rényi order (default 10)

    Returns:
        epsilon: privacy budget
    """
    rdp = alpha * (q ** 2) / (2 * sigma ** 2) * steps
    return rdp + np.log(1 / delta) / (alpha - 1)


def local_train(model, Xn: np.ndarray, yn: np.ndarray,
                epochs: int = LOCAL_EPOCHS,
                apply_dp: bool = True) -> nn.Module:
    """Honest local training with focal class weighting and optional DP-SGD.

    Class weight is capped at 40 to prevent gradient explosion on highly
    imbalanced shards where a client has very few malicious samples.

    Args:
        model:     ThreatNet instance (in-place modified and returned)
        Xn:        Feature matrix (numpy float32)
        yn:        Binary label vector (numpy float32)
        epochs:    Number of local epochs
        apply_dp:  If True, apply per-sample gradient clipping + Gaussian noise

    Returns:
        Trained model
    """
    model.train()
    n_pos = max(int((yn == 1).sum()), 1)
    n_neg = max(int((yn == 0).sum()), 1)
    pos_w = torch.tensor([min(n_neg / n_pos, 40.0)], dtype=torch.float32)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    ds    = TensorDataset(torch.tensor(Xn, dtype=torch.float32),
                          torch.tensor(yn, dtype=torch.float32))
    bs    = max(32, int(Q_SAMPLE * len(Xn)))
    dl    = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)

    for _ in range(epochs):
        for Xb, yb in dl:
            opt.zero_grad()
            wt   = torch.where(yb == 1, pos_w.expand_as(yb), torch.ones_like(yb))
            loss = (nn.functional.binary_cross_entropy(
                        model(Xb), yb, reduction='none') * wt).mean()
            loss.backward()
            if apply_dp:
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    norm = p.grad.norm(2)
                    p.grad.mul_(min(1.0, CLIP_NORM / (norm + 1e-9)))
                    p.grad.add_(torch.randn_like(p.grad) * NOISE_SCALE * CLIP_NORM)
            opt.step()
    return model


def malicious_train(model, Xn: np.ndarray, yn: np.ndarray,
                    epochs: int = LOCAL_EPOCHS,
                    scale: float = GRAD_SCALE) -> nn.Module:
    """Byzantine attack simulation: label flip (1->0) + gradient scaling.

    The attacker flips all insider labels to benign to poison the global
    model toward missing insiders, then amplifies gradients by `scale` to
    dominate the FedAvg weight update (model poisoning).

    This attack is empirically effective against standard FedAvg but is
    filtered by Multi-Krum's distance-based selection criterion.

    Args:
        model:   ThreatNet instance
        Xn:      Feature matrix
        yn:      True label vector (will be flipped internally)
        epochs:  Local training epochs
        scale:   Gradient amplification factor

    Returns:
        Poisoned model
    """
    model.train()
    yn_flipped = yn.copy()
    yn_flipped[yn == 1] = 0
    opt = optim.Adam(model.parameters(), lr=LR)
    ds  = TensorDataset(torch.tensor(Xn, dtype=torch.float32),
                        torch.tensor(yn_flipped, dtype=torch.float32))
    bs  = max(32, int(Q_SAMPLE * len(Xn)))
    dl  = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)

    for _ in range(epochs):
        for Xb, yb in dl:
            opt.zero_grad()
            nn.functional.binary_cross_entropy(model(Xb), yb).backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
            opt.step()
    return model


def fed_avg(global_model, local_models: list, weights: list) -> nn.Module:
    """Weighted FedAvg aggregation.

    Args:
        global_model:  Current global ThreatNet model (updated in-place)
        local_models:  List of locally-trained models
        weights:       List of weights (e.g. client_size / total_size)

    Returns:
        Updated global model
    """
    gd = global_model.state_dict()
    for k in gd:
        gd[k] = torch.stack(
            [lm.state_dict()[k].float() * w
             for lm, w in zip(local_models, weights)], 0
        ).sum(0)
    global_model.load_state_dict(gd)
    return global_model


def multi_krum(models: list, n_byzantine: int = N_BYZANTINE) -> nn.Module:
    """Multi-Krum Byzantine-robust aggregation.

    Scores each model by the sum of squared L2 distances to its
    k = n - n_byzantine - 2 nearest neighbors. Selects the top
    m = n - n_byzantine lowest-scoring (most honest) models and
    returns their equal average.

    Theoretical guarantee: tolerates up to n_byzantine < n/2 adversaries.
    Requires n >= 2*n_byzantine + 3 clients.

    Reference:
        Blanchard et al., "Machine Learning with Adversaries:
        Byzantine Tolerant Gradient Descent", NeurIPS 2017.

    Args:
        models:      List of local ThreatNet models
        n_byzantine: Number of assumed Byzantine clients

    Returns:
        Aggregated model from selected honest candidates
    """
    params = [torch.cat([p.data.view(-1) for p in m.parameters()])
              for m in models]
    n     = len(params)
    k     = max(1, n - n_byzantine - 2)
    m_sel = max(1, n - n_byzantine)

    scores = []
    for i in range(n):
        dists = sorted(
            [((params[i] - params[j]) ** 2).sum().item()
             for j in range(n) if j != i]
        )
        scores.append(sum(dists[:k]))

    selected_ids = np.argsort(scores)[:m_sel]
    selected     = [models[i] for i in selected_ids]

    gd = deepcopy(selected[0]).state_dict()
    for key in gd:
        gd[key] = torch.stack(
            [s.state_dict()[key].float() for s in selected], 0
        ).mean(0)
    result = deepcopy(selected[0])
    result.load_state_dict(gd)
    return result


def run_fl(Xtr: np.ndarray, ytr: np.ndarray, masks: list,
           apply_dp: bool = True, label: str = "FL",
           n_rounds: int = FL_ROUNDS,
           eval_fn=None, X_test=None, y_test=None):
    """Standard FL training loop (FedAvg).

    Args:
        Xtr:       Training feature matrix
        ytr:       Training label vector
        masks:     List of boolean index arrays, one per client
        apply_dp:  Enable DP-SGD in local training
        label:     Display label for progress output
        n_rounds:  Number of federation rounds
        eval_fn:   Optional evaluation function (model, X, y) -> metrics dict
        X_test:    Test features for mid-training evaluation
        y_test:    Test labels for mid-training evaluation

    Returns:
        (global_model, epsilon)
    """
    from src.model import ThreatNet
    gm = ThreatNet(Xtr.shape[1])

    for rnd in range(n_rounds):
        lms, sizes = [], []
        for mask in masks:
            if mask.sum() < 10:
                continue
            lm = deepcopy(gm)
            lm = local_train(lm, Xtr[mask], ytr[mask], apply_dp=apply_dp)
            lms.append(lm)
            sizes.append(int(mask.sum()))
        if not lms:
            continue
        total = sum(sizes)
        gm    = fed_avg(gm, lms, [s / total for s in sizes])

        if eval_fn is not None and X_test is not None and rnd % 3 == 2:
            r, _ = eval_fn(gm, X_test, y_test)
            print(f"  [{label}] Round {rnd+1}/{n_rounds} "
                  f"F1={r['F1']:.4f} AUC={r['AUC']:.4f}")

    eps = compute_epsilon(Q_SAMPLE, NOISE_SCALE, n_rounds * LOCAL_EPOCHS)
    return gm, eps


def run_fl_byzantine(Xtr: np.ndarray, ytr: np.ndarray, masks: list,
                     poison_ids: set,
                     aggregation: str = "fedavg",
                     label: str = "FL",
                     n_rounds: int = FL_ROUNDS):
    """FL training loop with simulated Byzantine clients.

    Honest clients train with DP-SGD (local_train).
    Byzantine clients apply label flip + gradient scaling (malicious_train).
    Aggregation is either standard FedAvg or Multi-Krum.

    Args:
        Xtr:         Training features
        ytr:         Training labels (unmodified — Byzantine clients flip internally)
        masks:       Client index masks
        poison_ids:  Set of client indices that are Byzantine
        aggregation: 'fedavg' or 'krum'
        label:       Display label
        n_rounds:    Federation rounds

    Returns:
        (global_model, epsilon)
    """
    from src.model import ThreatNet
    gm = ThreatNet(Xtr.shape[1])

    for _ in range(n_rounds):
        lms, sizes = [], []
        for i, mask in enumerate(masks):
            if mask.sum() < 10:
                continue
            lm = deepcopy(gm)
            if i in poison_ids:
                lm = malicious_train(lm, Xtr[mask], ytr[mask])
            else:
                lm = local_train(lm, Xtr[mask], ytr[mask], apply_dp=True)
            lms.append(lm)
            sizes.append(int(mask.sum()))
        if not lms:
            continue
        if aggregation == "krum":
            gm = multi_krum(lms, n_byzantine=N_BYZANTINE)
        else:
            total = sum(sizes)
            gm    = fed_avg(gm, lms, [s / total for s in sizes])

    eps = compute_epsilon(Q_SAMPLE, NOISE_SCALE, n_rounds * LOCAL_EPOCHS)
    print(f"  [{label}] ε={eps:.4f}  aggregation={aggregation}")
    return gm, eps
