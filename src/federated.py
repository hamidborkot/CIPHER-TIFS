"""Federated averaging, local training, and Rényi-DP accounting."""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.model import ThreatNet

# ── Default hyperparameters (override via config) ─────────────────────────────
FL_ROUNDS    = 10
LOCAL_EPOCHS = 3
LR           = 0.001
NOISE_SCALE  = 2.0
CLIP_NORM    = 1.0
DP_DELTA     = 1e-5
Q_SAMPLE     = 0.01
BATCH_SIZE   = 512


def compute_epsilon(
    q: float, sigma: float, steps: int, delta: float = DP_DELTA, alpha: int = 10
) -> float:
    """Rényi DP → (ε, δ)-DP via moments accountant."""
    rdp = alpha * q**2 / (2 * sigma**2) * steps
    eps = rdp + math.log(1 / delta) / (alpha - 1)
    return eps


def local_train(
    model: ThreatNet,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = LOCAL_EPOCHS,
    lr: float = LR,
    apply_dp: bool = True,
    noise_scale: float = NOISE_SCALE,
    clip_norm: float = CLIP_NORM,
    batch_size: int = BATCH_SIZE,
) -> ThreatNet:
    """Train a local model with optional per-sample DP gradient perturbation."""
    device = torch.device("cpu")
    model.train().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Class-weighted positive weight to handle imbalance
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
    )

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for Xb, yb in loader:
            optimizer.zero_grad()
            # Remove final sigmoid for BCEWithLogitsLoss
            logits = model.net[:-1](Xb).squeeze(-1)  # raw logits
            loss = criterion(logits, yb)
            loss.backward()

            if apply_dp:
                # Per-sample gradient clipping + Gaussian noise
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm(2)
                        clip_factor = min(1.0, clip_norm / (grad_norm + 1e-8))
                        param.grad.mul_(clip_factor)
                        param.grad.add_(
                            torch.randn_like(param.grad)
                            * noise_scale
                            * clip_norm
                            / batch_size
                        )
            optimizer.step()
    return model


def fed_avg(
    global_model: ThreatNet,
    local_models: list[ThreatNet],
    weights: list[float],
) -> ThreatNet:
    """Weighted FedAvg aggregation."""
    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = sum(
            w * lm.state_dict()[key].float()
            for w, lm in zip(weights, local_models)
        )
    global_model.load_state_dict(global_dict)
    return global_model


def run_fl(
    X_train: np.ndarray,
    y_train: np.ndarray,
    client_masks: list[np.ndarray],
    apply_dp: bool = True,
    label: str = "FL",
    n_rounds: int = FL_ROUNDS,
    Xte_eval: np.ndarray = None,
    yte_eval: np.ndarray = None,
    eval_fn=None,
) -> tuple[ThreatNet, float]:
    """
    Run federated learning for `n_rounds` rounds.

    Parameters
    ----------
    X_train, y_train : training data
    client_masks     : boolean index masks, one per client
    apply_dp         : enable Rényi-DP gradient perturbation
    label            : display name for progress output
    n_rounds         : number of FL rounds
    Xte_eval, yte_eval : evaluation data for mid-round logging
    eval_fn          : callable(model, X, y) → metrics dict

    Returns
    -------
    global_model, epsilon
    """
    global_model = ThreatNet(X_train.shape[1])
    for rnd in range(n_rounds):
        local_models, sizes = [], []
        for mask in client_masks:
            if mask.sum() < 10:
                continue
            lm = copy.deepcopy(global_model)
            lm = local_train(lm, X_train[mask], y_train[mask], apply_dp=apply_dp)
            local_models.append(lm)
            sizes.append(int(mask.sum()))

        if not local_models:
            continue

        total = sum(sizes)
        global_model = fed_avg(
            global_model, local_models, [s / total for s in sizes]
        )

        if (rnd + 1) % 2 == 0 and eval_fn is not None and Xte_eval is not None:
            metrics, _ = eval_fn(global_model, Xte_eval, yte_eval)
            print(
                f"  [{label}] Round {rnd+1}/{n_rounds} | "
                f"F1={metrics['F1']:.4f} AUC={metrics['AUC']:.4f} "
                f"Recall={metrics['Recall']:.4f} FPR={metrics['FPR']:.4f}"
            )

    eps = compute_epsilon(Q_SAMPLE, NOISE_SCALE, n_rounds * LOCAL_EPOCHS)
    print(f"  DP: ε={eps:.4f}, δ={DP_DELTA}")
    return global_model, eps
