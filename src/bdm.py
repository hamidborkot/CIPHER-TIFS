"""BDM — Behavioral Drift Monitor.

Paper module: Section III-B (CIPHER, IEEE TIFS submission)
Previously known as: PBI (Persona Behavioral Integrity) — name used in TDSC paper.
Renamed to BDM here to maintain clear separation between TDSC and TIFS submissions.

Function:
  Continuously monitors each user's behavioral distribution using KL-divergence.
  When DKL(P_t || P_{t-1}) > tau_BDM, the user is flagged and forwarded to PSE.
  Otherwise, the persona baseline is updated via exponential smoothing.

Key parameters (from config/cipher_config.yaml):
  tau_BDM = 0.25   (drift threshold, grid-searched on CERT r4.2 validation set)
  alpha   = 0.10   (exponential smoothing factor)
  epsilon_s = 1e-12 (smoothing constant to prevent log(0))

Paper reference: Eq. (2) — KL-divergence formula
"""

import numpy as np
from scipy.special import rel_entr


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Compute KL divergence D_KL(p || q) with smoothing."""
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(rel_entr(p, q)))


class BehavioralDriftMonitor:
    """BDM: stateful per-user drift detector.

    Args:
        tau (float): KL-divergence threshold for drift detection.
        alpha (float): Exponential smoothing factor for persona update.
        n_bins (int): Number of histogram bins for distribution summarization.
    """

    def __init__(self, tau: float = 0.25, alpha: float = 0.10, n_bins: int = 20):
        self.tau = tau
        self.alpha = alpha
        self.n_bins = n_bins
        self.personas: dict = {}   # user_id -> current persona distribution
        self.drift_log: list = []  # list of (user_id, window_t, kl_score)

    def enroll(self, user_id: str, feature_window: np.ndarray) -> None:
        """Enroll a user with their initial behavioral distribution."""
        hist, _ = np.histogram(feature_window.flatten(), bins=self.n_bins, density=True)
        self.personas[user_id] = hist + 1e-12

    def update(self, user_id: str, feature_window: np.ndarray) -> dict:
        """Process one time window for a user.

        Returns:
            dict with keys: user_id, drifted (bool), kl_score (float)
        """
        if user_id not in self.personas:
            self.enroll(user_id, feature_window)
            return {"user_id": user_id, "drifted": False, "kl_score": 0.0}

        hist, _ = np.histogram(feature_window.flatten(), bins=self.n_bins, density=True)
        p_current = hist + 1e-12
        p_prev = self.personas[user_id]

        kl = kl_divergence(p_current, p_prev)
        drifted = kl > self.tau

        if drifted:
            self.drift_log.append((user_id, kl))
        else:
            # Exponential smoothing update
            self.personas[user_id] = (
                (1 - self.alpha) * p_prev + self.alpha * p_current
            )

        return {"user_id": user_id, "drifted": drifted, "kl_score": kl}
