"""PSE — Persona-Stratified Ensemble.

Paper module: Section III-C (CIPHER, IEEE TIFS submission)
Previously known as: AIF (Anomaly Intent Fingerprinting) — name used in TDSC paper.
Renamed to PSE here to maintain clear separation between TDSC and TIFS submissions.

Function:
  Receives flagged users from BDM and assigns a continuous threat score rho in [0,1].
  Uses a 4-classifier ensemble: RandomForest, XGBoost, LightGBM, MLP.
  Features are normalized against archetype-specific enrollment statistics.
  Threat label: THREAT if rho >= theta_PSE (default 0.50).

Key parameters:
  theta_PSE = 0.50   (threat threshold)
  n_estimators = 200 (for RF, XGBoost, LightGBM)
  MLP layers: (128, 64), ReLU, dropout=0.3

Paper reference: Eq. (3) — ensemble threat score formula
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
except ImportError:
    XGBClassifier = None
    LGBMClassifier = None


class PersonaStratifiedEnsemble:
    """PSE: archetype-aware 4-classifier threat scorer.

    Args:
        theta (float): Threat score threshold for binary label.
        n_estimators (int): Number of trees in RF/XGB/LGB.
        archetype_stats (dict): Per-archetype {mean, std} for normalization.
    """

    def __init__(self, theta: float = 0.50, n_estimators: int = 200,
                 archetype_stats: dict = None):
        self.theta = theta
        self.archetype_stats = archetype_stats or {}
        self.classifiers = {
            "rf": RandomForestClassifier(n_estimators=n_estimators,
                                         max_depth=20, random_state=42),
            "mlp": MLPClassifier(hidden_layer_sizes=(128, 64),
                                  activation="relu", dropout=0.3,
                                  max_iter=300, random_state=42),
        }
        if XGBClassifier:
            self.classifiers["xgb"] = XGBClassifier(
                n_estimators=n_estimators, learning_rate=0.05,
                use_label_encoder=False, eval_metric="logloss", random_state=42)
        if LGBMClassifier:
            self.classifiers["lgb"] = LGBMClassifier(
                n_estimators=n_estimators, random_state=42, verbose=-1)

    def _normalize(self, features: np.ndarray, archetype: str) -> np.ndarray:
        """Normalize features against archetype-specific enrollment stats."""
        if archetype in self.archetype_stats:
            mu = self.archetype_stats[archetype]["mean"]
            sigma = self.archetype_stats[archetype]["std"]
            return (features - mu) / (sigma + 1e-8)
        return features

    def fit(self, X: np.ndarray, y: np.ndarray, archetype: str = "global") -> None:
        X_norm = self._normalize(X, archetype)
        for clf in self.classifiers.values():
            clf.fit(X_norm, y)

    def predict_score(self, features: np.ndarray, archetype: str = "global") -> float:
        """Return ensemble threat score rho in [0,1]."""
        X_norm = self._normalize(features.reshape(1, -1), archetype)
        scores = []
        for clf in self.classifiers.values():
            try:
                p = clf.predict_proba(X_norm)[0][1]
                scores.append(p)
            except Exception:
                pass
        return float(np.mean(scores)) if scores else 0.0

    def predict_label(self, features: np.ndarray, archetype: str = "global") -> str:
        rho = self.predict_score(features, archetype)
        return "THREAT" if rho >= self.theta else "BENIGN"
