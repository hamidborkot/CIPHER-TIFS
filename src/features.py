"""Behavioral feature engineering: BDM, PSE, and archetype derivation."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def compute_pbi(
    df: pd.DataFrame,
    feature_cols: list[str],
    user_col: str = "user",
    threshold: float = None,
) -> pd.DataFrame:
    """
    Behavioral Drift Monitor (BDM).
    For each row, compute the z-score distance of the user's feature
    vector from the mean of their peer group (same department/role).

    Parameters
    ----------
    df            : DataFrame with user activity rows
    feature_cols  : numeric columns to include in BDM
    user_col      : column identifying each user
    threshold     : BDM alert threshold (dynamic if None)

    Returns
    -------
    df with added columns: BDM_drift, BDM_alert
    """
    df = df.copy()
    X = df[feature_cols].fillna(0).values.astype(np.float32)

    # Global mean + std as peer baseline
    mu  = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    drift = np.sqrt(((X - mu) / std) ** 2).mean(axis=1)

    df["BDM_drift"] = drift

    if threshold is None:
        threshold = np.percentile(drift, 80)  # dynamic 80th percentile
    df["BDM_alert"] = (drift > threshold).astype(int)
    return df


def compute_aif(
    df: pd.DataFrame,
    signal_cols: list[str],
    weights: list[float] = None,
) -> pd.DataFrame:
    """
    Persona-Stratified Ensemble (PSE).
    Weighted sum of normalised anomaly signals.

    Parameters
    ----------
    df          : DataFrame with anomaly signal columns
    signal_cols : columns to fuse
    weights     : per-signal weights (equal if None)

    Returns
    -------
    df with added columns: PSE_score, PSE_alert
    """
    df   = df.copy()
    X    = df[signal_cols].fillna(0).values.astype(np.float32)
    X_n  = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

    if weights is None:
        weights = np.ones(len(signal_cols)) / len(signal_cols)
    weights = np.array(weights, dtype=np.float32)

    score = X_n @ weights
    df["PSE_score"] = score
    df["PSE_alert"] = (score > np.percentile(score, 85)).astype(int)
    return df


def derive_archetypes(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, KMeans]:
    """
    Derive behavioral archetypes via K-means clustering.

    Returns
    -------
    df with added column 'archetype', fitted KMeans model
    """
    df   = df.copy()
    X    = df[feature_cols].fillna(0).values.astype(np.float32)
    sc   = StandardScaler()
    X_sc = sc.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    df["archetype"] = km.fit_predict(X_sc)
    return df, km
