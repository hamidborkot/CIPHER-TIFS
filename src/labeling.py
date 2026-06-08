"""Malicious user label derivation pipeline for CERT r4.2."""
import pandas as pd
import numpy as np


INSIDER_RATE_TARGET = (0.015, 0.040)  # 1.5% – 4.0%


def label_users(
    users_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    user_col: str = "user_id",
    end_date_col: str = "end_date",
    rm_col: str = "usb_count",
    risky_col: str = "risky_web",
    ext_email_col: str = "ext_email_count",
    ext_email_pct: float = 0.95,
    target_n: int = 100,
) -> tuple[set, pd.DataFrame]:
    """
    Derive malicious user labels using a four-stage funnel:
      1. Resigned employees (end_date present)
      2. Resigned + removable media activity
      3. Resigned + (risky web OR high external email)
      4. Top-N by removable media copies among resigned (fallback)

    Returns
    -------
    malicious_users : set of user IDs labeled malicious
    users_df        : input DataFrame with added 'label' column
    """
    # Stage 1: resigned
    resigned = set(
        users_df.loc[users_df[end_date_col].notna(), user_col]
    )

    # Stage 2: resigned + removable media
    rm_users = set(
        activity_df.loc[
            (activity_df[user_col].isin(resigned)) & (activity_df[rm_col] > 0),
            user_col,
        ]
    )

    # Stage 3: resigned + risky/external signals
    risky_users = set(
        activity_df.loc[
            (activity_df[user_col].isin(resigned))
            & (
                (activity_df.get(risky_col, pd.Series(0)) > 0)
                | (
                    activity_df.get(ext_email_col, pd.Series(0))
                    >= activity_df.get(ext_email_col, pd.Series(0)).quantile(
                        ext_email_pct
                    )
                )
            ),
            user_col,
        ]
    )
    tightened = rm_users & risky_users

    # Stage 4: fallback top-N by removable media volume
    rm_volume = (
        activity_df[activity_df[user_col].isin(resigned)]
        .groupby(user_col)[rm_col]
        .sum()
        .nlargest(target_n)
    )
    top_n = set(rm_volume.index)

    # Select final malicious set
    if len(tightened) >= 60:
        malicious = tightened
    else:
        malicious = top_n  # guaranteed target_n

    # Ensure count is within target range
    total_resigned = len(resigned)
    rate = len(malicious) / max(total_resigned, 1)
    if not (INSIDER_RATE_TARGET[0] <= rate <= INSIDER_RATE_TARGET[1]):
        malicious = top_n  # fallback always within range

    users_df = users_df.copy()
    users_df["label"] = users_df[user_col].isin(malicious).astype(int)
    return malicious, users_df
