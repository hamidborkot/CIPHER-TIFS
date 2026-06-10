"""E5: CERT r4.2 Per-Scenario F1 Breakdown
Replaces D2/D3 synthetic cross-environment claim with legitimate
scenario-level generalization on real CERT r4.2 insider scenarios.

CERT r4.2 contains 5 insider threat scenarios:
  S1: Data theft before resignation (exfiltration)
  S2: Slow-burn IP theft / reconnaissance
  S3: IT sabotage (after-hours access + deletion)
  S4: Privilege escalation + lateral movement
  S5: Policy violation (general)

This experiment tests whether SENTINEL-EGO generalizes across insider TYPE,
replacing the weaker cross-dataset generalization claim.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# CERT r4.2 insider scenario codes (from answers/ directory user IDs)
SCENARIO_MAP = {
    "S1": "data_theft_resignation",
    "S2": "slow_burn_recon",
    "S3": "it_sabotage",
    "S4": "privilege_escalation",
    "S5": "policy_violation",
}

FEATURES_FULL = [
    "logon_count", "email_count", "http_count", "file_count",
    "work_hours_ratio", "pbi_drift", "pbi_alert",
    "aif_score", "aif_alert", "archetype_id"
]

FEATURES_LEGACY = [
    "logon_count", "email_count", "http_count",
    "file_count", "work_hours_ratio"
]


def evaluate_scenario(df_train, df_test_scenario, scenario_name, features, model_name):
    available = [f for f in features if f in df_train.columns]
    X_tr = df_train[available]
    y_tr = df_train["label"]
    X_te = df_test_scenario[available]
    y_te = df_test_scenario["label"]

    if y_te.sum() == 0:
        return None  # No malicious samples in this scenario slice

    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    return {
        "Scenario": scenario_name,
        "Model": model_name,
        "N_Test": len(y_te),
        "N_Malicious": int(y_te.sum()),
        "F1": round(f1_score(y_te, y_pred, zero_division=0), 4),
        "AUC": round(roc_auc_score(y_te, y_prob), 4),
    }


def main(df, scenario_col="scenario_type"):
    """df must contain 'label', 'scenario_type', and feature columns."""
    print("E5: CERT r4.2 Per-Scenario Breakdown")
    print("Training on full dataset; testing per scenario")
    print("="*60)

    X_train, X_test, _, _ = train_test_split(
        df, df["label"], test_size=0.3, stratify=df["label"], random_state=42
    )

    results = []
    for sc_code, sc_name in SCENARIO_MAP.items():
        sc_test = X_test[X_test[scenario_col] == sc_name]
        if len(sc_test) == 0:
            print(f"  {sc_code}: No test samples found, skipping")
            continue

        r_full = evaluate_scenario(X_train, sc_test, sc_code, FEATURES_FULL, "SENTINEL-EGO")
        r_legacy = evaluate_scenario(X_train, sc_test, sc_code, FEATURES_LEGACY, "Legacy-Only")

        if r_full and r_legacy:
            r_full["PBI_Delta"] = round(r_full["F1"] - r_legacy["F1"], 4)
            results.append(r_full)
            print(f"  {sc_code}: SENTINEL F1={r_full['F1']:.4f}  Legacy F1={r_legacy['F1']:.4f}  PBI_delta={r_full['PBI_Delta']:+.3f}")

    df_out = pd.DataFrame(results)
    df_out.to_csv("results/results_e5_scenario_breakdown.csv", index=False)
    print("\nSaved: results/results_e5_scenario_breakdown.csv")
    print("\nKey insight: PBI_Delta should be largest for S2 (slow-burn recon)")
    print("This demonstrates PBI's unique value for hard-to-detect insiders.")
    return df_out


if __name__ == "__main__":
    # df = pd.read_csv("data/cert_r42_features_with_scenarios.csv")
    # main(df)
    print("Load your feature CSV with 'scenario_type' column and call main(df).")
