# ============================================================
# CELL K_FINISH — SENTINEL-EGO r4.2 completion: E2 + E4 + E7
# ============================================================
# PREREQUISITES: Run the main E1 cell first so these are in memory:
#   X_train, X_test, y_train, y_test, test_df
#   FEAT_COLS, client_masks
#   fed_m, r_fed, eps_fed          (E1 federated model + results)
#   r_iso                          (isolated-DP result dict)
#   r_gbt                          (centralized GBT result dict)
#   run_fl(), best_eval()          (helper functions)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, recall_score, confusion_matrix


# ════════════════════════════════════════════════════════════
# E2: Ablation Study — FIXED
# Legacy-Only EXCLUDES usb_count, rm_copies, rm_ratio
# so those USB-correlated columns cannot carry detection alone
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("E2: Ablation Study — FIXED")
print("=" * 60)


def safe_cols(wanted):
    """Return only feature names that exist in FEAT_COLS."""
    return [c for c in wanted if c in FEAT_COLS]


# Feature sets — Legacy-Only is strictly behavioral, NO USB signals
ABLATION_SETS = {
    'Legacy-Only': safe_cols([
        'logon_count', 'after_hrs_logon', 'unique_pcs', 'ah_ratio',
        'file_ops', 'email_sent', 'ext_email', 'avg_mail_size', 'ext_ratio',
        'http_count', 'risky_count', 'risky_ratio',
        'role_changes', 'dept_changes', 'O', 'C', 'E', 'A', 'N'
    ]),
    '+PBI': safe_cols([
        'logon_count', 'after_hrs_logon', 'unique_pcs', 'ah_ratio',
        'file_ops', 'email_sent', 'ext_email', 'avg_mail_size', 'ext_ratio',
        'http_count', 'risky_count', 'risky_ratio',
        'role_changes', 'dept_changes', 'O', 'C', 'E', 'A', 'N',
        'pbi_drift', 'pbi_alert'
    ]),
    '+PBI+AIF': safe_cols([
        'logon_count', 'after_hrs_logon', 'unique_pcs', 'ah_ratio',
        'file_ops', 'email_sent', 'ext_email', 'avg_mail_size', 'ext_ratio',
        'http_count', 'risky_count', 'risky_ratio',
        'role_changes', 'dept_changes', 'O', 'C', 'E', 'A', 'N',
        'pbi_drift', 'pbi_alert', 'aif_score', 'aif_alert'
    ]),
    'Full SENTINEL-EGO': list(FEAT_COLS)
}

abl_results = {}
for name, cols in ABLATION_SETS.items():
    if len(cols) < 2:
        print(f"  SKIP {name}: only {len(cols)} feature(s)")
        continue
    idx = np.array([FEAT_COLS.index(c) for c in cols])
    Xtr_s = X_train[:, idx]
    Xte_s = X_test[:, idx]
    print(f"\n  Variant: {name} ({len(cols)} features)")
    m, _ = run_fl(Xtr_s, y_train, client_masks,
                  Xte_s, y_test, apply_dp=True, label=name[:16])
    r, _ = best_eval(m, Xte_s, y_test)
    r['n_features'] = len(cols)
    abl_results[name] = r
    print(f"  → {name}: F1={r['F1']:.4f} AUC={r['AUC']:.4f} "
          f"Recall={r['Recall']:.4f} FPR={r['FPR']:.4f}")

df_abl = pd.DataFrame(abl_results).T
df_abl.to_csv("results_e2_ablation_FIXED.csv")
print("\nE2 Final Table:")
print(df_abl[['F1', 'AUC', 'Recall', 'FPR', 'n_features']].to_string())
print("✓ Saved: results_e2_ablation_FIXED.csv")


# ════════════════════════════════════════════════════════════
# E4: No-DP Baseline + Final Comparison Table
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("No-DP Baseline + E4 Comparison Table")
print("=" * 60)

# Train FL without DP to quantify the privacy cost
m_nodp, _ = run_fl(X_train, y_train, client_masks,
                   X_test, y_test, apply_dp=False, label='No-DP-FL')
r_nodp, _ = best_eval(m_nodp, X_test, y_test)
r_nodp['DP_Protected'] = 'No'
r_nodp['Federated'] = 'Yes'

dp_f1 = round(r_nodp['F1'] - r_fed['F1'], 4)
dp_auc = round(r_nodp['AUC'] - r_fed['AUC'], 4)
print(f"\n  No-DP FL     : F1={r_nodp['F1']:.4f}  AUC={r_nodp['AUC']:.4f}")
print(f"  SENTINEL-EGO : F1={r_fed['F1']:.4f}   AUC={r_fed['AUC']:.4f}  "
      f"eps={eps_fed:.4f}")
print(f"  DP Cost  dF1={dp_f1:+.4f}  dAUC={dp_auc:+.4f}  "
      f"{'OK' if abs(dp_f1) < 0.03 else 'WARNING: large gap'}")

# Build full E4 comparison table
e4 = {
    'Yuan 2018 (LSTM-CNN)':     {'F1': 'N/A', 'AUC': 0.9449, 'Recall': 'N/A',
                                  'FPR': 'N/A', 'DP_Protected': 'No',
                                  'Federated': 'No', 'Byzantine_Robust': 'No',
                                  'MIA_Validated': 'No'},
    'LAN (TIFS 2024)':          {'F1': 'N/A', 'AUC': 0.9478, 'Recall': 'N/A',
                                  'FPR': 'N/A', 'DP_Protected': 'No',
                                  'Federated': 'No', 'Byzantine_Robust': 'No',
                                  'MIA_Validated': 'No'},
    'Ye 2025 (DeepInsight-FL)': {'F1': 0.9972, 'AUC': 'N/A', 'Recall': 'N/A',
                                  'FPR': 'N/A', 'DP_Protected': 'No',
                                  'Federated': 'Yes', 'Byzantine_Robust': 'No',
                                  'MIA_Validated': 'No'},
    'Isolated-DP (no FL)':      r_iso,
    'No-DP FL (no privacy)':    r_nodp,
    'Centralized-GBT (no DP)':  r_gbt,
    'SENTINEL-EGO (Ours)':      r_fed,
}
df_e4 = pd.DataFrame(e4).T
df_e4.to_csv("results_e4_comparison_FINAL.csv")
print("\nE4 Final Table:")
print(df_e4[['F1', 'AUC', 'Recall', 'FPR', 'DP_Protected', 'Federated']]
      .to_string())
print("✓ Saved: results_e4_comparison_FINAL.csv")


# ════════════════════════════════════════════════════════════
# E7: Scenario-Level Breakdown
# Uses fed_m trained in E1 — no re-training needed
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("E7: Scenario-Level Breakdown")
print("=" * 60)

df_test_sc = test_df.copy().reset_index(drop=True)


def classify_scenario(row):
    """Assign each user-week record to an insider threat scenario."""
    rm  = float(row.get('rm_copies',   0) or 0)
    ext = float(row.get('ext_ratio',   0) or 0)
    ah  = float(row.get('ah_ratio',    0) or 0)
    rs  = float(row.get('risky_count', 0) or 0)
    if   rm  > 0:                return 'S1: USB Exfiltration'
    elif ext > 0.5:              return 'S2: Email Exfiltration'
    elif ah  > 0.5:              return 'S3: After-Hours Access'
    elif rs  > 0 and ext > 0.3:  return 'S4: Risky Web+Email'
    else:                        return 'S5: General Accumulation'


df_test_sc['scenario'] = df_test_sc.apply(classify_scenario, axis=1)
print(f"  Distribution:\n{df_test_sc['scenario'].value_counts().to_string()}\n")

sc_results = {}
for sc, grp in df_test_sc.groupby('scenario'):
    idx = grp.index.values
    Xs = X_test[idx]
    ys = y_test[idx]
    if ys.sum() == 0 or len(np.unique(ys)) < 2:
        print(f"  {sc}: no positives — skip")
        continue
    r_sc, _ = best_eval(fed_m, Xs, ys)
    r_sc['n_total']     = len(ys)
    r_sc['n_malicious'] = int(ys.sum())
    sc_results[sc]      = r_sc
    print(f"  {sc:35s}: F1={r_sc['F1']:.4f}  AUC={r_sc['AUC']:.4f}  "
          f"n_pos={int(ys.sum())}/{len(ys)}")

pd.DataFrame(sc_results).T.to_csv("results_e7_scenario_breakdown.csv")
print("✓ Saved: results_e7_scenario_breakdown.csv")


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY — all r4.2 numbers
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ALL r4.2 RESULTS — COPY FOR PAPER")
print("=" * 60)

rows = {
    'Isolated-DP':     r_iso,
    'SENTINEL-EGO':    r_fed,
    'Centralized-GBT': r_gbt,
    'No-DP FL':        r_nodp,
    'Abl Legacy-Only': abl_results.get('Legacy-Only', {}),
    'Abl +PBI':        abl_results.get('+PBI', {}),
    'Abl +PBI+AIF':    abl_results.get('+PBI+AIF', {}),
    'Abl Full':        abl_results.get('Full SENTINEL-EGO', {}),
}
for k, v in rows.items():
    if isinstance(v, dict) and 'F1' in v:
        try:
            print(f"  {k:22s}: F1={float(v['F1']):.4f}  AUC={float(v['AUC']):.4f}  "
                  f"Recall={float(v['Recall']):.4f}  FPR={float(v['FPR']):.4f}")
        except Exception:
            print(f"  {k}: {v}")

print(f"\n  DP budget : eps={eps_fed:.4f}  delta=1e-05")
print(f"  DP cost   : dF1={dp_f1:+.4f}  dAUC={dp_auc:+.4f}")
print("\n  r4.2 COMPLETE. All CSVs saved.")
