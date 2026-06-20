# Results Key — What to Use for Writing

> **One rule**: always use the `_FINAL` or `_FIXED` version when two versions exist.

## Primary Dataset: CERT r4.2 (Kaggle)
This is your headline dataset. All Table numbers in the paper come from here.

| Experiment | File to Use | Key Number |
|---|---|---|
| E1 Primary Detection | `results_e1.csv` | **F1=0.8531, AUC=0.9601**, ε=1.28 |
| E2 Ablation | `results_e2_ablation_FIXED.csv` ✅ | Legacy→Full: **0.8438→0.8470 F1**, AUC +0.0093 |
| E3 ε-Sweep | `results_e3_privacy.csv` | Operating point σ=1.28 → **F1=0.8571, MIA=0.5171** |
| E4 SOTA Comparison | `results_e4_comparison_FINAL.csv` ✅ | SENTINEL-EGO only system with DP+FL+Byz+MIA |
| E5/E7 Scenario | `results_e7_scenario_breakdown.csv` | S1 (USB) best F1; S3 (after-hours) hardest |
| E8 MIA Audit | `results_e8_mia.csv` | **MIA AUC ≈ 0.51–0.52** across all datasets |
| E9 Byzantine | `results_e9_byzantine.csv` | **F1 drop <5pp** even at 30% poison |
| Convergence | `results_convergence.csv` | Used for Fig. 1 |

## Secondary Datasets: CERT r6.2 and r5.2 (Local GPU)
Used for cross-dataset validation tables only — NOT for headline numbers.

| Dataset | Folder | E1 F1 | Notes |
|---|---|---|---|
| CERT r6.2 | `results/r6.2/` | 0.5594 | Different PBI implementation (basic z-score vs exp-weighted) — explained in paper |
| CERT r5.2 | `results/r5.2/` | 0.6844 | 2,000-user dataset — higher mal% (5.41%) |

## ⚠️ Do NOT use
- `results_e2_ablation.csv` (archived to `results/archive/`) — biased baseline
- `results_e2_ablation_OLD.csv` — same issue
- Any file in `results/archive/`
