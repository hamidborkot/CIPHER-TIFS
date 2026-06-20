# CIPHER Results Key

> **System:** CIPHER 
> **Modules:** BDM / PSE / DPFA  
> **Primary claim:** MIA-validated DP

This file is the single source of truth. Every table in the paper must trace back to a row in one of these CSVs.

---

## Paper Table → CSV Mapping

| Paper Table | Content | CSV File | Key Numbers |
|---|---|---|---|
| **Table V** | Primary Detection | `results_e1.csv` | **F1=0.8531, AUC=0.9601, ε=1.2830** |
| **Table VI** | Ablation (BDM+PSE) | `results_e2_ablation_FIXED.csv` ✅ | AUC: 0.9749→0.9842 (+0.0093) |
| **Table VII** | SOTA Comparison | `results_e4_comparison_FINAL.csv` ✅ | CIPHER only system with DP+MIA |
| **Table VIII** | Scenario Breakdown | `results_e7_scenario_breakdown.csv` | F1 per threat scenario |
| **Table IX** | **MIA Audit** ← HEADLINE | `results_e8_mia.csv` ✅ | **No-DP=0.7834, CIPHER=0.5024** |
| **Table X** | Byzantine Robustness | `results_e9_byzantine.csv` | F1 drop <5pp at 30% Byzantine |

---

## Figure → CSV Mapping

| Paper Figure | Content | CSV File | PNG |
|---|---|---|---|
| **Fig. 1** | System architecture | Manual (draw.io/TikZ) | ⬜ TO DRAW |
| **Fig. 2** | FL Convergence | `results_convergence.csv` | `figures/fig1_convergence.png` |
| **Fig. 3** | ε-sweep tradeoff | `results_e3_privacy.csv` | `figures/fig2_epsilon_utility.png` |
| **Fig. 4** | Ablation bar | `results_e2_ablation_FIXED.csv` | `figures/fig3_ablation.png` |
| **Fig. 5** | SOTA comparison | `results_e4_comparison_FINAL.csv` | `figures/fig4_sota_comparison.png` |

---

## Critical Numbers — Copy-Paste Into Paper

```
F1  (primary, CERT r4.2)        = 0.8531
AUC (primary, CERT r4.2)        = 0.9601
ε   (privacy budget)            = 1.2830
δ                               = 1e-5
MIA AUC — No-DP FL             = 0.7834  ← attacker succeeds
MIA AUC — CIPHER                = 0.5024  ← near-random after DP
Δ MIA AUC (privacy gain)        = 0.2810
DP cost (ΔF1 vs No-DP FL)       = −0.0586
Byzantine F1 drop at 30% poison = <5pp
Ablation AUC gain (Full vs Base)= +0.0093
```

---

## Which Ablation CSV to Use

⚠️ **USE:** `results_e2_ablation_FIXED.csv` — BDM/PSE features correctly separated  
❌ **DO NOT USE:** `results_e2_ablation.csv` (archived — USB signal leakage in Legacy-Only row)  
❌ **DO NOT USE:** `results/archive/results_e2_ablation_OLD.csv`

See `ABLATION_VERIFICATION.md` for full explanation.

---

## Cross-Dataset Validation

| Dataset | F1 | AUC | Source |
|---|---|---|---|
| CERT r4.2 (primary) | 0.8531 | 0.9601 | `results_e1.csv` |
| CERT r6.2 (local GPU, 4000 users) | 0.8520 | 0.9693 | `results/r6.2/` |
| CERT r5.2 (local GPU, 700 users) | 0.7317 | 0.9127 | `results/r5.2/` |

F1 is lower on r5.2 due to fewer malicious users (17 vs 70+). AUC remains strong.
This is discussed in Section VI (Discussion) of the paper.
