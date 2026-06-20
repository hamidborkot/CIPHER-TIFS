# SENTINEL-EGO — Writing Checklist

Use this file to track writing progress. Check off each item as you complete it.

---

## Paper Structure (IEEE TIFS Format)

### Abstract (250 words max)
- [ ] Problem statement (1–2 sentences)
- [ ] Gap in existing work (1 sentence)
- [ ] Proposed solution: SENTINEL-EGO (2–3 sentences)
- [ ] Key results: F1=0.8531, AUC=0.9601, ε=1.28, MIA AUC≈0.51 (1–2 sentences)
- [ ] Significance statement (1 sentence)

### Section I — Introduction
- [ ] Insider threat problem motivation
- [ ] Why centralized approaches fail (privacy, deployment)
- [ ] FL + DP as the solution space
- [ ] Gap: no existing system has DP + FL + Byzantine + MIA audit
- [ ] Contributions list (4–5 bullet points)
- [ ] Paper organization paragraph

### Section II — Related Work
- [ ] Insider threat detection (Yuan 2018, LAN 2024, Ye 2025)
- [ ] Federated learning for security
- [ ] Differential privacy in FL
- [ ] Byzantine robustness in FL
- [ ] Positioning table (Table I or II)

### Section III — System Design (SENTINEL-EGO)
- [ ] III-A: Problem formulation
- [ ] III-B: Progressive Behavioral Indexing (PBI) — KL-divergence formulation
- [ ] III-C: Archetype-Informed Federated Learning (AIF)
- [ ] III-D: DP-SGD integration + privacy accounting
- [ ] III-E: Byzantine-robust aggregation
- [ ] Architecture figure (system diagram)

### Section IV — Privacy Analysis
- [ ] IV-A: (ε,δ)-DP guarantee proof
- [ ] IV-B: MIA resistance analysis
- [ ] IV-C: Convergence bound under DP noise

### Section V — Experiments
- [ ] V-A: Datasets (CERT r4.2 primary, r6.2 and r5.2 secondary)
- [ ] V-B: Baselines and metrics
- [ ] V-C: E1 Primary Detection → **Table V** (use `results_e1.csv`)
- [ ] V-D: E2 Ablation → **Table VI** (use `results_e2_ablation_FIXED.csv`)
- [ ] V-E: E3 Privacy–Utility Tradeoff → **Fig. 2** (use `results_e3_privacy.csv`)
- [ ] V-F: E4 SOTA Comparison → **Table VII** (use `results_e4_comparison_FINAL.csv`)
- [ ] V-G: E7 Scenario Breakdown → **Table VIII** (use `results_e7_scenario_breakdown.csv`)
- [ ] V-H: E8 MIA Audit → **Table IX** (use `results_e8_mia.csv`)
- [ ] V-I: E9 Byzantine Robustness → **Table X** (use `results_e9_byzantine.csv`)

### Section VI — Discussion
- [ ] Why F1 differs between r4.2 and r6.2/r5.2 (PBI implementation difference)
- [ ] DP cost analysis: F1 cost = 0.0586 vs No-DP FL
- [ ] Ye 2025 comparison: they have higher F1 but NO DP protection
- [ ] Limitations

### Section VII — Conclusion
- [ ] Summary (3–4 sentences)
- [ ] Future work

---

## Tables Checklist

| Table | Content | Source File | Status |
|---|---|---|---|
| Table I | Notation summary | Manual | ☐ |
| Table II | Related work comparison | Manual + E4 | ☐ |
| Table III | Dataset statistics | Manual | ☐ |
| Table IV | Hyperparameters | `config/` | ☐ |
| Table V | E1 Primary Detection | `results_e1.csv` | ☐ |
| Table VI | E2 Ablation | `results_e2_ablation_FIXED.csv` | ☐ |
| Table VII | E4 SOTA Comparison | `results_e4_comparison_FINAL.csv` | ☐ |
| Table VIII | E7 Scenario Breakdown | `results_e7_scenario_breakdown.csv` | ☐ |
| Table IX | E8 MIA Audit | `results_e8_mia.csv` | ☐ |
| Table X | E9 Byzantine | `results_e9_byzantine.csv` | ☐ |

## Figures Checklist

| Figure | Content | File | Status |
|---|---|---|---|
| Fig. 1 | System architecture diagram | To be created | ☐ |
| Fig. 2 | FL Convergence curves | `figures/fig1_convergence.png` | ✅ Ready |
| Fig. 3 | Privacy–Utility tradeoff | `figures/fig2_epsilon_utility.png` | ✅ Ready |
| Fig. 4 | Ablation bar chart | `figures/fig3_ablation.png` | ✅ Ready |
| Fig. 5 | SOTA comparison | `figures/fig4_sota_comparison.png` | ✅ Ready |

---

## Critical Numbers to Use

| Metric | Value | Source |
|---|---|---|
| F1 (primary) | **0.8531** | `results_e1.csv` |
| AUC (primary) | **0.9601** | `results_e1.csv` |
| Privacy budget | **ε=1.2830, δ=1e-5** | `results_e3_privacy.csv` |
| MIA AUC | **0.5171** (r4.2), 0.5048 (r6.2), 0.4996 (r5.2) | `results_e8_mia.csv` |
| DP cost vs No-DP FL | **ΔF1 = 0.0586** | `results_e4_comparison_FINAL.csv` |
| Byzantine resilience | **F1 drop <5pp at 30% poison** | `results_e9_byzantine.csv` |
| Ablation AUC gain | **+0.0093** (Full vs Legacy) | `results_e2_ablation_FIXED.csv` |

---

## Before Submission
- [ ] Notebook uploaded to `notebooks/sentinel_ego_r42_kaggle.ipynb`
- [ ] System architecture figure created (Fig. 1)
- [ ] All LaTeX tables match CSV numbers exactly
- [ ] `CITATION.cff` updated with final title and DOI
- [ ] `theory/` proofs written
- [ ] Anonymous version prepared (remove names from notebooks/comments)
