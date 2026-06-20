# CIPHER — Paper Figures

> Paper: CIPHER (IEEE TIFS Submission)  
> Repo: hamidborkot/CIPHER-TIFS

Regenerate all figures: `python scripts/generate_figures.py`

---

## Figure Inventory

| File | Paper Label | Content | Source CSV | Status |
|---|---|---|---|---|
| *(to be drawn)* | **Fig. 1** | System architecture diagram | Manual (draw.io / TikZ) | ⬜ TODO |
| `fig1_convergence.png` | **Fig. 2** | CIPHER FL convergence over 10 rounds | `results_convergence.csv` | ✅ |
| `fig2_epsilon_utility.png` | **Fig. 3** | ε-sweep: F1 vs MIA AUC tradeoff | `results_e3_privacy.csv` | ✅ |
| `fig3_ablation.png` | **Fig. 4** | Ablation: BDM + PSE contribution | `results_e2_ablation_FIXED.csv` | ✅ |
| `fig4_sota_comparison.png` | **Fig. 5** | CIPHER vs SOTA baselines | `results_e4_comparison_FINAL.csv` | ✅ |

---

## Figure Descriptions (for paper captions)

**Fig. 1 — System Architecture** *(to be drawn)*  
CIPHER pipeline: user behavioral logs → BDM (KL-divergence drift detection) → PSE (archetype-stratified ensemble) → DPFA (DP-SGD + Multi-Krum federated aggregation).

**Fig. 2 — FL Convergence**  
CIPHER detection F1 and AUC per federation round across CERT r4.2 and r6.2. Convergence achieved by round 6.

**Fig. 3 — Privacy-Utility Tradeoff**  
F1 score and MIA attacker AUC as ε varies. Operating point ε=1.28 sits at the knee: F1=0.8531 with MIA AUC suppressed to 0.5024 (near-random).

**Fig. 4 — Ablation Study**  
Incremental contribution of BDM (Behavioral Drift Monitor) and PSE (Persona-Stratified Ensemble). AUC improves +0.0093 from Legacy-Only to Full CIPHER.

**Fig. 5 — SOTA Comparison**  
CIPHER vs Yuan 2018, LAN-TIFS 2024, FedAT, Ye 2025, No-DP FL, Centralized-GBT. CIPHER is the only system with DP_Protected=Yes and MIA_Validated=Yes.

---

## Note on Ye 2025 (DeepInsight-FL)

Ye 2025 reports F1=0.9972 — higher than CIPHER (0.8531). This is expected and explained in Section VI:  
- Ye 2025 uses **no differential privacy** (DP_Protected=No)
- Ye 2025 has **no MIA validation** (MIA_Validated=No)
- F1 difference (0.0586 for No-DP FL) is the **measured cost of DP** in CIPHER
- CIPHER is solving a **different problem**: privacy-preserving detection, not maximum accuracy
