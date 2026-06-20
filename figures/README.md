# Figures

All figures are generated from real experimental data in `results/`. Do not edit manually — regenerate using `scripts/generate_figures.py`.

| File | Paper Figure | Experiment | Caption |
|---|---|---|---|
| `fig1_convergence.png` | Fig. 1 | E1 | FL convergence: SENTINEL-EGO vs DP-Isolated over 10 rounds, CERT r4.2 |
| `fig2_epsilon_utility.png` | Fig. 2 | E3 | Privacy–utility tradeoff: F1 and MIA AUC vs privacy budget ε |
| `fig3_ablation.png` | Fig. 3 | E2 | Ablation study: F1 and AUC per system variant (Legacy-Only → Full SENTINEL-EGO) |
| `fig4_sota_comparison.png` | Fig. 4 | E4 | SOTA comparison: SENTINEL-EGO vs 6 baselines (F1 and AUC) |

## Key Visual Takeaways
- **Fig 1**: FL converges in 10 rounds; gap to DP-Isolated closes from 0.23 → 0.02 F1
- **Fig 2**: Operating point ε=1.28 sits at the knee of the curve — best F1/privacy balance
- **Fig 3**: Full system AUC = 0.9655, +0.0093 over Legacy-Only; PBI+AIF contribution visible
- **Fig 4**: SENTINEL-EGO is the **only** system with DP + FL + Byzantine robustness + MIA audit
