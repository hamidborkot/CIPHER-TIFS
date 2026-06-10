# SENTINEL-EGO: Privacy-Preserving Federated Insider Threat Detection

**IEEE Transactions on Information Forensics and Security (TIFS) — Under Review**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Dataset: CERT r4.2](https://img.shields.io/badge/Dataset-CERT%20r4.2%2Fr5.2%2Fr6.2-green)](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099)
[![DP: ε=1.28](https://img.shields.io/badge/Privacy-ε%3D1.28%2C%20δ%3D1e--5-red)](theory/theorem3_dp_guarantee.md)

---

## What This Paper Proves

SENTINEL-EGO is the **first insider threat detection system to simultaneously satisfy three formal security properties**:

| Property | Guarantee | Evidence |
|----------|-----------|----------|
| **(ε, δ)-Differential Privacy** | ε=1.2830, δ=1e-5 via Rényi DP composition | [Theorem 3](theory/theorem3_dp_guarantee.md) + [MIA Audit](results/results_e8_mia.csv) |
| **Byzantine Robustness** | F1 drop <5pp under 30% poisoning | [E9](results/results_e9_byzantine.csv) |
| **MIA-Validated Privacy** | MIA AUC ≤ 0.535 across all datasets | [E8](results/results_e8_mia.csv) |

No prior work satisfies all three simultaneously. Ye 2025 (F1=0.9972) operates without any privacy protection and is **not comparable** under the same security constraints — see [comparison table](results/results_e4_comparison.csv).

---

## Novel Contributions

1. **Psycho-Behavioral Indexing (PBI):** Sliding-window Jensen-Shannon drift detector with provable minimum detectable deviation δ_min ≥ 0.93 (Theorem 1). Improves F1 by +8–15pp over legacy features on slow-burn reconnaissance attacks.

2. **Archetype-Indexed Federated Learning (AIF):** Behavioral archetype stratification for FL partitioning, provably tightening convergence vs. random partitioning (Theorem 2). +1.59pp F1 from stratification alone.

3. **Formal DP Guarantee with Empirical MIA Validation:** Rényi DP composition proven at ε=1.28; independently validated by Shokri et al. membership inference attack achieving near-random AUC ≤ 0.535 (Theorem 3).

---

## Results Summary

### E1: Primary Detection (CERT r4.2 — Primary Dataset)

| Method | F1 | AUC | Precision | Recall | ε | FPR |
|--------|-----|-----|-----------|--------|---|-----|
| SENTINEL-EGO (FL+DP) | **0.8571** | **0.9842** | 0.8228 | 0.8943 | 1.28 | 0.54% |
| Centralized-GBT (no-DP) | 0.9876 | 0.9981 | 0.9812 | 0.9942 | — | 0.06% |

*CERT r6.2 (4000 users): F1=0.8520, AUC=0.9693 — confirms scalability*
*CERT r5.2 (700 users): F1=0.7317, AUC=0.9127 — expected degradation at small scale*

### E2: Ablation Study (Fixed — Legacy-Only excludes USB features)

| Variant | F1 | AUC | Δ F1 vs Legacy |
|---------|----|-----|----------------|
| Legacy-Only (no USB) | 0.6821 | 0.8934 | — |
| +PBI | 0.7654 | 0.9312 | **+8.3pp** |
| +PBI+AIF | 0.8219 | 0.9687 | **+14.0pp** |
| Full SENTINEL-EGO | **0.8571** | **0.9842** | **+17.5pp** |

*The fixed ablation properly excludes label-correlated USB features from Legacy-Only, giving PBI/AIF a fair baseline.*

### E3: Privacy–Utility Trade-off (ε sweep)

| σ | ε | F1 | MIA AUC |
|---|---|-----|----------|
| 0.5 | 4.92 | 0.8814 | 0.5421 |
| 1.0 | 1.91 | 0.8712 | 0.5283 |
| **1.28** | **1.28** | **0.8571** | **0.5171** |
| 2.0 | 0.78 | 0.8341 | 0.5092 |
| 4.0 | 0.39 | 0.7823 | 0.5043 |
| 8.0 | 0.20 | 0.6941 | 0.5011 |

### E4: SOTA Comparison

| Method | F1 | AUC | DP | FL | Byzantine | MIA |
|--------|-----|-----|-----|-----|-----------|-----|
| Yuan 2018 | 0.7812 | 0.9449 | ❌ | ❌ | ❌ | ❌ |
| LAN TIFS 2024 | 0.8103 | 0.9478 | ❌ | ❌ | ❌ | ❌ |
| FedAT 2024 | 0.6795 | — | ❌ | ✅ | ❌ | ❌ |
| Ye 2025* | 0.9972 | 0.9989 | ❌ | ✅ | ❌ | ❌ |
| **SENTINEL-EGO** | **0.8571** | **0.9842** | **✅ ε=1.28** | **✅** | **✅** | **✅** |

*Ye 2025 uses 1,098 features, 100 FL rounds, and zero privacy protection — not comparable under identical security constraints. Among all DP-FL methods, SENTINEL-EGO is the only entry.*

### E5: Cross-Scenario Generalization (replaces D2/D3)

| Scenario | Threat Type | F1 | PBI Δ vs Legacy | Key Signal |
|----------|------------|-----|-----------------|------------|
| S1: Pre-resignation theft | Exfiltration | 0.9124 | +12.9pp | Temporal spike |
| S2: Slow-burn recon | Reconnaissance | 0.6891 | **+13.8pp** | Cumulative PBI drift |
| S3: IT sabotage | Sabotage | 0.8742 | +15.2pp | After-hours + AIF |
| S4: Privilege escalation | Escalation | 0.8103 | +11.6pp | LDAP + archetype |
| S5: Policy violation | General | 0.8831 | +7.3pp | High volume baseline |

*S2 shows the largest PBI contribution — the slow-burn reconnaissance scenario is precisely where legacy features fail and PBI's drift detection provides unique value.*

### E8: Membership Inference Audit

| Dataset | MIA AUC | Status | Notes |
|---------|---------|--------|-------|
| CERT r4.2 | 0.5024 | ✅ PASS | Near-random |
| CERT r6.2 | 0.5171 | ✅ PASS | Near-random |
| CERT r5.2 | 0.5352 | ✅ PASS | Higher class imbalance; still < 0.55 |
| No-DP baseline | 0.7834 | ❌ FAIL | Confirms DP is necessary |

### E9: Byzantine Robustness

| Attack | Poison Rate | F1 | F1 Drop | Status |
|--------|-------------|-----|---------|--------|
| Gradient Scaling (5×) | 10% | 0.8439 | −1.3pp | ✅ PASS |
| Gradient Scaling (5×) | 20% | 0.8287 | −2.8pp | ✅ PASS |
| Gradient Scaling (5×) | 30% | 0.8091 | −4.8pp | ✅ PASS |
| Label Flipping | 20% | 0.8204 | −3.7pp | ✅ PASS |

---

## Repository Structure

```
SENTINEL-EGO-TIFS/
├── README.md                          ← This file
├── CITATION.cff
├── requirements.txt
├── src/
│   ├── sentinel_ego_local.py          ← Main system implementation
│   ├── federated.py                   ← FL aggregation + DP-SGD
│   ├── features.py                    ← Feature engineering (PBI, AIF)
│   ├── evaluate.py                    ← Metrics + MIA evaluation
│   ├── labeling.py                    ← CERT dataset labeling
│   └── model.py                       ← GBT model wrapper
├── experiments/
│   ├── run_e2_ablation_fixed.py       ← FIXED ablation (USB features excluded)
│   └── run_e5_scenario_breakdown.py  ← CERT scenario-level evaluation
├── theory/
│   ├── theorem1_pbi_detectability.md  ← Pinsker-based PBI bound
│   ├── theorem2_convergence.md        ← Archetype FL convergence
│   └── theorem3_dp_guarantee.md       ← Rényi DP → (ε,δ)-DP proof
├── results/
│   ├── results_e1.csv                 ← E1: Primary detection
│   ├── results_e2_ablation.csv        ← E2: Ablation (FIXED)
│   ├── results_e3_privacy.csv         ← E3: ε sweep
│   ├── results_e4_comparison.csv      ← E4: SOTA comparison (with DP/FL/Byzantine cols)
│   ├── results_e5_scenario_breakdown.csv ← E5: Per-scenario F1
│   ├── results_e8_mia.csv             ← E8: MIA audit
│   ├── results_e9_byzantine.csv       ← E9: Byzantine robustness
│   ├── results_cross_env_final.csv    ← Cross-version generalization
│   ├── results_master.csv             ← All-in-one summary
│   ├── r4.2/                          ← Kaggle r4.2 raw outputs
│   ├── r5.2/                          ← Local GPU r5.2 raw outputs
│   └── r6.2/                          ← Local GPU r6.2 raw outputs
├── figures/                           ← Paper figures (Figure 1–5)
├── config/                            ← Experiment configuration YAML
└── data/                              ← Data README only (dataset not redistributed)
```

---

## Reproducibility

### Requirements
```bash
pip install -r requirements.txt
```

### CERT Dataset
Request access at: https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099  
Place under `data/CERT/r4.2/`, `data/CERT/r5.2/`, `data/CERT/r6.2/`

### Run Primary Experiment (E1)
```bash
python src/sentinel_ego_local.py --dataset r4.2 --epsilon 1.28 --sigma 1.28 --rounds 10
```

### Run Fixed Ablation (E2)
```bash
python experiments/run_e2_ablation_fixed.py
```

### Run Scenario Breakdown (E5)
```bash
python experiments/run_e5_scenario_breakdown.py
```

---

## Theoretical Contributions

See [`theory/`](theory/) for full proofs.

- **Theorem 1** ([PBI Detectability](theory/theorem1_pbi_detectability.md)): PBI fires whenever behavioral TV distance ≥ 0.93, using Pinsker's inequality for JS divergence.
- **Theorem 2** ([Convergence](theory/theorem2_convergence.md)): Archetype stratification reduces inter-client gradient divergence Δ, tightening the FedProx convergence bound.
- **Theorem 3** ([DP Guarantee](theory/theorem3_dp_guarantee.md)): Full Rényi DP composition proof for σ=1.28, T=10 rounds, yielding ε=1.2830 at δ=1e-5.

---

## Citation

```bibtex
@article{borkot2025sentinel,
  title   = {SENTINEL-EGO: Privacy-Preserving Federated Insider Threat Detection
             with Psycho-Behavioral Indexing and Formal Differential Privacy},
  author  = {Borkot, Hamid},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
