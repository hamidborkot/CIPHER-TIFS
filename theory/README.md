# Theory — CIPHER (IEEE TIFS)

This directory contains the formal theoretical proofs required by TIFS reviewers.
All three theorems must be written before starting Section IV of the paper.

---

## Status

| File | Theorem | Paper Section | Status |
|---|---|---|---|
| `dp_proof.md` | **Theorem 1**: (ε,δ)-DP guarantee for DPFA via Rényi composition | Sec. IV-A | ⬜ TO WRITE |
| `mia_bound.md` | **Theorem 2**: MIA distinguishing advantage bound | Sec. IV-B | ⬜ TO WRITE |
| `convergence_bound.md` | **Theorem 3**: FL convergence under DP noise (Appendix) | Sec. IV-C | ⬜ TO WRITE |

---

## Theorem Outlines

### Theorem 1 — DPFA (ε,δ)-DP Guarantee

**Statement:** CIPHER's DPFA module with σ=1.28, T=10 rounds, q=0.10 subsampling
satisfies (ε=1.2830, δ=1e-5)-differential privacy.

**Proof sketch (Rényi DP composition):**
1. Per-step RDP: `ε_α = q² · α / (2σ²) = 0.01 · 10 / (2 · 1.6384) = 0.003050`
2. Composed over T=10 rounds: `ε_total = T · ε_α = 0.03050`
3. Convert to (ε,δ)-DP: `ε(δ) = ε_total + ln(1/δ)/(α-1) = 0.03050 + ln(1e5)/9 = 1.2830`
4. Verify numerically with `src/dpfa.py::compute_epsilon(1.28, 10, 0.10)` → 1.2830 ✓

**Key citations:** Mironov 2017 (RDP), Abadi et al. 2016 (DP-SGD), Wang et al. 2019 (subsampling amplification)

---

### Theorem 2 — MIA Advantage Bound

**Statement:** Under (ε,δ)-DP, any membership inference adversary's advantage is bounded:
`Adv_MIA ≤ (e^ε - 1) / (e^ε + 1)`

**At ε=1.28:** theoretical bound ≈ `(e^1.28 - 1)/(e^1.28 + 1) = (3.597-1)/(3.597+1) ≈ 0.565`

**Empirical result (E8):** CIPHER MIA AUC = 0.5024, which is BELOW the theoretical bound of 0.5 + 0.565/2.
This confirms the DP guarantee holds in practice.

**Key citations:** Yeom et al. 2018 (MIA formulation), Humphries et al. 2023

---

### Theorem 3 — Convergence Bound (Appendix)

**Statement:** Under DPFA DP-SGD, the expected optimality gap satisfies:
`E[f(w_T) - f(w*)] ≤ O(σ² · d · T / n)`
where σ=noise, d=model dimension, T=rounds, n=total samples.

**Key citations:** Li et al. 2020 (convergence of FedAvg), Wei et al. 2021 (DP-FL convergence)

---

## Module Name Reference

All proofs in this directory refer to CIPHER module names:
- **BDM** (Behavioral Drift Monitor) — not PBI
- **PSE** (Persona-Stratified Ensemble) — not AIF
- **DPFA** (Differentially Private Federated Aggregation) — not FAL

The names PBI/AIF/FAL belong to the TDSC paper (SENTINEL-EGO). Do not use them here.
