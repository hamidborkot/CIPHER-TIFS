# Theoretical Foundations

This directory contains the three formal theorems that underpin SENTINEL-EGO's security and performance guarantees.

| File | Theorem | Claim |
|------|---------|-------|
| `theorem1_pbi_detectability.md` | Theorem 1 | PBI fires whenever behavioral deviation exceeds a provable minimum threshold (Pinsker inequality) |
| `theorem2_convergence.md` | Theorem 2 | Archetype-stratified FL converges faster than random partitioning (Li et al. 2020 extension) |
| `theorem3_dp_guarantee.md` | Theorem 3 | Full system satisfies (ε=1.28, δ=1e-5)-DP via Rényi DP composition + MIA empirical validation |

## How These Support the Paper

- **Section III** (System Model): cite Theorem 3 for DP guarantee
- **Section IV-A** (PBI): cite Theorem 1 for detectability bound  
- **Section IV-C** (Federated Aggregation): cite Theorem 2 for convergence claim
- **Theorems appear in Section V** (Formal Analysis) in the paper
