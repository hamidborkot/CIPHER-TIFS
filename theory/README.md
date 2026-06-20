# Theory

This directory will contain the formal theoretical components required by TIFS reviewers.
To be populated during the writing phase.

## Planned Contents

| File | Content | Paper Section |
|---|---|---|
| `dp_proof.md` | Formal (ε,δ)-DP proof for DP-SGD + FedAvg composition | Sec. IV-B |
| `convergence_bound.md` | FL convergence bound under DP noise (Theorem 1) | Sec. IV-C |
| `mia_bound.md` | MIA distinguishing advantage bound derivation | Sec. V-D |
| `pbi_kl_divergence.md` | KL-divergence formulation for PBI drift score | Sec. III-A |

## Key Theorems to Prove
1. **DP Composition**: Sequential composition of per-round DP-SGD gives (ε,δ)-DP for the full FL training
2. **Utility Bound**: E[F1_loss] ≤ O(σ²·T / n) where σ=noise, T=rounds, n=clients
3. **MIA Advantage**: Adv_MIA ≤ e^ε − 1 for any membership inference adversary
