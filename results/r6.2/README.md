# SENTINEL-EGO Results — CERT r6.2 (Local GPU Run)

All results produced on local machine: **NVIDIA GeForce RTX 5070 Laptop GPU (8.5 GB VRAM)**  
Dataset: **CERT Insider Threat Dataset r6.2**  
Date: June 2026

## Files

| File | Paper Table | Experiment |
|---|---|---|
| `table5_e1_final.csv` | Table V | E1 — Primary detection performance |
| `table6_e8_mia_final.csv` | Table VI | E8 — Membership inference attack audit |
| `table7_e9_stable.csv` | Table VII | E9 — Byzantine robustness |
| `eps_sweep.csv` | Figure 4 | ε-sweep across σ∈{1.0,1.5,2.0,2.5,3.0} |
| `SENTINEL_EGO_ALL_RESULTS.csv` | All | Master consolidated results |

## Key Numbers

### Table V — E1 Detection
- **F1 = 0.8520**, AUC = 0.9693, Recall = 0.8659, FPR = 0.0046
- DP guarantee: **ε = 1.28, δ = 1e-5**

### Table VI — E8 Privacy Audit (MIA)
- **MIA AUC = 0.5171** — indistinguishable from random guessing (0.50)
- Confirms DP-SGD provides effective membership privacy

### Table VII — E9 Byzantine Robustness
- Attack: Gradient scaling ×5 + label flip, 3/10 clients (30%)
- FedAvg degradation: F1 **0.8561 → 0.8064** (Δ = −0.0497)
- FedAvg Recall drop: **0.8370 → 0.7712** (Δ = −0.0658)
- Multi-Krum Recall recovery: **+0.0146** over FedAvg-under-attack
- Multi-Krum AUC gap from clean: **−0.0106** only
- Byzantine fraction 3/10 = 30% < n/2 (within provable tolerance)

### Figure 4 — ε-Sweep
- F1 variance across σ∈[1.0, 3.0]: **< 0.005**
- σ = 2.0 selected as operating point (ε = 1.28)

## Reproducibility

All experiments use `SEED=42`. Full code in `src/sentinel_ego_local.py`.

```bash
pip install torch torchvision scikit-learn pandas numpy
python src/sentinel_ego_local.py --dataset r6.2
```
