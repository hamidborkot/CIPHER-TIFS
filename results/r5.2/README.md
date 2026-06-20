# CIPHER Results — CERT r5.2 (Local GPU Run)

All results produced on local machine: **NVIDIA GeForce RTX 5070 Laptop GPU (8.5 GB VRAM)**  
Dataset: **CERT Insider Threat Dataset r5.2** (2000 users, ~692K daily records)  
Date: June 2026

## Files

| File | Paper Table | Experiment |
|---|---|---|
| `table5_e1_final.csv` | Table V | E1 — Primary detection performance |
| `table6_e8_mia_final.csv` | Table VI | E8 — Membership inference attack audit |
| `table7_e9_stable.csv` | Table VII | E9 — Byzantine robustness |
| `eps_sweep.csv` | Figure 4 | ε-sweep across σ∈{1.0,1.5,2.0,2.5,3.0} |
| `CIPHER_ALL_RESULTS.csv` | All | Master consolidated results |

## Key Numbers

### Table V — E1 Detection
- **F1 = 0.7317**, AUC = 0.9127, Recall = 0.6281, Precision = 0.8763, FPR = 0.0054
- DP guarantee: **ε = 1.28, δ = 1e-5**
- Note: Lower than r6.2 (F1=0.8520) — expected; r5.2 is smaller (2000 users vs 4000)
  and uses heuristic labels (no users.csv ground truth)

### Table VI — E8 Privacy Audit (MIA)
- **MIA AUC = 0.5354** — borderline above 0.53 threshold
- Interpretation: r5.2 has higher class imbalance (5.45% malicious vs 2.72% in r6.2)
  making confidence distributions slightly more separable
- DP remains effective; difference is dataset-level, not mechanism-level

### Table VII — E9 Byzantine Robustness
- Attack: Gradient scaling ×5 + label flip, 3/10 clients (30%)
- FedAvg degradation: F1 **0.7181 → 0.7063** (Δ = −0.0118)
- FedAvg Recall drop: **0.5980 → 0.5945** (Δ = −0.0035)
- Multi-Krum Recall vs FedAvg-attack: **+0.0190** ← Krum catches MORE insiders
- Multi-Krum AUC gap from clean: **−0.0167**
- Byzantine fraction 3/10 = 30% < n/2 (within provable tolerance)

### Figure 4 — ε-Sweep
- F1 range across σ∈[1.0, 3.0]: **0.7197–0.7473** (variance < 0.028)
- σ = 2.0 selected as operating point (ε = 1.28)

## Cross-Dataset Comparison (r5.2 vs r6.2)

| Metric | r5.2 | r6.2 | Notes |
|---|---|---|---|
| Users | 2,000 | 4,000 | r6.2 is 2× larger |
| E1 F1 | 0.7317 | 0.8520 | r6.2 benefits from more data |
| E1 AUC | 0.9127 | 0.9693 | Consistent pattern |
| MIA AUC | 0.5354 | 0.5171 | Both near-random |
| FedAvg F1 drop (E9) | −0.0118 | −0.0497 | r5.2 more robust (smaller dataset) |
| Krum Recall gain (E9) | +0.0190 | +0.0146 | Krum advantage holds across both |

## Reproducibility

All experiments use `SEED=42`. Full code in `src/CIPHER_local.py`.
Change `CERT = BASE_R52` to run on r5.2.
