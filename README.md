# SENTINEL-EGO: Privacy-Preserving Federated Insider Threat Detection

> **IEEE Transactions on Information Forensics and Security (TIFS) — Submission**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset: CERT r4.2](https://img.shields.io/badge/Dataset-CERT%20r4.2-lightgrey)](https://resources.sei.cmu.edu/tools/downloads/insider-threat/)

---

## Overview

**SENTINEL-EGO** is a federated learning framework for insider threat detection that jointly provides:

- ✅ **High detection performance** — AUC = 0.9842, F1 = 0.8571 on CERT r4.2
- ✅ **Formal differential privacy** — ε = 1.28, δ = 1e-5 (Rényi DP, σ = 2.0)
- ✅ **Cross-environment generalization** — evaluated on 3 independent datasets
- ✅ **User-stratified federation** — each client holds a balanced mix of malicious/benign users

No prior federated insider threat detection system simultaneously achieves **AUC > 0.98 with ε < 2** across multiple deployment environments.

---

## Architecture

```
SENTINEL-EGO
├── ThreatNet          MLP classifier (28 features → 256 → 128 → 64 → 1)
├── FedAvg             Weighted federated averaging across N=10 clients
├── Rényi-DP           Per-step gradient clipping + Gaussian noise (σ=2.0)
├── PBI                Peer Behavioral Index — drift detection vs. cohort
├── AIF                Anomaly Integration Fusion — combines PBI + raw signals
└── Archetypes         K-means behavioral clustering (k=10, silhouette=0.088)
```

---

## Results Summary

### E1 — Primary (CERT r4.2)

| Method | F1 | AUC | Recall | FPR | Privacy |
|---|---|---|---|---|---|
| Isolated-DP | 0.8753 | 0.9814 | 0.8808 | 0.0037 | ε=1.28 |
| **SENTINEL-EGO** | **0.8571** | **0.9842** | **0.8943** | 0.0054 | ε=1.28 |
| Centralized GBT | 1.0000 | 1.0000 | 1.0000 | 0.0000 | None |

> GBT perfect score is a dataset artifact — labels derived from removable media threshold GBT memorizes trivially.

### TABLE V — Cross-Environment Generalization

| Dataset | Method | F1 | AUC | FPR | ε |
|---|---|---|---|---|---|
| CERT r4.2 (D1) | SENTINEL-EGO | 0.8571 | 0.9842 | 0.0054 | 1.28 |
| Corporate (D2) | SENTINEL-EGO | 0.6033 | 0.8569 | 0.0093 | 1.28 |
| Classified (D3) | SENTINEL-EGO | 0.5381 | 0.9001 | 0.0090 | 1.28 |

### E3 — Privacy-Utility Tradeoff

| σ | ε | F1 | AUC |
|---|---|---|---|
| 0.5 | 1.3392 | 0.8838 | 0.9983 |
| 1.0 | 1.2942 | 0.8633 | 0.9936 |
| 2.0 | 1.2830 | 0.8702 | 0.9813 |
| 4.0 | 1.2802 | 0.8402 | 0.9789 |
| 8.0 | 1.2794 | 0.8499 | 0.9837 |

---

## Repository Structure

```
SENTINEL-EGO-TIFS/
├── notebooks/
│   ├── 01_labeling.ipynb          # Label derivation from CERT r4.2
│   ├── 02_features.ipynb          # PBI, AIF, archetype feature engineering
│   ├── 03_experiments_E1_E5.ipynb # E1–E5: primary + ablation + privacy
│   ├── 04_cross_env_E6_E7.ipynb   # E6–E7: D2 corporate, D3 classified
│   └── 05_figures.ipynb           # All publication figures
├── src/
│   ├── model.py                   # ThreatNet architecture
│   ├── federated.py               # FedAvg, local training, DP
│   ├── features.py                # PBI, AIF, archetype derivation
│   ├── labeling.py                # Malicious user label pipeline
│   └── evaluate.py                # Threshold search, metrics
├── results/
│   ├── results_e1.csv
│   ├── results_e2_ablation.csv
│   ├── results_e3_privacy.csv
│   ├── results_e4_comparison.csv
│   ├── results_convergence.csv
│   └── results_cross_env_final.csv
├── figures/                       # Publication-ready figures (PNG)
├── config/
│   └── config.yaml                # All hyperparameters
├── data/
│   └── README.md                  # Dataset download instructions
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Datasets

| Dataset | Source | Rows | Malicious Rate |
|---|---|---|---|
| CERT r4.2 (D1) | [CMU SEI](https://resources.sei.cmu.edu/tools/downloads/insider-threat/) | 1,394,010 | 2.72% |
| Corporate (D2) | [Kaggle](https://www.kaggle.com/datasets/ahmeduzaki/insider-threat-dataset-for-corporate-environments) | 118,614 | 5.38% |
| Classified (D3) | [Kaggle](https://www.kaggle.com/datasets/efchbd1013/insider-threat-dataset-for-classified-environments) | 299,880 | 2.51% |

Dataset files are **not included** in this repository. See `data/README.md` for download and placement instructions.

---

## Installation

```bash
git clone https://github.com/hamidborkot/SENTINEL-EGO-TIFS.git
cd SENTINEL-EGO-TIFS
pip install -r requirements.txt
```

---

## Reproducing Results

Run notebooks in order:

```bash
jupyter notebook notebooks/01_labeling.ipynb
jupyter notebook notebooks/03_experiments_E1_E5.ipynb
jupyter notebook notebooks/04_cross_env_E6_E7.ipynb
jupyter notebook notebooks/05_figures.ipynb
```

Or run on Kaggle — all experiments were developed and validated on Kaggle GPU (P100).

---

## Hyperparameters

| Parameter | Value |
|---|---|
| FL Rounds | 10 |
| Local Epochs | 3 |
| Clients (N) | 10 |
| Learning Rate | 0.001 |
| DP Noise σ | 2.0 |
| DP Clip Norm | 1.0 |
| DP δ | 1e-5 |
| Sampling Rate q | 0.01 |
| Batch Size | 512 |
| Hidden Dims | [256, 128, 64] |
| Seed | 42 |

---

## Citation

If you use this code, please cite:

```bibtex
@article{borkot2025sentinenego,
  title     = {SENTINEL-EGO: Privacy-Preserving Federated Insider Threat Detection
               with Differential Privacy and Behavioral Archetype Profiling},
  author    = {Borkot, Hamid},
  journal   = {IEEE Transactions on Information Forensics and Security},
  year      = {2025},
  note      = {Under review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
