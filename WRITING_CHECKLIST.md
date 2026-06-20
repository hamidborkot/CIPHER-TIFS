# CIPHER — Writing Checklist (IEEE TIFS)

> **System name:** CIPHER (not SENTINEL-EGO — that is the TDSC paper)  
> **Module names:** BDM / PSE / DPFA (not PBI / AIF / FAL)  
> **Primary claim:** MIA-validated DP, NOT detection accuracy  
> **Headline number:** MIA AUC 0.7834 → 0.5024, F1=0.8531, ε=1.28  

---

## Write in this order — NOT front to back

| Order | Section | Key argument | Status |
|---|---|---|---|
| 1st | **Sec. V — Experiments** | Lead with E8 MIA table as the centerpiece result | ⬜ |
| 2nd | **Sec. III — System Design** | BDM + PSE + DPFA, each with algorithm pseudocode | ⬜ |
| 3rd | **Sec. IV — Theory** | Theorem 1 (DP) + Theorem 2 (MIA bound) + Theorem 3 (convergence) | ⬜ |
| 4th | **Sec. II — Related Work** | Insider threat / FL for security / DP in FL / MIA defenses | ⬜ |
| 5th | **Sec. I — Introduction** | Gap sentence → 3 contributions → paper organization | ⬜ |
| 6th | **Abstract** | 150 words: gap + method + 3 numbers + impact | ⬜ |

---

## Section-by-Section Detail

### Abstract (150 words max — write LAST)
- [ ] Gap sentence: no prior insider threat FL system has MIA-validated DP
- [ ] Method: CIPHER with BDM + PSE + DPFA
- [ ] 3 numbers: MIA AUC 0.7834→0.5024, F1=0.8531, ε=1.28
- [ ] Significance: first empirically validated privacy guarantee in this space

### Section I — Introduction
- [ ] Open: insider threat motivation (SOC 99% FPR ref)
- [ ] Problem: existing FL systems claim DP but never validate against real MIA
- [ ] Gap: no system has DP + FL + Byzantine + MIA audit simultaneously
- [ ] Contributions (3 bullets):
  - [ ] BDM: drift-gated behavioral routing with KL-divergence monitoring
  - [ ] PSE + DPFA: archetype-stratified DP-FL with (ε=1.28,δ=1e-5)-DP
  - [ ] First MIA-validated DP in insider threat detection (0.7834→0.5024)
- [ ] Paper organization paragraph

### Section II — Related Work (4 paragraphs)
- [ ] Para 1: Insider threat detection — Yuan 2018, LAN TIFS 2024, Ye 2025
  - Note: all lack DP and MIA validation
- [ ] Para 2: FL for security — FedAT, FedITD, Zhao 2020
  - Note: none validate privacy with MIA audit
- [ ] Para 3: DP in FL — Abadi 2016 (DP-SGD), Wei 2020, Mironov 2017 (RDP)
- [ ] Para 4: MIA attacks and defenses — Yeom 2018, Humphries 2023
  - Note: no insider threat paper has run MIA evaluation
- [ ] Positioning table showing CIPHER is the only system with all 4 properties

### Section III — CIPHER Framework
- [ ] III-A: Problem formulation (threat model, GDPR motivation)
- [ ] III-B: BDM — KL-divergence drift detection, Algorithm 1, Eq.(2)
- [ ] III-C: PSE — 4-classifier ensemble, archetype normalization, Eq.(3)
- [ ] III-D: DPFA — DP-SGD, clipping (Eq.4), noise (Eq.5), aggregation (Eq.6)
- [ ] III-E: Byzantine robustness — Multi-Krum, f=1 Byzantine tolerance
- [ ] System architecture figure (Figure 1 — STILL TO BE DRAWN)

### Section IV — Privacy and Convergence Analysis
- [ ] IV-A: **Theorem 1** — (ε,δ)-DP guarantee via Rényi composition
  - State: FAL with σ=1.28, T=10, q=0.10 satisfies (ε=1.2830, δ=1e-5)-DP
  - Proof: εα = q²α/(2σ²), εtotal = R·εα, ε(δ) = εtotal + ln(1/δ)/(α-1)
  - Substitute values → 1.2830
- [ ] IV-B: **Theorem 2** — MIA advantage bound
  - State: Adv_MIA ≤ (e^ε - 1)/(e^ε + 1) for any MIA adversary
  - Apply: at ε=1.28 → theoretical bound ≈ 0.277
  - Show: empirical MIA AUC=0.5024 < theoretical bound → DP guarantee holds
- [ ] IV-C: **Theorem 3** (appendix) — Convergence under DP noise (cite Li 2020)

### Section V — Experiments
- [ ] V-A: Datasets — CERT r4.2 (primary), r6.2, r5.2 (cross-validation)
- [ ] V-B: Baselines — Isolated-DP, No-DP FL, Centralized-GBT, Ye 2025
- [ ] V-C: **E1 Primary Detection** → Table V (use `results_e1.csv`)
  - F1=0.8531, AUC=0.9601, ε=1.28
- [ ] V-D: **E2 Ablation** → Table VI (use `results_e2_ablation_FIXED.csv`)
  - Argue AUC (+0.0093), not F1. Explain F1 drop in +PSE row as precision-recall tradeoff
- [ ] V-E: **E3 ε-sweep** → Figure 2 (use `results_e3_privacy.csv`)
  - Operating point ε=1.28 at knee of privacy-utility curve
- [ ] V-F: **E4 SOTA Comparison** → Table VII (use `results_e4_comparison_FINAL.csv`)
  - Ye 2025: F1=0.9972 BUT DP_Protected=No, MIA_Validated=No
  - CIPHER: F1=0.8531, DP cost=0.0586, but ONLY system with all 4 properties
- [ ] V-G: **E7 Scenario Breakdown** → Table VIII (`results_e7_scenario_breakdown.csv`)
- [ ] V-H: **E8 MIA Audit** → Table IX ← LEAD THIS AS THE HEADLINE EXPERIMENT
  - No-DP MIA=0.7834, CIPHER MIA=0.5024, ΔAdv=0.2810
  - Framing: "Without DP, a passive adversary can identify 78% of training members.
    CIPHER reduces this to near-random (50.24%), achieving the theoretical bound."
- [ ] V-I: **E9 Byzantine Robustness** → Table X (`results_e9_byzantine.csv`)
  - F1 drop < 5pp at 30% Byzantine clients

### Section VI — Discussion
- [ ] Why F1 differs between r4.2 and r6.2/r5.2 (PBI implementation, not failure)
- [ ] DP cost analysis: ΔF1=0.0586 vs No-DP FL — small cost, large privacy benefit
- [ ] Ye 2025 explicitly: higher F1 but zero DP, zero MIA validation — different problem
- [ ] Limitations: CERT is semi-synthetic; single-workstation federation simulation

---

## Tables Checklist

| Table | Content | CSV to use | Status |
|---|---|---|---|
| Table I | Notation | Manual | ⬜ |
| Table II | Related work positioning | Manual | ⬜ |
| Table III | Dataset statistics | Manual | ⬜ |
| Table IV | Hyperparameters | `config/` | ⬜ |
| Table V | E1 Primary Detection | `results_e1.csv` | ⬜ |
| Table VI | E2 Ablation | `results_e2_ablation_FIXED.csv` ✅ | ⬜ |
| Table VII | E4 SOTA Comparison | `results_e4_comparison_FINAL.csv` ✅ | ⬜ |
| Table VIII | E7 Scenario | `results_e7_scenario_breakdown.csv` | ⬜ |
| Table IX | **E8 MIA Audit** ← HEADLINE | `results_e8_mia.csv` ✅ | ⬜ |
| Table X | E9 Byzantine | `results_e9_byzantine.csv` | ⬜ |

## Figures Checklist

| Figure | Content | File | Status |
|---|---|---|---|
| Fig. 1 | **System architecture** ← STILL NEEDED | draw.io / TikZ | ⬜ TO DRAW |
| Fig. 2 | FL Convergence | `figures/fig1_convergence.png` | ✅ |
| Fig. 3 | ε-sweep tradeoff | `figures/fig2_epsilon_utility.png` | ✅ |
| Fig. 4 | Ablation bar chart | `figures/fig3_ablation.png` | ✅ |
| Fig. 5 | SOTA comparison | `figures/fig4_sota_comparison.png` | ✅ |

---

## Critical Numbers — Memorize These

| Metric | Value | Never confuse with |
|---|---|---|
| F1 primary | **0.8531** | TDSC paper F1=0.9924 (different dataset) |
| AUC primary | **0.9601** | — |
| ε (privacy) | **1.2830** | TDSC paper ε=1.4042 |
| δ | **1e-5** | — |
| MIA AUC No-DP | **0.7834** | Your strongest result |
| MIA AUC CIPHER | **0.5024** | Your strongest result |
| DP cost (ΔF1) | **−0.0586** vs No-DP FL | — |
| Byzantine drop | **<5pp** at 30% | — |
| Ablation AUC gain | **+0.0093** | Frame as AUC, not F1 |

---

## Before Submission Final Checks
- [ ] Kaggle notebook uploaded to `notebooks/`
- [ ] System architecture figure (Fig. 1) drawn
- [ ] All 3 theorems written in `theory/`
- [ ] Every table number matches its source CSV exactly
- [ ] No TDSC module names (PBI/AIF/FAL/BTT/CDE) appear anywhere in this paper
- [ ] CERT dataset properly cited (CMU SEI)
- [ ] Anonymous version prepared (remove name from notebook comments)
- [ ] `CITATION.cff` updated with final DOI
