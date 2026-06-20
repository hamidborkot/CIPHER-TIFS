"""generate_figures.py — Regenerate all CIPHER paper figures from CSVs.

Paper: CIPHER (IEEE TIFS Submission)
Repo:  hamidborkot/CIPHER-TIFS

Usage:
    python scripts/generate_figures.py

Outputs:
    figures/fig1_convergence.png      <- Fig. 2 in paper (FL convergence)
    figures/fig2_epsilon_utility.png  <- Fig. 3 in paper (epsilon sweep)
    figures/fig3_ablation.png         <- Fig. 4 in paper (ablation)
    figures/fig4_sota_comparison.png  <- Fig. 5 in paper (SOTA)

Note: Fig. 1 (system architecture) must be drawn manually in draw.io or TikZ.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
})
import numpy as np
import os

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RES_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(FIG_DIR, exist_ok=True)


def fig1_convergence():
    """Fig. 2: FL convergence over rounds (r4.2 + r6.2)."""
    df = pd.read_csv(os.path.join(RES_DIR, 'results_convergence.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ['F1', 'AUC']):
        for dataset in df['Dataset'].unique():
            sub = df[df['Dataset'] == dataset].sort_values('Round')
            ax.plot(sub['Round'], sub[metric], marker='o', label=dataset)
        ax.set_xlabel('Federation Round')
        ax.set_ylabel(metric)
        ax.set_title(f'CIPHER FL Convergence — {metric}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle('CIPHER (DPFA): Convergence over 10 Federation Rounds', y=1.02)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig1_convergence.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def fig2_epsilon_utility():
    """Fig. 3: epsilon-sweep privacy-utility tradeoff."""
    df = pd.read_csv(os.path.join(RES_DIR, 'results_e3_privacy.csv'))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = '#1f77b4'
    color2 = '#d62728'
    ax2 = ax1.twinx()
    df_sorted = df.sort_values('Epsilon')
    ax1.plot(df_sorted['Epsilon'], df_sorted['F1'], 'o-', color=color1,
             label='F1 Score', linewidth=2)
    ax2.plot(df_sorted['Epsilon'], df_sorted['MIA_AUC'], 's--', color=color2,
             label='MIA AUC (attacker)', linewidth=2)
    ax1.axvline(x=1.283, color='gray', linestyle=':', label='Operating point ε=1.28')
    ax1.set_xlabel('Privacy Budget ε')
    ax1.set_ylabel('F1 Score', color=color1)
    ax2.set_ylabel('MIA Attacker AUC', color=color2)
    ax1.set_title('CIPHER Privacy-Utility Tradeoff (CERT r4.2)\n'
                  'Operating point: ε=1.28 → F1=0.8531, MIA AUC=0.5024')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig2_epsilon_utility.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def fig3_ablation():
    """Fig. 4: Ablation study bar chart (BDM + PSE contributions)."""
    df = pd.read_csv(os.path.join(RES_DIR, 'results_e2_ablation_FIXED.csv'))
    # Rename old module names if present in CSV
    df['Variant'] = df['Variant'].str.replace('PBI', 'BDM').str.replace('AIF', 'PSE')
    variants = df['Variant'].tolist()
    x = np.arange(len(variants))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, df['F1'], width, label='F1', color='#1f77b4')
    bars2 = ax.bar(x + width/2, df['AUC'], width, label='AUC', color='#ff7f0e')
    ax.set_xlabel('Model Variant')
    ax.set_ylabel('Score')
    ax.set_title('CIPHER Ablation Study\n'
                 'BDM (Behavioral Drift Monitor) + PSE (Persona-Stratified Ensemble)')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha='right')
    ax.set_ylim(0.7, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        ax.annotate(f'{bar.get_height():.4f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.4f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig3_ablation.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def fig4_sota():
    """Fig. 5: SOTA comparison bar chart."""
    df = pd.read_csv(os.path.join(RES_DIR, 'results_e4_comparison_FINAL.csv'))
    # Highlight CIPHER row
    colors = ['#d62728' if 'CIPHER' in str(m) else '#1f77b4'
              for m in df['Method']]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric in zip(axes, ['F1', 'AUC']):
        bars = ax.barh(df['Method'], df[metric], color=colors)
        ax.set_xlabel(metric)
        ax.set_title(f'SOTA Comparison — {metric}\n(Red = CIPHER)')
        ax.set_xlim(0.5, 1.05)
        for bar in bars:
            ax.annotate(f'{bar.get_width():.4f}',
                        xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                        xytext=(3, 0), textcoords='offset points',
                        va='center', fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
    plt.suptitle('CIPHER vs State-of-the-Art (CERT r4.2)\n'
                 'Note: Only CIPHER provides DP_Protected=Yes + MIA_Validated=Yes')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig4_sota_comparison.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


if __name__ == '__main__':
    print('[CIPHER] Regenerating all paper figures...')
    fig1_convergence()
    fig2_epsilon_utility()
    fig3_ablation()
    fig4_sota()
    print('[CIPHER] Done. All figures saved to figures/')
    print('Note: Fig. 1 (system architecture) must be drawn manually.')
