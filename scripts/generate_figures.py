"""Generate all 4 paper figures for SENTINEL-EGO.

Run: python scripts/generate_figures.py
Outputs saved to: figures/
Requires: pip install plotly kaleido pandas
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

os.makedirs('figures', exist_ok=True)

# ── Figure 1: FL Convergence (E1) ─────────────────────────────────────────────
df_conv = pd.read_csv('results/results_convergence.csv')

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=df_conv['round'], y=df_conv['F1_sentinel'],
    mode='lines+markers', name='SENTINEL-EGO (FL+DP)',
    line=dict(width=3.5), marker=dict(size=10)
))
fig1.add_trace(go.Scatter(
    x=df_conv['round'], y=df_conv['F1_isolated'],
    mode='lines+markers', name='DP-Isolated (no FL)',
    line=dict(width=3.5, dash='dash'), marker=dict(size=10)
))
for val, label in [(df_conv['F1_sentinel'].iloc[-1], 'SENTINEL-EGO'),
                   (df_conv['F1_isolated'].iloc[-1], 'DP-Isolated')]:
    fig1.add_annotation(x=10, y=val, text=f' {val:.4f}', showarrow=False,
                        font=dict(size=13), xanchor='left')
fig1.update_layout(
    title=dict(text='FL Convergence: SENTINEL-EGO vs DP-Isolated<br>'
               '<span style="font-size:15px;font-weight:normal;">'
               'CERT r4.2 | F1 per round, \u03b5=1.28, \u03b4=1e-5</span>',
               font=dict(size=19)),
    legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='center', x=0.5),
    margin=dict(r=80)
)
fig1.update_xaxes(title_text='Federated Round', tickvals=list(range(1, 11)))
fig1.update_yaxes(title_text='F1 Score', range=[0.42, 0.93])
fig1.write_image('figures/fig1_convergence.png', scale=2)
print('Fig1 saved.')

# ── Figure 2: epsilon-Utility Tradeoff (E3) ───────────────────────────────────
df_eps = pd.read_csv('results/results_e3_privacy.csv')

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df_eps['Epsilon'], y=df_eps['F1'],
    mode='lines+markers', name='F1 Score',
    line=dict(width=3.5), marker=dict(size=10, symbol='circle')
))
fig2.add_trace(go.Scatter(
    x=df_eps['Epsilon'], y=df_eps['MIA_AUC'],
    mode='lines+markers', name='MIA AUC (privacy)',
    line=dict(width=3.5, dash='dot'), marker=dict(size=10, symbol='diamond')
))
fig2.add_annotation(x=1.28, y=0.60, text='Operating point<br>\u03b5=1.28',
    showarrow=True, arrowhead=2, ax=50, ay=-40, font=dict(size=12))
fig2.add_vline(x=1.28, line_dash='dash', line_color='rgba(150,150,150,0.5)')
fig2.update_layout(
    title=dict(text='Privacy\u2013Utility Tradeoff (\u03b5-Sweep, E3)<br>'
               '<span style="font-size:15px;font-weight:normal;">'
               'CERT r4.2 | F1 and MIA AUC vs privacy budget \u03b5</span>',
               font=dict(size=19)),
    legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='center', x=0.5)
)
fig2.update_xaxes(title_text='Privacy Budget \u03b5', autorange='reversed')
fig2.update_yaxes(title_text='Score', range=[0.48, 0.93])
fig2.write_image('figures/fig2_epsilon_utility.png', scale=2)
print('Fig2 saved.')

# ── Figure 3: Ablation (E2) ───────────────────────────────────────────────────
df_abl = pd.read_csv('results/results_e2_ablation_FIXED.csv')

fig3 = go.Figure()
fig3.add_trace(go.Bar(
    name='F1 Score', x=df_abl['Variant'], y=df_abl['F1'],
    text=[f'{v:.4f}' for v in df_abl['F1']], textposition='outside'
))
fig3.add_trace(go.Bar(
    name='AUC', x=df_abl['Variant'], y=df_abl['AUC'],
    text=[f'{v:.4f}' for v in df_abl['AUC']], textposition='outside'
))
fig3.update_layout(
    barmode='group',
    title=dict(text='Ablation Study \u2014 Component Contributions (E2)<br>'
               '<span style="font-size:15px;font-weight:normal;">'
               'CERT r4.2 | F1 and AUC per variant, \u03b5=1.28</span>',
               font=dict(size=19)),
    legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='center', x=0.5)
)
fig3.update_xaxes(title_text='System Variant')
fig3.update_yaxes(title_text='Score', range=[0.78, 0.99])
fig3.update_traces(cliponaxis=False)
fig3.write_image('figures/fig3_ablation.png', scale=2)
print('Fig3 saved.')

# ── Figure 4: SOTA Comparison (E4) ────────────────────────────────────────────
df_e4 = pd.read_csv('results/results_e4_comparison_FINAL.csv')

fig4 = make_subplots(rows=1, cols=2, subplot_titles=['F1 Score', 'AUC'],
                     horizontal_spacing=0.10)

def safe_float(val):
    try: return float(val)
    except: return 0

methods = df_e4['Method'].tolist()
colors_f1 = ['#27ae60' if 'Ours' in m else '#888888' if 'Ye 2025' not in m
             else '#f39c12' for m in methods]
colors_auc = ['#27ae60' if 'Ours' in m else '#888888' for m in methods]

fig4.add_trace(go.Bar(
    y=methods,
    x=[safe_float(v) for v in df_e4['F1']],
    orientation='h',
    text=[f'{v:.4f}' if str(v) != 'N/A' else 'N/A' for v in df_e4['F1']],
    textposition='outside',
    marker_color=colors_f1, showlegend=False
), row=1, col=1)

fig4.add_trace(go.Bar(
    y=methods,
    x=[safe_float(v) for v in df_e4['AUC']],
    orientation='h',
    text=[f'{v:.4f}' if str(v) != 'N/A' else 'N/A' for v in df_e4['AUC']],
    textposition='outside',
    marker_color=colors_auc, showlegend=False
), row=1, col=2)

fig4.update_layout(
    title=dict(text='SOTA Comparison \u2014 SENTINEL-EGO vs Baselines (E4)<br>'
               '<span style="font-size:15px;font-weight:normal;">'
               'CERT r4.2 | Only SENTINEL-EGO has DP+FL+Byzantine+MIA audit</span>',
               font=dict(size=19)),
    height=600
)
fig4.update_xaxes(range=[0, 1.12])
fig4.update_traces(cliponaxis=False)
fig4.write_image('figures/fig4_sota_comparison.png', scale=2)
print('Fig4 saved.')

print('\nAll figures generated. Check figures/ directory.')
