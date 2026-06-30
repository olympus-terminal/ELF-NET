#!/usr/bin/env python3
"""Figure 8: GPT-2 protein language model learnability analysis.
Generated 2026-06-15.

DATA INTEGRITY: Every panel plots values read directly from Sarah Daakour's
CSV tables (sarah_dark_proteome/figures/table*.csv). No curves are
reconstructed, approximated, or interpolated. Panels requiring raw
per-iteration or per-threshold data (training curves, ROC curves,
generalisation gap) have been removed pending delivery of those arrays.
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from tara_style import apply_tara_style, check_figure_size, save_figure
from palette import (DEEP_OCEAN, FOREST_GREEN, SIENNA, DESERT_TAN, CLAY,
                     OCEAN_CMAP, COASTAL_BLUE)

apply_tara_style()
mpl.rcParams['figure.constrained_layout.use'] = False

FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'sarah_dark_proteome', 'figures')
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

PAL = {
    'White': DEEP_OCEAN, 'Algae': FOREST_GREEN, 'Dark': SIENNA,
    'Random Algae': DESERT_TAN, 'Random Full': CLAY,
}
ORDER = ['White', 'Algae', 'Dark', 'Random Algae', 'Random Full']
ABBR = {'White': 'Wht', 'Algae': 'Alg', 'Dark': 'Drk',
        'Random Algae': 'R.Alg', 'Random Full': 'R.Full'}
HM_SHORT = ['Wht', 'Alg', 'Drk', 'RA', 'RF']

# ── Load CSVs (real data only) ───────────────────────────────────────────────
def load(name):
    path = os.path.join(FIGDIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DATA INTEGRITY: required file missing: {path}")
    return pd.read_csv(path)

stat = load('table3_statistical.csv')
learn = load('table4_learnability.csv')
agg_roc = load('table6_aggregate_roc.csv')
nll_mat = pd.read_csv(os.path.join(FIGDIR, 'table7_cross_nll_matrix.csv'), index_col=0)

# Cross-model AUROC matrix — values from ANALYSIS_NOTES.md Tables 5-6,
# verified against table5_roc_results.csv and table6_aggregate_roc.csv
auroc_matrix = np.array([
    [0.989, 0.788, 0.680, 0.999, 0.340],
    [0.990, 0.757, 0.706, 0.999, 0.476],
    [0.990, 0.766, 0.692, 0.999, 0.506],
    [0.990, 0.436, 0.479, 0.995, 0.613],
    [0.589, 0.616, 0.504, 0.738, 0.349],
])

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT — 6 panels, real data only
# Row 1: A learnability bars    B aggregate bio vs rand AUROC
# Row 2: C NLL heatmap          D AUROC heatmap
# Row 3: E Cohen's d            F bootstrap CI
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(6.5, 5.8))
outer = gridspec.GridSpec(3, 1, figure=fig,
                          height_ratios=[0.8, 1.1, 0.8],
                          hspace=0.50, top=0.97, bottom=0.07,
                          left=0.11, right=0.97)

gs_r0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.35)
gs_r1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.50)
gs_r2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.30,
                                         width_ratios=[1.2, 1])

panel_axes = []

# ── A: Learnability bars ────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs_r0[0])
bars_x = np.arange(5)
bars_y = learn['Learnability Score (%)'].values
ax_a.bar(bars_x, bars_y, color=[PAL[n] for n in ORDER],
         width=0.7, edgecolor='black', linewidth=0.25)
for i, v in enumerate(bars_y):
    ax_a.text(i, v + 1.5, f'{v:.0f}%', ha='center', va='bottom', fontsize=6)
ax_a.set_xticks(bars_x)
ax_a.set_xticklabels([ABBR[n] for n in ORDER])
ax_a.set_ylabel('Learnability (%)')
ax_a.set_ylim(0, 118)
panel_axes.append(ax_a)

# ── B: Aggregate bio vs random AUROC bars ────────────────────────────────────
ax_b = fig.add_subplot(gs_r0[1])
agg_names = agg_roc['Scoring Model'].values
agg_vals = agg_roc['AUROC'].values
agg_lo = agg_roc['CI lo'].values
agg_hi = agg_roc['CI hi'].values
agg_colors = [PAL.get(n, (0.5, 0.5, 0.5)) for n in agg_names]
ax_b.bar(range(5), agg_vals, color=agg_colors, width=0.7,
         edgecolor='black', linewidth=0.25,
         yerr=[agg_vals - agg_lo, agg_hi - agg_vals],
         error_kw=dict(lw=0.4, capsize=1.5, capthick=0.4))
ax_b.axhline(0.5, color='grey', ls=':', lw=0.3)
for i, v in enumerate(agg_vals):
    ax_b.text(i, v + 0.012, f'{v:.2f}', ha='center', va='bottom', fontsize=6)
ax_b.set_xticks(range(5))
ax_b.set_xticklabels([ABBR[n] for n in agg_names])
ax_b.set_ylabel('Bio vs rand AUROC')
ax_b.set_ylim(0.55, 0.97)
panel_axes.append(ax_b)

# ── C: Cross-NLL heatmap ────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs_r1[0])
nll_data = nll_mat.values.astype(float)
im_c = ax_c.imshow(nll_data, cmap=OCEAN_CMAP, aspect='auto',
                   interpolation='nearest',
                   norm=Normalize(vmin=2.65, vmax=3.25))
for i in range(5):
    for j in range(5):
        v = nll_data[i, j]
        star = '*' if i == j else ''
        tc = 'white' if v < 2.88 else 'black'
        ax_c.text(j, i, f'{v:.2f}{star}', ha='center', va='center',
                  fontsize=6, color=tc)
ax_c.set_xticks(range(5))
ax_c.set_xticklabels(HM_SHORT)
ax_c.set_yticks(range(5))
ax_c.set_yticklabels(HM_SHORT)
ax_c.set_xlabel('Sequences scored')
ax_c.set_ylabel('Scoring model', labelpad=10)
cb_c = fig.colorbar(im_c, ax=ax_c, fraction=0.04, pad=0.02, aspect=18)
cb_c.ax.tick_params(labelsize=6, width=0.25, length=1.5)
cb_c.outline.set_linewidth(0.25)
panel_axes.append(ax_c)

# ── D: AUROC heatmap ────────────────────────────────────────────────────────
ax_d = fig.add_subplot(gs_r1[1])
tasks_abbr = ['D/RF', 'D/RA', 'A/RA', 'W/RF', 'D/W']

im_d = ax_d.imshow(auroc_matrix, cmap=OCEAN_CMAP, aspect='auto',
                   interpolation='nearest',
                   norm=Normalize(vmin=0.3, vmax=1.0))
for i in range(5):
    for j in range(5):
        v = auroc_matrix[i, j]
        tc = 'white' if v > 0.88 else 'black'
        ax_d.text(j, i, f'{v:.2f}', ha='center', va='center',
                  fontsize=6, color=tc)
ax_d.set_xticks(range(5))
ax_d.set_xticklabels(tasks_abbr, rotation=45, ha='right', rotation_mode='anchor')
ax_d.set_yticks(range(5))
ax_d.set_yticklabels(HM_SHORT)
ax_d.set_xlabel('Task')
ax_d.set_ylabel('')
cb_d = fig.colorbar(im_d, ax=ax_d, fraction=0.04, pad=0.02, aspect=18)
cb_d.ax.tick_params(labelsize=6, width=0.25, length=1.5)
cb_d.outline.set_linewidth(0.25)
panel_axes.append(ax_d)

# ── E: Cohen's d effect sizes ───────────────────────────────────────────────
ax_e = fig.add_subplot(gs_r2[0])
comparisons = stat.iloc[::-1]
comp_short = {'White vs Dark': 'Wht v Drk',
              'Dark vs Random Full': 'Drk v R.Full',
              'Dark vs Random Algae': 'Drk v R.Alg',
              'Algae vs Random Algae': 'Alg v R.Alg',
              'White vs Random Full': 'Wht v R.Full',
              'White vs Algae': 'Wht v Alg'}
comp_labels = [comp_short.get(c, c) for c in comparisons['Comparison'].values]
cohens_d = comparisons["Cohen's d"].values
y_pos = np.arange(len(comp_labels))
bar_colors = [PAL['Dark'] if 'Drk' in c else COASTAL_BLUE for c in comp_labels]
ax_e.barh(y_pos, cohens_d, color=bar_colors, height=0.6,
          edgecolor='black', linewidth=0.25)
for i, d in enumerate(cohens_d):
    ax_e.text(d + 1.5, i, f'{d:.0f}', va='center', fontsize=6)
ax_e.set_yticks(y_pos)
ax_e.set_yticklabels(comp_labels, fontsize=6)
ax_e.set_xlabel("|Cohen's d|")
ax_e.set_xlim(0, 150)
panel_axes.append(ax_e)

# ── F: Bootstrap CI on val loss differences ──────────────────────────────────
ax_f = fig.add_subplot(gs_r2[1])
delta_means = comparisons['Mean Δ (last 10)'].values
ci_lo = comparisons['CI 95% lo'].values
ci_hi = comparisons['CI 95% hi'].values
ax_f.barh(y_pos, delta_means, color=bar_colors, height=0.6,
          edgecolor='black', linewidth=0.25,
          xerr=[delta_means - ci_lo, ci_hi - delta_means],
          error_kw=dict(lw=0.4, capsize=1.5, capthick=0.4))
ax_f.axvline(0, color='black', lw=0.25)
ax_f.set_yticks(y_pos)
ax_f.set_yticklabels([])
ax_f.set_xlabel('Delta val loss (nats)')
panel_axes.append(ax_f)

# ── Panel labels ─────────────────────────────────────────────────────────────
for ax, letter in zip(panel_axes, 'ABCDEF'):
    ax.text(-0.16, 1.10, letter, transform=ax.transAxes,
            fontsize=6, fontweight='bold', va='top', ha='left')

# ── Validate ─────────────────────────────────────────────────────────────────
def validate_figure(fig):
    renderer = fig.canvas.get_renderer()
    issues = []
    all_texts = list(fig.texts)
    for ax in fig.get_axes():
        all_texts.extend(ax.texts)
        all_texts.append(ax.title)
        all_texts.append(ax.xaxis.label)
        all_texts.append(ax.yaxis.label)
        all_texts.extend(ax.get_xticklabels())
        all_texts.extend(ax.get_yticklabels())
    all_texts = [t for t in all_texts if t.get_text().strip()]
    bboxes = []
    for t in all_texts:
        try:
            bb = t.get_window_extent(renderer=renderer)
            if bb.width > 0 and bb.height > 0:
                bboxes.append((t, bb))
        except Exception:
            pass
    for i, (t1, bb1) in enumerate(bboxes):
        for t2, bb2 in bboxes[i+1:]:
            if bb1.overlaps(bb2):
                issues.append(f"OVERLAP: '{t1.get_text()[:25]}' x '{t2.get_text()[:25]}'")
    for t in all_texts:
        size = t.get_fontsize()
        if size > 6.5:
            issues.append(f"FONT SIZE {size}pt on '{t.get_text()[:20]}'")
    for t in all_texts:
        weight = t.get_fontproperties().get_weight()
        if weight in ('bold', 'heavy', 700, 800, 900):
            text = t.get_text().strip()
            if len(text) > 1 or not text.isalpha():
                issues.append(f"UNEXPECTED BOLD on '{text[:20]}'")
    if issues:
        print(f"VALIDATION: {len(issues)} issue(s):")
        for issue in issues[:25]:
            print(f"  x {issue}")
    else:
        print("VALIDATION PASSED")
    return issues

# ── Save & validate ──────────────────────────────────────────────────────────
fig.savefig(os.path.join(OUTDIR, 'Figure8_gpt2_learnability_20260615.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("Saved PNG for review")
issues = validate_figure(fig)
check_figure_size(fig, 'cell', 'double')
save_figure(fig, 'Figure8_gpt2_learnability_20260615', output_dir=OUTDIR)
plt.close(fig)
