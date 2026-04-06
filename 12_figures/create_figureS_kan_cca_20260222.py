#!/usr/bin/env python3
"""
create_figureS_kan_cca_20260222.py — KAN-CCA supplementary figure.

Two full-width panels:
  (A) Paired strip plot: Linear vs KAN-CCA at k=13 sparse subset (3 CCs, 10-fold)
  (B) All 13 learned domain spline activations, colored by function (TE vs non-TE)

Provenance:
  - kan_cca_results/kan_cca_ablation_20260222_192952.tsv
  - kan_cca_results/kan_cca_spline_activations_20260222_190340.tsv
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Figure protocol
sys.path.insert(0, str(Path(__file__).parent))
from palette import (
    DEEP_OCEAN, TURQUOISE, COASTAL_BLUE, OCEAN_BLUE,
    FOREST_GREEN, DESERT_TAN, CLAY, SAND, SEAFOAM,
    SIENNA, SAVANNA, PALE_GREEN, PALE_AQUA,
    EARTH_CATEGORICAL, OCEAN_CMAP, get_sequential_cmap,
)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['axes.linewidth'] = 0.25
mpl.rcParams['xtick.major.width'] = 0.25
mpl.rcParams['ytick.major.width'] = 0.25
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['axes.labelpad'] = 1
mpl.rcParams['xtick.major.pad'] = 1
mpl.rcParams['ytick.major.pad'] = 1
mpl.rcParams['pdf.fonttype'] = 42  # editable text

RESULTS_DIR = Path(__file__).parent.parent / 'kan_cca_results'
ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# TE domains (10 of 13 sparse-selected) vs non-TE (3)
TE_DOMAINS = {
    'PF00078',   # RVT_1 (reverse transcriptase)
    'PF17917',   # RT_RNaseH
    'PF17919',   # RT_RNaseH_2
    'PF00665',   # rve (integrase core)
    'PF17921',   # Integrase_H2C2
    'PF05380',   # Peptidase_A17 (Pao retrotransposon)
    'PF14214',   # Helitron_like_N
    'PF14529',   # Exo_endo_phos_2
    'PF18701',   # DUF5641 (retrotransposon)
    'PF20209',   # DUF6570 (eukaryotic transposon)
}
NON_TE_DOMAINS = {
    'PF01576',   # myosin tail
    'PF05970',   # PIF1 helicase
    'PF13927',   # Ig-like domain
}

def panel_a(ax):
    """Paired strip plot: Linear vs KAN-CCA at k=13, showing every fold."""
    from scipy import stats

    df = pd.read_csv(RESULTS_DIR / 'kan_cca_ablation_20260222_192952.tsv', sep='\t')
    df13 = df[df['k_domain'] == 13]

    comps = [1, 2, 3]
    gap = 0.22  # horizontal gap between paired columns within each CC

    for i, c in enumerate(comps):
        cdata = df13[df13['component'] == c].sort_values('fold')
        lin_vals = cdata['linear_test_corr'].values
        kan_vals = cdata['kan_test_corr'].values

        x_lin = i - gap
        x_kan = i + gap

        # Connecting lines (fold-paired): color by direction
        for lv, kv in zip(lin_vals, kan_vals):
            color = TURQUOISE if kv > lv else CLAY
            alpha = 0.5 if kv > lv else 0.35
            ax.plot([x_lin, x_kan], [lv, kv], color=color,
                    linewidth=0.4, alpha=alpha, zorder=2)

        # Individual fold dots
        ax.scatter([x_lin] * len(lin_vals), lin_vals,
                   color=DEEP_OCEAN, s=8, zorder=3, edgecolors='white',
                   linewidths=0.2, alpha=0.85)
        ax.scatter([x_kan] * len(kan_vals), kan_vals,
                   color=TURQUOISE, s=8, zorder=3, edgecolors='white',
                   linewidths=0.2, alpha=0.85)

        # Mean markers (horizontal bar)
        for xpos, vals, col in [(x_lin, lin_vals, DEEP_OCEAN),
                                (x_kan, kan_vals, TURQUOISE)]:
            ax.plot([xpos - 0.08, xpos + 0.08], [vals.mean(), vals.mean()],
                    color=col, linewidth=1.2, zorder=4, solid_capstyle='round')

        # Significance bracket
        _, p = stats.ttest_rel(kan_vals, lin_vals)
        star = '' if p >= 0.05 else '*' if p >= 0.01 else '**' if p >= 0.001 else '***'
        y_top = max(lin_vals.max(), kan_vals.max()) + 0.03
        bracket_y = y_top + 0.01

        if star:
            # Draw bracket
            ax.plot([x_lin, x_lin, x_kan, x_kan],
                    [y_top, bracket_y, bracket_y, y_top],
                    color='black', linewidth=0.4, zorder=5)
            ax.text(i, bracket_y + 0.01, star, ha='center', va='bottom',
                    fontsize=7, fontweight='bold')
        else:
            ax.text(i, bracket_y + 0.01, 'n.s.', ha='center', va='bottom',
                    fontsize=5, color='gray')

    # Axes
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([f'CC{c}' for c in comps])
    ax.set_ylabel('Test correlation (10-fold spatial CV)')
    ax.set_ylim(-0.02, 0.88)
    ax.set_xlim(-0.55, 2.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0, color='black', linewidth=0.25, zorder=0)

    # Legend (manual, compact)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=DEEP_OCEAN,
               markersize=3, label='Linear CCA'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=TURQUOISE,
               markersize=3, label='KAN-CCA'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False,
              fontsize=5, handletextpad=0.3)
    ax.set_title('Linear vs KAN-CCA (k = 13 sparse subset)', fontsize=6,
                 fontweight='bold')

def panel_b(axes_row):
    """All 13 domain spline activations as small multiples, colored TE vs non-TE."""
    df = pd.read_csv(RESULTS_DIR / 'kan_cca_spline_activations_20260222_190340.tsv', sep='\t')
    domain = df[(df['branch'] == 'domain') & (df['output_idx'] == 0)]

    # Compute all spline curves and nonlinearity
    feat_nonlin = {}
    feat_curves = {}
    for feat_name in domain['feature_name'].unique():
        feat_data = domain[domain['feature_name'] == feat_name]
        all_x, all_y = [], []
        for _, row in feat_data.iterrows():
            x = np.array([float(v) for v in row['x_vals'].split(',')])
            y = np.array([float(v) for v in row['y_vals'].split(',')])
            all_y.append(y)
            all_x.append(x)
        mean_y = np.mean(all_y, axis=0)
        mean_x = all_x[0]

        # Also collect per-fold curves for uncertainty band
        all_y_arr = np.array(all_y)
        lo = np.percentile(all_y_arr, 10, axis=0)
        hi = np.percentile(all_y_arr, 90, axis=0)

        coeffs = np.polyfit(mean_x, mean_y, 1)
        y_lin = np.polyval(coeffs, mean_x)
        ss_res = np.sum((mean_y - y_lin) ** 2)
        ss_tot = np.sum((mean_y - mean_y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0

        feat_nonlin[feat_name] = r2
        feat_curves[feat_name] = (mean_x, mean_y, y_lin, lo, hi)

    # Sort by nonlinearity (most nonlinear first)
    sorted_feats = sorted(feat_nonlin, key=feat_nonlin.get)

    for idx, feat in enumerate(sorted_feats):
        ax = axes_row[idx]
        x, y, y_lin, lo, hi = feat_curves[feat]
        r2 = feat_nonlin[feat]
        short = feat.split('.')[0]
        is_te = short in TE_DOMAINS

        color = DEEP_OCEAN if is_te else FOREST_GREEN
        fill_color = COASTAL_BLUE if is_te else PALE_GREEN

        # Uncertainty band (10th-90th percentile across folds)
        ax.fill_between(x, lo, hi, color=fill_color, alpha=0.25, zorder=1)
        # Mean spline
        ax.plot(x, y, color=color, linewidth=0.8, zorder=3)
        # Linear reference
        ax.plot(x, y_lin, color='gray', linewidth=0.4, linestyle='--',
                alpha=0.5, zorder=2)

        # Title with Pfam ID and R2
        label = 'TE' if is_te else ''
        ax.set_title(f'{short}\nR$^2$={r2:.2f}', fontsize=5, fontweight='bold',
                     color=color, pad=2)

        ax.set_xlim(-2, 2)
        ax.axhline(0, color='gray', linewidth=0.2, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Only label leftmost axis
        if idx == 0:
            ax.set_ylabel('Activation')
        else:
            ax.set_yticklabels([])

        # Shared x label only on a few
        if idx == 6:  # middle
            ax.set_xlabel('Standardized input')
        ax.tick_params(axis='x', labelbottom=(idx in (0, 6, 12)))

def main():
    fig = plt.figure(figsize=(7.0, 4.0))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.0, 0.8],
                           hspace=0.45)

    # Panel A: full-width paired strip plot
    ax_a = fig.add_subplot(gs[0])
    panel_a(ax_a)
    ax_a.set_title('A', loc='left', fontweight='bold', fontsize=8)

    # Panel B: 13 small-multiple spline axes
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 13, subplot_spec=gs[1],
                                            wspace=0.15)
    axes_b = [fig.add_subplot(gs_b[0, i]) for i in range(13)]
    panel_b(axes_b)

    # Panel B label on first sub-axis
    axes_b[0].text(-0.5, 1.25, 'B', transform=axes_b[0].transAxes,
                   fontsize=8, fontweight='bold', va='top')

    # Add TE / non-TE legend below panel B
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color=DEEP_OCEAN, linewidth=1.0, label='TE domain (10/13)'),
        Line2D([0], [0], color=FOREST_GREEN, linewidth=1.0, label='Non-TE domain (3/13)'),
        Patch(facecolor=COASTAL_BLUE, alpha=0.25, label='10th-90th %ile across folds'),
        Line2D([0], [0], color='gray', linewidth=0.5, linestyle='--', label='Linear fit'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', frameon=False,
               fontsize=5, ncol=4, bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.3, columnspacing=1.0)

    # Save
    out_base = Path(__file__).parent / f'FigureS_kan_cca_{ts}'
    for fmt in ['pdf', 'svg']:
        fig.savefig(f'{out_base}.{fmt}', format=fmt, dpi=300,
                    bbox_inches='tight', transparent=True, edgecolor='none')
    plt.close()

    print(f"Saved: {out_base}.pdf")
    print(f"Saved: {out_base}.svg")
    print(f"Timestamp: {ts}")

if __name__ == '__main__':
    main()
