#!/usr/bin/env python3
"""
create_figureS11_spline_curves_20260223.py — KAN-CCA learned spline activations.

11-panel figure (3x4 grid) showing the learned B-spline activation functions
φ_kj(x) for top domain features and ψ_kℓ(y) for top environmental features.
Each panel shows all 10 CV fold curves (thin gray) with mean (thick color) and
linear reference (dashed), annotated with linearity R² and amplitude.

Layout:
  Row 1 (A-D): Top 4 most nonlinear domain splines
  Row 2 (E-H): 2 more domain + 2 env splines
  Row 3 (I-K): 3 more env splines + legend panel

Provenance:
  - kan_cca_results/kan_cca_spline_activations_20260222_190340.tsv
  - Nonlinearity rankings from source_data/kan_cca_results.md lines 167-192
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Figure protocol
sys.path.insert(0, str(Path(__file__).parent))
from palette import (
    DEEP_OCEAN, TURQUOISE, COASTAL_BLUE, OCEAN_BLUE,
    FOREST_GREEN, DESERT_TAN, CLAY, SAND, SIENNA,
    SEAFOAM, PALE_AQUA, SAVANNA,
)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['pdf.fonttype'] = 42  # editable text

RESULTS_DIR = Path(__file__).parent.parent / 'kan_cca_results'

# Domain features ordered by increasing linearity R² (most nonlinear first)
DOMAIN_FEATURES = [
    ('PF01576.24', 'Myosin_tail_1', DEEP_OCEAN),
    ('PF17921.7', 'Integrase_H2C2', TURQUOISE),
    ('PF00078.32', 'RVT_1', COASTAL_BLUE),
    ('PF14214.11', 'Helitron_like_N', OCEAN_BLUE),
    ('PF13927.12', 'Ig_3', FOREST_GREEN),
    ('PF05380.18', 'Peptidase_A17', DESERT_TAN),
]

# Env features ordered by increasing linearity R²
ENV_FEATURES = [
    ('rrs_443', 'R$_{rs}$(443 nm)', CLAY),
    ('rrs_488', 'R$_{rs}$(488 nm)', SIENNA),
    ('rrs_547', 'R$_{rs}$(547 nm)', SAVANNA),
    ('rrs_667', 'R$_{rs}$(667 nm)', SEAFOAM),
    ('nflh_mean', 'nFLH (mean)', FOREST_GREEN),
]

def parse_spline_data(df, branch, feature_name):
    """Extract per-fold spline curves for a feature."""
    mask = (df['branch'] == branch) & (df['feature_name'] == feature_name) & (df['output_idx'] == 0)
    subset = df[mask]
    fold_curves = []
    x_grid = None
    for _, row in subset.iterrows():
        x = np.array([float(v) for v in row['x_vals'].split(',')])
        y = np.array([float(v) for v in row['y_vals'].split(',')])
        fold_curves.append(y)
        if x_grid is None:
            x_grid = x
    return x_grid, np.array(fold_curves)

def compute_stats(x, fold_curves):
    """Compute mean curve, linear fit, R², and amplitude."""
    mean_y = np.mean(fold_curves, axis=0)

    # Linear fit to mean curve
    coeffs = np.polyfit(x, mean_y, 1)
    y_lin = np.polyval(coeffs, x)

    # R² of linear fit
    ss_res = np.sum((mean_y - y_lin) ** 2)
    ss_tot = np.sum((mean_y - mean_y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0

    # Amplitude (range of mean activation)
    amplitude = np.max(mean_y) - np.min(mean_y)

    return mean_y, y_lin, r2, amplitude

def plot_spline_panel(ax, x_grid, fold_curves, mean_y, y_lin, r2, amplitude,
                      label, color, panel_letter):
    """Plot a single spline panel."""
    # Individual folds as thin gray lines
    for curve in fold_curves:
        ax.plot(x_grid, curve, color='0.75', linewidth=0.3, alpha=0.5, zorder=1)

    # Linear reference
    ax.plot(x_grid, y_lin, color='black', linewidth=0.6, linestyle='--',
            alpha=0.6, zorder=2, label='Linear')

    # Mean spline (thick colored)
    ax.plot(x_grid, mean_y, color=color, linewidth=1.2, zorder=3, label='KAN spline')

    # Styling
    ax.set_xlim(-2, 2)
    ax.axhline(0, color='0.85', linewidth=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Title with feature name
    ax.set_title(label, fontsize=5.5, fontweight='bold', pad=3)

    # Annotation: R² and amplitude
    ax.text(0.97, 0.95, f'R²={r2:.2f}\nA={amplitude:.2f}',
            transform=ax.transAxes, fontsize=4.5, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='0.8', linewidth=0.3, alpha=0.9))

    # Panel letter
    ax.text(-0.15, 1.08, panel_letter, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top')

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load spline data
    df = pd.read_csv(RESULTS_DIR / 'kan_cca_spline_activations_20260222_190340.tsv', sep='\t')

    # Create figure: 3 rows x 4 columns
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.0))
    fig.subplots_adjust(hspace=0.55, wspace=0.35, left=0.08, right=0.97,
                        top=0.92, bottom=0.08)

    letters = 'ABCDEFGHIJKLMNOP'
    panel_idx = 0

    # Row 1-2: Domain splines (6 panels)
    for feat_id, feat_name, color in DOMAIN_FEATURES:
        row, col = divmod(panel_idx, 4)
        ax = axes[row, col]

        x_grid, fold_curves = parse_spline_data(df, 'domain', feat_id)
        mean_y, y_lin, r2, amplitude = compute_stats(x_grid, fold_curves)

        label = f'{feat_id.split(".")[0]}\n({feat_name})'
        plot_spline_panel(ax, x_grid, fold_curves, mean_y, y_lin, r2, amplitude,
                          label, color, letters[panel_idx])

        if col == 0:
            ax.set_ylabel('Spline activation φ(x)', fontsize=5)
        if row == 2 or (row == 1 and col >= 2):
            ax.set_xlabel('Standardized input', fontsize=5)

        panel_idx += 1

    # Row 2-3: Env splines (5 panels)
    for feat_id, feat_name, color in ENV_FEATURES:
        row, col = divmod(panel_idx, 4)
        ax = axes[row, col]

        x_grid, fold_curves = parse_spline_data(df, 'env', feat_id)
        mean_y, y_lin, r2, amplitude = compute_stats(x_grid, fold_curves)

        label = f'{feat_name}'
        plot_spline_panel(ax, x_grid, fold_curves, mean_y, y_lin, r2, amplitude,
                          label, color, letters[panel_idx])

        if col == 0:
            ax.set_ylabel('Spline activation ψ(y)', fontsize=5)
        ax.set_xlabel('Standardized input', fontsize=5)

        panel_idx += 1

    # Last panel (row 2, col 3): legend and summary
    ax_legend = axes[2, 3]
    ax_legend.axis('off')

    # Legend elements
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='0.75', linewidth=0.5, alpha=0.7, label='Individual folds (n=10)'),
        Line2D([0], [0], color=DEEP_OCEAN, linewidth=1.2, label='Mean KAN spline'),
        Line2D([0], [0], color='black', linewidth=0.6, linestyle='--', label='Linear reference'),
    ]
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=5.5,
                     frameon=True, fancybox=False, edgecolor='0.7',
                     title='Legend', title_fontsize=6)

    # Summary text
    ax_legend.text(0.5, 0.08,
                   'R² = linearity of spline\n'
                   '(lower = more nonlinear)\n'
                   'A = activation amplitude\n'
                   'A–F: domain features\n'
                   'G–K: environmental features',
                   transform=ax_legend.transAxes, fontsize=4.5,
                   ha='center', va='bottom', color='0.4',
                   linespacing=1.4)

    # Thin horizontal divider between domain (A-F) and env (G-K) sections
    fig.add_artist(mpl.lines.Line2D(
        [0.06, 0.96], [0.385, 0.385],
        transform=fig.transFigure, color='0.7', linewidth=0.5,
        clip_on=False, zorder=5
    ))

    # Suptitle
    fig.suptitle('Learned B-spline activation functions (KAN-CCA, CC1)',
                 fontsize=8, fontweight='bold', y=0.98)

    # Save
    for ext in ['pdf', 'svg', 'png']:
        out = Path(__file__).parent / f'FigureS11_spline_curves_{ts}.{ext}'
        fig.savefig(out, dpi=300 if ext == 'png' else None)
        print(f'Saved: {out}')

    plt.close(fig)
    print(f'\nFigure S11 generated with {panel_idx} panels.')
    print(f'Provenance: kan_cca_results/kan_cca_spline_activations_20260222_190340.tsv')

if __name__ == '__main__':
    main()
