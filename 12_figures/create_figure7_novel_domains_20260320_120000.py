#!/usr/bin/env python3
"""
Figure 7: Novel Domain Characterization
# --------------------------------------------------------------------------

Single-row 4-panel figure (7.5 x 2.5 in) — compact, high data density.

Panel A — Cluster size distribution (log-log power law + threshold)
Panel B — Novel domain prevalence across 2,044 assemblies (histogram)
Panel C — Novel vs Pfam environmental coupling comparison
Panel D — Cluster size vs prevalence landscape, colored by source type

Data: source_data/dark_proteome/*.tsv
Created: 2026-03-20
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

_palette_dir = str(Path(__file__).resolve().parent)
if _palette_dir not in sys.path:
    sys.path.insert(0, _palette_dir)

from palette import (
    DEEP_OCEAN, TURQUOISE, DESERT_TAN, OCEAN_BLUE, COASTAL_BLUE,
    FOREST_GREEN, CLAY, CLOUD_WHITE, PALE_AQUA, SAND, SIENNA,
)

# ── rcParams ─────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 6, 'axes.labelsize': 6, 'axes.titlesize': 6,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 5.5,
    'axes.linewidth': 0.25,
    'xtick.major.width': 0.25, 'ytick.major.width': 0.25,
    'xtick.major.size': 2, 'ytick.major.size': 2,
    'axes.labelpad': 1, 'xtick.major.pad': 1, 'ytick.major.pad': 1,
})

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / 'source_data' / 'dark_proteome'
OUT_DIR = SCRIPT_DIR
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Source-type colors (consistent across figures)
SRC_COLORS = {
    'Metagenome': TURQUOISE,
    'Cultured reference': FOREST_GREEN,
    'NCBI genome': DESERT_TAN,
    'Other': COASTAL_BLUE,
}
SRC_ORDER = ['Metagenome', 'NCBI genome', 'Cultured reference', 'Other']

def classify_source(name):
    if '-NODE-' in name or 'MGYA' in name:
        return 'Metagenome'
    elif '.PRE.fa' in name or 'MMETSP' in name:
        return 'Cultured reference'
    elif 'GCA_' in name or 'GCF_' in name:
        return 'NCBI genome'
    return 'Other'

# --------------------------------------------------------------------------
# PANEL A — Cluster size distribution (log-log)
# --------------------------------------------------------------------------

def draw_panel_a(ax):
    df = pd.read_csv(DATA_DIR / 'cluster_size_distribution.tsv', sep='\t')
    sizes = df['cluster_size'].values
    counts = df['n_clusters'].values

    ax.scatter(sizes, counts, s=1, color=TURQUOISE, alpha=0.6,
               edgecolors='none', zorder=2, rasterized=True)

    # Power-law fit
    mask = (sizes >= 2) & (counts > 0)
    log_s = np.log10(sizes[mask].astype(float))
    log_c = np.log10(counts[mask].astype(float))
    slope, intercept = np.polyfit(log_s, log_c, 1)
    fit_x = np.linspace(np.log10(2), np.log10(sizes.max()), 100)
    fit_y = slope * fit_x + intercept
    keep = fit_y >= 0
    ax.plot(10**fit_x[keep], 10**fit_y[keep], color=DEEP_OCEAN, lw=0.5,
            ls='--', alpha=0.7, zorder=1,
            label=f'$\\alpha$ = {slope:.2f}')

    # Threshold line at 10
    ax.axvline(10, color=CLAY, lw=0.4, ls=':', alpha=0.7, zorder=1)
    y_at_10 = counts[sizes == 10][0] if 10 in sizes else 1e5
    ax.annotate('$\\geq$10\n33,950 HMMs',
                xy=(10, y_at_10), xytext=(80, y_at_10 * 1.5),
                fontsize=5, color=CLAY, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=CLAY, lw=0.3),
                ha='left', va='bottom')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Cluster size')
    ax.set_ylabel('Clusters')
    total = counts.sum()
    ax.text(0.97, 0.97, f'{total/1e6:.1f}M total',
            transform=ax.transAxes, fontsize=5, ha='right', va='top',
            color='#555555')
    ax.legend(fontsize=5, frameon=False, loc='lower left',
              handlelength=1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('A', loc='left', fontweight='bold', fontsize=8, pad=2)

# --------------------------------------------------------------------------
# PANEL B — Domain prevalence histogram
# --------------------------------------------------------------------------

def draw_panel_b(ax):
    df = pd.read_csv(DATA_DIR / 'novel_domain_prevalence.tsv', sep='\t')
    vals = df['n_samples'].values

    bins = np.linspace(0, 2044, 30)
    ax.hist(vals, bins=bins, color=TURQUOISE, edgecolor='white',
            linewidth=0.15, alpha=0.85, zorder=2)

    med = np.median(vals)
    ax.axvline(med, color=DEEP_OCEAN, lw=0.5, ls='--', zorder=3)
    ax.text(med + 30, ax.get_ylim()[1] * 0.92,
            f'med={med:.0f}', fontsize=5, color=DEEP_OCEAN,
            fontweight='bold', va='top')

    ax.text(0.97, 0.97,
            f'n={len(vals):,}\nmax={vals.max():,}',
            transform=ax.transAxes, fontsize=5, ha='right', va='top',
            color='#555555')
    ax.set_xlabel('Assemblies detected')
    ax.set_ylabel('Novel domains')
    ax.set_xlim(0, 2100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('B', loc='left', fontweight='bold', fontsize=8, pad=2)

# --------------------------------------------------------------------------
# PANEL C — Novel vs Pfam environmental effect sizes
# --------------------------------------------------------------------------

def draw_panel_c(ax):
    """Compact grouped comparison: Pfam vs novel domain environmental coupling.
    Two metrics shown as grouped horizontal bars."""
    df = pd.read_csv(DATA_DIR / 'novel_vs_pfam_effect_sizes.tsv', sep='\t')
    df = dict(zip(df['metric'], zip(df['pfam'], df['novel'])))

    pfam_n = float(df['n_correlations'][0])
    novel_n = float(df['n_correlations'][1])
    pfam_sig = float(df['n_significant'][0])
    novel_sig = float(df['n_significant'][1])
    pfam_med = float(df['median_abs_rho'][0])
    novel_med = float(df['median_abs_rho'][1])

    pct_sig_pfam = 100 * pfam_sig / pfam_n
    pct_sig_novel = 100 * novel_sig / novel_n

    # Two metrics, two groups: horizontal grouped bars
    y = np.array([1, 0])   # top = % sig, bottom = median |rho| (scaled)
    h = 0.32
    labels = ['% significant\n(FDR < 0.05)', 'Median |$\\rho$|\n(\u00d7100)']

    # Normalize median |rho| to same visual scale (x100)
    pfam_vals = [pct_sig_pfam, pfam_med * 100]
    novel_vals = [pct_sig_novel, novel_med * 100]

    ax.barh(y + h/2, pfam_vals, h, color=DEEP_OCEAN, alpha=0.85,
            edgecolor='white', linewidth=0.2, label='Pfam', zorder=2)
    ax.barh(y - h/2, novel_vals, h, color=DESERT_TAN, alpha=0.85,
            edgecolor='white', linewidth=0.2, label='Novel', zorder=2)

    # Value labels at bar ends
    for i, (pv, nv) in enumerate(zip(pfam_vals, novel_vals)):
        pfmt = f'{pv:.1f}%' if i == 0 else f'{pfam_med:.3f}'
        nfmt = f'{nv:.1f}%' if i == 0 else f'{novel_med:.3f}'
        ax.text(pv + 1, y[i] + h/2, pfmt, va='center', ha='left',
                fontsize=5, color=DEEP_OCEAN, fontweight='bold')
        fold = nv / pv
        ax.text(nv + 1, y[i] - h/2,
                f'{nfmt}  ({fold:.1f}\u00d7)',
                va='center', ha='left', fontsize=5,
                color=CLAY, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel('')
    ax.set_xlim(0, 100)
    ax.tick_params(axis='x', labelbottom=False, length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=5, frameon=False, loc='upper right',
              handlelength=0.8, handletextpad=0.3)
    ax.set_title('C', loc='left', fontweight='bold', fontsize=8, pad=2)

# --------------------------------------------------------------------------
# PANEL D — Family size vs prevalence scatter (by source type)
# --------------------------------------------------------------------------

def draw_panel_d(ax):
    """Scatter: cluster_size vs n_samples for all 33,950 novel domains,
    colored by source type."""
    df = pd.read_csv(DATA_DIR / 'novel_domain_summary.tsv', sep='\t')
    df['source'] = df['domain'].apply(classify_source)

    # Plot each source type (rasterized for density)
    for src in reversed(SRC_ORDER):
        sub = df[df['source'] == src]
        ax.scatter(sub['n_samples'], sub['cluster_size'],
                   s=1.5, alpha=0.35, color=SRC_COLORS[src],
                   edgecolors='none', label=f'{src} ({len(sub):,})',
                   rasterized=True, zorder=2 if src == 'Metagenome' else 3)

    # Marginal summary: median lines
    med_prev = df['n_samples'].median()
    med_size = df['cluster_size'].median()
    ax.axvline(med_prev, color='#aaaaaa', lw=0.25, ls=':', zorder=1)
    ax.axhline(med_size, color='#aaaaaa', lw=0.25, ls=':', zorder=1)

    ax.set_xlabel('Assemblies detected')
    ax.set_ylabel('Cluster size')
    ax.set_xlim(0, 2100)
    ax.set_yscale('log')
    ax.set_ylim(8, 2500)

    ax.legend(fontsize=4.5, frameon=True, facecolor='white', edgecolor='none',
              loc='upper right', markerscale=3, handletextpad=0.2,
              borderpad=0.2, labelspacing=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('D', loc='left', fontweight='bold', fontsize=8, pad=2)

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    print('=' * 70)
    print('Figure 7: Novel Domain Characterization')
    print(f'  {datetime.now().isoformat()}')
    print('=' * 70)

    fig = plt.figure(figsize=(7.5, 2.5))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.50,
                           width_ratios=[1.0, 1.0, 0.85, 1.0])

    print('Panel A (cluster size power law)...')
    draw_panel_a(fig.add_subplot(gs[0]))

    print('Panel B (prevalence histogram)...')
    draw_panel_b(fig.add_subplot(gs[1]))

    print('Panel C (novel vs Pfam effect sizes)...')
    draw_panel_c(fig.add_subplot(gs[2]))

    print('Panel D (size vs prevalence landscape)...')
    draw_panel_d(fig.add_subplot(gs[3]))

    for fmt in ['pdf', 'svg']:
        out = OUT_DIR / f'Figure7_novel_domains_{TIMESTAMP}.{fmt}'
        fig.savefig(str(out), format=fmt, bbox_inches='tight',
                    transparent=True, edgecolor='none', dpi=300)
        print(f'Saved: {out}')

    prov = OUT_DIR / f'Figure7_novel_domains_{TIMESTAMP}_provenance.txt'
    with open(prov, 'w') as f:
        f.write(f'# Provenance — Figure 7: Novel Domain Characterization\n')
        f.write(f'Script: {Path(__file__).resolve()}\n')
        f.write(f'Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n')
        f.write(f'Data: {DATA_DIR}\n\n')
        f.write('Panel A: cluster_size_distribution.tsv (power-law + threshold)\n')
        f.write('Panel B: novel_domain_prevalence.tsv (histogram)\n')
        f.write('Panel C: novel_vs_pfam_effect_sizes.tsv (env coupling comparison)\n')
        f.write('Panel D: novel_domain_summary.tsv (scatter, all 33,950 domains)\n')
    print(f'Saved: {prov}')
    plt.close()
    print('DONE')

if __name__ == '__main__':
    main()
