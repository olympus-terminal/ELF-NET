#!/usr/bin/env python3
"""
Figure S7: Dark proteome physicochemical characterization
==========================================================

Artist-mode layout (Section 0 plan):

Target: full-page supplemental, 7 x 9 inches
Panel count: 10 (A-J)

ASCII Layout:
+----------+----------+----------+----------+----------+
| A Length  | B pI     | C GRAVY  | D Entropy| E GC-prx |  Row 0: 50%  (KDE)
| (KDE)    | (KDE)    | (KDE)    | (KDE)    | (KDE)    |
+----------+----------+----------+----------+----------+
| F Disord | G NCPR/  | H AA     | I Chou-  | J Summary|  Row 1: 50%
| (KDE)    |  FCR sct | (bars)   | Fasman   | (violin) |
+----------+----------+----------+----------+----------+

Layout: SubFigures, 2 rows x 5 cols
Datasets: Dark (500k) vs Annotated (500k) — Algae bulk excluded

Data: source_data/dark_proteome/physicochemical/
Created: 2026-05-13
"""

import gzip
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

_palette_dir = str(Path(__file__).resolve().parent)
if _palette_dir not in sys.path:
    sys.path.insert(0, _palette_dir)

from palette import OCEAN_BLUE, SIENNA, COASTAL_BLUE

mpl.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 6, 'axes.labelsize': 6, 'axes.titlesize': 6,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
    'axes.linewidth': 0.25,
    'xtick.major.width': 0.25, 'ytick.major.width': 0.25,
    'xtick.major.size': 2, 'ytick.major.size': 2,
    'axes.labelpad': 1, 'xtick.major.pad': 1, 'ytick.major.pad': 1,
})

SCRIPT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = SCRIPT_DIR.parent
DATA_DIR = MANUSCRIPT_DIR / 'source_data' / 'dark_proteome' / 'physicochemical'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

COLOR_ANNOTATED = OCEAN_BLUE
COLOR_DARK = SIENNA
ALPHA_FILL = 0.35
ALPHA_LINE = 0.9
LW = 0.6

N_SUBSAMPLE = 50000
RNG_SEED = 42


def load_summary_stats():
    df = pd.read_csv(DATA_DIR / 'summary_stats.tsv', sep='\t', comment='#')
    return df


def load_subsampled_per_orf(dataset, cols=None, n=N_SUBSAMPLE):
    """Stream-read gzipped per-orf file, subsample n rows."""
    fname = DATA_DIR / f'{dataset}_per_orf_metrics.tsv.gz'
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    header = None
    with gzip.open(fname, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if header is None:
                header = line.strip().split('\t')
                continue
            rows.append(line.strip().split('\t'))

    n_total = len(rows)
    idx = rng.choice(n_total, size=min(n, n_total), replace=False)
    idx.sort()
    sampled = [rows[i] for i in idx]
    df = pd.DataFrame(sampled, columns=header)
    if cols:
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    else:
        for c in header[1:]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def kde_from_data(values, xmin=None, xmax=None, n_pts=500):
    """Compute KDE density curve from array of values."""
    vals = values.dropna().values
    if len(vals) < 10:
        return np.array([]), np.array([])
    if xmin is None:
        xmin = np.percentile(vals, 0.5)
    if xmax is None:
        xmax = np.percentile(vals, 99.5)
    x = np.linspace(xmin, xmax, n_pts)
    kde = gaussian_kde(vals, bw_method='scott')
    y = kde(x)
    return x, y


def plot_kde_panel(ax, ann_vals, dark_vals, xlabel, xmin=None, xmax=None,
                   letter=None, ann_median=None, dark_median=None):
    """Overlaid KDE density for annotated vs dark."""
    x_a, y_a = kde_from_data(ann_vals, xmin, xmax)
    x_d, y_d = kde_from_data(dark_vals, xmin, xmax)
    ax.fill_between(x_a, y_a, alpha=ALPHA_FILL, color=COLOR_ANNOTATED, lw=0)
    ax.plot(x_a, y_a, color=COLOR_ANNOTATED, lw=LW, alpha=ALPHA_LINE)
    ax.fill_between(x_d, y_d, alpha=ALPHA_FILL, color=COLOR_DARK, lw=0)
    ax.plot(x_d, y_d, color=COLOR_DARK, lw=LW, alpha=ALPHA_LINE)
    if ann_median is not None:
        ax.axvline(ann_median, color=COLOR_ANNOTATED, ls='--', lw=0.4, alpha=0.7)
    if dark_median is not None:
        ax.axvline(dark_median, color=COLOR_DARK, ls='--', lw=0.4, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if letter:
        ax.text(-0.12, 1.08, letter, transform=ax.transAxes,
                fontsize=6, fontweight='bold', va='top', ha='left')


def main():
    print("Loading summary statistics...")
    ss = load_summary_stats()

    def get_median(dataset, metric):
        row = ss[(ss['dataset'] == dataset) & (ss['metric'] == metric)]
        return row['median'].values[0] if len(row) > 0 else None

    print("Loading subsampled per-orf data (annotated)...")
    ann = load_subsampled_per_orf('annotated')
    print("Loading subsampled per-orf data (dark)...")
    dark = load_subsampled_per_orf('dark')

    # ── Figure layout: 2 rows x 5 cols via SubFigures ──
    fig = plt.figure(figsize=(7, 5.5), layout='constrained')
    (row0, row1) = fig.subfigures(2, 1, height_ratios=[1, 1])
    axs_r0 = row0.subplots(1, 5, gridspec_kw={'wspace': 0.05})
    axs_r1 = row1.subplots(1, 5, gridspec_kw={'wspace': 0.05})

    # ══════════════════════════════════════════════════════
    # Row 0: KDE panels A-E
    # ══════════════════════════════════════════════════════

    # A: Length
    print("Panel A: Length...")
    plot_kde_panel(axs_r0[0], ann['length'], dark['length'],
                   'Length (aa)', xmin=30, xmax=800,
                   letter='A',
                   ann_median=get_median('annotated', 'length'),
                   dark_median=get_median('dark', 'length'))

    # B: pI
    print("Panel B: pI...")
    plot_kde_panel(axs_r0[1], ann['pI'], dark['pI'],
                   'Isoelectric point (pI)', xmin=4, xmax=12,
                   letter='B',
                   ann_median=get_median('annotated', 'pI'),
                   dark_median=get_median('dark', 'pI'))

    # C: GRAVY
    print("Panel C: GRAVY...")
    plot_kde_panel(axs_r0[2], ann['gravy'], dark['gravy'],
                   'GRAVY', xmin=-2.0, xmax=1.5,
                   letter='C',
                   ann_median=get_median('annotated', 'gravy'),
                   dark_median=get_median('dark', 'gravy'))

    # D: Shannon entropy
    print("Panel D: Shannon entropy...")
    plot_kde_panel(axs_r0[3], ann['shannon_entropy'], dark['shannon_entropy'],
                   'Shannon entropy (bits)', xmin=2.5, xmax=4.3,
                   letter='D',
                   ann_median=get_median('annotated', 'shannon_entropy'),
                   dark_median=get_median('dark', 'shannon_entropy'))

    # E: GC-proxy
    print("Panel E: GC-proxy...")
    plot_kde_panel(axs_r0[4], ann['gc_proxy'], dark['gc_proxy'],
                   'GC-proxy', xmin=-0.5, xmax=0.8,
                   letter='E',
                   ann_median=get_median('annotated', 'gc_proxy'),
                   dark_median=get_median('dark', 'gc_proxy'))

    # ══════════════════════════════════════════════════════
    # Row 1: Panels F-J
    # ══════════════════════════════════════════════════════

    # F: Disorder fraction KDE
    print("Panel F: Disorder fraction...")
    plot_kde_panel(axs_r1[0], ann['disorder_frac'], dark['disorder_frac'],
                   'Disorder fraction', xmin=0.0, xmax=1.0,
                   letter='F',
                   ann_median=get_median('annotated', 'disorder_frac'),
                   dark_median=get_median('dark', 'disorder_frac'))

    # G: NCPR vs FCR (2D KDE contours)
    print("Panel G: NCPR vs FCR...")
    ax_g = axs_r1[1]
    n_kde2d = 20000
    rng = np.random.default_rng(RNG_SEED + 1)
    idx_a = rng.choice(len(ann), n_kde2d, replace=False)
    idx_d = rng.choice(len(dark), n_kde2d, replace=False)
    xgrid = np.linspace(0, 0.7, 120)
    ygrid = np.linspace(-0.3, 0.3, 120)
    xx, yy = np.meshgrid(xgrid, ygrid)
    grid_pts = np.vstack([xx.ravel(), yy.ravel()])
    for idx_set, color, label in [
        (idx_a, COLOR_ANNOTATED, 'Annotated'),
        (idx_d, COLOR_DARK, 'Dark'),
    ]:
        fcr = ann['fcr'].values[idx_set] if label == 'Annotated' else dark['fcr'].values[idx_set]
        ncpr = ann['ncpr'].values[idx_set] if label == 'Annotated' else dark['ncpr'].values[idx_set]
        mask = np.isfinite(fcr) & np.isfinite(ncpr)
        kde2d = gaussian_kde(np.vstack([fcr[mask], ncpr[mask]]), bw_method='scott')
        zz = kde2d(grid_pts).reshape(xx.shape)
        levels = np.percentile(zz[zz > 0], [50, 75, 92])
        ax_g.contourf(xx, yy, zz, levels=[levels[0], zz.max()], colors=[color],
                      alpha=0.18)
        ax_g.contour(xx, yy, zz, levels=levels, colors=[color], linewidths=0.4,
                     alpha=0.85)
    ax_g.axhline(0, color='gray', lw=0.25, alpha=0.5)
    ax_g.axvline(0.25, color='gray', lw=0.25, alpha=0.5, ls=':')
    ax_g.set_xlabel('FCR')
    ax_g.set_ylabel('NCPR')
    ax_g.set_xlim(0, 0.7)
    ax_g.set_ylim(-0.3, 0.3)
    ax_g.spines['top'].set_visible(False)
    ax_g.spines['right'].set_visible(False)
    ax_g.text(-0.12, 1.08, 'G', transform=ax_g.transAxes,
              fontsize=6, fontweight='bold', va='top', ha='left')

    # H: Key median comparisons (grouped horizontal bars with value annotations)
    print("Panel H: Key metric comparison...")
    ax_h = axs_r1[2]
    metrics_h = ['disorder_frac', 'order_frac', 'ncpr', 'gc_proxy']
    labels_h = ['Disorder\nfrac.', 'Order\nfrac.', 'NCPR', 'GC-\nproxy']
    ann_vals_h = [get_median('annotated', m) for m in metrics_h]
    dark_vals_h = [get_median('dark', m) for m in metrics_h]
    y_pos = np.arange(len(metrics_h))
    bar_h = 0.32
    ax_h.barh(y_pos + bar_h / 2, dark_vals_h, bar_h, color=COLOR_DARK,
              alpha=0.8)
    ax_h.barh(y_pos - bar_h / 2, ann_vals_h, bar_h, color=COLOR_ANNOTATED,
              alpha=0.8)
    for k in range(len(metrics_h)):
        for val, yoff, color in [
            (ann_vals_h[k], -bar_h / 2, COLOR_ANNOTATED),
            (dark_vals_h[k], bar_h / 2, COLOR_DARK),
        ]:
            if val >= 0:
                xoff = max(abs(ann_vals_h[k]), abs(dark_vals_h[k])) * 0.08 + 0.01
                ha = 'left'
            else:
                xoff = -(abs(val) * 0.08 + 0.01)
                ha = 'right'
            ax_h.text(val + xoff, y_pos[k] + yoff,
                      f'{val:.3f}', va='center', ha=ha, fontsize=5,
                      color=color)
    ax_h.set_yticks(y_pos)
    ax_h.set_yticklabels(labels_h)
    ax_h.set_xlabel('Median value')
    ax_h.invert_yaxis()
    ax_h.spines['top'].set_visible(False)
    ax_h.spines['right'].set_visible(False)
    ax_h.text(-0.12, 1.08, 'H', transform=ax_h.transAxes,
              fontsize=6, fontweight='bold', va='top', ha='left')

    # I: Chou-Fasman helix/strand propensity (violin)
    print("Panel I: Chou-Fasman propensity...")
    ax_i = axs_r1[3]
    rng_i = np.random.default_rng(RNG_SEED + 2)
    n_violin = 10000
    idx_va = rng_i.choice(len(ann), n_violin, replace=False)
    idx_vd = rng_i.choice(len(dark), n_violin, replace=False)

    positions_i = [1, 2, 3.5, 4.5]
    data_i = [
        ann['helix_propensity'].values[idx_va],
        dark['helix_propensity'].values[idx_vd],
        ann['strand_propensity'].values[idx_va],
        dark['strand_propensity'].values[idx_vd],
    ]
    colors_i = [COLOR_ANNOTATED, COLOR_DARK, COLOR_ANNOTATED, COLOR_DARK]
    parts = ax_i.violinplot(data_i, positions=positions_i, showmedians=True,
                            showextrema=False, widths=0.7)
    for i_body, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_i[i_body])
        pc.set_alpha(0.5)
        pc.set_edgecolor(colors_i[i_body])
        pc.set_linewidth(0.4)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(0.5)
    ax_i.axhline(1.0, color='gray', lw=0.25, ls=':', alpha=0.5)
    ax_i.set_xticks([1.5, 4.0])
    ax_i.set_xticklabels(['Helix', 'Strand'])
    ax_i.set_ylabel('Propensity')
    ax_i.spines['top'].set_visible(False)
    ax_i.spines['right'].set_visible(False)
    ax_i.text(-0.12, 1.08, 'I', transform=ax_i.transAxes,
              fontsize=6, fontweight='bold', va='top', ha='left')

    # J: Summary horizontal violin of key metrics (Z-scored)
    print("Panel J: Summary violin (horizontal)...")
    ax_j = axs_r1[4]
    summary_metrics = ['length', 'pI', 'gravy', 'shannon_entropy',
                       'disorder_frac', 'gc_proxy']
    summary_labels = ['Length', 'pI', 'GRAVY', 'Entropy', 'Disorder', 'GC-prx']
    n_summary = 10000
    rng_j = np.random.default_rng(RNG_SEED + 3)
    idx_ja = rng_j.choice(len(ann), n_summary, replace=False)
    idx_jd = rng_j.choice(len(dark), n_summary, replace=False)

    # Z-score normalize each metric for comparability
    pos = []
    data_j = []
    colors_j = []
    tick_pos = []
    spacing = 2.0
    for k, m in enumerate(summary_metrics):
        combined = np.concatenate([ann[m].values[idx_ja], dark[m].values[idx_jd]])
        mu, sigma = combined.mean(), combined.std()
        if sigma == 0:
            sigma = 1
        a_z = (ann[m].values[idx_ja] - mu) / sigma
        d_z = (dark[m].values[idx_jd] - mu) / sigma
        p_a = k * spacing + 0.5
        p_d = k * spacing + 1.2
        pos.extend([p_a, p_d])
        data_j.extend([a_z, d_z])
        colors_j.extend([COLOR_ANNOTATED, COLOR_DARK])
        tick_pos.append(k * spacing + 0.85)

    parts_j = ax_j.violinplot(data_j, positions=pos, showmedians=True,
                              showextrema=False, widths=0.55,
                              vert=False)
    for i_body, pc in enumerate(parts_j['bodies']):
        pc.set_facecolor(colors_j[i_body])
        pc.set_alpha(0.45)
        pc.set_edgecolor(colors_j[i_body])
        pc.set_linewidth(0.3)
    parts_j['cmedians'].set_color('black')
    parts_j['cmedians'].set_linewidth(0.5)
    ax_j.axvline(0, color='gray', lw=0.25, ls=':', alpha=0.5)
    ax_j.set_yticks(tick_pos)
    ax_j.set_yticklabels(summary_labels)
    ax_j.set_xlabel('Z-score')
    ax_j.set_xlim(-4, 4)
    ax_j.invert_yaxis()
    ax_j.spines['top'].set_visible(False)
    ax_j.spines['right'].set_visible(False)
    ax_j.text(-0.12, 1.08, 'J', transform=ax_j.transAxes,
              fontsize=6, fontweight='bold', va='top', ha='left')

    # ── Legend (shared) ──
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_ANNOTATED, alpha=0.6, label='Annotated (n=500,000)'),
        Patch(facecolor=COLOR_DARK, alpha=0.6, label='Dark (n=500,000)'),
    ]
    row0.legend(handles=legend_elements, loc='upper right',
                frameon=True, edgecolor='none', facecolor='white',
                framealpha=0.8, fontsize=6, ncol=2,
                bbox_to_anchor=(0.98, 1.02))

    # ── Save ──
    out_stem = SCRIPT_DIR / 'figureS7_physicochemical'
    for fmt in ['pdf', 'svg']:
        fig.savefig(f'{out_stem}.{fmt}', format=fmt,
                    transparent=True, edgecolor='none')
        print(f"Saved: {out_stem}.{fmt}")

    plt.close(fig)
    print("Done.")


if __name__ == '__main__':
    main()
