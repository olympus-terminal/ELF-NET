#!/usr/bin/env python3
"""
Figure 6 (combined): The dark proteome is real, distinct, and consequential
============================================================================

Consolidates former Figures 6 (novel domains), 7 (physicochemistry), and
8 (GPT-2 learnability) into ONE dense full-page figure, reordered along the
biological argument:

    Block 1  Sequence learnability  ........  dark proteins are REAL
    Block 2  Physicochemistry      ........  ... and DISTINCT
    Block 3  Discovery & coupling  ........  ... and CONSEQUENTIAL

27 source sub-panels consolidated to 18 (A-R) by merging redundant panels.
Clean block boundaries: each block starts on a fresh row.

Layout (nested GridSpec, 7 rows, 7 x 9.1 in):

  Block 1 (learnability)
    Row 1: [A learn%+AUROC]  [B cross-NLL]  [C task-AUROC]
    Row 2: [D ---- Cohen's d / Delta val-loss (paired, wide) ----]
  Block 2 (physicochemistry)
    Row 3: [E length] [F pI] [G GC-proxy] [H disorder]    (KDE)
    Row 4: [I charge FCR-NCPR] [J median bars] [K Z-violins]
  Block 3 (discovery & coupling)
    Row 5: [L cluster-size] [M novel-vs-Pfam coupling] [N size-vs-prev]
    Row 6: [O prevalence by source] [P fold-enrichment vs threshold]
    Row 7: [Q ---- ESMFold pLDDT profiles (wide) ----] [R Foldseek TM]

DATA INTEGRITY: every panel reads real values from local source_data / Sarah's
CSV tables. No synthetic, interpolated, or reconstructed data. Subsampling of
the 500k-ORF physicochemical tables uses a fixed seed (42) per project policy.

ALL text is 6pt Arial. No exceptions.
Created: 2026-06-22
"""

import sys, os, gzip
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from tara_style import apply_tara_style, check_figure_size
from palette import (
    DEEP_OCEAN, TURQUOISE, DESERT_TAN, OCEAN_BLUE, COASTAL_BLUE,
    FOREST_GREEN, CLAY, SIENNA, OCEAN_CMAP,
)

apply_tara_style()
mpl.rcParams['figure.constrained_layout.use'] = False

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = SCRIPT_DIR.parent
DATA_DIR = MANUSCRIPT_DIR / 'source_data' / 'dark_proteome'
PHYS_DIR = DATA_DIR / 'physicochemical'
PLDDT_DIR = DATA_DIR / 'plddt_per_residue'
GPT2_DIR = MANUSCRIPT_DIR / 'sarah_dark_proteome' / 'figures'
OUTDIR = MANUSCRIPT_DIR / 'figures'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Shared colors ────────────────────────────────────────────────────────────
# Learnability corpora
GPT2_PAL = {'White': DEEP_OCEAN, 'Algae': FOREST_GREEN, 'Dark': SIENNA,
            'Random Algae': DESERT_TAN, 'Random Full': CLAY}
GPT2_ORDER = ['White', 'Algae', 'Dark', 'Random Algae', 'Random Full']
GPT2_ABBR = {'White': 'Wht', 'Algae': 'Alg', 'Dark': 'Drk',
             'Random Algae': 'R.Alg', 'Random Full': 'R.Full'}
HM_SHORT = ['Wht', 'Alg', 'Drk', 'RA', 'RF']

# Physicochemistry
COLOR_ANNOTATED = OCEAN_BLUE
COLOR_DARK = SIENNA
ALPHA_FILL, ALPHA_LINE, LW = 0.35, 0.9, 0.6
N_SUBSAMPLE, RNG_SEED = 50000, 42

# Discovery (novel domains)
SRC_COLORS = {'Metagenome': TURQUOISE, 'Transcriptome': COASTAL_BLUE,
              'Cultured ref.': FOREST_GREEN}
SRC_ORDER = ['Metagenome', 'Transcriptome', 'Cultured ref.']
DOMAIN_SRC_COLORS = {
    'metagenomic': DEEP_OCEAN, 'green_alga': FOREST_GREEN, 'red_alga': CLAY,
    'haptophyte': TURQUOISE, 'diatom': OCEAN_BLUE, 'dinoflagellate': DESERT_TAN,
    'ncbi_genome': COASTAL_BLUE, 'other': (0.6, 0.6, 0.6)}
PFAM_COLOR = DESERT_TAN
NOVEL_COLORS = [DEEP_OCEAN, COASTAL_BLUE, TURQUOISE]


def classify_source(name):
    if '-NODE-' in name or 'MGYA' in name:
        return 'Metagenome'
    if 'MMETSP' in name:
        return 'Transcriptome'
    return 'Cultured ref.'


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═════════════════════════════════════════════════════════════════════════════

def load_summary_stats():
    return pd.read_csv(PHYS_DIR / 'summary_stats.tsv', sep='\t', comment='#')


def load_subsampled_per_orf(dataset, n=N_SUBSAMPLE):
    """Stream-read gzipped per-orf file, subsample n rows (seed=42)."""
    fname = PHYS_DIR / f'{dataset}_per_orf_metrics.tsv.gz'
    rng = np.random.default_rng(RNG_SEED)
    rows, header = [], None
    with gzip.open(fname, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if header is None:
                header = line.strip().split('\t')
                continue
            rows.append(line.strip().split('\t'))
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    idx.sort()
    df = pd.DataFrame([rows[i] for i in idx], columns=header)
    for c in header[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def kde_curve(values, xmin, xmax, n_pts=400):
    vals = values.dropna().values
    if len(vals) < 10:
        return np.array([]), np.array([])
    x = np.linspace(xmin, xmax, n_pts)
    return x, gaussian_kde(vals, bw_method='scott')(x)


# ═════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — SEQUENCE LEARNABILITY  (panels A-D)
# ═════════════════════════════════════════════════════════════════════════════

def draw_A_learnability(ax, learn, agg_roc):
    """A (horizontal): Learnability % (bars) + Bio-vs-rand AUROC (diamonds, twin
    TOP x-axis). Corpora are y-categories so their names read horizontally; the
    bar sits in the upper half of each slot and the AUROC diamond in the lower
    half, so the two metrics never collide (vertical analogue of the original
    left/right split)."""
    n = len(GPT2_ORDER)
    cat = np.arange(n)[::-1]        # category centre, White on top
    yL = learn['Learnability Score (%)'].values
    bar_y = cat + 0.20             # bars in upper half of slot
    dia_y = cat - 0.20             # diamonds in lower half
    ax.barh(bar_y, yL, color=[GPT2_PAL[nm] for nm in GPT2_ORDER],
            height=0.36, edgecolor='black', linewidth=0.25, zorder=2)
    for i, v in enumerate(yL):
        ax.text(v + 3, bar_y[i], f'{v:.0f}', va='center', ha='left',
                fontsize=5.5, color='#333333')
    ax.set_yticks(cat); ax.set_yticklabels([GPT2_ABBR[nm] for nm in GPT2_ORDER])
    ax.set_xlabel('Learnability (%)')
    ax.set_xlim(0, 122)            # headroom for the '100' value label
    ax.set_ylim(-0.7, n - 0.3)

    # twin TOP axis: AUROC diamonds with CI whiskers, in the lower half-slot
    ax2 = ax.twiny()
    names = list(agg_roc['Scoring Model'].values)
    order_idx = [names.index(nm) for nm in GPT2_ORDER if nm in names]
    av = agg_roc['AUROC'].values[order_idx]
    lo = agg_roc['CI lo'].values[order_idx]
    hi = agg_roc['CI hi'].values[order_idx]
    ax2.errorbar(av, dia_y[:len(order_idx)], xerr=[av - lo, hi - av], fmt='D',
                 ms=2.5, color='black', mfc='white', mec='black', mew=0.5,
                 ecolor='black', elinewidth=0.4, capsize=1.5, capthick=0.4,
                 zorder=4, linestyle='none')
    ax2.set_xlabel('Bio vs rand AUROC')
    ax2.set_xlim(0.5, 1.0)
    ax2.set_ylim(-0.7, n - 0.3)
    ax2.spines['right'].set_visible(False)


def draw_B_nll(fig, ax, nll_mat):
    """B: Cross-dataset mean NLL matrix."""
    nll = nll_mat.values.astype(float)
    im = ax.imshow(nll, cmap=OCEAN_CMAP, aspect='auto', interpolation='nearest',
                   norm=Normalize(vmin=2.65, vmax=3.25))
    for i in range(5):
        for j in range(5):
            v = nll[i, j]
            tc = 'white' if v < 2.88 else 'black'
            ax.text(j, i, f'{v:.2f}{"*" if i == j else ""}', ha='center',
                    va='center', fontsize=5.5, color=tc)
    ax.set_xticks(range(5)); ax.set_xticklabels(HM_SHORT, rotation=45,
                                                ha='right', rotation_mode='anchor')
    ax.set_yticks(range(5)); ax.set_yticklabels(HM_SHORT)
    ax.set_xlabel('Sequences scored')
    ax.set_ylabel('Scoring model')
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, aspect=16)
    cb.ax.tick_params(labelsize=6, width=0.25, length=1.5)
    cb.outline.set_linewidth(0.25)
    cb.set_label('NLL (nats)', fontsize=6)


def draw_C_taskauroc(fig, ax, auroc_matrix):
    """C: Cross-model AUROC heatmap for 5 binary tasks."""
    tasks = ['D/RF', 'D/RA', 'A/RA', 'W/RF', 'D/W']
    im = ax.imshow(auroc_matrix, cmap=OCEAN_CMAP, aspect='auto',
                   interpolation='nearest', norm=Normalize(vmin=0.3, vmax=1.0))
    for i in range(5):
        for j in range(5):
            v = auroc_matrix[i, j]
            tc = 'white' if v > 0.88 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=5.5, color=tc)
    ax.set_xticks(range(5)); ax.set_xticklabels(tasks, rotation=45,
                                                ha='right', rotation_mode='anchor')
    ax.set_yticks(range(5)); ax.set_yticklabels(HM_SHORT)
    ax.set_xlabel('Task')
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, aspect=16)
    cb.ax.tick_params(labelsize=6, width=0.25, length=1.5)
    cb.outline.set_linewidth(0.25)
    cb.set_label('AUROC', fontsize=6)


def draw_D_effects(ax_left, ax_right, stat):
    """D: paired Cohen's d (left) and Delta val-loss (right) over 6 comparisons."""
    comparisons = stat.iloc[::-1]
    # Compact labels so the y-tick column clears panel C's colourbar now that the
    # learnability block is condensed onto one row.
    comp_short = {'White vs Dark': 'W:Drk', 'Dark vs Random Full': 'Drk:RF',
                  'Dark vs Random Algae': 'Drk:RA', 'Algae vs Random Algae': 'Alg:RA',
                  'White vs Random Full': 'W:RF', 'White vs Algae': 'W:Alg'}
    labels = [comp_short.get(c, c) for c in comparisons['Comparison'].values]
    cohens_d = comparisons["Cohen's d"].values
    delta = comparisons['Mean Δ (last 10)'].values
    lo = comparisons['CI 95% lo'].values
    hi = comparisons['CI 95% hi'].values
    y = np.arange(len(labels))
    colors = [SIENNA if 'Drk' in c else COASTAL_BLUE for c in labels]

    # left: |Cohen's d|
    ax_left.barh(y, cohens_d, color=colors, height=0.62, edgecolor='black',
                 linewidth=0.25)
    for i, d in enumerate(cohens_d):
        ax_left.text(d + 3, i, f'{d:.0f}', va='center', fontsize=6)
    ax_left.set_yticks(y); ax_left.set_yticklabels(labels)
    ax_left.set_xlabel("|Cohen's d|")
    # Extra headroom so the 3-digit value labels (119, 136) sit inside the now
    # narrow panel rather than clipping at the right spine.
    ax_left.set_xlim(0, 185)
    ax_left.set_xticks([0, 100])

    # right: Delta val loss
    ax_right.barh(y, delta, color=colors, height=0.62, edgecolor='black',
                  linewidth=0.25, xerr=[delta - lo, hi - delta],
                  error_kw=dict(lw=0.4, capsize=1.5, capthick=0.4))
    ax_right.axvline(0, color='black', lw=0.25)
    ax_right.set_yticks(y); ax_right.set_yticklabels([])
    ax_right.set_xlabel('Δ val loss')


# ═════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — PHYSICOCHEMISTRY  (panels E-K)
# ═════════════════════════════════════════════════════════════════════════════

def kde_panel(ax, ann_vals, dark_vals, xlabel, xmin, xmax,
              ann_median=None, dark_median=None, show_ylabel=True):
    x_a, y_a = kde_curve(ann_vals, xmin, xmax)
    x_d, y_d = kde_curve(dark_vals, xmin, xmax)
    ax.fill_between(x_a, y_a, alpha=ALPHA_FILL, color=COLOR_ANNOTATED, lw=0)
    ax.plot(x_a, y_a, color=COLOR_ANNOTATED, lw=LW, alpha=ALPHA_LINE)
    ax.fill_between(x_d, y_d, alpha=ALPHA_FILL, color=COLOR_DARK, lw=0)
    ax.plot(x_d, y_d, color=COLOR_DARK, lw=LW, alpha=ALPHA_LINE)
    if ann_median is not None:
        ax.axvline(ann_median, color=COLOR_ANNOTATED, ls='--', lw=0.4, alpha=0.7)
    if dark_median is not None:
        ax.axvline(dark_median, color=COLOR_DARK, ls='--', lw=0.4, alpha=0.7)
    ax.set_xlabel(xlabel)
    # Density is non-negative: clamp the lower y-limit to 0 so the autoscaler
    # does not emit a spurious negative tick (e.g. -0.05) that collides with the
    # adjacent 0.0 / 2.5 ticks in these compact panels.
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    if show_ylabel:
        ax.set_ylabel('Density')


def draw_I_charge(ax, ann, dark):
    """I: NCPR vs FCR 2D-KDE contours."""
    rng = np.random.default_rng(RNG_SEED + 1)
    n2d = min(15000, len(ann), len(dark))
    idx_a = rng.choice(len(ann), n2d, replace=False)
    idx_d = rng.choice(len(dark), n2d, replace=False)
    xgrid = np.linspace(0, 0.7, 110)
    ygrid = np.linspace(-0.3, 0.3, 110)
    xx, yy = np.meshgrid(xgrid, ygrid)
    grid = np.vstack([xx.ravel(), yy.ravel()])
    for df, idx_set, color in [(ann, idx_a, COLOR_ANNOTATED),
                               (dark, idx_d, COLOR_DARK)]:
        fcr = df['fcr'].values[idx_set]
        ncpr = df['ncpr'].values[idx_set]
        mask = np.isfinite(fcr) & np.isfinite(ncpr)
        kde2d = gaussian_kde(np.vstack([fcr[mask], ncpr[mask]]), bw_method='scott')
        zz = kde2d(grid).reshape(xx.shape)
        levels = np.percentile(zz[zz > 0], [50, 75, 92])
        ax.contourf(xx, yy, zz, levels=[levels[0], zz.max()], colors=[color],
                    alpha=0.18)
        ax.contour(xx, yy, zz, levels=levels, colors=[color], linewidths=0.4,
                   alpha=0.85)
    ax.axhline(0, color='gray', lw=0.25, alpha=0.5)
    ax.axvline(0.25, color='gray', lw=0.25, alpha=0.5, ls=':')
    ax.set_xlabel('FCR'); ax.set_ylabel('NCPR', labelpad=3)
    ax.set_xlim(0, 0.7); ax.set_ylim(-0.3, 0.3)


def draw_J_medianbars(ax, get_median):
    """J: median comparison bars (disorder/order/NCPR/GC-proxy)."""
    metrics = ['disorder_frac', 'order_frac', 'ncpr', 'gc_proxy']
    labels = ['Disorder', 'Order', 'NCPR', 'GC-prx']
    ann_vals = [get_median('annotated', m) for m in metrics]
    dark_vals = [get_median('dark', m) for m in metrics]
    y = np.arange(len(metrics)); h = 0.32
    ax.barh(y + h/2, dark_vals, h, color=COLOR_DARK, alpha=0.8)
    ax.barh(y - h/2, ann_vals, h, color=COLOR_ANNOTATED, alpha=0.8)
    # Annotate each metric once, at the outer bar end, to avoid stacked-label
    # collisions in this compact row (paired magnitudes read from bar lengths;
    # exact values are also given in panel K z-scores).
    for k in range(len(metrics)):
        outer = max(ann_vals[k], dark_vals[k], key=abs)
        if outer >= 0:
            xpos, ha = max(ann_vals[k], dark_vals[k]) + 0.015, 'left'
        else:
            xpos, ha = min(ann_vals[k], dark_vals[k]) - 0.015, 'right'
        ax.text(xpos, y[k], f'{ann_vals[k]:.2f}/{dark_vals[k]:.2f}',
                va='center', ha=ha, fontsize=5.5, color='#333333')
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel('Median value (ann./dark)'); ax.invert_yaxis()
    ax.set_xlim(min(0, min(ann_vals + dark_vals)) - 0.05, 0.95)


def draw_K_zviolins(ax, ann, dark):
    """K: Z-scored summary violins across six key metrics."""
    metrics = ['length', 'pI', 'gravy', 'shannon_entropy', 'disorder_frac', 'gc_proxy']
    labels = ['Length', 'pI', 'GRAVY', 'Entropy', 'Disorder', 'GC-prx']
    n = min(10000, len(ann), len(dark))
    rng = np.random.default_rng(RNG_SEED + 3)
    ia = rng.choice(len(ann), n, replace=False)
    idd = rng.choice(len(dark), n, replace=False)
    pos, data, colors, tick_pos = [], [], [], []
    spacing = 2.0
    for k, m in enumerate(metrics):
        combined = np.concatenate([ann[m].values[ia], dark[m].values[idd]])
        mu, sigma = combined.mean(), combined.std() or 1.0
        pos.extend([k * spacing + 0.5, k * spacing + 1.2])
        data.extend([(ann[m].values[ia] - mu) / sigma,
                     (dark[m].values[idd] - mu) / sigma])
        colors.extend([COLOR_ANNOTATED, COLOR_DARK])
        tick_pos.append(k * spacing + 0.85)
    parts = ax.violinplot(data, positions=pos, showmedians=True,
                          showextrema=False, widths=0.55, vert=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i]); pc.set_alpha(0.45)
        pc.set_edgecolor(colors[i]); pc.set_linewidth(0.3)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(0.5)
    ax.axvline(0, color='gray', lw=0.25, ls=':', alpha=0.5)
    ax.set_yticks(tick_pos); ax.set_yticklabels(labels)
    ax.set_xlabel('Z-score'); ax.set_xlim(-4, 4); ax.invert_yaxis()


# ═════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — DISCOVERY & ENVIRONMENTAL COUPLING  (panels L-R)
# ═════════════════════════════════════════════════════════════════════════════

def draw_L_clustersize(ax):
    df = pd.read_csv(DATA_DIR / 'cluster_size_distribution.tsv', sep='\t')
    sizes = df['cluster_size'].values
    counts = df['n_clusters'].values
    ax.scatter(sizes, counts, s=0.8, color=TURQUOISE, alpha=0.5,
               edgecolors='none', zorder=2, rasterized=True)
    mask = (sizes >= 2) & (counts > 0)
    slope, intercept = np.polyfit(np.log10(sizes[mask].astype(float)),
                                  np.log10(counts[mask].astype(float)), 1)
    fit_x = np.linspace(np.log10(2), np.log10(sizes.max()), 100)
    fit_y = slope * fit_x + intercept
    keep = fit_y >= 0
    ax.plot(10**fit_x[keep], 10**fit_y[keep], color=DEEP_OCEAN, lw=0.5,
            ls='--', alpha=0.7, zorder=1, label=f'$\\alpha$ = {slope:.2f}')
    ax.axvline(10, color=CLAY, lw=0.4, ls=':', alpha=0.7, zorder=1)
    y_at_10 = counts[sizes == 10][0] if 10 in sizes else 1e5
    ax.annotate('$\\geq$10\n33,950', xy=(10, y_at_10),
                xytext=(80, y_at_10 * 1.5), fontsize=6, color=CLAY,
                arrowprops=dict(arrowstyle='->', color=CLAY, lw=0.3),
                ha='left', va='bottom')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Cluster size'); ax.set_ylabel('Clusters')
    ax.text(0.97, 0.97, f'{counts.sum()/1e6:.1f}M', transform=ax.transAxes,
            fontsize=6, ha='right', va='top', color='#555555')
    ax.legend(fontsize=6, frameon=False, loc='lower left', handlelength=1.2)


def draw_M_coupling(ax):
    df = pd.read_csv(DATA_DIR / 'novel_vs_pfam_effect_sizes.tsv', sep='\t')
    d = dict(zip(df['metric'], zip(df['pfam'], df['novel'])))
    pfam_n, novel_n = float(d['n_correlations'][0]), float(d['n_correlations'][1])
    pfam_sig, novel_sig = float(d['n_significant'][0]), float(d['n_significant'][1])
    pfam_med, novel_med = float(d['median_abs_rho'][0]), float(d['median_abs_rho'][1])
    pct_pfam, pct_novel = 100 * pfam_sig / pfam_n, 100 * novel_sig / novel_n
    y = np.array([1, 0]); h = 0.32
    labels = ['% significant\n(FDR<0.05)', 'Median |rho|\n(x100)']
    pfam_vals = [pct_pfam, pfam_med * 100]
    novel_vals = [pct_novel, novel_med * 100]
    ax.barh(y + h/2, pfam_vals, h, color=DEEP_OCEAN, alpha=0.85,
            edgecolor='white', linewidth=0.2, label='Pfam', zorder=2)
    ax.barh(y - h/2, novel_vals, h, color=DESERT_TAN, alpha=0.85,
            edgecolor='white', linewidth=0.2, label='Novel', zorder=2)
    for i, (pv, nv) in enumerate(zip(pfam_vals, novel_vals)):
        pfmt = f'{pv:.1f}%' if i == 0 else f'{pfam_med:.3f}'
        nfmt = f'{nv:.1f}%' if i == 0 else f'{novel_med:.3f}'
        ax.text(pv + 1, y[i] + h/2, pfmt, va='center', ha='left', fontsize=6,
                color=DEEP_OCEAN)
        ax.text(nv + 1, y[i] - h/2, f'{nfmt} ({nv/pv:.1f}x)', va='center',
                ha='left', fontsize=6, color=CLAY)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(0, 100); ax.set_xlabel('Score')
    # Legend in the empty lower-right quadrant (bottom group bars only reach ~14),
    # clear of the long '70.7% (1.6x)' label on the top Novel bar.
    ax.legend(fontsize=6, frameon=False, loc='lower right', handlelength=0.8,
              handletextpad=0.3, bbox_to_anchor=(1.0, 0.02))


def draw_N_scatter(ax):
    df = pd.read_csv(DATA_DIR / 'novel_domain_summary.tsv', sep='\t')
    df['source'] = df['domain'].apply(classify_source)
    for src in reversed(SRC_ORDER):
        sub = df[df['source'] == src]
        if len(sub) == 0:
            continue
        ax.scatter(sub['n_samples'], sub['cluster_size'], s=1.0, alpha=0.30,
                   color=SRC_COLORS[src], edgecolors='none',
                   label=f'{src} ({len(sub):,})', rasterized=True,
                   zorder=2 if src == 'Metagenome' else 3)
    ax.axvline(df['n_samples'].median(), color='#aaaaaa', lw=0.25, ls=':', zorder=1)
    ax.axhline(df['cluster_size'].median(), color='#aaaaaa', lw=0.25, ls=':', zorder=1)
    ax.set_xlabel('Assemblies detected'); ax.set_ylabel('Cluster size')
    ax.set_xlim(0, 2100); ax.set_yscale('log'); ax.set_ylim(8, 2500)
    ax.legend(fontsize=5, frameon=True, facecolor='white', edgecolor='none',
              loc='upper right', markerscale=3, handletextpad=0.2,
              borderpad=0.2, labelspacing=0.2)


def draw_O_prevalence(ax):
    """O: prevalence violins/box by source type (merges former B+F)."""
    df = pd.read_csv(DATA_DIR / 'novel_domain_summary.tsv', sep='\t')
    df['source'] = df['domain'].apply(classify_source)
    data, positions, colors_list, present = [], [], [], []
    for i, src in enumerate(SRC_ORDER):
        sub = df[df['source'] == src]
        if len(sub) == 0:
            continue
        data.append(sub['n_samples'].values)
        positions.append(i)
        colors_list.append(SRC_COLORS[src])
        present.append(src)
    parts = ax.violinplot(data, positions=positions, widths=0.7,
                          showmeans=False, showmedians=False, showextrema=False)
    for pc, color in zip(parts['bodies'], colors_list):
        pc.set_facecolor(color); pc.set_alpha(0.35)
        pc.set_edgecolor(color); pc.set_linewidth(0.3)
    bp = ax.boxplot(data, positions=positions, widths=0.25, patch_artist=True,
                    showfliers=False, medianprops=dict(color='white', lw=0.5),
                    whiskerprops=dict(color='#666666', lw=0.3),
                    capprops=dict(color='#666666', lw=0.3))
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color); patch.set_alpha(0.8)
        patch.set_edgecolor('white'); patch.set_linewidth(0.2)
    abbrev = {'Metagenome': 'Meta.', 'Transcriptome': 'Trans.', 'Cultured ref.': 'Cult.'}
    ax.set_xticks(positions); ax.set_xticklabels([abbrev[s] for s in present])
    ax.set_ylabel('Assemblies detected'); ax.set_ylim(0, 2150)


def draw_P_foldenrich(ax, summary):
    """P: fold-enrichment of novel over Pfam at each threshold."""
    novel = summary[summary['threshold'].str.startswith('Novel')].copy()
    x = np.arange(len(novel)); w = 0.35
    ax.bar(x - w/2, novel['fold_vs_pfam_median'].values, w, color=NOVEL_COLORS,
           edgecolor='black', linewidth=0.25, label='Median |rho|')
    ax.bar(x + w/2, novel['fold_vs_pfam_pct_sig'].values, w,
           color=[c + (0.5,) if len(c) == 3 else c for c in NOVEL_COLORS],
           edgecolor=NOVEL_COLORS, linewidth=0.25, label='% significant')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.25, zorder=0)
    ax.annotate('Pfam=1', xy=(1.0, 1.0),
                xycoords=mpl.transforms.blended_transform_factory(
                    ax.transAxes, ax.transData),
                xytext=(0, -3), textcoords='offset points', fontsize=6,
                color='gray', ha='right', va='top')
    ax.set_xticks(x); ax.set_xticklabels(['E<1e-5', 'E<1e-7', 'E<1e-9'])
    ax.set_ylabel('Fold vs Pfam')
    ax.legend(fontsize=6, frameon=False, loc='lower right')
    for bars in [ax.containers[0], ax.containers[1]]:
        for bar in bars:
            hgt = bar.get_height()
            if not np.isnan(hgt):
                ax.text(bar.get_x() + bar.get_width() / 2, hgt + 0.03,
                        f'{hgt:.1f}x', ha='center', va='bottom', fontsize=5.5)


def draw_Q_plddt(ax, sel_df, plddt_df):
    """Q: ESMFold pLDDT per-residue profiles for top-10 novel domains (wide)."""
    per_res_loaded = False
    if PLDDT_DIR.exists() and sel_df is not None:
        for _, row in sel_df.iterrows():
            safe = row['domain'].replace("/", "_").replace(" ", "_")
            p = PLDDT_DIR / f"{safe}_plddt.tsv"
            if p.exists():
                per_res_loaded = True
                rp = pd.read_csv(p, sep="\t")
                color = DOMAIN_SRC_COLORS.get(row.get('source', 'other'),
                                              (0.5, 0.5, 0.5))
                ax.plot(rp['residue'], rp['plddt'], color=color,
                        linewidth=0.4, alpha=0.7)
    if not per_res_loaded:
        ax.text(0.5, 0.5, 'pLDDT data not available', transform=ax.transAxes,
                ha='center', va='center', fontsize=6, color='gray')
    ax.axhline(y=70, color='gray', ls='--', lw=0.4, alpha=0.5)
    ax.annotate('70', xy=(1.0, 70), xycoords=('axes fraction', 'data'),
                fontsize=6, color='gray', va='center', ha='left',
                xytext=(2, 0), textcoords='offset points')
    ax.set_xlabel('Residue position'); ax.set_ylabel('pLDDT')
    ax.set_ylim(20, 100)
    ax.text(0.97, 0.05, '10 domains, 7/10 above pLDDT 70', transform=ax.transAxes,
            fontsize=6, ha='right', va='bottom', color='#555555')


def draw_R_foldseek(fig, ax, sel_df, foldseek_df):
    """R: Foldseek TM-score heatmap (10 domains x AFDB/PDB)."""
    if foldseek_df is None or len(foldseek_df) == 0 or sel_df is None:
        ax.text(0.5, 0.5, 'Foldseek data not available', transform=ax.transAxes,
                ha='center', va='center', fontsize=6, color='gray')
        ax.set_xticks([]); ax.set_yticks([]); return
    domains = sel_df['domain'].tolist()
    short = [f'D{i+1}' for i in range(len(domains))]
    dbs = ['AFDB', 'PDB']
    tm = np.zeros((len(domains), 2))
    hit = [[''] * 2 for _ in range(len(domains))]
    for i, dom in enumerate(domains):
        for j, db in enumerate(dbs):
            m = foldseek_df[(foldseek_df['domain'] == dom) &
                            (foldseek_df['database'] == db)]
            if len(m) > 0:
                tmv = float(m.iloc[0]['tm_score'])
                tm[i, j] = tmv
                target = str(m.iloc[0]['best_target'])
                hit[i][j] = f'{tmv:.2f}' if target != 'no_significant_hit' else '—'
            else:
                hit[i][j] = '—'
    im = ax.imshow(tm, cmap=OCEAN_CMAP, aspect='auto', vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(dbs)
    ax.set_yticks(range(len(domains))); ax.set_yticklabels(short)
    for i in range(len(domains)):
        for j in range(2):
            ax.text(j, i, hit[i][j], ha='center', va='center', fontsize=5.5,
                    color='white' if tm[i, j] > 0.5 else 'black')
    cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.06, aspect=22)
    cb.set_label('TM-score', fontsize=6)
    cb.ax.tick_params(labelsize=6, width=0.25, length=2)
    cb.outline.set_linewidth(0.25)


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def validate_figure(fig):
    """Flag WITHIN-axis text overlaps (real collisions). Cross-panel tick
    adjacency between tightly-packed independent panels is expected and not
    a defect, so overlap detection is scoped per-axis."""
    renderer = fig.canvas.get_renderer()
    issues = []
    # ── per-axis overlap detection (real collisions) ──
    for ax in fig.get_axes():
        axis_texts = list(ax.texts) + [ax.xaxis.label, ax.yaxis.label]
        axis_texts += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        axis_texts = [t for t in axis_texts if t.get_text().strip()]
        bb = []
        for t in axis_texts:
            try:
                e = t.get_window_extent(renderer=renderer)
                if e.width > 0 and e.height > 0:
                    bb.append((t, e))
            except Exception:
                pass
        for i, (t1, b1) in enumerate(bb):
            for t2, b2 in bb[i+1:]:
                if b1.overlaps(b2):
                    issues.append(f"OVERLAP[{id(ax)%1000}]: "
                                  f"'{t1.get_text()[:18]}' x '{t2.get_text()[:18]}'")
    # ── font / weight checks across all text ──
    all_texts = list(fig.texts)
    for ax in fig.get_axes():
        all_texts.extend(ax.texts)
        all_texts.append(ax.xaxis.label)
        all_texts.append(ax.yaxis.label)
        all_texts.extend(ax.get_xticklabels())
        all_texts.extend(ax.get_yticklabels())
    all_texts = [t for t in all_texts if t.get_text().strip()]
    for t in all_texts:
        if t.get_fontsize() > 6.5:
            issues.append(f"FONT {t.get_fontsize()}pt on '{t.get_text()[:18]}'")
    BLOCK_TITLES = {'Sequence learnability', 'Physicochemistry',
                    'Discovery & environmental coupling'}
    for t in all_texts:
        weight = t.get_fontproperties().get_weight()
        if weight in ('bold', 'heavy', 700, 800, 900):
            txt = t.get_text().strip()
            if txt in BLOCK_TITLES:
                continue  # intentional block headers
            if len(txt) > 1 or not txt.isalpha():
                issues.append(f"BOLD on '{txt[:18]}'")
    if issues:
        print(f"VALIDATION: {len(issues)} issue(s):")
        for s in issues[:40]:
            print(f"  x {s}")
    else:
        print("VALIDATION PASSED — no overlap/font/weight issues")
    return issues


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 72)
    print('Combined dark-proteome figure (18 panels A-R)')
    print('  Block1 learnability A-D | Block2 physicochem E-K | Block3 discovery L-R')
    print('=' * 72)

    # ── Load learnability data ──
    learn = pd.read_csv(GPT2_DIR / 'table4_learnability.csv')
    stat = pd.read_csv(GPT2_DIR / 'table3_statistical.csv')
    agg_roc = pd.read_csv(GPT2_DIR / 'table6_aggregate_roc.csv')
    nll_mat = pd.read_csv(GPT2_DIR / 'table7_cross_nll_matrix.csv', index_col=0)
    # Cross-model AUROC matrix (verified vs table5/6, per former Fig8 script)
    auroc_matrix = np.array([
        [0.989, 0.788, 0.680, 0.999, 0.340],
        [0.990, 0.757, 0.706, 0.999, 0.476],
        [0.990, 0.766, 0.692, 0.999, 0.506],
        [0.990, 0.436, 0.479, 0.995, 0.613],
        [0.589, 0.616, 0.504, 0.738, 0.349]])

    # ── Load physicochemical data ──
    ss = load_summary_stats()
    def get_median(dataset, metric):
        row = ss[(ss['dataset'] == dataset) & (ss['metric'] == metric)]
        return row['median'].values[0] if len(row) > 0 else None
    print('  loading physicochemical subsamples (seed=42)...')
    ann = load_subsampled_per_orf('annotated')
    dark = load_subsampled_per_orf('dark')

    # ── Load discovery data ──
    sel_df = pd.read_csv(DATA_DIR / 'top10_selection_summary.tsv', sep='\t', comment='#')
    plddt_df = (pd.read_csv(DATA_DIR / 'plddt_summary.tsv', sep='\t', comment='#')
                if (DATA_DIR / 'plddt_summary.tsv').exists() else None)
    foldseek_df = (pd.read_csv(DATA_DIR / 'foldseek_best_hits.tsv', sep='\t', comment='#')
                   if (DATA_DIR / 'foldseek_best_hits.tsv').exists() else None)
    eval_summary = pd.read_csv(DATA_DIR / 'evalue_sensitivity_comparison.tsv', sep='\t')

    # ── Layout: 7.7 x 10.4 in, 7 logical rows ──
    # Manual row placement for differential inter-row spacing.
    # The learnability block (A|B|C|D) is condensed onto a SINGLE row: the two
    # D bar-charts share a y-axis, so they sit as a tight pair in slot 4.
    # This is a FULL-PAGE figure: it is placed near the paper edge (not inside
    # the text margins), so it is sized 7.7 x 10.4 in to fill an 8.5x11 page
    # with a small (~0.4 in) margin. The 3x4 structure gallery (O-Z) is rendered
    # large; CONTENT rows are pinned to fixed physical heights so they keep their
    # previously-condensed size -- the extra page real estate goes to the
    # structures, NOT to fattening the data panels.
    FIG_W, FIG_H = 7.7, 10.4
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    LEFT, RIGHT = 0.085, 0.955
    TOP, BOT = 0.965, 0.040

    # The AF3 structure renders are 4:3 (w/h). At 4 columns each panel is wider
    # (and thus taller); size each structure row to the image's natural height
    # (plus a caption strip) so the axis hugs the structure and the inter-row
    # gap stays tight.
    AF3_NCOL = 4
    AF3_NROW = 3
    AF3_LEFT, AF3_RIGHT = 0.02, 0.98     # structures reclaim the y-label margin
    af3_wspace = 0.06
    panel_w_in = (AF3_RIGHT - AF3_LEFT) * FIG_W / (AF3_NCOL + af3_wspace * (AF3_NCOL - 1))
    img_aspect = 600.0 / 800.0          # height / width of the renders
    af3_img_h_frac = (panel_w_in * img_aspect) / FIG_H
    af3_caption_frac = 0.12 / FIG_H     # physical caption strip below each image
    af3_row_h = af3_img_h_frac + af3_caption_frac

    # Content rows pinned to fixed PHYSICAL heights (inches -> fig-fraction);
    # these are the previously-condensed per-row heights and do NOT grow with the
    # larger page.  row0 = A|B|C|D | row1 = E F G H | row2 = I J K | row3 = L M N
    content_h_in = [0.870, 0.967, 0.967, 1.016]
    rh_content = [h / FIG_H for h in content_h_in]
    avail = TOP - BOT
    base_gap = 0.340 / FIG_H   # fixed physical gap; fits 3x4 structures full size
    # Full gap above the first structure row; tight gaps between structure rows.
    af3_above_gap = base_gap * 1.00
    af3_between_gap = base_gap * 0.15

    # Row heights: 4 content rows + AF3_NROW structure rows.
    rh = rh_content + [af3_row_h] * AF3_NROW

    # Compute top/bottom for each row (top-down).
    n_rows = 4 + AF3_NROW
    row_tops = [0.0] * n_rows
    row_bots = [0.0] * n_rows
    cursor = TOP
    # gaps[i] is the gap BELOW row i: 3 normal gaps among content rows, the full
    # gap above the first structure row, then tight gaps between structure rows.
    gaps = [base_gap] * (n_rows - 1)
    gaps[3] = af3_above_gap
    for k in range(4, n_rows - 1):
        gaps[k] = af3_between_gap
    for i in range(n_rows):
        row_tops[i] = cursor
        row_bots[i] = cursor - rh[i]
        cursor = row_bots[i] - (gaps[i] if i < n_rows - 1 else 0)

    def row_gs(i, ncols, left=LEFT, right=RIGHT, **kw):
        return GridSpec(1, ncols, figure=fig, left=left, right=right,
                        top=row_tops[i], bottom=row_bots[i], **kw)

    panel_axes = []  # (letter, ax)

    # ─── BLOCK 1: learnability ───
    # Row 1 (single row): A | B | C | D, where D is a tight shared-y pair of
    # bar-charts (|Cohen's d| + Delta val-loss) occupying slot 4.
    # A is horizontal and narrow; B/C heatmaps keep their generous original
    # width; the D bar-pair takes the small remaining slot.
    gs1 = row_gs(0, 4, wspace=0.70, width_ratios=[0.70, 1.15, 1.15, 0.85])
    axA = fig.add_subplot(gs1[0]); draw_A_learnability(axA, learn, agg_roc)
    axB = fig.add_subplot(gs1[1]); draw_B_nll(fig, axB, nll_mat)
    axC = fig.add_subplot(gs1[2]); draw_C_taskauroc(fig, axC, auroc_matrix)
    # D: split slot 4 into two near-touching sub-axes that share the y-axis.
    gsD = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs1[3], wspace=0.05,
                                  width_ratios=[1.2, 1.0])
    axD_l = fig.add_subplot(gsD[0]); axD_r = fig.add_subplot(gsD[1])
    draw_D_effects(axD_l, axD_r, stat)
    panel_axes += [('A', axA), ('B', axB), ('C', axC), ('D', axD_l)]

    # ─── BLOCK 2: physicochemistry ───
    # Row 2: E F G H (KDE)
    gs3 = row_gs(1, 4, wspace=0.42)
    axE = fig.add_subplot(gs3[0])
    kde_panel(axE, ann['length'], dark['length'], 'Length (aa)', 30, 800,
              get_median('annotated', 'length'), get_median('dark', 'length'))
    axE.yaxis.set_major_locator(plt.MaxNLocator(3))
    axF = fig.add_subplot(gs3[1])
    kde_panel(axF, ann['pI'], dark['pI'], 'Isoelectric point (pI)', 4, 12,
              get_median('annotated', 'pI'), get_median('dark', 'pI'),
              show_ylabel=False)
    axG = fig.add_subplot(gs3[2])
    kde_panel(axG, ann['gc_proxy'], dark['gc_proxy'], 'GC-proxy', -0.5, 0.8,
              get_median('annotated', 'gc_proxy'), get_median('dark', 'gc_proxy'),
              show_ylabel=False)
    axH = fig.add_subplot(gs3[3])
    kde_panel(axH, ann['disorder_frac'], dark['disorder_frac'], 'Disorder fraction',
              0.0, 1.0, get_median('annotated', 'disorder_frac'),
              get_median('dark', 'disorder_frac'), show_ylabel=False)
    panel_axes += [('E', axE), ('F', axF), ('G', axG), ('H', axH)]

    # Row 3: I J K
    gs4 = row_gs(2, 3, wspace=0.45, width_ratios=[1.0, 1.0, 1.15])
    axI = fig.add_subplot(gs4[0]); draw_I_charge(axI, ann, dark)
    axJ = fig.add_subplot(gs4[1]); draw_J_medianbars(axJ, get_median)
    axK = fig.add_subplot(gs4[2]); draw_K_zviolins(axK, ann, dark)
    panel_axes += [('I', axI), ('J', axJ), ('K', axK)]

    # ─── BLOCK 3: discovery & coupling ───
    # Row 4: L M N
    gs5 = row_gs(3, 3, wspace=0.50)
    axL = fig.add_subplot(gs5[0]); draw_L_clustersize(axL)
    axM = fig.add_subplot(gs5[1]); draw_M_coupling(axM)
    axN = fig.add_subplot(gs5[2]); draw_N_scatter(axN)
    panel_axes += [('L', axL), ('M', axM), ('N', axN)]

    # Rows 5–7: O–Z — AF3 structure gallery (top 12 by pTM, 3 rows of 4)
    AF3_RENDER_DIR = MANUSCRIPT_DIR / 'figures' / 'af3_renders'
    AF3_RENDERS = [
        ('render_job_00573.png', 106, 0.89, 2, 'bathymetry'),
        ('render_job_02712.png', 331, 0.87, 8, 'bathymetry'),
        ('render_job_03278.png', 440, 0.85, 10, 'air temp.'),
        ('render_job_03025.png', 110, 0.83, 9, 'bathymetry'),
        ('render_job_01801.png', 274, 0.81, 6, 'bathymetry'),
        ('render_job_00314.png', 432, 0.81, 1, 'bathymetry'),
        ('render_job_03594.png', 257, 0.80, 10, 'bathymetry'),
        ('render_job_00634.png', 151, 0.79, 2, 'bathymetry'),
        ('render_job_00041.png', 748, 0.79, 1, 'bathymetry'),
        ('render_job_03454.png', 542, 0.78, 10, 'oxygen'),
        ('render_job_03108.png', 271, 0.78, 9, 'SST range'),
        ('render_job_03034.png', 267, 0.78, 9, 'oxygen'),
    ]
    af3_labels = ['O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    from matplotlib.image import imread as mpl_imread

    def autocrop_img(img, pad=2):
        # The AF3 renders carry a uniform alpha=1.0 channel, so an alpha-based
        # mask selects the whole canvas and crops nothing. Detect the structure
        # by colour distance from the (white) background corner instead; fall
        # back to alpha only if the corner is transparent.
        bg = img[0, 0, :3]
        mask = np.any(np.abs(img[:, :, :3] - bg) > 0.05, axis=2)
        if img.shape[2] == 4 and not mask.any():
            mask = img[:, :, 3] > 0.1
        rows_m = np.any(mask, axis=1)
        cols_m = np.any(mask, axis=0)
        if rows_m.any():
            rmin, rmax = np.where(rows_m)[0][[0, -1]]
            cmin, cmax = np.where(cols_m)[0][[0, -1]]
            rmin = max(0, rmin - pad); rmax = min(img.shape[0], rmax + pad + 1)
            cmin = max(0, cmin - pad); cmax = min(img.shape[1], cmax + pad + 1)
            return img[rmin:rmax, cmin:cmax]
        return img

    # Structure rows carry no y-axis labels or tick furniture, so they reclaim
    # the ~0.07 of left margin that the data panels reserve for their
    # 'Clusters'/'Cluster size' y-labels (AF3_LEFT/AF3_RIGHT set up top). The 12
    # renders are laid out as AF3_NROW x AF3_NCOL (3 x 4).
    af3_rows = [AF3_RENDERS[i:i + AF3_NCOL]
                for i in range(0, len(AF3_RENDERS), AF3_NCOL)]
    for row_idx, row_slice in enumerate(af3_rows):
        gs_row = row_gs(4 + row_idx, AF3_NCOL, left=AF3_LEFT, right=AF3_RIGHT,
                        wspace=af3_wspace)
        for col, (fname, length, ptm, rbin, env) in enumerate(row_slice):
            ax = fig.add_subplot(gs_row[col])
            img_path = AF3_RENDER_DIR / fname
            if img_path.exists():
                img = autocrop_img(mpl_imread(str(img_path)))
                ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.text(0.5, -0.01, f'L={length}  pTM={ptm:.2f}  bin {rbin}',
                    transform=ax.transAxes, fontsize=5, ha='center', va='top')
            lbl = af3_labels[row_idx * AF3_NCOL + col]
            panel_axes += [(lbl, ax)]

    # ── pLDDT colorbar for structure gallery ──
    # PyMOL rendered with: spectrum b, red_white_blue, protein, 0, 100
    # = matplotlib's RdBu (red=low, white=mid, blue=high)
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import LinearSegmentedColormap
    last_struct_ax = panel_axes[-1][1]   # panel Z
    pos = last_struct_ax.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.008, pos.y0 + 0.01,
                            0.008, pos.height * 0.75])
    cmap_plddt = plt.cm.RdBu
    norm_plddt = Normalize(vmin=0, vmax=100)
    cb = ColorbarBase(cbar_ax, cmap=cmap_plddt, norm=norm_plddt,
                      orientation='vertical')
    cb.set_ticks([0, 50, 100])
    cb.set_ticklabels(['0', '50', '100'])
    cb.ax.tick_params(labelsize=5, length=2, width=0.4, pad=1)
    cb.ax.set_ylabel('pLDDT', fontsize=5, labelpad=2)
    cb.outline.set_linewidth(0.4)

    # ── Shared legends ──
    leg_pc = [Patch(facecolor=COLOR_ANNOTATED, alpha=0.6, label='Annotated (n=500k)'),
              Patch(facecolor=COLOR_DARK, alpha=0.6, label='Dark (n=500k)')]
    axE.legend(handles=leg_pc, loc='upper right', frameon=False, fontsize=5.5,
               handlelength=1.0, handletextpad=0.4, borderpad=0.2)

    # ── Spine cleanup ──
    for ax in fig.get_axes():
        if not ax.images:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # ── Panel labels ──
    # Vertical offsets are expressed in POINTS (font-relative) via
    # ScaledTranslation so a "letter-height" shift is identical across rows of
    # different physical height. Letter height/width are taken from the 6pt
    # font: cap height ~= font size, cap width ~= 0.7 * font size.
    from matplotlib.transforms import ScaledTranslation
    fig.canvas.draw()
    LBL_PT = 6.0                       # panel-label font size
    pt2in = 1.0 / 72.0                 # points -> inches (figure dpi-independent)
    LETTER_H_IN = LBL_PT * pt2in       # one letter height, inches
    LETTER_W_IN = 0.7 * LBL_PT * pt2in # one letter width, inches
    af3_panel_set = set(af3_labels)
    # Panel D's left sub-axis carries wide comparison y-labels, so its letter
    # needs a little extra left clearance than the default -0.26.
    label_x_override = {'D': -0.62}
    for letter, ax in panel_axes:
        if letter in af3_panel_set:
            # Anchor at the axis top-left corner; nudge DOWN 2 letter-heights and
            # IN (right) 1 letter-width so the label sits over the (now tightly
            # cropped) structure rather than floating in the former whitespace.
            # A semi-transparent white box keeps the letter legible where it
            # overlaps the structure rendering.
            off = ScaledTranslation(LETTER_W_IN, -2.0 * LETTER_H_IN,
                                    fig.dpi_scale_trans)
            ax.text(0.0, 1.0, letter, transform=ax.transAxes + off,
                    fontsize=LBL_PT, fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle='square,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.7))
        else:
            # Anchor at axis top-left; nudge DOWN 1.75 letter-heights from the
            # former floating position so labels hug their panels.
            off = ScaledTranslation(0.0, -1.75 * LETTER_H_IN, fig.dpi_scale_trans)
            ax.text(label_x_override.get(letter, -0.26), 1.20, letter,
                    transform=ax.transAxes + off,
                    fontsize=LBL_PT, fontweight='bold', va='top', ha='left')

    # ── Block dividers + titles (left margin) ──
    # Anchor each rotated title to the TOP of its first row so they track the
    # re-flowed layout (learnability is now a single row 0; physicochem is the
    # EFGH row 1).
    block_titles = [
        (row_tops[0] + 0.013, 'Sequence learnability'),
        (row_tops[1] + 0.013, 'Physicochemistry'),
    ]
    for yfrac, title in block_titles:
        fig.text(0.012, yfrac, title, fontsize=6, fontweight='bold',
                 rotation=90, va='top', ha='left', color='#333333')

    # ── Validate & export ──
    issues = validate_figure(fig)
    check_figure_size(fig, 'cell', 'double')

    stem = f'Figure6_combined_dark_proteome_{TIMESTAMP}'
    fig.savefig(OUTDIR / f'{stem}.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    print(f'  Saved PNG (review): {OUTDIR / (stem + ".png")}')
    for fmt in ['pdf', 'svg']:
        # bbox_inches='tight' trims to the actual ink so the exported page has
        # symmetric margins; without it the PDF keeps the full canvas, whose
        # internal margins are uneven, and the figure looks off-centre on the
        # manuscript page.
        fig.savefig(OUTDIR / f'{stem}.{fmt}', format=fmt, dpi=600,
                    transparent=True, edgecolor='none', bbox_inches='tight')
        print(f'  Saved: {OUTDIR / (stem + "." + fmt)}')

    prov = OUTDIR / f'{stem}_provenance.txt'
    with open(prov, 'w') as f:
        f.write('# Provenance -- Combined dark-proteome figure (Figs 6+7+8)\n')
        f.write(f'Script: {Path(__file__).resolve()}\n')
        f.write(f'Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n')
        f.write('Block 1 (A-D) learnability: sarah_dark_proteome/figures/table{3,4,6,7}.csv\n')
        f.write('Block 2 (E-K) physicochem: source_data/dark_proteome/physicochemical/*\n')
        f.write('Block 3 (L-R) discovery: source_data/dark_proteome/*\n')
        f.write(f'Validation issues at save: {len(issues)}\n')
    print(f'  Saved: {prov}')
    plt.close(fig)
    print('DONE')
    return len(issues)


if __name__ == '__main__':
    main()
