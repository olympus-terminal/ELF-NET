#!/usr/bin/env python3
"""
Figure 7: Novel Domain Discovery, Structural Validation, and Sensitivity
=========================================================================

Consolidated main figure merging former Figure 7, S6 (panels A-B, E),
and S7 into a single 11-panel composite. Former S6-C (contig neighborhoods)
and S6-D (pLDDT profiles) moved to standalone supplemental figures.

Layout (nested GridSpec, 4 rows, 7 x 7.5 in):

    Row 1 (28%): [A cluster-size]  [B prevalence]  [C coupling]  [D scatter]
    Row 2 (24%): [E dark-frac]     [F violins]     [G sig-rates]
    Row 3 (14%): [H ──────── pLDDT profiles (full width) ──────── ]
    Row 4 (34%): [I Foldseek]  [J effect-size violins]  [K fold-enrichment]

ALL text is 6pt. No exceptions.

Created: 2026-04-12
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from tara_style import apply_tara_style, check_figure_size, save_figure
from palette import (
    DEEP_OCEAN, TURQUOISE, DESERT_TAN, OCEAN_BLUE, COASTAL_BLUE,
    FOREST_GREEN, CLAY, OCEAN_CMAP,
)

apply_tara_style()
mpl.rcParams['figure.constrained_layout.use'] = False

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = SCRIPT_DIR.parent
DATA_DIR = MANUSCRIPT_DIR / 'source_data' / 'dark_proteome'
PLDDT_DIR = DATA_DIR / 'plddt_per_residue'
SENSITIVITY_DIR = DATA_DIR / 'sensitivity_per_threshold_details'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Shared colors ────────────────────────────────────────────────────────────
# Three-category framework (matches manuscript):
#   Metagenome     — TARA Oceans + OSD ocean metagenome assemblies
#   Transcriptome  — MMETSP marine eukaryotic transcriptomes
#   Cultured ref.  — NCBI GCA/GCF genomes, Phytozome, culture-collection assemblies
SRC_COLORS = {
    'Metagenome':       TURQUOISE,
    'Transcriptome':    COASTAL_BLUE,
    'Cultured ref.':    FOREST_GREEN,
}
SRC_ORDER = ['Metagenome', 'Transcriptome', 'Cultured ref.']

DOMAIN_SRC_COLORS = {
    'metagenomic': DEEP_OCEAN, 'green_alga': FOREST_GREEN,
    'red_alga': CLAY, 'haptophyte': TURQUOISE, 'diatom': OCEAN_BLUE,
    'dinoflagellate': DESERT_TAN, 'ncbi_genome': COASTAL_BLUE,
    'other': (0.6, 0.6, 0.6),
}

PFAM_COLOR = DESERT_TAN
NOVEL_COLORS = [DEEP_OCEAN, COASTAL_BLUE, TURQUOISE]
ALL_EVAL_COLORS = [PFAM_COLOR, DEEP_OCEAN, COASTAL_BLUE, TURQUOISE]


def classify_source(name):
    # Ocean metagenomes: MGnify assembly IDs (TARA + OSD) or SPAdes/megahit contigs
    if '-NODE-' in name or 'MGYA' in name:
        return 'Metagenome'
    # MMETSP transcriptomes
    if 'MMETSP' in name:
        return 'Transcriptome'
    # Everything else is a cultured reference proteome:
    # NCBI GCA/GCF, Phytozome (.PRE.fa), culture collections, other curated refs
    return 'Cultured ref.'


def shorten_domain_name(name, max_len=22):
    s = name.replace("NOVEL_", "")
    for org in ["Dunaliella", "Chlamydomonas", "Chondrus", "Chrysochromulina",
                "Characiochloris", "Micromonas", "Emiliania", "Galdieria"]:
        if org in s:
            idx = s.index(org)
            return s[idx:idx + len(org) + 12][:max_len]
    # For metagenomic domains: extract ERZ accession
    if 'ERZ' in s:
        erz = s.split('.')[0]  # e.g. ERZ17094789
        return erz
    if len(s) > max_len:
        s = s[:max_len - 1] + '\u2026'
    return s


# ═════════════════════════════════════════════════════════════════════════════
# ROW 0 — Characterization (A-D)
# ═════════════════════════════════════════════════════════════════════════════

def draw_panel_a(ax):
    """A: Cluster size distribution (log-log power law)."""
    df = pd.read_csv(DATA_DIR / 'cluster_size_distribution.tsv', sep='\t')
    sizes = df['cluster_size'].values
    counts = df['n_clusters'].values

    ax.scatter(sizes, counts, s=0.8, color=TURQUOISE, alpha=0.5,
               edgecolors='none', zorder=2, rasterized=True)

    mask = (sizes >= 2) & (counts > 0)
    log_s = np.log10(sizes[mask].astype(float))
    log_c = np.log10(counts[mask].astype(float))
    slope, intercept = np.polyfit(log_s, log_c, 1)
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
    ax.set_xlabel('Cluster size')
    ax.set_ylabel('Clusters')
    total = counts.sum()
    ax.text(0.97, 0.97, f'{total/1e6:.1f}M',
            transform=ax.transAxes, fontsize=6, ha='right', va='top',
            color='#555555')
    ax.legend(fontsize=6, frameon=False, loc='lower left', handlelength=1.2)


def draw_panel_b(ax):
    """B: Novel domain prevalence histogram."""
    df = pd.read_csv(DATA_DIR / 'novel_domain_prevalence.tsv', sep='\t')
    vals = df['n_samples'].values

    bins = np.linspace(0, 2044, 30)
    ax.hist(vals, bins=bins, color=TURQUOISE, edgecolor='white',
            linewidth=0.15, alpha=0.85, zorder=2)

    med = np.median(vals)
    ax.axvline(med, color=DEEP_OCEAN, lw=0.5, ls='--', zorder=3)
    ax.text(med + 40, ax.get_ylim()[1] * 0.92,
            f'med={med:.0f}', fontsize=6, color=DEEP_OCEAN, va='top')

    ax.text(0.97, 0.97, f'n={len(vals):,}',
            transform=ax.transAxes, fontsize=6, ha='right', va='top',
            color='#555555')
    ax.set_xlabel('Assemblies detected')
    ax.set_ylabel('Novel domains')
    ax.set_xlim(0, 2100)


def draw_panel_c(ax):
    """C: Novel vs Pfam environmental coupling."""
    df = pd.read_csv(DATA_DIR / 'novel_vs_pfam_effect_sizes.tsv', sep='\t')
    df = dict(zip(df['metric'], zip(df['pfam'], df['novel'])))

    pfam_n = float(df['n_correlations'][0])
    novel_n = float(df['n_correlations'][1])
    pfam_sig = float(df['n_significant'][0])
    novel_sig = float(df['n_significant'][1])
    pfam_med = float(df['median_abs_rho'][0])
    novel_med = float(df['median_abs_rho'][1])

    pct_pfam = 100 * pfam_sig / pfam_n
    pct_novel = 100 * novel_sig / novel_n

    y = np.array([1, 0])
    h = 0.32
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
        fold = nv / pv
        ax.text(pv + 1, y[i] + h/2, pfmt, va='center', ha='left',
                fontsize=6, color=DEEP_OCEAN)
        ax.text(nv + 1, y[i] - h/2, f'{nfmt} ({fold:.1f}x)',
                va='center', ha='left', fontsize=6, color=CLAY)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Score')
    ax.legend(fontsize=6, frameon=False, loc='upper right',
              handlelength=0.8, handletextpad=0.3,
              bbox_to_anchor=(1.0, 0.85))


def draw_panel_d(ax):
    """D: Cluster size vs prevalence scatter."""
    df = pd.read_csv(DATA_DIR / 'novel_domain_summary.tsv', sep='\t')
    df['source'] = df['domain'].apply(classify_source)

    for src in reversed(SRC_ORDER):
        sub = df[df['source'] == src]
        if len(sub) == 0:
            continue
        ax.scatter(sub['n_samples'], sub['cluster_size'],
                   s=1.0, alpha=0.30, color=SRC_COLORS[src],
                   edgecolors='none', label=f'{src} ({len(sub):,})',
                   rasterized=True, zorder=2 if src == 'Metagenome' else 3)

    ax.axvline(df['n_samples'].median(), color='#aaaaaa', lw=0.25,
               ls=':', zorder=1)
    ax.axhline(df['cluster_size'].median(), color='#aaaaaa', lw=0.25,
               ls=':', zorder=1)

    ax.set_xlabel('Assemblies detected')
    ax.set_ylabel('Cluster size')
    ax.set_xlim(0, 2100); ax.set_yscale('log'); ax.set_ylim(8, 2500)
    ax.legend(fontsize=6, frameon=True, facecolor='white', edgecolor='none',
              loc='upper right', markerscale=3, handletextpad=0.2,
              borderpad=0.2, labelspacing=0.2)


# ═════════════════════════════════════════════════════════════════════════════
# ROW 1 — Source breakdown + E-value overview (E-H)
# ═════════════════════════════════════════════════════════════════════════════

def draw_panel_e(ax):
    """E: Dark fraction by source category (3-category framework).
    Collapses 'NCBI genome' + 'Other cultured reference' → 'Cultured refs.'
    by pooling weighted medians/means across the two sub-categories."""
    df = pd.read_csv(MANUSCRIPT_DIR / 'source_data' / 'pfam_density_summary.tsv',
                     sep='\t')
    df = df[df['category'] != 'GLOBAL'].copy()

    # Collapse NCBI genome + Other cultured reference into a single
    # "Cultured refs." row, weighting median/mean/std by sample count.
    cultured_mask = df['category'].isin(['NCBI genome', 'Other cultured reference'])
    cultured = df[cultured_mask]
    other = df[~cultured_mask].copy()
    if len(cultured) > 0:
        n_tot = cultured['n_samples'].sum()
        w_med = np.average(cultured['pct_dark_median'], weights=cultured['n_samples'])
        w_mean = np.average(cultured['pct_dark_mean'], weights=cultured['n_samples'])
        # pooled SD via weighted variance of means + weighted within-group variance
        w_var = (np.average(cultured['pct_dark_std']**2, weights=cultured['n_samples']) +
                 np.average((cultured['pct_dark_mean'] - w_mean)**2,
                            weights=cultured['n_samples']))
        w_std = np.sqrt(w_var)
        pooled = pd.DataFrame([{
            'category': 'Cultured refs.', 'n_samples': int(n_tot),
            'pct_dark_median': w_med, 'pct_dark_mean': w_mean, 'pct_dark_std': w_std,
        }])
        cats = pd.concat([other, pooled], ignore_index=True)
    else:
        cats = other
    cats = cats.sort_values('n_samples', ascending=False)

    xlabels = {'Metagenome ORFs': 'Meta.', 'Transcriptome ORFs': 'Trans.',
               'Cultured refs.': 'Cult.'}
    labs = [xlabels.get(c, c) for c in cats['category']]
    n = len(labs)
    x = np.arange(n)
    medians = cats['pct_dark_median'].values
    means = cats['pct_dark_mean'].values
    stds = cats['pct_dark_std'].values
    colors = [TURQUOISE, COASTAL_BLUE, FOREST_GREEN][:n]
    w = 0.35

    ax.bar(x - w/2, medians, w, label='Median', color=colors, alpha=0.85,
           edgecolor='white', linewidth=0.2)
    ax.bar(x + w/2, means, w, yerr=stds, label='Mean +/- SD',
           color=[(*c[:3], 0.45) for c in colors],
           edgecolor=colors, linewidth=0.3,
           error_kw=dict(lw=0.3, capsize=1.5, capthick=0.3, color='#555555'))

    glob = df[df['category'] == 'GLOBAL']
    if len(glob):
        gm = float(glob['pct_dark_median'].iloc[0])
        ax.axhline(gm, color=DEEP_OCEAN, lw=0.4, ls='--', alpha=0.7)
        ax.text(-0.4, gm - 4, f'{gm:.1f}%', fontsize=6,
                color=DEEP_OCEAN, ha='left', va='top')

    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel('Dark fraction (%)')
    ax.set_ylim(0, 108)
    ax.legend(fontsize=6, frameon=False, loc='lower left')


def draw_panel_f(ax):
    """F: Prevalence violins by source type."""
    df = pd.read_csv(DATA_DIR / 'novel_domain_summary.tsv', sep='\t')
    df['source'] = df['domain'].apply(classify_source)
    data, positions, colors_list = [], [], []
    for i, src in enumerate(SRC_ORDER):
        sub = df[df['source'] == src]
        if len(sub) == 0:
            continue
        data.append(sub['n_samples'].values)
        positions.append(i)
        colors_list.append(SRC_COLORS[src])

    parts = ax.violinplot(data, positions=positions, widths=0.7,
                          showmeans=False, showmedians=False, showextrema=False)
    for pc, color in zip(parts['bodies'], colors_list):
        pc.set_facecolor(color); pc.set_alpha(0.35)
        pc.set_edgecolor(color); pc.set_linewidth(0.3)

    bp = ax.boxplot(data, positions=positions, widths=0.25, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color='white', lw=0.5),
                    whiskerprops=dict(color='#666666', lw=0.3),
                    capprops=dict(color='#666666', lw=0.3))
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color); patch.set_alpha(0.8)
        patch.set_edgecolor('white'); patch.set_linewidth(0.2)

    ax.set_xticks(positions)
    abbrev = {'Metagenome': 'Meta.', 'Transcriptome': 'Trans.',
              'Cultured ref.': 'Cult.'}
    ax.set_xticklabels([abbrev.get(s, s)
                        for s in SRC_ORDER if any(df['source'] == s)])
    ax.set_ylabel('Assemblies detected')
    ax.set_ylim(0, 2150)


def draw_panel_g(ax, summary):
    """G: Domains retained at each E-value threshold."""
    novel = summary[summary['threshold'].str.startswith('Novel')]
    x = np.arange(len(novel))
    w = 0.35

    ax.bar(x - w/2, novel['n_domains_total'].values, w,
           color=[c + (0.5,) if len(c) == 3 else c for c in NOVEL_COLORS],
           edgecolor=NOVEL_COLORS, linewidth=0.25, label='Total')
    ax.bar(x + w/2, novel['n_domains_prevalent'].values, w,
           color=NOVEL_COLORS, edgecolor='black', linewidth=0.25,
           label='Prevalent')

    ax.set_xticks(x)
    ax.set_xticklabels(['E<1e-5', 'E<1e-7', 'E<1e-9'])
    ax.set_ylabel('Domains')
    ax.set_ylim(0, 42000)
    ax.legend(fontsize=6, frameon=False, loc='upper center')


def draw_panel_h(ax, summary):
    """H: Significance rates across thresholds."""
    x = np.arange(len(summary))
    bars = ax.bar(x, summary['pct_significant'].values,
                  color=ALL_EVAL_COLORS, edgecolor='black', linewidth=0.25)

    pfam_pct = summary.loc[summary['threshold'] == 'Pfam_1e-9',
                           'pct_significant'].values
    if len(pfam_pct) > 0:
        ax.axhline(pfam_pct[0], color=PFAM_COLOR, linestyle='--',
                   linewidth=0.4, alpha=0.6, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(['Pfam', 'Nov\n1e-5', 'Nov\n1e-7', 'Nov\n1e-9'])
    ax.set_ylabel('Significant (%)')
    ax.set_ylim(0, 95)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f'{h:.0f}%', ha='center', va='bottom', fontsize=6)


# ═════════════════════════════════════════════════════════════════════════════
# ROW 2 — pLDDT profiles (I, full width, compact height)
# ═════════════════════════════════════════════════════════════════════════════

def draw_panel_plddt(ax, sel_df, plddt_df):
    """H: ESMFold pLDDT per-residue profiles for top-10 novel domains (full width)."""
    per_res_loaded = False
    if PLDDT_DIR.exists() and sel_df is not None:
        for _, row in sel_df.iterrows():
            domain = row['domain']
            safe_name = domain.replace("/", "_").replace(" ", "_")
            plddt_path = PLDDT_DIR / f"{safe_name}_plddt.tsv"
            if plddt_path.exists():
                per_res_loaded = True
                residue_plddt = pd.read_csv(plddt_path, sep="\t")
                color = DOMAIN_SRC_COLORS.get(
                    row.get('source', 'other'), (0.5, 0.5, 0.5))
                ax.plot(residue_plddt['residue'], residue_plddt['plddt'],
                        color=color, linewidth=0.4, alpha=0.7)

    if not per_res_loaded:
        if plddt_df is not None and sel_df is not None:
            merged = sel_df.merge(plddt_df, on='domain', how='left')
            colors = [DOMAIN_SRC_COLORS.get(r.get('source', 'other'),
                      (0.5, 0.5, 0.5)) for _, r in merged.iterrows()]
            labels = [shorten_domain_name(d, 15) for d in merged['domain']]
            x = np.arange(len(merged))
            ax.bar(x, merged['mean_plddt'], color=colors, edgecolor='white',
                   linewidth=0.3, width=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylim(0, 100)
        else:
            ax.text(0.5, 0.5, 'pLDDT data not available',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=6, color='gray')

    ax.axhline(y=70, color='gray', ls='--', lw=0.4, alpha=0.5)
    ax.annotate('70', xy=(1.0, 70), xycoords=('axes fraction', 'data'),
                fontsize=6, color='gray', va='center', ha='left',
                xytext=(2, 0), textcoords='offset points')
    ax.set_xlabel('Residue position')
    ax.set_ylabel('pLDDT')
    ax.set_ylim(20, 100)
    ax.text(0.97, 0.05, '10 domains, 7/10 above pLDDT 70',
            transform=ax.transAxes, fontsize=6, ha='right', va='bottom',
            color='#555555')


# ═════════════════════════════════════════════════════════════════════════════
# ROW 3 — Foldseek + E-value sensitivity detail (J-L)
# ═════════════════════════════════════════════════════════════════════════════

def draw_panel_foldseek(ax, sel_df, foldseek_df):
    """I: Foldseek TM-score heatmap (10 domains x 2 databases)."""
    if foldseek_df is None or len(foldseek_df) == 0 or sel_df is None:
        ax.text(0.5, 0.5, 'Foldseek data not available',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=6, color='gray')
        ax.set_xticks([]); ax.set_yticks([])
        return

    domains = sel_df['domain'].tolist()
    short_labels = [f'D{i+1}' for i in range(len(domains))]
    databases = ['AFDB', 'PDB']
    tm_matrix = np.zeros((len(domains), len(databases)))
    hit_labels = [[''] * 2 for _ in range(len(domains))]

    for i, domain in enumerate(domains):
        for j, db in enumerate(databases):
            match = foldseek_df[(foldseek_df['domain'] == domain) &
                                (foldseek_df['database'] == db)]
            if len(match) > 0:
                tm = float(match.iloc[0]['tm_score'])
                tm_matrix[i, j] = tm
                target = str(match.iloc[0]['best_target'])
                hit_labels[i][j] = (f'{tm:.2f}'
                                    if target != 'no_significant_hit'
                                    else '\u2014')
            else:
                hit_labels[i][j] = '\u2014'

    im = ax.imshow(tm_matrix, cmap=OCEAN_CMAP, aspect='auto', vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['AFDB', 'PDB'])
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(short_labels)

    for i in range(len(domains)):
        for j in range(2):
            ax.text(j, i, hit_labels[i][j], ha='center', va='center',
                    fontsize=6,
                    color='white' if tm_matrix[i, j] > 0.5 else 'black')

    cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04, aspect=25)
    cbar.set_label('TM-score', fontsize=6)
    cbar.ax.tick_params(labelsize=6, width=0.25, length=2)
    cbar.outline.set_linewidth(0.25)


def draw_panel_j(ax, summary):
    """J: Effect size distributions as violin/box plots."""
    detail_data = {}
    for label, fname in [
        ("Pfam", "correlations_pfam_baseline.tsv"),
        ("Novel_1e-05", "correlations_1e-05.tsv"),
        ("Novel_1e-07", "correlations_1e-07.tsv"),
        ("Novel_1e-09", "correlations_1e-09.tsv"),
    ]:
        path = SENSITIVITY_DIR / fname
        if path.exists():
            rho_vals = []
            with open(path) as f:
                header = f.readline().strip().split('\t')
                rho_idx = header.index('rho')
                for line in f:
                    fields = line.strip().split('\t')
                    if len(fields) > rho_idx:
                        try:
                            rho_vals.append(abs(float(fields[rho_idx])))
                        except ValueError:
                            pass
            if rho_vals:
                detail_data[label] = np.array(rho_vals)

    colors_map = {"Pfam": PFAM_COLOR, "Novel_1e-05": DEEP_OCEAN,
                  "Novel_1e-07": COASTAL_BLUE, "Novel_1e-09": TURQUOISE}
    display_map = {"Pfam": "Pfam", "Novel_1e-05": "Nov\n1e-5",
                   "Novel_1e-07": "Nov\n1e-7", "Novel_1e-09": "Nov\n1e-9"}

    if detail_data:
        data_used, labels_used, colors_used = [], [], []
        for label in ["Pfam", "Novel_1e-05", "Novel_1e-07", "Novel_1e-09"]:
            if label in detail_data:
                data_used.append(detail_data[label])
                labels_used.append(display_map[label])
                colors_used.append(colors_map[label])

        if data_used:
            parts = ax.violinplot(data_used, positions=range(len(data_used)),
                                  showextrema=False, widths=0.7)
            for i, body in enumerate(parts['bodies']):
                body.set_facecolor(colors_used[i])
                body.set_alpha(0.3)
                body.set_edgecolor(colors_used[i])

            ax.boxplot(data_used, positions=range(len(data_used)),
                       widths=0.3, showfliers=False,
                       medianprops=dict(color='black', linewidth=0.5),
                       boxprops=dict(linewidth=0.25),
                       whiskerprops=dict(linewidth=0.25),
                       capprops=dict(linewidth=0.25))

            ax.set_xticks(range(len(labels_used)))
            ax.set_xticklabels(labels_used)

            for i, d in enumerate(data_used):
                med = np.median(d)
                ax.text(i, med - 0.015, f'{med:.3f}', ha='center',
                        va='top', fontsize=6)
    else:
        med_values = summary['median_abs_rho'].values
        x = np.arange(len(summary))
        ax.bar(x, med_values, color=ALL_EVAL_COLORS, edgecolor='black',
               linewidth=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(['Pfam', 'Nov\n1e-5', 'Nov\n1e-7', 'Nov\n1e-9'])

    ax.set_ylabel('|rho| (Spearman)')


def draw_panel_k(ax, summary):
    """K: Fold-enrichment vs Pfam at each threshold."""
    novel = summary[summary['threshold'].str.startswith('Novel')].copy()
    x = np.arange(len(novel))
    w = 0.35

    ax.bar(x - w/2, novel['fold_vs_pfam_median'].values, w,
           color=NOVEL_COLORS, edgecolor='black', linewidth=0.25,
           label='Median |rho|')
    ax.bar(x + w/2, novel['fold_vs_pfam_pct_sig'].values, w,
           color=[c + (0.5,) if len(c) == 3 else c for c in NOVEL_COLORS],
           edgecolor=NOVEL_COLORS, linewidth=0.25, label='% significant')

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.25, zorder=0)
    ax.annotate('Pfam=1', xy=(1.0, 1.0),
                xycoords=mpl.transforms.blended_transform_factory(
                    ax.transAxes, ax.transData),
                xytext=(0, -3), textcoords='offset points',
                fontsize=6, color='gray', ha='right', va='top')

    ax.set_xticks(x)
    ax.set_xticklabels(['E<1e-5', 'E<1e-7', 'E<1e-9'])
    ax.set_ylabel('Fold vs Pfam')
    ax.legend(fontsize=6, frameon=False, loc='lower right')

    for bars in [ax.containers[0], ax.containers[1]]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03,
                        f'{h:.1f}x', ha='center', va='bottom', fontsize=6)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 70)
    print('Figure 7: Novel Domains — Consolidated (11 panels, A-K)')
    print()
    print('  Row 1: [A cluster-size] [B prevalence] [C coupling] [D scatter]')
    print('  Row 2: [E dark-frac] [F violins-by-source] [G sig-rates]')
    print('  Row 3: [H ──── pLDDT profiles (full width) ────]')
    print('  Row 4: [I Foldseek] [J effect-size violins] [K fold-enrichment]')
    print()
    print(f'  {datetime.now().isoformat()}')
    print('=' * 70)

    # ── Load data ────────────────────────────────────────────────────────
    sel_df = pd.read_csv(DATA_DIR / 'top10_selection_summary.tsv', sep='\t',
                         comment='#')
    plddt_df = (pd.read_csv(DATA_DIR / 'plddt_summary.tsv', sep='\t',
                            comment='#')
                if (DATA_DIR / 'plddt_summary.tsv').exists() else None)
    foldseek_df = (pd.read_csv(DATA_DIR / 'foldseek_best_hits.tsv', sep='\t',
                               comment='#')
                   if (DATA_DIR / 'foldseek_best_hits.tsv').exists() else None)
    eval_summary = pd.read_csv(DATA_DIR / 'evalue_sensitivity_comparison.tsv',
                               sep='\t')

    print(f'  {len(sel_df)} selected domains, pLDDT={plddt_df is not None}, '
          f'Foldseek={foldseek_df is not None}, E-val={len(eval_summary)} rows')

    # ── Layout: 7 x 7.5 in, 4 rows ─────────────────────────────────────
    fig = plt.figure(figsize=(7.0, 7.5))
    outer = GridSpec(4, 1, figure=fig,
                     height_ratios=[0.22, 0.19, 0.14, 0.27],
                     hspace=0.28,
                     left=0.10, right=0.95, top=0.98, bottom=0.05)

    # Collect panel axes explicitly (avoids colorbar indexing issues)
    panel_axes = []

    # Row 1: A-D (4 panels)
    gs_r1 = outer[0, 0].subgridspec(1, 4, wspace=0.50)
    print('  Row 1: A-D ...')
    for i, draw_fn in enumerate([draw_panel_a, draw_panel_b,
                                  draw_panel_c, draw_panel_d]):
        ax = fig.add_subplot(gs_r1[0, i])
        draw_fn(ax)
        panel_axes.append(ax)

    # Row 2: E-G (3 panels)
    gs_r2 = outer[1, 0].subgridspec(1, 3, wspace=0.45)
    print('  Row 2: E-G ...')
    for i, (draw_fn, extra) in enumerate([
        (draw_panel_e, ()), (draw_panel_f, ()),
        (draw_panel_h, (eval_summary,)),
    ]):
        ax = fig.add_subplot(gs_r2[0, i])
        draw_fn(ax, *extra)
        panel_axes.append(ax)

    # Row 3: H — pLDDT profiles (full width)
    gs_r3 = outer[2, 0].subgridspec(1, 1)
    print('  Row 3: H (pLDDT) ...')
    ax_h = fig.add_subplot(gs_r3[0, 0])
    draw_panel_plddt(ax_h, sel_df, plddt_df)
    panel_axes.append(ax_h)

    # Row 4: I-K (Foldseek + sensitivity detail)
    gs_r4 = outer[3, 0].subgridspec(1, 3, wspace=0.45,
                                      width_ratios=[1.0, 1.0, 1.0])
    print('  Row 4: I-K ...')
    ax_i = fig.add_subplot(gs_r4[0, 0])
    draw_panel_foldseek(ax_i, sel_df, foldseek_df)
    panel_axes.append(ax_i)
    ax_j = fig.add_subplot(gs_r4[0, 1])
    draw_panel_j(ax_j, eval_summary)
    panel_axes.append(ax_j)
    ax_k = fig.add_subplot(gs_r4[0, 2])
    draw_panel_k(ax_k, eval_summary)
    panel_axes.append(ax_k)

    # ── Panel labels (A-K on exactly the 11 panel axes) ──────────────────
    labels = 'ABCDEFGHIJK'
    ax_a_bbox = panel_axes[0].get_position()
    label_x_fig = ax_a_bbox.x0 - 0.02

    for letter, ax in zip(labels, panel_axes):
        if letter == 'H':
            # Full-width panel: align label with row 1 labels
            ax_pos = ax.get_position()
            x_axes = (label_x_fig - ax_pos.x0) / ax_pos.width
            ax.text(x_axes, 1.08, letter, transform=ax.transAxes,
                    fontsize=6, fontweight='bold', va='top', ha='left')
        else:
            ax.text(-0.12, 1.08, letter, transform=ax.transAxes,
                    fontsize=6, fontweight='bold', va='top', ha='left')

    # ── Global spine cleanup ─────────────────────────────────────────────
    for ax in fig.get_axes():
        if not ax.images:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # ── Export ────────────────────────────────────────────────────────────
    check_figure_size(fig, 'cell', 'double')

    for fmt in ['pdf', 'svg']:
        out = SCRIPT_DIR / f'Figure6_novel_domains_{TIMESTAMP}.{fmt}'
        fig.savefig(str(out), format=fmt, dpi=600,
                    transparent=True, edgecolor='none')
        print(f'  Saved: {out}')

    prov = SCRIPT_DIR / f'Figure6_novel_domains_{TIMESTAMP}_provenance.txt'
    with open(prov, 'w') as f:
        f.write('# Provenance -- Figure 7: Novel Domain Discovery '
                '(Consolidated)\n')
        f.write(f'Script: {Path(__file__).resolve()}\n')
        f.write(f'Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n')
        f.write('Panels A-D: Characterization (original Figure 7)\n')
        f.write('Panels E-F: Source breakdown (former Figure S6 A-B)\n')
        f.write('Panel G: Significance rates (former Figure S7 B)\n')
        f.write('Panel H: pLDDT profiles (former Figure S6 D)\n')
        f.write('Panel I: Foldseek heatmap (former Figure S6 E)\n')
        f.write('Panels J-K: E-value sensitivity detail '
                '(former Figure S7 C, D)\n')
    print(f'  Saved: {prov}')

    plt.close(fig)
    print('DONE')


if __name__ == '__main__':
    main()
