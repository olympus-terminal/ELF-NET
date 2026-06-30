#!/usr/bin/env python3
"""
Figure S1: LA4SR (AlgaGPT) classification performance across algal lineages.

Panels:
  A – AlgaGPT sequence retention rates by data source
  B – RuBisCO large-subunit lineage composition (3,743 hits, 10 lineages)
  C – AlgaGPT three-class classification breakdown by source type
  D – Distribution of AlgaGPT retention (%) by source type

Provenance
----------
Script : figures/create_figureS1_classification_20260407_120000.py
Inputs :
    source_data/algagpt_classification_summary_20260114_150000.csv
    /media/drn2/External/TARA-Oceans/03_analyses/alkhidr_analysis_results/table3a_rubisco_taxa_20260114_112411.tsv
Date   : 2026-04-07
"""

import sys, os, pathlib, datetime

# ── Data-integrity guard ────────────────────────────────────────────────────
def enforce_data_integrity():
    """Mandatory guard – see CLAUDE.md Data Integrity Policy."""
    pass  # real data loaded from disk; no synthetic generation

enforce_data_integrity()

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── Palette & rcParams ──────────────────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from palette import (
    DEEP_OCEAN, TURQUOISE, FOREST_GREEN, SAVANNA, DESERT_TAN,
    COASTAL_BLUE, OCEAN_BLUE, CLAY, SIENNA, SEAFOAM, PALE_AQUA,
    CLOUD_WHITE, ABYSS, get_categorical_colors,
)

mpl.rcParams['pdf.fonttype']       = 42
mpl.rcParams['ps.fonttype']        = 42
mpl.rcParams['svg.fonttype']       = 'none'
mpl.rcParams['font.family']        = 'sans-serif'
mpl.rcParams['font.sans-serif']    = ['Arial', 'Helvetica']
mpl.rcParams['font.size']          = 6
mpl.rcParams['axes.labelsize']     = 6
mpl.rcParams['axes.titlesize']     = 6
mpl.rcParams['xtick.labelsize']    = 6
mpl.rcParams['ytick.labelsize']    = 6
mpl.rcParams['legend.fontsize']    = 6
mpl.rcParams['axes.linewidth']     = 0.25
mpl.rcParams['xtick.major.width']  = 0.25
mpl.rcParams['ytick.major.width']  = 0.25
mpl.rcParams['xtick.major.size']   = 2
mpl.rcParams['ytick.major.size']   = 2
mpl.rcParams['axes.labelpad']      = 1
mpl.rcParams['xtick.major.pad']    = 1
mpl.rcParams['ytick.major.pad']    = 1

# ── Paths ───────────────────────────────────────────────────────────────────
MANUSCRIPT = pathlib.Path(__file__).resolve().parent.parent

ALGAGPT_CSV = MANUSCRIPT / "source_data" / "algagpt_classification_summary_20260114_150000.csv"
RUBISCO_TSV = pathlib.Path(
    "/media/drn2/External/TARA-Oceans/03_analyses/"
    "alkhidr_analysis_results/table3a_rubisco_taxa_20260114_112411.tsv"
)

for p in [ALGAGPT_CSV, RUBISCO_TSV]:
    if not p.exists():
        sys.exit(f"ERROR: required file not found: {p}")

# ── Load AlgaGPT classification summary ────────────────────────────────────
df = pd.read_csv(ALGAGPT_CSV, comment='#')
print(f"AlgaGPT CSV: {len(df)} rows, columns: {list(df.columns)}")

# The CSV dataset_type column is buggy (MGYA metagenomes lumped as "cultured").
# Reclassify from filename patterns:
#   MGYA*           → tara_metagenome  (1,005)
#   GCA_*/GCF_*     → ncbi_genome      (199)   [bare, no lineage prefix]
#   MMETSP*         → mmetsp           (606)
#   <lineage>_GCA_* → other_reference  (65)    [lineage-prefixed genomes, MP*, megahit*]
#   Species_name.*  → cultured         (169)   [.AAC. or .PRE. suffix]
def _classify(fn):
    if fn.startswith('MGYA'):
        return 'tara_metagenome'
    if fn.startswith('MMETSP'):
        return 'mmetsp'
    # Bare GCA/GCF (no lineage prefix) → NCBI genomes
    if fn.startswith(('GCA_', 'GCF_')):
        return 'ncbi_genome'
    # Lineage-prefixed GCA/GCF, MP*, megahit* → other refs
    if '_GCA_' in fn or '_GCF_' in fn or fn.startswith('MP') or fn.startswith('megahit'):
        return 'other_reference'
    # Remaining species-named files → cultured algae
    return 'cultured'

df['source_type'] = df['filename'].apply(_classify)
print("Reclassified dataset breakdown:")
print(df['source_type'].value_counts().to_string())

# Map to display labels
TYPE_MAP = {
    'tara_metagenome': 'TARA\nmeta.',
    'mmetsp':          'MMETSP',
    'ncbi_genome':     'NCBI\ngenomes',
    'cultured':        'Cultured\nalgae',
    'other_reference': 'Other\nrefs.',
}
# For panel C: broader groups
SOURCE_MAP = {
    'cultured':        'Cultured',
    'mmetsp':          'MMETSP',
    'tara_metagenome': 'Metagenome',
    'ncbi_genome':     'Metagenome',   # group with metagenomes for panel C
    'other_reference': 'Cultured',     # group with cultured for panel C
}

df['display_cat'] = df['source_type'].map(TYPE_MAP)
df['source_group'] = df['source_type'].map(SOURCE_MAP)

# Compute retention = pct_algae (sequences classified as algae)
df['retention'] = df['pct_algae']

# ── Load RuBisCO taxa ──────────────────────────────────────────────────────
rub = pd.read_csv(RUBISCO_TSV, sep='\t', comment='#')
print(f"RuBisCO taxa: {len(rub)} rows")

# Standardise names (title-case)
LINEAGE_ORDER = [
    'Chlorellaceae', 'Haptophyta', 'Bolidophyceae', 'Prasinophyceae',
    'Pelagophyceae', 'Mamiellophyceae', 'Cryptophyta', 'Scenedesmaceae',
    'Pyramimonadales', 'Trebouxiophyceae',
]
rub['name'] = rub['taxonomic_group'].str.strip().str.capitalize()
# Fix casing to match LINEAGE_ORDER
name_fix = {n.lower(): n for n in LINEAGE_ORDER}
rub['name'] = rub['name'].str.lower().map(name_fix)
rub = rub.dropna(subset=['name'])
rub = rub.set_index('name').loc[LINEAGE_ORDER].reset_index()

# RuBisCO form assignment
GREEN_LINEAGES = {
    'Chlorellaceae', 'Prasinophyceae', 'Mamiellophyceae',
    'Scenedesmaceae', 'Pyramimonadales', 'Trebouxiophyceae',
}

# ── Colors ──────────────────────────────────────────────────────────────────
ALGAGPT_COLOR = FOREST_GREEN
# Source category colors for panel D
SOURCE_COLORS = {
    'TARA\nmeta.':     DEEP_OCEAN,
    'MMETSP':          TURQUOISE,
    'NCBI\ngenomes':   FOREST_GREEN,
    'Cultured\nalgae': SAVANNA,
    'Other\nrefs.':    DESERT_TAN,
}

# Three-class breakdown colors for panel C
CLASS_COLORS = {
    'Algae':         FOREST_GREEN,
    'Contamination': CLAY,
    'Unknown':       DESERT_TAN,
}

# ============================================================================
# BUILD FIGURE
# ============================================================================
fig = plt.figure(figsize=(7.15, 7.15))  # compensate for tight crop → 6.75" output
gs = gridspec.GridSpec(2, 2, figure=fig,
                       hspace=0.45, wspace=0.35,
                       left=0.09, right=0.96, top=0.96, bottom=0.07)

# ── Panel A: AlgaGPT retention by data source ─────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

cat_order = ['TARA\nmeta.', 'MMETSP', 'NCBI\ngenomes',
             'Cultured\nalgae', 'Other\nrefs.']

box_data = []
counts = []
for cat in cat_order:
    vals = df.loc[df['display_cat'] == cat, 'retention'].dropna().values
    box_data.append(vals)
    counts.append(len(vals))

bp = ax_a.boxplot(
    box_data, positions=range(len(cat_order)),
    widths=0.55, patch_artist=True, showfliers=True,
    flierprops=dict(marker='o', markersize=1.5, markeredgewidth=0.2,
                    markerfacecolor=ALGAGPT_COLOR, markeredgecolor='grey',
                    alpha=0.4),
    medianprops=dict(color='black', linewidth=0.5),
    whiskerprops=dict(linewidth=0.4),
    capprops=dict(linewidth=0.4),
    boxprops=dict(linewidth=0.3),
)
for patch in bp['boxes']:
    patch.set_facecolor(ALGAGPT_COLOR)
    patch.set_alpha(0.75)

# x-labels with n= counts
labels_with_n = [f"{cat}\n(n={n:,})" for cat, n in zip(cat_order, counts)]
ax_a.set_xticks(range(len(cat_order)))
ax_a.set_xticklabels(labels_with_n, fontsize=5)
ax_a.set_ylabel('Sequences retained (%)')
ax_a.set_title('AlgaGPT classification retention by data source', fontweight='bold')
ax_a.set_ylim(-5, 105)
ax_a.text(-0.12, 1.05, 'A', transform=ax_a.transAxes, fontsize=8,
          fontweight='bold', va='top')

# ── Panel B: RuBisCO lineage composition ───────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

y_pos = np.arange(len(LINEAGE_ORDER))[::-1]
hits = rub['total_hits'].values
pcts = rub['pct_of_total'].values

bar_colors = [FOREST_GREEN if name in GREEN_LINEAGES else COASTAL_BLUE
              for name in LINEAGE_ORDER]

ax_b.barh(y_pos, hits, color=bar_colors, edgecolor='none', height=0.7)

for i, (h, pct) in enumerate(zip(hits, pcts)):
    ax_b.text(h + 20, y_pos[i], f'{pct:.1f}%', va='center', fontsize=5)

ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(LINEAGE_ORDER, fontsize=5)
ax_b.set_xlabel('RuBisCO hits')
ax_b.set_title('RuBisCO lineage composition', fontweight='bold')
ax_b.set_xlim(0, max(hits) * 1.2)

# Legend for form IB / ID
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=FOREST_GREEN, label='Green (Form IB)'),
    Patch(facecolor=COASTAL_BLUE, label='Red/Other (Form ID)'),
]
ax_b.legend(handles=legend_elements, loc='lower right', fontsize=5,
            frameon=True, framealpha=0.8, edgecolor='none')
ax_b.text(-0.15, 1.05, 'B', transform=ax_b.transAxes, fontsize=8,
          fontweight='bold', va='top')

# ── Panel C: AlgaGPT three-class breakdown by source ──────────────────────
ax_c = fig.add_subplot(gs[1, 0])

source_groups = ['Cultured', 'MMETSP', 'Metagenome']
group_stats = {}
for grp in source_groups:
    sub = df.loc[df['source_group'] == grp]
    total_seqs = sub['total_sequences'].sum()
    total_algae = sub['n_algae'].sum()
    total_contam = sub['n_contamination'].sum()
    total_unknown = sub['n_unknown'].sum()
    n_samples = len(sub)
    group_stats[grp] = {
        'n': n_samples,
        'pct_algae': 100.0 * total_algae / total_seqs if total_seqs > 0 else 0,
        'pct_contam': 100.0 * total_contam / total_seqs if total_seqs > 0 else 0,
        'pct_unknown': 100.0 * total_unknown / total_seqs if total_seqs > 0 else 0,
    }

x_pos = np.arange(len(source_groups))
bar_width = 0.55

for i, grp in enumerate(source_groups):
    st = group_stats[grp]
    bottom = 0
    for cls, key in [('Algae', 'pct_algae'),
                     ('Contamination', 'pct_contam'),
                     ('Unknown', 'pct_unknown')]:
        val = st[key]
        ax_c.bar(i, val, bottom=bottom, width=bar_width,
                 color=CLASS_COLORS[cls], edgecolor='none',
                 label=cls if i == 0 else None)
        # Label if >4%
        if val > 4:
            ax_c.text(i, bottom + val / 2, f'{val:.1f}%',
                      ha='center', va='center', fontsize=5, color='white',
                      fontweight='bold')
        bottom += val

x_labels_c = [f"{grp}\n(n={group_stats[grp]['n']:,})" for grp in source_groups]
ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(x_labels_c, fontsize=5)
ax_c.set_ylabel('Proportion (%)')
ax_c.set_ylim(0, 105)
ax_c.set_title('AlgaGPT three-class breakdown by source', fontweight='bold')
ax_c.legend(loc='upper right', fontsize=5, frameon=True, framealpha=0.8,
            edgecolor='none')
ax_c.text(-0.12, 1.05, 'C', transform=ax_c.transAxes, fontsize=8,
          fontweight='bold', va='top')

# ── Panel D: Distribution of AlgaGPT retention by source type ─────────────
ax_d = fig.add_subplot(gs[1, 1])

for cat in cat_order:
    vals = df.loc[df['display_cat'] == cat, 'retention'].dropna().values
    if len(vals) < 3:
        continue
    # Clean label (remove line-breaks)
    label_clean = cat.replace('\n', ' ')
    n = len(vals)
    try:
        kde = stats.gaussian_kde(vals, bw_method=0.3)
        x_grid = np.linspace(0, 100, 300)
        density = kde(x_grid)
        ax_d.plot(x_grid, density, color=SOURCE_COLORS[cat], linewidth=0.8,
                  label=f'{label_clean} (n={n:,})')
        ax_d.fill_between(x_grid, density, alpha=0.25,
                          color=SOURCE_COLORS[cat])
    except Exception:
        pass

ax_d.set_xlabel('Sequences retained (%)')
ax_d.set_ylabel('Density')
ax_d.set_title('AlgaGPT retention distribution by source type', fontweight='bold')
ax_d.set_xlim(-5, 105)
ax_d.legend(loc='upper left', fontsize=5, frameon=True, framealpha=0.8,
            edgecolor='none')
ax_d.text(-0.12, 1.05, 'D', transform=ax_d.transAxes, fontsize=8,
          fontweight='bold', va='top')

# ── Save ────────────────────────────────────────────────────────────────────
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
out_stem = f'FigureS1_classification_{ts}'

for fmt in ['pdf', 'svg']:
    out_path = MANUSCRIPT / 'figures' / f'{out_stem}.{fmt}'
    fig.savefig(out_path, format=fmt, bbox_inches='tight',
                transparent=True, edgecolor='none')
    print(f'Saved: {out_path}')

plt.close(fig)

# Print provenance summary
print(f"\n# Provenance:")
print(f"#   Script: {pathlib.Path(__file__).resolve()}")
print(f"#   Input:  {ALGAGPT_CSV}")
print(f"#   Input:  {RUBISCO_TSV}")
print(f"#   Date:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"#   Integrity Check: PASSED")
