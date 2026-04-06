#!/usr/bin/env python3
"""
Create supplemental figure: AlgaGPT retention extremes.

Panel A: Top 20 cleanest assemblies (highest % algal retention)
         — full 0–100% x-axis so bars visually dominate
Panel B: Top 20 most-cleaned assemblies (most sequences REMOVED in absolute terms)
         — stacked bar: algal (colored) + removed (grey)

Shows that well-curated reference genomes achieve >99% retention while
metagenome assemblies undergo heavy filtering (up to 674K seqs removed).

Artist mode: 6pt Arial, 0.25pt lines, transparent, PDF+SVG
"""

import os
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# -- Matplotlib setup ----------------------------------------------------------
import matplotlib as mpl
mpl.use('Agg')

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['axes.titlesize'] = 7
mpl.rcParams['xtick.labelsize'] = 5.5
mpl.rcParams['ytick.labelsize'] = 5.5
mpl.rcParams['legend.fontsize'] = 5
mpl.rcParams['axes.linewidth'] = 0.25
mpl.rcParams['xtick.major.width'] = 0.25
mpl.rcParams['ytick.major.width'] = 0.25
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['axes.labelpad'] = 1
mpl.rcParams['xtick.major.pad'] = 1
mpl.rcParams['ytick.major.pad'] = 1

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

# -- Config --------------------------------------------------------------------
MANUSCRIPT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(MANUSCRIPT, 'source_data',
                        'algagpt_classification_summary_20260114_150000.csv')
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Source type colors (colorblind-friendly)
SOURCE_COLORS = {
    'Cultured reference':  '#2166ac',  # blue
    'NCBI reference':      '#4393c3',  # light blue
    'MMETSP transcriptome':'#f4a582',  # salmon
    'TARA metagenome':     '#b2182b',  # red
    'Cryptophyta ref':     '#762a83',  # purple
    'Other reference':     '#878787',  # grey
}

# -- Parse source type from filename -------------------------------------------
def classify_source(fn):
    if fn.startswith('MMETSP') or fn.startswith('MP0'):
        return 'MMETSP transcriptome'
    if fn.startswith('MGYA'):
        return 'TARA metagenome'
    if fn.startswith('cryptophyta_'):
        return 'Cryptophyta ref'
    if '.AAC.' in fn or '.PRE.' in fn:
        return 'Cultured reference'
    if fn.startswith('GCA_') or fn.startswith('GCF_'):
        return 'NCBI reference'
    if fn.startswith('euglenozoa_'):
        return 'Other reference'
    return 'Other reference'

# Organism names for GCA/GCF accessions in the top-20 (from NCBI)
GCA_NAMES = {
    'GCA_014080715.1': 'Scenedesmus sp.',
    'GCA_048987025.1': 'Coelastrum sp. PABB002',
    'GCA_008037345.1': 'Messastrum gracile',
    'GCA_014905635.1': 'Scenedesmus sp. PABB004',
    'GCA_004335835.1': 'Scenedesmus sp. ARA3',
    'GCA_052674805.1': 'Scenedesmus obliquus',
    'GCA_043380735.1': 'Tetradesmus obliquus',
    'GCA_051107615.1': 'Scenedesmus bijugus',
}

def clean_name(fn, source=None):
    """Make a readable label from filename."""
    name = fn.replace('_algagpt.tsv', '').replace('.aa', '')
    # Trim long trinity suffixes
    name = re.sub(r'\.trinity_out_2\.2\.0\.Trinitysta\.transdecoder', '', name)
    # Trim _contigs / _assembly suffix
    name = re.sub(r'_(contigs|assembly)$', '', name)
    # Trim _genomic suffix
    name = name.replace('_genomic', '')
    # For GCA/GCF: resolve to organism name if known
    m = re.match(r'(GC[AF]_\d+\.\d+)', name)
    if m and m.group(1) in GCA_NAMES:
        return GCA_NAMES[m.group(1)]
    # Otherwise keep accession, trim assembly name suffix
    name = re.sub(r'(GC[AF]_\d+\.\d+)_\S+', r'\1', name)
    # For organism names: underscores → spaces
    if not name.startswith(('GCA', 'GCF', 'MGYA', 'MMETSP', 'MP0', 'cryptophyta')):
        name = name.replace('_', ' ').replace('.', ' ')
        name = re.sub(r'\s+', ' ', name).strip()
    return name

# -- Load data -----------------------------------------------------------------
df = pd.read_csv(CSV_PATH, comment='#')
df['source'] = df['filename'].apply(classify_source)
df['label'] = df.apply(lambda r: clean_name(r['filename'], r['source']), axis=1)
df['n_removed'] = df['total_sequences'] - df['n_algae']
df['pct_removed'] = 100.0 - df['pct_algae']

# Filter out tiny assemblies (nucleomorphs, bacterial contaminants <1000 seqs)
df_substantial = df[df['total_sequences'] >= 1000].copy()

# -- Panel A: Top 20 cleanest (highest % retention) ---------------------------
top_clean = df_substantial.nlargest(20, 'pct_algae').iloc[::-1]

# -- Panel B: Top 20 lowest retention (by %) ----------------------------------
top_dirty = df_substantial.nsmallest(20, 'pct_algae').iloc[::-1]

# -- Summary stats -------------------------------------------------------------
ref_genomes = df[df['source'].isin(['Cultured reference', 'NCBI reference',
                                     'Other reference', 'Cryptophyta ref'])]
metagenomes = df[df['source'] == 'TARA metagenome']
transcriptomes = df[df['source'] == 'MMETSP transcriptome']

# -- Create figure -------------------------------------------------------------
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.35),
                                  gridspec_kw={'wspace': 0.45,
                                               'left': 0.18, 'right': 0.92})

# ============================================================================
# Panel A: Cleanest assemblies — full 0–100% scale
# ============================================================================
colors_a = [SOURCE_COLORS[s] for s in top_clean['source']]
ax_a.barh(range(len(top_clean)), top_clean['pct_algae'].values,
          color=colors_a, edgecolor='none', height=0.7, zorder=2)

ax_a.set_yticks(range(len(top_clean)))
ax_a.set_yticklabels(top_clean['label'].values, fontsize=5)
ax_a.set_xlabel('Sequences classified as algal (%)')
ax_a.set_xlim(0, 100)
ax_a.set_title('A  Top 20 highest-retention assemblies',
                fontweight='bold', loc='left', fontsize=7)

# Annotate with percentage + n
for i, (_, row) in enumerate(top_clean.iterrows()):
    ax_a.text(row['pct_algae'] - 1, i, f"{row['pct_algae']:.1f}%",
              va='center', ha='right', fontsize=4.5, color='white',
              fontweight='bold')

# Cross-panel context: show where the lowest-retention assemblies fall
dirty_mean = top_dirty['pct_algae'].mean()
ax_a.axvline(x=dirty_mean, color='#b2182b', linewidth=0.6, linestyle='--',
             zorder=1, alpha=0.6)
ax_a.annotate(f'Lowest-retention 20\nmean: {dirty_mean:.0f}%',
              xy=(dirty_mean, 0), xycoords=('data', 'axes fraction'),
              xytext=(0, -18), textcoords='offset points',
              fontsize=4.5, color='#b2182b', ha='center', va='top',
              annotation_clip=False,
              arrowprops=dict(arrowstyle='-', color='#b2182b', lw=0.3, alpha=0.5))

# ============================================================================
# Panel B: Lowest-retention assemblies — % retained (colored) + % removed (grey)
# ============================================================================
colors_b = [SOURCE_COLORS[s] for s in top_dirty['source']]
# Retained portion (colored)
ax_b.barh(range(len(top_dirty)), top_dirty['pct_algae'].values,
          color=colors_b, edgecolor='none', height=0.7, zorder=2)
# Removed portion (grey)
ax_b.barh(range(len(top_dirty)), top_dirty['pct_removed'].values,
          left=top_dirty['pct_algae'].values,
          color='#d9d9d9', edgecolor='none', height=0.7, zorder=1)

ax_b.set_yticks(range(len(top_dirty)))
ax_b.set_yticklabels(top_dirty['label'].values, fontsize=5)
ax_b.set_xlabel('Sequences classified as algal (%)')
ax_b.set_xlim(0, 100)
ax_b.set_title('B  Top 20 lowest-retention assemblies',
                fontweight='bold', loc='left', fontsize=7)

# Annotate with % + sequence count
for i, (_, row) in enumerate(top_dirty.iterrows()):
    ax_b.text(row['pct_algae'] + 1, i, f"{row['pct_algae']:.1f}%",
              va='center', ha='left', fontsize=4.5, color='#333333')
    ax_b.annotate(f"n={row['total_sequences']:,}",
                  xy=(1.02, i), xycoords=('axes fraction', 'data'),
                  va='center', ha='left', fontsize=4, color='#888888',
                  annotation_clip=False)

# Small legend for grey = removed
from matplotlib.patches import Patch as _P
ax_b.legend(handles=[_P(facecolor='#d9d9d9', edgecolor='none', label='Removed (non-algal)')],
            loc='upper right', frameon=True, framealpha=0.9,
            edgecolor='#cccccc', fontsize=5, handlelength=1)

# ============================================================================
# Shared source-type legend at bottom
# ============================================================================
all_sources = set(top_clean['source']) | set(top_dirty['source'])
legend_elements = [Patch(facecolor=SOURCE_COLORS[s], edgecolor='none', label=s)
                   for s in SOURCE_COLORS if s in all_sources]
fig.legend(handles=legend_elements, loc='lower center',
           ncol=len(legend_elements), frameon=False, fontsize=5,
           bbox_to_anchor=(0.5, -0.03))

# ============================================================================
# Summary banner
# ============================================================================
clean_mean_val = top_clean['pct_algae'].mean()
dirty_mean_val = top_dirty['pct_algae'].mean()
summary = (
    f"Cleanest 20: {clean_mean_val:.1f}% mean retention (all reference genomes)"
    f"     vs.     "
    f"Lowest-retention 20: {dirty_mean_val:.1f}% mean retention (metagenomes + transcriptomes)"
)
fig.text(0.5, 1.02, summary, ha='center', va='bottom', fontsize=5.5,
         transform=fig.transFigure, style='italic', color='#444444')

# ============================================================================
# Save
# ============================================================================
for ext in ('pdf', 'svg'):
    outpath = os.path.join(MANUSCRIPT, 'supplement',
                           f'FigureS_retention_extremes_{TIMESTAMP}.{ext}')
    fig.savefig(outpath, dpi=300, bbox_inches='tight', transparent=True)
    print(f'Saved: {outpath}')

plt.close(fig)
print('Done.')
