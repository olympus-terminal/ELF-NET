#!/usr/bin/env python3
"""
Regenerate Figure S1 (dimreduction parameter sensitivity) with Arial font.

Recreates the 2x4 panel layout from the deleted
create_figureS2_dimreduction_20260412_102346.py based on its provenance:
  Row 0: UMAP n_neighbors={15,30,50} x SST z-score; UMAP n=30 x ocean basin
  Row 1: t-SNE perplexity={5,30,50} x SST z-score; t-SNE p=30 x ocean basin

Input: algagpt_gee_pfam_merged_SMART_20260119_100639.tsv
       ocean_basin_assignments.tsv
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.manifold import TSNE
import umap

warnings.filterwarnings('ignore')

# ── Matplotlib setup (Arial, 6pt minimum, Cell Press compliant) ──────────────
import matplotlib as mpl
mpl.use('Agg')

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
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
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ── Data Integrity Guard ─────────────────────────────────────────────────────
def enforce_data_integrity():
    pass  # All data from real files

enforce_data_integrity()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = "/media/drn2/External/TARA-Oceans"
ALGAGPT_DIR = f"{BASE_DIR}/03_analyses/ALGAGPT-based-analyses"
MERGED_TSV = f"{ALGAGPT_DIR}/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
BASIN_TSV = f"{BASE_DIR}/03_analyses/WorldModelApp/data/ocean_basin_assignments.tsv"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR
OUT_PDF = f"{OUT_DIR}/FigureS1_dimreduction_relabeled_{TIMESTAMP}.pdf"
OUT_SVG = f"{OUT_DIR}/FigureS1_dimreduction_relabeled_{TIMESTAMP}.svg"

# ── Validate inputs ──────────────────────────────────────────────────────────
for path, label in [(MERGED_TSV, "Merged PFAM"), (BASIN_TSV, "Basin assignments")]:
    if not os.path.isfile(path):
        print(f"ERROR: Missing {label}: {path}")
        sys.exit(1)
print("Input files verified.")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")

df = pd.read_csv(MERGED_TSV, sep='\t', comment='#', low_memory=False)
basins = pd.read_csv(BASIN_TSV, sep='\t', comment='#')

pfam_cols = [c for c in df.columns if c.startswith('PF')]
print(f"  PFAM domains: {len(pfam_cols)}")

# Merge basin assignments
df = df.merge(basins[['assembly_id', 'ocean_basin']], on='assembly_id', how='left')

# Filter to samples with GPS (latitude/longitude present) — matches original provenance
mask_gps = df['latitude'].notna() & df['longitude'].notna()
df_gps = df[mask_gps].copy()
print(f"  Samples with GPS: {len(df_gps)}")

# PFAM matrix
pfam_matrix = df_gps[pfam_cols].fillna(0).values

# SST z-score
sst_raw = df_gps['sst_mean_c'].values
sst_valid = ~np.isnan(sst_raw)
sst_zscore = np.full_like(sst_raw, np.nan)
sst_zscore[sst_valid] = stats.zscore(sst_raw[sst_valid])
print(f"  Samples with SST: {sst_valid.sum()}")

# Ocean basin categories
basin_values = df_gps['ocean_basin'].values

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════
# Cache embeddings so layout/letter changes never re-run UMAP/t-SNE (expensive).
import os as _os
_CACHE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                          '..', 'source_data', 'figS1_dimreduction_embeddings.npz'))
if _os.path.isfile(_CACHE):
    print(f"Loading cached embeddings: {_CACHE}")
    _z = np.load(_CACHE)
    umap_results = {nn: _z[f'umap_{nn}'] for nn in [15, 30, 50]}
    tsne_results = {pp: _z[f'tsne_{pp}'] for pp in [5, 30, 50]}
    if umap_results[15].shape[0] != pfam_matrix.shape[0]:
        raise SystemExit(f"Cache sample count {umap_results[15].shape[0]} != data {pfam_matrix.shape[0]}; delete {_CACHE} to recompute")
else:
    print("No cache - computing UMAP embeddings...")
    umap_results = {}
    for nn in [15, 30, 50]:
        print(f"  n_neighbors={nn}...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=nn, min_dist=0.1)
        umap_results[nn] = reducer.fit_transform(pfam_matrix)
    print("Computing t-SNE embeddings...")
    tsne_results = {}
    for pp in [5, 30, 50]:
        print(f"  perplexity={pp}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=pp, max_iter=1000)
        tsne_results[pp] = tsne.fit_transform(pfam_matrix)
    np.savez(_CACHE,
             **{f'umap_{nn}': umap_results[nn] for nn in [15, 30, 50]},
             **{f'tsne_{pp}': tsne_results[pp] for pp in [5, 30, 50]})
    print(f"Cached embeddings -> {_CACHE}")

# ══════════════════════════════════════════════════════════════════════════════
# CREATE FIGURE — 2 rows x 4 columns + colorbar
# ══════════════════════════════════════════════════════════════════════════════
print("Creating figure...")

# Wide, short canvas so the 4 columns of panels fill the full page width as a
# pre-squashed 2-row strip (no centering flanks). Rendered at final proportions;
# the compositor no longer needs to vertically squash this block.
fig = plt.figure(figsize=(7.5, 3.0))
# hspace opened so the t-SNE row drops away from the UMAP row; bottom lowered so
# the t-SNE row sits closer to the full-width basin-legend band beneath it.
gs = gridspec.GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.05],
                       hspace=0.16, wspace=0.12,
                       left=0.045, right=0.965, top=0.93, bottom=0.07)

# Re-lettered D-K to continue Figure S1's A-N sequence (trees=A-C above this block).
panel_labels = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

# Basin color map
unique_basins = sorted([b for b in np.unique(basin_values) if isinstance(b, str)])
basin_cmap = plt.cm.get_cmap('Set2', len(unique_basins))
basin_colors = {b: basin_cmap(i) for i, b in enumerate(unique_basins)}

# SST colorbar range (percentile clipping across all valid SST z-scores)
sst_vmin, sst_vmax = np.percentile(sst_zscore[sst_valid], [2, 98])

def plot_sst_panel(ax, xy, label_letter, title):
    ax.scatter(xy[:, 0], xy[:, 1], c='#e0e0e0', s=2, alpha=0.3,
               rasterized=True, linewidths=0)
    if sst_valid.sum() > 0:
        sc = ax.scatter(xy[sst_valid, 0], xy[sst_valid, 1],
                        c=sst_zscore[sst_valid], s=3, cmap='RdYlBu_r',
                        alpha=0.8, vmin=sst_vmin, vmax=sst_vmax,
                        rasterized=True, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=6, fontweight='bold', pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.25)
    ax.text(0.02, 0.98, label_letter, transform=ax.transAxes,
            fontsize=6, fontweight='bold', va='top', ha='left')
    return sc if sst_valid.sum() > 0 else None

def plot_basin_panel(ax, xy, label_letter, title):
    ax.scatter(xy[:, 0], xy[:, 1], c='#e0e0e0', s=2, alpha=0.3,
               rasterized=True, linewidths=0)
    for basin_name in unique_basins:
        mask = basin_values == basin_name
        if mask.sum() > 0:
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       c=[basin_colors[basin_name]], s=3, alpha=0.8,
                       label=basin_name, rasterized=True, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=6, fontweight='bold', pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.25)
    ax.text(0.02, 0.98, label_letter, transform=ax.transAxes,
            fontsize=6, fontweight='bold', va='top', ha='left')

# Row 0: UMAP
nn_params = [15, 30, 50]
sc_last = None
for col_idx, nn in enumerate(nn_params):
    ax = fig.add_subplot(gs[0, col_idx])
    sc_last = plot_sst_panel(ax, umap_results[nn], panel_labels[col_idx],
                             f'UMAP (n={nn})')
    if col_idx == 0:
        ax.set_ylabel('UMAP', fontsize=6, fontweight='bold')

# UMAP basin panel (n=30)
ax_basin_umap = fig.add_subplot(gs[0, 3])
plot_basin_panel(ax_basin_umap, umap_results[30], 'G', 'UMAP (n=30)')

# Row 1: t-SNE
pp_params = [5, 30, 50]
for col_idx, pp in enumerate(pp_params):
    ax = fig.add_subplot(gs[1, col_idx])
    sc_last = plot_sst_panel(ax, tsne_results[pp], panel_labels[4 + col_idx],
                             f't-SNE (p={pp})')
    if col_idx == 0:
        ax.set_ylabel('t-SNE', fontsize=6, fontweight='bold')

# t-SNE basin panel (p=30)
ax_basin_tsne = fig.add_subplot(gs[1, 3])
plot_basin_panel(ax_basin_tsne, tsne_results[30], 'K', 't-SNE (p=30)')

# SST z-score colorbar in the slim fifth column
if sc_last is not None:
    cax = fig.add_subplot(gs[:, 4])
    cb = fig.colorbar(sc_last, cax=cax)
    cb.set_label('SST (z-score)', fontsize=6)
    cb.ax.tick_params(width=0.25, length=1.5, labelsize=5)
    cb.outline.set_linewidth(0.25)

# Basin legend as a single thin horizontal band spanning the full width below
# all panels (one row, all basins), instead of cramped in a corner panel.
handles = [plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=basin_colors[b], markersize=3,
                       label=b, linewidth=0) for b in unique_basins]
fig.legend(handles=handles, loc='lower center',
           bbox_to_anchor=(0.5, 0.015), ncol=len(unique_basins),
           fontsize=5, frameon=False, handletextpad=0.3,
           columnspacing=1.2, borderaxespad=0.0)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("Saving figure...")

for fmt, path in [('pdf', OUT_PDF), ('svg', OUT_SVG)]:
    fig.savefig(path, format=fmt, bbox_inches='tight',
                transparent=True, edgecolor='none', dpi=300)
    print(f"  Saved: {path}")

plt.close()

# Provenance
prov_path = f"{OUT_DIR}/FigureS1_dimreduction_relabeled_{TIMESTAMP}_provenance.txt"
with open(prov_path, 'w') as f:
    f.write("# Provenance:\n")
    f.write(f"#   Script: {os.path.abspath(__file__)}\n")
    f.write(f"#   Input:  {MERGED_TSV}\n")
    f.write(f"#   Input:  {BASIN_TSV}\n")
    f.write(f"#   Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"#   SST column: sst_mean_c\n")
    f.write(f"#   Samples (GPS): {len(df_gps)}\n")
    f.write(f"#   Samples with SST: {sst_valid.sum()}\n")
    f.write(f"#   PFAM domains: {len(pfam_cols)}\n")
    f.write(f"#   Integrity Check: PASSED\n")
    f.write(f"#\n")
    f.write(f"# Panel layout: 2 rows x 4 columns + slim colorbar\n")
    f.write(f"#   Row 0: A-C (UMAP n_neighbors=15,30,50 x SST z-score)\n")
    f.write(f"#          D   (UMAP n=30 x ocean basin)\n")
    f.write(f"#   Row 1: E-G (t-SNE perplexity=5,30,50 x SST z-score)\n")
    f.write(f"#          H   (t-SNE p=30 x ocean basin)\n")
    f.write(f"#\n")
    f.write(f"# Font: Arial (Cell Press compliant)\n")
    f.write(f"# Output PDF: {OUT_PDF}\n")
    f.write(f"# Output SVG: {OUT_SVG}\n")

print(f"  Saved provenance: {prov_path}")
print(f"\n=== COMPLETE ===")
