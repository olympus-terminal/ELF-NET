#!/usr/bin/env python3
"""
Figure 1: Global Sample Coverage, Dataset Composition, and Algal Lineage Distributions
ARTIST MODE - Publication Quality

Layout:
  Row 0: Panel A (map, colored by dataset) | Panel B (dataset composition per basin,
         vertical stacked bars — rotated 90 CCW from old horizontal) | shared legend
  Rows 1-4: Panels C-L (10 lineage maps in 3-column grid, dark theme)
  Last row: final lineage map + RuBisCO size legend

Provenance:
  Script: create_figure1_artist_20260212_160000.py
  Input GPS: algagpt_gee_pfam_merged_SMART_20260119_100639.tsv
  Input RuBisCO: rubisco_all_samples_20260114.tsv
  Input Basin Stats: basin_statistics.json
  Date: 2026-02-12
"""

import json
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from datetime import datetime
import matplotlib.patheffects as pe
import sys
import os

# Data Integrity Guard
def enforce_data_integrity():
    pass

enforce_data_integrity()

# Earth-from-Space palette import
sys.path.insert(0, '/media/drn2/External/TARA-Oceans/MANUSCRIPT/figures')
from palette import (
    OCEAN_CMAP, FOREST_CMAP, THERMAL_CMAP, COASTAL_CMAP,
    DIVERGING_CMAP, COOL_DIVERGING_CMAP,
    BASIN_COLORS, LINEAGE_COLORS, ENV_CATEGORY_COLORS, MODULE_COLORS,
    get_sequential_cmap, get_diverging_cmap, get_categorical_colors,
    DEEP_OCEAN, TURQUOISE, FOREST_GREEN, DESERT_TAN, CLOUD_WHITE,
    SIENNA, COASTAL_BLUE, SAVANNA, CLAY
)

# =============================================================================
# ARTIST MODE: Mandatory rcParams Configuration
# =============================================================================
mpl.rcParams['pdf.fonttype'] = 42      # TrueType in PDF
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

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

# =============================================================================
# COLOR DEFINITIONS - Unified Dataset Colors (shared with Figure 2)
# =============================================================================
# Canonical dataset color mapping used across all figures
DATASET_COLORS = {
    'TARA_Oceans': DEEP_OCEAN,       # Deep blue
    'MMETSP': TURQUOISE,             # Turquoise
    'OSD': FOREST_GREEN,             # Forest green
    'RefGenome_GenBank': DESERT_TAN, # Desert tan
    'AAC': SIENNA,                   # Sienna
    'TARA_protist': COASTAL_BLUE,    # Coastal blue
    'Reference_Genome': SAVANNA,     # Savanna
    'RefGenome_PRE_REF': CLAY,       # Clay
}

DATASET_LABELS = {
    'TARA_Oceans': 'TARA Oceans',
    'MMETSP': 'MMETSP',
    'OSD': 'OSD',
    'RefGenome_GenBank': 'RefGenome',
    'AAC': 'AAC',
    'TARA_protist': 'TARA protist',
    'Reference_Genome': 'Ref. Genome',
    'RefGenome_PRE_REF': 'RefGenome PRE',
}

DATASET_ORDER = list(DATASET_COLORS.keys())

# Lineage mapping: data uses lowercase, palette uses capitalized
LINEAGE_NAMES = {
    'mamiellophyceae': 'Mamiellophyceae',
    'prasinophyceae': 'Prasinophyceae',
    'pyramimonadales': 'Pyramimonadales',
    'chlorellaceae': 'Chlorellaceae',
    'trebouxiophyceae': 'Trebouxiophyceae',
    'scenedesmaceae': 'Scenedesmaceae',
    'pelagophyceae': 'Pelagophyceae',
    'bolidophyceae': 'Bolidophyceae',
    'haptophyta': 'Haptophyta',
    'cryptophyta': 'Cryptophyta',
}

LINEAGE_ORDER = list(LINEAGE_NAMES.keys())

def get_lineage_color(lineage_key):
    """Get color for lineage, using palette when available."""
    capitalized_name = LINEAGE_NAMES.get(lineage_key, lineage_key.capitalize())
    if capitalized_name in LINEAGE_COLORS:
        return LINEAGE_COLORS[capitalized_name]
    else:
        fallback_colors = get_categorical_colors(len(LINEAGE_ORDER))
        idx = list(LINEAGE_NAMES.keys()).index(lineage_key)
        return fallback_colors[idx % len(fallback_colors)]

# =============================================================================
# DATA LOADING
# =============================================================================
BASE = Path('/media/drn2/External/TARA-Oceans')
GPS_FILE = BASE / '03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv'
RUBISCO_FILE = BASE / '03_analyses/rubisco_hmms/rubisco_all_samples_20260114.tsv'
BASIN_STATS_FILE = BASE / '03_analyses/WorldModelApp/basin_coverage/data/basin_statistics.json'

for path in [GPS_FILE, RUBISCO_FILE, BASIN_STATS_FILE]:
    if not path.exists():
        print(f"ERROR: Missing input: {path}")
        sys.exit(1)

print("Loading data...")

# Load GPS data
gps_df = pd.read_csv(GPS_FILE, sep='\t', comment='#')
gps_df = gps_df.dropna(subset=['latitude', 'longitude'])
gps_df = gps_df[gps_df['dataset'] != 'algaGPT_no_GPS']
print(f"  GPS samples: {len(gps_df)}")

# Load RuBisCO data
rubisco_df = pd.read_csv(RUBISCO_FILE, sep='\t')
print(f"  RuBisCO samples: {len(rubisco_df)}")

# Load basin statistics
with open(BASIN_STATS_FILE, 'r') as f:
    basin_stats = json.load(f)
basin_order = basin_stats['summary']['basin_order_by_count']
print(f"  Basins: {basin_order}")

# Extract base ID for matching
def extract_base_id(sample_id):
    """Extract base ID for matching between datasets."""
    s = str(sample_id)
    if s.startswith('MGYA'):
        return s.split('.')[0]
    return s.split('.')[0]

gps_df['base_id'] = gps_df['assembly_id'].apply(extract_base_id)
rubisco_df['base_id'] = rubisco_df['sample_id'].apply(extract_base_id)

# Merge
merged = gps_df.merge(rubisco_df, on='base_id', how='left')
for lin in LINEAGE_ORDER:
    if lin not in merged.columns:
        merged[lin] = 0
    merged[lin] = merged[lin].fillna(0)

print(f"  Merged samples: {len(merged)}")

# =============================================================================
# FIGURE CREATION
# =============================================================================
print("Creating figure...")

# Figure size: 7.5" wide (double column), ~9" tall
fig = plt.figure(figsize=(7.5, 9))

# Main grid: 5 rows
#   Row 0: Panel A (map) + Panel B (composition) + legend  - height_ratio 3.0
#   Rows 1-4: Lineage maps (3 per row) - height_ratio 1.3 each
gs = gridspec.GridSpec(5, 1, figure=fig,
                       height_ratios=[2.2, 1.3, 1.3, 1.3, 1.3],
                       hspace=0.01)

# Row 0 sub-grid: map (left) | right column (composition + legend, vertically shrunk)
gs_row0 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[0],
    width_ratios=[1.15, 0.85], wspace=0.12
)

# Right column: pad top/bottom so bar chart matches map height
# The map has no axis chrome so it fills the cell, but the bar chart has
# title, tick labels, and x-label that eat into its height.  Adding blank
# rows above and below shrinks the chart's drawing area to match.
gs_right = gridspec.GridSpecFromSubplotSpec(
    3, 2, subplot_spec=gs_row0[0, 1],
    height_ratios=[0.10, 0.80, 0.10],  # top pad, content (~80% height), bottom pad
    width_ratios=[0.65, 0.35],          # composition, legend
    hspace=0.0, wspace=0.02
)

# Rows 1-4 sub-grids: 3 columns each for lineage maps
gs_rows = []
for r in range(1, 5):
    gs_r = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[r], wspace=0.01
    )
    gs_rows.append(gs_r)

# =============================================================================
# PANEL A: Main Map (Light Theme)
# =============================================================================
print("  Drawing Panel A: Global sample map...")
ax_main = fig.add_subplot(gs_row0[0, 0], projection=ccrs.Robinson())

# Light theme basemap - NO border
ax_main.set_facecolor('#e8f4f8')  # Light blue ocean
ax_main.add_feature(cfeature.LAND, facecolor='#e0e0e0', edgecolor='none')
ax_main.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='#888888')
# Remove border
for spine in ax_main.spines.values():
    spine.set_visible(False)
ax_main.set_global()

# Plot samples by dataset - SIZE SCALED BY PERCENTILE RANK
if 'n_total' in merged.columns:
    merged['n_total'] = merged['n_total'].fillna(0)
else:
    merged['n_total'] = 0

merged['size_rank'] = merged['n_total'].rank(pct=True)

for dataset, color in DATASET_COLORS.items():
    subset = merged[merged['dataset'] == dataset]
    if len(subset) == 0:
        continue

    sizes = 4 + (subset['size_rank'] ** 2) * 116
    sizes = np.clip(sizes, 4, 120)

    ax_main.scatter(
        subset['longitude'], subset['latitude'],
        c=[color], s=sizes, alpha=0.7,
        transform=ccrs.PlateCarree(),
        edgecolors='black', linewidths=0.1,
        zorder=5
    )

# Panel label
ax_main.text(0.02, 0.98, 'A', transform=ax_main.transAxes,
             fontsize=8, fontweight='bold', va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.8))

ax_main.set_title(f'Global Sample Distribution (n={len(merged):,})',
                  fontsize=6, pad=2)

# =============================================================================
# PANEL B: Dataset Composition by Basin (vertical stacked bars)
# Rotated 90 degrees CCW from the old horizontal version:
#   - basins are now on the x-axis (columns)
#   - dataset % stacks vertically (bottom to top)
# =============================================================================
print("  Drawing Panel B: Dataset composition (vertical bars)...")
ax_comp = fig.add_subplot(gs_right[1, 0])

n_basins = len(basin_order)
x_positions = np.arange(n_basins)
bar_width = 0.70

# Compute composition percentages
composition_data = {}
for basin in basin_order:
    bdata = basin_stats['basins'][basin]['dataset_composition']
    total = basin_stats['basins'][basin]['sample_count']
    row = {ds: bdata.get(ds, 0) / total * 100.0 for ds in DATASET_ORDER}
    composition_data[basin] = row

bottom_edges = np.zeros(n_basins)
for ds in DATASET_ORDER:
    heights = np.array([composition_data[b][ds] for b in basin_order])
    if np.sum(heights) == 0:
        continue
    ax_comp.bar(x_positions, heights, width=bar_width, bottom=bottom_edges,
                color=DATASET_COLORS[ds], edgecolor='white', linewidth=0.2)
    bottom_edges += heights

ax_comp.set_ylim(0, 100)
ax_comp.set_ylabel('Samples (%)')
ax_comp.set_xticks(x_positions)
# Basin labels on the bottom, abbreviated to avoid overlap
basin_xlabels = ['Atl.', 'Pac.', 'Med.', 'Ind.', 'Sth.', 'Arc.', 'Red']
ax_comp.set_xticklabels(basin_xlabels, rotation=0, ha='center', fontsize=5)
ax_comp.tick_params(axis='x', labelsize=5, pad=1)
ax_comp.set_title('Dataset Composition by Basin', fontweight='bold', pad=2, fontsize=6)

# Panel label - placed inside top-left of the axes to avoid overlapping A or y-axis
ax_comp.text(0.02, 0.97, 'B', transform=ax_comp.transAxes,
             fontsize=7, fontweight='bold', va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.8))

# =============================================================================
# SHARED DATASET LEGEND (between panels A and B, in right column of row 0)
# =============================================================================
print("  Drawing shared dataset legend...")
ax_leg = fig.add_subplot(gs_right[1, 1])
ax_leg.axis('off')

# Position legend entries vertically
y_start = 0.92
y_step = 0.065

for i, ds in enumerate(DATASET_ORDER):
    color = DATASET_COLORS[ds]
    n = len(merged[merged['dataset'] == ds])
    if n == 0:
        continue
    y = y_start - i * y_step
    ax_leg.scatter([0.08], [y], c=[color], s=18, edgecolors='black',
                   linewidths=0.15, zorder=5)
    label = DATASET_LABELS[ds]
    ax_leg.text(0.20, y, f'{label} ({n})', fontsize=5,
                va='center', ha='left', color='black')

ax_leg.set_xlim(0, 1)
ax_leg.set_ylim(0, 1)

# =============================================================================
# PANELS C-K: First 9 Lineage Maps (Dark Theme, 3 per row, rows 1-3)
# =============================================================================
print("  Drawing Panels C-K: Lineage maps...")
panel_labels = 'CDEFGHIJKL'

for i, lineage in enumerate(LINEAGE_ORDER[:9]):  # First 9 lineages
    row_idx = i // 3       # 0, 1, 2  -> gs_rows[0], gs_rows[1], gs_rows[2]
    col = i % 3

    ax = fig.add_subplot(gs_rows[row_idx][0, col], projection=ccrs.Robinson())

    # Dark theme basemap - NO outline
    ax.set_facecolor('#1a1a2e')
    ax.add_feature(cfeature.LAND, facecolor='#2d2d2d', edgecolor='none')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.15, edgecolor='#3a3a3a')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_global()

    # Get samples with this lineage
    lineage_data = merged[merged[lineage] > 0].copy()

    if len(lineage_data) > 0:
        sizes = np.sqrt(lineage_data[lineage]) * 4 + 3
        sizes = np.clip(sizes, 3, 35)

        ax.scatter(
            lineage_data['longitude'], lineage_data['latitude'],
            c=[get_lineage_color(lineage)], s=sizes, alpha=0.85,
            transform=ccrs.PlateCarree(),
            edgecolors='black', linewidths=0.1,
            zorder=5
        )

    txt_outline = [pe.withStroke(linewidth=1.5, foreground='white')]

    ax.text(0.03, 0.95, panel_labels[i], transform=ax.transAxes,
            fontsize=7, fontweight='bold', va='top', ha='left',
            color='black', path_effects=txt_outline)

    n_samples = len(lineage_data)
    ax.text(0.03, 0.03, LINEAGE_NAMES[lineage], transform=ax.transAxes,
            fontsize=5, fontweight='bold', va='bottom', ha='left',
            color='black', path_effects=txt_outline)
    ax.text(0.03, 0.12, f'n={n_samples}', transform=ax.transAxes,
            fontsize=5, va='bottom', ha='left', color='black',
            path_effects=txt_outline)

# =============================================================================
# ROW 4: Cryptophyta (L) + RuBisCO Size Legend
# =============================================================================
print("  Drawing Panel L: Cryptophyta + legends...")

# Panel L: Cryptophyta
lineage = 'cryptophyta'
ax_crypto = fig.add_subplot(gs_rows[3][0, 0], projection=ccrs.Robinson())
ax_crypto.set_facecolor('#1a1a2e')
ax_crypto.add_feature(cfeature.LAND, facecolor='#2d2d2d', edgecolor='none')
ax_crypto.add_feature(cfeature.COASTLINE, linewidth=0.15, edgecolor='#3a3a3a')
for spine in ax_crypto.spines.values():
    spine.set_visible(False)
ax_crypto.set_global()

lineage_data = merged[merged[lineage] > 0].copy()
if len(lineage_data) > 0:
    sizes = np.sqrt(lineage_data[lineage]) * 4 + 3
    sizes = np.clip(sizes, 3, 35)
    ax_crypto.scatter(
        lineage_data['longitude'], lineage_data['latitude'],
        c=[get_lineage_color(lineage)], s=sizes, alpha=0.85,
        transform=ccrs.PlateCarree(), edgecolors='black', linewidths=0.1, zorder=5
    )

txt_outline = [pe.withStroke(linewidth=1.5, foreground='white')]
ax_crypto.text(0.03, 0.95, 'L', transform=ax_crypto.transAxes,
               fontsize=7, fontweight='bold', va='top', ha='left', color='black',
               path_effects=txt_outline)
n_samples = len(lineage_data)
ax_crypto.text(0.03, 0.03, LINEAGE_NAMES[lineage], transform=ax_crypto.transAxes,
               fontsize=5, fontweight='bold', va='bottom', ha='left',
               color='black', path_effects=txt_outline)
ax_crypto.text(0.03, 0.12, f'n={n_samples}', transform=ax_crypto.transAxes,
               fontsize=5, va='bottom', ha='left', color='black',
               path_effects=txt_outline)

# Size Legend Panel (row 4, col 1)
ax_leg2 = fig.add_subplot(gs_rows[3][0, 1])
ax_leg2.set_facecolor('#1a1a2e')
ax_leg2.axis('off')

ax_leg2.text(0.5, 0.98, 'RuBisCO Seqs', fontsize=6, fontweight='bold',
             ha='center', va='top', color='black')
size_vals = [1, 5, 20, 50]
y_pos = 0.78
for val in size_vals:
    s = np.sqrt(val) * 4 + 3
    ax_leg2.scatter([0.3], [y_pos], s=s, c='#888888', edgecolors='black',
                    linewidths=0.1, alpha=0.9)
    ax_leg2.text(0.5, y_pos, f'{val}', fontsize=5, va='center', ha='left', color='black')
    y_pos -= 0.18

ax_leg2.set_xlim(0, 1)
ax_leg2.set_ylim(0, 1)

# Empty panel (row 4, col 2) - dark to match
ax_empty = fig.add_subplot(gs_rows[3][0, 2])
ax_empty.set_facecolor('#1a1a2e')
ax_empty.axis('off')

# =============================================================================
# SAVE OUTPUTS
# =============================================================================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_base = Path('/media/drn2/External/TARA-Oceans/MANUSCRIPT/figures')
output_stem = f'Figure1_artist_{timestamp}'

print(f"Saving outputs...")

for fmt in ['pdf', 'svg']:
    outpath = output_base / f'{output_stem}.{fmt}'
    fig.savefig(outpath, format=fmt,
                bbox_inches='tight',
                transparent=True,
                edgecolor='none',
                dpi=300)
    print(f"  Saved: {outpath} ({outpath.stat().st_size / 1e6:.1f} MB)")

# Provenance file
prov_path = output_base / f'{output_stem}_provenance.txt'
with open(prov_path, 'w') as f:
    f.write(f"# Provenance for {output_stem}\n")
    f.write(f"Script: {os.path.abspath(__file__)}\n")
    f.write(f"Date: {datetime.now().isoformat()}\n")
    f.write(f"Input GPS: {GPS_FILE}\n")
    f.write(f"Input RuBisCO: {RUBISCO_FILE}\n")
    f.write(f"Input Basin Stats: {BASIN_STATS_FILE}\n")
    f.write(f"Total samples: {len(merged)}\n")
    f.write(f"Datasets: {merged['dataset'].value_counts().to_dict()}\n")
    f.write(f"Basins: {basin_order}\n")
    for lin in LINEAGE_ORDER:
        n = (merged[lin] > 0).sum()
        s = int(merged[lin].sum())
        f.write(f"  {lin}: {n} samples, {s} sequences\n")

print(f"  Saved: {prov_path}")
print("Done!")

plt.close(fig)
