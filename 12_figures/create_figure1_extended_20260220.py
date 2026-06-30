#!/usr/bin/env python3
"""
Figure 1 Extended: Global Sample Coverage and Algal Lineage Distributions
Includes Form I (10 lineages) and Form II myzozoan (5 lineages)

Layout:
  - Panel A: Large light-theme map (all samples, colored by dataset)
  - Panel B: Basin composition stacked bars
  - Panels C-L: 10 Form I lineage maps (dark theme)
  - Panels M-Q: 5 Form II lineage maps (dark theme)
  - Legend panel

Provenance:
  Script: create_figure1_extended_20260220.py
  Input GPS: source_data/ASSEMBLY_GPS_MASTER_KEY_20260119_101500.tsv
  Input RuBisCO: rubisco_merged_formI_formII_20260223.tsv (corrected Form I + Form II)
  Date: 2026-02-23
"""

import socket
from pathlib import Path

def get_base_dir(project_name: str) -> Path:
    hpc = Path(f"/scratch/drn2/PROJECTS/{project_name}")
    if hpc.exists():
        return hpc
    local_ext = Path("/media/drn2/External/TARA-Oceans")
    if local_ext.exists():
        return local_ext
    local_ext1 = Path("/media/drn/External1/TARA-Oceans")
    if local_ext1.exists():
        return local_ext1
    return Path(f"/home/drn/Documents/projects/TARA-OMEN/MANUSCRIPT")

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import sys

# =============================================================================
# ARTIST MODE: Mandatory rcParams
# =============================================================================
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
# COLOR DEFINITIONS
# =============================================================================

# Earth-from-Space primary colors
ABYSS = (0.004, 0.098, 0.220)
DEEP_OCEAN = (0.031, 0.188, 0.420)
OCEAN_BLUE = (0.129, 0.400, 0.545)
COASTAL_BLUE = (0.255, 0.573, 0.612)
TURQUOISE = (0.369, 0.678, 0.651)
SEAFOAM = (0.498, 0.757, 0.718)
PALE_AQUA = (0.725, 0.875, 0.839)
FOREST_GREEN = (0.180, 0.420, 0.239)
SAVANNA = (0.420, 0.533, 0.290)
PALE_GREEN = (0.682, 0.800, 0.565)
DESERT_TAN = (0.678, 0.600, 0.420)

# Dataset colors — three-category framework (matches manuscript):
#   Metagenome    — TARA Oceans + OSD ocean metagenome assemblies
#                   (TARA_Oceans, TARA_protist, and OSD are collapsed at load time)
#   Transcriptome — MMETSP marine eukaryotic transcriptomes
#   Cultured ref. — NCBI GenBank/RefSeq, Phytozome, culture-collection assemblies
#                   (RefGenome_* variants, Reference_Genome, AAC are collapsed
#                    at load time)
DATASET_COLORS = {
    'Metagenome':    DEEP_OCEAN,
    'Transcriptome': TURQUOISE,
    'Cultured ref.': FOREST_GREEN,
}

# Form I lineage colors (cool palette)
FORM_I_COLORS = {
    'mamiellophyceae': SAVANNA,
    'prasinophyceae': PALE_GREEN,
    'pyramimonadales': SEAFOAM,
    'chlorellaceae': FOREST_GREEN,
    'trebouxiophyceae': DEEP_OCEAN,
    'scenedesmaceae': (0.35, 0.55, 0.35),
    'pelagophyceae': OCEAN_BLUE,
    'bolidophyceae': COASTAL_BLUE,
    'haptophyta': TURQUOISE,
    'cryptophyta': PALE_AQUA,
}

# Form II lineage colors (warm palette — fire tones)
FORM_II_COLORS = {
    'symbiodiniaceae': (0.85, 0.35, 0.25),   # Coral
    'peridiniales': (0.80, 0.55, 0.15),       # Amber
    'gonyaulacales': (0.70, 0.15, 0.20),      # Crimson
    'prorocentrales': (0.85, 0.70, 0.20),     # Gold
    'chromerida': (0.60, 0.30, 0.15),         # Rust
}

# Display names
LINEAGE_DISPLAY = {
    # Form I
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
    # Form II
    'symbiodiniaceae': 'Symbiodiniaceae',
    'peridiniales': 'Peridiniales',
    'gonyaulacales': 'Gonyaulacales',
    'prorocentrales': 'Prorocentrales',
    'chromerida': 'Chromerida',
}

FORM_I_ORDER = [
    'mamiellophyceae', 'prasinophyceae', 'pyramimonadales',
    'chlorellaceae', 'trebouxiophyceae', 'scenedesmaceae',
    'pelagophyceae', 'bolidophyceae', 'haptophyta', 'cryptophyta',
]

FORM_II_ORDER = [
    'symbiodiniaceae', 'peridiniales', 'gonyaulacales',
    'prorocentrales', 'chromerida',
]

ALL_LINEAGES = FORM_I_ORDER + FORM_II_ORDER
ALL_COLORS = {**FORM_I_COLORS, **FORM_II_COLORS}

# Dataset display order and labels (three-category framework)
DATASET_ORDER = list(DATASET_COLORS.keys())
DATASET_LABELS = {
    'Metagenome':    'Metagenome (TARA + OSD)',
    'Transcriptome': 'Transcriptome (MMETSP)',
    'Cultured ref.': 'Cultured ref.',
}

# =============================================================================
# BASIN ASSIGNMENT from GPS coordinates
# =============================================================================
BASIN_ORDER = ['Atlantic', 'Pacific', 'Mediterranean', 'Indian', 'Southern', 'Arctic', 'Red Sea']
BASIN_XLABELS = ['Atl.', 'Pac.', 'Med.', 'Ind.', 'Sth.', 'Arc.', 'Red']

def assign_basin(lat, lon):
    """Assign an ocean basin based on latitude/longitude."""
    # Southern Ocean: south of -60
    if lat < -60:
        return 'Southern'
    # Arctic: north of 66
    if lat > 66:
        return 'Arctic'
    # Mediterranean: 30-46N, -6 to 36E
    if 30 <= lat <= 46 and -6 <= lon <= 36:
        return 'Mediterranean'
    # Red Sea: 12-30N, 32-44E
    if 12 <= lat <= 30 and 32 <= lon <= 44:
        return 'Red Sea'
    # Indian Ocean: south of 30N, 20-146.5E (east of Africa, west of Australia)
    if lat < 30 and 20 <= lon <= 146.5:
        # Exclude Mediterranean overlap
        if not (lat > 30 and lon < 36):
            return 'Indian'
    # Pacific: east of 146.5E or west of -70W (Americas)
    if lon > 146.5 or lon < -70:
        if lat > -60:
            return 'Pacific'
    # Atlantic: between -70W and 20E (broadly)
    if -70 <= lon <= 20:
        return 'Atlantic'
    # Fallback
    return 'Atlantic'

# =============================================================================
# DATA LOADING
# =============================================================================
BASE = get_base_dir("TARA-LA4SR")

# Paths — try MANUSCRIPT source_data first, then HPC paths
GPS_CANDIDATES = [
    Path(__file__).parent / 'ASSEMBLY_GPS_MASTER_KEY_20260119_101500.tsv',
    BASE / 'source_data' / 'ASSEMBLY_GPS_MASTER_KEY_20260119_101500.tsv',
    BASE / 'omen-work' / 'ASSEMBLY_GPS_MASTER_KEY_20260119_101500.tsv',
    BASE / '03_analyses/ALGAGPT-based-analyses/ASSEMBLY_GPS_MASTER_KEY_20260119_101500.tsv',
]
RUBISCO_CANDIDATES = [
    Path(__file__).parent / 'rubisco_merged_formI_formII_20260223.tsv',
    Path(__file__).parent / 'rubisco_extended_combined_20260218_081615.tsv',
    BASE / 'source_data' / 'rubisco_extended_combined_20260218_081615.tsv',
    BASE / 'omen-work' / 'rubisco_extended_combined_20260218_081615.tsv',
    BASE / '03_analyses/rubisco_hmms/rubisco_extended_combined_20260218_081615.tsv',
]

def find_file(candidates):
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"None found: {[str(c) for c in candidates]}")

GPS_FILE = find_file(GPS_CANDIDATES)
RUBISCO_FILE = find_file(RUBISCO_CANDIDATES)

print(f"GPS file: {GPS_FILE}")
print(f"RuBisCO file: {RUBISCO_FILE}")

# Load GPS data
gps_df = pd.read_csv(GPS_FILE, sep='\t', comment='#')
gps_df = gps_df.dropna(subset=['latitude', 'longitude'])
print(f"  GPS samples: {len(gps_df)}")

# Load extended RuBisCO data
rubisco_df = pd.read_csv(RUBISCO_FILE, sep='\t', comment='#')
print(f"  RuBisCO samples: {len(rubisco_df)}")

# Extract base ID for matching
def extract_base_id(sample_id):
    s = str(sample_id)
    # Strip common suffixes
    for suffix in ['.aa', '.fasta', '.fa']:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    # For MGYA IDs, keep the MGYA part before _contigs
    if '_contigs' in s:
        s = s.split('_contigs')[0]
    return s

gps_df['base_id'] = gps_df['assembly_id'].apply(extract_base_id)
rubisco_df['base_id'] = rubisco_df['sample_id'].apply(extract_base_id)

# Merge
merged = gps_df.merge(rubisco_df, on='base_id', how='left')

# Collapse the 8 raw dataset labels from the GPS master key into the
# three-category framework used throughout the manuscript:
#   Metagenome    = TARA_Oceans + TARA_protist + OSD (all MGnify assemblies)
#   Transcriptome = MMETSP
#   Cultured ref. = RefGenome_* + Reference_Genome + AAC
_DATASET_ALIASES = {
    'TARA_Oceans':       'Metagenome',
    'TARA_protist':      'Metagenome',
    'OSD':               'Metagenome',
    'MMETSP':            'Transcriptome',
    'RefGenome_GenBank': 'Cultured ref.',
    'Reference_Genome':  'Cultured ref.',
    'RefGenome_PRE_REF': 'Cultured ref.',
    'AAC':               'Cultured ref.',
}
if 'dataset' in merged.columns:
    _before_counts = merged['dataset'].value_counts().to_dict()
    merged['dataset'] = merged['dataset'].replace(_DATASET_ALIASES)
    print(f"  Consolidated 8 raw dataset labels into 3 categories:")
    for new_cat in ['Metagenome', 'Transcriptome', 'Cultured ref.']:
        srcs = [s for s, t in _DATASET_ALIASES.items() if t == new_cat]
        total = sum(_before_counts.get(s, 0) for s in srcs)
        print(f"    {new_cat:15s} = {total:4d} samples ({', '.join(srcs)})")

for lin in ALL_LINEAGES:
    if lin not in merged.columns:
        merged[lin] = 0
    merged[lin] = merged[lin].fillna(0)

print(f"  Merged samples with GPS: {len(merged)}")

# Assign basins
merged['basin'] = merged.apply(lambda r: assign_basin(r['latitude'], r['longitude']), axis=1)
print(f"\nBasin assignment:")
for basin in BASIN_ORDER:
    n = (merged['basin'] == basin).sum()
    if n > 0:
        print(f"  {basin:20s}: {n:5d} samples")

# Report lineage coverage
print("\nLineage detection in GPS-mapped samples:")
for lin in ALL_LINEAGES:
    n = (merged[lin] > 0).sum()
    s = int(merged[lin].sum())
    form = "I " if lin in FORM_I_ORDER else "II"
    print(f"  Form {form} {LINEAGE_DISPLAY[lin]:20s}: {n:5d} samples, {s:6d} seqs")

# =============================================================================
# FIGURE CREATION
# =============================================================================
print("\nCreating figure...")

# Layout:
#   Row 0: Panel A (big map, left ~59%) sized so the Robinson map FILLS its
#          cell (no top/bottom whitespace) | right ~38% column with Panel C
#          (Mamiellophyceae map) on top and a SMALL Panel B (basin bars) below.
#   Row 1: thin full-width Data Sources legend strip.
#   Row 2: Form I row 1 (3 maps: D, E, F)
#   Row 3: Form I row 2 (3 maps: G, H, I)
#   Row 4: Form I row 3 (3 maps: J, K, L)
#   Row 5: Form II row 1 (3 maps: M, N, O)
#   Row 6: Form II row 2 (2 maps: P, Q + size legend)
#
# Total: 7 rows. Row 0 height is matched to the map's ~2.1:1 aspect at the
# chosen width so Panel A has no empty band above/below it.

# Figure height (7.7 in) and row ratios are matched to the world maps' ~2.1:1
# aspect so each map FILLS its cell — taller cells would letterbox the maps and
# leave large vertical gaps between rows. row0 (A + B/C block) ~1.78x a lineage
# row; the legend strip ~0.26x.
fig = plt.figure(figsize=(7, 7.7))

gs = gridspec.GridSpec(7, 1, figure=fig,
                       height_ratios=[1.78, 0.26, 1.0, 1.0, 1.0, 1.0, 1.0],
                       hspace=0.02)

# Row 0 sub-grid: big map A (left ~59%) | right column (B over C, ~38%)
gs_row0 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[0],
    width_ratios=[1.6, 1.0], wspace=0.10
)

# Right column, in reading order: small Panel B (basin bars) on top, then
# Panel C (Mamiellophyceae map, full size) below. B is shorter than C and is
# narrowed to ~80% of the column width (it's a bar chart, not a map).
gs_right = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs_row0[0, 1],
    height_ratios=[0.62, 1.0],
    hspace=0.40
)
# B occupies the left ~80% of the top cell; the rest is left empty.
gs_b = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_right[0, 0],
    width_ratios=[0.8, 0.2], wspace=0.0
)

# Legend strip (row 1) spans full width.
gs_legend = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1])

# Rows 2-6: 3-column grids for lineage maps
gs_rows = []
for r in range(2, 7):
    gs_r = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[r], wspace=0.02
    )
    gs_rows.append(gs_r)

# =============================================================================
# PANEL A: Main Map (Light Theme)
# =============================================================================
ax_main = fig.add_subplot(gs_row0[0, 0], projection=ccrs.Robinson())
ax_main.set_facecolor('#e8f4f8')
ax_main.add_feature(cfeature.LAND, facecolor='#e0e0e0', edgecolor='none')
ax_main.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='#888888')
for spine in ax_main.spines.values():
    spine.set_visible(False)
ax_main.set_extent([-180, 180, -75, 80], crs=ccrs.PlateCarree())

# Plot samples by dataset — use n_total or n_proteins for size scaling
if 'n_total' not in merged.columns:
    if 'n_proteins' in merged.columns:
        merged['n_total'] = merged['n_proteins'].fillna(0)
    else:
        merged['n_total'] = 0
merged['n_total'] = merged['n_total'].fillna(0)
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
        label=f"{dataset.replace('_', ' ')} (n={len(subset)})",
        zorder=5
    )

ax_main.text(0.02, 0.98, 'A', transform=ax_main.transAxes,
             fontsize=8, fontweight='bold', va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor='none', alpha=0.8))
ax_main.set_title(f'Global Sample Distribution (n={len(merged):,})',
                  fontsize=6, pad=2)

# =============================================================================
# PANEL B: Dataset Composition by Basin (vertical stacked bars)
# =============================================================================
print("  Drawing Panel B: Dataset composition by basin...")
ax_comp = fig.add_subplot(gs_b[0, 0])  # top of right column, ~80% width (small B)

# Compute composition percentages per basin
basins_present = [b for b in BASIN_ORDER if (merged['basin'] == b).sum() > 0]
n_basins = len(basins_present)
x_positions = np.arange(n_basins)
bar_width = 0.70

composition_data = {}
for basin in basins_present:
    basin_mask = merged['basin'] == basin
    total = basin_mask.sum()
    row = {}
    for ds in DATASET_ORDER:
        ds_count = ((merged['dataset'] == ds) & basin_mask).sum()
        row[ds] = ds_count / total * 100.0 if total > 0 else 0.0
    composition_data[basin] = row

bottom_edges = np.zeros(n_basins)
for ds in DATASET_ORDER:
    heights = np.array([composition_data[b][ds] for b in basins_present])
    if np.sum(heights) == 0:
        continue
    ax_comp.bar(x_positions, heights, width=bar_width, bottom=bottom_edges,
                color=DATASET_COLORS[ds], edgecolor='white', linewidth=0.2)
    bottom_edges += heights

ax_comp.set_ylim(0, 100)
ax_comp.set_ylabel('Samples (%)')
ax_comp.set_xticks(x_positions)
basin_labels = [BASIN_XLABELS[BASIN_ORDER.index(b)] for b in basins_present]
ax_comp.set_xticklabels(basin_labels, rotation=0, ha='center', fontsize=6)
ax_comp.tick_params(axis='x', labelsize=6, pad=1)
ax_comp.set_title('Dataset Composition by Basin', pad=2, fontsize=6)

# B panel label — same white-box style as A and the map panels.
ax_comp.text(-0.20, 1.12, 'B', transform=ax_comp.transAxes,
             fontsize=8, fontweight='bold', va='top', ha='left', color='black',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor='none', alpha=0.8))

# Dataset legend placed in Row 1 (cols 0-1) directly below Panel A

# =============================================================================
# HELPER: Plot a lineage map panel
# =============================================================================
def plot_lineage_panel(ax, lineage, panel_label, color, form_label=None):
    """Plot a single lineage map panel with dark theme."""
    ax.set_facecolor('#1a1a2e')
    ax.add_feature(cfeature.LAND, facecolor='#2d2d2d', edgecolor='none')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.15, edgecolor='#3a3a3a')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_extent([-180, 180, -75, 80], crs=ccrs.PlateCarree())

    lineage_data = merged[merged[lineage] > 0].copy()

    if len(lineage_data) > 0:
        sizes = np.sqrt(lineage_data[lineage]) * 4 + 3
        sizes = np.clip(sizes, 3, 35)
        ax.scatter(
            lineage_data['longitude'], lineage_data['latitude'],
            c=[color], s=sizes, alpha=0.85,
            transform=ccrs.PlateCarree(),
            edgecolors='black', linewidths=0.1, zorder=5
        )

    # Panel label — standardized to match Panel A (white box, black bold text)
    # so lettering is consistent across all panels (A, B, C-Q).
    ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left',
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.8))

    # Lineage name + count
    n_samples = len(lineage_data)
    display_name = LINEAGE_DISPLAY[lineage]
    ax.text(0.03, 0.03, f'{display_name} (n={n_samples})',
            transform=ax.transAxes,
            fontsize=6, fontweight='bold', va='bottom', ha='left',
            color='white',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='black',
                      edgecolor='none', alpha=0.4))

    # Form label badge (top right)
    if form_label:
        badge_color = '#2a5a2a' if 'I ' in form_label else '#8b2500'
        ax.text(0.97, 0.95, form_label, transform=ax.transAxes,
                fontsize=6, fontweight='bold', va='top', ha='right',
                color='white',
                bbox=dict(boxstyle='round,pad=0.15', facecolor=badge_color,
                          edgecolor='none', alpha=0.7))

# =============================================================================
# DATA SOURCES LEGEND — thin full-width strip below the A/C/B block (row 1)
# =============================================================================
ds_entries = [(ds, DATASET_COLORS[ds], len(merged[merged['dataset'] == ds]))
              for ds in DATASET_ORDER
              if len(merged[merged['dataset'] == ds]) > 0]

ax_dsleg = fig.add_subplot(gs_legend[0, 0])
ax_dsleg.axis('off')

# Compact single-row legend via matplotlib's legend engine (tight, reliable
# spacing). "Data Sources:" label sits at the far left, the 3 entries packed
# immediately after it.
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=4,
                  markerfacecolor=color, markeredgecolor='#333333',
                  markeredgewidth=0.2,
                  label='{} ({})'.format(DATASET_LABELS.get(ds, ds.replace('_', ' ')), n))
           for ds, color, n in ds_entries]
ax_dsleg.text(0.005, 0.5, 'Data Sources:', fontsize=6, fontweight='bold',
              va='center', ha='left', color='#1a1a1a',
              transform=ax_dsleg.transAxes)
ax_dsleg.legend(handles=handles, loc='center left', bbox_to_anchor=(0.135, 0.5),
                ncol=len(handles), frameon=False, fontsize=6,
                handletextpad=0.25, columnspacing=0.9, borderpad=0.0)

# =============================================================================
# PANEL C: Mamiellophyceae — bottom of the right column in row 0 (below B),
# full column width (not shrunk).
# =============================================================================
ax = fig.add_subplot(gs_right[1, 0], projection=ccrs.Robinson())
plot_lineage_panel(ax, FORM_I_ORDER[0], 'C',
                   FORM_I_COLORS[FORM_I_ORDER[0]], form_label='Form I')

# =============================================================================
# PANELS D-L: Remaining Form I Lineage Maps (rows 1-3, 3 per row)
# =============================================================================
panel_labels_formI = 'DEFGHIJKL'

for i, lineage in enumerate(FORM_I_ORDER[1:]):
    row_idx = i // 3       # gs_rows[0], gs_rows[1], gs_rows[2]
    col = i % 3
    ax = fig.add_subplot(gs_rows[row_idx][0, col], projection=ccrs.Robinson())
    plot_lineage_panel(ax, lineage, panel_labels_formI[i],
                       FORM_I_COLORS[lineage], form_label='Form I')

# =============================================================================
# PANELS M-Q: Form II Lineage Maps (rows 4-5)
# =============================================================================
panel_labels_ii = 'MNOPQ'

# Row 4 (gs_rows[3]): 3 Form II lineages
for i, lineage in enumerate(FORM_II_ORDER[:3]):
    ax = fig.add_subplot(gs_rows[3][0, i], projection=ccrs.Robinson())
    plot_lineage_panel(ax, lineage, panel_labels_ii[i],
                       FORM_II_COLORS[lineage], form_label='Form II')

# Row 5 (gs_rows[4]): 2 Form II lineages + size legend
for i, lineage in enumerate(FORM_II_ORDER[3:5]):
    ax = fig.add_subplot(gs_rows[4][0, i], projection=ccrs.Robinson())
    plot_lineage_panel(ax, lineage, panel_labels_ii[3 + i],
                       FORM_II_COLORS[lineage], form_label='Form II')

# =============================================================================
# SIZE LEGEND PANEL (row 5, col 2)
# =============================================================================
ax_leg = fig.add_subplot(gs_rows[4][0, 2])
ax_leg.set_facecolor('#f5f5f5')
ax_leg.axis('off')

# Size legend
ax_leg.text(0.5, 0.97, 'RuBisCO Sequences', fontsize=6,
            fontweight='bold', ha='center', va='top', color='#1a1a1a')

size_vals = [1, 5, 20, 50, 100]
y_pos = 0.82
for val in size_vals:
    s = np.sqrt(val) * 4 + 3
    ax_leg.scatter([0.25], [y_pos], s=s, c='#888888',
                   edgecolors='#333333', linewidths=0.2, alpha=0.9)
    ax_leg.text(0.45, y_pos, f'{val}', fontsize=6,
                va='center', ha='left', color='#1a1a1a')
    y_pos -= 0.12

# Form color key
y_pos -= 0.08
ax_leg.text(0.5, y_pos, 'RuBisCO Form', fontsize=6,
            fontweight='bold', ha='center', va='top', color='#1a1a1a')
y_pos -= 0.10
ax_leg.scatter([0.2], [y_pos], s=20, c='#2a5a2a', edgecolors='#333333',
               linewidths=0.2, marker='s')
ax_leg.text(0.35, y_pos, 'Form I (chloroplast)', fontsize=6,
            va='center', ha='left', color='#1a1a1a')
y_pos -= 0.09
ax_leg.scatter([0.2], [y_pos], s=20, c='#8b2500', edgecolors='#333333',
               linewidths=0.2, marker='s')
ax_leg.text(0.35, y_pos, 'Form II (nuclear)', fontsize=6,
            va='center', ha='left', color='#1a1a1a')

ax_leg.set_xlim(0, 1)
ax_leg.set_ylim(0, 1)

# =============================================================================
# SAVE
fig.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98)

# =============================================================================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = Path(__file__).parent
figures_dir = Path(__file__).parent.parent / 'figures'
output_stem = f'Figure1_extended_{timestamp}'

print(f"\nSaving outputs...")
for fmt in ['pdf', 'svg']:
    outpath = output_dir / f'{output_stem}.{fmt}'
    fig.savefig(outpath, format=fmt, bbox_inches='tight',
                transparent=True, edgecolor='none', dpi=300)
    print(f"  Saved: {outpath} ({outpath.stat().st_size / 1e6:.1f} MB)")
    # Also save to figures/ directory
    if figures_dir.exists():
        fig_path = figures_dir / f'{output_stem}.{fmt}'
        fig.savefig(fig_path, format=fmt, bbox_inches='tight',
                    transparent=True, edgecolor='none', dpi=300)
        print(f"  Saved: {fig_path} ({fig_path.stat().st_size / 1e6:.1f} MB)")

# Provenance
prov_path = output_dir / f'{output_stem}_provenance.txt'
with open(prov_path, 'w') as f:
    f.write(f"# Provenance for {output_stem}\n")
    f.write(f"Script: {__file__}\n")
    f.write(f"Date: {datetime.now().isoformat()}\n")
    f.write(f"Input GPS: {GPS_FILE}\n")
    f.write(f"Input RuBisCO: {RUBISCO_FILE}\n")
    f.write(f"Total GPS-mapped samples: {len(merged)}\n")
    f.write(f"\nForm I lineages (chloroplast-encoded) [from rubisco_all_samples_20260114.tsv]:\n")
    for lin in FORM_I_ORDER:
        n = (merged[lin] > 0).sum()
        s = int(merged[lin].sum())
        f.write(f"  {LINEAGE_DISPLAY[lin]:20s}: {n:5d} samples, {s:6d} seqs\n")
    f.write(f"\nForm II lineages (nuclear-encoded, myzozoan):\n")
    for lin in FORM_II_ORDER:
        n = (merged[lin] > 0).sum()
        s = int(merged[lin].sum())
        f.write(f"  {LINEAGE_DISPLAY[lin]:20s}: {n:5d} samples, {s:6d} seqs\n")

print(f"  Saved: {prov_path}")
print("Done!")
plt.close(fig)
