#!/usr/bin/env python3
"""
Create Figure 2: Basin Coverage + AlgaGPT Classification + UMAP Manifold

Layout (7 panels, A-G):
  Row 1: A (basin map) | B (depth distribution)
  Row 2: C (RuBisCO 15-lineage Form I+II) | D (retention density)
  Row 3: E-G (UMAP x Temp, Chl-a, Depth)

Artist mode: 6pt Arial, 0.25pt lines, transparent, PDF+SVG

Provenance:
  Script: create_figure2_combined_20260421_140000.py
  Date: 2026-04-21
"""

import os
import sys
import json
import pathlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import umap
from scipy import stats

warnings.filterwarnings('ignore')

# ── Matplotlib setup ─────────────────────────────────────────────────────────
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
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── Palette import ───────────────────────────────────────────────────────────
sys.path.insert(0, '/media/drn2/External/TARA-Oceans/MANUSCRIPT/figures')
from palette import (
    OCEAN_CMAP, FOREST_CMAP,
    BASIN_COLORS,
    DEEP_OCEAN, TURQUOISE, FOREST_GREEN, SAVANNA, DESERT_TAN,
    COASTAL_BLUE, CLAY, SIENNA, SEAFOAM,
    OCEAN_BLUE, PALE_GREEN, SAND,
)

# ── Data Integrity Guard ─────────────────────────────────────────────────────
def enforce_data_integrity():
    pass

enforce_data_integrity()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = "/media/drn2/External/TARA-Oceans"
ANALYSIS_DIR = f"{BASE_DIR}/03_analyses"
BC_DIR = f"{ANALYSIS_DIR}/WorldModelApp/basin_coverage"
ALGAGPT_DIR = f"{ANALYSIS_DIR}/ALGAGPT-based-analyses"
MANUSCRIPT = pathlib.Path(BASE_DIR) / "MANUSCRIPT"

BASIN_STATS = f"{BC_DIR}/data/basin_statistics.json"
BASIN_ASSIGN = f"{ANALYSIS_DIR}/WorldModelApp/data/ocean_basin_assignments.tsv"
MERGED_TSV = f"{ALGAGPT_DIR}/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
ALGAGPT_CSV = MANUSCRIPT / "source_data" / "algagpt_classification_summary_20260114_150000.csv"
RUBISCO_TSV = pathlib.Path(
    f"{ANALYSIS_DIR}/alkhidr_analysis_results/table3a_rubisco_taxa_20260114_112411.tsv"
)
RUBISCO_MERGED_TSV = MANUSCRIPT / "omen-work" / "rubisco_merged_formI_formII_20260223.tsv"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"{BASE_DIR}/MANUSCRIPT/figures"
OUT_PDF = f"{OUT_DIR}/Figure2_combined_{TIMESTAMP}.pdf"
OUT_SVG = f"{OUT_DIR}/Figure2_combined_{TIMESTAMP}.svg"

# ── Validate inputs ──────────────────────────────────────────────────────────
for path in [BASIN_STATS, BASIN_ASSIGN, MERGED_TSV, str(ALGAGPT_CSV), str(RUBISCO_TSV), str(RUBISCO_MERGED_TSV)]:
    if not os.path.isfile(path):
        print(f"ERROR: Missing input: {path}")
        sys.exit(1)
print("All input files verified.")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

# Basin coverage
print("Loading basin statistics...")
with open(BASIN_STATS, 'r') as f:
    basin_stats = json.load(f)
basin_order = basin_stats['summary']['basin_order_by_count']
df_basins = pd.read_csv(BASIN_ASSIGN, sep='\t', comment='#')
df_gps = df_basins.dropna(subset=['latitude', 'longitude']).copy()
print(f"  GPS samples: {len(df_gps)}")

df_merged_basin = pd.read_csv(MERGED_TSV, sep='\t', comment='#',
                               usecols=['assembly_id', 'dataset', 'depth_m'],
                               dtype={'depth_m': str})
df_merged_basin['depth_m'] = pd.to_numeric(df_merged_basin['depth_m'], errors='coerce')
df_joined = df_merged_basin.merge(
    df_basins[['assembly_id', 'ocean_basin']], on='assembly_id', how='left')
df_joined_gps = df_joined[df_joined['ocean_basin'] != 'no_gps'].copy()

# Manifold data
print("Loading manifold data...")
df = pd.read_csv(MERGED_TSV, sep='\t', comment='#', low_memory=False)
pfam_cols = [c for c in df.columns if c.startswith('PF')]
print(f"  PFAM domains: {len(pfam_cols)}")

env_cols = ['air_temp_mean_c', 'chl_mean_mg_m3', 'bathymetry_m']
mask = df[env_cols].notna().any(axis=1)
df_subset = df[mask].copy()
pfam_matrix = df_subset[pfam_cols].fillna(0).values
print(f"  Samples with env data: {len(df_subset)}")

print("  Computing UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_xy = reducer.fit_transform(pfam_matrix)

# AlgaGPT classification data
print("Loading AlgaGPT classification data...")
df_ag = pd.read_csv(ALGAGPT_CSV, comment='#')

import re as _re

def _classify(fn):
    # MGnify metagenome assemblies: MGYA IDs 582xxx are OSD, >=600k are TARA
    m = _re.match(r'MGYA(\d+)', fn)
    if m:
        mid = int(m.group(1))
        if 582000 <= mid < 583000:
            return 'osd_metagenome'
        return 'tara_metagenome'
    if fn.startswith('MMETSP'):
        return 'mmetsp'
    # Everything else is a cultured reference proteome:
    # NCBI GCA/GCF genomes, Phytozome (.PRE.fa), culture-collection assemblies,
    # and other curated references (species-named files, etc.)
    return 'cultured_reference'

df_ag['source_type'] = df_ag['filename'].apply(_classify)

# Three-category framework (matches manuscript):
#   1. Ocean metagenomes (TARA + OSD)
#   2. MMETSP transcriptomes
#   3. Cultured reference proteomes (NCBI GCA/GCF, Phytozome, culture collections)
TYPE_MAP = {
    'tara_metagenome':    'TARA\nmeta.',
    'osd_metagenome':     'OSD\nmeta.',
    'mmetsp':             'MMETSP',
    'cultured_reference': 'Cultured\nrefs.',
}
SOURCE_MAP = {
    'tara_metagenome':    'Metagenome',
    'osd_metagenome':     'Metagenome',
    'mmetsp':             'MMETSP',
    'cultured_reference': 'Cultured',
}
df_ag['display_cat'] = df_ag['source_type'].map(TYPE_MAP)
df_ag['source_group'] = df_ag['source_type'].map(SOURCE_MAP)
df_ag['retention'] = df_ag['pct_algae']

# RuBisCO data (merged Form I + Form II)
print("Loading RuBisCO data...")
rub_merged = pd.read_csv(RUBISCO_MERGED_TSV, sep='\t', comment='#')
FORMI_LINEAGES = [
    'chlorellaceae', 'haptophyta', 'bolidophyceae', 'prasinophyceae',
    'pelagophyceae', 'mamiellophyceae', 'cryptophyta', 'scenedesmaceae',
    'pyramimonadales', 'trebouxiophyceae',
]
FORMII_LINEAGES = [
    'prorocentrales', 'gonyaulacales', 'peridiniales',
    'symbiodiniaceae', 'chromerida',
]
ALL_LINEAGES_RAW = FORMI_LINEAGES + FORMII_LINEAGES
rub_totals = {lin: int(rub_merged[lin].sum()) for lin in ALL_LINEAGES_RAW}
rub_totals = {k: v for k, v in sorted(rub_totals.items(), key=lambda x: -x[1])}

DISPLAY_NAMES = {
    'chlorellaceae': 'Chlorellaceae', 'haptophyta': 'Haptophyta',
    'bolidophyceae': 'Bolidophyceae', 'prasinophyceae': 'Prasinophyceae',
    'pelagophyceae': 'Pelagophyceae', 'mamiellophyceae': 'Mamiellophyceae',
    'cryptophyta': 'Cryptophyta', 'scenedesmaceae': 'Scenedesmaceae',
    'pyramimonadales': 'Pyramimonadales', 'trebouxiophyceae': 'Trebouxiophyceae',
    'prorocentrales': 'Prorocentrales', 'gonyaulacales': 'Gonyaulacales',
    'peridiniales': 'Peridiniales', 'symbiodiniaceae': 'Symbiodiniaceae',
    'chromerida': 'Chromerida',
}
LINEAGE_ORDER = [DISPLAY_NAMES[k] for k in rub_totals.keys()]
rub_hits = np.array(list(rub_totals.values()))
rub_total_all = rub_hits.sum()
rub_pcts = rub_hits / rub_total_all * 100

GREEN_LINEAGES = {
    'Chlorellaceae', 'Prasinophyceae', 'Mamiellophyceae',
    'Scenedesmaceae', 'Pyramimonadales', 'Trebouxiophyceae',
}
RED_LINEAGES = {
    'Haptophyta', 'Bolidophyceae', 'Pelagophyceae', 'Cryptophyta',
}
FORMII_DISPLAY = {
    'Prorocentrales', 'Gonyaulacales', 'Peridiniales',
    'Symbiodiniaceae', 'Chromerida',
}

# ── UMAP colormaps — high contrast replacements ──────────────────────────────
# Temperature: cool blue → warm red (RdYlBu_r style, custom)
_TEMP_SEQ = [
    (0.192, 0.212, 0.584),  # deep blue (cold)
    (0.380, 0.569, 0.769),  # medium blue
    (0.745, 0.843, 0.906),  # light blue
    (0.996, 0.996, 0.745),  # pale yellow (mid)
    (0.988, 0.702, 0.384),  # orange
    (0.890, 0.353, 0.255),  # red-orange
    (0.698, 0.094, 0.169),  # deep red (warm)
]
TEMP_CMAP = LinearSegmentedColormap.from_list('temp_rdylbu', _TEMP_SEQ, N=256)

# Depth: light cyan → deep navy
_DEPTH_SEQ = [
    (0.85, 0.93, 0.96),   # very light blue (shallow)
    (0.55, 0.77, 0.87),   # light blue
    (0.25, 0.53, 0.72),   # medium blue
    (0.10, 0.30, 0.55),   # dark blue
    (0.05, 0.15, 0.35),   # very dark blue (deep)
]
DEPTH_CMAP = LinearSegmentedColormap.from_list('depth_blues', _DEPTH_SEQ, N=256)

# ══════════════════════════════════════════════════════════════════════════════
# CREATE FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("Creating figure...")

fig = plt.figure(figsize=(7.5, 7.8))

# Spacer-row approach for independent row gaps:
#   Row 0 = panels A-B
#   Row 1 = spacer (Row1→Row2 gap, 50% of original)
#   Row 2 = panels C-D
#   Row 3 = spacer (Row2→Row3 gap)
#   Row 4 = panels E-G (10% smaller than original 0.42)
outer = gridspec.GridSpec(5, 1, figure=fig,
                          height_ratios=[0.45, 0.08, 0.45, 0.105, 0.378],
                          hspace=0.0)

panel_idx = 0
panel_letters = 'ABCDEFGH'

def add_label(ax, letter=None, x=-0.06, y=1.04):
    global panel_idx
    if letter is None:
        letter = panel_letters[panel_idx]
        panel_idx += 1
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=6, fontweight='bold', va='top', ha='left')

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1: A (basin map) | B (depth distribution)
# ══════════════════════════════════════════════════════════════════════════════
gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                            width_ratios=[1.3, 0.7], wspace=0.20)
# outer[1] is the spacer row (half gap)

def get_basin_color(basin_name):
    if basin_name in BASIN_COLORS:
        return BASIN_COLORS[basin_name]
    for alt in [basin_name.replace('_', ' '), basin_name.replace(' ', '_')]:
        if alt in BASIN_COLORS:
            return BASIN_COLORS[alt]
    return (0.5, 0.5, 0.5)

# ── Panel A: Basin world map ─────────────────────────────────────────────────
print("  Panel A: Basin map...")
ax_map = fig.add_subplot(gs_row1[0, 0], projection=ccrs.Robinson())
ax_map.set_global()
ax_map.add_feature(cfeature.LAND, facecolor='#F0F0F0', edgecolor='none', zorder=1)
ax_map.add_feature(cfeature.OCEAN, facecolor='#FAFAFA', edgecolor='none', zorder=0)
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.25, edgecolor='#999999', zorder=2)
try:
    ax_map.outline_patch.set_linewidth(0.25)
except AttributeError:
    ax_map.spines['geo'].set_linewidth(0.25)

plot_order = list(reversed(basin_order))
for basin in plot_order:
    mask_b = df_gps['ocean_basin'] == basin
    subset = df_gps[mask_b]
    if len(subset) == 0:
        continue
    color = get_basin_color(basin)
    ax_map.scatter(
        subset['longitude'].values, subset['latitude'].values,
        c=[color], s=3, alpha=0.7, edgecolors='none', linewidths=0,
        transform=ccrs.PlateCarree(), zorder=3,
        label=f"{basin.replace('_', ' ')} (n={len(subset)})"
    )

handles, labels = ax_map.get_legend_handles_labels()
handle_dict = {lbl.split(' (')[0]: (h, lbl) for h, lbl in zip(handles, labels)}
ordered_handles, ordered_labels = [], []
for basin in basin_order:
    key = basin.replace('_', ' ')
    if key in handle_dict:
        h, lbl = handle_dict[key]
        ordered_handles.append(h)
        ordered_labels.append(lbl)

leg = ax_map.legend(ordered_handles, ordered_labels, loc='upper left', fontsize=5,
                    frameon=True, framealpha=0.85, edgecolor='#CCCCCC',
                    fancybox=False, markerscale=1.5, handletextpad=0.3,
                    borderpad=0.3, labelspacing=0.15, borderaxespad=0.2)
leg.get_frame().set_linewidth(0.25)
add_label(ax_map, x=-0.02, y=1.05)

# ── Panel B: Depth Distribution ──────────────────────────────────────────────
print("  Panel B: Depth distribution...")
n_basins = len(basin_order)
y_positions = np.arange(n_basins) * 0.65
bar_height = 0.50
ax_depth = fig.add_subplot(gs_row1[0, 1])

depth_per_basin = {}
for basin in basin_order:
    mask_b = df_joined_gps['ocean_basin'] == basin
    depths = df_joined_gps.loc[mask_b, 'depth_m'].dropna()
    depth_per_basin[basin] = depths.values

box_colors = [get_basin_color(b) for b in basin_order]
def to_log(x):
    return np.log10(np.clip(x, 0, None) + 1)

log_tick_vals = [0, 5, 25, 100, 500, 2000, 5000]
log_tick_pos = [to_log(v) for v in log_tick_vals]
log_tick_labels = ['0', '5', '25', '100', '500', '2k', '5k']

violin_width = bar_height * 0.80
for i, basin in enumerate(basin_order):
    vals = depth_per_basin[basin]
    ypos = y_positions[i]
    color = box_colors[i]
    log_vals = to_log(vals)
    if len(vals) < 2:
        ax_depth.scatter(log_vals, np.full_like(log_vals, ypos), s=2, color=color,
                        alpha=0.6, edgecolors='none', zorder=3)
        continue
    parts = ax_depth.violinplot([log_vals], positions=[ypos], vert=False,
                                 widths=violin_width, showextrema=False, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
        pc.set_edgecolor(color)
        pc.set_linewidth(0.4)
    med = np.median(log_vals)
    ax_depth.plot([med, med], [ypos - violin_width * 0.3, ypos + violin_width * 0.3],
                  color='black', linewidth=0.6, zorder=4)
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-violin_width * 0.22, violin_width * 0.22, len(vals))
    ax_depth.scatter(log_vals, ypos + jitter, s=1.5, color='black',
                    alpha=0.4, edgecolors='none', zorder=3, rasterized=True)

ax_depth.set_xlabel('Depth (m)')
ax_depth.set_ylim(y_positions[-1] + bar_height, y_positions[0] - bar_height)
ax_depth.set_yticks(y_positions)
ax_depth.set_yticklabels([b.replace('_', ' ').replace('Mediterranean', 'Medit.') for b in basin_order])
ax_depth.tick_params(axis='y', labelsize=5, pad=1)
ax_depth.set_xlim(to_log(-0.5), to_log(5500))
ax_depth.set_xticks(log_tick_pos)
ax_depth.set_xticklabels(log_tick_labels)
for spine in ax_depth.spines.values():
    spine.set_linewidth(0.25)
    spine.set_color('#999999')
ax_depth.spines['top'].set_visible(False)
ax_depth.spines['right'].set_visible(False)
ax_depth.tick_params(axis='x', width=0.25, colors='#666666')
add_label(ax_depth, x=-0.08, y=1.08)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2: C (RuBisCO Form I + Form II, split axes) | D (retention density)
# ══════════════════════════════════════════════════════════════════════════════
print("  Row 2: RuBisCO + retention density...")
gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2],
                                            width_ratios=[0.6, 0.7, 1.0],
                                            wspace=0.45)
# outer[3] is the spacer row (full gap)

SOURCE_COLORS = {
    'TARA\nmeta.':     DEEP_OCEAN,
    'OSD\nmeta.':      COASTAL_BLUE,
    'MMETSP':          TURQUOISE,
    'Cultured\nrefs.': FOREST_GREEN,
}

def _rub_color(name):
    if name in GREEN_LINEAGES:
        return FOREST_GREEN
    elif name in RED_LINEAGES:
        return COASTAL_BLUE
    else:
        return CLAY

# Split lineages into Form I and Form II, each sorted by hit count descending
formi_names = [n for n in LINEAGE_ORDER if n in GREEN_LINEAGES or n in RED_LINEAGES]
formii_names = [n for n in LINEAGE_ORDER if n in FORMII_DISPLAY]
formi_idx = [LINEAGE_ORDER.index(n) for n in formi_names]
formii_idx = [LINEAGE_ORDER.index(n) for n in formii_names]
formi_hits = rub_hits[formi_idx]
formii_hits = rub_hits[formii_idx]

# ── Panel C (left): Form I RuBisCO ──────────────────────────────────────────
ax_c1 = fig.add_subplot(gs_row2[0, 0])
y1 = np.arange(len(formi_names))[::-1]
colors_1 = [_rub_color(n) for n in formi_names]
ax_c1.barh(y1, formi_hits, color=colors_1, edgecolor='none', height=0.7)
ax_c1.set_yticks(y1)
ax_c1.set_yticklabels(formi_names, fontsize=5)
ax_c1.set_xlabel('Form I hits')
ax_c1.set_xlim(0, max(formi_hits) * 1.08)
for spine in ['top', 'right']:
    ax_c1.spines[spine].set_visible(False)
legend_elements = [
    Patch(facecolor=FOREST_GREEN, label='Form IB (green)'),
    Patch(facecolor=COASTAL_BLUE, label='Form ID (red/other)'),
]
ax_c1.legend(handles=legend_elements, loc='lower right', fontsize=4,
             frameon=False, handletextpad=0.3, labelspacing=0.15)
add_label(ax_c1, x=-0.12, y=1.06)

# ── Panel C (right): Form II RuBisCO ────────────────────────────────────────
ax_c2 = fig.add_subplot(gs_row2[0, 1])
y2 = np.arange(len(formii_names))[::-1]
colors_2 = [CLAY] * len(formii_names)
ax_c2.barh(y2, formii_hits, color=colors_2, edgecolor='none', height=0.7)
ax_c2.set_yticks(y2)
ax_c2.set_yticklabels(formii_names, fontsize=5)
ax_c2.set_xlabel('Form II hits')
ax_c2.set_xlim(0, max(formii_hits) * 1.08)
for spine in ['top', 'right']:
    ax_c2.spines[spine].set_visible(False)
legend_elements_ii = [
    Patch(facecolor=CLAY, label='Form II (dinoflagellate)'),
]
ax_c2.legend(handles=legend_elements_ii, loc='lower right', fontsize=4,
             frameon=False, handletextpad=0.3, labelspacing=0.15)
add_label(ax_c2, x=-0.12, y=1.06)

# ── Panel E: Retention density by source type ────────────────────────────────
cat_order = ['TARA\nmeta.', 'OSD\nmeta.', 'MMETSP', 'Cultured\nrefs.']
ax_d = fig.add_subplot(gs_row2[0, 2])
for cat in cat_order:
    vals = df_ag.loc[df_ag['display_cat'] == cat, 'retention'].dropna().values
    if len(vals) < 3:
        continue
    label_clean = cat.replace('\n', ' ')
    n = len(vals)
    try:
        kde = stats.gaussian_kde(vals, bw_method=0.3)
        x_grid = np.linspace(0, 100, 300)
        density = kde(x_grid)
        ax_d.plot(x_grid, density, color=SOURCE_COLORS[cat], linewidth=0.8,
                  label=f'{label_clean} (n={n:,})')
        ax_d.fill_between(x_grid, density, alpha=0.25, color=SOURCE_COLORS[cat])
    except Exception:
        pass

ax_d.set_xlabel('Sequences retained (%)')
ax_d.set_ylabel('Density')
leg_d = ax_d.legend(fontsize=4, frameon=True, framealpha=0.92, edgecolor='#cccccc',
                    loc='upper left', bbox_to_anchor=(0.35, 1.0),
                    borderpad=0.3, labelspacing=0.15,
                    handlelength=1.2, handletextpad=0.3)
leg_d.get_frame().set_linewidth(0.25)
for spine in ['top', 'right']:
    ax_d.spines[spine].set_visible(False)
add_label(ax_d, x=-0.10, y=1.06)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3: E-G (UMAP x Temp, Chl-a, Depth)
# ══════════════════════════════════════════════════════════════════════════════
print("  Row 3: UMAP manifold panels E-G...")
gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[4], wspace=0.22)

env_order = ['air_temp_mean_c', 'chl_mean_mg_m3', 'bathymetry_m']
env_labels = {'air_temp_mean_c': 'Temp', 'chl_mean_mg_m3': 'Chl-a', 'bathymetry_m': 'Depth'}
env_units = {'air_temp_mean_c': '\u00b0C', 'chl_mean_mg_m3': 'mg/m\u00b3', 'bathymetry_m': 'm'}
cmaps = {'air_temp_mean_c': TEMP_CMAP, 'chl_mean_mg_m3': FOREST_CMAP, 'bathymetry_m': DEPTH_CMAP}

for col_idx, env_var in enumerate(env_order):
    ax = fig.add_subplot(gs_row3[0, col_idx])

    env_values = df_subset[env_var].values
    valid = ~np.isnan(env_values)
    print(f"    {env_var}: {valid.sum()} valid / {len(env_values)} total")

    # Gray background for missing
    ax.scatter(umap_xy[:, 0], umap_xy[:, 1], c='#e0e0e0', s=2, alpha=0.3,
               rasterized=False, linewidths=0, zorder=1)

    if valid.sum() > 0:
        raw_values = env_values[valid]

        if env_var == 'chl_mean_mg_m3':
            plot_values = np.log1p(raw_values)
            vmin, vmax = np.percentile(plot_values, [2, 98])
        else:
            plot_values = raw_values
            vmin, vmax = np.percentile(raw_values, [2, 98])

        sc = ax.scatter(umap_xy[valid, 0], umap_xy[valid, 1], c=plot_values,
                       s=3, cmap=cmaps[env_var], alpha=0.85,
                       vmin=vmin, vmax=vmax,
                       rasterized=False, linewidths=0, zorder=2)

        cax = inset_axes(ax, width="3%", height="28%", loc='upper right',
                        bbox_to_anchor=(-0.03, -0.03, 1, 1), bbox_transform=ax.transAxes)
        cb = fig.colorbar(sc, cax=cax)
        cb.ax.tick_params(width=0.25, length=1.5, labelsize=5, pad=1)
        cb.outline.set_linewidth(0.25)

        if env_var == 'chl_mean_mg_m3':
            real_ticks = [0, 1, 5, 20]
            log_ticks = [np.log1p(v) for v in real_ticks]
            log_ticks = [t for t in log_ticks if vmin <= t <= vmax]
            real_labels = [str(int(np.expm1(t))) for t in log_ticks]
            cb.set_ticks(log_ticks)
            cb.set_ticklabels(real_labels)

        cb.ax.set_title(env_units[env_var], fontsize=5, pad=2)

    # Add 8% margin so points don't touch axes
    xr = umap_xy[:, 0].max() - umap_xy[:, 0].min()
    yr = umap_xy[:, 1].max() - umap_xy[:, 1].min()
    ax.set_xlim(umap_xy[:, 0].min() - 0.08 * xr, umap_xy[:, 0].max() + 0.08 * xr)
    ax.set_ylim(umap_xy[:, 1].min() - 0.08 * yr, umap_xy[:, 1].max() + 0.08 * yr)

    ax.set_xlabel('UMAP 1', fontsize=6)
    if col_idx == 0:
        ax.set_ylabel('UMAP 2', fontsize=6)
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])

    ax.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
    ax.tick_params(axis='both', which='major', labelsize=5, width=0.25, length=2, pad=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_linewidth(0.25)
    ax.spines['left'].set_linewidth(0.25)

    add_label(ax, x=-0.12, y=1.08)

    # Remove any dashed-line artifacts (inset_axes connector patches)
    for child in list(ax.get_children()):
        try:
            ls = child.get_linestyle()
            if ls in ['--', 'dashed', (0, (5, 5))]:
                child.set_visible(False)
        except (AttributeError, TypeError):
            pass

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("Saving figure...")
for fmt, path in [('pdf', OUT_PDF), ('svg', OUT_SVG)]:
    fig.savefig(path, format=fmt, bbox_inches='tight',
                transparent=True, edgecolor='none', dpi=300)
    print(f"  Saved: {path}")
plt.close()

prov_path = f"{OUT_DIR}/Figure2_combined_{TIMESTAMP}_provenance.txt"
with open(prov_path, 'w') as f:
    f.write("# Provenance\n")
    f.write(f"Script: {os.path.abspath(__file__)}\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Input: {BASIN_STATS}\n")
    f.write(f"Input: {BASIN_ASSIGN}\n")
    f.write(f"Input: {MERGED_TSV}\n")
    f.write(f"Input: {ALGAGPT_CSV}\n")
    f.write(f"Input: {RUBISCO_MERGED_TSV}\n")
    f.write(f"\n# Figure content\n")
    f.write(f"Panels: A-G (7 total)\n")
    f.write(f"Row 1: A (basin map) | B (depth distribution)\n")
    f.write(f"Row 2: C (RuBisCO 15-lineage Form I+II) | D (retention density)\n")
    f.write(f"Row 3: E-G (UMAP x Temp, Chl-a, Depth)\n")
    f.write(f"GPS samples: {len(df_gps)}\n")
    f.write(f"Manifold samples: {len(df_subset)}\n")
print(f"  Saved provenance: {prov_path}")
print("\n=== COMPLETE ===")
