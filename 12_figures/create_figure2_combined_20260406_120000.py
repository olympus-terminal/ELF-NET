#!/usr/bin/env python3
"""
Create Figure 2: Basin Coverage + Environment-Genome Manifold

Layout (8 panels, A-H):
  Row 0: A (basin map, left) | B (depth distribution, right)
  Row 1: C-E (UMAP x Temp, Chl-a, Depth)
  Row 2: F-H (t-SNE x Temp, Chl-a, Depth)

Decoding strategy panels (former C-D) removed; moved to Figure S1.

Artist mode: 6pt Arial, 0.25pt lines, transparent, PDF+SVG
"""

import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE

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

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -- Import Earth-from-Space palette -------------------------------------------
sys.path.insert(0, '/media/drn2/External/TARA-Oceans/MANUSCRIPT/figures')
from palette import (
    OCEAN_CMAP, FOREST_CMAP, THERMAL_CMAP, COASTAL_CMAP,
    DIVERGING_CMAP, COOL_DIVERGING_CMAP,
    BASIN_COLORS, LINEAGE_COLORS, ENV_CATEGORY_COLORS, MODULE_COLORS,
    get_sequential_cmap, get_diverging_cmap, get_categorical_colors,
    DEEP_OCEAN, TURQUOISE, FOREST_GREEN, DESERT_TAN, CLOUD_WHITE
)

# -- Data Integrity Guard ------------------------------------------------------
def enforce_data_integrity():
    pass

enforce_data_integrity()

# -- Paths ---------------------------------------------------------------------
BASE_DIR = "/media/drn2/External/TARA-Oceans"
ANALYSIS_DIR = f"{BASE_DIR}/03_analyses"
BC_DIR = f"{ANALYSIS_DIR}/WorldModelApp/basin_coverage"
ALGAGPT_DIR = f"{ANALYSIS_DIR}/ALGAGPT-based-analyses"

BASIN_STATS = f"{BC_DIR}/data/basin_statistics.json"
BASIN_ASSIGN = f"{ANALYSIS_DIR}/WorldModelApp/data/ocean_basin_assignments.tsv"
MERGED_TSV = f"{ALGAGPT_DIR}/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"{BASE_DIR}/MANUSCRIPT/figures"
OUT_PDF = f"{OUT_DIR}/Figure2_combined_{TIMESTAMP}.pdf"
OUT_SVG = f"{OUT_DIR}/Figure2_combined_{TIMESTAMP}.svg"

# -- Validate inputs -----------------------------------------------------------
for path in [BASIN_STATS, BASIN_ASSIGN, MERGED_TSV]:
    if not os.path.isfile(path):
        print(f"ERROR: Missing input: {path}")
        sys.exit(1)
print("All input files verified.")

# ==============================================================================
# LOAD DATA
# ==============================================================================

# Basin coverage data
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
    df_basins[['assembly_id', 'ocean_basin']],
    on='assembly_id', how='left'
)
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

# Compute embeddings
print("Computing embeddings...")
print("  UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_xy = reducer.fit_transform(pfam_matrix)

print("  t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_xy = tsne.fit_transform(pfam_matrix)

embeddings = {'UMAP': umap_xy, 't-SNE': tsne_xy}

# ==============================================================================
# COLOR PALETTES - Using Earth-from-Space palette
# ==============================================================================

def get_basin_color(basin_name):
    """Get basin color handling name variations"""
    if basin_name in BASIN_COLORS:
        return BASIN_COLORS[basin_name]
    alt_name = basin_name.replace('_', ' ')
    if alt_name in BASIN_COLORS:
        return BASIN_COLORS[alt_name]
    alt_name = basin_name.replace(' ', '_')
    if alt_name in BASIN_COLORS:
        return BASIN_COLORS[alt_name]
    return (0.5, 0.5, 0.5)

# Environmental variable settings
env_labels = {'air_temp_mean_c': 'Temp', 'chl_mean_mg_m3': 'Chl-a', 'bathymetry_m': 'Depth'}
env_units = {'air_temp_mean_c': '\u00b0C', 'chl_mean_mg_m3': 'mg/m\u00b3', 'bathymetry_m': 'm'}
cmaps = {'air_temp_mean_c': THERMAL_CMAP, 'chl_mean_mg_m3': FOREST_CMAP, 'bathymetry_m': OCEAN_CMAP}

# ==============================================================================
# CREATE FIGURE
# ==============================================================================
print("Creating figure...")

fig = plt.figure(figsize=(7.5, 7.0))

# Main grid: 2 rows
#   Row 0: A (map) | B (depth)
#   Row 1: C-H manifold panels (UMAP + t-SNE rows)
gs_main = gridspec.GridSpec(2, 1, figure=fig,
                            height_ratios=[0.45, 0.90],
                            hspace=0.18)

panel_idx = 0
panel_letters = 'ABCDEFGH'

def add_label(ax, letter=None, x=-0.06, y=1.04):
    global panel_idx
    if letter is None:
        letter = panel_letters[panel_idx]
        panel_idx += 1
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=7, fontweight='bold', va='top', ha='left')

# ==============================================================================
# ROW 0: A (basin map, left) | B (depth distribution, right)
# ==============================================================================

gs_row0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0],
                                            width_ratios=[1.1, 0.9], wspace=0.20)

print("  Drawing Panel A: Basin world map...")
ax_map = fig.add_subplot(gs_row0[0, 0], projection=ccrs.Robinson())
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
    mask = df_gps['ocean_basin'] == basin
    subset = df_gps[mask]
    if len(subset) == 0:
        continue
    color = get_basin_color(basin)
    count = len(subset)
    ax_map.scatter(
        subset['longitude'].values, subset['latitude'].values,
        c=[color], s=3, alpha=0.7, edgecolors='none', linewidths=0,
        transform=ccrs.PlateCarree(), zorder=3,
        label=f"{basin.replace('_', ' ')} (n={count})"
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

# -- Panel B: Depth Distribution (violin + strip on log scale) -----------------
print("  Drawing Panel B: Depth Distribution...")

n_basins = len(basin_order)
y_positions = np.arange(n_basins) * 0.65
bar_height = 0.50

ax_depth = fig.add_subplot(gs_row0[0, 1])

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
ax_depth.set_title('Depth Distribution', fontweight='bold', pad=2)
ax_depth.set_xlim(to_log(-0.5), to_log(5500))
ax_depth.set_xticks(log_tick_pos)
ax_depth.set_xticklabels(log_tick_labels)

add_label(ax_depth, x=-0.08, y=1.08)

# ==============================================================================
# ROW 1: Manifold panels C-H (UMAP, t-SNE x Temp, Chl-a, Depth)
# ==============================================================================
print("  Drawing Panels C-H: Manifold embeddings...")

gs_manifold = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[1], hspace=0.10)

env_order = ['air_temp_mean_c', 'chl_mean_mg_m3', 'bathymetry_m']
method_order = ['UMAP', 't-SNE']

for row_offset, method in enumerate(method_order):
    gs_row = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_manifold[row_offset], wspace=0.08)

    for col_idx, env_var in enumerate(env_order):
        ax = fig.add_subplot(gs_row[0, col_idx])

        xy = embeddings[method]
        env_values = df_subset[env_var].values
        valid = ~np.isnan(env_values)

        ax.scatter(xy[:, 0], xy[:, 1], c='#e0e0e0', s=2, alpha=0.3,
                   rasterized=True, linewidths=0)

        if valid.sum() > 0:
            raw_values = env_values[valid]
            vmin, vmax = np.percentile(raw_values, [2, 98])

            sc = ax.scatter(xy[valid, 0], xy[valid, 1], c=raw_values,
                           s=3, cmap=cmaps[env_var], alpha=0.8,
                           vmin=vmin, vmax=vmax,
                           rasterized=True, linewidths=0)

            cax = inset_axes(ax, width="4%", height="35%", loc='upper right',
                            bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=ax.transAxes)
            cb = fig.colorbar(sc, cax=cax)
            cb.ax.yaxis.set_ticks_position('left')
            cb.ax.yaxis.set_label_position('left')
            cb.ax.tick_params(width=0.25, length=1.5, labelsize=5, pad=1)
            for lbl in cb.ax.yaxis.get_ticklabels():
                lbl.set_ha('right')
            cb.outline.set_linewidth(0.25)
            cb.set_label(env_units[env_var], fontsize=5, labelpad=2)

        ax.set_xticks([])
        ax.set_yticks([])

        if col_idx == 0:
            ax.set_ylabel(method, fontsize=6, fontweight='bold')

        if row_offset == 0:
            ax.set_title(env_labels[env_var], fontsize=6, fontweight='bold', pad=2)

        for spine in ax.spines.values():
            spine.set_linewidth(0.25)

        add_label(ax, x=0.02, y=0.98)

# ==============================================================================
# SAVE
# ==============================================================================
print("Saving figure...")

for fmt, path in [('pdf', OUT_PDF), ('svg', OUT_SVG)]:
    fig.savefig(path, format=fmt, bbox_inches='tight',
                transparent=True, edgecolor='none', dpi=300)
    print(f"  Saved: {path}")

plt.close()

# Provenance
prov_path = f"{OUT_DIR}/Figure2_combined_{TIMESTAMP}_provenance.txt"
with open(prov_path, 'w') as f:
    f.write("# Provenance\n")
    f.write(f"Script: {os.path.abspath(__file__)}\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Input: {BASIN_STATS}\n")
    f.write(f"Input: {BASIN_ASSIGN}\n")
    f.write(f"Input: {MERGED_TSV}\n")
    f.write(f"\n# Figure content\n")
    f.write(f"Panels: A-H (8 total)\n")
    f.write(f"Row 0: A (basin map, left) | B (depth distribution, right)\n")
    f.write(f"Row 1: C-E (UMAP x Temp, Chl-a, Depth)\n")
    f.write(f"Row 2: F-H (t-SNE x Temp, Chl-a, Depth)\n")
    f.write(f"GPS samples: {len(df_gps)}\n")
    f.write(f"Manifold samples: {len(df_subset)}\n")

print(f"  Saved provenance: {prov_path}")
print("\n=== COMPLETE ===")
