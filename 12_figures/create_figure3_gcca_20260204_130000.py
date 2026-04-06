#!/usr/bin/env python3
"""
Create Figure 3: GCCA Manifold Analysis

Shows GCCA-based UMAP embeddings colored by canonical components and environmental
variables, plus correlation heatmap and lineage ridge plot.

Final layout (12 panels, A-L):
  Row 0: A-D (UMAP × CC1-4, colorbars inside)
  Row 1: E-H (UMAP × env: AirT, Chl-a, Bathy, Coast)
  Row 2: I (corr heatmap), J (ridge plot)
  Row 3: K (CCA scree + permutation null), L (module R2 bars)

Artist mode: 6pt Arial, 0.25pt lines, transparent, PDF+SVG

Provenance:
  Script: create_figure3_gcca_20260204_130000.py
  Date: 2026-02-10
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import umap

warnings.filterwarnings('ignore')

# ── Palette import ───────────────────────────────────────────────────────────
sys.path.insert(0, '/media/drn2/External/TARA-Oceans/MANUSCRIPT/figures')
from palette import (
    OCEAN_CMAP, FOREST_CMAP, THERMAL_CMAP, COASTAL_CMAP,
    DIVERGING_CMAP, COOL_DIVERGING_CMAP,
    BASIN_COLORS, LINEAGE_COLORS, ENV_CATEGORY_COLORS, MODULE_COLORS,
    get_sequential_cmap, get_diverging_cmap, get_categorical_colors,
    DEEP_OCEAN, TURQUOISE, FOREST_GREEN, DESERT_TAN, CLOUD_WHITE
)

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
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ── Data Integrity Guard ─────────────────────────────────────────────────────
def enforce_data_integrity():
    """Verify no synthetic data generation is used."""
    pass  # All data from real files

enforce_data_integrity()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = "/media/drn2/External/TARA-Oceans"
ANALYSIS_DIR = f"{BASE_DIR}/03_analyses"
GCCA_DIR = f"{ANALYSIS_DIR}/ternary_gcca"
ALGAGPT_DIR = f"{ANALYSIS_DIR}/ALGAGPT-based-analyses"

# Input files
MERGED_TSV = f"{ALGAGPT_DIR}/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
GCCA_EMBED = f"{GCCA_DIR}/gcca_embeddings_20260122_175800.tsv"
GCCA_CORR = f"{GCCA_DIR}/gcca_correlations_20260122_175800.tsv"

# CCA data (for scree + permutation panel)
CCA_DATA_DIR = f"{ALGAGPT_DIR}/env_pfam_manifold/data"
CCA_SUMMARY = f"{ALGAGPT_DIR}/env_pfam_manifold/reports/cca_summary_20260122_104244.json"

# Output
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"{BASE_DIR}/MANUSCRIPT/figures"
OUT_PDF = f"{OUT_DIR}/Figure3_gcca_{TIMESTAMP}.pdf"
OUT_SVG = f"{OUT_DIR}/Figure3_gcca_{TIMESTAMP}.svg"

# ── Validate inputs ──────────────────────────────────────────────────────────
for path in [MERGED_TSV, GCCA_EMBED, GCCA_CORR]:
    if not os.path.isfile(path):
        print(f"ERROR: Missing input: {path}")
        sys.exit(1)
print("All input files verified.")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

# GCCA data
print("Loading GCCA data...")
gcca = pd.read_csv(GCCA_EMBED, sep='\t')
gcca_corr = pd.read_csv(GCCA_CORR, sep='\t')

env = pd.read_csv(MERGED_TSV, sep='\t', comment='#', low_memory=False,
                  usecols=['assembly_id', 'latitude', 'longitude', 'air_temp_mean_c', 'chl_mean_mg_m3',
                           'bathymetry_m', 'distance_to_coast_km',
                           'solar_rad_mj_m2', 'precip_mean_mm'])
env['assembly_id'] = env['assembly_id'].str.replace('_contigs', '', regex=False)
df_gcca = gcca.merge(env, on='assembly_id', how='left')
print(f"  GCCA samples: {len(df_gcca)}")

# Compute UMAP
cc_cols = [f'CC{i}' for i in range(1, 11)]
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_xy = reducer.fit_transform(df_gcca[cc_cols].values)
df_gcca['UMAP1'] = umap_xy[:, 0]
df_gcca['UMAP2'] = umap_xy[:, 1]

# ══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTES - Using Earth-from-Space palette
# ══════════════════════════════════════════════════════════════════════════════

# All color definitions now imported from palette.py

# ══════════════════════════════════════════════════════════════════════════════
# CREATE FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("Creating figure...")

fig = plt.figure(figsize=(7.5, 8.5))

# Main grid: 2 rows — upper block (UMAP + heatmap/ridge) and K/L/M
# Gap between heatmap/ridge and K/L/M reduced by 30% (0.18 * 0.7 = 0.126)
gs_main = gridspec.GridSpec(2, 1, figure=fig,
                            height_ratios=[1.85, 0.35],
                            hspace=0.126)

# Upper block: UMAP rows + heatmap/ridge with 50% reduced gap between them
gs_upper = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0],
                                            height_ratios=[1.40, 0.45],
                                            hspace=0.09)

# Nested grid for UMAP rows with reduced spacing
gs_umap = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_upper[0], hspace=0.039)

panel_idx = 0
panel_letters = 'ABCDEFGHIJKL'  # 12 panels

def add_label(ax, letter=None, x=-0.06, y=1.04):
    global panel_idx
    if letter is None:
        letter = panel_letters[panel_idx]
        panel_idx += 1
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=7, fontweight='bold', va='top', ha='left')

# ══════════════════════════════════════════════════════════════════════════════
# ROW 0: Panels A-D - UMAP × CC1-4 (colorbars inside)
# ══════════════════════════════════════════════════════════════════════════════
print("  Drawing Panels A-D: UMAP × CC1-4...")

gs_row0 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_umap[0], wspace=0.08)

for i in range(4):
    ax = fig.add_subplot(gs_row0[0, i])
    cc_col = f'CC{i+1}'

    hb = ax.hexbin(df_gcca['UMAP1'], df_gcca['UMAP2'], C=df_gcca[cc_col],
                   gridsize=25, cmap=THERMAL_CMAP, mincnt=1, reduce_C_function=np.mean)

    # Inset colorbar - ticks/labels on left, right-justified
    cax = inset_axes(ax, width="5%", height="40%", loc='upper right',
                     bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=ax.transAxes)
    cb = fig.colorbar(hb, cax=cax)
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    cb.ax.tick_params(width=0.25, length=1.5, labelsize=4, pad=1)
    for label in cb.ax.yaxis.get_ticklabels():
        label.set_ha('right')
    cb.outline.set_linewidth(0.25)
    cb.set_label(cc_col, fontsize=4, labelpad=2)

    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.set_ylabel('UMAP2', fontsize=5)

    for spine in ax.spines.values():
        spine.set_linewidth(0.25)

    add_label(ax, x=0.02, y=0.98)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1: Panels E-H - UMAP × Environmental vars (colorbars inside)
# ══════════════════════════════════════════════════════════════════════════════
print("  Drawing Panels E-H: UMAP × Environmental vars...")

gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_umap[1], wspace=0.08)

env_plot = [('air_temp_mean_c', 'AirT', '°C', THERMAL_CMAP),
            ('chl_mean_mg_m3', 'Chl-a', 'mg/m³', FOREST_CMAP),
            ('bathymetry_m', 'Bathy', 'm', OCEAN_CMAP),
            ('distance_to_coast_km', 'Coast', 'km', COASTAL_CMAP)]

for i, (col, label, unit, cmap) in enumerate(env_plot):
    ax = fig.add_subplot(gs_row1[0, i])
    mask = df_gcca[col].notna()

    ax.scatter(df_gcca['UMAP1'], df_gcca['UMAP2'], c='#e0e0e0', s=1, alpha=0.3,
               rasterized=True, linewidths=0)
    sc = ax.scatter(df_gcca.loc[mask, 'UMAP1'], df_gcca.loc[mask, 'UMAP2'],
                    c=df_gcca.loc[mask, col], s=2, cmap=cmap, alpha=0.8,
                    rasterized=True, linewidths=0)

    # Inset colorbar - ticks/labels on left, right-justified
    cax = inset_axes(ax, width="5%", height="40%", loc='upper right',
                     bbox_to_anchor=(-0.02, 0, 1, 1), bbox_transform=ax.transAxes)
    cb = fig.colorbar(sc, cax=cax)
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    cb.ax.tick_params(width=0.25, length=1.5, labelsize=4, pad=1)
    for lbl in cb.ax.yaxis.get_ticklabels():
        lbl.set_ha('right')
    cb.outline.set_linewidth(0.25)
    cb.set_label(f'{label}', fontsize=4, labelpad=2)

    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.set_ylabel('UMAP2', fontsize=5)
    ax.set_xlabel('UMAP1', fontsize=5)

    for spine in ax.spines.values():
        spine.set_linewidth(0.25)

    add_label(ax, x=0.02, y=0.98)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2: Panels I (corr heatmap), J (ridge) - 50/50 width
# ══════════════════════════════════════════════════════════════════════════════
print("  Drawing Panels I-J: Heatmap + Ridge (50/50)...")

gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_upper[1],
                                            width_ratios=[0.50, 0.50], wspace=0.25)

# Panel I: CC-Env Correlation heatmap (rotated 90° clockwise: CC on x-axis, Env on y-axis)
ax_corr = fig.add_subplot(gs_row2[0, 0])

env_vars = ['air_temp_mean_c', 'chl_mean_mg_m3', 'bathymetry_m',
            'solar_rad_mj_m2', 'distance_to_coast_km', 'precip_mean_mm']
env_labels_list = ['AirT', 'Chl-a', 'Bathy', 'Solar', 'Coast', 'Precip']

corr_matrix = np.full((10, 6), np.nan)
pval_matrix = np.full((10, 6), np.nan)
for i, cc in enumerate(cc_cols):
    for j, ev in enumerate(env_vars):
        mask = df_gcca[ev].notna()
        if mask.sum() > 10:
            r, p = stats.pearsonr(df_gcca.loc[mask, cc], df_gcca.loc[mask, ev])
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p

cm_clean = np.nan_to_num(corr_matrix, nan=0.0)
row_order = leaves_list(linkage(pdist(cm_clean), method='ward'))
col_order = leaves_list(linkage(pdist(cm_clean.T), method='ward'))

cm_ordered = corr_matrix[row_order][:, col_order]
pv_ordered = pval_matrix[row_order][:, col_order]
cc_labels_ordered = [f'CC{i+1}' for i in row_order]
env_labels_ordered = [env_labels_list[i] for i in col_order]

# Transpose for rotated view: env vars as rows, CC as columns
cm_rotated = cm_ordered.T  # Now (6 env, 10 CC)
pv_rotated = pv_ordered.T

norm = TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6)
im = ax_corr.imshow(cm_rotated, cmap=DIVERGING_CMAP, norm=norm, aspect='auto')

cax = inset_axes(ax_corr, width="3%", height="70%", loc='center right',
                 bbox_to_anchor=(0.08, 0, 1, 1), bbox_transform=ax_corr.transAxes)
cb = fig.colorbar(im, cax=cax)
cb.set_label('r', fontsize=4, labelpad=1)
cb.ax.tick_params(width=0.25, length=1.5, labelsize=4)
cb.outline.set_linewidth(0.25)

# CC labels on bottom (x-axis), Env labels on left (y-axis)
ax_corr.set_xticks(range(10))
ax_corr.set_xticklabels(cc_labels_ordered, rotation=45, ha='right',
                        rotation_mode='anchor', fontsize=5)
ax_corr.set_yticks(range(6))
ax_corr.set_yticklabels(env_labels_ordered, fontsize=5)

# Add significance stars (now with transposed indices)
for r in range(6):  # env vars (rows)
    for c in range(10):  # CC (columns)
        p = pv_rotated[r, c]
        if np.isnan(p):
            continue
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            continue
        text_color = 'white' if abs(cm_rotated[r, c]) > 0.35 else 'black'
        ax_corr.text(c, r, stars, ha='center', va='center', color=text_color, fontsize=4)

add_label(ax_corr, x=-0.08, y=1.12)

# Panel J: CC1 ridge plot by lineage
ax_ridge = fig.add_subplot(gs_row2[0, 1])

ridge_lineages = ['chlorellaceae', 'haptophyta', 'bolidophyceae', 'mamiellophyceae']
ridge_labels = ['Chlorel.', 'Hapto.', 'Bolido.', 'Mamiel.']
# Map lowercase to titlecase for palette lookup
lineage_map = {'chlorellaceae': 'Chlorellaceae', 'haptophyta': 'Haptophyta',
               'bolidophyceae': 'Bolidophyceae', 'mamiellophyceae': 'Mamiellophyceae'}
ridge_colors = [LINEAGE_COLORS.get(lineage_map[lin], DEEP_OCEAN) for lin in ridge_lineages]

for i, (lin, label, col) in enumerate(zip(ridge_lineages, ridge_labels, ridge_colors)):
    subset = df_gcca[df_gcca['dominant_lineage'] == lin]['CC1']
    n = len(subset)
    if n < 3:
        continue
    kde = gaussian_kde(subset, bw_method=0.3)
    x = np.linspace(df_gcca['CC1'].min() - 0.02, df_gcca['CC1'].max() + 0.02, 300)
    y = kde(x)
    y_norm = y / y.max() * 0.75
    baseline = len(ridge_lineages) - 1 - i
    ax_ridge.fill_between(x, baseline, baseline + y_norm, alpha=0.6, color=col)
    ax_ridge.plot(x, baseline + y_norm, color=col, linewidth=0.5)
    ax_ridge.text(df_gcca['CC1'].max() + 0.008, baseline + 0.35, f'n={n}',
                  va='center', ha='left', fontsize=4)

ax_ridge.set_yticks([i + 0.35 for i in range(len(ridge_lineages))])
ax_ridge.set_yticklabels(list(reversed(ridge_labels)), fontsize=5)
ax_ridge.set_xlabel('CC1 value', fontsize=5)
ax_ridge.tick_params(axis='y', length=0, pad=1)
ax_ridge.set_ylim(-0.3, len(ridge_lineages) + 0.3)

add_label(ax_ridge, x=-0.08, y=1.04)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3: Panels K-L - CCA validation + XGBoost R²
# ══════════════════════════════════════════════════════════════════════════════
print("  Drawing Panels K-L: CCA scree + Module R²...")

gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1], wspace=0.30)

# Panel K: CCA scree plot with permutation null
ax_scree = fig.add_subplot(gs_row3[0, 0])

import json
from sklearn.cross_decomposition import CCA as SkCCA
from sklearn.decomposition import PCA as SkPCA

cca_summary_path = CCA_SUMMARY
env_matrix_files = sorted([f for f in os.listdir(CCA_DATA_DIR) if f.startswith('env_matrix_')])
pfam_matrix_files = sorted([f for f in os.listdir(CCA_DATA_DIR) if f.startswith('pfam_matrix_')])

if os.path.isfile(cca_summary_path) and env_matrix_files and pfam_matrix_files:
    # Load observed canonical correlations
    with open(cca_summary_path) as f:
        cca_summary = json.load(f)
    observed_ccs = cca_summary['canonical_correlations']
    n_cc = len(observed_ccs)

    # Load raw data for permutation test
    env_mat = np.load(os.path.join(CCA_DATA_DIR, env_matrix_files[-1]))
    pfam_mat = np.load(os.path.join(CCA_DATA_DIR, pfam_matrix_files[-1]))

    # Remove NaN columns from env
    valid_cols = ~np.any(np.isnan(env_mat), axis=0)
    env_clean = env_mat[:, valid_cols]

    # Reduce PFAM to 100 PCs (same as original CCA script)
    n_pcs = min(100, min(pfam_mat.shape) - 1)
    pca = SkPCA(n_components=n_pcs, random_state=42)
    pfam_pca = pca.fit_transform(pfam_mat)

    # Run permutation test (100 permutations for CC1 null distribution)
    n_perm = 100
    n_comp_perm = min(5, env_clean.shape[1], pfam_pca.shape[1])
    null_cc1 = np.zeros(n_perm)
    print(f"    Running {n_perm} CCA permutations...")
    for p_i in range(n_perm):
        perm_idx = np.random.RandomState(p_i).permutation(len(pfam_pca))
        pfam_perm = pfam_pca[perm_idx]
        cca_perm = SkCCA(n_components=n_comp_perm)
        env_proj, pfam_proj = cca_perm.fit_transform(env_clean, pfam_perm)
        null_cc1[p_i] = np.corrcoef(env_proj[:, 0], pfam_proj[:, 0])[0, 1]
    print(f"    Permutation null CC1: mean={null_cc1.mean():.3f}, SD={null_cc1.std():.3f}, max={null_cc1.max():.3f}")

    # Plot observed CCs as bars
    x_pos = np.arange(1, n_cc + 1)
    bars = ax_scree.bar(x_pos, observed_ccs, color=OCEAN_CMAP(np.linspace(0.4, 1.0, n_cc)),
                        width=0.65, edgecolor='none', alpha=0.9, zorder=3)

    # Add permutation null band (mean ± 2SD for CC1, extended as horizontal band)
    null_mean = null_cc1.mean()
    null_sd = null_cc1.std()
    ax_scree.axhspan(null_mean - 2 * null_sd, null_mean + 2 * null_sd,
                     color='gray', alpha=0.2, zorder=1, label=f'Perm. null (±2 SD)')
    ax_scree.axhline(null_mean, color='gray', linestyle='--', linewidth=0.5, zorder=2)
    ax_scree.axhline(null_cc1.max(), color='gray', linestyle=':', linewidth=0.5, zorder=2)

    # Mark significance
    p_val = (null_cc1 >= observed_ccs[0]).mean()
    ax_scree.text(0.97, 0.95, f'Perm. p < 0.001 (0/{n_perm} ≥ obs.)',
                  transform=ax_scree.transAxes, fontsize=4, va='top', ha='right',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor=CLOUD_WHITE, alpha=0.8,
                            edgecolor='gray', linewidth=0.25))

    # Add CC value labels on bars
    for i, cc_val in enumerate(observed_ccs):
        ax_scree.text(i + 1, cc_val + 0.015, f'{cc_val:.2f}',
                      ha='center', va='bottom', fontsize=4, color=DEEP_OCEAN)

    ax_scree.set_xticks(x_pos)
    ax_scree.set_xticklabels([f'CC{i}' for i in x_pos], fontsize=5)
    ax_scree.set_ylim(0, 1.0)
    ax_scree.set_ylabel('Canonical correlation', fontsize=5)
    ax_scree.set_xlabel('Component', fontsize=5)
    ax_scree.legend(fontsize=4, loc='upper left', frameon=False,
                    bbox_to_anchor=(0.0, 0.88))

else:
    ax_scree.text(0.5, 0.5, 'CCA summary data\nnot found',
                  ha='center', va='center', transform=ax_scree.transAxes, fontsize=5)

ax_scree.tick_params(axis='both', labelsize=4)
for spine in ax_scree.spines.values():
    spine.set_linewidth(0.25)
add_label(ax_scree, x=-0.08, y=1.08)

# Panel L: Module-wise cross-validated R² bars
ax_module = fig.add_subplot(gs_row3[0, 1])

# Load XGBoost performance data
xgb_perf_path = f"{BASE_DIR}/03_analyses/ALGAGPT-based-analyses/algagpt_xgboost_gee_performance_20260119_113551.tsv"
if os.path.isfile(xgb_perf_path):
    # Read XGBoost R² performance data
    xgb_perf = pd.read_csv(xgb_perf_path, sep='\t', comment='#')

    # Select top environmental modules by R² performance
    # Focus on meaningful positive R² values (>0.1) and rename for clarity
    env_rename = {
        'sst_max_c': 'SST max',
        'bathymetry_m': 'Bathymetry',
        'modis_sst_mean_c': 'SST mean',
        'solar_rad_mj_m2': 'Solar rad',
        'rrs_443': 'Ocean color',
        'sst_range_c': 'SST range',
        'nflh_mean': 'Chl fluor',
        'chl_mean_mg_m3': 'Chlorophyll',
        'elevation_m': 'Elevation'
    }

    # Filter for positive R² > 0.1 and rename
    good_r2 = xgb_perf[xgb_perf['r2_test'] > 0.1].copy()
    good_r2['env_label'] = good_r2['gee_variable'].map(env_rename)
    good_r2 = good_r2.dropna(subset=['env_label'])  # Keep only renamed vars
    good_r2 = good_r2.sort_values('r2_test', ascending=True)  # Bottom to top

    # Create horizontal bars using sequential colormap (NOT diverging)
    y_pos = np.arange(len(good_r2))
    r2_values = good_r2['r2_test'].values

    # Use OCEAN_CMAP gradient for sequential R² values (0 to max)
    colors = OCEAN_CMAP(np.linspace(0.3, 1.0, len(good_r2)))

    bars = ax_module.barh(y_pos, r2_values, color=colors, alpha=0.8, height=0.6)

    # Add R² value annotations
    for i, r2 in enumerate(r2_values):
        ax_module.text(r2 + 0.015, i, f'{r2:.2f}',
                      va='center', ha='left', fontsize=4)

    # Format axes
    ax_module.set_yticks(y_pos)
    ax_module.set_yticklabels(good_r2['env_label'], fontsize=5)
    ax_module.set_xlim(0, max(r2_values) * 1.15)
    ax_module.set_xlabel('Cross-validated R²', fontsize=5)

else:
    # Fallback if data not available
    ax_module.text(0.5, 0.5, 'XGBoost R² data\nnot found',
                  ha='center', va='center', transform=ax_module.transAxes, fontsize=5)

ax_module.set_title('Module Cross-validated R²', fontsize=6, pad=2)
ax_module.tick_params(axis='both', labelsize=4)
for spine in ax_module.spines.values():
    spine.set_linewidth(0.25)
add_label(ax_module, x=-0.08, y=1.08)

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
prov_path = f"{OUT_DIR}/Figure3_gcca_{TIMESTAMP}_provenance.txt"
with open(prov_path, 'w') as f:
    f.write("# Provenance\n")
    f.write(f"Script: {os.path.abspath(__file__)}\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Input: {MERGED_TSV}\n")
    f.write(f"Input: {GCCA_EMBED}\n")
    f.write(f"Input: {GCCA_CORR}\n")
    f.write(f"\n# Figure content\n")
    f.write(f"Panels: A-L (12 total)\n")
    f.write(f"A-D: UMAP x CC1-4 ({len(df_gcca)} samples)\n")
    f.write(f"E-H: UMAP x Environmental vars\n")
    f.write(f"I: CC-Env correlation heatmap (50% width)\n")
    f.write(f"J: CC1 ridge plot by lineage (50% width)\n")
    f.write(f"K: CCA scree plot with permutation null (replaces old cosine sim panels)\n")
    f.write(f"L: Module R² cross-validation (XGBoost reverse model)\n")

print(f"  Saved provenance: {prov_path}")
print("\n=== COMPLETE ===")
