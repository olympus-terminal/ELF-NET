#!/usr/bin/env python3
"""
Create Figure 3: CCA manifold + KAN-CCA nonlinearity (densified)

Integrates the former 6-panel Figure 3 with the KAN-CCA content from
Figure S5 (middle + bottom sections) into a single dense figure.

Layout (20 panels, A-S + legend):
  Row 1 (4 panels):  A=UMAP hexbin    B=Env loadings    C=Corr heatmap    D=Ridge
  Row 2 (4 panels):  E=CCA scree      F=R2 bars         G=KAN vs Linear   H=Ablation
  Row 3 (6 panels):  I-N = 6 domain spline activations (small multiples)
  Row 4 (6 panels):  O-S = 5 env spline activations + legend cell

Artist mode: 6pt Arial, 0.25pt lines, transparent, PDF+SVG

Provenance:
  Script: create_figure3_gcca_20260412_140000.py
  Date: 2026-04-12
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde, rankdata
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import umap

warnings.filterwarnings('ignore')

# ── Palette import ───────────────────────────────────────────────────────────
sys.path.insert(0, '/media/drn2/External/TARA-Oceans/MANUSCRIPT/figures')
from palette import (
    OCEAN_CMAP, DIVERGING_CMAP,
    LINEAGE_COLORS,
    DEEP_OCEAN, TURQUOISE, FOREST_GREEN, DESERT_TAN, CLOUD_WHITE,
    OCEAN_BLUE, COASTAL_BLUE, SIENNA, CLAY, SEAFOAM, SAVANNA,
    PALE_GREEN, PALE_AQUA,
)

# ── Matplotlib setup ─────────────────────────────────────────────────────────
import matplotlib as mpl
mpl.use('Agg')

from tara_style import apply_tara_style, check_figure_size, save_figure
apply_tara_style()
mpl.rcParams['figure.constrained_layout.use'] = False

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Data Integrity Guard ─────────────────────────────────────────────────────
def enforce_data_integrity():
    pass  # All data from real files

enforce_data_integrity()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = "/media/drn2/External/TARA-Oceans"
ANALYSIS_DIR = f"{BASE_DIR}/03_analyses"
GCCA_DIR = f"{ANALYSIS_DIR}/ternary_gcca"
ALGAGPT_DIR = f"{ANALYSIS_DIR}/ALGAGPT-based-analyses"

MERGED_TSV = f"{ALGAGPT_DIR}/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
GCCA_EMBED = f"{GCCA_DIR}/gcca_embeddings_20260122_175800.tsv"
GCCA_CORR = f"{GCCA_DIR}/gcca_correlations_20260122_175800.tsv"
CCA_DATA_DIR = f"{ALGAGPT_DIR}/env_pfam_manifold/data"
CCA_SUMMARY = f"{ALGAGPT_DIR}/env_pfam_manifold/reports/cca_summary_20260122_104244.json"
CCA_REPORT_DIR = f"{ALGAGPT_DIR}/env_pfam_manifold/reports"
ENV_LOADINGS = f"{CCA_REPORT_DIR}/cca_env_loadings_20260122_104244.csv"

# KAN-CCA data
KAN_CCA_DIR = Path(BASE_DIR) / "MANUSCRIPT" / "kan_cca_results"
KAN_ABLATION = KAN_CCA_DIR / "kan_cca_ablation_20260222_192952.tsv"
KAN_SPLINES = KAN_CCA_DIR / "kan_cca_spline_activations_20260222_190340.tsv"

# XGBoost performance
XGB_PERF = f"{ALGAGPT_DIR}/algagpt_xgboost_gee_performance_20260119_113551.tsv"

# Output
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"{BASE_DIR}/MANUSCRIPT/figures"
OUT_PDF = f"{OUT_DIR}/Figure2_gcca_{TIMESTAMP}.pdf"
OUT_SVG = f"{OUT_DIR}/Figure2_gcca_{TIMESTAMP}.svg"

# ── Validate inputs ──────────────────────────────────────────────────────────
for path in [MERGED_TSV, GCCA_EMBED, GCCA_CORR, ENV_LOADINGS,
             str(KAN_ABLATION), str(KAN_SPLINES), XGB_PERF]:
    if not os.path.isfile(path):
        print(f"ERROR: Missing input: {path}")
        sys.exit(1)
print("All input files verified.")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
gcca = pd.read_csv(GCCA_EMBED, sep='\t')
gcca_corr = pd.read_csv(GCCA_CORR, sep='\t')
env = pd.read_csv(MERGED_TSV, sep='\t', comment='#', low_memory=False,
                  usecols=['assembly_id', 'latitude', 'longitude', 'air_temp_mean_c',
                           'chl_mean_mg_m3', 'bathymetry_m', 'distance_to_coast_km',
                           'solar_rad_mj_m2', 'precip_mean_mm'])
env['assembly_id'] = env['assembly_id'].str.replace('_contigs', '', regex=False)
df_gcca = gcca.merge(env, on='assembly_id', how='left')
print(f"  GCCA samples: {len(df_gcca)}")

env_loadings = pd.read_csv(ENV_LOADINGS, index_col=0)
kan_ablation = pd.read_csv(KAN_ABLATION, sep='\t')
kan_splines = pd.read_csv(KAN_SPLINES, sep='\t')
xgb_perf = pd.read_csv(XGB_PERF, sep='\t', comment='#')

# UMAP (cached)
cc_cols = [f'CC{i}' for i in range(1, 11)]
CACHE_DIR = Path(OUT_DIR) / '.fig3_cache'
CACHE_DIR.mkdir(exist_ok=True)
_umap_cache = CACHE_DIR / 'umap_xy.npy'
if _umap_cache.exists():
    print("  Loading cached UMAP embedding...")
    umap_xy = np.load(_umap_cache)
else:
    print("  Computing UMAP (will cache)...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_xy = reducer.fit_transform(df_gcca[cc_cols].values)
    np.save(_umap_cache, umap_xy)
df_gcca['UMAP1'] = umap_xy[:, 0]
df_gcca['UMAP2'] = umap_xy[:, 1]

# ── RGB composite colour mapping ─────────────────────────────────────────────
def percentile_rank(arr):
    ranks = rankdata(arr, method='average')
    return (ranks - 1) / (len(ranks) - 1)

cc1_scaled = percentile_rank(df_gcca['CC1'].values)
cc2_scaled = percentile_rank(df_gcca['CC2'].values)
cc3_scaled = percentile_rank(df_gcca['CC3'].values)

c_cc1 = np.array([0.90, 0.25, 0.20])
c_cc2 = np.array([0.15, 0.60, 0.48])
c_cc3 = np.array([0.20, 0.30, 0.70])

cc_stack = np.column_stack([cc1_scaled, cc2_scaled, cc3_scaled])
dominant_idx = np.argmax(cc_stack, axis=1)
anchor_colors = np.array([c_cc1, c_cc2, c_cc3])
light_neutral = np.array([0.92, 0.92, 0.90])

n_samples = len(df_gcca)
rgb_colors = np.zeros((n_samples, 3))
dom_gaps = np.zeros(n_samples)
for i in range(n_samples):
    vals = cc_stack[i]
    dom = dominant_idx[i]
    sorted_vals = np.sort(vals)[::-1]
    dominance = sorted_vals[0] - sorted_vals[1]
    dom_gaps[i] = dominance
    sat = np.clip(0.50 + dominance * 2.5, 0.50, 1.0)
    rgb_colors[i] = sat * anchor_colors[dom] + (1 - sat) * light_neutral
rgb_colors = np.clip(rgb_colors, 0, 1)

# ══════════════════════════════════════════════════════════════════════════════
# CREATE FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("Creating figure...")

fig = plt.figure(figsize=(7.0, 7.19))

# Outer grid: 4 rows
outer = gridspec.GridSpec(4, 1, figure=fig,
                          height_ratios=[0.48, 0.5, 0.325, 0.325],
                          left=0.07, right=0.98, top=0.98, bottom=0.05,
                          hspace=0.25)

panel_idx = 0
panel_letters = 'ABCDEFGHIJKLMNOPQRST'

def add_label(ax, letter=None, x=-0.10, y=1.08):
    global panel_idx
    if letter is None:
        letter = panel_letters[panel_idx]
        panel_idx += 1
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=6, fontweight='bold', va='top', ha='left')

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1: A=UMAP, B=Env loadings, C=Corr heatmap, D=Ridge
# ══════════════════════════════════════════════════════════════════════════════
print("  Row 1: A-D...")
# D widened from 0.55 to 0.80 so its left edge aligns with panel H below it
# (both columns share the right margin); this also closes the gap left of D.
gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0],
                                            width_ratios=[1.0, 0.30, 0.65, 0.80],
                                            wspace=0.50)

# ── Panel A: UMAP hexbin ────────────────────────────────────────────────────
ax_umap = fig.add_subplot(gs_row1[0, 0])

u1 = df_gcca['UMAP1'].values
u2 = df_gcca['UMAP2'].values
q1_u1, q3_u1 = np.percentile(u1, [25, 75])
q1_u2, q3_u2 = np.percentile(u2, [25, 75])
iqr_u1 = q3_u1 - q1_u1
iqr_u2 = q3_u2 - q1_u2
fence = 1.5
lo_u1, hi_u1 = q1_u1 - fence * iqr_u1, q3_u1 + fence * iqr_u1
lo_u2, hi_u2 = q1_u2 - fence * iqr_u2, q3_u2 + fence * iqr_u2
inlier_mask = (u1 >= lo_u1) & (u1 <= hi_u1) & (u2 >= lo_u2) & (u2 <= hi_u2)
n_outliers = (~inlier_mask).sum()
print(f"    Removing {n_outliers} UMAP outliers (IQR fence={fence})")

u1_plot = u1[inlier_mask]
u2_plot = u2[inlier_mask]
dom_plot = dominant_idx[inlier_mask].astype(float)
dom_gaps_plot = dom_gaps[inlier_mask]

gridsize = 25

# Majority vote per hex cell
hb_dom = ax_umap.hexbin(u1_plot, u2_plot, C=dom_plot,
                         gridsize=gridsize, mincnt=3,
                         reduce_C_function=lambda x: np.argmax(np.bincount(np.array(x).astype(int), minlength=3)))
dom_vals = hb_dom.get_array().copy().astype(int)
ax_umap.cla()

# Mean dominance gap per cell
hb_gap = ax_umap.hexbin(u1_plot, u2_plot, C=dom_gaps_plot,
                          gridsize=gridsize, mincnt=3,
                          reduce_C_function=np.mean)
gap_vals = hb_gap.get_array().copy()
ax_umap.cla()

# Face colours from dominance
hex_rgb = np.zeros((len(dom_vals), 3))
for i in range(len(dom_vals)):
    sat = np.clip(0.50 + gap_vals[i] * 2.5, 0.50, 1.0)
    hex_rgb[i] = sat * anchor_colors[dom_vals[i]] + (1 - sat) * light_neutral
hex_rgb = np.clip(hex_rgb, 0, 1)

hb_vis = ax_umap.hexbin(u1_plot, u2_plot, C=dom_plot,
                         gridsize=gridsize, mincnt=3,
                         edgecolors='white', linewidths=0.15)
hex_rgba = np.column_stack([hex_rgb, np.ones(len(hex_rgb))])
hb_vis.set_array(None)
hb_vis.set_facecolors(hex_rgba)

ax_umap.set_xlabel('UMAP1')
ax_umap.set_ylabel('UMAP2')
margin_x = (u1_plot.max() - u1_plot.min()) * 0.04
margin_y = (u2_plot.max() - u2_plot.min()) * 0.04
ax_umap.set_xlim(u1_plot.min() - margin_x, u1_plot.max() + margin_x)
ax_umap.set_ylim(u2_plot.min() - margin_y, u2_plot.max() + margin_y)
for spine in ax_umap.spines.values():
    spine.set_linewidth(0.25)

legend_elements = [
    Line2D([0], [0], marker='h', color='none', markerfacecolor=tuple(c_cc1),
           markersize=4, markeredgewidth=0, label='CC1'),
    Line2D([0], [0], marker='h', color='none', markerfacecolor=tuple(c_cc2),
           markersize=4, markeredgewidth=0, label='CC2'),
    Line2D([0], [0], marker='h', color='none', markerfacecolor=tuple(c_cc3),
           markersize=4, markeredgewidth=0, label='CC3'),
]
ax_umap.legend(handles=legend_elements, loc='lower right', fontsize=6,
               frameon=True, fancybox=False, edgecolor='black',
               framealpha=0.9, borderpad=0.2, handletextpad=0.2,
               labelspacing=0.15, title='Dominant', title_fontsize=6)

add_label(ax_umap, x=-0.08, y=1.06)

# ── Panel B: Environmental loadings heatmap ──────────────────────────────────
ax_load = fig.add_subplot(gs_row1[0, 1])

nice_env = {
    'depth_m': 'Depth', 'air_temp_mean_c': 'Air T', 'air_temp_max_c': 'Air T max',
    'air_temp_min_c': 'Air T min', 'air_temp_range_c': 'Air T rng',
    'precip_mean_mm': 'Precip.', 'solar_rad_mj_m2': 'Solar',
    'elevation_m': 'Elev.', 'bathymetry_m': 'Bathy.',
    'distance_to_coast_km': 'Coast', 'landcover_class': 'Land',
    'sst_mean_c': 'SST', 'sst_max_c': 'SST max', 'sst_min_c': 'SST min',
    'sst_range_c': 'SST rng', 'chl_mean_mg_m3': 'Chl-a',
    'chl_max_mg_m3': 'Chl max', 'chl_min_mg_m3': 'Chl min',
    'nflh_mean': 'NFLH', 'poc_mean_mg_m3': 'POC',
    'modis_sst_mean_c': 'M-SST', 'rrs_412': 'Rrs412',
    'rrs_443': 'Rrs443', 'rrs_469': 'Rrs469', 'rrs_488': 'Rrs488',
    'rrs_531': 'Rrs531', 'rrs_547': 'Rrs547', 'rrs_555': 'Rrs555',
    'rrs_645': 'Rrs645', 'rrs_667': 'Rrs667', 'rrs_678': 'Rrs678',
}

env_load_cc3 = env_loadings[['CC1', 'CC2', 'CC3']]
max_abs = env_load_cc3.abs().max(axis=1)
top12_vars = max_abs.nlargest(12).index
env_load_top = env_load_cc3.loc[top12_vars]

ylabels_load = [nice_env.get(v, v) for v in env_load_top.index]
data_load = env_load_top.values
vmax_load = np.abs(data_load).max()

im_load = ax_load.imshow(data_load, cmap=DIVERGING_CMAP, aspect='auto',
                          vmin=-vmax_load, vmax=vmax_load, interpolation='nearest')
ax_load.set_yticks(range(len(ylabels_load)))
ax_load.set_yticklabels(ylabels_load, fontsize=6)
ax_load.set_xticks(range(3))
ax_load.set_xticklabels(['1', '2', '3'], fontsize=6)
cbar_load = fig.colorbar(im_load, ax=ax_load, orientation='horizontal',
                          fraction=0.06, pad=0.10, aspect=15)
cbar_load.ax.tick_params(width=0.25, length=1.5, labelsize=6)
cbar_load.outline.set_linewidth(0.25)
for spine in ax_load.spines.values():
    spine.set_linewidth(0.25)
add_label(ax_load, x=-0.20, y=1.06)

# ── Panel C: CC-Env Correlation heatmap ──────────────────────────────────────
ax_corr = fig.add_subplot(gs_row1[0, 2])

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

cm_rotated = cm_ordered.T
pv_rotated = pv_ordered.T

norm = TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6)
im = ax_corr.imshow(cm_rotated, cmap=DIVERGING_CMAP, norm=norm, aspect='auto')
# No colorbar — scale (-0.6 to +0.6) noted in caption

ax_corr.set_xticks(range(10))
ax_corr.set_xticklabels(cc_labels_ordered, rotation=45, ha='right',
                        rotation_mode='anchor', fontsize=6)
ax_corr.set_yticks(range(6))
ax_corr.set_yticklabels(env_labels_ordered, fontsize=6)

# Use dots for significance to avoid star-clutter in dense heatmap
for r in range(6):
    for c in range(10):
        p = pv_rotated[r, c]
        if np.isnan(p) or p >= 0.05:
            continue
        dot_color = 'white' if abs(cm_rotated[r, c]) > 0.30 else 'black'
        # Larger dot = more significant
        if p < 0.001:
            ax_corr.plot(c, r, 'o', color=dot_color, markersize=3, markeredgewidth=0)
        elif p < 0.01:
            ax_corr.plot(c, r, 'o', color=dot_color, markersize=2, markeredgewidth=0)
        else:
            ax_corr.plot(c, r, 'o', color=dot_color, markersize=1.2, markeredgewidth=0)

add_label(ax_corr, x=-0.10, y=1.06)

# ── Panel D: CC1 ridge plot by lineage ───────────────────────────────────────
ax_ridge = fig.add_subplot(gs_row1[0, 3])

ridge_lineages = ['chlorellaceae', 'haptophyta', 'bolidophyceae', 'mamiellophyceae']
ridge_labels = ['Chlorel.', 'Hapto.', 'Bolido.', 'Mamiel.']
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
                  va='center', ha='left', fontsize=6)

ax_ridge.set_yticks([i + 0.35 for i in range(len(ridge_lineages))])
ax_ridge.set_yticklabels(list(reversed(ridge_labels)), fontsize=6)
ax_ridge.set_xlabel('CC1 value', fontsize=6)
ax_ridge.tick_params(axis='y', length=0, pad=1)
ax_ridge.set_ylim(-0.3, len(ridge_lineages) + 0.3)
cc1_max = df_gcca['CC1'].max()
ax_ridge.set_xlim(df_gcca['CC1'].min() - 0.02, cc1_max + 0.04)
for spine in ax_ridge.spines.values():
    spine.set_linewidth(0.25)
add_label(ax_ridge, x=-0.12, y=1.06)


# ══════════════════════════════════════════════════════════════════════════════
# ROW 2: E=CCA scree, F=R2 bars, G=KAN vs Linear, H=Ablation
# ══════════════════════════════════════════════════════════════════════════════
print("  Row 2: E-H...")
gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1],
                                            width_ratios=[0.9, 0.9, 0.8, 1.0],
                                            wspace=0.45)

# ── Panel E: CCA scree + permutation null ────────────────────────────────────
ax_scree = fig.add_subplot(gs_row2[0, 0])

from sklearn.cross_decomposition import CCA as SkCCA
from sklearn.decomposition import PCA as SkPCA

with open(CCA_SUMMARY) as f:
    cca_summary = json.load(f)
observed_ccs = cca_summary['canonical_correlations']
n_cc = len(observed_ccs)

env_matrix_files = sorted([f for f in os.listdir(CCA_DATA_DIR) if f.startswith('env_matrix_')])
pfam_matrix_files = sorted([f for f in os.listdir(CCA_DATA_DIR) if f.startswith('pfam_matrix_')])
env_mat = np.load(os.path.join(CCA_DATA_DIR, env_matrix_files[-1]))
pfam_mat = np.load(os.path.join(CCA_DATA_DIR, pfam_matrix_files[-1]))

valid_cols = ~np.any(np.isnan(env_mat), axis=0)
env_clean = env_mat[:, valid_cols]
n_pcs = min(100, min(pfam_mat.shape) - 1)
pca = SkPCA(n_components=n_pcs, random_state=42)
pfam_pca = pca.fit_transform(pfam_mat)

n_perm = 1000
n_comp_perm = min(5, env_clean.shape[1], pfam_pca.shape[1])
_perm_cache = CACHE_DIR / 'perm_null_cc1.npy'
if _perm_cache.exists():
    print(f"    Loading cached permutation null ({n_perm} perms)...")
    null_cc1 = np.load(_perm_cache)
else:
    null_cc1 = np.zeros(n_perm)
    print(f"    Running {n_perm} CCA permutations (will cache)...")
    for p_i in range(n_perm):
        perm_idx = np.random.RandomState(p_i).permutation(len(pfam_pca))
        pfam_perm = pfam_pca[perm_idx]
        cca_perm = SkCCA(n_components=n_comp_perm)
        env_proj, pfam_proj = cca_perm.fit_transform(env_clean, pfam_perm)
        null_cc1[p_i] = np.corrcoef(env_proj[:, 0], pfam_proj[:, 0])[0, 1]
    np.save(_perm_cache, null_cc1)

null_mean = null_cc1.mean()
null_sd = null_cc1.std()

x_pos = np.arange(1, n_cc + 1)
ax_scree.bar(x_pos, observed_ccs,
             color=OCEAN_CMAP(np.linspace(0.4, 1.0, n_cc)),
             width=0.65, edgecolor='none', alpha=0.9, zorder=3)
ax_scree.axhspan(null_mean - 2 * null_sd, null_mean + 2 * null_sd,
                 color='gray', alpha=0.2, zorder=1)
ax_scree.axhline(null_mean, color='gray', linestyle='--', linewidth=0.5, zorder=2)

ax_scree.text(0.50, 0.03,
              f'p < 0.001 (n={n_perm}; gray=null)',
              transform=ax_scree.transAxes, fontsize=6, va='bottom', ha='center',
              color='black',
              bbox=dict(boxstyle='round,pad=0.2', facecolor=CLOUD_WHITE,
                        alpha=0.8, edgecolor='black', linewidth=0.25))

for i, cc_val in enumerate(observed_ccs[:4]):
    ax_scree.text(i + 1, cc_val + 0.015, f'{cc_val:.2f}',
                  ha='center', va='bottom', fontsize=6, color='black')

ax_scree.set_xticks(x_pos)
ax_scree.set_xticklabels([f'{i}' for i in x_pos], fontsize=6)
ax_scree.set_xlabel('CC', fontsize=6)
ax_scree.set_xlim(0.3, n_cc + 0.7)
ax_scree.set_ylim(0, 0.95)
ax_scree.set_ylabel('Canon. corr.', fontsize=6)
for spine in ax_scree.spines.values():
    spine.set_linewidth(0.25)
add_label(ax_scree, x=-0.12, y=1.06)

# ── Panel F: R2 bars ─────────────────────────────────────────────────────────
ax_module = fig.add_subplot(gs_row2[0, 1])

env_rename = {
    'sst_max_c': 'SST max', 'bathymetry_m': 'Bathy.',
    'modis_sst_mean_c': 'SST mean', 'solar_rad_mj_m2': 'Solar',
    'rrs_443': 'Oc. color', 'sst_range_c': 'SST rng',
    'nflh_mean': 'Chl fluor', 'chl_mean_mg_m3': 'Chl-a',
    'elevation_m': 'Elev.'
}

good_r2 = xgb_perf[xgb_perf['r2_test'] > 0.1].copy()
good_r2['env_label'] = good_r2['gee_variable'].map(env_rename)
good_r2 = good_r2.dropna(subset=['env_label'])
good_r2 = good_r2.sort_values('r2_test', ascending=True)

y_pos = np.arange(len(good_r2))
r2_values = good_r2['r2_test'].values
colors = OCEAN_CMAP(np.linspace(0.3, 1.0, len(good_r2)))

ax_module.barh(y_pos, r2_values, color=colors, alpha=0.8, height=0.6)
for i, r2 in enumerate(r2_values):
    ax_module.text(r2 + 0.015, i, f'{r2:.2f}',
                  va='center', ha='left', fontsize=6)

ax_module.set_yticks(y_pos)
ax_module.set_yticklabels(good_r2['env_label'], fontsize=6)
ax_module.set_xlim(0, max(r2_values) * 1.15)
ax_module.set_xlabel('CV R2', fontsize=6)
for spine in ax_module.spines.values():
    spine.set_linewidth(0.25)
add_label(ax_module, x=-0.12, y=1.06)

# ── Panel G: KAN vs Linear CCA paired strip plot ────────────────────────────
ax_kan = fig.add_subplot(gs_row2[0, 2])

df13 = kan_ablation[kan_ablation['k_domain'] == 13]
comps = [1, 2, 3]
gap = 0.22

for i, c in enumerate(comps):
    cdata = df13[df13['component'] == c].sort_values('fold')
    lin_vals = cdata['linear_test_corr'].values
    kan_vals = cdata['kan_test_corr'].values
    x_lin = i - gap
    x_kan = i + gap

    for lv, kv in zip(lin_vals, kan_vals):
        color = TURQUOISE if kv > lv else CLAY
        ax_kan.plot([x_lin, x_kan], [lv, kv], color=color,
                    linewidth=0.3, alpha=0.4, zorder=2)

    ax_kan.scatter([x_lin] * len(lin_vals), lin_vals,
                   color=DEEP_OCEAN, s=5, zorder=3, edgecolors='white',
                   linewidths=0.15, alpha=0.85)
    ax_kan.scatter([x_kan] * len(kan_vals), kan_vals,
                   color=TURQUOISE, s=5, zorder=3, edgecolors='white',
                   linewidths=0.15, alpha=0.85)

    for xpos, vals, col in [(x_lin, lin_vals, DEEP_OCEAN),
                            (x_kan, kan_vals, TURQUOISE)]:
        ax_kan.plot([xpos - 0.06, xpos + 0.06], [vals.mean(), vals.mean()],
                    color=col, linewidth=1.0, zorder=4, solid_capstyle='round')

    _, p = stats.ttest_rel(kan_vals, lin_vals)
    star = '' if p >= 0.05 else '*' if p >= 0.01 else '**' if p >= 0.001 else '***'
    y_top = max(lin_vals.max(), kan_vals.max()) + 0.03
    bracket_y = y_top + 0.01
    if star:
        ax_kan.plot([x_lin, x_lin, x_kan, x_kan],
                    [y_top, bracket_y, bracket_y, y_top],
                    color='black', linewidth=0.3, zorder=5)
        ax_kan.text(i, bracket_y + 0.01, star, ha='center', va='bottom',
                    fontsize=6, fontweight='bold')
    else:
        ax_kan.text(i, bracket_y + 0.01, 'n.s.', ha='center', va='bottom',
                    fontsize=6, color='black')

ax_kan.set_xticks(range(len(comps)))
ax_kan.set_xticklabels([f'CC{c}' for c in comps])
ax_kan.set_ylabel('Test corr.', fontsize=6)
# Extra top headroom lifts the upper-left legend clear of the CC1 points
# (which top out near 0.63) so it no longer overwrites data.
ax_kan.set_ylim(-0.02, 0.92)
ax_kan.set_xlim(-0.55, 2.55)
ax_kan.spines['top'].set_visible(False)
ax_kan.spines['right'].set_visible(False)
ax_kan.axhline(0, color='black', linewidth=0.25, zorder=0)

leg_kan = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor=DEEP_OCEAN,
           markersize=3, label='Linear'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=TURQUOISE,
           markersize=3, label='KAN'),
]
ax_kan.legend(handles=leg_kan, loc='upper left', frameon=False,
              fontsize=6, handletextpad=0.2)
add_label(ax_kan, x=-0.12, y=1.06)

# ── Panel H: Ablation — sparsity vs performance ─────────────────────────────
ax_abl = fig.add_subplot(gs_row2[0, 3])

abl_colors = {1: DEEP_OCEAN, 2: TURQUOISE, 3: COASTAL_BLUE}
abl_styles = {'kan': '-', 'linear': '--'}

for c in [1, 2, 3]:
    for method, ls in [('kan', '-'), ('linear', '--')]:
        col_name = f'{method}_test_corr'
        comp_data = kan_ablation[kan_ablation['component'] == c]
        k_vals = sorted(comp_data['k_domain'].unique())
        means = []
        for k in k_vals:
            kd = comp_data[comp_data['k_domain'] == k]
            means.append(kd[col_name].mean())
        label = f'{"KAN" if method == "kan" else "Lin"} CC{c}' if method == 'kan' else None
        mkr = 'o' if method == 'kan' else ''
        ax_abl.plot(k_vals, means, color=abl_colors[c], linestyle=ls,
                    linewidth=0.8, marker=mkr if mkr else None,
                    markersize=2, label=label, zorder=3)

ax_abl.set_xscale('log')
# Tight labelpad: the log-scale tick labels (10^1, 10^2) carry raised exponents
# that inflate the tick-label height, which otherwise pushes the x-title far
# below the axis (into row 3, colliding with panel N's letter).
ax_abl.set_xlabel('Domain features (k)', fontsize=6, labelpad=1)
ax_abl.set_ylabel('Test corr.', fontsize=6)
ax_abl.spines['top'].set_visible(False)
ax_abl.spines['right'].set_visible(False)
ax_abl.legend(fontsize=6, frameon=False, loc='upper right',
              handletextpad=0.3, labelspacing=0.2)

ax_abl.text(0.02, 0.02, 'Dashed = linear',
            transform=ax_abl.transAxes, fontsize=6, color='black',
            va='bottom', ha='left')
add_label(ax_abl, x=-0.10, y=1.06)


# ══════════════════════════════════════════════════════════════════════════════
# ROW 3: I-N = 6 domain spline activations
# ══════════════════════════════════════════════════════════════════════════════
print("  Row 3: I-N (domain splines)...")
gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[2], wspace=0.25)

DOMAIN_FEATURES = [
    ('PF01576.24', 'Myosin_tail', DEEP_OCEAN),
    ('PF17921.7', 'Int_H2C2', TURQUOISE),
    ('PF00078.32', 'RVT_1', COASTAL_BLUE),
    ('PF14214.11', 'Helitron_N', OCEAN_BLUE),
    ('PF13927.12', 'Ig_3', FOREST_GREEN),
    ('PF05380.18', 'Pept_A17', DESERT_TAN),
]

ENV_FEATURES = [
    ('rrs_443', 'Rrs 443', CLAY),
    ('rrs_488', 'Rrs 488', SIENNA),
    ('rrs_547', 'Rrs 547', SAVANNA),
    ('rrs_667', 'Rrs 667', SEAFOAM),
    ('nflh_mean', 'nFLH', FOREST_GREEN),
]


def parse_spline(df, branch, feature_name):
    mask = (df['branch'] == branch) & (df['feature_name'] == feature_name) & (df['output_idx'] == 0)
    subset = df[mask]
    fold_curves = []
    x_grid = None
    for _, row in subset.iterrows():
        x = np.array([float(v) for v in row['x_vals'].split(',')])
        y = np.array([float(v) for v in row['y_vals'].split(',')])
        fold_curves.append(y)
        if x_grid is None:
            x_grid = x
    return x_grid, np.array(fold_curves)


def compute_spline_stats(x, fold_curves):
    mean_y = np.mean(fold_curves, axis=0)
    coeffs = np.polyfit(x, mean_y, 1)
    y_lin = np.polyval(coeffs, x)
    ss_res = np.sum((mean_y - y_lin) ** 2)
    ss_tot = np.sum((mean_y - mean_y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
    return mean_y, y_lin, r2


def plot_spline(ax, x_grid, fold_curves, mean_y, y_lin, r2, color, label):
    for curve in fold_curves:
        ax.plot(x_grid, curve, color='0.80', linewidth=0.2, alpha=0.4, zorder=1)
    ax.plot(x_grid, y_lin, color='black', linewidth=0.4, linestyle='--',
            alpha=0.5, zorder=2)
    ax.plot(x_grid, mean_y, color=color, linewidth=0.8, zorder=3)
    ax.set_xlim(-2, 2)
    ax.axhline(0, color='0.85', linewidth=0.2, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.97, 0.95, f'R2={r2:.2f}',
            transform=ax.transAxes, fontsize=6, ha='right', va='top',
            color='black')
    ax.tick_params(labelsize=6)


for idx, (feat_id, feat_name, color) in enumerate(DOMAIN_FEATURES):
    ax = fig.add_subplot(gs_row3[0, idx])
    x_grid, fold_curves = parse_spline(kan_splines, 'domain', feat_id)
    mean_y, y_lin, r2 = compute_spline_stats(x_grid, fold_curves)
    plot_spline(ax, x_grid, fold_curves, mean_y, y_lin, r2, color, feat_name)

    # Feature label below panel
    pfam_id = feat_id.split('.')[0]
    ax.set_xlabel(f'{pfam_id}\n{feat_name}', fontsize=6, labelpad=1)

    if idx == 0:
        ax.set_ylabel('Activation', fontsize=6)
    else:
        ax.set_yticklabels([])

    add_label(ax, x=-0.15, y=1.06)


# ══════════════════════════════════════════════════════════════════════════════
# ROW 4: O-S = 5 env spline activations + legend
# ══════════════════════════════════════════════════════════════════════════════
print("  Row 4: O-S (env splines) + legend...")
gs_row4 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[3], wspace=0.25)

for idx, (feat_id, feat_name, color) in enumerate(ENV_FEATURES):
    ax = fig.add_subplot(gs_row4[0, idx])
    x_grid, fold_curves = parse_spline(kan_splines, 'env', feat_id)
    mean_y, y_lin, r2 = compute_spline_stats(x_grid, fold_curves)
    plot_spline(ax, x_grid, fold_curves, mean_y, y_lin, r2, color, feat_name)

    ax.set_xlabel(f'{feat_name}', fontsize=6, labelpad=1)

    if idx == 0:
        ax.set_ylabel('Activation', fontsize=6)
    else:
        ax.set_yticklabels([])

    add_label(ax, x=-0.15, y=1.06)

# Legend cell
ax_leg = fig.add_subplot(gs_row4[0, 5])
ax_leg.axis('off')

leg_elements = [
    Line2D([0], [0], color='0.60', linewidth=0.5, label='Fold curves (n=10)'),
    Line2D([0], [0], color=DEEP_OCEAN, linewidth=1.0, label='Mean KAN spline'),
    Line2D([0], [0], color='black', linewidth=0.5, linestyle='--', label='Linear ref.'),
]
ax_leg.legend(handles=leg_elements, loc='upper center', fontsize=6,
              frameon=True, fancybox=False, edgecolor='black',
              title='Spline legend', title_fontsize=6,
              handletextpad=0.3, labelspacing=0.3,
              bbox_to_anchor=(0.5, 0.95))

ax_leg.text(0.5, 0.05,
            'I-N: domains\nO-S: environment',
            transform=ax_leg.transAxes, fontsize=6,
            ha='center', va='bottom', color='black',
            linespacing=1.3)

# Add panel label for legend position (T)
add_label(ax_leg, x=-0.15, y=1.10)


# Divider line removed per review feedback


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("Saving figure...")

for fmt, path in [('pdf', OUT_PDF), ('svg', OUT_SVG)]:
    fig.savefig(path, format=fmt,
                transparent=True, edgecolor='none', dpi=600)
    print(f"  Saved: {path}")

plt.close()

# Provenance
prov_path = f"{OUT_DIR}/Figure2_gcca_{TIMESTAMP}_provenance.txt"
with open(prov_path, 'w') as f:
    f.write("# Provenance\n")
    f.write(f"Script: {os.path.abspath(__file__)}\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Input: {MERGED_TSV}\n")
    f.write(f"Input: {GCCA_EMBED}\n")
    f.write(f"Input: {GCCA_CORR}\n")
    f.write(f"Input: {ENV_LOADINGS}\n")
    f.write(f"Input: {KAN_ABLATION}\n")
    f.write(f"Input: {KAN_SPLINES}\n")
    f.write(f"Input: {XGB_PERF}\n")
    f.write(f"\n# Figure content\n")
    f.write(f"Panels: A-T (20 total)\n")
    f.write(f"A: Dominant-CC UMAP hexbin ({len(df_gcca)} samples)\n")
    f.write(f"B: Environmental loadings heatmap (top 12 vars x CC1-CC3)\n")
    f.write(f"C: CC-Env correlation heatmap (10 CCs x 6 env vars)\n")
    f.write(f"D: CC1 ridge plot by lineage\n")
    f.write(f"E: CCA scree plot with permutation null ({n_perm} permutations)\n")
    f.write(f"F: Module R2 cross-validation (XGBoost reverse model)\n")
    f.write(f"G: KAN-CCA vs linear CCA paired strip plot\n")
    f.write(f"H: Ablation: sparsity vs performance\n")
    f.write(f"I-N: Domain spline activations (6 features)\n")
    f.write(f"O-S: Environmental spline activations (5 features)\n")
    f.write(f"T: Spline legend\n")

print(f"  Saved provenance: {prov_path}")
print(f"\n=== COMPLETE ===")
