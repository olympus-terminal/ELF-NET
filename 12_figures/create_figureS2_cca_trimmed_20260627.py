#!/usr/bin/env python3
"""
Merged Figure S3: Comprehensive Canonical Correlation Analysis

Combines old Figure S3 (CCA cross-validation, 5 panels) and old Figure S9
(extended CCA, 5 panels) into a single 10-panel comprehensive figure.

Layout (4 rows):
  Row 1: (A) CCs bar + permutation null | (B) Permutation test CC1 | (C) Shared variance
  Row 2: (D) Env loadings top-15 heatmap | (E) CC1 scatter env vs PFAM
  Row 3: (F) CC1 top-20 PFAM loadings | (G) CC2/CC3 top-10 PFAM loadings (stacked)
  Row 4: (H) World map CC1 scores (full width)
  Row 5: (I) Cross-validated shrinkage | (J) GCCA comparison

Provenance:
  Input: cca_summary_20260122_104244.json (observed CCs)
  Input: cca_permutation.tsv (permutation null)
  Input: cca_env_loadings_20260122_104244.csv (env loadings CC1-CC5)
  Input: cca_env_embedding_20260122_104244.npy (env CCA scores)
  Input: cca_pfam_embedding_20260122_104244.npy (pfam CCA scores)
  Input: cca_pfam_loadings_20260212_100221.tsv (PFAM loadings through PCA)
  Input: cca_cc2_cc5_interpretation_20260212_100532.tsv (CC interpretations)
  Input: cca_shrinkage_cv_20260212_101811.tsv (shrinkage summary)
  Input: cca_shrinkage_cv_distribution_20260212_101811.tsv (CV distributions)
  Input: gcca_ternary_comparison_20260212_102910.tsv (GCCA comparison)
  Input: algagpt_gee_pfam_merged_SMART_20260119_100639.tsv (GPS coords)
  Output: figures/FigureS3_comprehensive_cca_*.pdf, *.svg
  Date: 2026-02-12
"""

import numpy as np
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from datetime import datetime
import sys

# ============================================================================
# Data integrity enforcement
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
MANUSCRIPT_DIR = SCRIPT_DIR.parent
BASE_DIR = Path("/media/drn2/External/TARA-Oceans")
CCA_DATA_DIR = BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "env_pfam_manifold" / "data"
CCA_REPORT_DIR = BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "env_pfam_manifold" / "reports"
SOURCE_DATA_DIR = MANUSCRIPT_DIR / "source_data"
RALPH4_DIR = MANUSCRIPT_DIR / "ralph4_statistical_reanalysis"

sys.path.insert(0, str(BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "env_pfam_manifold"))
from DataIntegrityGuard import enforce_data_integrity, validate_input_source, create_provenance_header
enforce_data_integrity()

# Import palette
sys.path.insert(0, str(SCRIPT_DIR))
from palette import (
    get_diverging_cmap, get_sequential_cmap,
    DEEP_OCEAN, OCEAN_BLUE, COASTAL_BLUE, TURQUOISE, SEAFOAM, PALE_AQUA,
    FOREST_GREEN, SAVANNA, DESERT_TAN, CLAY, SIENNA,
    CLOUD_WHITE, ABYSS, SAND, PALE_GREEN,
    EARTH_CATEGORICAL, OCEAN_CMAP, DIVERGING_CMAP, THERMAL_CMAP,
    ENV_CATEGORY_COLORS,
)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# FIGURE_PROTOCOL.md: matplotlib settings
# ============================================================================
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

# ============================================================================
# LOAD ALL DATA
# ============================================================================
print("=" * 70)
print("LOADING DATA FOR COMPREHENSIVE CCA FIGURE S3")
print("=" * 70)

# --- CCA summary (observed canonical correlations) ---
cca_summary_path = CCA_REPORT_DIR / "cca_summary_20260122_104244.json"
validate_input_source(cca_summary_path)
with open(cca_summary_path) as f:
    cca_summary = json.load(f)
observed_ccs = np.array(cca_summary['canonical_correlations'])
print(f"  Observed CCs: {observed_ccs}")

# --- Permutation test results ---
# (Permutation null data not needed in trimmed S2 — panels A-C moved to Figure S1.)

# --- Env loadings ---
env_load_path = CCA_REPORT_DIR / "cca_env_loadings_20260122_104244.csv"
validate_input_source(env_load_path)
env_loadings = pd.read_csv(env_load_path, index_col=0)
print(f"  Env loadings: {env_loadings.shape}")

# --- CCA embeddings (for CC1 scatter) ---
env_emb_path = CCA_DATA_DIR / "cca_env_embedding_20260122_104244.npy"
pfam_emb_path = CCA_DATA_DIR / "cca_pfam_embedding_20260122_104244.npy"
sample_ids_path = CCA_DATA_DIR / "sample_ids_20260122_101559.npy"
env_matrix_path = CCA_DATA_DIR / "env_matrix_20260122_101559.npy"
env_cols_path = CCA_DATA_DIR / "env_columns_20260122_101559.txt"

for p in [env_emb_path, pfam_emb_path, sample_ids_path, env_matrix_path, env_cols_path]:
    validate_input_source(p)

env_embedding = np.load(env_emb_path)
pfam_embedding = np.load(pfam_emb_path)
sample_ids = np.load(sample_ids_path, allow_pickle=True)
env_matrix = np.load(env_matrix_path)
env_col_names = [line.strip() for line in open(env_cols_path)]
print(f"  CCA env embedding: {env_embedding.shape}")
print(f"  CCA pfam embedding: {pfam_embedding.shape}")

# Find SST column for CC1 scatter coloring
sst_idx = None
for i, name in enumerate(env_col_names):
    if name == 'sst_mean_c':
        sst_idx = i
        break
if sst_idx is None:
    for i, name in enumerate(env_col_names):
        if 'sst' in name.lower() and 'mean' in name.lower():
            sst_idx = i
            break
if sst_idx is not None:
    sst_values = env_matrix[:, sst_idx]
    print(f"  SST column: {env_col_names[sst_idx]}, range [{sst_values.min():.2f}, {sst_values.max():.2f}]")
else:
    sst_values = env_embedding[:, 0]  # fallback
    print("  WARNING: SST column not found, using CC1 for coloring")

# --- PFAM loadings (through PCA) ---
pfam_loadings_path = SOURCE_DATA_DIR / "cca_pfam_loadings_20260212_100221.tsv"
validate_input_source(pfam_loadings_path)
pfam_loadings = pd.read_csv(pfam_loadings_path, sep='\t', comment='#')
print(f"  PFAM loadings: {pfam_loadings.shape[0]} rows")

# --- Shrinkage data ---
shrinkage_path = SOURCE_DATA_DIR / "cca_shrinkage_cv_20260212_101811.tsv"
validate_input_source(shrinkage_path)
shrinkage = pd.read_csv(shrinkage_path, sep='\t', comment='#')

shrinkage_dist_path = SOURCE_DATA_DIR / "cca_shrinkage_cv_distribution_20260212_101811.tsv"
validate_input_source(shrinkage_dist_path)
shrinkage_dist = pd.read_csv(shrinkage_dist_path, sep='\t', comment='#')
print(f"  Shrinkage: {shrinkage.shape[0]} CCs, {shrinkage_dist.shape[0]} CV iterations")

# --- GCCA comparison ---
gcca_comp_path = SOURCE_DATA_DIR / "gcca_ternary_comparison_20260212_102910.tsv"
validate_input_source(gcca_comp_path)
gcca_comp = pd.read_csv(gcca_comp_path, sep='\t', comment='#')
print(f"  GCCA comparison: {gcca_comp.shape}")

# --- GPS coordinates for world map ---
merged_path = BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
validate_input_source(merged_path)
merged = pd.read_csv(merged_path, sep='\t', comment='#',
                     usecols=['assembly_id', 'latitude', 'longitude'])
gps_lookup = merged.set_index('assembly_id')[['latitude', 'longitude']]
lats = np.array([gps_lookup.loc[sid, 'latitude'] if sid in gps_lookup.index else np.nan for sid in sample_ids])
lons = np.array([gps_lookup.loc[sid, 'longitude'] if sid in gps_lookup.index else np.nan for sid in sample_ids])
valid_gps = ~np.isnan(lats)
print(f"  GPS matched: {valid_gps.sum()} of {len(sample_ids)} samples")

# ============================================================================
# PREPARE PANEL DATA
# ============================================================================
print("\n" + "=" * 70)
print("PREPARING PANEL DATA")
print("=" * 70)

# (Former panels A/B/C — CC bars, permutation test, shared variance — moved to
# Figure S1; their data prep is no longer needed in the trimmed S2.)

# Panel D: Env loadings heatmap (top 15 by max absolute loading)
max_abs_loading = env_loadings.abs().max(axis=1)
top15_vars = max_abs_loading.nlargest(15).index
env_load_top15 = env_loadings.loc[top15_vars]
print(f"  Panel D: Top 15 env variables by max |loading|")

# Panel E: CC1 scatter
cc1_env = env_embedding[:, 0]
cc1_pfam = pfam_embedding[:, 0]
print(f"  Panel E: CC1 scatter, r={np.corrcoef(cc1_env, cc1_pfam)[0,1]:.4f}")

# Panel F: CC1 top-20 PFAM domains
cc1_pfam_data = pfam_loadings[pfam_loadings['canonical_component'] == 'CC1'].head(20).copy()
cc1_pfam_data = cc1_pfam_data.sort_values('cca_pca_loading', ascending=True)
cc1_pfam_data['pfam_short'] = cc1_pfam_data['pfam_domain'].str.replace(r'\.\d+$', '', regex=True)
print(f"  Panel F: CC1 top {len(cc1_pfam_data)} PFAM domains")

# Panel G: CC2 top-10 + CC3 top-10
cc2_pfam = pfam_loadings[pfam_loadings['canonical_component'] == 'CC2'].head(10).copy()
cc2_pfam = cc2_pfam.sort_values('cca_pca_loading', ascending=True)
cc2_pfam['pfam_short'] = cc2_pfam['pfam_domain'].str.replace(r'\.\d+$', '', regex=True)

cc3_pfam = pfam_loadings[pfam_loadings['canonical_component'] == 'CC3'].head(10).copy()
cc3_pfam = cc3_pfam.sort_values('cca_pca_loading', ascending=True)
cc3_pfam['pfam_short'] = cc3_pfam['pfam_domain'].str.replace(r'\.\d+$', '', regex=True)
print(f"  Panel G: CC2 top {len(cc2_pfam)}, CC3 top {len(cc3_pfam)} PFAM domains")

# Panel H: World map (using CC1 env scores)
cc1_map = env_embedding[:, 0]
sort_idx = np.argsort(np.abs(cc1_map))
lats_sorted = lats[sort_idx]
lons_sorted = lons[sort_idx]
cc1_sorted = cc1_map[sort_idx]
vmax_map = max(abs(cc1_map[valid_gps].min()), abs(cc1_map[valid_gps].max()))
vmin_map = -vmax_map
print(f"  Panel H: World map, {valid_gps.sum()} samples, range [{cc1_map[valid_gps].min():.4f}, {cc1_map[valid_gps].max():.4f}]")

# Panel I: Shrinkage data
shrinkage_pcts = shrinkage['shrinkage_pct'].values
observed_ccs_shrinkage = shrinkage['observed_correlation'].values
cv_dist_cols = [f'CC{i+1}' for i in range(10)]
cv_dist_data = [shrinkage_dist[col].values for col in cv_dist_cols]
print(f"  Panel I: Shrinkage — CC1={shrinkage_pcts[0]:.1f}%")

# Panel J: GCCA comparison
gcca_ccs = gcca_comp[gcca_comp['component'] != 'mean'].copy()
print(f"  Panel J: {len(gcca_ccs)} CCs × {len(gcca_comp.columns)-1} analyses")

# ============================================================================
# CREATE COMPOSITE FIGURE
# ============================================================================
print("\n" + "=" * 70)
print("CREATING COMPREHENSIVE CCA FIGURE S3")
print("=" * 70)

fig = plt.figure(figsize=(7.1, 4.0))

# Master grid: 2 rows. Row 1 packs all five small panels 5-up (heatmap, scatter,
# CC1/CC2/CC3 PFAM bars); Row 2 is a 3-up row holding the world map, the CV-
# shrinkage chart, and the CCA/GCCA chart. hspace opened to clear inter-row
# title/x-label collisions.
outer_gs = gridspec.GridSpec(2, 1, figure=fig,
                             height_ratios=[2.3, 2.4],
                             hspace=0.48,
                             top=0.94, bottom=0.07, left=0.06, right=0.975)

# ── Row 1 (5-up): (A) heatmap | (B) CC1 scatter | (C) CC1 | (D) CC2 | (E) CC3 ──
row1_gs = outer_gs[0].subgridspec(1, 5, width_ratios=[1.25, 1.0, 1.0, 0.9, 0.9],
                                  wspace=0.62)

# Panel A: Env loadings heatmap
ax_d = fig.add_subplot(row1_gs[0, 0])
# Nice labels for env variables
nice_env = {
    'depth_m': 'Depth', 'air_temp_mean_c': 'Air T (mean)', 'air_temp_max_c': 'Air T (max)',
    'air_temp_min_c': 'Air T (min)', 'air_temp_range_c': 'Air T (range)',
    'precip_mean_mm': 'Precipitation', 'solar_rad_mj_m2': 'Solar rad.',
    'elevation_m': 'Elevation', 'bathymetry_m': 'Bathymetry',
    'distance_to_coast_km': 'Dist. coast', 'landcover_class': 'Landcover',
    'sst_mean_c': 'SST (mean)', 'sst_max_c': 'SST (max)', 'sst_min_c': 'SST (min)',
    'sst_range_c': 'SST (range)', 'chl_mean_mg_m3': 'Chl-a (mean)',
    'chl_max_mg_m3': 'Chl-a (max)', 'chl_min_mg_m3': 'Chl-a (min)',
    'nflh_mean': 'NFLH', 'poc_mean_mg_m3': 'POC',
    'modis_sst_mean_c': 'MODIS SST', 'rrs_412': 'Rrs 412',
    'rrs_443': 'Rrs 443', 'rrs_469': 'Rrs 469', 'rrs_488': 'Rrs 488',
    'rrs_531': 'Rrs 531', 'rrs_547': 'Rrs 547', 'rrs_555': 'Rrs 555',
    'rrs_645': 'Rrs 645', 'rrs_667': 'Rrs 667', 'rrs_678': 'Rrs 678',
}
ylabels_d = [nice_env.get(v, v) for v in env_load_top15.index]

data_d = env_load_top15.values
vmax_d = np.abs(data_d).max()
im_d = ax_d.imshow(data_d, cmap=DIVERGING_CMAP, aspect='auto',
                   vmin=-vmax_d, vmax=vmax_d, interpolation='nearest')
ax_d.set_yticks(range(len(ylabels_d)))
ax_d.set_yticklabels(ylabels_d, fontsize=5)
ax_d.set_xticks(range(5))
ax_d.set_xticklabels(['CC1', 'CC2', 'CC3', 'CC4', 'CC5'], fontsize=5)
ax_d.set_title('Env. loadings on CC1-CC5 (top 15)', fontsize=6, fontweight='bold')

# Compact colorbar
cbar_d = plt.colorbar(im_d, ax=ax_d, fraction=0.03, pad=0.02, aspect=25)
cbar_d.set_label('Loading weight', fontsize=5)
cbar_d.ax.tick_params(labelsize=4.5, width=0.25, length=2)
cbar_d.outline.set_linewidth(0.25)
ax_d.text(-0.15, 1.05, 'A', transform=ax_d.transAxes, fontsize=6, fontweight='bold', va='top')

# Panel B: CC1 scatter (env vs PFAM canonical variates)
ax_e = fig.add_subplot(row1_gs[0, 1])
sc_e = ax_e.scatter(cc1_env, cc1_pfam, c=sst_values, cmap=THERMAL_CMAP,
                    s=2, alpha=0.6, linewidth=0, rasterized=True)
ax_e.set_xlabel('Env canonical variate (CC1)')
ax_e.set_ylabel('PFAM canonical variate (CC1)')
r_cc1 = np.corrcoef(cc1_env, cc1_pfam)[0, 1]
ax_e.set_title(f'CC1: env vs PFAM (r = {r_cc1:.3f})', fontsize=6, fontweight='bold')
# Identity line
lims = [min(cc1_env.min(), cc1_pfam.min()), max(cc1_env.max(), cc1_pfam.max())]
ax_e.plot(lims, lims, '--', color=(0.5, 0.5, 0.5), linewidth=0.5, alpha=0.5)
ax_e.set_xlim(lims)
ax_e.set_ylim(lims)
cbar_e = plt.colorbar(sc_e, ax=ax_e, fraction=0.03, pad=0.02, aspect=25)
cbar_e.set_label('SST (z-scored)', fontsize=5)
cbar_e.ax.tick_params(labelsize=4.5, width=0.25, length=2)
cbar_e.outline.set_linewidth(0.25)
ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)
ax_e.text(-0.15, 1.05, 'B', transform=ax_e.transAxes, fontsize=6, fontweight='bold', va='top')

# Panel C: CC1 top-20 PFAM domain loadings (Row 1, slot 3)
ax_f = fig.add_subplot(row1_gs[0, 2])
bar_colors_f = [DEEP_OCEAN if v < 0 else SIENNA for v in cc1_pfam_data['cca_pca_loading']]
ax_f.barh(range(len(cc1_pfam_data)), cc1_pfam_data['cca_pca_loading'].values,
          color=bar_colors_f, edgecolor='none', height=0.7)
ax_f.set_yticks(range(len(cc1_pfam_data)))
ax_f.set_yticklabels(cc1_pfam_data['pfam_short'].values, fontsize=4.5)
ax_f.set_xlabel('CCA loading (through PCA)')
ax_f.set_title('CC1 top-20 PFAM domain loadings', fontsize=6, fontweight='bold')
ax_f.axvline(0, color='black', linewidth=0.25, zorder=0)
ax_f.invert_yaxis()
# (Per-bar Spearman r annotations dropped: too cramped at 1/5-width; the loadings
#  and PFAM labels carry the panel. r-values remain in source_data.)
ax_f.spines['top'].set_visible(False)
ax_f.spines['right'].set_visible(False)
ax_f.text(-0.18, 1.05, 'C', transform=ax_f.transAxes, fontsize=6, fontweight='bold', va='top')

# Panel D: CC2 top-10 PFAM loadings (Row 1, slot 4)
ax_g1 = fig.add_subplot(row1_gs[0, 3])
bar_colors_g1 = [DEEP_OCEAN if v < 0 else SIENNA for v in cc2_pfam['cca_pca_loading']]
ax_g1.barh(range(len(cc2_pfam)), cc2_pfam['cca_pca_loading'].values,
           color=bar_colors_g1, edgecolor='none', height=0.7)
ax_g1.set_yticks(range(len(cc2_pfam)))
ax_g1.set_yticklabels(cc2_pfam['pfam_short'].values, fontsize=4.5)
ax_g1.set_xlabel('CCA loading (through PCA)')
ax_g1.set_title('CC2 top-10 PFAM loadings', fontsize=6, fontweight='bold')
ax_g1.axvline(0, color='black', linewidth=0.25, zorder=0)
ax_g1.invert_yaxis()
ax_g1.spines['top'].set_visible(False)
ax_g1.spines['right'].set_visible(False)
ax_g1.text(-0.22, 1.05, 'D', transform=ax_g1.transAxes, fontsize=6, fontweight='bold', va='top')

# Panel E: CC3 top-10 PFAM loadings (Row 1, slot 5)
ax_g2 = fig.add_subplot(row1_gs[0, 4])
bar_colors_g2 = [DEEP_OCEAN if v < 0 else SIENNA for v in cc3_pfam['cca_pca_loading']]
ax_g2.barh(range(len(cc3_pfam)), cc3_pfam['cca_pca_loading'].values,
           color=bar_colors_g2, edgecolor='none', height=0.7)
ax_g2.set_yticks(range(len(cc3_pfam)))
ax_g2.set_yticklabels(cc3_pfam['pfam_short'].values, fontsize=4.5)
ax_g2.set_xlabel('CCA loading (through PCA)')
ax_g2.set_title('CC3 top-10 PFAM loadings', fontsize=6, fontweight='bold')
ax_g2.axvline(0, color='black', linewidth=0.25, zorder=0)
ax_g2.invert_yaxis()
ax_g2.spines['top'].set_visible(False)
ax_g2.spines['right'].set_visible(False)
ax_g2.text(-0.22, 1.05, 'E', transform=ax_g2.transAxes, fontsize=6, fontweight='bold', va='top')

# ── Row 2 (3-up): (F) world map | (G) CV shrinkage | (H) GCCA comparison ──
row2_gs = outer_gs[1].subgridspec(1, 3, width_ratios=[1.0, 1.05, 1.05], wspace=0.42)

# Panel F: World map CC1 scores
ax_h = fig.add_subplot(row2_gs[0, 0], projection=ccrs.Robinson())
cmap_div = get_diverging_cmap('ocean_land')

ax_h.add_feature(cfeature.LAND, facecolor=(0.92, 0.92, 0.92), edgecolor='none', zorder=0)
ax_h.add_feature(cfeature.OCEAN, facecolor=(0.97, 0.97, 0.98), edgecolor='none', zorder=0)
ax_h.add_feature(cfeature.COASTLINE, linewidth=0.15, edgecolor=(0.5, 0.5, 0.5), zorder=1)

# Only plot points with valid GPS
valid_mask = valid_gps[sort_idx]
sc_h = ax_h.scatter(lons_sorted[valid_mask], lats_sorted[valid_mask],
                    c=cc1_sorted[valid_mask], cmap=cmap_div,
                    s=3, alpha=0.85, vmin=vmin_map, vmax=vmax_map,
                    linewidth=0.1, edgecolor=(0.3, 0.3, 0.3),
                    transform=ccrs.PlateCarree(), zorder=2)

ax_h.gridlines(draw_labels=False, linewidth=0.15, color=(0.7, 0.7, 0.7), alpha=0.5, linestyle='--')
ax_h.set_title(f'CC1 env-side scores (n = {valid_gps.sum()})',
               fontsize=6, fontweight='bold', pad=4)

# Colorbar for map — thin HORIZONTAL bar under the map, kept inside the map's own
# 1/3-width slot so it cannot overlap the (G) panel to its right. Anchored to the
# map axes' drawn position (robust to layout changes).
fig.canvas.draw()
hpos = ax_h.get_position()
cax_h = fig.add_axes([hpos.x0 + 0.18 * hpos.width, hpos.y0 - 0.012,
                      0.45 * hpos.width, 0.012])
cbar_h = plt.colorbar(sc_h, cax=cax_h, orientation='horizontal')
cbar_h.set_label('CC1 score', fontsize=5, labelpad=1)
cbar_h.ax.tick_params(labelsize=4.5, width=0.25, length=2)
cbar_h.outline.set_linewidth(0.25)

# Panel label at the map slot's left edge, lifted above its (centered) title.
fig.text(hpos.x0 - 0.005, hpos.y1 + 0.005, 'F', fontsize=6, fontweight='bold',
         va='bottom', ha='left')

# Panel G: Cross-validated shrinkage (Row 2, slot 2)
ax_i = fig.add_subplot(row2_gs[0, 1])
x_i = np.arange(10)
bar_width = 0.35

bars_obs = ax_i.bar(x_i, observed_ccs_shrinkage, width=bar_width, label='Observed',
                    color=COASTAL_BLUE, edgecolor='none', alpha=0.9)

vp = ax_i.violinplot(cv_dist_data, positions=x_i + bar_width,
                     showmeans=False, showmedians=True, showextrema=False,
                     widths=bar_width * 1.5)
for body in vp['bodies']:
    body.set_facecolor(DEEP_OCEAN)
    body.set_edgecolor('none')
    body.set_alpha(0.6)
vp['cmedians'].set_color('white')
vp['cmedians'].set_linewidth(0.5)

for i, pct in enumerate(shrinkage_pcts):
    ax_i.text(x_i[i] + bar_width / 2, observed_ccs_shrinkage[i] + 0.02,
              f'{pct:.0f}%', fontsize=4, ha='center', va='bottom',
              color=SIENNA, fontweight='bold')

ax_i.set_xticks(x_i + bar_width / 2)
ax_i.set_xticklabels([f'CC{i+1}' for i in range(10)], fontsize=5, rotation=45, ha='right')
ax_i.set_ylabel('Canonical correlation')
ax_i.set_title('Observed vs. cross-validated CCs\n(100 split-half iterations, basin-stratified)',
               fontsize=6, fontweight='bold')
ax_i.set_ylim(0, 1.0)
ax_i.legend(loc='upper right', frameon=False, fontsize=4.5, labels=['Observed', 'CV distribution'])
ax_i.spines['top'].set_visible(False)
ax_i.spines['right'].set_visible(False)
ax_i.text(-0.12, 1.08, 'G', transform=ax_i.transAxes, fontsize=6, fontweight='bold', va='top')

# Panel H: GCCA pairwise correlation comparison (Row 2, slot 3)
ax_j = fig.add_subplot(row2_gs[0, 2])
analysis_cols = ['gcca_3view_n995', 'env_pfam_2view_n995',
                 'alpha_pfam_2view_n995', 'alpha_env_2view_n995',
                 'env_pfam_2view_n1809']
analysis_labels = ['GCCA 3-view (n=995)', 'Env x PFAM (n=995)',
                   'AE x PFAM (n=995)', 'AE x Env (n=995)',
                   'Env x PFAM (n=1809)']
analysis_colors = [FOREST_GREEN, COASTAL_BLUE, TURQUOISE, DESERT_TAN, DEEP_OCEAN]

x_j = np.arange(10)
width_j = 0.15
for j, (col, label, color) in enumerate(zip(analysis_cols, analysis_labels, analysis_colors)):
    vals = gcca_ccs[col].values
    offset = (j - 2) * width_j
    ax_j.bar(x_j + offset, vals, width=width_j, label=label,
             color=color, edgecolor='none', alpha=0.85)

ax_j.set_xticks(x_j)
ax_j.set_xticklabels([f'CC{i+1}' for i in range(10)], fontsize=5, rotation=45, ha='right')
ax_j.set_ylabel('Canonical correlation')
ax_j.set_title('CCA vs. GCCA: pairwise comparison', fontsize=6, fontweight='bold')
ax_j.set_ylim(0, 1.05)
ax_j.legend(loc='upper right', frameon=False, fontsize=4, ncol=1,
            handlelength=1.0, handletextpad=0.3, labelspacing=0.3)
ax_j.spines['top'].set_visible(False)
ax_j.spines['right'].set_visible(False)
ax_j.text(-0.12, 1.08, 'H', transform=ax_j.transAxes, fontsize=6, fontweight='bold', va='top')

# ============================================================================
# SAVE OUTPUTS
# ============================================================================
print("\n" + "=" * 70)
print("SAVING COMPREHENSIVE CCA FIGURE S3")
print("=" * 70)

OUT_DIR = MANUSCRIPT_DIR / "figures"
for fmt in ['pdf', 'svg']:
    filename = OUT_DIR / f"FigureS2_cca_trimmed_{TIMESTAMP}.{fmt}"
    fig.savefig(filename, format=fmt,
                bbox_inches='tight',
                transparent=True,
                edgecolor='none',
                dpi=300)
    print(f"  Saved: {filename}")
fig.savefig(OUT_DIR / f"FigureS2_cca_trimmed_{TIMESTAMP}_check.png",
            dpi=300, bbox_inches='tight', transparent=False)

plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY — Comprehensive CCA Figure S3")
print("=" * 70)
print(f"  10 panels (A-J) combining old S3 + old S9")
print(f"  Panel A: 10 CCs with permutation null bounds")
print(f"  Panel C: Shared variance decomposition")
print(f"  Panel D: Env loadings heatmap (top 15 vars x CC1-CC5)")
print(f"  Panel E: CC1 scatter env vs PFAM (r={r_cc1:.3f})")
print(f"  Panel F: CC1 top-20 PFAM domains (all negative loadings)")
print(f"  Panel G: CC2/CC3 top-10 PFAM domains")
print(f"  Panel H: World map with {valid_gps.sum()} samples")
print(f"  Panel I: Shrinkage — CC1={shrinkage_pcts[0]:.1f}%, mean={shrinkage_pcts.mean():.1f}%")
print(f"  Panel J: GCCA comparison (5 analyses x 10 CCs)")
print(f"\n{'=' * 70}")
print(f"COMPLETE — {datetime.now().isoformat()}")
print(f"{'=' * 70}")
