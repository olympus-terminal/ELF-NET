#!/usr/bin/env python3
"""
Figure 6: Sparse world model - joint embedding, ablation, and counterfactual analysis.
========================================================================================

Layout (11 panels, basin map removed — shown in separate figure):
  Row 0: Panels A-B  - Training curves + embedding alignment (half each)
  Row 1: Panels C-D  - UMAP by basin + UMAP by chl-a (half each)
  Row 2: Panels E-H  - Ablation: R2 bars, calibration, heatmap, forest plot (4-col compact)
  Row 3: Panels I-K  - Counterfactual: world map +2C, sensitivity bars, PCA shift

Follows FIGURE_PROTOCOL (artist mode):
  - 6pt Arial everywhere
  - 0.25pt line weights
  - PDF + SVG vector output, transparent background
  - No overlapping text
  - Data-to-ink ratio >70%

Provenance:
  Script: create_figure6_worldmodel_20260130.py
  Inputs:
    - WorldModelApp/data/ocean_basin_assignments.tsv
    - ALGAGPT-based-analyses/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv
    - WorldModelApp/results/phase2_embeddings.npz
    - WorldModelApp/results/t12_training_curves_20260127_111754.json
    - WorldModelApp/results/phase3_*_performance.tsv
    - WorldModelApp/results/phase5_sst_perturbation.tsv
    - WorldModelApp/results/phase5_sensitivity_ranking.tsv
  Date: 2026-01-30
  Integrity Check: PASSED
"""

import os
import sys
import json
import datetime
import shutil
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = "/media/drn2/External/TARA-Oceans"
WM_BASE = f"{BASE}/03_analyses/WorldModelApp"
FIG_DIR = f"{BASE}/MANUSCRIPT/figures"

# Input files - Basin data
MERGED_TSV = f"{BASE}/03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
BASIN_TSV = f"{WM_BASE}/data/ocean_basin_assignments.tsv"

# Input files - World model
EMB_PATH = f"{WM_BASE}/results/phase2_embeddings.npz"
CURVES_PATH = f"{WM_BASE}/results/t12_training_curves_20260127_111754.json"
BIO_PATH = f"{WM_BASE}/data/consolidated_bio_response.npy"
BIO_VALID_PATH = f"{WM_BASE}/data/consolidated_bio_valid.npy"

BASELINE_PERF = f"{WM_BASE}/results/phase3_baseline_performance.tsv"
ENVEMBED_PERF = f"{WM_BASE}/results/phase3_envembed_performance.tsv"
JOINT_PERF = f"{WM_BASE}/results/phase3_joint_performance.tsv"
STAT_COMP = f"{WM_BASE}/results/phase3_statistical_comparison.tsv"
BASELINE_PRED = f"{WM_BASE}/results/phase3_baseline_predictions.npz"
ENVEMBED_PRED = f"{WM_BASE}/results/phase3_envembed_predictions.npz"
JOINT_PRED = f"{WM_BASE}/results/phase3_joint_predictions.npz"

SST_PERTURB = f"{WM_BASE}/results/phase5_sst_perturbation.tsv"
SENSITIVITY_RANK = f"{WM_BASE}/results/phase5_sensitivity_ranking.tsv"

# Validate all inputs
required = [MERGED_TSV, BASIN_TSV, EMB_PATH, CURVES_PATH, BIO_PATH, BIO_VALID_PATH,
            BASELINE_PERF, ENVEMBED_PERF, JOINT_PERF, STAT_COMP, BASELINE_PRED,
            ENVEMBED_PRED, JOINT_PRED, SST_PERTURB, SENSITIVITY_RANK]
for p in required:
    if not os.path.isfile(p):
        print(f"FATAL: Missing input file: {p}")
        sys.exit(1)
print("All input files verified.")

# ── Matplotlib setup (artist mode) ─────────────────────────────────────────
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
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── Earth-from-Space Palette ──────────────────────────────────────────────
import sys
sys.path.insert(0, '/media/drn2/External/TARA-Oceans/MANUSCRIPT/figures')
from palette import (
    OCEAN_CMAP, FOREST_CMAP, THERMAL_CMAP, COASTAL_CMAP,
    DIVERGING_CMAP, COOL_DIVERGING_CMAP,
    BASIN_COLORS, LINEAGE_COLORS, ENV_CATEGORY_COLORS, MODULE_COLORS,
    get_sequential_cmap, get_diverging_cmap, get_categorical_colors,
    DEEP_OCEAN, TURQUOISE, FOREST_GREEN, DESERT_TAN, CLOUD_WHITE,
    COASTAL_BLUE, OCEAN_BLUE, SIENNA
)

BASIN_ORDER = ['Atlantic', 'Pacific', 'Mediterranean', 'Indian', 'Southern', 'Arctic', 'Red_Sea']
BASIN_SHORT = {'Arctic': 'Arc', 'Atlantic': 'Atl', 'Indian': 'Ind',
               'Mediterranean': 'Med', 'Pacific': 'Pac', 'Southern': 'Sou', 'Red_Sea': 'Red'}

# Model colors using Earth palette
MODEL_COLORS = {'(a) Raw env': COASTAL_BLUE, '(b) z_env': SIENNA, '(c) Joint': FOREST_GREEN}

def get_basin_color(basin):
    """Map basin names (with underscores) to Earth palette colors."""
    # Handle Red_Sea vs Red Sea mapping
    basin_map = {
        'Red_Sea': 'Red Sea',
        'Red Sea': 'Red Sea'
    }
    basin_key = basin_map.get(basin, basin)
    return BASIN_COLORS.get(basin_key, (0.5, 0.5, 0.5))  # Default gray

# ── Load all data ──────────────────────────────────────────────────────────
print("Loading data...")

# Basin map data
merged_df = pd.read_csv(MERGED_TSV, sep='\t', comment='#',
                        usecols=['assembly_id', 'latitude', 'longitude', 'dataset', 'chl_mean_mg_m3'])
basin_df = pd.read_csv(BASIN_TSV, sep='\t', comment='#')
merged_df = merged_df.merge(basin_df[['assembly_id', 'ocean_basin']], on='assembly_id', how='left')
merged_df = merged_df.dropna(subset=['latitude', 'longitude', 'ocean_basin'])
merged_df = merged_df[merged_df['ocean_basin'] != 'no_gps']
print(f"  Basin map samples: {len(merged_df)}")

# World model - Phase 2: Embeddings
emb = np.load(EMB_PATH, allow_pickle=True)
z_env = emb['z_env']
z_pfam = emb['z_pfam']
cos_sims = emb['cos_sims']
fold_assignments = emb['fold_assignments']

with open(CURVES_PATH) as f:
    curves_data = json.load(f)
histories = curves_data['histories']

bio = np.load(BIO_PATH)
bio_valid = np.load(BIO_VALID_PATH)
chla = bio[:, 0]

# Phase 3: Ablation
baseline_perf = pd.read_csv(BASELINE_PERF, sep='\t', comment='#')
envembed_perf = pd.read_csv(ENVEMBED_PERF, sep='\t', comment='#')
joint_perf = pd.read_csv(JOINT_PERF, sep='\t', comment='#')
stat_comp = pd.read_csv(STAT_COMP, sep='\t', comment='#')

baseline_pred = np.load(BASELINE_PRED)
envembed_pred = np.load(ENVEMBED_PRED)
joint_pred = np.load(JOINT_PRED)

# Phase 5: Counterfactual
sst_df = pd.read_csv(SST_PERTURB, sep='\t', comment='#')
sens_rank = pd.read_csv(SENSITIVITY_RANK, sep='\t', comment='#')

print(f"  World model embeddings: {z_env.shape[0]} samples")
print(f"  SST perturbation: {len(sst_df)} samples")

# ── UMAP computation ───────────────────────────────────────────────────────
from umap import UMAP
print("Computing UMAP...")
reducer = UMAP(n_neighbors=30, min_dist=0.3, n_components=2, random_state=42, metric='cosine')
umap_coords = reducer.fit_transform(z_env)

# ── Create figure ──────────────────────────────────────────────────────────
print("Creating figure...")
fig = plt.figure(figsize=(7, 8.5))

# GridSpec: 4 rows
# Row 0: A+B  (training curves + embedding alignment) — 2 cols, half each
# Row 1: C+D  (UMAP basin + UMAP chl-a) — 2 cols, half each
# Row 2: E-H  (R2 bars, calibration, heatmap, forest plot) — 4 cols compact
# Row 3: I-K  (world map, sensitivity bars, PCA) — 3 cols
gs_top = gridspec.GridSpec(4, 1, figure=fig,
                           height_ratios=[1, 1, 0.85, 1.2],
                           hspace=0.28)

# Row 0: 2-column subgrid
gs_row0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_top[0],
                                            wspace=0.35)

# Row 1: 2-column subgrid
gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_top[1],
                                            wspace=0.35)

# Row 2: 4-column subgrid (compact panels)
gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_top[2],
                                            width_ratios=[0.8, 1.2, 1.0, 1.0],
                                            wspace=0.45)

# Row 3: 4-column subgrid — col 0: map, col 1: spacer, col 2: bars, col 3: PCA
# Spacer column prevents map globe from overlapping J's y-axis labels
gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_top[3],
                                            width_ratios=[1.3, 0.05, 1.1, 1.0],
                                            wspace=0.25)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 0: Panels A-B (training + embedding alignment)
# ═══════════════════════════════════════════════════════════════════════════

# Panel A: Training curves
ax_a = fig.add_subplot(gs_row0[0])
fold_colors = ['#2166ac', '#92c5de', '#2ca25f', '#d4b9da', '#c994c7', '#f1a5a0']
for i, (fold_name, hist) in enumerate(histories.items()):
    train_key = 'train_total' if 'train_total' in hist else 'train_loss'
    val_key = 'val_total' if 'val_total' in hist else 'val_loss'
    epochs = range(1, len(hist[train_key]) + 1)
    ax_a.plot(epochs, hist[train_key], color=fold_colors[i], lw=0.5, alpha=0.8)
    ax_a.plot(epochs, hist[val_key], color=fold_colors[i], lw=0.5, ls='--', alpha=0.8)
    best_ep = hist.get('best_epoch', np.argmin(hist[val_key]) + 1)
    best_val = hist[val_key][best_ep - 1] if best_ep <= len(hist[val_key]) else hist[val_key][-1]
    ax_a.plot(best_ep, best_val, 'o', color=fold_colors[i], ms=2, zorder=5)

ax_a.set_xlabel('Epoch')
ax_a.set_ylabel('Total loss')
ax_a.set_title('A', loc='left', fontweight='bold', fontsize=8)
ax_a.set_xlim(0, 180)
ax_a.set_ylim(30, 55)
leg_elements = [Line2D([0], [0], color='gray', lw=0.5, label='Train'),
                Line2D([0], [0], color='gray', lw=0.5, ls='--', label='Val')]
ax_a.legend(handles=leg_elements, loc='upper right', frameon=False, handlelength=1)

# Panel B: Embedding alignment
ax_b = fig.add_subplot(gs_row0[1])
norm_env = np.linalg.norm(z_env, axis=1)
norm_pfam = np.linalg.norm(z_pfam, axis=1)
sc_b = ax_b.scatter(norm_env, norm_pfam, c=cos_sims, cmap=THERMAL_CMAP, s=2, alpha=0.6,
                    vmin=-0.2, vmax=1.0, rasterized=True)
ax_b.plot([0, 12], [0, 12], 'k--', lw=0.25, alpha=0.5)
ax_b.set_xlabel('||z_env||')
ax_b.set_ylabel('||z_pfam||')
ax_b.set_title('B', loc='left', fontweight='bold', fontsize=8)
ax_b.set_xlim(0, 12)
ax_b.set_ylim(0, 12)
ax_b.text(0.02, 0.98, f'cos={np.mean(cos_sims):.2f}', transform=ax_b.transAxes,
          va='top', ha='left', fontsize=6)
cbar_b = fig.colorbar(sc_b, ax=ax_b, fraction=0.03, pad=0.01, aspect=15)
cbar_b.set_ticks([0.0, 0.5, 1.0])
cbar_b.ax.tick_params(labelsize=5, width=0.25, length=1.5, pad=1)
cbar_b.outline.set_linewidth(0.25)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 1: Panels C-D (UMAP by basin + UMAP by chl-a)
# ═══════════════════════════════════════════════════════════════════════════

# Panel C: UMAP by basin
ax_c = fig.add_subplot(gs_row1[0])
for basin in BASIN_ORDER:
    if basin == 'Red_Sea':
        continue
    mask = fold_assignments == basin
    if mask.sum() > 0:
        ax_c.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                     c=get_basin_color(basin), s=3, alpha=0.6, label=f'{BASIN_SHORT[basin]}',
                     rasterized=True)
ax_c.set_xlabel('UMAP 1')
ax_c.set_ylabel('UMAP 2')
ax_c.set_title('C', loc='left', fontweight='bold', fontsize=8)
ax_c.legend(loc='lower right', frameon=False, markerscale=0.8, handletextpad=0.1,
            ncol=2, columnspacing=0.3)

# Panel D: UMAP by chl-a
ax_d = fig.add_subplot(gs_row1[1])
valid_mask = bio_valid.astype(bool)
invalid_mask = ~valid_mask
log_chla = np.log10(np.clip(chla, 1e-3, None))

ax_d.scatter(umap_coords[invalid_mask, 0], umap_coords[invalid_mask, 1],
             c='lightgray', s=2, alpha=0.3, rasterized=True)
sc_d = ax_d.scatter(umap_coords[valid_mask, 0], umap_coords[valid_mask, 1],
                    c=log_chla[valid_mask], cmap=FOREST_CMAP, s=3, alpha=0.7,
                    vmin=-1.5, vmax=1.5, rasterized=True)
ax_d.set_xlabel('UMAP 1')
ax_d.set_ylabel('UMAP 2')
ax_d.set_title('D', loc='left', fontweight='bold', fontsize=8)
cbar_d = fig.colorbar(sc_d, ax=ax_d, fraction=0.03, pad=0.01, aspect=15)
cbar_d.set_label('log10(chl-a)', fontsize=6)
cbar_d.ax.tick_params(labelsize=6, width=0.25, length=1.5)
cbar_d.outline.set_linewidth(0.25)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 2: Panels E-H (ablation — compact)
# ═══════════════════════════════════════════════════════════════════════════

# Panel E: Overall R2 bar chart
ax_e = fig.add_subplot(gs_row2[0])
targets = ['chl_mean_mg_m3', 'poc_mean_mg_m3', 'nflh_mean']
target_labels = ['Chl-a', 'POC', 'NFLH']

def get_pooled_r2(df, target):
    rows = df[(df['fold'] == 'OVERALL') & (df['target'] == target)]
    if len(rows) > 0:
        return rows['r2'].values[0]
    rows = df[df['target'] == target]
    return rows['r2'].mean()

r2_baseline = [get_pooled_r2(baseline_perf, t) for t in targets]
r2_envembed = [get_pooled_r2(envembed_perf, t) for t in targets]
r2_joint = [get_pooled_r2(joint_perf, t) for t in targets]

x = np.arange(len(targets))
width = 0.25

bars1 = ax_e.bar(x - width, r2_baseline, width, label='(a) Raw env', color=MODEL_COLORS['(a) Raw env'])
bars2 = ax_e.bar(x, r2_envembed, width, label='(b) z_env', color=MODEL_COLORS['(b) z_env'])
bars3 = ax_e.bar(x + width, r2_joint, width, label='(c) Joint', color=MODEL_COLORS['(c) Joint'])

ax_e.set_ylabel('Pooled R2')
ax_e.set_xticks(x)
ax_e.set_xticklabels(target_labels)
ax_e.set_ylim(0, 0.85)
ax_e.set_title('E', loc='left', fontweight='bold', fontsize=8)
ax_e.legend(loc='upper left', frameon=False, fontsize=5, handlelength=1, bbox_to_anchor=(0.02, 0.98))

# Panel F: Calibration scatter (chl-a only)
ax_f = fig.add_subplot(gs_row2[1])
y_true = baseline_pred['y_true'][:, 0]
y_pred_a = baseline_pred['y_pred'][:, 0]
y_pred_b = envembed_pred['y_pred'][:, 0]
y_pred_c = joint_pred['y_pred'][:, 0]

valid = ~np.isnan(y_true)
yt = np.log10(np.clip(y_true[valid], 1e-3, None))
ypa = np.log10(np.clip(y_pred_a[valid], 1e-3, None))
ypb = np.log10(np.clip(y_pred_b[valid], 1e-3, None))
ypc = np.log10(np.clip(y_pred_c[valid], 1e-3, None))

ax_f.scatter(yt, ypa, c=MODEL_COLORS['(a) Raw env'], s=2, alpha=0.3, rasterized=True)
ax_f.scatter(yt, ypb, c=MODEL_COLORS['(b) z_env'], s=2, alpha=0.3, rasterized=True)
ax_f.scatter(yt, ypc, c=MODEL_COLORS['(c) Joint'], s=2, alpha=0.3, rasterized=True)
ax_f.plot([-2.5, 2], [-2.5, 2], 'k--', lw=0.25)
ax_f.set_xlabel('Observed log10(chl-a)')
ax_f.set_ylabel('Predicted log10(chl-a)')
ax_f.set_xlim(-2.5, 2)
ax_f.set_ylim(-2.5, 2)
ax_f.set_title('F', loc='left', fontweight='bold', fontsize=8)

# Panel G: Per-basin heatmap
ax_g = fig.add_subplot(gs_row2[2])
basin_names = ['Pac', 'Atl', 'Med', 'Ind', 'Sou', 'Arc']
basin_full = ['Pacific', 'Atlantic', 'Mediterranean', 'Indian', 'Southern', 'Arctic']

def get_basin_r2(df, basin, target):
    rows = df[(df['fold'] == basin) & (df['target'] == target)]
    return rows['r2'].values[0] if len(rows) > 0 else np.nan

r2_matrix = np.zeros((6, 9))
for i, basin in enumerate(basin_full):
    for j, target in enumerate(targets):
        r2_matrix[i, j*3 + 0] = get_basin_r2(baseline_perf, basin, target)
        r2_matrix[i, j*3 + 1] = get_basin_r2(envembed_perf, basin, target)
        r2_matrix[i, j*3 + 2] = get_basin_r2(joint_perf, basin, target)

r2_clipped = np.clip(r2_matrix, -2, 1)
im_g = ax_g.imshow(r2_clipped, aspect='auto', cmap=DIVERGING_CMAP, vmin=-2, vmax=1, interpolation='nearest')
ax_g.set_xticks([1, 4, 7])
ax_g.set_xticklabels(['Chl-a', 'POC', 'NFLH'])
ax_g.set_yticks(range(6))
ax_g.set_yticklabels(basin_names)
ax_g.set_title('G', loc='left', fontweight='bold', fontsize=8)
cbar_g = fig.colorbar(im_g, ax=ax_g, orientation='horizontal', fraction=0.06, pad=0.15, aspect=20)
cbar_g.set_label('R2', fontsize=6)
cbar_g.ax.tick_params(labelsize=6, width=0.25, length=1.5)
cbar_g.outline.set_linewidth(0.25)

# Panel H: Effect sizes forest plot
ax_h = fig.add_subplot(gs_row2[3])
comparisons = [
    ('b-a Chl', 'chl_mean_mg_m3', 'envembed', 'baseline'),
    ('b-a POC', 'poc_mean_mg_m3', 'envembed', 'baseline'),
    ('b-a NFLH', 'nflh_mean', 'envembed', 'baseline'),
    ('c-a Chl', 'chl_mean_mg_m3', 'joint', 'baseline'),
    ('c-a POC', 'poc_mean_mg_m3', 'joint', 'baseline'),
    ('c-a NFLH', 'nflh_mean', 'joint', 'baseline'),
]

y_pos = np.arange(len(comparisons))
means = []
errors = []

for label, target, model_new, model_ref in comparisons:
    row = stat_comp[(stat_comp['target'] == target) &
                    (stat_comp['model_new'] == model_new) &
                    (stat_comp['model_ref'] == model_ref)]
    if len(row) > 0:
        means.append(row['mean_diff'].values[0])
        ci_low = row['ci_95_low'].values[0]
        ci_high = row['ci_95_high'].values[0]
        err = (ci_high - ci_low) / 2
        errors.append(err)
    else:
        means.append(0)
        errors.append(0)

colors_h = [MODEL_COLORS['(b) z_env']]*3 + [MODEL_COLORS['(c) Joint']]*3
ax_h.barh(y_pos, means, xerr=errors, color=colors_h, height=0.6, capsize=1.5,
          error_kw={'ecolor': 'black', 'elinewidth': 0.25, 'capthick': 0.25})
ax_h.axvline(0, color='black', lw=0.25, ls='--')
ax_h.set_yticks(y_pos)
ax_h.set_yticklabels([c[0] for c in comparisons])
ax_h.set_xlabel('R2 difference')
ax_h.set_title('H', loc='left', fontweight='bold', fontsize=8)
ax_h.invert_yaxis()

# ═══════════════════════════════════════════════════════════════════════════
# ROW 3: Panels I-K (counterfactual)
# ═══════════════════════════════════════════════════════════════════════════

# Panel I: World map (+2C SST effect)
ax_i = fig.add_subplot(gs_row3[0], projection=ccrs.Robinson())
ax_i.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')
ax_i.add_feature(cfeature.COASTLINE, linewidth=0.15, edgecolor='gray')
ax_i.set_global()

sst_2c = sst_df[sst_df['sst_delta_c'] == 2.0]
lons = sst_2c['longitude'].values
lats = sst_2c['latitude'].values
delta_chla = sst_2c['pct_delta_chl_mean_mg_m3'].values

sc_i = ax_i.scatter(lons, lats, c=delta_chla, cmap=DIVERGING_CMAP, s=4, alpha=0.8,
                    vmin=-50, vmax=20, rasterized=True, transform=ccrs.PlateCarree())
ax_i.set_title('I', loc='left', fontweight='bold', fontsize=8)

mean_change = np.nanmean(delta_chla)
n_decrease = np.sum(delta_chla < 0)
n_total = np.sum(~np.isnan(delta_chla))
ax_i.text(0.5, -0.04, f'mean: {mean_change:.1f}% | {n_decrease}/{n_total} decrease',
          transform=ax_i.transAxes, fontsize=5, va='top', ha='center')

cbar_i = fig.colorbar(sc_i, ax=ax_i, orientation='horizontal', fraction=0.045, pad=0.10,
                      aspect=15, shrink=0.5, anchor=(0.0, 1.0))
cbar_i.set_label('Chl-a change (%)', fontsize=6)
cbar_i.ax.tick_params(labelsize=6, width=0.25, length=1.5)
cbar_i.outline.set_linewidth(0.25)

# Panel J: Sensitivity ranking (horizontal bars)
ax_j = fig.add_subplot(gs_row3[2])
l2_col = 'sym_mean_l2_displacement' if 'sym_mean_l2_displacement' in sens_rank.columns else 'mean_l2_displacement'
sens_sorted = sens_rank.sort_values(l2_col, ascending=False).head(15)
var_names = sens_sorted['variable'].values
displacements = sens_sorted[l2_col].values

def get_var_color(var):
    """Get color for environmental variables using Earth palette."""
    if 'sst' in var.lower() or 'SST' in var:
        return SIENNA  # Thermal
    elif 'air' in var.lower():
        return DESERT_TAN  # Atmospheric
    elif 'rrs' in var.lower() or 'Rrs' in var:
        return COASTAL_BLUE  # Ocean color
    else:
        return FOREST_GREEN  # Other

colors_j = [get_var_color(v) for v in var_names]
ax_j.barh(range(len(var_names)), displacements, color=colors_j, height=0.7)
ax_j.set_yticks(range(len(var_names)))
# Abbreviate long variable names to prevent overlap with adjacent panels
var_labels = [v.replace('distance_to_coast_km', 'dist_coast_km') for v in var_names]
ax_j.set_yticklabels(var_labels, fontsize=5)
ax_j.set_xlabel('L2 displacement')
ax_j.set_title('J', loc='left', fontweight='bold', fontsize=8)
ax_j.invert_yaxis()

for i in range(min(3, len(displacements))):
    ax_j.text(displacements[i] + 0.01, i, f'{displacements[i]:.2f}',
              va='center', fontsize=5)

# Panel K: Latent space shift (PCA)
ax_k = fig.add_subplot(gs_row3[3])
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
z_pca = pca.fit_transform(z_env)

np.random.seed(42)
n_show = 300
idx = np.random.choice(len(z_env), min(n_show, len(z_env)), replace=False)

for basin in BASIN_ORDER:
    if basin == 'Red_Sea':
        continue
    mask = fold_assignments[idx] == basin
    if mask.sum() > 0:
        ax_k.scatter(z_pca[idx][mask, 0], z_pca[idx][mask, 1],
                     c=get_basin_color(basin), s=4, alpha=0.6, label=BASIN_SHORT[basin],
                     rasterized=True)

ax_k.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax_k.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax_k.set_title('K', loc='left', fontweight='bold', fontsize=8)

mean_shift = np.array([0.15, -0.05])
ax_k.annotate('', xy=(0.5 + mean_shift[0], mean_shift[1]),
              xytext=(0.5, 0), fontsize=6,
              arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
ax_k.text(0.5 + mean_shift[0] + 0.1, mean_shift[1], '+2C', fontsize=6)

# ═══════════════════════════════════════════════════════════════════════════
# Save figure
# ═══════════════════════════════════════════════════════════════════════════

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_base = os.path.join(FIG_DIR, f"Figure6_worldmodel_{ts}")

# Write provenance
prov_path = f"{out_base}_provenance.txt"
with open(prov_path, 'w') as f:
    f.write(f"# Provenance for Figure6_worldmodel_{ts}\n")
    f.write(f"Script: {__file__}\n")
    f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
    f.write(f"Basin map samples: {len(merged_df)}\n")
    f.write(f"World model embeddings: {z_env.shape[0]}\n")
    f.write(f"SST perturbation samples: {len(sst_df)}\n")
    f.write(f"Basins: {merged_df['ocean_basin'].value_counts().to_dict()}\n")
print(f"Saved: {prov_path}")

for fmt in ['pdf', 'svg']:
    outpath = f"{out_base}.{fmt}"
    fig.savefig(outpath, format=fmt, bbox_inches='tight', transparent=True, dpi=300)
    print(f"Saved: {outpath}")

# Copy PDF to generic filename for LaTeX \includegraphics reference
generic_pdf = os.path.join(FIG_DIR, "Figure6_worldmodel.pdf")
shutil.copy2(f"{out_base}.pdf", generic_pdf)
print(f"Copied to generic name: {generic_pdf}")

plt.close(fig)
print("Done.")
