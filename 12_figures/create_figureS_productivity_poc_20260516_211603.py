#!/usr/bin/env python3
"""
Figure S8: Productivity Proof-of-Concept — XGBoost Spatial Block CV

Three panels:
  A) Observed vs predicted hexbin for chl_mean_mg_m3 (combined model)
  B) R² comparison: grouped bars (domain-only vs env-only vs combined, 3 targets)
  C) Top 15 SHAP features (horizontal bar)

Input:
  - algagpt_gee_pfam_merged_GPS_RECOVERED_20260124_114445.tsv (for Panel A CV)
  - source_data/ralph54/productivity_results.tsv (Panel B)
  - source_data/ralph54/shap_productivity_top15.tsv (Panel C)

Generated: 2026-05-16
"""

import os
import sys
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

SCRIPT_PATH = os.path.abspath(__file__)
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

sys.path.insert(0, os.path.dirname(__file__))
from palette import (OCEAN_CMAP, FOREST_CMAP, get_categorical_colors,
                     DEEP_OCEAN, TURQUOISE, FOREST_GREEN, OCEAN_BLUE,
                     COASTAL_BLUE, SAVANNA, DESERT_TAN)


def enforce_data_integrity():
    pass


enforce_data_integrity()


def apply_tara_style():
    style = {
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 6,
        'axes.labelsize': 6,
        'axes.titlesize': 6,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'axes.linewidth': 0.25,
        'xtick.major.width': 0.25,
        'ytick.major.width': 0.25,
        'xtick.major.size': 2,
        'ytick.major.size': 2,
        'axes.labelpad': 1,
        'xtick.major.pad': 1,
        'ytick.major.pad': 1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'axes.axisbelow': True,
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'legend.frameon': False,
        'figure.dpi': 100,
        'figure.facecolor': 'white',
        'lines.linewidth': 1.2,
        'lines.markersize': 3,
        'lines.markeredgewidth': 0.4,
        'savefig.dpi': 600,
        'savefig.format': 'pdf',
        'savefig.transparent': True,
        'savefig.facecolor': 'none',
        'savefig.edgecolor': 'none',
        'image.cmap': 'viridis',
    }
    mpl.rcParams.update(style)


apply_tara_style()

# ── Environment detection ──
if os.path.exists('/media/drn2/External/TARA-Oceans'):
    BASE_DIR = '/media/drn2/External/TARA-Oceans'
elif os.path.exists('/media/drn/External1/TARA-Oceans'):
    BASE_DIR = '/media/drn/External1/TARA-Oceans'
elif os.path.exists('/scratch/drn2/PROJECTS/TARA-LA4SR'):
    BASE_DIR = '/scratch/drn2/PROJECTS/TARA-LA4SR'
else:
    print("ERROR: Unknown environment")
    sys.exit(1)

MERGED_PATH = os.path.join(
    BASE_DIR,
    '03_analyses/ALGAGPT-based-analyses/'
    'algagpt_gee_pfam_merged_GPS_RECOVERED_20260124_114445.tsv'
)
MANUSCRIPT_DIR = os.path.dirname(os.path.dirname(SCRIPT_PATH))
RESULTS_PATH = os.path.join(MANUSCRIPT_DIR, 'source_data/ralph54/productivity_results.tsv')
SHAP_PATH = os.path.join(MANUSCRIPT_DIR, 'source_data/ralph54/shap_productivity_top15.tsv')
OUTPUT_DIR = os.path.dirname(SCRIPT_PATH)

for p in [MERGED_PATH, RESULTS_PATH, SHAP_PATH]:
    if not os.path.isfile(p):
        print(f"ERROR: File not found: {p}")
        sys.exit(1)

print(f"Dataset: {MERGED_PATH}")
print(f"Results: {RESULTS_PATH}")
print(f"SHAP:    {SHAP_PATH}")
sys.stdout.flush()

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL A: Re-run combined CV for chl_mean_mg_m3 to get obs-vs-pred
# ═══════════════════════════════════════════════════════════════════════════════

BLOCK_SIZE = 2.0
N_FOLDS = 10
N_PCA = 100
CLR_PSEUDOCOUNT = 0.5
PREVALENCE_THRESHOLD = 0.05

XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': 4,
    'verbosity': 0,
}

TARGET = 'chl_mean_mg_m3'
EXCLUSIONS = ['chl_mean_mg_m3', 'chl_max_mg_m3', 'chl_min_mg_m3', 'nflh_mean']

ALL_ENV_CANDIDATES = [
    'modis_sst_mean_c', 'sst_mean_c', 'sst_max_c', 'sst_min_c',
    'bathymetry_m', 'sst_range_c', 'air_temp_range_c',
    'rrs_412', 'solar_rad_mj_m2',
    'air_temp_min_c', 'rrs_443', 'distance_to_coast_km',
    'rrs_469', 'rrs_555', 'rrs_547', 'rrs_531', 'rrs_488', 'rrs_645',
    'air_temp_mean_c', 'rrs_667', 'rrs_678',
    'elevation_m', 'air_temp_max_c', 'precip_mean_mm',
    'depth_m', 'salinity_psu_est',
]

print("\n=== Panel A: CV for obs-vs-pred ===")
print("Loading dataset...")
sys.stdout.flush()

with open(MERGED_PATH, 'r') as f:
    for line in f:
        if not line.startswith('#'):
            header = line.strip().split('\t')
            break

PFAM_COLS = sorted([c for c in header if c.startswith('PF')])

needed_cols = list(set(
    ['assembly_id', 'latitude', 'longitude', TARGET,
     'chl_max_mg_m3', 'chl_min_mg_m3', 'nflh_mean'] +
    ALL_ENV_CANDIDATES + PFAM_COLS
))
needed_cols = [c for c in needed_cols if c in header]

df = pd.read_csv(MERGED_PATH, sep='\t', comment='#', usecols=needed_cols, low_memory=False)
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df = df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)
print(f"  Samples with GPS: {len(df)}")

# Spatial blocks
df['block_lat'] = np.floor(df['latitude'] / BLOCK_SIZE) * BLOCK_SIZE
df['block_lon'] = np.floor(df['longitude'] / BLOCK_SIZE) * BLOCK_SIZE
df['block_id'] = df['block_lat'].astype(str) + '_' + df['block_lon'].astype(str)

block_counts = df['block_id'].value_counts()
blocks_sorted = sorted(block_counts.index, key=lambda b: block_counts[b], reverse=True)
fold_assignment = {}
fold_sizes = [0] * N_FOLDS
for block in blocks_sorted:
    min_fold = int(np.argmin(fold_sizes))
    fold_assignment[block] = min_fold
    fold_sizes[min_fold] += block_counts[block]
df['fold'] = df['block_id'].map(fold_assignment)
folds_array = df['fold'].values

# PFAM features
X_pfam_raw = df[PFAM_COLS].fillna(0).values.astype(np.float64)
n_samples = X_pfam_raw.shape[0]
prevalence = (X_pfam_raw > 0).sum(axis=0) / n_samples
prev_mask = prevalence >= PREVALENCE_THRESHOLD
X_pfam_filtered = X_pfam_raw[:, prev_mask]
del X_pfam_raw
print(f"  PFAMs after filter: {X_pfam_filtered.shape[1]}")


def clr_transform(X, pseudocount=CLR_PSEUDOCOUNT):
    X_pseudo = X + pseudocount
    log_X = np.log(X_pseudo)
    geometric_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geometric_mean


# Env features
available_env = [c for c in ALL_ENV_CANDIDATES if c in df.columns]
for col in available_env:
    df[col] = pd.to_numeric(df[col], errors='coerce')
env_cols_for_target = [c for c in available_env if c not in EXCLUSIONS]
env_vals = df[env_cols_for_target].values.astype(np.float64)
col_medians = np.nanmedian(env_vals, axis=0)
for j in range(env_vals.shape[1]):
    mask_nan = np.isnan(env_vals[:, j])
    env_vals[mask_nan, j] = col_medians[j]

y_all = pd.to_numeric(df[TARGET], errors='coerce').values
valid_all = ~np.isnan(y_all)

from xgboost import XGBRegressor

all_observed = []
all_predicted = []

print("Running combined CV...")
sys.stdout.flush()

for fold_idx in range(N_FOLDS):
    test_mask = folds_array == fold_idx
    train_mask = ~test_mask
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    # CLR + PCA per fold
    X_train_clr = clr_transform(X_pfam_filtered[train_idx])
    X_test_clr = clr_transform(X_pfam_filtered[test_idx])

    n_comp = min(N_PCA, X_train_clr.shape[0], X_train_clr.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca.fit_transform(X_train_clr)
    X_test_pca = pca.transform(X_test_clr)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_pca)
    X_test_sc = scaler.transform(X_test_pca)

    # Filter to valid target
    train_valid = valid_all[train_idx]
    test_valid = valid_all[test_idx]
    if test_valid.sum() < 5 or train_valid.sum() < 20:
        continue

    y_train = y_all[train_idx[train_valid]]
    y_test = y_all[test_idx[test_valid]]

    X_train = np.hstack([X_train_sc[train_valid], env_vals[train_idx[train_valid]]])
    X_test = np.hstack([X_test_sc[test_valid], env_vals[test_idx[test_valid]]])

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    all_observed.extend(y_test.tolist())
    all_predicted.extend(y_pred.tolist())

    r2 = r2_score(y_test, y_pred)
    print(f"  Fold {fold_idx}: n_test={len(y_test)}, R²={r2:.4f}")
    sys.stdout.flush()

obs = np.array(all_observed)
pred = np.array(all_predicted)
overall_r2 = r2_score(obs, pred)
print(f"  Overall R² (pooled): {overall_r2:.4f}")
print(f"  Total samples: {len(obs)}")
sys.stdout.flush()

del df, X_pfam_filtered, env_vals

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL B: Read productivity results
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== Panel B: Loading R² results ===")
results_df = pd.read_csv(RESULTS_PATH, sep='\t', comment='#')
print(results_df[['target', 'config', 'median_r2']].to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL C: Read SHAP data
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== Panel C: Loading SHAP ===")
shap_df = pd.read_csv(SHAP_PATH, sep='\t', comment='#')
print(shap_df.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== Building figure ===")
sys.stdout.flush()

fig = plt.figure(figsize=(7.0, 3.2))
mpl.rcParams['figure.constrained_layout.use'] = False
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.1],
              left=0.06, right=0.97, bottom=0.15, top=0.90, wspace=0.45)

panel_axes = []

# ── Panel A: Obs vs Pred hexbin ──
ax_a = fig.add_subplot(gs[0, 0])
panel_axes.append(ax_a)

vmin_data = min(obs.min(), pred.min())
vmax_data = max(obs.max(), pred.max())
margin = (vmax_data - vmin_data) * 0.05
plot_min = vmin_data - margin
plot_max = vmax_data + margin

hb = ax_a.hexbin(obs, pred, gridsize=30, cmap=OCEAN_CMAP,
                 mincnt=1, linewidths=0.0, edgecolors='none')
ax_a.plot([plot_min, plot_max], [plot_min, plot_max],
          color='black', linewidth=0.5, linestyle='--', zorder=5)
ax_a.set_xlim(plot_min, plot_max)
ax_a.set_ylim(plot_min, plot_max)
ax_a.set_xlabel('Observed chl-a (mg/m3)')
ax_a.set_ylabel('Predicted chl-a (mg/m3)')
ax_a.set_aspect('equal', adjustable='box')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cbar_ax = inset_axes(ax_a, width="4%", height="50%", loc='right',
                     bbox_to_anchor=(0.08, 0, 1, 1),
                     bbox_transform=ax_a.transAxes, borderpad=0)
cbar_a = fig.colorbar(hb, cax=cbar_ax)
cbar_a.ax.tick_params(labelsize=6, width=0.25, length=2)
cbar_a.outline.set_linewidth(0.25)

ax_a.text(0.05, 0.95,
          f'R² = {overall_r2:.2f}\nn = {len(obs):,}',
          transform=ax_a.transAxes, fontsize=6, va='top', ha='left',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='black', linewidth=0.25, alpha=0.8))

# ── Panel B: R² grouped bars ──
ax_b = fig.add_subplot(gs[0, 1])
panel_axes.append(ax_b)

target_labels = {
    'chl_mean_mg_m3': 'Chl-a',
    'poc_mean_mg_m3': 'POC',
    'nflh_mean': 'NFLH',
}
config_labels = {
    'domain_only': 'Domain',
    'env_only': 'Environ.',
    'combined': 'Combined',
}
config_colors = {
    'domain_only': TURQUOISE,
    'env_only': DESERT_TAN,
    'combined': DEEP_OCEAN,
}
config_order = ['domain_only', 'env_only', 'combined']
target_order = ['chl_mean_mg_m3', 'poc_mean_mg_m3', 'nflh_mean']

n_targets = len(target_order)
n_configs = len(config_order)
bar_width = 0.22
x_positions = np.arange(n_targets)

for i, config in enumerate(config_order):
    heights = []
    err_lo = []
    err_hi = []
    for target in target_order:
        row = results_df[(results_df['target'] == target) &
                         (results_df['config'] == config)]
        if len(row) == 0:
            heights.append(0)
            err_lo.append(0)
            err_hi.append(0)
        else:
            med = row['median_r2'].values[0]
            q25 = row['iqr_25'].values[0]
            q75 = row['iqr_75'].values[0]
            heights.append(med)
            err_lo.append(max(0, med - q25))
            err_hi.append(q75 - med)

    offset = (i - (n_configs - 1) / 2) * bar_width
    bars = ax_b.bar(x_positions + offset, heights, bar_width,
                    color=config_colors[config],
                    edgecolor='black', linewidth=0.25,
                    label=config_labels[config], zorder=3)
    ax_b.errorbar(x_positions + offset, heights,
                  yerr=[err_lo, err_hi],
                  fmt='none', ecolor='black', elinewidth=0.5,
                  capsize=2, capthick=0.5, zorder=4)

ax_b.set_xticks(x_positions)
ax_b.set_xticklabels([target_labels[t] for t in target_order])
ax_b.set_ylabel('Median R²', labelpad=2)
ax_b.set_ylim(-0.20, 1.05)
ax_b.axhline(0, color='black', linewidth=0.25, zorder=1)
ax_b.legend(loc='upper right', frameon=False, fontsize=6)

# ── Panel C: SHAP horizontal bars ──
ax_c = fig.add_subplot(gs[0, 2])
panel_axes.append(ax_c)

shap_sorted = shap_df.sort_values('mean_abs_shap', ascending=True).reset_index(drop=True)
y_pos = np.arange(len(shap_sorted))

bar_colors = [DEEP_OCEAN if pfam else DESERT_TAN
              for pfam in shap_sorted['is_pfam']]

ax_c.barh(y_pos, shap_sorted['mean_abs_shap'], height=0.7,
          color=bar_colors, edgecolor='black', linewidth=0.25, zorder=3)

label_map = {
    'rrs_443': 'RRS 443 nm',
    'rrs_412': 'RRS 412 nm',
    'rrs_645': 'RRS 645 nm',
    'rrs_678': 'RRS 678 nm',
    'rrs_469': 'RRS 469 nm',
    'rrs_667': 'RRS 667 nm',
    'rrs_555': 'RRS 555 nm',
    'rrs_488': 'RRS 488 nm',
    'air_temp_mean_c': 'Air temp (mean)',
    'air_temp_min_c': 'Air temp (min)',
    'sst_range_c': 'SST range',
}
labels = []
for feat in shap_sorted['feature']:
    if feat.startswith('PF'):
        labels.append(feat.split('.')[0])
    elif feat in label_map:
        labels.append(label_map[feat])
    else:
        labels.append(feat.replace('_', ' '))

ax_c.set_yticks(y_pos)
ax_c.set_yticklabels(labels)
ax_c.set_xlabel('Mean |SHAP|')
ax_c.tick_params(axis='y', length=0)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=DEEP_OCEAN, edgecolor='black', linewidth=0.25, label='PFAM domain'),
    Patch(facecolor=DESERT_TAN, edgecolor='black', linewidth=0.25, label='Env. variable'),
]
ax_c.legend(handles=legend_elements, loc='lower right', frameon=False, fontsize=6)

# ── Panel labels ──
for ax, letter in zip(panel_axes, 'ABC'):
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes,
            fontsize=6, fontweight='bold', va='top', ha='left')

# ── Validate ──
def validate_figure(fig):
    renderer = fig.canvas.get_renderer()
    issues = []
    all_texts = list(fig.texts)
    for ax in fig.get_axes():
        all_texts.extend(ax.texts)
        all_texts.append(ax.title)
        all_texts.append(ax.xaxis.label)
        all_texts.append(ax.yaxis.label)
        all_texts.extend(ax.get_xticklabels())
        all_texts.extend(ax.get_yticklabels())
    all_texts = [t for t in all_texts if t.get_text().strip()]
    bboxes = []
    for t in all_texts:
        try:
            bb = t.get_window_extent(renderer=renderer)
            if bb.width > 0 and bb.height > 0:
                bboxes.append((t, bb))
        except Exception:
            pass
    for i, (t1, bb1) in enumerate(bboxes):
        for t2, bb2 in bboxes[i+1:]:
            if bb1.overlaps(bb2):
                overlap_x = min(bb1.x1, bb2.x1) - max(bb1.x0, bb2.x0)
                overlap_y = min(bb1.y1, bb2.y1) - max(bb1.y0, bb2.y0)
                rot1 = t1.get_rotation()
                rot2 = t2.get_rotation()
                both_rotated = (rot1 != 0) and (rot2 != 0)
                threshold = 15 if both_rotated else 5
                if overlap_x > threshold and overlap_y > threshold:
                    issues.append(f"OVERLAP: '{t1.get_text()[:30]}' x '{t2.get_text()[:30]}' ({overlap_x:.0f}x{overlap_y:.0f}px)")
    for t in all_texts:
        size = t.get_fontsize()
        if size > 6.5:
            issues.append(f"FONT SIZE {size}pt on '{t.get_text()[:20]}'")
    for t in all_texts:
        weight = t.get_fontproperties().get_weight()
        if weight in ('bold', 'heavy', 700, 800, 900):
            text = t.get_text().strip()
            if len(text) > 1 or not text.isalpha():
                issues.append(f"UNEXPECTED BOLD on '{text[:20]}'")
    if issues:
        print(f"VALIDATION FAILED - {len(issues)} issue(s):")
        for issue in issues:
            print(f"  x {issue}")
    else:
        print("VALIDATION PASSED - no overlap, font, or weight issues detected")
    return issues


issues = validate_figure(fig)

# ── Save ──
output_name = f"FigureS_productivity_poc_{TIMESTAMP}"
for fmt in ['pdf', 'svg']:
    path = os.path.join(OUTPUT_DIR, f"{output_name}.{fmt}")
    fig.savefig(path, format=fmt, transparent=True, edgecolor='none')
    print(f"  Saved: {path}")

# Also save a check PNG for validation
check_path = os.path.join(OUTPUT_DIR, f"{output_name}_check.png")
fig.savefig(check_path, dpi=300, transparent=False)
print(f"  Check: {check_path}")

plt.close(fig)
print("\nDONE.")
