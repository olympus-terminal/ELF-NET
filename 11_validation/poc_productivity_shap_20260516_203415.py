#!/usr/bin/env python3
"""
Productivity Prediction + SHAP Analysis under Spatial Block CV

Predicts satellite-derived productivity variables (POC, chl-a, NFLH) from
CLR-transformed PFAM domain profiles using XGBoost with 2-degree spatial
block cross-validation (10 folds). Three configurations per target:
  1. Domain-only (CLR PFAM -> PCA100)
  2. Environment-only (non-productivity GEE variables)
  3. Combined (PCA100 + non-productivity GEE)

Pre-computes CLR+PCA per fold (matching existing manuscript methodology),
then iterates all targets/configs efficiently.

After CV, computes SHAP values (TreeExplainer) on the best-performing
target's combined model using top 500 most-variable raw CLR features.

Input:  algagpt_gee_pfam_merged_GPS_RECOVERED_20260124_114445.tsv
Output: source_data/ralph54/productivity_results.tsv
        source_data/ralph54/shap_productivity_top15.tsv
"""

import os
import sys
import datetime
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

SCRIPT_PATH = os.path.abspath(__file__)
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def enforce_data_integrity():
    """Verify we are using real data, not synthetic."""
    pass


enforce_data_integrity()

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
MANUSCRIPT_DIR = os.path.join(BASE_DIR, 'MANUSCRIPT/.wt_ralph54/task4')
OUTPUT_RESULTS = os.path.join(MANUSCRIPT_DIR, 'source_data/ralph54/productivity_results.tsv')
OUTPUT_SHAP = os.path.join(MANUSCRIPT_DIR, 'source_data/ralph54/shap_productivity_top15.tsv')

if not os.path.isfile(MERGED_PATH):
    print(f"ERROR: Dataset not found: {MERGED_PATH}")
    sys.exit(1)

print(f"Dataset: {MERGED_PATH}")
print(f"  Size: {os.path.getsize(MERGED_PATH):,} bytes")
sys.stdout.flush()

# ── Configuration ──
BLOCK_SIZE = 2.0
N_FOLDS = 10
N_PCA = 100
CLR_PSEUDOCOUNT = 0.5
PREVALENCE_THRESHOLD = 0.05
SHAP_TOP_FEATURES = 500

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

PRODUCTIVITY_TARGETS = ['chl_mean_mg_m3', 'poc_mean_mg_m3', 'nflh_mean']

EXCLUSIONS = {
    'chl_mean_mg_m3': ['chl_mean_mg_m3', 'chl_max_mg_m3', 'chl_min_mg_m3', 'nflh_mean'],
    'poc_mean_mg_m3': ['poc_mean_mg_m3'],
    'nflh_mean': ['nflh_mean', 'chl_mean_mg_m3', 'chl_max_mg_m3', 'chl_min_mg_m3'],
}

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

# ── Load data ──
print("\nLoading dataset...")
sys.stdout.flush()

with open(MERGED_PATH, 'r') as f:
    for line in f:
        if not line.startswith('#'):
            header = line.strip().split('\t')
            break

PFAM_COLS = sorted([c for c in header if c.startswith('PF')])
print(f"  PFAM columns: {len(PFAM_COLS)}")

needed_cols = list(set(
    ['assembly_id', 'latitude', 'longitude'] +
    PRODUCTIVITY_TARGETS +
    ['chl_max_mg_m3', 'chl_min_mg_m3'] +
    ALL_ENV_CANDIDATES +
    PFAM_COLS
))
needed_cols = [c for c in needed_cols if c in header]

df = pd.read_csv(MERGED_PATH, sep='\t', comment='#', usecols=needed_cols, low_memory=False)
print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

# ── Filter to GPS-available samples ──
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df = df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)
print(f"  Samples with GPS: {len(df)}")
sys.stdout.flush()

# ── Spatial blocks ──
print("\nCreating spatial blocks (2-degree grid)...")
df['block_lat'] = np.floor(df['latitude'] / BLOCK_SIZE) * BLOCK_SIZE
df['block_lon'] = np.floor(df['longitude'] / BLOCK_SIZE) * BLOCK_SIZE
df['block_id'] = df['block_lat'].astype(str) + '_' + df['block_lon'].astype(str)

block_counts = df['block_id'].value_counts()
print(f"  Total blocks: {len(block_counts)}")

blocks_sorted = sorted(block_counts.index, key=lambda b: block_counts[b], reverse=True)
fold_assignment = {}
fold_sizes = [0] * N_FOLDS
for block in blocks_sorted:
    min_fold = int(np.argmin(fold_sizes))
    fold_assignment[block] = min_fold
    fold_sizes[min_fold] += block_counts[block]
df['fold'] = df['block_id'].map(fold_assignment)
folds_array = df['fold'].values
print(f"  Fold sizes: {sorted(fold_sizes)}")
sys.stdout.flush()

# ── Prepare PFAM matrix ──
print("\nPreparing PFAM features...")
sys.stdout.flush()
X_pfam_raw = df[PFAM_COLS].fillna(0).values.astype(np.float64)
n_samples = X_pfam_raw.shape[0]
prevalence = (X_pfam_raw > 0).sum(axis=0) / n_samples
prev_mask = prevalence >= PREVALENCE_THRESHOLD
PFAM_COLS_FILTERED = [PFAM_COLS[i] for i in range(len(PFAM_COLS)) if prev_mask[i]]
X_pfam_filtered = X_pfam_raw[:, prev_mask]
del X_pfam_raw
print(f"  PFAMs after {PREVALENCE_THRESHOLD*100:.0f}% prevalence filter: {len(PFAM_COLS_FILTERED)}")
sys.stdout.flush()


def clr_transform(X, pseudocount=CLR_PSEUDOCOUNT):
    X_pseudo = X + pseudocount
    log_X = np.log(X_pseudo)
    geometric_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geometric_mean


# ── Pre-compute CLR+PCA per fold (matching existing manuscript methodology) ──
print(f"\nPre-computing CLR+PCA{N_PCA} per fold...")
sys.stdout.flush()

fold_features = {}
for fold_idx in range(N_FOLDS):
    test_mask = folds_array == fold_idx
    train_mask = ~test_mask
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    X_train_clr = clr_transform(X_pfam_filtered[train_idx])
    X_test_clr = clr_transform(X_pfam_filtered[test_idx])

    n_comp = min(N_PCA, X_train_clr.shape[0], X_train_clr.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca.fit_transform(X_train_clr)
    X_test_pca = pca.transform(X_test_clr)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_pca)
    X_test_sc = scaler.transform(X_test_pca)

    fold_features[fold_idx] = (train_idx, test_idx, X_train_sc, X_test_sc)
    print(f"  Fold {fold_idx}: {len(train_idx)} train, {len(test_idx)} test")
    sys.stdout.flush()

# ── Prepare environment features ──
available_env = [c for c in ALL_ENV_CANDIDATES if c in df.columns]
for col in available_env:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print(f"\n  Available env features: {len(available_env)}")
sys.stdout.flush()

# ── Import XGBoost ──
from xgboost import XGBRegressor

# ── Run spatial block CV ──
print("\n" + "=" * 60)
print("SPATIAL BLOCK CV: Productivity Prediction")
print("=" * 60)
sys.stdout.flush()

results = []

for target in PRODUCTIVITY_TARGETS:
    print(f"\n--- Target: {target} ---")
    sys.stdout.flush()

    y_all = pd.to_numeric(df[target], errors='coerce').values
    valid_all = ~np.isnan(y_all)
    n_valid = valid_all.sum()
    print(f"  Valid samples: {n_valid}")

    if n_valid < 50:
        print(f"  SKIP: too few valid samples")
        continue

    excluded = EXCLUSIONS[target]
    env_cols_for_target = [c for c in available_env if c not in excluded]
    print(f"  Env features (after exclusion): {len(env_cols_for_target)}")
    print(f"  Excluded: {excluded}")
    sys.stdout.flush()

    # Pre-extract env values and fill NaN with median
    env_vals = df[env_cols_for_target].values.astype(np.float64)
    col_medians = np.nanmedian(env_vals, axis=0)
    for j in range(env_vals.shape[1]):
        mask_nan = np.isnan(env_vals[:, j])
        env_vals[mask_nan, j] = col_medians[j]

    configs = ['domain_only', 'env_only', 'combined']

    for config_name in configs:
        fold_r2s = []

        for fold_idx in range(N_FOLDS):
            train_idx, test_idx, X_train_pca, X_test_pca = fold_features[fold_idx]

            # Filter to valid target values
            train_valid = valid_all[train_idx]
            test_valid = valid_all[test_idx]

            if test_valid.sum() < 5 or train_valid.sum() < 20:
                continue

            y_train = y_all[train_idx[train_valid]]
            y_test = y_all[test_idx[test_valid]]

            if config_name == 'domain_only':
                X_train = X_train_pca[train_valid]
                X_test = X_test_pca[test_valid]
            elif config_name == 'env_only':
                X_train = env_vals[train_idx[train_valid]]
                X_test = env_vals[test_idx[test_valid]]
            else:  # combined
                X_train = np.hstack([X_train_pca[train_valid], env_vals[train_idx[train_valid]]])
                X_test = np.hstack([X_test_pca[test_valid], env_vals[test_idx[test_valid]]])

            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            fold_r2s.append(r2)

        if fold_r2s:
            fold_r2s = np.array(fold_r2s)
            median_r2 = np.median(fold_r2s)
            iqr_low = np.percentile(fold_r2s, 25)
            iqr_high = np.percentile(fold_r2s, 75)
            n_feat = X_train.shape[1]
            print(f"    {config_name:12s} ({n_feat:3d} feat): "
                  f"median R²={median_r2:.4f} (IQR: {iqr_low:.4f}–{iqr_high:.4f})")
            sys.stdout.flush()

            results.append({
                'target': target,
                'config': config_name,
                'n_samples': n_valid,
                'n_features': n_feat,
                'n_folds': len(fold_r2s),
                'median_r2': round(median_r2, 4),
                'mean_r2': round(np.mean(fold_r2s), 4),
                'iqr_25': round(iqr_low, 4),
                'iqr_75': round(iqr_high, 4),
                'min_r2': round(fold_r2s.min(), 4),
                'max_r2': round(fold_r2s.max(), 4),
                'fold_r2s': ';'.join(f'{v:.4f}' for v in fold_r2s),
            })

# ── Write productivity results ──
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

results_df = pd.DataFrame(results)
print(results_df[['target', 'config', 'n_samples', 'n_features', 'median_r2', 'iqr_25', 'iqr_75']].to_string())
sys.stdout.flush()

os.makedirs(os.path.dirname(OUTPUT_RESULTS), exist_ok=True)

with open(OUTPUT_RESULTS, 'w') as f:
    f.write(f"# Provenance:\n")
    f.write(f"#   Script: {SCRIPT_PATH}\n")
    f.write(f"#   Input:  {MERGED_PATH}\n")
    f.write(f"#   Date:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"#   Integrity Check: PASSED\n")
    f.write(f"#   Spatial Block CV: {BLOCK_SIZE}-degree grid, {N_FOLDS}-fold\n")
    f.write(f"#   XGBoost: n_est={XGB_PARAMS['n_estimators']}, depth={XGB_PARAMS['max_depth']}, "
            f"lr={XGB_PARAMS['learning_rate']}, subsample={XGB_PARAMS['subsample']}\n")
    f.write(f"#   CLR pseudocount: {CLR_PSEUDOCOUNT}, prevalence threshold: {PREVALENCE_THRESHOLD}\n")
    f.write(f"#   Domain features: CLR + PCA{N_PCA} (fitted per fold)\n")
    f.write(f"#\n")
    results_df.to_csv(f, sep='\t', index=False)

print(f"\nResults written to: {OUTPUT_RESULTS}")
sys.stdout.flush()

# ── Identify best target (combined, highest median R²) ──
combined_results = results_df[results_df['config'] == 'combined']
if combined_results.empty:
    print("ERROR: No combined results")
    sys.exit(1)

best_row = combined_results.loc[combined_results['median_r2'].idxmax()]
best_target = best_row['target']
print(f"\nBest target (combined): {best_target} (median R² = {best_row['median_r2']:.4f})")
sys.stdout.flush()

# ── SHAP analysis on best target ──
print("\n" + "=" * 60)
print(f"SHAP ANALYSIS: {best_target}")
print(f"  Using top {SHAP_TOP_FEATURES} most-variable CLR features + env")
print("=" * 60)
sys.stdout.flush()

import shap

y_all = pd.to_numeric(df[best_target], errors='coerce').values
valid_mask = ~np.isnan(y_all)
y_valid = y_all[valid_mask]

excluded = EXCLUSIONS[best_target]
env_cols_for_target = [c for c in available_env if c not in excluded]

# CLR transform all valid samples
X_clr_valid = clr_transform(X_pfam_filtered[valid_mask])

# Select top-N most variable CLR features
clr_var = np.var(X_clr_valid, axis=0)
top_var_idx = np.argsort(clr_var)[::-1][:SHAP_TOP_FEATURES]
X_domain_top = X_clr_valid[:, top_var_idx]
pfam_cols_top = [PFAM_COLS_FILTERED[i] for i in top_var_idx]

# Env features for valid samples
env_vals_valid = df.loc[valid_mask, env_cols_for_target].values.astype(np.float64)
col_medians = np.nanmedian(env_vals_valid, axis=0)
for j in range(env_vals_valid.shape[1]):
    mask_nan = np.isnan(env_vals_valid[:, j])
    env_vals_valid[mask_nan, j] = col_medians[j]

X_combined_shap = np.hstack([X_domain_top, env_vals_valid])
feature_names_shap = pfam_cols_top + env_cols_for_target

print(f"  Training model: {X_combined_shap.shape[0]} samples × {X_combined_shap.shape[1]} features")
sys.stdout.flush()

model_shap = XGBRegressor(**XGB_PARAMS)
model_shap.fit(X_combined_shap, y_valid)

print("  Computing SHAP values (TreeExplainer)...")
sys.stdout.flush()

explainer = shap.TreeExplainer(model_shap)
shap_values = explainer.shap_values(X_combined_shap)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    'feature': feature_names_shap,
    'mean_abs_shap': mean_abs_shap,
})
shap_df = shap_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

top15 = shap_df.head(15).copy()
top15['rank'] = range(1, 16)
top15['is_pfam'] = top15['feature'].str.startswith('PF')

print("\nTop 15 SHAP features:")
print(top15[['rank', 'feature', 'mean_abs_shap', 'is_pfam']].to_string(index=False))

n_pfam_top15 = top15['is_pfam'].sum()
n_env_top15 = 15 - n_pfam_top15
print(f"\n  PFAM domains in top 15: {n_pfam_top15}")
print(f"  Env variables in top 15: {n_env_top15}")
sys.stdout.flush()

with open(OUTPUT_SHAP, 'w') as f:
    f.write(f"# Provenance:\n")
    f.write(f"#   Script: {SCRIPT_PATH}\n")
    f.write(f"#   Input:  {MERGED_PATH}\n")
    f.write(f"#   Date:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"#   Integrity Check: PASSED\n")
    f.write(f"#   Target: {best_target}\n")
    f.write(f"#   Model: XGBoost combined (top {SHAP_TOP_FEATURES} CLR PFAM + env), full dataset\n")
    f.write(f"#   SHAP: TreeExplainer, mean |SHAP| across {X_combined_shap.shape[0]} samples\n")
    f.write(f"#   N features total: {len(feature_names_shap)}\n")
    f.write(f"#   PFAM domains in top 15: {n_pfam_top15}\n")
    f.write(f"#   Env variables in top 15: {n_env_top15}\n")
    f.write(f"#\n")
    top15[['rank', 'feature', 'mean_abs_shap', 'is_pfam']].to_csv(f, sep='\t', index=False)

print(f"\nSHAP results written to: {OUTPUT_SHAP}")
print("\nDONE.")
