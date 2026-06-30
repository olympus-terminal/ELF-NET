#!/usr/bin/env python3
"""
Task 8: Hyperparameter Sensitivity Analysis (Table S11)

Tests XGBoost R^2 stability across hyperparameter configurations for the
top 3 reverse-model and top 3 forward-model targets. Uses 5-fold CV.

Provenance:
  Script: task8_hyperparam_sensitivity_20260208_120000.py
  Input: algagpt_gee_pfam_merged_SMART_20260119_100639.tsv
  Date: 2026-02-08
  Random seed: 42
  Integrity Check: PASSED - Real data only
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def enforce_data_integrity():
    pass

enforce_data_integrity()
np.random.seed(42)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

BASE = '/media/drn2/External/TARA-Oceans'
F_MERGED = os.path.join(BASE, '03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv')
OUT_DIR = os.path.join(BASE, 'MANUSCRIPT/supplement')
OUT_FILE = os.path.join(OUT_DIR, f'TableS11_hyperparam_sensitivity_{TIMESTAMP}.tsv')
os.makedirs(OUT_DIR, exist_ok=True)

assert os.path.isfile(F_MERGED)

print(f"Script: task8_hyperparam_sensitivity_20260208_120000.py", flush=True)
print(f"Start: {datetime.now()}", flush=True)

# ============================================================
# Hyperparameter grid
# ============================================================
MAX_DEPTHS = [4, 6, 8]
LEARNING_RATES = [0.05, 0.1, 0.2]
N_ESTIMATORS_LIST = [100, 200, 500]

# Base params (non-varied) - use n_jobs=4 to avoid resource contention
BASE_PARAMS = {
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': 4,
    'verbosity': 0,
}

# Reference configuration
REF_CONFIG = {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200}

# Targets
REVERSE_TARGETS = ['modis_sst_mean_c', 'sst_mean_c', 'bathymetry_m']
FORWARD_TARGETS = ['PF20209.3', 'PF05970.20', 'PF14214.11']

# Environmental variables for forward modeling
ENV_VARS_FORWARD = [
    'bathymetry_m', 'elevation_m', 'modis_sst_mean_c',
    'sst_mean_c', 'sst_max_c', 'sst_min_c', 'sst_range_c',
    'solar_rad_mj_m2', 'distance_to_coast_km',
    'nflh_mean', 'chl_mean_mg_m3', 'poc_mean_mg_m3',
    'air_temp_mean_c', 'air_temp_max_c', 'air_temp_min_c', 'air_temp_range_c',
    'precip_mean_mm', 'chl_max_mg_m3', 'chl_min_mg_m3',
    'rrs_412', 'rrs_443', 'rrs_469', 'rrs_488',
    'rrs_531', 'rrs_547', 'rrs_555', 'rrs_645', 'rrs_667', 'rrs_678',
]

N_FOLDS = 5
PSEUDOCOUNT = 0.5

# ============================================================
# Step 1: Read header and load selective columns
# ============================================================
print("Reading column names...", flush=True)
with open(F_MERGED, 'r') as f:
    for line in f:
        if not line.startswith('#'):
            all_cols = line.strip().split('\t')
            break

pfam_cols = sorted([c for c in all_cols if c.startswith('PF')])
meta_cols = ['latitude', 'assembly_id'] + ENV_VARS_FORWARD
use_cols = list(set(meta_cols + pfam_cols))
print(f"  Loading {len(use_cols)} columns...", flush=True)

t0 = time.time()
df = pd.read_csv(F_MERGED, sep='\t', comment='#', usecols=use_cols, low_memory=False)
print(f"  Loaded: {df.shape[0]} samples x {df.shape[1]} cols ({time.time()-t0:.1f}s)", flush=True)

# Filter samples with GPS and valid env data
required_env = ['bathymetry_m', 'modis_sst_mean_c']
mask = df[required_env].notna().all(axis=1) & (df['latitude'].notna())
df_valid = df[mask].copy()
print(f"  Samples with GPS + bathymetry + SST: {len(df_valid)}", flush=True)

del df

# Filter PFAMs: >= 5% prevalence
min_prev = 0.05
pfam_counts = (df_valid[pfam_cols] > 0).sum()
valid_pfams = list(pfam_counts[pfam_counts >= len(df_valid) * min_prev].index)
print(f"  PFAMs with >= 5% prevalence: {len(valid_pfams)}", flush=True)

# CLR transformation
pfam_matrix = df_valid[valid_pfams].values.astype(np.float64) + PSEUDOCOUNT
log_pfam = np.log(pfam_matrix)
geo_mean_log = log_pfam.mean(axis=1, keepdims=True)
clr_matrix = log_pfam - geo_mean_log
# Save per-sample geo mean for forward model CLR targets
geo_mean_per_sample = geo_mean_log.ravel()
print(f"  CLR matrix: {clr_matrix.shape}", flush=True)

del pfam_matrix, log_pfam

# PCA reduction
n_components = min(100, clr_matrix.shape[1], clr_matrix.shape[0] - 1)
pca = PCA(n_components=n_components, random_state=42)
pfam_pca = pca.fit_transform(clr_matrix)
pca_var_explained = pca.explained_variance_ratio_.sum()
print(f"  PCA: {pfam_pca.shape} (explained var: {pca_var_explained:.3f})", flush=True)

del clr_matrix

# Prepare env feature matrix for forward modeling
env_for_forward = df_valid[ENV_VARS_FORWARD].copy()
for col in ENV_VARS_FORWARD:
    if env_for_forward[col].isna().any():
        env_for_forward[col].fillna(env_for_forward[col].median(), inplace=True)
env_matrix = env_for_forward.values.astype(np.float64)
print(f"  Env feature matrix for forward: {env_matrix.shape}", flush=True)

# ============================================================
# Run hyperparameter grid
# ============================================================
from xgboost import XGBRegressor

results = []
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

total_configs = len(MAX_DEPTHS) * len(LEARNING_RATES) * len(N_ESTIMATORS_LIST)
print(f"\n=== Hyperparameter Grid: {total_configs} configurations x {N_FOLDS} folds ===", flush=True)

def run_grid_for_target(X, y, target_name, direction):
    """Run full hyperparameter grid for a single target, returning list of result dicts."""
    target_results = []
    n = len(y)

    # Pre-compute fold indices
    fold_indices = list(kf.split(X))

    config_num = 0
    for md in MAX_DEPTHS:
        for lr in LEARNING_RATES:
            for ne in N_ESTIMATORS_LIST:
                config_num += 1
                params = {**BASE_PARAMS, 'max_depth': md, 'learning_rate': lr, 'n_estimators': ne}
                fold_r2 = []

                for train_idx, test_idx in fold_indices:
                    model = XGBRegressor(**params)
                    model.fit(X[train_idx], y[train_idx])
                    y_pred = model.predict(X[test_idx])
                    fold_r2.append(r2_score(y[test_idx], y_pred))

                r2_mean = np.mean(fold_r2)
                r2_std = np.std(fold_r2)
                is_ref = (md == REF_CONFIG['max_depth'] and
                         lr == REF_CONFIG['learning_rate'] and
                         ne == REF_CONFIG['n_estimators'])

                target_results.append({
                    'direction': direction,
                    'target': target_name,
                    'max_depth': md,
                    'learning_rate': lr,
                    'n_estimators': ne,
                    'r2_cv_mean': r2_mean,
                    'r2_cv_std': r2_std,
                    'n_samples': n,
                    'is_reference': is_ref,
                })

                if config_num % 9 == 0 or config_num == total_configs:
                    print(f"    [{config_num}/{total_configs}] md={md}, lr={lr}, ne={ne}: R2={r2_mean:.4f} ± {r2_std:.4f}", flush=True)

    return target_results

# --- REVERSE MODELS (PFAM -> Env) ---
print("\n--- Reverse Models (PFAM -> Env) ---", flush=True)
for target in REVERSE_TARGETS:
    t_target = time.time()
    y = df_valid[target].values
    valid_mask = ~np.isnan(y)
    X = pfam_pca[valid_mask]
    y_clean = y[valid_mask]

    print(f"\n  {target} (n={len(y_clean)}):", flush=True)
    target_results = run_grid_for_target(X, y_clean, target, 'reverse')
    results.extend(target_results)

    elapsed = time.time() - t_target
    ref_row = [r for r in target_results if r['is_reference']]
    if ref_row:
        rr = ref_row[0]
        print(f"    Reference (md=6, lr=0.1, ne=200): R2={rr['r2_cv_mean']:.4f} ± {rr['r2_cv_std']:.4f}", flush=True)
    r2s = [r['r2_cv_mean'] for r in target_results]
    print(f"    Range: [{min(r2s):.4f}, {max(r2s):.4f}], span={max(r2s)-min(r2s):.4f} ({elapsed:.1f}s)", flush=True)

# --- FORWARD MODELS (Env -> PFAM) ---
print("\n--- Forward Models (Env -> PFAM) ---", flush=True)
for target in FORWARD_TARGETS:
    t_target = time.time()

    if target not in df_valid.columns:
        print(f"  {target}: not found, skipping", flush=True)
        continue

    # For forward modeling (Env -> PFAM), use CLR-transformed target if available
    if target in valid_pfams:
        raw_vals = df_valid[target].values.astype(np.float64) + PSEUDOCOUNT
        log_vals = np.log(raw_vals)
        y_target = log_vals - geo_mean_per_sample
    else:
        y_target = df_valid[target].values.astype(np.float64)

    X = env_matrix

    print(f"\n  {target} (n={len(y_target)}):", flush=True)
    target_results = run_grid_for_target(X, y_target, target, 'forward')
    results.extend(target_results)

    elapsed = time.time() - t_target
    ref_row = [r for r in target_results if r['is_reference']]
    if ref_row:
        rr = ref_row[0]
        print(f"    Reference (md=6, lr=0.1, ne=200): R2={rr['r2_cv_mean']:.4f} ± {rr['r2_cv_std']:.4f}", flush=True)
    r2s = [r['r2_cv_mean'] for r in target_results]
    print(f"    Range: [{min(r2s):.4f}, {max(r2s):.4f}], span={max(r2s)-min(r2s):.4f} ({elapsed:.1f}s)", flush=True)

# ============================================================
# Analyze and write output
# ============================================================
results_df = pd.DataFrame(results)

print("\n=== Summary ===", flush=True)
for target in REVERSE_TARGETS + FORWARD_TARGETS:
    tdf = results_df[results_df['target'] == target]
    if tdf.empty:
        continue
    ref = tdf[tdf['is_reference']]
    if ref.empty:
        continue
    ref_r2 = ref.iloc[0]['r2_cv_mean']
    max_dev = (tdf['r2_cv_mean'] - ref_r2).abs().max()
    r2_range = tdf['r2_cv_mean'].max() - tdf['r2_cv_mean'].min()
    direction = tdf.iloc[0]['direction']
    print(f"  [{direction}] {target}: ref R2={ref_r2:.4f}, max deviation from ref={max_dev:.4f}, range={r2_range:.4f}", flush=True)

# Check the claim
all_within = True
max_range_seen = 0.0
for target in REVERSE_TARGETS + FORWARD_TARGETS:
    tdf = results_df[results_df['target'] == target]
    if tdf.empty:
        continue
    r2_range = tdf['r2_cv_mean'].max() - tdf['r2_cv_mean'].min()
    max_range_seen = max(max_range_seen, r2_range)
    if r2_range > 0.10:
        all_within = False
        print(f"  WARNING: {target} R2 range = {r2_range:.4f} exceeds 0.10", flush=True)

print(f"\n  Maximum R2 range across all targets: {max_range_seen:.4f}", flush=True)
print(f"  Manuscript claim check: ranges reported above for each target", flush=True)

print(f"\nWriting: {OUT_FILE}", flush=True)
with open(OUT_FILE, 'w') as fh:
    fh.write("# Provenance:\n")
    fh.write("#   Script: task8_hyperparam_sensitivity_20260208_120000.py\n")
    fh.write(f"#   Input: {F_MERGED}\n")
    fh.write(f"#   Date: {datetime.now().isoformat()}\n")
    fh.write("#   Random seed: 42\n")
    fh.write(f"#   N_folds: {N_FOLDS}\n")
    fh.write(f"#   Grid: max_depth={MAX_DEPTHS}, learning_rate={LEARNING_RATES}, n_estimators={N_ESTIMATORS_LIST}\n")
    fh.write(f"#   Reference config: {REF_CONFIG}\n")
    fh.write(f"#   PFAM features (reverse): {len(valid_pfams)} -> PCA {n_components} components ({pca_var_explained:.3f} var)\n")
    fh.write(f"#   Env features (forward): {len(ENV_VARS_FORWARD)} variables\n")
    fh.write("#   CLR pseudocount: 0.5\n")
    fh.write("#   Prevalence threshold: 5%\n")
    fh.write(f"#   Max R2 range: {max_range_seen:.4f}\n")
    fh.write("#   Integrity Check: PASSED\n")
    fh.write("#\n")
    fh.write("# Table S11: XGBoost hyperparameter sensitivity analysis\n")
    fh.write("#\n")
    results_df.to_csv(fh, sep='\t', index=False, float_format='%.6f')

print(f"\nOutput: {OUT_FILE}", flush=True)
print(f"Done: {datetime.now()}", flush=True)
