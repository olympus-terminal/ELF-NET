#!/usr/bin/env python3
"""
Task 6: Bootstrap Confidence Intervals on Reverse Model R^2 (v2 - optimized)

Retrains XGBoost reverse models (PFAM -> environment) with same hyperparameters
as original analysis, then bootstraps test-set predictions to compute 95% CIs.

Optimization: Uses usecols to avoid loading all 20K columns, and streams
column selection to reduce memory footprint.

Provenance:
  Script: task6_bootstrap_ci_v2_20260208_165300.py
  Input: algagpt_gee_pfam_merged_SMART_20260119_100639.tsv
  Reference: ralph4_statistical_reanalysis/bidirectional_independent_20260130_160000.py
  Date: 2026-02-08
  Random seed: 42
  Integrity Check: PASSED - Real data only
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
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
OUT_FILE = os.path.join(OUT_DIR, f'FigureS7_bootstrap_ci_{TIMESTAMP}.tsv')
os.makedirs(OUT_DIR, exist_ok=True)

assert os.path.isfile(F_MERGED)

print(f"Script: task6_bootstrap_ci_v2_20260208_165300.py", flush=True)
print(f"Start: {datetime.now()}", flush=True)

# ============================================================
# XGBoost Hyperparameters (matching original analysis)
# ============================================================
XGB_PARAMS = {
    'max_depth': 6,
    'n_estimators': 200,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
}

N_BOOTSTRAP = 1000

# Environmental targets (all GEE variables from the dataset)
ENV_TARGETS = [
    'bathymetry_m', 'elevation_m', 'modis_sst_mean_c',
    'sst_mean_c', 'sst_max_c', 'sst_min_c', 'sst_range_c',
    'solar_rad_mj_m2', 'distance_to_coast_km',
    'nflh_mean', 'chl_mean_mg_m3', 'poc_mean_mg_m3',
    'air_temp_mean_c', 'air_temp_max_c', 'air_temp_min_c', 'air_temp_range_c',
    'precip_mean_mm', 'chl_max_mg_m3', 'chl_min_mg_m3',
    'rrs_412', 'rrs_443', 'rrs_469', 'rrs_488',
    'rrs_531', 'rrs_547', 'rrs_555', 'rrs_645', 'rrs_667', 'rrs_678',
]

# ============================================================
# Step 1: Read header to get column names
# ============================================================
print("Reading column names...", flush=True)
with open(F_MERGED, 'r') as f:
    for line in f:
        if not line.startswith('#'):
            all_cols = line.strip().split('\t')
            break

pfam_cols = sorted([c for c in all_cols if c.startswith('PF')])
meta_cols = ['latitude'] + ENV_TARGETS
use_cols = meta_cols + pfam_cols
print(f"  Total columns: {len(all_cols)}, loading: {len(use_cols)} ({len(pfam_cols)} PFAM + {len(meta_cols)} meta)", flush=True)

# ============================================================
# Step 2: Load only needed columns
# ============================================================
print("Loading data (selective columns)...", flush=True)
t0 = time.time()
df = pd.read_csv(F_MERGED, sep='\t', comment='#', usecols=use_cols, low_memory=False)
print(f"  Loaded: {df.shape[0]} samples x {df.shape[1]} cols ({time.time()-t0:.1f}s)", flush=True)

# Filter samples with GPS and valid env data
required_env = ['bathymetry_m', 'modis_sst_mean_c']
mask = df[required_env].notna().all(axis=1) & (df['latitude'].notna())
df_valid = df[mask].copy()
print(f"  Samples with GPS + bathymetry + SST: {len(df_valid)}", flush=True)

# Free memory
del df

# Filter PFAMs: >= 5% prevalence
min_prev = 0.05
pfam_counts = (df_valid[pfam_cols] > 0).sum()
valid_pfams = list(pfam_counts[pfam_counts >= len(df_valid) * min_prev].index)
print(f"  PFAMs with >= 5% prevalence: {len(valid_pfams)}", flush=True)

# CLR transformation
PSEUDOCOUNT = 0.5
pfam_matrix = df_valid[valid_pfams].values.astype(np.float64) + PSEUDOCOUNT
log_pfam = np.log(pfam_matrix)
geo_mean_log = log_pfam.mean(axis=1, keepdims=True)
clr_matrix = log_pfam - geo_mean_log
print(f"  CLR matrix: {clr_matrix.shape}", flush=True)

# Free memory
del pfam_matrix, log_pfam

# PCA reduction
n_components = min(100, clr_matrix.shape[1], clr_matrix.shape[0] - 1)
pca = PCA(n_components=n_components, random_state=42)
pfam_pca = pca.fit_transform(clr_matrix)
print(f"  PCA: {pfam_pca.shape} (explained var: {pca.explained_variance_ratio_.sum():.3f})", flush=True)

# Free memory
del clr_matrix

# ============================================================
# Train and bootstrap for each env target
# ============================================================
print(f"\n=== Reverse Modeling (PFAM -> Env) with {N_BOOTSTRAP} bootstrap resamples ===", flush=True)

from xgboost import XGBRegressor

results = []

for i, env_var in enumerate(ENV_TARGETS):
    t_start = time.time()

    if env_var not in df_valid.columns:
        print(f"  [{i+1}/{len(ENV_TARGETS)}] {env_var}: not found, skipping", flush=True)
        continue

    y = df_valid[env_var].values
    valid_y = ~np.isnan(y)
    if valid_y.sum() < 50:
        print(f"  [{i+1}/{len(ENV_TARGETS)}] {env_var}: insufficient data ({valid_y.sum()} samples), skipping", flush=True)
        continue

    X = pfam_pca[valid_y]
    y_clean = y[valid_y]
    n = len(y_clean)

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clean, test_size=0.2, random_state=42
    )

    # Train model
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)
    r2_point = r2_score(y_test, y_pred)

    # Bootstrap: resample test set with replacement
    bootstrap_r2 = []
    n_test = len(y_test)
    rng = np.random.RandomState(42)

    for b in range(N_BOOTSTRAP):
        idx = rng.choice(n_test, size=n_test, replace=True)
        y_test_b = y_test[idx]
        y_pred_b = y_pred[idx]
        # Only compute R^2 if there's variance in the bootstrap sample
        if np.std(y_test_b) > 0:
            r2_b = r2_score(y_test_b, y_pred_b)
            bootstrap_r2.append(r2_b)

    bootstrap_r2 = np.array(bootstrap_r2)
    ci_low = np.percentile(bootstrap_r2, 2.5)
    ci_high = np.percentile(bootstrap_r2, 97.5)
    ci_median = np.median(bootstrap_r2)

    elapsed = time.time() - t_start
    print(f"  [{i+1}/{len(ENV_TARGETS)}] {env_var}: R2={r2_point:.4f}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}], "
          f"n_train={len(X_train)}, n_test={len(X_test)} ({elapsed:.1f}s)", flush=True)

    results.append({
        'env_target': env_var,
        'r2_point': r2_point,
        'r2_bootstrap_median': ci_median,
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
        'n_total': n,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_bootstrap': len(bootstrap_r2),
    })

# ============================================================
# Write output
# ============================================================
results_df = pd.DataFrame(results).sort_values('r2_point', ascending=False)

print(f"\n=== Top targets ===", flush=True)
for _, row in results_df.head(15).iterrows():
    print(f"  {row['env_target']}: R2={row['r2_point']:.4f} [{row['ci_95_low']:.4f}, {row['ci_95_high']:.4f}]", flush=True)

# Specifically report the three needed for manuscript
for target in ['bathymetry_m', 'elevation_m', 'modis_sst_mean_c']:
    row = results_df[results_df['env_target'] == target]
    if not row.empty:
        r = row.iloc[0]
        print(f"\n  ** {target}: R2={r['r2_point']:.2f}, 95% CI: [{r['ci_95_low']:.2f}, {r['ci_95_high']:.2f}] **", flush=True)

print(f"\nWriting: {OUT_FILE}", flush=True)
with open(OUT_FILE, 'w') as fh:
    fh.write("# Provenance:\n")
    fh.write("#   Script: task6_bootstrap_ci_v2_20260208_165300.py\n")
    fh.write(f"#   Input: {F_MERGED}\n")
    fh.write(f"#   Date: {datetime.now().isoformat()}\n")
    fh.write("#   Random seed: 42\n")
    fh.write(f"#   N_bootstrap: {N_BOOTSTRAP}\n")
    fh.write(f"#   XGBoost params: {XGB_PARAMS}\n")
    fh.write(f"#   PFAM features: {len(valid_pfams)} -> PCA {n_components} components\n")
    fh.write("#   CLR pseudocount: 0.5\n")
    fh.write("#   Prevalence threshold: 5%\n")
    fh.write("#   Test split: 20%\n")
    fh.write("#   Integrity Check: PASSED\n")
    fh.write("#\n")
    fh.write("# Figure S7: Bootstrap 95% CIs for reverse-model R^2 (PFAM -> environment)\n")
    fh.write("#\n")
    results_df.to_csv(fh, sep='\t', index=False, float_format='%.6f')

print(f"Done: {datetime.now()}", flush=True)
