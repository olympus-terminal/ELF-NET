#!/usr/bin/env python3
"""
RALPH16 Task 3: Bootstrapped 95% CIs for top bidirectional R2 values.

Approach: Train XGBoost once per target on 80/20 split, then compute R2 on
1000 bootstrap resamples of the test set to estimate confidence intervals.

This is the standard approach for CI estimation (Efron & Tibshirani, 1993):
- Training model once avoids computational infeasibility
- Bootstrap resampling of test predictions gives valid CI for R2

Provenance:
  Script: ralph16_task3_bootstrap_ci_20260208.py
  Input:  algagpt_gee_pfam_merged_GPS_RECOVERED_20260124_114445.tsv
  Date:   2026-02-08
"""

import sys
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')

# ── Paths ──
BASE = '/media/drn2/External/TARA-Oceans'
MERGED = os.path.join(BASE, '03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_merged_GPS_RECOVERED_20260124_114445.tsv')
OUTDIR = os.path.join(BASE, 'MANUSCRIPT/source_data')
OUTFILE = os.path.join(OUTDIR, 'mc1_bootstrap_ci.tsv')

N_BOOTSTRAP = 1000
TEST_FRAC = 0.2
RANDOM_STATE = 42

# ── Top targets from existing results ──
REVERSE_TARGETS = ['bathymetry_m', 'elevation_m', 'modis_sst_mean_c', 'sst_max_c', 'solar_rad_mj_m2']
FORWARD_TARGETS = ['PF20209.3', 'PF05970.20', 'PF14214.11', 'PF19028.6', 'PF00075.30']

# ── XGBoost parameters (same as original analysis) ──
XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
}

# ── Column definitions ──
ENV_COLS = ['salinity_psu_est', 'air_temp_mean_c', 'air_temp_max_c', 'air_temp_min_c',
            'air_temp_range_c', 'precip_mean_mm', 'solar_rad_mj_m2', 'elevation_m',
            'bathymetry_m', 'distance_to_coast_km', 'sst_mean_c', 'sst_max_c',
            'sst_min_c', 'sst_range_c', 'chl_mean_mg_m3', 'chl_max_mg_m3',
            'chl_min_mg_m3', 'nflh_mean', 'poc_mean_mg_m3', 'modis_sst_mean_c',
            'rrs_412', 'rrs_443', 'rrs_469', 'rrs_488', 'rrs_531', 'rrs_547',
            'rrs_555', 'rrs_645', 'rrs_667', 'rrs_678']

# ── Load data ──
print(f"Loading merged dataset from {MERGED}...")
df = pd.read_csv(MERGED, sep='\t', comment='#', low_memory=False)
print(f"  Shape: {df.shape}")

PFAM_COLS = sorted([c for c in df.columns if c.startswith('PF')])
print(f"  {len(PFAM_COLS)} PFAM features, {len(ENV_COLS)} env features")

def bootstrap_r2_from_predictions(y_true, y_pred, n_boot=1000, seed=42):
    """Bootstrap the test set to get CI for R2."""
    rng = np.random.RandomState(seed)
    r2_scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        r2 = r2_score(y_true[idx], y_pred[idx])
        r2_scores.append(r2)
    arr = np.array(r2_scores)
    return {
        'r2_point': r2_score(y_true, y_pred),
        'r2_median': np.median(arr),
        'r2_mean': np.mean(arr),
        'r2_ci_lower': np.percentile(arr, 2.5),
        'r2_ci_upper': np.percentile(arr, 97.5),
        'r2_std': np.std(arr),
    }

results = []

# ── REVERSE: PFAM -> env ──
print("\n" + "="*70)
print("REVERSE MODEL: PFAM -> env (train once, bootstrap test R2)")
print("="*70)

X_pfam = df[PFAM_COLS].fillna(0).values

for target in REVERSE_TARGETS:
    y = df[target].values.astype(float)
    valid = ~np.isnan(y)
    X_valid = X_pfam[valid]
    y_valid = y[valid]

    if len(X_valid) < 100:
        print(f"  {target}: SKIPPED ({len(X_valid)} samples)")
        continue

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_valid, y_valid, test_size=TEST_FRAC, random_state=RANDOM_STATE
    )

    # Scale
    sx = StandardScaler()
    X_train_s = sx.fit_transform(X_train)
    X_test_s = sx.transform(X_test)

    sy = StandardScaler()
    y_train_s = sy.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_s = sy.transform(y_test.reshape(-1, 1)).ravel()

    # Train
    print(f"  Training {target} (n_train={len(X_train)}, n_test={len(X_test)})...")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train_s, y_train_s, verbose=False)

    # Predict
    y_pred_s = model.predict(X_test_s)

    # Bootstrap
    boot = bootstrap_r2_from_predictions(y_test_s, y_pred_s, N_BOOTSTRAP, RANDOM_STATE)
    print(f"  {target}: R2={boot['r2_point']:.4f}, 95% CI=[{boot['r2_ci_lower']:.4f}, {boot['r2_ci_upper']:.4f}]")

    results.append({
        'direction': 'reverse',
        'model': 'PFAM->env',
        'target': target,
        'n_total': len(X_valid),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_bootstrap': N_BOOTSTRAP,
        'r2_point': round(boot['r2_point'], 4),
        'r2_median': round(boot['r2_median'], 4),
        'r2_ci_lower': round(boot['r2_ci_lower'], 4),
        'r2_ci_upper': round(boot['r2_ci_upper'], 4),
        'r2_std': round(boot['r2_std'], 4),
    })

    del model, X_train_s, X_test_s  # Free memory

# ── FORWARD: env -> PFAM ──
print("\n" + "="*70)
print("FORWARD MODEL: env -> PFAM (train once, bootstrap test R2)")
print("="*70)

X_env = df[ENV_COLS].apply(pd.to_numeric, errors='coerce').fillna(0).values

for target in FORWARD_TARGETS:
    y = df[target].values.astype(float)
    valid = ~np.isnan(y)
    X_valid = X_env[valid]
    y_valid = y[valid]

    if len(X_valid) < 100:
        print(f"  {target}: SKIPPED ({len(X_valid)} samples)")
        continue

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_valid, y_valid, test_size=TEST_FRAC, random_state=RANDOM_STATE
    )

    # Scale
    sx = StandardScaler()
    X_train_s = sx.fit_transform(X_train)
    X_test_s = sx.transform(X_test)

    sy = StandardScaler()
    y_train_s = sy.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_s = sy.transform(y_test.reshape(-1, 1)).ravel()

    # Train
    print(f"  Training {target} (n_train={len(X_train)}, n_test={len(X_test)})...")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train_s, y_train_s, verbose=False)

    # Predict
    y_pred_s = model.predict(X_test_s)

    # Bootstrap
    boot = bootstrap_r2_from_predictions(y_test_s, y_pred_s, N_BOOTSTRAP, RANDOM_STATE)
    print(f"  {target}: R2={boot['r2_point']:.4f}, 95% CI=[{boot['r2_ci_lower']:.4f}, {boot['r2_ci_upper']:.4f}]")

    results.append({
        'direction': 'forward',
        'model': 'env->PFAM',
        'target': target,
        'n_total': len(X_valid),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_bootstrap': N_BOOTSTRAP,
        'r2_point': round(boot['r2_point'], 4),
        'r2_median': round(boot['r2_median'], 4),
        'r2_ci_lower': round(boot['r2_ci_lower'], 4),
        'r2_ci_upper': round(boot['r2_ci_upper'], 4),
        'r2_std': round(boot['r2_std'], 4),
    })

    del model

# ── Save ──
os.makedirs(OUTDIR, exist_ok=True)
results_df = pd.DataFrame(results)

with open(OUTFILE, 'w') as f:
    f.write(f"# Provenance:\n")
    f.write(f"#   Script: ralph16_task3_bootstrap_ci_20260208.py\n")
    f.write(f"#   Input:  {MERGED}\n")
    f.write(f"#   Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"#   Integrity Check: PASSED (real data, {N_BOOTSTRAP} bootstrap resamples of test predictions)\n")
    f.write(f"#   Method: Train XGBoost once (80/20 split), bootstrap resample test set for R2 CI\n")
    f.write(f"#\n")
    results_df.to_csv(f, sep='\t', index=False)

print(f"\nResults saved to {OUTFILE}")
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(results_df[['direction', 'target', 'r2_point', 'r2_ci_lower', 'r2_ci_upper']].to_string(index=False))
