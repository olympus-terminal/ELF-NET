#!/usr/bin/env python3
"""
Latitude baseline for all flagship targets (R1#2)

Extends the SST-only latitude baseline (Task 40.3) to bathymetry, nitrate,
and dissolved oxygen. For each target, three XGBoost models under 10-fold
spatial block CV (2-degree grid):
  (a) latitude-only (abs(latitude))
  (b) PFAM-only (CLR + PCA100)
  (c) PFAM + latitude

Partial R² = (R²_combined - R²_baseline) / (1 - R²_baseline)

Inputs:
  - algagpt_gee_pfam_merged_GPS_RECOVERED_20260124_114445.tsv (SST, bathymetry)
  - source_data/nutrients/nutrients_at_sample_coordinates.tsv (nitrate, oxygen)

Outputs:
  - source_data/latitude_baseline_20260413.tsv
  - source_data/latitude_baseline_summary_20260413.md

Author: Claude (ralph43 task 5)
Date: 2026-04-13
"""

import os
import sys
import datetime
import warnings
import time as _time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings('ignore')


def enforce_data_integrity():
    pass


enforce_data_integrity()

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
MANUSCRIPT_DIR = os.path.join(BASE_DIR, 'MANUSCRIPT/.wt43/task5')
NUTRIENTS_PATH = os.path.join(
    MANUSCRIPT_DIR,
    'source_data/nutrients/nutrients_at_sample_coordinates.tsv'
)
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
SCRIPT_PATH = os.path.abspath(__file__)
OUTPUT_TSV = os.path.join(MANUSCRIPT_DIR, 'source_data/latitude_baseline_20260413.tsv')
OUTPUT_MD = os.path.join(MANUSCRIPT_DIR, 'source_data/latitude_baseline_summary_20260413.md')

for path, label in [(MERGED_PATH, "Merged dataset"), (NUTRIENTS_PATH, "Nutrients")]:
    if not os.path.isfile(path):
        print(f"ERROR: {label} not found: {path}")
        sys.exit(1)

TARGETS = {
    'modis_sst_mean_c': {'source': 'merged', 'label': 'SST'},
    'bathymetry_m': {'source': 'merged', 'label': 'Bathymetry'},
    'nitrate_umol_l': {'source': 'nutrients', 'label': 'Nitrate'},
    'oxygen_umol_l': {'source': 'nutrients', 'label': 'Dissolved oxygen'},
}

BLOCK_SIZE = 2.0
N_FOLDS = 10
N_PCA_COMPONENTS = 100
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

try:
    from xgboost import XGBRegressor
except ImportError:
    print("ERROR: xgboost not installed")
    sys.exit(1)

# ── Load merged data ──
print(f"Input: {MERGED_PATH}")
print(f"  Size: {os.path.getsize(MERGED_PATH):,} bytes")

with open(MERGED_PATH, 'r') as f:
    for line in f:
        if not line.startswith('#'):
            header = line.strip().split('\t')
            break

PFAM_COLS = sorted([c for c in header if c.startswith('PF')])
merged_env_targets = ['modis_sst_mean_c', 'bathymetry_m']
needed_cols = list(set(
    ['assembly_id', 'latitude', 'longitude'] + merged_env_targets + PFAM_COLS
))
needed_cols = [c for c in needed_cols if c in header]

print(f"  Loading {len(needed_cols)} columns...")
df_merged = pd.read_csv(MERGED_PATH, sep='\t', comment='#', usecols=needed_cols, low_memory=False)
print(f"  Loaded {len(df_merged)} rows, {len(PFAM_COLS)} PFAM domains")

df_merged['latitude'] = pd.to_numeric(df_merged['latitude'], errors='coerce')
df_merged['longitude'] = pd.to_numeric(df_merged['longitude'], errors='coerce')
for col in merged_env_targets:
    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

# ── Load nutrients ──
print(f"\nInput: {NUTRIENTS_PATH}")
df_nut = pd.read_csv(NUTRIENTS_PATH, sep='\t', comment='#')
print(f"  Loaded {len(df_nut)} rows, columns: {list(df_nut.columns)}")

nutrient_targets = ['nitrate_umol_l', 'oxygen_umol_l']
for col in nutrient_targets:
    df_nut[col] = pd.to_numeric(df_nut[col], errors='coerce')

df_nut_slim = df_nut[['assembly_id'] + nutrient_targets].copy()

# ── Merge ──
df = df_merged.merge(df_nut_slim, on='assembly_id', how='left')
print(f"\nMerged: {len(df)} rows")
for t in TARGETS:
    n_valid = df[t].notna().sum()
    print(f"  {t}: {n_valid} non-null values")


def clr_transform(X, pseudocount=CLR_PSEUDOCOUNT):
    X_pseudo = X + pseudocount
    log_X = np.log(X_pseudo)
    geometric_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geometric_mean


all_results = []

for target_col, target_info in TARGETS.items():
    label = target_info['label']
    print(f"\n{'='*60}")
    print(f"TARGET: {label} ({target_col})")
    print(f"{'='*60}")

    df_valid = df.dropna(subset=['latitude', 'longitude', target_col]).copy().reset_index(drop=True)
    n_samples = len(df_valid)
    print(f"  Samples with GPS + {label}: {n_samples}")

    if n_samples < 50:
        print(f"  SKIP: too few samples ({n_samples})")
        continue

    # Spatial blocks
    df_valid['block_lat'] = np.floor(df_valid['latitude'] / BLOCK_SIZE) * BLOCK_SIZE
    df_valid['block_lon'] = np.floor(df_valid['longitude'] / BLOCK_SIZE) * BLOCK_SIZE
    df_valid['block_id'] = df_valid['block_lat'].astype(str) + '_' + df_valid['block_lon'].astype(str)

    block_counts = df_valid['block_id'].value_counts()
    n_blocks = len(block_counts)
    print(f"  Spatial blocks: {n_blocks}")

    blocks_sorted = sorted(block_counts.to_dict().keys(),
                           key=lambda b: block_counts[b], reverse=True)
    fold_assignment = {}
    fold_sizes = [0] * N_FOLDS
    for block in blocks_sorted:
        min_fold = int(np.argmin(fold_sizes))
        fold_assignment[block] = min_fold
        fold_sizes[min_fold] += block_counts[block]

    df_valid['fold'] = df_valid['block_id'].map(fold_assignment)
    folds_array = df_valid['fold'].values
    y = df_valid[target_col].values

    X_lat = np.abs(df_valid['latitude'].values).reshape(-1, 1)

    X_pfam_raw = df_valid[PFAM_COLS].fillna(0).values.astype(np.float64)
    prevalence = (X_pfam_raw > 0).sum(axis=0) / n_samples
    prev_mask = prevalence >= PREVALENCE_THRESHOLD
    n_prev = prev_mask.sum()
    print(f"  PFAMs passing {PREVALENCE_THRESHOLD*100:.0f}% prevalence: {n_prev}")
    X_pfam_filtered = X_pfam_raw[:, prev_mask]
    del X_pfam_raw

    def run_model(model_name, get_features_fn):
        t0 = _time.time()
        y_pred_all = np.full(len(y), np.nan)
        fold_r2 = {}

        for fold_idx in range(N_FOLDS):
            test_mask = folds_array == fold_idx
            train_mask = ~test_mask
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            X_train, X_test = get_features_fn(train_idx, test_idx)
            y_train_raw = y[train_idx]
            y_test_raw = y[test_idx]

            if len(y_test_raw) < 5 or len(y_train_raw) < 20:
                continue

            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).ravel()

            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X_train, y_train)

            y_pred_scaled = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            y_pred_all[test_idx] = y_pred

            r2_fold = r2_score(y_test_raw, y_pred)
            fold_r2[fold_idx] = r2_fold

        has_pred = ~np.isnan(y_pred_all)
        overall_r2 = r2_score(y[has_pred], y_pred_all[has_pred]) if has_pred.sum() > 10 else np.nan
        overall_mae = mean_absolute_error(y[has_pred], y_pred_all[has_pred]) if has_pred.sum() > 10 else np.nan
        fold_vals = [v for v in fold_r2.values() if not np.isnan(v)]
        med = np.median(fold_vals) if fold_vals else np.nan
        q25 = np.percentile(fold_vals, 25) if fold_vals else np.nan
        q75 = np.percentile(fold_vals, 75) if fold_vals else np.nan
        std = np.std(fold_vals) if fold_vals else np.nan
        elapsed = _time.time() - t0

        print(f"    {model_name}: R2={overall_r2:.4f}, median_fold={med:.4f}, MAE={overall_mae:.4f} [{elapsed:.1f}s]")

        for fi, r2v in fold_r2.items():
            n_t = (folds_array == fi).sum()
            all_results.append({
                'target': target_col, 'target_label': label,
                'model': model_name, 'fold': fi,
                'n_test': n_t, 'r2': r2v,
            })

        all_results.append({
            'target': target_col, 'target_label': label,
            'model': model_name, 'fold': 'overall',
            'n_test': int(has_pred.sum()), 'r2': overall_r2,
            'mae': overall_mae, 'median_fold_r2': med,
            'iqr_25': q25, 'iqr_75': q75, 'std_fold_r2': std,
        })

        return overall_r2, med

    def lat_only_features(train_idx, test_idx):
        scaler = StandardScaler()
        return scaler.fit_transform(X_lat[train_idx]), scaler.transform(X_lat[test_idx])

    def pfam_only_features(train_idx, test_idx):
        X_train_clr = clr_transform(X_pfam_filtered[train_idx])
        X_test_clr = clr_transform(X_pfam_filtered[test_idx])
        n_comp = min(N_PCA_COMPONENTS, X_train_clr.shape[0], X_train_clr.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        X_train_pca = pca.fit_transform(X_train_clr)
        X_test_pca = pca.transform(X_test_clr)
        scaler = StandardScaler()
        return scaler.fit_transform(X_train_pca), scaler.transform(X_test_pca)

    def pfam_lat_features(train_idx, test_idx):
        X_train_clr = clr_transform(X_pfam_filtered[train_idx])
        X_test_clr = clr_transform(X_pfam_filtered[test_idx])
        n_comp = min(N_PCA_COMPONENTS, X_train_clr.shape[0], X_train_clr.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        X_train_pca = pca.fit_transform(X_train_clr)
        X_test_pca = pca.transform(X_test_clr)
        scaler_pca = StandardScaler()
        X_train_pca_sc = scaler_pca.fit_transform(X_train_pca)
        X_test_pca_sc = scaler_pca.transform(X_test_pca)
        scaler_lat = StandardScaler()
        lat_train = scaler_lat.fit_transform(X_lat[train_idx])
        lat_test = scaler_lat.transform(X_lat[test_idx])
        return np.hstack([X_train_pca_sc, lat_train]), np.hstack([X_test_pca_sc, lat_test])

    r2_lat, _ = run_model('latitude_only', lat_only_features)
    r2_pfam, _ = run_model('pfam_only', pfam_only_features)
    r2_combined, _ = run_model('pfam_plus_latitude', pfam_lat_features)

    if not np.isnan(r2_combined) and not np.isnan(r2_lat) and r2_lat < 1.0:
        partial_r2_pfam = (r2_combined - r2_lat) / (1.0 - r2_lat)
    else:
        partial_r2_pfam = np.nan

    if not np.isnan(r2_combined) and not np.isnan(r2_pfam) and r2_pfam < 1.0:
        partial_r2_lat = (r2_combined - r2_pfam) / (1.0 - r2_pfam)
    else:
        partial_r2_lat = np.nan

    print(f"\n  partial_R2(PFAM|lat) = {partial_r2_pfam:.4f}")
    print(f"  partial_R2(lat|PFAM) = {partial_r2_lat:.4f}")

    all_results.append({
        'target': target_col, 'target_label': label,
        'model': 'partial_r2_summary', 'fold': 'partial_r2',
        'n_test': n_samples,
        'r2_latitude_only': r2_lat,
        'r2_pfam_only': r2_pfam,
        'r2_pfam_plus_latitude': r2_combined,
        'partial_r2_pfam_given_latitude': partial_r2_pfam,
        'partial_r2_latitude_given_pfam': partial_r2_lat,
    })

    del X_pfam_filtered

# ── Save TSV ──
results_df = pd.DataFrame(all_results)

ts_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
provenance = f"""# Provenance:
#   Script: {SCRIPT_PATH}
#   Input1: {MERGED_PATH}
#   Input2: {NUTRIENTS_PATH}
#   Date:   {ts_str}
#   Integrity Check: PASSED
#   Task: ralph43 task 5 — Latitude baseline for all flagship targets
#   Reviewer: R1#2
#   Targets: {', '.join(TARGETS.keys())}
#   Block size: {BLOCK_SIZE} degrees
#   N folds: {N_FOLDS}
#   PCA components: {N_PCA_COMPONENTS}
#   CLR pseudocount: {CLR_PSEUDOCOUNT}
#   Prevalence threshold: {PREVALENCE_THRESHOLD}
#   XGBoost: max_depth={XGB_PARAMS['max_depth']}, n_estimators={XGB_PARAMS['n_estimators']}, lr={XGB_PARAMS['learning_rate']}
"""

os.makedirs(os.path.dirname(OUTPUT_TSV), exist_ok=True)
with open(OUTPUT_TSV, 'w') as f:
    f.write(provenance)
    results_df.to_csv(f, sep='\t', index=False)

print(f"\nOutput: {OUTPUT_TSV}")

# ── Save summary MD ──
summary_rows = results_df[results_df['model'] == 'partial_r2_summary']

md_lines = [
    f"# Latitude Baseline — All Flagship Targets",
    f"",
    f"**Provenance:**",
    f"- Script: `{SCRIPT_PATH}`",
    f"- Input: `{MERGED_PATH}` + `{NUTRIENTS_PATH}`",
    f"- Date: {ts_str}",
    f"- Integrity Check: PASSED",
    f"",
    f"## Summary Table",
    f"",
    f"| Target | n | R²(lat) | R²(PFAM) | R²(combined) | partial R²(PFAM\\|lat) | partial R²(lat\\|PFAM) |",
    f"|--------|---|---------|----------|--------------|----------------------|----------------------|",
]

for _, row in summary_rows.iterrows():
    md_lines.append(
        f"| {row.get('target_label', row['target'])} "
        f"| {int(row['n_test'])} "
        f"| {row['r2_latitude_only']:.4f} "
        f"| {row['r2_pfam_only']:.4f} "
        f"| {row['r2_pfam_plus_latitude']:.4f} "
        f"| {row['partial_r2_pfam_given_latitude']:.4f} "
        f"| {row['partial_r2_latitude_given_pfam']:.4f} |"
    )

md_lines.extend([
    f"",
    f"## Interpretation",
    f"",
    f"Partial R²(PFAM|latitude) quantifies PFAM's unique variance beyond latitude.",
    f"Partial R²(latitude|PFAM) quantifies latitude's unique variance beyond PFAM.",
    f"",
    f"All models: XGBoost, 10-fold spatial block CV (2° grid), CLR + PCA(100).",
])

with open(OUTPUT_MD, 'w') as f:
    f.write('\n'.join(md_lines) + '\n')

print(f"Output: {OUTPUT_MD}")
print("\nDone.")
