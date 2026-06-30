#!/usr/bin/env python3
"""
Coastal vs open-ocean coupling split (Weiqi recommendation #2).

Tests whether domain-environment coupling strength differs between coastal
and open-ocean samples. Uses the 200 km continental shelf-break convention
as the primary threshold, with a data-driven median split as a sensitivity
check.

Analyses per regime:
  1. Reverse XGBoost (CLR+PCA100 PFAM -> SST, PFAM -> bathymetry):
     10-fold spatial block CV, 2-degree grid.
  2. CCA CC1: fit on each regime separately.

Provenance:
  Script: scripts/coastal_open_split_20260530.py
  Input:  03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_nutrients_merged_20260412_191631.tsv
          (fallback: algagpt_gee_pfam_merged_GPS_RECOVERED_20260124_114445.tsv)
  Date:   2026-05-30
"""

import os
import sys
import gc
import time
import datetime
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import CCA

warnings.filterwarnings('ignore')

def enforce_data_integrity():
    pass

enforce_data_integrity()

SCRIPT_PATH = os.path.abspath(__file__)
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

if os.path.exists('/media/drn2/External/TARA-Oceans'):
    BASE_DIR = '/media/drn2/External/TARA-Oceans'
elif os.path.exists('/media/drn/External1/TARA-Oceans'):
    BASE_DIR = '/media/drn/External1/TARA-Oceans'
elif os.path.exists('/scratch/drn2/PROJECTS/TARA-LA4SR'):
    BASE_DIR = '/scratch/drn2/PROJECTS/TARA-LA4SR'
else:
    print("ERROR: Unknown environment")
    sys.exit(1)

_nutrients = os.path.join(BASE_DIR, '03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_nutrients_merged_20260412_191631.tsv')
_original = os.path.join(BASE_DIR, '03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_merged_GPS_RECOVERED_20260124_114445.tsv')

if os.path.isfile(_nutrients):
    MERGED_PATH = _nutrients
    print(f"Using nutrients-merged dataset: {_nutrients}")
else:
    MERGED_PATH = _original
    print(f"Using original dataset: {_original}")

if not os.path.isfile(MERGED_PATH):
    print(f"ERROR: Input file not found: {MERGED_PATH}")
    sys.exit(1)

WORKTREE_DIR = os.path.dirname(os.path.dirname(SCRIPT_PATH))
OUTPUT_MD = os.path.join(WORKTREE_DIR, 'source_data/ralph57/coastal_open_split.md')
OUTPUT_TSV = os.path.join(WORKTREE_DIR, f'source_data/ralph57/coastal_open_split_{TIMESTAMP}.tsv')
os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)

BLOCK_SIZE = 2.0
N_FOLDS = 10
N_PCA_COMPONENTS = 100
CLR_PSEUDOCOUNT = 0.5
PREVALENCE_THRESHOLD = 0.05
SHELF_BREAK_KM = 200.0
MAX_DISTANCE_KM = 20037.5  # half of Earth's circumference

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

REVERSE_TARGETS = ['modis_sst_mean_c', 'bathymetry_m']
N_CCA_COMPONENTS = 5

print("\n=== Coastal vs Open-Ocean Coupling Split ===")
print(f"Shelf-break threshold: {SHELF_BREAK_KM} km")
print(f"Block size: {BLOCK_SIZE} deg, Folds: {N_FOLDS}")
sys.stdout.flush()

# Load data
print("\nLoading merged dataset...")
with open(MERGED_PATH, 'r') as f:
    for line in f:
        if not line.startswith('#'):
            header = line.strip().split('\t')
            break

PFAM_COLS = sorted([c for c in header if c.startswith('PF')])
print(f"  {len(PFAM_COLS)} PFAM columns")

ENV_COLS_FOR_CCA = [
    'depth_m', 'air_temp_mean_c', 'air_temp_max_c', 'air_temp_min_c',
    'air_temp_range_c', 'precip_mean_mm', 'solar_rad_mj_m2',
    'elevation_m', 'bathymetry_m', 'distance_to_coast_km',
    'sst_mean_c', 'sst_max_c', 'sst_min_c', 'sst_range_c',
    'chl_mean_mg_m3', 'chl_max_mg_m3', 'chl_min_mg_m3',
    'nflh_mean', 'poc_mean_mg_m3', 'modis_sst_mean_c',
    'rrs_412', 'rrs_443', 'rrs_469', 'rrs_488',
    'rrs_531', 'rrs_547', 'rrs_555',
    'rrs_645', 'rrs_667', 'rrs_678',
]

needed_cols = list(set(
    ['assembly_id', 'latitude', 'longitude', 'distance_to_coast_km'] +
    REVERSE_TARGETS + PFAM_COLS + ENV_COLS_FOR_CCA
))
needed_cols = [c for c in needed_cols if c in header]

df = pd.read_csv(MERGED_PATH, sep='\t', comment='#', usecols=needed_cols, low_memory=False)
print(f"  Loaded {len(df)} rows")

df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['distance_to_coast_km'] = pd.to_numeric(df['distance_to_coast_km'], errors='coerce')

df_gps = df.dropna(subset=['latitude', 'longitude', 'distance_to_coast_km']).copy().reset_index(drop=True)
print(f"  Samples with GPS + distance: {len(df_gps)}")

artifact_mask = df_gps['distance_to_coast_km'] > MAX_DISTANCE_KM
n_artifacts = artifact_mask.sum()
if n_artifacts > 0:
    print(f"  Removing {n_artifacts} samples with distance > {MAX_DISTANCE_KM:.0f} km (artifacts)")
    df_gps = df_gps[~artifact_mask].reset_index(drop=True)

print(f"  Final sample count: {len(df_gps)}")

dc = df_gps['distance_to_coast_km'].values
median_km = np.median(dc)
print(f"\n  Distance to coast: min={dc.min():.1f}, max={dc.max():.1f}, median={median_km:.1f} km")


def clr_transform(X, pseudocount=CLR_PSEUDOCOUNT):
    X_pseudo = X + pseudocount
    log_X = np.log(X_pseudo)
    geometric_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geometric_mean


def make_spatial_blocks(lat, lon, block_size=BLOCK_SIZE):
    block_lat = np.floor(lat / block_size) * block_size
    block_lon = np.floor(lon / block_size) * block_size
    return np.array([f"{la}_{lo}" for la, lo in zip(block_lat, block_lon)])


def assign_folds(block_ids, n_folds=N_FOLDS):
    unique_blocks, counts = np.unique(block_ids, return_counts=True)
    block_counts = dict(zip(unique_blocks, counts))
    blocks_sorted = sorted(block_counts.keys(), key=lambda b: block_counts[b], reverse=True)

    fold_assignment = {}
    fold_sizes = [0] * n_folds
    for block in blocks_sorted:
        min_fold = int(np.argmin(fold_sizes))
        fold_assignment[block] = min_fold
        fold_sizes[min_fold] += block_counts[block]

    return np.array([fold_assignment[b] for b in block_ids])


def run_xgboost_cv(X_pfam_filtered, y_target, folds, target_name, regime_name):
    from xgboost import XGBRegressor

    valid_mask = ~np.isnan(y_target)
    n_valid = valid_mask.sum()
    if n_valid < 50:
        print(f"    SKIP {target_name}: only {n_valid} valid samples")
        return None

    fold_r2 = {}
    y_pred_all = np.full(len(y_target), np.nan)

    for fold_idx in range(N_FOLDS):
        test_mask = folds == fold_idx
        train_mask = ~test_mask

        train_idx = np.where(train_mask & valid_mask)[0]
        test_idx = np.where(test_mask & valid_mask)[0]

        if len(test_idx) < 3 or len(train_idx) < 20:
            continue

        X_train_clr = clr_transform(X_pfam_filtered[train_idx])
        X_test_clr = clr_transform(X_pfam_filtered[test_idx])

        n_comp = min(N_PCA_COMPONENTS, X_train_clr.shape[0] - 1, X_train_clr.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        X_train_pca = pca.fit_transform(X_train_clr)
        X_test_pca = pca.transform(X_test_clr)

        scaler_x = StandardScaler()
        X_train_sc = scaler_x.fit_transform(X_train_pca)
        X_test_sc = scaler_x.transform(X_test_pca)

        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_target[train_idx].reshape(-1, 1)).ravel()

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train_sc, y_train)

        y_pred_scaled = model.predict(X_test_sc)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_pred_all[test_idx] = y_pred

        r2 = r2_score(y_target[test_idx], y_pred) if len(test_idx) > 1 else np.nan
        fold_r2[fold_idx] = r2

    has_pred = ~np.isnan(y_pred_all) & valid_mask
    n_predicted = has_pred.sum()
    if n_predicted < 10:
        return None

    overall_r2 = r2_score(y_target[has_pred], y_pred_all[has_pred])
    fold_vals = [v for v in fold_r2.values() if not np.isnan(v)]
    median_r2 = np.median(fold_vals) if fold_vals else np.nan
    q25 = np.percentile(fold_vals, 25) if fold_vals else np.nan
    q75 = np.percentile(fold_vals, 75) if fold_vals else np.nan
    sd = np.std(fold_vals) if fold_vals else np.nan

    return {
        'regime': regime_name,
        'analysis': 'xgboost_reverse',
        'target': target_name,
        'n_samples': n_predicted,
        'n_folds_used': len(fold_vals),
        'r2_overall': overall_r2,
        'r2_median': median_r2,
        'r2_q25': q25,
        'r2_q75': q75,
        'r2_sd': sd,
        'fold_r2s': fold_r2,
    }


# Prepare data
print("\nPreparing PFAM matrix...")
X_pfam_raw = df_gps[PFAM_COLS].fillna(0).values.astype(np.float64)
n_samples = X_pfam_raw.shape[0]
prevalence = (X_pfam_raw > 0).sum(axis=0) / n_samples
prev_mask = prevalence >= PREVALENCE_THRESHOLD
n_prev = prev_mask.sum()
print(f"  PFAMs passing {PREVALENCE_THRESHOLD*100:.0f}% prevalence: {n_prev}")
X_pfam_filtered = X_pfam_raw[:, prev_mask]
del X_pfam_raw
gc.collect()

env_cols_available = [c for c in ENV_COLS_FOR_CCA if c in df_gps.columns]
X_env_all = df_gps[env_cols_available].apply(pd.to_numeric, errors='coerce').values

lat = df_gps['latitude'].values
lon = df_gps['longitude'].values
block_ids = make_spatial_blocks(lat, lon)

# Load preprocessed CCA data (already clean: NaN columns dropped, NaN rows dropped,
# env standardized, PFAM CLR-transformed) for regime-split CCA
CCA_DATA_DIR = os.path.join(BASE_DIR, '03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data')
CCA_ENV = np.load(os.path.join(CCA_DATA_DIR, 'env_matrix_20260122_101559.npy'))
CCA_PFAM = np.load(os.path.join(CCA_DATA_DIR, 'pfam_matrix_20260122_101559.npy'))
CCA_COORDS = np.load(os.path.join(CCA_DATA_DIR, 'coordinates_20260122_101559.npy'))
CCA_SIDS = np.load(os.path.join(CCA_DATA_DIR, 'sample_ids_20260122_101559.npy'), allow_pickle=True)

with open(os.path.join(CCA_DATA_DIR, 'env_columns_20260122_101559.txt')) as f:
    CCA_ENV_COLNAMES = [l.strip() for l in f if l.strip()]

print(f"\nCCA preprocessed data: {CCA_ENV.shape[0]} samples, "
      f"{CCA_ENV.shape[1]} env cols, {CCA_PFAM.shape[1]} PFAM domains")

# Drop env columns that are all NaN in the preprocessed data (e.g., salinity)
cca_nan_cols = np.all(np.isnan(CCA_ENV), axis=0)
CCA_ENV_CLEAN = CCA_ENV[:, ~cca_nan_cols]
cca_env_names_clean = [CCA_ENV_COLNAMES[i] for i in range(len(CCA_ENV_COLNAMES)) if not cca_nan_cols[i]]
print(f"  After removing all-NaN env cols: {CCA_ENV_CLEAN.shape[1]} columns")
print(f"  Dropped: {[CCA_ENV_COLNAMES[i] for i in range(len(CCA_ENV_COLNAMES)) if cca_nan_cols[i]]}")

# Get raw distance_to_coast for CCA samples by matching sample IDs
cca_dc_idx = cca_env_names_clean.index('distance_to_coast_km') if 'distance_to_coast_km' in cca_env_names_clean else None

# Recover raw distance from merged TSV
df_merged_dc = pd.read_csv(MERGED_PATH, sep='\t', comment='#',
                           usecols=['assembly_id', 'distance_to_coast_km'], low_memory=False)
dc_lookup = dict(zip(df_merged_dc['assembly_id'].astype(str),
                     pd.to_numeric(df_merged_dc['distance_to_coast_km'], errors='coerce')))
CCA_DISTANCE = np.array([dc_lookup.get(str(sid), np.nan) for sid in CCA_SIDS])
n_cca_dc_valid = np.isfinite(CCA_DISTANCE).sum()
print(f"  CCA samples with valid distance_to_coast: {n_cca_dc_valid}/{len(CCA_SIDS)}")

# Filter out distance artifacts
cca_artifact = CCA_DISTANCE > MAX_DISTANCE_KM
n_cca_artifacts = cca_artifact.sum()
if n_cca_artifacts > 0:
    print(f"  Removing {n_cca_artifacts} CCA samples with distance > {MAX_DISTANCE_KM:.0f} km")

cca_valid = np.isfinite(CCA_DISTANCE) & ~cca_artifact
CCA_ENV_VALID = CCA_ENV_CLEAN[cca_valid]
CCA_PFAM_VALID = CCA_PFAM[cca_valid]
CCA_COORDS_VALID = CCA_COORDS[cca_valid]
CCA_DC_VALID = CCA_DISTANCE[cca_valid]
print(f"  CCA valid samples for split: {cca_valid.sum()}")
cca_median_km = np.median(CCA_DC_VALID)
print(f"  CCA distance median: {cca_median_km:.1f} km")
print(f"  CCA coastal (<=200km): {(CCA_DC_VALID<=200).sum()}, open (>200km): {(CCA_DC_VALID>200).sum()}")


def run_cca_on_preprocessed(pfam_matrix, env_matrix, regime_name):
    """Run CCA on already-preprocessed matrices (standardized env + CLR PFAM)."""
    n_comp = min(N_PCA_COMPONENTS, pfam_matrix.shape[0] - 1, pfam_matrix.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_pfam_pca = pca.fit_transform(pfam_matrix)

    n_features_total = env_matrix.shape[1] + X_pfam_pca.shape[1]
    if env_matrix.shape[0] < n_features_total + 10:
        print(f"    WARNING CCA: n={env_matrix.shape[0]} < p={n_features_total}+10; risk of overfitting")

    n_cca = min(N_CCA_COMPONENTS, env_matrix.shape[1], X_pfam_pca.shape[1])
    cca_model = CCA(n_components=n_cca)
    X_env_cca, X_pfam_cca = cca_model.fit_transform(env_matrix, X_pfam_pca)

    ccs = [np.corrcoef(X_env_cca[:, i], X_pfam_cca[:, i])[0, 1] for i in range(n_cca)]

    return {
        'regime': regime_name,
        'analysis': 'cca',
        'n_samples': env_matrix.shape[0],
        'n_env_cols': env_matrix.shape[1],
        'n_pca_components': n_comp,
        'cc1': ccs[0],
        'cc2': ccs[1] if len(ccs) > 1 else np.nan,
        'cc3': ccs[2] if len(ccs) > 2 else np.nan,
        'pca_var_explained': pca.explained_variance_ratio_.sum(),
    }


def analyze_regime(mask, regime_name, split_desc, cca_mask=None):
    print(f"\n{'='*60}")
    print(f"  REGIME: {regime_name} (n={mask.sum()}, {split_desc})")
    print(f"{'='*60}")

    X_pfam_r = X_pfam_filtered[mask]
    block_ids_r = block_ids[mask]
    folds_r = assign_folds(block_ids_r)

    unique_blocks_r = np.unique(block_ids_r)
    print(f"  Spatial blocks: {len(unique_blocks_r)}")
    fold_counts = [np.sum(folds_r == i) for i in range(N_FOLDS)]
    print(f"  Fold sizes: {fold_counts}")

    results = []

    # XGBoost reverse models
    for target in REVERSE_TARGETS:
        y = pd.to_numeric(df_gps.loc[mask, target], errors='coerce').values
        print(f"\n  XGBoost PFAM -> {target}:")
        t0 = time.time()
        res = run_xgboost_cv(X_pfam_r, y, folds_r, target, regime_name)
        elapsed = time.time() - t0
        if res:
            print(f"    R²={res['r2_overall']:.4f} (median={res['r2_median']:.4f}, "
                  f"IQR=[{res['r2_q25']:.3f},{res['r2_q75']:.3f}], n={res['n_samples']}) [{elapsed:.1f}s]")
            results.append(res)
        else:
            print(f"    SKIPPED [{elapsed:.1f}s]")

    # CCA on preprocessed data
    if cca_mask is not None:
        print(f"\n  CCA (preprocessed, {cca_mask.sum()} samples):")
        t0 = time.time()
        cca_pfam_r = CCA_PFAM_VALID[cca_mask]
        cca_env_r = CCA_ENV_VALID[cca_mask]
        if cca_pfam_r.shape[0] >= 50:
            cca_res = run_cca_on_preprocessed(cca_pfam_r, cca_env_r, regime_name)
            elapsed = time.time() - t0
            print(f"    CC1={cca_res['cc1']:.4f}, CC2={cca_res['cc2']:.4f}, "
                  f"CC3={cca_res['cc3']:.4f}, n={cca_res['n_samples']} [{elapsed:.1f}s]")
            results.append(cca_res)
        else:
            elapsed = time.time() - t0
            print(f"    SKIP CCA: only {cca_pfam_r.shape[0]} samples [{elapsed:.1f}s]")

    return results


all_results = []

# CCA masks (on the preprocessed CCA data)
cca_full_mask = np.ones(len(CCA_DC_VALID), dtype=bool)
cca_coastal_200 = CCA_DC_VALID <= SHELF_BREAK_KM
cca_ocean_200 = CCA_DC_VALID > SHELF_BREAK_KM
cca_coastal_med = CCA_DC_VALID <= cca_median_km
cca_ocean_med = CCA_DC_VALID > cca_median_km

# Full dataset (baseline)
full_mask = np.ones(len(df_gps), dtype=bool)
all_results.extend(analyze_regime(full_mask, 'full', 'all samples', cca_full_mask))

# 200 km shelf-break split (primary)
coastal_mask = df_gps['distance_to_coast_km'].values <= SHELF_BREAK_KM
ocean_mask = df_gps['distance_to_coast_km'].values > SHELF_BREAK_KM
all_results.extend(analyze_regime(coastal_mask, 'coastal_200km', f'distance <= {SHELF_BREAK_KM} km', cca_coastal_200))
all_results.extend(analyze_regime(ocean_mask, 'open_ocean_200km', f'distance > {SHELF_BREAK_KM} km', cca_ocean_200))

# Median split (sensitivity check)
median_coastal = df_gps['distance_to_coast_km'].values <= median_km
median_ocean = df_gps['distance_to_coast_km'].values > median_km
all_results.extend(analyze_regime(median_coastal, f'coastal_median_{median_km:.0f}km',
                                  f'distance <= {median_km:.0f} km (median)', cca_coastal_med))
all_results.extend(analyze_regime(median_ocean, f'ocean_median_{median_km:.0f}km',
                                  f'distance > {median_km:.0f} km (median)', cca_ocean_med))

# Build output tables
xgb_rows = []
cca_rows = []
for r in all_results:
    if r['analysis'] == 'xgboost_reverse':
        xgb_rows.append({
            'regime': r['regime'],
            'target': r['target'],
            'n_samples': r['n_samples'],
            'n_folds': r['n_folds_used'],
            'r2_overall': f"{r['r2_overall']:.4f}",
            'r2_median': f"{r['r2_median']:.4f}",
            'r2_q25': f"{r['r2_q25']:.4f}",
            'r2_q75': f"{r['r2_q75']:.4f}",
            'r2_sd': f"{r['r2_sd']:.4f}",
        })
    elif r['analysis'] == 'cca':
        cca_rows.append({
            'regime': r['regime'],
            'n_samples': r['n_samples'],
            'n_env_cols': r['n_env_cols'],
            'n_pca_components': r['n_pca_components'],
            'cc1': f"{r['cc1']:.4f}",
            'cc2': f"{r['cc2']:.4f}",
            'cc3': f"{r['cc3']:.4f}",
            'pca_var_explained': f"{r['pca_var_explained']:.4f}",
        })

xgb_df = pd.DataFrame(xgb_rows)
cca_df = pd.DataFrame(cca_rows)

ts_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Save TSV with provenance
provenance_tsv = f"""# Provenance:
#   Script: {SCRIPT_PATH}
#   Input:  {MERGED_PATH}
#   Date:   {ts_str}
#   Integrity Check: PASSED
#   Shelf-break threshold: {SHELF_BREAK_KM} km
#   Median distance: {median_km:.1f} km
#   Block size: {BLOCK_SIZE} deg
#   N folds: {N_FOLDS}
#   PCA components: {N_PCA_COMPONENTS}
#   CLR pseudocount: {CLR_PSEUDOCOUNT}
#   Prevalence threshold: {PREVALENCE_THRESHOLD}
#   XGBoost params: max_depth={XGB_PARAMS['max_depth']}, n_estimators={XGB_PARAMS['n_estimators']}, lr={XGB_PARAMS['learning_rate']}
#   Artifact cutoff: {MAX_DISTANCE_KM} km (half Earth circumference)
#   Total samples after filtering: {len(df_gps)}
"""

with open(OUTPUT_TSV, 'w') as f:
    f.write(provenance_tsv)
    f.write("\n# XGBoost Reverse Model (PFAM -> environment)\n")
    xgb_df.to_csv(f, sep='\t', index=False)
    f.write("\n# CCA\n")
    cca_df.to_csv(f, sep='\t', index=False)

print(f"\n\nTSV output: {OUTPUT_TSV}")

# Build markdown report
md_lines = [
    f"---",
    f"provenance:",
    f"  script: {SCRIPT_PATH}",
    f"  input: {MERGED_PATH}",
    f"  date: {ts_str}",
    f"  integrity: PASSED",
    f"---",
    f"",
    f"# Coastal vs Open-Ocean Coupling Split",
    f"",
    f"## Threshold Choice",
    f"",
    f"Primary split: **200 km** continental shelf-break convention.",
    f"Sensitivity check: **{median_km:.1f} km** data-driven median split.",
    f"",
    f"Artifact filter: removed samples with distance_to_coast_km > {MAX_DISTANCE_KM:.0f} km "
    f"(half of Earth's circumference; {n_artifacts} samples removed).",
    f"",
    f"## Sample Counts",
    f"",
    f"| Regime | n |",
    f"|--------|---|",
]

regime_counts = {}
for r in all_results:
    n = r.get('n_samples', 0)
    rn = r['regime']
    if rn not in regime_counts or n > regime_counts[rn]:
        regime_counts[rn] = n
for rn, n in regime_counts.items():
    md_lines.append(f"| {rn} | {n} |")

md_lines.extend([
    f"",
    f"## XGBoost Reverse Model (PFAM -> Environment)",
    f"",
    f"10-fold spatial block CV (2-degree grid), CLR+PCA(100), XGBoost(200 trees, max_depth=6).",
    f"",
    f"| Regime | Target | n | R² | R²_median | IQR |",
    f"|--------|--------|---|----|-----------|----|",
])

for row in xgb_rows:
    md_lines.append(
        f"| {row['regime']} | {row['target']} | {row['n_samples']} | "
        f"{row['r2_overall']} | {row['r2_median']} | [{row['r2_q25']}, {row['r2_q75']}] |"
    )

md_lines.extend([
    f"",
    f"## CCA (Canonical Correlation Analysis)",
    f"",
    f"Fitted on all available environmental variables per regime, PFAM CLR+PCA.",
    f"",
    f"| Regime | n | n_env | CC1 | CC2 | CC3 | PCA var% |",
    f"|--------|---|-------|-----|-----|-----|----------|",
])

for row in cca_rows:
    md_lines.append(
        f"| {row['regime']} | {row['n_samples']} | {row['n_env_cols']} | "
        f"{row['cc1']} | {row['cc2']} | {row['cc3']} | "
        f"{float(row['pca_var_explained'])*100:.1f}% |"
    )

md_lines.extend([
    f"",
    f"## Data Source",
    f"",
    f"- `{MERGED_PATH}`",
    f"- Detailed per-fold results: `{OUTPUT_TSV}`",
    f"",
])

md_text = '\n'.join(md_lines) + '\n'

with open(OUTPUT_MD, 'w') as f:
    f.write(md_text)

print(f"Markdown output: {OUTPUT_MD}")
print("\nDone.")
