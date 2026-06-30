#!/usr/bin/env python3
"""
XGBoost productivity prediction: POC, chl-a, NFLH from PFAM domain profiles
under spatial block cross-validation.

Three configurations per target:
  1. Domain-only: CLR-transformed PFAM counts
  2. Environment-only: non-productivity GEE variables
  3. Combined: PFAM + environment

Provenance:
  Script: scripts/poc_productivity_proof_of_concept_20260516_203400.py
  Input:  /media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/
          algagpt_gee_pfam_merged_GPS_RECOVERED_20260124_114445.tsv
  Date:   2026-05-16
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
from math import floor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_PATH = os.path.abspath(__file__)
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

DATA_PATH = "/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_merged_GPS_RECOVERED_20260124_114445.tsv"

BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_PATH))
OUTPUT_PATH = os.path.join(BASE_DIR, "source_data", "ralph54", f"productivity_results_{TIMESTAMP}.tsv")


def enforce_data_integrity():
    if not os.path.isfile(DATA_PATH):
        raise RuntimeError(f"Data file not found: {DATA_PATH}")
    sz = os.path.getsize(DATA_PATH)
    if sz < 1000:
        raise RuntimeError(f"Data file suspiciously small ({sz} bytes): {DATA_PATH}")
    print(f"[integrity] Verified data file: {DATA_PATH} ({sz / 1e6:.1f} MB)")


enforce_data_integrity()

# --- Hyperparameters (match existing manuscript models) ---
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 4,
    "verbosity": 0,
}

BLOCK_SIZE = 2.0
N_FOLDS = 10
CLR_PSEUDOCOUNT = 0.5
PREVALENCE_THRESHOLD = 0.05
PCA_COMPONENTS = 100

TARGETS = {
    "chl_mean_mg_m3": ["chl_mean_mg_m3", "chl_max_mg_m3", "chl_min_mg_m3", "nflh_mean"],
    "poc_mean_mg_m3": ["poc_mean_mg_m3"],
    "nflh_mean": ["nflh_mean", "chl_mean_mg_m3", "chl_max_mg_m3", "chl_min_mg_m3"],
}

METADATA_COLS = {
    "assembly_id", "matched_to", "matched_sample", "latitude", "longitude",
    "dataset", "depth_m", "collection_date", "species", "habitat",
    "gps_source", "gps_confidence", "landcover_class",
}

PRODUCTIVITY_COLS = {"chl_mean_mg_m3", "chl_max_mg_m3", "chl_min_mg_m3", "nflh_mean", "poc_mean_mg_m3"}


def clr_transform(X, pseudocount=CLR_PSEUDOCOUNT):
    X_pseudo = X + pseudocount
    log_X = np.log(X_pseudo)
    geometric_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geometric_mean


def assign_spatial_blocks(lat, lon, block_size=BLOCK_SIZE):
    block_lat = np.floor(lat / block_size) * block_size
    block_lon = np.floor(lon / block_size) * block_size
    return [f"{la:.0f}_{lo:.0f}" for la, lo in zip(block_lat, block_lon)]


def greedy_balanced_folds(block_ids, n_folds=N_FOLDS):
    block_counts = pd.Series(block_ids).value_counts()
    sorted_blocks = block_counts.sort_values(ascending=False).index.tolist()
    fold_sizes = [0] * n_folds
    block_to_fold = {}
    for block in sorted_blocks:
        min_fold = int(np.argmin(fold_sizes))
        block_to_fold[block] = min_fold
        fold_sizes[min_fold] += block_counts[block]
    return block_to_fold


print("Loading dataset...")
df = pd.read_csv(DATA_PATH, sep="\t", comment="#", low_memory=False)
print(f"  Loaded {len(df)} samples, {len(df.columns)} columns")

pfam_cols = sorted([c for c in df.columns if c.startswith("PF")])
print(f"  PFAM domain columns: {len(pfam_cols)}")

env_cols = [
    c for c in df.columns
    if c not in METADATA_COLS and c not in PRODUCTIVITY_COLS and not c.startswith("PF")
]
print(f"  Environment columns: {len(env_cols)}: {env_cols}")

# --- Filter to samples with GPS + all targets ---
required = ["latitude", "longitude", "chl_mean_mg_m3", "poc_mean_mg_m3", "nflh_mean"]
mask = df[required].notna().all(axis=1)
df_work = df.loc[mask].copy()
print(f"  Samples with GPS + all 3 targets: {len(df_work)}")

# --- Prevalence filter on PFAM columns ---
pfam_data = df_work[pfam_cols].fillna(0).values
nonzero_frac = (pfam_data > 0).mean(axis=0)
keep_mask = nonzero_frac >= PREVALENCE_THRESHOLD
pfam_cols_filtered = [pfam_cols[i] for i in range(len(pfam_cols)) if keep_mask[i]]
print(f"  PFAM after {PREVALENCE_THRESHOLD*100:.0f}% prevalence filter: {len(pfam_cols_filtered)}")
del pfam_data

# --- Spatial blocks ---
blocks = assign_spatial_blocks(df_work["latitude"].values, df_work["longitude"].values)
block_to_fold = greedy_balanced_folds(blocks, N_FOLDS)
df_work["fold"] = [block_to_fold[b] for b in blocks]

fold_counts = df_work["fold"].value_counts().sort_index()
print(f"  Fold distribution: {dict(fold_counts)}")
n_blocks = len(set(blocks))
print(f"  Total spatial blocks: {n_blocks}")

# --- Prepare feature matrices ---
X_pfam_raw = df_work[pfam_cols_filtered].fillna(0).values
X_pfam_clr = clr_transform(X_pfam_raw)
print(f"  CLR-transformed PFAM matrix: {X_pfam_clr.shape}")

X_env_all = df_work[env_cols].copy()
for col in env_cols:
    X_env_all[col] = pd.to_numeric(X_env_all[col], errors="coerce")
# Drop columns with >50% missing — XGBoost handles remaining NaN natively
env_missing = X_env_all.isna().mean()
env_cols_keep = [c for c in env_cols if env_missing[c] < 0.5]
env_cols_dropped = [c for c in env_cols if env_missing[c] >= 0.5]
if env_cols_dropped:
    print(f"  Dropped env columns (>50% missing): {env_cols_dropped}")
X_env_all = X_env_all[env_cols_keep]
env_cols = env_cols_keep
print(f"  Environment matrix: {X_env_all.shape}")

folds = df_work["fold"].values
results = []


def pca_reduce(X_train, X_test, n_components=PCA_COMPONENTS):
    n_comp = min(n_components, X_train.shape[1], X_train.shape[0])
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    pca = PCA(n_components=n_comp, random_state=42)
    return pca.fit_transform(X_train_s), pca.transform(X_test_s)


def run_cv(target_name, X, feature_type, folds_arr, apply_pca=False,
           X_extra=None):
    """Run spatial block CV. If X_extra is provided, PCA is applied only to X
    and X_extra is concatenated afterward (for the combined config)."""
    y = df_work[target_name].values
    fold_r2 = {}
    for fold_idx in range(N_FOLDS):
        train_mask = folds_arr != fold_idx
        test_mask = folds_arr == fold_idx
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        if len(y_test) < 5:
            continue
        if apply_pca:
            X_train, X_test = pca_reduce(X_train, X_test)
        if X_extra is not None:
            X_train = np.hstack([X_train, X_extra[train_mask]])
            X_test = np.hstack([X_test, X_extra[test_mask]])
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        fold_r2[fold_idx] = r2
        results.append({
            "target": target_name,
            "feature_set": feature_type,
            "fold": fold_idx,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "r2": r2,
        })
        print(f"    fold {fold_idx}: R² = {r2:.4f} (n_test={int(test_mask.sum())})")
    if fold_r2:
        vals = list(fold_r2.values())
        med = np.median(vals)
        q25 = np.percentile(vals, 25)
        q75 = np.percentile(vals, 75)
        results.append({
            "target": target_name,
            "feature_set": feature_type,
            "fold": "summary",
            "n_train": "",
            "n_test": len(y),
            "r2": np.mean(vals),
            "median_r2": med,
            "iqr_25": q25,
            "iqr_75": q75,
            "std_r2": np.std(vals),
        })
        print(f"  {target_name} [{feature_type}]: median R² = {med:.4f} (IQR {q25:.4f}–{q75:.4f})")


for target_name, exclude_cols in TARGETS.items():
    print(f"\n=== Target: {target_name} ===")
    print(f"  Excluding from env features: {exclude_cols}")

    # 1. Domain-only (PCA to reduce ~10K CLR features to 100 components per fold)
    print(f"  Config: domain-only ({X_pfam_clr.shape[1]} CLR features → PCA {PCA_COMPONENTS})")
    run_cv(target_name, X_pfam_clr, "domain_only", folds, apply_pca=True)

    # 2. Environment-only (exclude productivity-related)
    env_feature_cols = [c for c in env_cols if c not in exclude_cols]
    X_env_sub = X_env_all[env_feature_cols].values
    print(f"  Config: environment-only ({len(env_feature_cols)} features)")
    run_cv(target_name, X_env_sub, "environment_only", folds)

    # 3. Combined (PCA on PFAM portion, concatenate env)
    print(f"  Config: combined (PCA({PCA_COMPONENTS}) PFAM + {len(env_feature_cols)} env)")
    run_cv(target_name, X_pfam_clr, "combined", folds, apply_pca=True,
           X_extra=X_env_sub)

# --- Save results ---
results_df = pd.DataFrame(results)

provenance = f"""# Provenance:
#   Script: {SCRIPT_PATH}
#   Input:  {DATA_PATH}
#   Date:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#   Integrity Check: PASSED
#   Hyperparameters: n_estimators={XGB_PARAMS['n_estimators']}, max_depth={XGB_PARAMS['max_depth']}, lr={XGB_PARAMS['learning_rate']}, subsample={XGB_PARAMS['subsample']}, colsample={XGB_PARAMS['colsample_bytree']}
#   Spatial block CV: {BLOCK_SIZE}-degree grid, {N_FOLDS}-fold
#   CLR pseudocount: {CLR_PSEUDOCOUNT}
#   Prevalence threshold: {PREVALENCE_THRESHOLD}
#   PCA components: {PCA_COMPONENTS}
#   PFAM features (post-filter): {len(pfam_cols_filtered)}
#   Samples: {len(df_work)}
"""

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    f.write(provenance)
    results_df.to_csv(f, sep="\t", index=False)

print(f"\nResults written to: {OUTPUT_PATH}")
print(f"Total result rows: {len(results_df)}")

# Also write a simpler summary for quick reference
summary_path = os.path.join(BASE_DIR, "source_data", "ralph54", "productivity_results.tsv")
summary_rows = results_df[results_df["fold"] == "summary"].copy()
with open(summary_path, "w") as f:
    f.write(provenance)
    summary_rows.to_csv(f, sep="\t", index=False)
print(f"Summary written to: {summary_path}")
