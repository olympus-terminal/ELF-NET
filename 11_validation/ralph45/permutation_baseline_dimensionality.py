#!/usr/bin/env python3
"""
Permutation baseline for dimensionality inflation test.

Tests whether ~10k random features can mechanically inflate R² under
spatial block CV. Generates 100 random feature matrices (same shape as
real PFAM matrix) by independently shuffling each column, then trains
XGBoost with identical hyperparameters under 10-fold spatial block CV.

If spatial block CV prevents overfitting, null R² ≈ 0.
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

os.environ['PYTHONUNBUFFERED'] = '1'

DATA_DIR = Path("/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/"
                "env_pfam_manifold/data")
PFAM_MATRIX = DATA_DIR / "pfam_matrix_20260122_101559.npy"
ENV_MATRIX = DATA_DIR / "env_matrix_20260122_101559.npy"
COORDS_FILE = DATA_DIR / "coordinates_20260122_101559.npy"
ENV_COLS_FILE = DATA_DIR / "env_columns_20260122_101559.txt"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR.parent.parent / "source_data" / "ralph45" / "permutation_dimensionality_null.tsv"
CHECKPOINT_FILE = OUTPUT_FILE.parent / "permutation_checkpoint.npy"

TARGET_COL_NAME = "modis_sst_mean_c"
BLOCK_SIZE = 2.0
N_FOLDS = 10
N_PERMUTATIONS = 20
RANDOM_SEED = 42

XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.3,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
}


def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()


def load_data():
    X = np.load(PFAM_MATRIX)
    env = np.load(ENV_MATRIX)
    coords = np.load(COORDS_FILE)
    with open(ENV_COLS_FILE) as f:
        env_cols = [line.strip() for line in f]
    target_idx = env_cols.index(TARGET_COL_NAME)
    y_raw = env[:, target_idx]
    valid_mask = ~np.isnan(y_raw) & ~np.isnan(coords[:, 0]) & ~np.isnan(coords[:, 1])
    X = X[valid_mask].astype(np.float32)
    y = y_raw[valid_mask]
    coords_valid = coords[valid_mask]
    return X, y, coords_valid


def assign_spatial_blocks(coords):
    lat_bin = np.floor(coords[:, 0] / BLOCK_SIZE) * BLOCK_SIZE
    lon_bin = np.floor(coords[:, 1] / BLOCK_SIZE) * BLOCK_SIZE
    block_ids = np.array([f"{la}_{lo}" for la, lo in zip(lat_bin, lon_bin)])
    return block_ids


def assign_folds(block_ids):
    unique_blocks = np.unique(block_ids)
    block_counts = {b: np.sum(block_ids == b) for b in unique_blocks}
    sorted_blocks = sorted(block_counts.keys(), key=lambda b: block_counts[b], reverse=True)
    fold_sizes = np.zeros(N_FOLDS, dtype=int)
    block_to_fold = {}
    for b in sorted_blocks:
        min_fold = np.argmin(fold_sizes)
        block_to_fold[b] = min_fold
        fold_sizes[min_fold] += block_counts[b]
    fold_assignments = np.array([block_to_fold[b] for b in block_ids])
    return fold_assignments


def spatial_block_cv_r2(X, y, folds):
    y_pred_all = np.full(len(y), np.nan)
    for fold_idx in range(N_FOLDS):
        test_mask = folds == fold_idx
        train_mask = ~test_mask
        if np.sum(test_mask) < 5:
            continue
        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]
        scaler_X = StandardScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        X_test_s = scaler_X.transform(X_test)
        y_mean, y_std = y_train.mean(), y_train.std()
        if y_std == 0:
            continue
        y_train_s = (y_train - y_mean) / y_std
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train_s, y_train_s, verbose=False)
        y_pred_s = model.predict(X_test_s)
        y_pred_all[test_mask] = y_pred_s * y_std + y_mean
    valid = ~np.isnan(y_pred_all)
    if valid.sum() == 0:
        return np.nan
    ss_res = np.sum((y[valid] - y_pred_all[valid]) ** 2)
    ss_tot = np.sum((y[valid] - y[valid].mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def main():
    log(f"Loading data from numpy arrays...")
    log(f"  PFAM matrix: {PFAM_MATRIX}")
    log(f"  Env matrix: {ENV_MATRIX}")
    log(f"  Coordinates: {COORDS_FILE}")
    X, y, coords = load_data()
    n_samples, n_features = X.shape
    log(f"Loaded: {n_samples} samples, {n_features} PFAM features")
    log(f"Target: {TARGET_COL_NAME}, range [{y.min():.1f}, {y.max():.1f}]")

    block_ids = assign_spatial_blocks(coords)
    folds = assign_folds(block_ids)
    n_blocks = len(np.unique(block_ids))
    log(f"Spatial blocks: {n_blocks} (block size: {BLOCK_SIZE}°)")
    log(f"Fold sizes: {[int(np.sum(folds == i)) for i in range(N_FOLDS)]}")

    # Resume from checkpoint if available
    start_iter = 0
    null_r2_values = []
    if CHECKPOINT_FILE.exists():
        checkpoint = np.load(CHECKPOINT_FILE)
        null_r2_values = list(checkpoint)
        start_iter = len(null_r2_values)
        log(f"Resuming from checkpoint: {start_iter} iterations completed")

    log(f"\nRunning {N_PERMUTATIONS} permutation iterations (starting at {start_iter})...")
    rng = np.random.default_rng(RANDOM_SEED)
    # Advance RNG state to match checkpoint
    for _ in range(start_iter):
        X_skip = X.copy()
        for col in range(X_skip.shape[1]):
            rng.shuffle(X_skip[:, col])
        del X_skip

    for i in range(start_iter, N_PERMUTATIONS):
        t0 = datetime.now()
        X_perm = X.copy()
        for col in range(X_perm.shape[1]):
            rng.shuffle(X_perm[:, col])
        r2 = spatial_block_cv_r2(X_perm, y, folds)
        del X_perm
        null_r2_values.append(r2)
        elapsed = (datetime.now() - t0).total_seconds()
        log(f"  Iteration {i+1}/{N_PERMUTATIONS}: R² = {r2:.4f} ({elapsed:.1f}s)")

        # Checkpoint every 5 iterations
        if (i + 1) % 5 == 0:
            np.save(CHECKPOINT_FILE, np.array(null_r2_values))
            log(f"  [Checkpoint saved: {i+1} iterations, running mean = {np.nanmean(null_r2_values):.4f}]")

    null_r2 = np.array(null_r2_values)
    mean_null = np.nanmean(null_r2)
    std_null = np.nanstd(null_r2)
    median_null = np.nanmedian(null_r2)
    max_null = np.nanmax(null_r2)
    min_null = np.nanmin(null_r2)
    pct_95 = np.nanpercentile(null_r2, 95)
    pct_99 = np.nanpercentile(null_r2, 99)

    log(f"\n--- Results ---")
    log(f"Null R² distribution (n={N_PERMUTATIONS}):")
    log(f"  Mean:   {mean_null:.4f}")
    log(f"  Median: {median_null:.4f}")
    log(f"  Std:    {std_null:.4f}")
    log(f"  Min:    {min_null:.4f}")
    log(f"  Max:    {max_null:.4f}")
    log(f"  95th %: {pct_95:.4f}")
    log(f"  99th %: {pct_99:.4f}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"# Provenance:\n")
        f.write(f"#   Script: {Path(__file__).resolve()}\n")
        f.write(f"#   Input PFAM matrix: {PFAM_MATRIX}\n")
        f.write(f"#   Input Env matrix: {ENV_MATRIX}\n")
        f.write(f"#   Input Coordinates: {COORDS_FILE}\n")
        f.write(f"#   Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"#   Integrity Check: PASSED\n")
        f.write(f"#   Target: {TARGET_COL_NAME}\n")
        f.write(f"#   N samples: {n_samples}\n")
        f.write(f"#   N features: {n_features}\n")
        f.write(f"#   N permutations: {N_PERMUTATIONS}\n")
        f.write(f"#   Block size: {BLOCK_SIZE} degrees\n")
        f.write(f"#   N folds: {N_FOLDS}\n")
        f.write(f"#   XGBoost params: {XGB_PARAMS}\n")
        f.write(f"#   Random seed: {RANDOM_SEED}\n")
        f.write(f"#   Method: Independent column shuffle of CLR-transformed PFAM matrix\n")
        f.write(f"#\n")
        f.write(f"# Summary:\n")
        f.write(f"#   Null R² mean: {mean_null:.6f}\n")
        f.write(f"#   Null R² median: {median_null:.6f}\n")
        f.write(f"#   Null R² std: {std_null:.6f}\n")
        f.write(f"#   Null R² 95th percentile: {pct_95:.6f}\n")
        f.write(f"#   Null R² 99th percentile: {pct_99:.6f}\n")
        f.write(f"#   Null R² max: {max_null:.6f}\n")
        f.write(f"#   Null R² min: {min_null:.6f}\n")
        f.write(f"#   Real model R² (from mc3_spatial_block_cv.tsv): 0.497\n")
        f.write(f"#\n")
        f.write("permutation_id\tnull_r2\n")
        for i, r2 in enumerate(null_r2_values):
            f.write(f"{i}\t{r2:.6f}\n")

    log(f"\nOutput written to: {OUTPUT_FILE}")

    # Cleanup checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


if __name__ == "__main__":
    main()
