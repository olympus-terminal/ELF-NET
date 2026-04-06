#!/usr/bin/env python3
"""
B4: Dark Proteome Environment Coupling (same framework as A8)

Provenance:
    Script: scripts/novel_families/B4_dark_env_coupling.py
    Generated: 2026-02-21
    Pipeline: Novel Domain Discovery — Track B

Purpose:
    Apply XGBoost forward model to dark proteome families using the same
    spatial block CV framework as A8b. Supports SLURM array parallelism
    for large numbers of families.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import socket

warnings.filterwarnings("ignore")

def get_base_dir(project_name: str) -> Path:
    hostname = socket.gethostname()
    if os.path.isdir("/scratch/drn2") or "dn" in hostname or "cn" in hostname or "gpu" in hostname or "jubail" in hostname:
        return Path(f"/scratch/drn2/PROJECTS/{project_name}")
    return Path(f"/media/drn/External1/{project_name}")

# Exact parameter match
XGB_PARAMS = {
    "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    "n_jobs": 4, "verbosity": 0,
}

ENV_FEATURES = [
    "modis_sst_mean_c", "sst_mean_c", "sst_max_c", "sst_min_c",
    "bathymetry_m", "sst_range_c", "nflh_mean", "poc_mean_mg_m3",
    "rrs_412", "solar_rad_mj_m2", "chl_max_mg_m3", "chl_mean_mg_m3",
    "rrs_443", "distance_to_coast_km", "rrs_469", "rrs_555", "rrs_547",
    "rrs_531", "rrs_488", "rrs_645", "rrs_667", "rrs_678",
    "chl_min_mg_m3", "elevation_m", "depth_m",
]

BLOCK_SIZE = 2.0
N_FOLDS = 10
CLR_PSEUDOCOUNT = 0.5

def clr_transform(X, pseudocount=CLR_PSEUDOCOUNT):
    X_pseudo = X + pseudocount
    log_X = np.log(X_pseudo)
    geom_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geom_mean

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-start", type=int, default=0,
                        help="Start index for family chunk (array jobs)")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Number of families per chunk (0 = all)")
    args = parser.parse_args()

    BASE = get_base_dir("TARA-LA4SR")
    RESULTS_DIR = BASE / "novel_families" / "results" / "track_b"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    COUNTS_RAW = BASE / "novel_families" / "data" / "abundance_matrices" / "dark_family_counts_raw.tsv"
    MERGED_PATH = (BASE / "03_analyses" / "ALGAGPT-based-analyses" /
                   "algagpt_gee_pfam_merged_GPS_RECOVERED_20260124_114445.tsv")

    print("=" * 60)
    print("  B4: Dark Proteome Environment Coupling")
    print("=" * 60)

    if not COUNTS_RAW.exists():
        print(f"  ERROR: {COUNTS_RAW} not found. Run B3 first.")
        sys.exit(1)

    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  ERROR: xgboost not installed")
        sys.exit(1)

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    from collections import Counter

    # Load data
    dark_df = pd.read_csv(COUNTS_RAW, sep="\t", index_col=0)

    # Apply chunking for array jobs
    family_ids = list(dark_df.columns)
    if args.chunk_size > 0:
        family_ids = family_ids[args.chunk_start:args.chunk_start + args.chunk_size]
        print(f"  Processing chunk: families {args.chunk_start} to "
              f"{args.chunk_start + len(family_ids)}")

    with open(MERGED_PATH) as f:
        for line in f:
            if not line.startswith("#"):
                avail_cols = line.strip().split("\t")
                break
    use_cols = ["assembly_id", "latitude", "longitude"] + \
               [c for c in ENV_FEATURES if c in avail_cols]
    merged_df = pd.read_csv(MERGED_PATH, sep="\t", comment="#",
                            usecols=use_cols, low_memory=False)
    merged_df = merged_df.set_index("assembly_id")

    available_env = [f for f in ENV_FEATURES if f in merged_df.columns]

    # Intersect
    common = dark_df.index.intersection(merged_df.index)
    merged_gps = merged_df.loc[common].dropna(subset=["latitude", "longitude"])
    common = merged_gps.index

    print(f"  Samples: {len(common)}, Families: {len(family_ids)}")

    # CLR
    dark_vals = dark_df.loc[common].values.astype(np.float64)
    dark_clr = clr_transform(dark_vals)

    # Spatial blocks
    X_env = merged_df.loc[common, available_env].apply(
        pd.to_numeric, errors="coerce").values
    lat = pd.to_numeric(merged_df.loc[common, "latitude"], errors="coerce").values
    lon = pd.to_numeric(merged_df.loc[common, "longitude"], errors="coerce").values

    block_ids = [f"{np.floor(la/BLOCK_SIZE)*BLOCK_SIZE}_{np.floor(lo/BLOCK_SIZE)*BLOCK_SIZE}"
                 for la, lo in zip(lat, lon)]
    block_counts = Counter(block_ids)
    blocks_sorted = sorted(block_counts, key=lambda b: block_counts[b], reverse=True)

    fold_assignment = {}
    fold_sizes = [0] * N_FOLDS
    for block in blocks_sorted:
        min_fold = int(np.argmin(fold_sizes))
        fold_assignment[block] = min_fold
        fold_sizes[min_fold] += block_counts[block]

    folds = np.array([fold_assignment[b] for b in block_ids])

    # Pre-compute env features
    fold_env = {}
    for fi in range(N_FOLDS):
        train_idx = np.where(folds != fi)[0]
        test_idx = np.where(folds == fi)[0]
        X_tr = X_env[train_idx].copy()
        X_te = X_env[test_idx].copy()
        col_means = np.nanmean(X_tr, axis=0)
        for j in range(X_tr.shape[1]):
            m = col_means[j] if not np.isnan(col_means[j]) else 0.0
            X_tr[np.isnan(X_tr[:, j]), j] = m
            X_te[np.isnan(X_te[:, j]), j] = m
        sc = StandardScaler()
        fold_env[fi] = (train_idx, test_idx, sc.fit_transform(X_tr), sc.transform(X_te))

    # Run forward model
    all_family_ids = list(dark_df.columns)
    results = []
    import time as _time

    for fi_idx, fam in enumerate(family_ids):
        t0 = _time.time()
        col_i = all_family_ids.index(fam)
        y_clr = dark_clr[:, col_i]
        y_pred_all = np.full(len(y_clr), np.nan)
        fold_r2 = {}

        for fold_idx in range(N_FOLDS):
            tri, tei, X_tr, X_te = fold_env[fold_idx]
            y_tr = y_clr[tri]
            y_te = y_clr[tei]
            if len(y_te) < 5:
                continue
            sc_y = StandardScaler()
            y_tr_s = sc_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X_tr, y_tr_s)
            y_pred_s = model.predict(X_te)
            y_pred = sc_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
            y_pred_all[tei] = y_pred
            fold_r2[fold_idx] = r2_score(y_te, y_pred) if len(y_te) > 1 else np.nan

        has_pred = ~np.isnan(y_pred_all)
        overall_r2 = r2_score(y_clr[has_pred], y_pred_all[has_pred]) if has_pred.sum() > 10 else np.nan
        fv = [v for v in fold_r2.values() if not np.isnan(v)]

        prevalence = int((dark_vals[:, col_i] > 0).sum())
        results.append({
            "family_id": fam, "r2_overall": overall_r2,
            "r2_median": np.median(fv) if fv else np.nan,
            "r2_q25": np.percentile(fv, 25) if fv else np.nan,
            "r2_q75": np.percentile(fv, 75) if fv else np.nan,
            "n_samples": int(has_pred.sum()), "prevalence": prevalence,
        })

        elapsed = _time.time() - t0
        print(f"  [{fi_idx+1}/{len(family_ids)}] {fam}: R2={overall_r2:.4f} [{elapsed:.1f}s]")
        sys.stdout.flush()

    # Save
    suffix = f"_chunk{args.chunk_start}" if args.chunk_size > 0 else ""
    r2_path = RESULTS_DIR / f"dark_family_xgboost_forward_r2{suffix}.tsv"
    pd.DataFrame(results).to_csv(r2_path, sep="\t", index=False)
    print(f"\n  Written: {r2_path}")

if __name__ == "__main__":
    main()
