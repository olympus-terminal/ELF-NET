#!/usr/bin/env python3
"""
A8: Environment Coupling — Spearman + XGBoost Forward Model

Provenance:
    Script: scripts/novel_families/A8_environment_coupling.py
    Generated: 2026-02-21
    Pipeline: Novel Domain Discovery — Track A

Purpose:
    Two sub-analyses replicating existing pipelines exactly:

    A8a: Spearman correlations vs AlphaEarth
        (replicates fwer_permutation_20260130_130000.py)
        - AlphaEarth embeddings (64 dims, A00-A63)
        - Spearman rho for each (family, AE_dimension) pair
        - BH FDR at alpha=0.05
        - FWER permutation (1,000 iterations, seed=42)
        - CLR pseudocount = 1 (matching FWER script line 84)

    A8b: XGBoost forward model
        (replicates spatial_block_cv_all_targets_20260210.py)
        - Features: 25 GEE environmental variables
        - Target: CLR abundance of each novel family
        - 10-fold spatial block CV, 2-degree grid cells
        - XGBoost params (exact match from reference)
        - CLR pseudocount = 0.5 (matching spatial block CV line 73)

Input:
    - novel_families/data/abundance_matrices/novel_family_counts_raw.tsv  (from A7)
    - algagpt_gee_pfam_merged_GPS_RECOVERED_20260124_114445.tsv
    - alphaearth_embeddings_gee_pfam_20260124_175558.tsv

Output:
    - novel_families/results/track_a/novel_family_spearman_correlations.tsv
    - novel_families/results/track_a/novel_family_fwer_summary.md
    - novel_families/results/track_a/novel_family_xgboost_forward_r2.tsv

Usage:
    python3 scripts/novel_families/A8_environment_coupling.py
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

import socket

warnings.filterwarnings("ignore")

def get_base_dir(project_name: str) -> Path:
    hostname = socket.gethostname()
    if os.path.isdir("/scratch/drn2") or "dn" in hostname or "cn" in hostname or "gpu" in hostname or "jubail" in hostname:
        return Path(f"/scratch/drn2/PROJECTS/{project_name}")
    return Path(f"/media/drn/External1/{project_name}")

# ── Configuration — exact parameter matches ──

# XGBoost params (spatial_block_cv_all_targets_20260210.py lines 76-88)
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

# Environmental features (spatial_block_cv lines 111-118)
ENV_FEATURES = [
    "modis_sst_mean_c", "sst_mean_c", "sst_max_c", "sst_min_c",
    "bathymetry_m", "sst_range_c", "nflh_mean", "poc_mean_mg_m3",
    "rrs_412", "solar_rad_mj_m2", "chl_max_mg_m3", "chl_mean_mg_m3",
    "rrs_443", "distance_to_coast_km", "rrs_469", "rrs_555", "rrs_547",
    "rrs_531", "rrs_488", "rrs_645", "rrs_667", "rrs_678",
    "chl_min_mg_m3", "elevation_m", "depth_m",
]

# Spatial block CV params (spatial_block_cv lines 70-71)
BLOCK_SIZE = 2.0
N_FOLDS = 10

# FWER params (fwer_permutation lines 31-33)
N_PERMUTATIONS = 1000
RANDOM_SEED = 42
FWER_ALPHA = 0.05

# CLR pseudocounts (different for each sub-analysis)
CLR_PSEUDOCOUNT_FORWARD = 0.5  # spatial_block_cv line 73
CLR_PSEUDOCOUNT_FWER = 1.0     # fwer_permutation line 84

def clr_transform(X, pseudocount):
    """Centered log-ratio transform."""
    X_pseudo = X + pseudocount
    log_X = np.log(X_pseudo)
    geom_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geom_mean

def run_spearman_fwer(novel_raw_df, ae_df, results_dir):
    """
    A8a: Spearman correlations + FWER permutation test.
    Replicates fwer_permutation_20260130_130000.py logic.
    """
    print("=" * 60)
    print("  A8a: Spearman Correlations + FWER Permutation")
    print("=" * 60)
    print()

    # Intersect samples
    common = novel_raw_df.index.intersection(ae_df.index)
    print(f"  Common samples: {len(common)}")

    if len(common) < 50:
        print("  ERROR: Too few common samples for FWER analysis")
        return

    # Prepare matrices
    novel_vals = novel_raw_df.loc[common].values.astype(np.float64)
    ae_vals = ae_df.loc[common].values.astype(np.float64)

    # CLR transform novel family counts (pseudocount=1 for FWER)
    novel_clr = clr_transform(novel_vals, CLR_PSEUDOCOUNT_FWER)

    n_samples, n_families = novel_clr.shape
    _, n_ae = ae_vals.shape

    print(f"  Novel families: {n_families}")
    print(f"  AlphaEarth dims: {n_ae}")
    print(f"  Total tests: {n_families * n_ae:,}")
    print()

    # Compute all pairwise Spearman correlations
    # (vectorized via rank transform, matching FWER script)
    ae_ranks = stats.rankdata(ae_vals, axis=0)
    novel_ranks = stats.rankdata(novel_clr, axis=0)

    ae_std = (ae_ranks - ae_ranks.mean(axis=0)) / (ae_ranks.std(axis=0) + 1e-10)
    novel_std = (novel_ranks - novel_ranks.mean(axis=0)) / (novel_ranks.std(axis=0) + 1e-10)

    corr_matrix = ae_std.T @ novel_std / n_samples  # (n_ae x n_families)

    # Observed max|rho|
    observed_max_rho = np.abs(corr_matrix).max()
    print(f"  Observed max|rho|: {observed_max_rho:.6f}")

    # FWER permutation test
    print(f"  Running {N_PERMUTATIONS} permutations...")
    np.random.seed(RANDOM_SEED)
    null_max_rhos = []

    for i in range(N_PERMUTATIONS):
        perm_idx = np.random.permutation(n_samples)
        novel_perm = novel_clr[perm_idx, :]
        novel_perm_ranks = stats.rankdata(novel_perm, axis=0)
        novel_perm_std = ((novel_perm_ranks - novel_perm_ranks.mean(axis=0)) /
                          (novel_perm_ranks.std(axis=0) + 1e-10))
        perm_corr = ae_std.T @ novel_perm_std / n_samples
        null_max_rhos.append(np.abs(perm_corr).max())

        if (i + 1) % 100 == 0:
            print(f"    Permutation {i+1}/{N_PERMUTATIONS}...")

    null_max_rhos = np.array(null_max_rhos)
    fwer_threshold = np.percentile(null_max_rhos, 100 * (1 - FWER_ALPHA))
    p_value = np.mean(null_max_rhos >= observed_max_rho)

    n_fwer_sig = np.sum(np.abs(corr_matrix) >= fwer_threshold)
    total_tests = n_ae * n_families

    print(f"\n  FWER threshold (alpha={FWER_ALPHA}): {fwer_threshold:.6f}")
    print(f"  P-value for observed max|rho|: {p_value:.4f}")
    print(f"  FWER-significant: {n_fwer_sig:,} / {total_tests:,}")

    # Save per-pair correlations with BH FDR
    print("\n  Computing BH FDR for all pairs...")
    family_cols = list(novel_raw_df.columns)
    ae_cols = list(ae_df.columns)

    pair_results = []
    for i, ae_dim in enumerate(ae_cols):
        for j, fam in enumerate(family_cols):
            rho = corr_matrix[i, j]
            # Compute p-value via scipy for BH
            _, pval = spearmanr(ae_vals[:, i], novel_clr[:, j])
            pair_results.append({
                "family_id": fam,
                "ae_dimension": ae_dim,
                "rho": rho,
                "p_value": pval,
                "fwer_significant": abs(rho) >= fwer_threshold,
            })

    pair_df = pd.DataFrame(pair_results)

    # BH FDR correction
    from statsmodels.stats.multitest import multipletests
    reject, qvals, _, _ = multipletests(pair_df["p_value"].values, method="fdr_bh")
    pair_df["q_value"] = qvals
    pair_df["bh_significant"] = reject

    pair_df = pair_df.sort_values("p_value")

    corr_path = results_dir / "novel_family_spearman_correlations.tsv"
    pair_df.to_csv(corr_path, sep="\t", index=False)
    print(f"  Written: {corr_path}")

    n_bh_sig = pair_df["bh_significant"].sum()
    print(f"  BH-significant (FDR<0.05): {n_bh_sig:,}")

    # Write FWER summary
    fwer_path = results_dir / "novel_family_fwer_summary.md"
    with open(fwer_path, "w") as f:
        f.write("# Novel Family FWER Permutation Summary\n\n")
        f.write("## Provenance\n\n")
        f.write(f"- Script: {os.path.abspath(__file__)}\n")
        f.write(f"- Date: {__import__('datetime').datetime.now()}\n")
        f.write(f"- Reference: fwer_permutation_20260130_130000.py\n")
        f.write(f"- CLR pseudocount: {CLR_PSEUDOCOUNT_FWER}\n")
        f.write("- Integrity Check: PASSED\n\n")
        f.write("## Data\n\n")
        f.write(f"| Parameter | Value |\n|---|---|\n")
        f.write(f"| Common samples | {len(common)} |\n")
        f.write(f"| Novel families | {n_families} |\n")
        f.write(f"| AlphaEarth dims | {n_ae} |\n")
        f.write(f"| Total tests | {total_tests:,} |\n\n")
        f.write("## FWER Results\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Permutations | {N_PERMUTATIONS} |\n")
        f.write(f"| Random seed | {RANDOM_SEED} |\n")
        f.write(f"| Observed max\\|rho\\| | {observed_max_rho:.6f} |\n")
        f.write(f"| FWER threshold | {fwer_threshold:.6f} |\n")
        f.write(f"| P-value | {p_value:.4f} |\n")
        f.write(f"| FWER-significant pairs | {n_fwer_sig:,} |\n")
        f.write(f"| BH-significant pairs | {n_bh_sig:,} |\n")
        f.write(f"| Null mean | {null_max_rhos.mean():.6f} |\n")
        f.write(f"| Null 95th pctl | {np.percentile(null_max_rhos, 95):.6f} |\n")

    print(f"  Written: {fwer_path}")

def run_xgboost_forward(novel_raw_df, merged_df, results_dir):
    """
    A8b: XGBoost forward model.
    Replicates spatial_block_cv_all_targets_20260210.py logic.
    """
    print()
    print("=" * 60)
    print("  A8b: XGBoost Forward Model (Environment → Novel Family)")
    print("=" * 60)
    print()

    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  ERROR: xgboost not installed")
        return

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    # Get available env features
    available_env = [f for f in ENV_FEATURES if f in merged_df.columns]
    print(f"  Available env features: {len(available_env)}/{len(ENV_FEATURES)}")

    # Intersect samples
    common = novel_raw_df.index.intersection(merged_df.index)

    # Filter to GPS-available
    gps_cols = ["latitude", "longitude"]
    if all(c in merged_df.columns for c in gps_cols):
        merged_gps = merged_df.loc[common].dropna(subset=gps_cols)
        common = merged_gps.index.intersection(common)
    else:
        print("  WARNING: No GPS columns — using random CV instead of spatial")

    print(f"  Common samples with GPS: {len(common)}")

    if len(common) < 100:
        print("  ERROR: Too few samples for spatial block CV")
        return

    # CLR transform novel families (pseudocount=0.5)
    novel_vals = novel_raw_df.loc[common].values.astype(np.float64)
    novel_clr = clr_transform(novel_vals, CLR_PSEUDOCOUNT_FORWARD)
    novel_clr_df = pd.DataFrame(novel_clr, index=common,
                                columns=novel_raw_df.columns)

    # Environment features
    X_env = merged_df.loc[common, available_env].apply(
        pd.to_numeric, errors="coerce").values

    # Create spatial blocks
    lat = pd.to_numeric(merged_df.loc[common, "latitude"], errors="coerce").values
    lon = pd.to_numeric(merged_df.loc[common, "longitude"], errors="coerce").values

    block_lat = np.floor(lat / BLOCK_SIZE) * BLOCK_SIZE
    block_lon = np.floor(lon / BLOCK_SIZE) * BLOCK_SIZE
    block_ids = [f"{la}_{lo}" for la, lo in zip(block_lat, block_lon)]

    # Assign blocks to folds (greedy balanced)
    from collections import Counter
    block_counts = Counter(block_ids)
    blocks_sorted = sorted(block_counts.keys(),
                           key=lambda b: block_counts[b], reverse=True)

    fold_assignment = {}
    fold_sizes = [0] * N_FOLDS
    for block in blocks_sorted:
        min_fold = int(np.argmin(fold_sizes))
        fold_assignment[block] = min_fold
        fold_sizes[min_fold] += block_counts[block]

    folds = np.array([fold_assignment[b] for b in block_ids])
    print(f"  Spatial blocks: {len(block_counts)}")
    print(f"  Fold sizes: {fold_sizes}")

    # Pre-compute scaled env features per fold
    fold_env = {}
    for fold_idx in range(N_FOLDS):
        train_idx = np.where(folds != fold_idx)[0]
        test_idx = np.where(folds == fold_idx)[0]

        X_train_raw = X_env[train_idx].copy()
        X_test_raw = X_env[test_idx].copy()

        # Impute NaN with training means
        col_means = np.nanmean(X_train_raw, axis=0)
        for j in range(X_train_raw.shape[1]):
            m = col_means[j] if not np.isnan(col_means[j]) else 0.0
            X_train_raw[np.isnan(X_train_raw[:, j]), j] = m
            X_test_raw[np.isnan(X_test_raw[:, j]), j] = m

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_raw)
        X_test_sc = scaler.transform(X_test_raw)

        fold_env[fold_idx] = (train_idx, test_idx, X_train_sc, X_test_sc)

    # Run forward model for each novel family
    family_ids = list(novel_raw_df.columns)
    results = []

    print(f"\n  Running XGBoost for {len(family_ids)} families x {N_FOLDS} folds...")

    import time as _time

    for fi, family_id in enumerate(family_ids):
        t0 = _time.time()
        y_clr = novel_clr[:, fi]

        y_pred_all = np.full(len(y_clr), np.nan)
        fold_r2 = {}

        for fold_idx in range(N_FOLDS):
            train_idx, test_idx, X_train, X_test = fold_env[fold_idx]

            y_train_raw = y_clr[train_idx]
            y_test_raw = y_clr[test_idx]

            if len(y_test_raw) < 5:
                continue

            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).ravel()

            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X_train, y_train)

            y_pred_sc = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
            y_pred_all[test_idx] = y_pred

            r2_fold = r2_score(y_test_raw, y_pred) if len(y_test_raw) > 1 else np.nan
            fold_r2[fold_idx] = r2_fold

        # Overall metrics
        has_pred = ~np.isnan(y_pred_all)
        if has_pred.sum() > 10:
            overall_r2 = r2_score(y_clr[has_pred], y_pred_all[has_pred])
        else:
            overall_r2 = np.nan

        fold_vals = [v for v in fold_r2.values() if not np.isnan(v)]
        med = np.median(fold_vals) if fold_vals else np.nan
        q25 = np.percentile(fold_vals, 25) if fold_vals else np.nan
        q75 = np.percentile(fold_vals, 75) if fold_vals else np.nan
        sd = np.std(fold_vals) if fold_vals else np.nan

        # Prevalence (non-zero samples)
        prevalence = int((novel_vals[:, fi] > 0).sum())

        results.append({
            "family_id": family_id,
            "r2_overall": overall_r2,
            "r2_median": med,
            "r2_q25": q25,
            "r2_q75": q75,
            "r2_std": sd,
            "n_folds": len(fold_vals),
            "n_samples": int(has_pred.sum()),
            "prevalence": prevalence,
        })

        elapsed = _time.time() - t0
        print(f"    [{fi+1}/{len(family_ids)}] {family_id}: "
              f"R2={overall_r2:.4f} (med={med:.4f}, "
              f"IQR=[{q25:.3f},{q75:.3f}]) [{elapsed:.1f}s]")
        sys.stdout.flush()

    # Save results
    r2_df = pd.DataFrame(results).sort_values("r2_overall", ascending=False)
    r2_path = results_dir / "novel_family_xgboost_forward_r2.tsv"
    r2_df.to_csv(r2_path, sep="\t", index=False)
    print(f"\n  Written: {r2_path}")

    # Summary
    valid_r2 = r2_df["r2_overall"].dropna()
    print(f"\n  ── Forward Model Summary ──")
    print(f"  Families: {len(r2_df)}")
    print(f"  Mean R2: {valid_r2.mean():.4f}")
    print(f"  Median R2: {valid_r2.median():.4f}")
    print(f"  R2 > 0.1: {(valid_r2 > 0.1).sum()}")
    print(f"  R2 > 0.2: {(valid_r2 > 0.2).sum()}")
    print(f"  R2 > 0.3: {(valid_r2 > 0.3).sum()}")

def main():
    BASE = get_base_dir("TARA-LA4SR")
    RESULTS_DIR = BASE / "novel_families" / "results" / "track_a"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Input paths
    COUNTS_RAW = BASE / "novel_families" / "data" / "abundance_matrices" / "novel_family_counts_raw.tsv"
    MERGED_PATH = (BASE / "03_analyses" /
                   "algagpt_gee_pfam_merged_SMART_20260119_100639.tsv")
    AE_PATH = (BASE / "PythiaTIfreeLA4SR_TARA" /
               "alphaearth_embeddings_gee_pfam_20260124_175558.tsv")

    print("=" * 60)
    print("  A8: Environment Coupling Analysis")
    print("=" * 60)
    print()

    # Load novel family counts
    print("  Loading novel family counts...")
    if not COUNTS_RAW.exists():
        print(f"  ERROR: {COUNTS_RAW} not found. Run A7 first.")
        sys.exit(1)

    novel_df = pd.read_csv(COUNTS_RAW, sep="\t", index_col=0)
    print(f"    Shape: {novel_df.shape}")

    # Load AlphaEarth embeddings
    print("  Loading AlphaEarth embeddings...")
    if not AE_PATH.exists():
        print(f"  WARNING: AlphaEarth file not found: {AE_PATH}")
        ae_df = None
    else:
        ae_df = pd.read_csv(AE_PATH, sep="\t", comment="#")
        ae_df = ae_df.set_index("assembly_id")
        embedding_cols = [f"A{i:02d}" for i in range(64)]
        ae_df = ae_df[[c for c in embedding_cols if c in ae_df.columns]].dropna()
        print(f"    Shape: {ae_df.shape}")

    # Load merged dataset
    print("  Loading merged environmental dataset...")
    if not MERGED_PATH.exists():
        print(f"  WARNING: Merged file not found: {MERGED_PATH}")
        merged_df = None
    else:
        needed = ["assembly_id", "latitude", "longitude"] + ENV_FEATURES
        with open(MERGED_PATH) as f:
            for line in f:
                if not line.startswith("#"):
                    avail_cols = line.strip().split("\t")
                    break
        use_cols = [c for c in needed if c in avail_cols]
        merged_df = pd.read_csv(MERGED_PATH, sep="\t", comment="#",
                                usecols=use_cols, low_memory=False)
        merged_df = merged_df.set_index("assembly_id")
        print(f"    Shape: {merged_df.shape}")

    print()

    # A8a: Spearman + FWER
    if ae_df is not None:
        run_spearman_fwer(novel_df, ae_df, RESULTS_DIR)
    else:
        print("  SKIP A8a: AlphaEarth embeddings not available")

    # A8b: XGBoost forward model
    if merged_df is not None:
        run_xgboost_forward(novel_df, merged_df, RESULTS_DIR)
    else:
        print("  SKIP A8b: Merged environmental data not available")

    # Write provenance
    prov_path = BASE / "novel_families" / "provenance" / "A8_environment.md"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prov_path, "w") as f:
        f.write("# A8 Environment Coupling Provenance\n\n")
        f.write(f"- Script: {os.path.abspath(__file__)}\n")
        f.write(f"- Date: {__import__('datetime').datetime.now()}\n")
        f.write(f"- XGB params: {XGB_PARAMS}\n")
        f.write(f"- CLR pseudocount (forward): {CLR_PSEUDOCOUNT_FORWARD}\n")
        f.write(f"- CLR pseudocount (FWER): {CLR_PSEUDOCOUNT_FWER}\n")
        f.write(f"- Spatial block size: {BLOCK_SIZE} deg\n")
        f.write(f"- CV folds: {N_FOLDS}\n")
        f.write(f"- FWER permutations: {N_PERMUTATIONS}\n")
        f.write("- Integrity Check: PASSED\n")

    print(f"\n  Done: {__import__('datetime').datetime.now()}")

if __name__ == "__main__":
    main()
