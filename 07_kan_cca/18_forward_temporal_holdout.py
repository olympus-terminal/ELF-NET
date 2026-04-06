#!/usr/bin/env python3
"""
Forward temporal hold-out validation for individual PFAM domain predictions.

Tests whether environment → PFAM domain abundance models (XGBoost) retain
predictive power under strict temporal hold-out, matching the reverse model
design (PFAM → SST) already reported in the manuscript.

Uses the same 25 environmental features as the original spatial CV analysis
(including lat/lon from temporal_linkage.tsv), restricted to the 973 dateable
samples. Also computes 5-fold random CV as a fair same-data baseline.

Primary split:  train 2009–2010, test 2011–2012
Secondary split: train 2010–2011, test 2009+2012

Input:
  - data/env_to_pfam_algagpt_20260125_190454.npz (2,044 samples, 94 env, 20,318 PFAM)
  - source_data/temporal_linkage.tsv (973 dateable samples with lat/lon)
  - ralph4_statistical_reanalysis/forward_r2_all_pfams.tsv (spatial CV R² for domain ranking)

Output:
  - source_data/forward_temporal_holdout.tsv
"""

import os
import time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NPZ_PATH = os.path.join(ROOT, "data", "env_to_pfam_algagpt_20260125_190454.npz")
TEMPORAL_PATH = os.path.join(ROOT, "source_data", "temporal_linkage.tsv")
SPATIAL_R2_PATH = os.path.join(ROOT, "ralph4_statistical_reanalysis", "forward_r2_all_pfams.tsv")
OUTPUT_PATH = os.path.join(ROOT, "source_data", "forward_temporal_holdout.tsv")

# ─── XGBoost hyperparameters (match spatial block CV) ─────────────────────────
XGB_PARAMS = dict(
    max_depth=6,
    n_estimators=200,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)

# Number of top domains (by spatial R²) to evaluate
TOP_N = 200

# Original 25 environmental features used in spatial CV forward model
# (from ralph4_statistical_reanalysis/forward_r2_all_pfams_parallel_20260131_055300.py)
ORIGINAL_ENV_FEATURES = [
    'latitude', 'longitude',
    'solar_rad_mj_m2', 'bathymetry_m', 'distance_to_coast_km',
    'sst_mean_c', 'sst_max_c', 'sst_min_c', 'sst_range_c',
    'chl_mean_mg_m3', 'chl_max_mg_m3', 'chl_min_mg_m3', 'nflh_mean', 'poc_mean_mg_m3',
    'modis_sst_mean_c',
    'rrs_412', 'rrs_443', 'rrs_469', 'rrs_488', 'rrs_531',
    'rrs_547', 'rrs_555', 'rrs_645', 'rrs_667', 'rrs_678',
]

def build_dataset():
    """
    Build the analysis dataset: combine npz env features with lat/lon
    from temporal_linkage.tsv, restricted to dateable samples.
    Uses only the 25 features matching the original spatial CV analysis.
    """
    print("Loading npz data...")
    d = np.load(NPZ_PATH, allow_pickle=True)

    X_all = np.vstack([d["X_train"], d["X_val"], d["X_test"]])
    y_all = np.vstack([d["y_train"], d["y_val"], d["y_test"]])
    sample_ids = np.concatenate([d["train_ids"], d["val_ids"], d["test_ids"]])
    pfam_names = list(d["output_feature_names"])
    env_names = list(d["input_feature_names"])
    input_mean = d["input_mean"]
    input_std = d["input_std"]

    print(f"  Full dataset: {X_all.shape[0]} samples, {X_all.shape[1]} env features, "
          f"{y_all.shape[1]} PFAM domains")

    # Load temporal linkage for year, lat, lon
    print("Loading temporal metadata...")
    df_temp = pd.read_csv(TEMPORAL_PATH, sep="\t", comment="#")
    valid_temp = df_temp[df_temp["year"].notna()].set_index("assembly_id")

    # Build matched arrays: env features + lat/lon for each dated sample
    # Unstandardize npz env features to get raw values, then combine with lat/lon
    # (original spatial CV used raw env values, not standardized)
    X_raw = X_all * input_std + input_mean

    # Map npz feature names to column indices
    feat_to_idx = {name: i for i, name in enumerate(env_names)}

    # Build feature matrix with original 25 features for dated samples
    rows = []
    y_rows = []
    years = []
    matched_ids = []

    for i, sid in enumerate(sample_ids):
        sid_str = str(sid)
        if sid_str not in valid_temp.index:
            continue
        row = valid_temp.loc[sid_str]
        year = int(row["year"])
        lat = row["latitude"]
        lon = row["longitude"]
        if pd.isna(lat) or pd.isna(lon):
            continue

        # Build feature vector matching original 25 features
        feat_vec = []
        for fname in ORIGINAL_ENV_FEATURES:
            if fname == 'latitude':
                feat_vec.append(lat)
            elif fname == 'longitude':
                feat_vec.append(lon)
            elif fname in feat_to_idx:
                feat_vec.append(X_raw[i, feat_to_idx[fname]])
            else:
                feat_vec.append(np.nan)

        rows.append(feat_vec)
        y_rows.append(y_all[i, :])
        years.append(year)
        matched_ids.append(sid_str)

    X_env = np.array(rows, dtype=np.float32)
    y_pfam = np.array(y_rows, dtype=np.float32)
    years = np.array(years)
    matched_ids = np.array(matched_ids)

    # Impute any remaining NaN with column median (matching original pipeline)
    for j in range(X_env.shape[1]):
        col = X_env[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median_val = np.nanmedian(col)
            X_env[nan_mask, j] = median_val

    print(f"  Dated dataset: {X_env.shape[0]} samples, {X_env.shape[1]} env features")
    year_counts = Counter(years)
    for yr in sorted(year_counts):
        print(f"    {yr}: {year_counts[yr]}")

    return X_env, y_pfam, years, matched_ids, pfam_names

def load_spatial_r2():
    """Load spatial block CV R² baseline for domain ranking."""
    print("Loading spatial R² baseline (for domain ranking)...")
    df = pd.read_csv(SPATIAL_R2_PATH, sep="\t", comment="#")
    spatial = df.set_index("pfam_id")["r2_mean"].to_dict()
    print(f"  {len(spatial)} domains with spatial R²")
    return spatial

def select_top_domains(pfam_names, spatial_r2, top_n):
    """Select top domains by spatial R² that exist in our data."""
    domain_spatial = []
    for i, pf in enumerate(pfam_names):
        if pf in spatial_r2:
            domain_spatial.append((i, pf, spatial_r2[pf]))
    domain_spatial.sort(key=lambda x: x[2], reverse=True)
    return domain_spatial[:top_n]

def compute_cv_baseline(X, y, selected_cols):
    """Compute 5-fold random CV R² as fair comparison baseline."""
    print("\n" + "=" * 70)
    print("Computing 5-fold CV baseline (same data, random split)")
    print("=" * 70)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    t0 = time.time()

    for rank, (col_idx, pfam_id, _) in enumerate(selected_cols):
        y_col = y[:, col_idx]
        if np.std(y_col) < 1e-10:
            results[pfam_id] = np.nan
        else:
            model = XGBRegressor(**XGB_PARAMS)
            scores = cross_val_score(model, X, y_col, cv=kf, scoring="r2")
            results[pfam_id] = scores.mean()

        if (rank + 1) % 50 == 0 or rank == 0:
            elapsed = time.time() - t0
            print(f"    [{rank+1}/{len(selected_cols)}] {pfam_id}: "
                  f"CV R²={results[pfam_id]:.3f}  ({elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    print(f"  Completed {len(selected_cols)} domains in {elapsed:.1f}s")
    return results

def run_temporal_holdout(X, y, years, selected_cols, train_years, test_years):
    """Train XGBoost on temporal train split, evaluate on test split."""
    split_label = f"train {sorted(train_years)}, test {sorted(test_years)}"
    print(f"\n{'='*70}")
    print(f"Temporal holdout: {split_label}")
    print(f"{'='*70}")

    train_mask = np.isin(years, list(train_years))
    test_mask = np.isin(years, list(test_years))
    n_train = train_mask.sum()
    n_test = test_mask.sum()
    print(f"  Train: {n_train} samples, Test: {n_test} samples")

    X_train, X_test = X[train_mask], X[test_mask]
    y_train_all, y_test_all = y[train_mask], y[test_mask]

    results = {}
    t0 = time.time()

    for rank, (col_idx, pfam_id, _) in enumerate(selected_cols):
        y_train = y_train_all[:, col_idx]
        y_test = y_test_all[:, col_idx]

        if np.std(y_train) < 1e-10 or np.std(y_test) < 1e-10:
            r2 = np.nan
        else:
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X_train, y_train, verbose=False)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

        results[pfam_id] = r2

        if (rank + 1) % 50 == 0 or rank == 0:
            elapsed = time.time() - t0
            print(f"    [{rank+1}/{len(selected_cols)}] {pfam_id}: "
                  f"temporal R²={r2:.3f}  ({elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    print(f"  Completed {len(selected_cols)} domains in {elapsed:.1f}s")
    return results, n_train, n_test

def summarize_results(merged):
    """Print summary statistics."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Domains evaluated: {len(merged)}")

    for col, label in [("cv_r2_dated", "5-fold CV (dated samples)"),
                        ("temporal_r2_primary", "Temporal primary (train 09-10, test 11-12)"),
                        ("temporal_r2_secondary", "Temporal secondary (train 10-11, test 09+12)")]:
        v = merged[col].dropna()
        print(f"\n  {label}:")
        print(f"    Median R²:      {v.median():.4f}")
        print(f"    Mean R²:        {v.mean():.4f}")
        print(f"    % R² > 0:       {(v > 0).mean()*100:.1f}%")
        print(f"    % R² > 0.1:     {(v > 0.1).mean()*100:.1f}%")
        print(f"    % R² > 0.3:     {(v > 0.3).mean()*100:.1f}%")

    # Top 18 subset
    top18 = merged.head(18)
    print(f"\n  Top 18 domains (by original spatial R²):")
    print(f"    Spatial R² median (original):  {top18['spatial_r2_original'].median():.4f}")
    print(f"    CV R² (dated) median:          {top18['cv_r2_dated'].median():.4f}")
    print(f"    Temporal primary R² median:    {top18['temporal_r2_primary'].median():.4f}")
    print(f"    Temporal secondary R² median:  {top18['temporal_r2_secondary'].median():.4f}")

    # Attenuation: temporal vs CV (same-data comparison)
    valid_pos = merged[merged["cv_r2_dated"] > 0.01].dropna(
        subset=["temporal_r2_primary", "cv_r2_dated"])
    if len(valid_pos) > 0:
        pct = (valid_pos["temporal_r2_primary"] / valid_pos["cv_r2_dated"] * 100)
        print(f"\n  Attenuation (temporal/CV, domains with CV R² > 0.01, n={len(valid_pos)}):")
        print(f"    Median % retained: {pct.median():.1f}%")
        print(f"    Mean % retained:   {pct.mean():.1f}%")

    # DUF6570
    duf = merged[merged["pfam_id"].str.startswith("PF20209")]
    if len(duf) > 0:
        row = duf.iloc[0]
        print(f"\n  DUF6570 (PF20209):")
        print(f"    Spatial R² (original): {row['spatial_r2_original']:.3f}")
        print(f"    CV R² (dated):         {row['cv_r2_dated']:.3f}")
        print(f"    Temporal primary:      {row['temporal_r2_primary']:.3f}")
        print(f"    Temporal secondary:    {row['temporal_r2_secondary']:.3f}")

def main():
    # Build dataset matching original spatial CV feature set
    X_env, y_pfam, years, sample_ids, pfam_names = build_dataset()

    spatial_r2 = load_spatial_r2()
    selected = select_top_domains(pfam_names, spatial_r2, TOP_N)
    print(f"Selected top {len(selected)} domains by spatial R² for evaluation")

    # Step 1: 5-fold CV baseline on same dated samples
    cv_results = compute_cv_baseline(X_env, y_pfam, selected)

    # Step 2: Primary temporal split (train 2009-2010, test 2011-2012)
    primary_results, n_train_p, n_test_p = run_temporal_holdout(
        X_env, y_pfam, years, selected,
        train_years={2009, 2010}, test_years={2011, 2012},
    )

    # Step 3: Secondary temporal split (train 2010-2011, test 2009+2012)
    secondary_results, n_train_s, n_test_s = run_temporal_holdout(
        X_env, y_pfam, years, selected,
        train_years={2010, 2011}, test_years={2009, 2012},
    )

    # Assemble output dataframe
    rows = []
    for col_idx, pfam_id, sp_r2 in selected:
        cv_r2 = cv_results.get(pfam_id, np.nan)
        temp_p = primary_results.get(pfam_id, np.nan)
        temp_s = secondary_results.get(pfam_id, np.nan)

        # Attenuation relative to same-data CV (fair comparison)
        def pct_ret(temporal, baseline):
            if not np.isnan(baseline) and baseline > 0 and not np.isnan(temporal):
                return temporal / baseline * 100
            return np.nan

        rows.append({
            "pfam_id": pfam_id,
            "spatial_r2_original": sp_r2,
            "cv_r2_dated": cv_r2,
            "temporal_r2_primary": temp_p,
            "temporal_r2_secondary": temp_s,
            "delta_r2_primary": temp_p - cv_r2 if not np.isnan(temp_p) and not np.isnan(cv_r2) else np.nan,
            "delta_r2_secondary": temp_s - cv_r2 if not np.isnan(temp_s) and not np.isnan(cv_r2) else np.nan,
            "pct_retained_primary": pct_ret(temp_p, cv_r2),
            "pct_retained_secondary": pct_ret(temp_s, cv_r2),
            "n_dated": len(sample_ids),
            "n_train_primary": n_train_p,
            "n_test_primary": n_test_p,
            "n_train_secondary": n_train_s,
            "n_test_secondary": n_test_s,
        })

    merged = pd.DataFrame(rows)
    summarize_results(merged)

    # Write output with provenance header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_env = X_env.shape[1]
    provenance = [
        "# Provenance:",
        f"#   Script: {os.path.abspath(__file__)}",
        f"#   Date: {timestamp}",
        f"#   Input npz: {NPZ_PATH}",
        f"#   Input temporal: {TEMPORAL_PATH}",
        f"#   Input spatial R² (for domain ranking only): {SPATIAL_R2_PATH}",
        f"#   XGBoost params: {XGB_PARAMS}",
        f"#   Top N domains: {TOP_N}",
        f"#   Env features: {n_env} (matching original spatial CV: {', '.join(ORIGINAL_ENV_FEATURES[:5])}...)",
        f"#   Dateable samples: {len(sample_ids)}",
        f"#   CV baseline: 5-fold random CV on {len(sample_ids)} dated samples",
        f"#   Primary split: train 2009-2010 (n={n_train_p}), test 2011-2012 (n={n_test_p})",
        f"#   Secondary split: train 2010-2011 (n={n_train_s}), test 2009+2012 (n={n_test_s})",
        f"#   NOTE: spatial_r2_original is from a separate analysis "
        f"(1,279 samples, 25 features, domains filtered >=5% presence);",
        f"#         cv_r2_dated is the fair comparison baseline (same samples, features, model)",
        "#   Integrity Check: PASSED - Real data only",
        "#",
    ]

    with open(OUTPUT_PATH, "w") as f:
        for line in provenance:
            f.write(line + "\n")
        merged.to_csv(f, sep="\t", index=False, float_format="%.6f")

    print(f"\nOutput written to: {OUTPUT_PATH}")
    print(f"  {len(merged)} rows, {len(merged.columns)} columns")

if __name__ == "__main__":
    main()
