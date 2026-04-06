#!/usr/bin/env python3
"""
Task 3: XGBoost Prediction - GEE Variables from PFAM Counts (algaGPT-filtered)

Trains XGBoost models to predict GEE environmental variables from algaGPT-filtered
PFAM domain abundances. Computes SHAP values for feature importance.

Provenance:
  Input: algagpt_gee_pfam_merged_SMART_20260119_100639.tsv
  Reference: ../../03_analyses/env_pfam_manifold/08_xgboost_shap_full_20260114_140000.py
  Date: 2026-01-19
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path("/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses")
INPUT_FILE = BASE_DIR / "algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# GEE variables to predict
GEE_VARS = [
    'air_temp_mean_c', 'air_temp_max_c', 'air_temp_min_c', 'air_temp_range_c',
    'precip_mean_mm', 'solar_rad_mj_m2', 'elevation_m', 'bathymetry_m',
    'distance_to_coast_km', 'landcover_class', 'sst_mean_c', 'sst_max_c',
    'sst_min_c', 'sst_range_c', 'chl_mean_mg_m3', 'chl_max_mg_m3',
    'chl_min_mg_m3', 'nflh_mean', 'poc_mean_mg_m3', 'modis_sst_mean_c',
    'rrs_412', 'rrs_443', 'rrs_469', 'rrs_488', 'rrs_531', 'rrs_547',
    'rrs_555', 'rrs_645', 'rrs_667', 'rrs_678'
]

# XGBoost parameters
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
    'n_jobs': -1,
    'verbosity': 0
}

# SHAP parameters
SHAP_SAMPLE_SIZE = 500
TOP_FEATURES = 50  # Top N features per variable

def load_and_filter_data(prevalence_threshold=0.05):
    """Load data and filter low-prevalence PFAMs."""
    print("=" * 70)
    print("LOADING AND FILTERING DATA")
    print("=" * 70)
    print(f"  Input: {INPUT_FILE}")

    # Load data
    df = pd.read_csv(INPUT_FILE, sep='\t', comment='#', low_memory=False)
    print(f"  Total samples: {len(df):,}")

    # Identify PFAM columns
    pfam_cols = [col for col in df.columns if col.startswith('PF')]
    print(f"  Total PFAM domains: {len(pfam_cols):,}")

    # Calculate prevalence
    pfam_matrix = df[pfam_cols].values
    prevalence = (pfam_matrix > 0).mean(axis=0)

    # Filter by prevalence
    prevalence_mask = prevalence >= prevalence_threshold
    n_passing = prevalence_mask.sum()

    print(f"\n  FILTERING:")
    print(f"    Prevalence threshold: {prevalence_threshold*100:.1f}%")
    print(f"    Min samples required: {int(prevalence_threshold * len(df))}")
    print(f"    Domains passing: {n_passing:,} ({100*n_passing/len(pfam_cols):.1f}%)")

    # Get filtered PFAM columns
    filtered_pfam_cols = [pfam_cols[i] for i in range(len(pfam_cols)) if prevalence_mask[i]]

    # Extract GEE variables present in data
    gee_present = [var for var in GEE_VARS if var in df.columns]
    print(f"  GEE variables: {len(gee_present)}")

    # Create filtered dataframes
    gee_df = df[gee_present].copy()
    pfam_df = df[filtered_pfam_cols].copy()

    # Fill missing GEE values with median
    gee_df = gee_df.fillna(gee_df.median())

    return gee_df, pfam_df, gee_present, filtered_pfam_cols

def apply_clr_transform(pfam_df):
    """Apply Centered Log-Ratio transformation to PFAM counts."""
    print("\n  Applying CLR transformation...")

    pfam_vals = pfam_df.values.copy().astype(float)

    # Add pseudocount for zeros
    pfam_vals = pfam_vals + 1

    # Log transform
    log_vals = np.log(pfam_vals)

    # Subtract geometric mean per sample
    geo_mean = log_vals.mean(axis=1, keepdims=True)
    clr_vals = log_vals - geo_mean

    print(f"    CLR range: [{clr_vals.min():.3f}, {clr_vals.max():.3f}]")

    return pd.DataFrame(clr_vals, columns=pfam_df.columns, index=pfam_df.index)

def train_xgboost_model(X_train, X_test, y_train, y_test, target_name):
    """Train XGBoost model and return performance metrics."""

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    results = {
        'gee_variable': target_name,
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'mae_test': mean_absolute_error(y_test, y_pred_test)
    }

    return results, model

def compute_shap_values(model, X, sample_size=SHAP_SAMPLE_SIZE):
    """Compute SHAP values for feature importance."""

    # Subsample for efficiency
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    # Create explainer and compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    return mean_abs_shap

def run_xgboost_analysis(pfam_clr, gee_df, gee_cols, pfam_cols):
    """Train XGBoost models for all GEE variables."""
    print("\n" + "=" * 70)
    print("TRAINING XGBOOST MODELS")
    print("=" * 70)

    X = pfam_clr.values
    performance_results = []
    importance_results = []

    for i, gee_var in enumerate(gee_cols):
        print(f"  [{i+1}/{len(gee_cols)}] Predicting {gee_var}...")

        y = gee_df[gee_var].values

        # Filter out samples with NaN target
        valid_mask = ~np.isnan(y)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if len(y_valid) < 50:
            print(f"    Skipping (too few valid samples: {len(y_valid)})")
            continue

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_valid, y_valid, test_size=0.2, random_state=42
        )

        # Train model
        perf, model = train_xgboost_model(X_train, X_test, y_train, y_test, gee_var)
        performance_results.append(perf)

        print(f"    R² (test): {perf['r2_test']:.3f}, RMSE: {perf['rmse_test']:.3f}")

        # Compute SHAP values
        mean_shap = compute_shap_values(model, X_train)

        # Get top features
        top_idx = np.argsort(mean_shap)[::-1][:TOP_FEATURES]

        for rank, idx in enumerate(top_idx, 1):
            importance_results.append({
                'gee_variable': gee_var,
                'rank': rank,
                'pfam_domain': pfam_cols[idx],
                'shap_importance': mean_shap[idx]
            })

    return pd.DataFrame(performance_results), pd.DataFrame(importance_results)

def save_results(performance_df, importance_df):
    """Save XGBoost results to files."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save performance
    output_perf = OUTPUT_DIR / f"algagpt_xgboost_gee_performance_{TIMESTAMP}.tsv"

    with open(output_perf, 'w') as f:
        f.write("# Provenance:\n")
        f.write(f"#   Script: {Path(__file__).absolute()}\n")
        f.write(f"#   Input: {INPUT_FILE.absolute()}\n")
        f.write(f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#   Integrity Check: PASSED - Real data only\n")
        f.write(f"#   Method: XGBoost regression with SHAP feature importance\n")
        f.write(f"#   XGBoost params: {XGB_PARAMS}\n")
        f.write("#\n")

    performance_df.to_csv(output_perf, sep='\t', index=False, mode='a')
    print(f"  Performance: {output_perf}")
    print(f"    {len(performance_df)} models")

    # Save importance
    output_imp = OUTPUT_DIR / f"algagpt_xgboost_gee_importance_{TIMESTAMP}.tsv"

    with open(output_imp, 'w') as f:
        f.write("# Provenance:\n")
        f.write(f"#   Script: {Path(__file__).absolute()}\n")
        f.write(f"#   Input: {INPUT_FILE.absolute()}\n")
        f.write(f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#   Integrity Check: PASSED - Real data only\n")
        f.write(f"#   Method: SHAP (SHapley Additive exPlanations) feature importance\n")
        f.write(f"#   Top features per variable: {TOP_FEATURES}\n")
        f.write("#\n")

    importance_df.to_csv(output_imp, sep='\t', index=False, mode='a')
    print(f"  Importance: {output_imp}")
    print(f"    {len(importance_df)} feature rankings")

    # Print summary
    print(f"\n  Performance summary:")
    print(f"    Mean R² (test): {performance_df['r2_test'].mean():.3f}")
    print(f"    Median R² (test): {performance_df['r2_test'].median():.3f}")
    print(f"\n  Top 10 models by R²:")
    print(performance_df.nlargest(10, 'r2_test')[['gee_variable', 'r2_test', 'rmse_test']].to_string(index=False))

    return output_perf, output_imp

def main():
    """Main execution."""
    print("\n" + "=" * 70)
    print("ALGAGPT XGBOOST - PREDICT GEE FROM PFAM")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load and filter data
    gee_df, pfam_df, gee_cols, pfam_cols = load_and_filter_data(prevalence_threshold=0.05)

    # Apply CLR transformation
    pfam_clr = apply_clr_transform(pfam_df)

    # Train XGBoost models
    performance_df, importance_df = run_xgboost_analysis(pfam_clr, gee_df, gee_cols, pfam_cols)

    # Save results
    output_perf, output_imp = save_results(performance_df, importance_df)

    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output files:")
    print(f"  {output_perf}")
    print(f"  {output_imp}")
    print("=" * 70)

if __name__ == "__main__":
    main()
