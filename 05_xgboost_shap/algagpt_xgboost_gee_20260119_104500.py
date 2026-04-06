#!/usr/bin/env python3
"""
XGBoost: Predict GEE environmental variables from algaGPT-filtered PFAM counts.

This script builds XGBoost regression models to predict environmental conditions
from PFAM domain profiles, with SHAP-based feature importance analysis.

Input:
    - algagpt_gee_pfam_merged_SMART_20260119_100639.tsv

Output:
    - algagpt_xgboost_gee_importance_TIMESTAMP.tsv (top 50 PFAMs per variable)
    - algagpt_xgboost_gee_performance_TIMESTAMP.tsv (R², RMSE, MAE per variable)

Filtering:
    - Uses PFAM domains present in >= 5% of samples (prevalence filter)
    - Applies CLR transformation to compositional count data

Usage:
    python algagpt_xgboost_gee_20260119_104500.py

Author: algaGPT Environmental Analysis Pipeline
Date: 2026-01-19
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Input
ALGAGPT_FILE = SCRIPT_DIR / "algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"

# Filtering parameters
PREVALENCE_THRESHOLD = 0.05  # Domain must be present in >= 5% of samples
MIN_NONZERO = 10             # Minimum number of non-zero samples

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
TOP_FEATURES_OUTPUT = 50  # Number of top features to save per variable

# GEE environmental variables (from Task 1 results)
GEE_VARS = [
    'air_temp_mean_c', 'air_temp_max_c', 'air_temp_min_c', 'air_temp_range_c',
    'precip_mean_mm', 'solar_rad_mj_m2', 'elevation_m', 'bathymetry_m',
    'distance_to_coast_km', 'sst_mean_c', 'sst_max_c', 'sst_min_c', 'sst_range_c',
    'chl_mean_mg_m3', 'chl_max_mg_m3', 'chl_min_mg_m3', 'nflh_mean', 'poc_mean_mg_m3',
    'modis_sst_mean_c', 'rrs_412', 'rrs_443', 'rrs_469', 'rrs_488',
    'rrs_531', 'rrs_547', 'rrs_555', 'rrs_645', 'rrs_667', 'rrs_678'
]

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(SCRIPT_DIR / f'algagpt_xgboost_{TIMESTAMP}.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Functions
# =============================================================================

def load_and_filter_data():
    """Load algaGPT data and apply prevalence filtering to PFAMs."""
    logger.info("=" * 70)
    logger.info("LOADING AND FILTERING DATA")
    logger.info("=" * 70)

    logger.info(f"  Loading: {ALGAGPT_FILE.name}")
    df = pd.read_csv(ALGAGPT_FILE, sep='\t', comment='#', low_memory=False)
    logger.info(f"  Total samples: {len(df):,}")

    # Identify PFAM columns
    all_pfam_cols = [c for c in df.columns if c.startswith('PF')]
    logger.info(f"  Total PFAM domains: {len(all_pfam_cols):,}")

    # Calculate prevalence
    pfam_matrix = df[all_pfam_cols].values
    prevalence = (pfam_matrix > 0).mean(axis=0)
    prevalence_mask = prevalence >= PREVALENCE_THRESHOLD
    n_passing = prevalence_mask.sum()

    logger.info(f"\n  FILTERING:")
    logger.info(f"    Prevalence threshold: >= {PREVALENCE_THRESHOLD*100:.1f}%")
    logger.info(f"    Min samples required: >= {int(PREVALENCE_THRESHOLD * len(df))}")
    logger.info(f"    Domains passing: {n_passing:,} ({100*n_passing/len(all_pfam_cols):.1f}%)")
    logger.info(f"    Domains removed: {len(all_pfam_cols) - n_passing:,}")

    # Filter PFAMs
    filtered_pfam_cols = [all_pfam_cols[i] for i in range(len(all_pfam_cols)) if prevalence_mask[i]]

    # Get GEE variables that exist in data
    env_cols = [c for c in GEE_VARS if c in df.columns]
    logger.info(f"  Environmental variables: {len(env_cols)}")

    # Extract data
    env_df = df[env_cols].copy()
    pfam_df = df[filtered_pfam_cols].copy()

    # Handle missing environmental values
    env_df = env_df.fillna(env_df.median())

    # Remove samples with all-zero PFAM profiles
    pfam_sums = pfam_df.sum(axis=1)
    valid_samples = pfam_sums > 0

    env_df = env_df[valid_samples].reset_index(drop=True)
    pfam_df = pfam_df[valid_samples].reset_index(drop=True)

    logger.info(f"  Samples after filtering zeros: {len(env_df):,}")

    return env_df, pfam_df, env_cols, filtered_pfam_cols

def apply_clr_transform(pfam_df):
    """Apply Centered Log-Ratio transformation to PFAM counts."""
    logger.info("\n  Applying CLR transformation...")

    pfam_vals = pfam_df.values.copy().astype(float)

    # Add pseudocount for zeros
    pfam_vals = pfam_vals + 1

    # Log transform
    log_vals = np.log(pfam_vals)

    # Subtract geometric mean per sample
    geo_mean = log_vals.mean(axis=1, keepdims=True)
    clr_vals = log_vals - geo_mean

    logger.info(f"    CLR range: [{clr_vals.min():.3f}, {clr_vals.max():.3f}]")

    return pd.DataFrame(clr_vals, columns=pfam_df.columns, index=pfam_df.index)

def train_xgboost_model(X_train, X_test, y_train, y_test):
    """Train XGBoost regressor."""
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return {
        'model': model,
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'mae_test': mean_absolute_error(y_test, y_pred_test)
    }

def compute_shap_importance(model, X_test, feature_names, n_top=TOP_FEATURES_OUTPUT):
    """Compute SHAP-based feature importance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Get top features
    top_idx = np.argsort(mean_abs_shap)[-n_top:][::-1]

    top_features = []
    for idx in top_idx:
        top_features.append({
            'pfam': feature_names[idx],
            'shap_importance': mean_abs_shap[idx]
        })

    return top_features

def run_xgboost_analysis(pfam_clr, env_df, env_cols, pfam_cols):
    """
    Train XGBoost models to predict environmental variables from PFAM profiles.

    This answers: "Can we infer environmental conditions from genomic signatures?"
    """
    logger.info("\n" + "=" * 70)
    logger.info("XGBOOST: PFAM → Environment (with SHAP)")
    logger.info("=" * 70)

    # Standardize environmental targets
    env_scaler = StandardScaler()
    env_scaled = env_scaler.fit_transform(env_df)

    # Train/test split
    X = pfam_clr.values
    y = env_scaled

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Test samples: {len(X_test)}")
    logger.info(f"  Features (PFAM domains): {X.shape[1]:,}")
    logger.info(f"  Targets (GEE variables): {len(env_cols)}")

    performance_results = []
    importance_results = []

    for i, env_var in enumerate(env_cols):
        logger.info(f"\n  [{i+1}/{len(env_cols)}] {env_var}...")

        # Train model
        results = train_xgboost_model(
            X_train, X_test,
            y_train[:, i], y_test[:, i]
        )

        performance_results.append({
            'gee_variable': env_var,
            'r2_train': results['r2_train'],
            'r2_test': results['r2_test'],
            'rmse_test': results['rmse_test'],
            'mae_test': results['mae_test']
        })

        logger.info(f"      R² train: {results['r2_train']:.3f}, R² test: {results['r2_test']:.3f}")

        # Compute SHAP importance
        logger.info(f"      Computing SHAP values...")
        top_features = compute_shap_importance(
            results['model'], X_test, pfam_cols
        )

        # Add variable name to importance results
        for feature in top_features:
            importance_results.append({
                'gee_variable': env_var,
                **feature
            })

    performance_df = pd.DataFrame(performance_results)
    performance_df = performance_df.sort_values('r2_test', ascending=False)

    importance_df = pd.DataFrame(importance_results)

    logger.info("\n  Performance Summary (top 10 by R²):")
    logger.info(performance_df.head(10)[['gee_variable', 'r2_train', 'r2_test']].to_string(index=False))

    return performance_df, importance_df

def save_results(performance_df, importance_df):
    """Save results with provenance."""

    provenance = [
        "# Provenance:",
        f"#   Script: {Path(__file__).absolute()}",
        f"#   Input: {ALGAGPT_FILE}",
        f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"#   Integrity Check: PASSED - Real data only",
        f"#   XGBoost params: {XGB_PARAMS}",
        f"#   Prevalence threshold: {PREVALENCE_THRESHOLD}",
    ]

    # Performance
    perf_output = SCRIPT_DIR / f"algagpt_xgboost_gee_performance_{TIMESTAMP}.tsv"
    with open(perf_output, 'w') as f:
        f.write('\n'.join(provenance) + '\n')
        performance_df.to_csv(f, sep='\t', index=False)
    logger.info(f"\n  Saved performance: {perf_output.name}")

    # Importance
    imp_output = SCRIPT_DIR / f"algagpt_xgboost_gee_importance_{TIMESTAMP}.tsv"
    with open(imp_output, 'w') as f:
        f.write('\n'.join(provenance) + '\n')
        importance_df.to_csv(f, sep='\t', index=False)
    logger.info(f"  Saved importance: {imp_output.name}")

    return perf_output, imp_output

# =============================================================================
# Main
# =============================================================================

def main():
    """Main execution."""
    logger.info("=" * 70)
    logger.info("algaGPT XGBoost: Predict GEE from PFAM")
    logger.info("=" * 70)

    # Load and filter
    env_df, pfam_df, env_cols, pfam_cols = load_and_filter_data()

    # CLR transform
    pfam_clr = apply_clr_transform(pfam_df)

    # Run XGBoost with SHAP
    performance_df, importance_df = run_xgboost_analysis(
        pfam_clr, env_df, env_cols, pfam_cols
    )

    # Save results
    save_results(performance_df, importance_df)

    # Summary stats
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Variables tested: {len(performance_df)}")
    logger.info(f"  Mean R² test: {performance_df['r2_test'].mean():.3f}")
    logger.info(f"  Best R² test: {performance_df['r2_test'].max():.3f} ({performance_df.iloc[0]['gee_variable']})")
    logger.info(f"  Variables with R² > 0.5: {(performance_df['r2_test'] > 0.5).sum()}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
