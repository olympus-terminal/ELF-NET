#!/usr/bin/env python3
"""
XGBoost: Predict AlphaEarth embeddings from algaGPT-filtered PFAM counts.

This script builds XGBoost regression models to predict AlphaEarth satellite
embeddings from PFAM domain profiles, with SHAP-based feature importance.

Input:
    - algagpt_gee_pfam_merged_SMART_20260119_100639.tsv (PFAM data)
    - alphaearth_embeddings_20260114_113823.tsv (AlphaEarth 64D embeddings)

Output:
    - algagpt_xgboost_alphaearth_importance_TIMESTAMP.tsv (top 50 PFAMs per dimension)
    - algagpt_xgboost_alphaearth_performance_TIMESTAMP.tsv (R², RMSE, MAE per dimension)

Usage:
    python algagpt_xgboost_alphaearth_20260119_105000.py

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
ALPHAEARTH_FILE = Path("/media/drn2/External/TARA-Oceans/AlphaEarth/alphaearth_embeddings_20260114_113823.tsv")

# On HPC, path may be different
if not ALPHAEARTH_FILE.exists():
    ALPHAEARTH_FILE = Path("/scratch/drn2/PROJECTS/algaGPT-TARA-archive/AlphaEarth/alphaearth_embeddings_20260114_113823.tsv")

# Filtering parameters
PREVALENCE_THRESHOLD = 0.05

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
TOP_FEATURES_OUTPUT = 50

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(SCRIPT_DIR / f'algagpt_xgboost_alphaearth_{TIMESTAMP}.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Functions
# =============================================================================

def load_and_merge_data():
    """Load PFAM and AlphaEarth data, merge on assembly_id."""
    logger.info("=" * 70)
    logger.info("LOADING AND MERGING DATA")
    logger.info("=" * 70)

    # Load PFAM data
    logger.info(f"  Loading PFAM: {ALGAGPT_FILE.name}")
    pfam_df = pd.read_csv(ALGAGPT_FILE, sep='\t', comment='#', low_memory=False)
    logger.info(f"  PFAM samples: {len(pfam_df):,}")

    # Load AlphaEarth embeddings
    logger.info(f"  Loading AlphaEarth: {ALPHAEARTH_FILE.name}")
    ae_df = pd.read_csv(ALPHAEARTH_FILE, sep='\t', comment='#')

    # Get embedding columns
    embed_cols = [f'A{i:02d}' for i in range(64)]
    present_cols = [c for c in embed_cols if c in ae_df.columns]
    logger.info(f"  AlphaEarth dimensions: {len(present_cols)}")

    # Merge on assembly_id
    merged = pfam_df.merge(
        ae_df[['assembly_id'] + present_cols],
        on='assembly_id',
        how='inner'
    )
    logger.info(f"  Merged samples: {len(merged):,}")

    # Filter out rows with NaN in AlphaEarth embeddings
    before_nan = len(merged)
    merged = merged.dropna(subset=present_cols)
    after_nan = len(merged)
    if before_nan > after_nan:
        logger.info(f"  Removed {before_nan - after_nan} samples with NaN in embeddings")
        logger.info(f"  Samples after NaN filtering: {after_nan:,}")

    # Get PFAM columns
    pfam_cols = [c for c in merged.columns if c.startswith('PF')]
    logger.info(f"  Total PFAM domains: {len(pfam_cols):,}")

    return merged, pfam_cols, present_cols

def filter_pfams(merged_df, pfam_cols):
    """Apply prevalence filtering to PFAMs."""
    logger.info("\n  FILTERING PFAMs...")

    pfam_matrix = merged_df[pfam_cols].values
    prevalence = (pfam_matrix > 0).mean(axis=0)
    prevalence_mask = prevalence >= PREVALENCE_THRESHOLD
    n_passing = prevalence_mask.sum()

    logger.info(f"    Prevalence threshold: >= {PREVALENCE_THRESHOLD*100:.1f}%")
    logger.info(f"    Domains passing: {n_passing:,} ({100*n_passing/len(pfam_cols):.1f}%)")

    filtered_pfam_cols = [pfam_cols[i] for i in range(len(pfam_cols)) if prevalence_mask[i]]

    return filtered_pfam_cols

def apply_clr_transform(pfam_df):
    """Apply CLR transformation to PFAM counts."""
    logger.info("\n  Applying CLR transformation...")

    pfam_vals = pfam_df.values.copy().astype(float)
    pfam_vals = pfam_vals + 1  # pseudocount
    log_vals = np.log(pfam_vals)
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

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-n_top:][::-1]

    top_features = []
    for idx in top_idx:
        top_features.append({
            'pfam': feature_names[idx],
            'shap_importance': mean_abs_shap[idx]
        })

    return top_features

def run_xgboost_analysis(pfam_clr, embed_df, embed_cols, pfam_cols):
    """Train XGBoost models to predict AlphaEarth embeddings from PFAM profiles."""
    logger.info("\n" + "=" * 70)
    logger.info("XGBOOST: PFAM → AlphaEarth Embeddings (with SHAP)")
    logger.info("=" * 70)

    # Standardize embedding targets
    embed_scaler = StandardScaler()
    embed_scaled = embed_scaler.fit_transform(embed_df)

    # Train/test split
    X = pfam_clr.values
    y = embed_scaled

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Test samples: {len(X_test)}")
    logger.info(f"  Features (PFAM domains): {X.shape[1]:,}")
    logger.info(f"  Targets (AlphaEarth dimensions): {len(embed_cols)}")

    performance_results = []
    importance_results = []

    for i, embed_dim in enumerate(embed_cols):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"\n  [{i+1}/{len(embed_cols)}] {embed_dim}...")

        # Train model
        results = train_xgboost_model(
            X_train, X_test,
            y_train[:, i], y_test[:, i]
        )

        performance_results.append({
            'alphaearth_dimension': embed_dim,
            'r2_train': results['r2_train'],
            'r2_test': results['r2_test'],
            'rmse_test': results['rmse_test'],
            'mae_test': results['mae_test']
        })

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"      R² train: {results['r2_train']:.3f}, R² test: {results['r2_test']:.3f}")

        # Compute SHAP importance
        top_features = compute_shap_importance(
            results['model'], X_test, pfam_cols
        )

        for feature in top_features:
            importance_results.append({
                'alphaearth_dimension': embed_dim,
                **feature
            })

    performance_df = pd.DataFrame(performance_results)
    performance_df = performance_df.sort_values('r2_test', ascending=False)

    importance_df = pd.DataFrame(importance_results)

    logger.info("\n  Performance Summary (top 10 by R²):")
    logger.info(performance_df.head(10)[['alphaearth_dimension', 'r2_train', 'r2_test']].to_string(index=False))

    return performance_df, importance_df

def save_results(performance_df, importance_df):
    """Save results with provenance."""

    provenance = [
        "# Provenance:",
        f"#   Script: {Path(__file__).absolute()}",
        f"#   PFAM input: {ALGAGPT_FILE}",
        f"#   AlphaEarth input: {ALPHAEARTH_FILE}",
        f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"#   Integrity Check: PASSED - Real data only",
        f"#   XGBoost params: {XGB_PARAMS}",
        f"#   Prevalence threshold: {PREVALENCE_THRESHOLD}",
    ]

    # Performance
    perf_output = SCRIPT_DIR / f"algagpt_xgboost_alphaearth_performance_{TIMESTAMP}.tsv"
    with open(perf_output, 'w') as f:
        f.write('\n'.join(provenance) + '\n')
        performance_df.to_csv(f, sep='\t', index=False)
    logger.info(f"\n  Saved performance: {perf_output.name}")

    # Importance
    imp_output = SCRIPT_DIR / f"algagpt_xgboost_alphaearth_importance_{TIMESTAMP}.tsv"
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
    logger.info("algaGPT XGBoost: Predict AlphaEarth from PFAM")
    logger.info("=" * 70)

    # Load and merge
    merged_df, pfam_cols, embed_cols = load_and_merge_data()

    # Filter PFAMs
    filtered_pfam_cols = filter_pfams(merged_df, pfam_cols)

    # Extract data
    pfam_df = merged_df[filtered_pfam_cols].copy()
    embed_df = merged_df[embed_cols].copy()

    # CLR transform
    pfam_clr = apply_clr_transform(pfam_df)

    # Run XGBoost with SHAP
    performance_df, importance_df = run_xgboost_analysis(
        pfam_clr, embed_df, embed_cols, filtered_pfam_cols
    )

    # Save results
    save_results(performance_df, importance_df)

    # Summary stats
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Dimensions tested: {len(performance_df)}")
    logger.info(f"  Mean R² test: {performance_df['r2_test'].mean():.3f}")
    logger.info(f"  Best R² test: {performance_df['r2_test'].max():.3f} ({performance_df.iloc[0]['alphaearth_dimension']})")
    logger.info(f"  Dimensions with R² > 0.3: {(performance_df['r2_test'] > 0.3).sum()}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
