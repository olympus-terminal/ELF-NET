#!/usr/bin/env python3
"""
Phase 8: XGBoost Modeling with SHAP Interpretation (Full PFAM Domain Set)

This script builds XGBoost models for:
  1. Forward modeling: Environment → PFAM abundances
  2. Reverse modeling: PFAM profile → Environmental conditions

Key improvements over Phase 4/5:
  - Uses ALL PFAM domains (not just top 100)
  - Applies documented prevalence/variance filtering
  - Includes SHAP (SHapley Additive exPlanations) for interpretability
  - Documents filtering methodology for reproducibility

Filtering Methodology:
======================
PFAM domains are filtered based on PREVALENCE (fraction of samples where
domain is detected, i.e., count > 0). This is standard practice in
metagenomics to:
  - Remove rare/singleton domains with insufficient statistical power
  - Reduce noise from sequencing artifacts or annotation errors
  - Focus on ecologically meaningful domains present across samples

Threshold Options:
  - Conservative (10%): Domain in >= 205/2049 samples → ~13,862 domains
  - Moderate (5%): Domain in >= 102/2049 samples → ~17,232 domains
  - Inclusive (1%): Domain in >= 20/2049 samples → ~20,709 domains

Default: 5% prevalence threshold (moderate filtering)

Provenance:
  Input: /media/drn/External1/TARA-Oceans/03_analyses/env_pfam_manifold/data/processed_env_pfam_*.parquet
  Output: reports/xgboost_shap_*.csv, figures/shap_*.png
  Date: 2026-01-14
  XGBoost Version: 3.x
  SHAP Version: 0.50.x

"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import json
import sys

# Data integrity enforcement (CLAUDE.md requirement)
sys.path.insert(0, str(Path(__file__).parent))
from DataIntegrityGuard import enforce_data_integrity
enforce_data_integrity()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
BASE_DIR = Path("/media/drn/External1/TARA-Oceans/03_analyses/env_pfam_manifold")
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Filtering parameters
PREVALENCE_THRESHOLD = 0.05  # 5% - domain must be present in >= 5% of samples
MIN_VARIANCE = 0.0           # Additional variance filter (0 = disabled)

# XGBoost parameters
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,        # L1 regularization
    'reg_lambda': 1.0,       # L2 regularization
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

# SHAP parameters
SHAP_SAMPLE_SIZE = 500  # Number of samples for SHAP computation (for speed)
TOP_FEATURES_PLOT = 30  # Number of top features to show in plots

def load_and_filter_data(prevalence_threshold=PREVALENCE_THRESHOLD):
    """
    Load data and apply prevalence-based filtering to PFAM domains.

    Filtering Rationale:
    - Rare domains (low prevalence) have high variance in estimates
    - Cannot reliably model domains present in only a few samples
    - Prevalence filtering is standard in microbiome/metagenomics analysis

    Returns:
        env_df: Environmental variables (n_samples x n_env_vars)
        pfam_df: Filtered PFAM abundances (n_samples x n_filtered_pfam)
        filtering_stats: Dictionary with filtering statistics
    """
    print("=" * 70)
    print("LOADING AND FILTERING DATA")
    print("=" * 70)

    # Find most recent processed file
    parquet_files = sorted(DATA_DIR.glob("processed_env_pfam_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No processed data found. Run Phase 1 first.")

    input_file = parquet_files[-1]
    print(f"  Input: {input_file.name}")

    df = pd.read_parquet(input_file)
    print(f"  Total samples: {len(df):,}")

    # Identify raw PFAM columns (exclude _norm suffix)
    all_pfam_cols = [c for c in df.columns if c.startswith('PF') and not c.endswith('_norm')]
    print(f"  Total raw PFAM domains: {len(all_pfam_cols):,}")

    # Extract PFAM matrix
    pfam_raw = df[all_pfam_cols].values

    # Calculate prevalence for each domain
    prevalence = (pfam_raw > 0).mean(axis=0)

    # Apply prevalence filter
    prevalence_mask = prevalence >= prevalence_threshold
    n_passing = prevalence_mask.sum()

    print(f"\n  FILTERING METHODOLOGY:")
    print(f"    Criterion: Prevalence >= {prevalence_threshold*100:.1f}%")
    print(f"    Interpretation: Domain detected in >= {int(prevalence_threshold * len(df))} samples")
    print(f"    Domains passing filter: {n_passing:,} ({100*n_passing/len(all_pfam_cols):.1f}%)")
    print(f"    Domains removed: {len(all_pfam_cols) - n_passing:,}")

    # Get filtered column names
    filtered_pfam_cols = [all_pfam_cols[i] for i in range(len(all_pfam_cols)) if prevalence_mask[i]]

    # Environmental columns (numeric only, exclude metadata)
    env_numeric_cols = [
        'air_temp_mean_c', 'air_temp_max_c', 'air_temp_min_c', 'air_temp_range_c',
        'precip_mean_mm', 'solar_rad_mj_m2', 'elevation_m', 'bathymetry_m',
        'distance_to_coast_km', 'sst_mean_c', 'sst_max_c', 'sst_min_c', 'sst_range_c',
        'chl_mean_mg_m3', 'chl_max_mg_m3', 'chl_min_mg_m3', 'nflh_mean', 'poc_mean_mg_m3',
        'modis_sst_mean_c', 'rrs_412', 'rrs_443', 'rrs_469', 'rrs_488',
        'rrs_531', 'rrs_547', 'rrs_555', 'rrs_645', 'rrs_667', 'rrs_678'
    ]

    # Keep only columns that exist
    env_cols = [c for c in env_numeric_cols if c in df.columns]
    print(f"  Environmental variables: {len(env_cols)}")

    # Extract data
    env_df = df[env_cols].copy()
    pfam_df = df[filtered_pfam_cols].copy()

    # Handle missing values in environmental data
    env_df = env_df.fillna(env_df.median())

    # Store filtering statistics
    filtering_stats = {
        'input_file': str(input_file),
        'n_samples': len(df),
        'n_pfam_total': len(all_pfam_cols),
        'n_pfam_filtered': n_passing,
        'prevalence_threshold': prevalence_threshold,
        'min_samples_required': int(prevalence_threshold * len(df)),
        'n_env_vars': len(env_cols),
        'env_vars': env_cols,
        'prevalence_distribution': {
            'min': float(prevalence.min()),
            'max': float(prevalence.max()),
            'mean': float(prevalence.mean()),
            'median': float(np.median(prevalence)),
        }
    }

    return env_df, pfam_df, filtering_stats

def apply_clr_transform(pfam_df):
    """
    Apply Centered Log-Ratio (CLR) transformation to PFAM counts.

    CLR is appropriate for compositional data (counts that sum to a total).
    CLR(x) = log(x / geometric_mean(x)) for each sample.

    Handles zeros by adding a pseudocount of 1.
    """
    print("\n  Applying CLR transformation...")

    pfam_vals = pfam_df.values.copy().astype(float)

    # Add pseudocount for zeros
    pfam_vals = pfam_vals + 1

    # Log transform
    log_vals = np.log(pfam_vals)

    # Subtract geometric mean (mean of log values) per sample
    geo_mean = log_vals.mean(axis=1, keepdims=True)
    clr_vals = log_vals - geo_mean

    print(f"    CLR range: [{clr_vals.min():.3f}, {clr_vals.max():.3f}]")

    return pd.DataFrame(clr_vals, columns=pfam_df.columns, index=pfam_df.index)

def train_xgboost_model(X_train, X_test, y_train, y_test, feature_names, target_name):
    """Train XGBoost model and return results with predictions."""

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    results = {
        'target': target_name,
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'mae_test': mean_absolute_error(y_test, y_pred_test),
        'model': model
    }

    return results

def compute_shap_values(model, X, feature_names, sample_size=SHAP_SAMPLE_SIZE):
    """
    Compute SHAP values for model interpretation.

    SHAP (SHapley Additive exPlanations) provides:
    - Feature importance based on game theory
    - Direction of feature effects (positive/negative)
    - Interaction effects between features
    """
    # Subsample for computational efficiency
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    return shap_values, mean_abs_shap, X_sample

def run_reverse_modeling_with_shap(pfam_clr, env_df, env_cols):
    """
    Reverse modeling: Predict environmental conditions from PFAM profile.

    This answers: "Can we infer habitat characteristics from genomic signatures?"
    """
    print("\n" + "=" * 70)
    print("REVERSE MODELING: PFAM → Environment (with SHAP)")
    print("=" * 70)

    # Standardize environmental targets
    env_scaler = StandardScaler()
    env_scaled = env_scaler.fit_transform(env_df)

    # Train/test split
    X = pfam_clr.values
    feature_names = pfam_clr.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, env_scaled, test_size=0.2, random_state=42
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features (PFAM domains): {X.shape[1]:,}")
    print(f"  Targets (Env vars): {len(env_cols)}")

    results = []
    shap_results = {}

    for i, env_var in enumerate(env_cols):
        print(f"\n  [{i+1}/{len(env_cols)}] {env_var}...")

        # Train model
        model_results = train_xgboost_model(
            X_train, X_test,
            y_train[:, i], y_test[:, i],
            feature_names, env_var
        )

        results.append({
            'env_var': env_var,
            'r2_train': model_results['r2_train'],
            'r2_test': model_results['r2_test'],
            'rmse_test': model_results['rmse_test'],
            'mae_test': model_results['mae_test']
        })

        print(f"      R² train: {model_results['r2_train']:.3f}, R² test: {model_results['r2_test']:.3f}")

        # Compute SHAP for top-performing models
        if model_results['r2_test'] > 0.2:
            print(f"      Computing SHAP values...")
            shap_vals, mean_shap, X_shap = compute_shap_values(
                model_results['model'], X_test, feature_names
            )

            # Get top features by SHAP importance
            top_idx = np.argsort(mean_shap)[-TOP_FEATURES_PLOT:]
            shap_results[env_var] = {
                'shap_values': shap_vals,
                'mean_abs_shap': mean_shap,
                'top_features': [feature_names[j] for j in top_idx],
                'top_shap': mean_shap[top_idx],
                'X_shap': X_shap,
                'model': model_results['model']
            }

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('r2_test', ascending=False)

    print("\n  Results Summary (sorted by R²):")
    print(results_df[['env_var', 'r2_train', 'r2_test', 'rmse_test']].to_string(index=False))

    return results_df, shap_results, feature_names

def run_forward_modeling_with_shap(env_df, pfam_clr, env_cols, pfam_cols, n_targets=50):
    """
    Forward modeling: Predict PFAM abundances from environment.

    This answers: "Which environmental factors drive domain abundances?"

    For computational efficiency, we model the top n_targets most variable PFAM domains.
    """
    print("\n" + "=" * 70)
    print(f"FORWARD MODELING: Environment → PFAM (top {n_targets} targets, with SHAP)")
    print("=" * 70)

    # Standardize environmental features
    env_scaler = StandardScaler()
    X = env_scaler.fit_transform(env_df)

    # Select top variable PFAM domains as targets
    pfam_vars = pfam_clr.var().sort_values(ascending=False)
    top_pfam_cols = pfam_vars.head(n_targets).index.tolist()
    y = pfam_clr[top_pfam_cols].values

    print(f"  Features (Env vars): {X.shape[1]}")
    print(f"  Targets (top {n_targets} variable PFAM): {y.shape[1]}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []
    shap_results = {}

    for i, pfam_name in enumerate(top_pfam_cols):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_targets}] {pfam_name}...")

        model_results = train_xgboost_model(
            X_train, X_test,
            y_train[:, i], y_test[:, i],
            env_cols, pfam_name
        )

        results.append({
            'pfam': pfam_name,
            'r2_train': model_results['r2_train'],
            'r2_test': model_results['r2_test'],
            'rmse_test': model_results['rmse_test'],
            'mae_test': model_results['mae_test']
        })

        # Compute SHAP for well-predicted targets
        if model_results['r2_test'] > 0.3 and i < 10:  # Limit SHAP computation
            shap_vals, mean_shap, X_shap = compute_shap_values(
                model_results['model'], X_test, env_cols
            )
            shap_results[pfam_name] = {
                'shap_values': shap_vals,
                'mean_abs_shap': mean_shap,
                'top_features': [env_cols[j] for j in np.argsort(mean_shap)[-10:]],
                'X_shap': X_shap,
                'model': model_results['model']
            }

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('r2_test', ascending=False)

    print("\n  Top 20 Results (sorted by R²):")
    print(results_df.head(20)[['pfam', 'r2_train', 'r2_test', 'rmse_test']].to_string(index=False))

    return results_df, shap_results

def plot_shap_summary(shap_results, feature_names, output_prefix, direction='reverse'):
    """Create SHAP summary plots for top-performing models."""
    print("\n" + "=" * 70)
    print("GENERATING SHAP PLOTS")
    print("=" * 70)

    if not shap_results:
        print("  No models with R² > threshold for SHAP analysis.")
        return

    # Aggregate feature importance across all targets
    n_features = len(feature_names)
    agg_importance = np.zeros(n_features)

    for target, data in shap_results.items():
        agg_importance += data['mean_abs_shap']

    agg_importance /= len(shap_results)

    # Plot aggregated importance
    top_idx = np.argsort(agg_importance)[-TOP_FEATURES_PLOT:]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top_idx))

    ax.barh(y_pos, agg_importance[top_idx], color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i].split('.')[0] for i in top_idx], fontsize=8)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'Aggregated Feature Importance ({direction.upper()} modeling)\n'
                 f'Averaged across {len(shap_results)} well-predicted targets')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{output_prefix}_aggregated_{TIMESTAMP}.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_prefix}_aggregated_{TIMESTAMP}.png")

    # Plot individual SHAP beeswarm for best target
    best_target = max(shap_results.keys(),
                      key=lambda k: shap_results[k]['mean_abs_shap'].max())

    fig, ax = plt.subplots(figsize=(10, 8))
    shap_data = shap_results[best_target]

    # Get top features for this target
    top_feat_idx = np.argsort(shap_data['mean_abs_shap'])[-20:]

    # Create simplified beeswarm-style plot
    for j, feat_idx in enumerate(top_feat_idx):
        shap_vals = shap_data['shap_values'][:, feat_idx]
        x_vals = np.random.normal(j, 0.1, len(shap_vals))
        ax.scatter(shap_vals, x_vals, alpha=0.3, s=5)

    ax.set_yticks(range(len(top_feat_idx)))
    ax.set_yticklabels([feature_names[i].split('.')[0] for i in top_feat_idx], fontsize=8)
    ax.set_xlabel('SHAP value')
    ax.set_title(f'SHAP values for {best_target}')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{output_prefix}_beeswarm_{best_target.split('.')[0]}_{TIMESTAMP}.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_prefix}_beeswarm_{best_target.split('.')[0]}_{TIMESTAMP}.png")

def save_results(reverse_df, forward_df, filtering_stats, shap_reverse, shap_forward, pfam_cols, env_cols):
    """Save all results with full provenance."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save reverse modeling results
    reverse_df.to_csv(REPORTS_DIR / f"xgboost_reverse_full_{TIMESTAMP}.csv", index=False)
    print(f"  Saved: xgboost_reverse_full_{TIMESTAMP}.csv")

    # Save forward modeling results
    forward_df.to_csv(REPORTS_DIR / f"xgboost_forward_full_{TIMESTAMP}.csv", index=False)
    print(f"  Saved: xgboost_forward_full_{TIMESTAMP}.csv")

    # Save SHAP feature importances
    if shap_reverse:
        shap_df = []
        for env_var, data in shap_reverse.items():
            for i, (feat, shap_val) in enumerate(zip(pfam_cols, data['mean_abs_shap'])):
                shap_df.append({
                    'target': env_var,
                    'feature': feat,
                    'mean_abs_shap': shap_val,
                    'rank': i
                })
        pd.DataFrame(shap_df).to_csv(
            REPORTS_DIR / f"shap_reverse_importance_{TIMESTAMP}.csv", index=False
        )
        print(f"  Saved: shap_reverse_importance_{TIMESTAMP}.csv")

    # Save comprehensive summary
    summary = {
        'timestamp': TIMESTAMP,
        'xgboost_version': xgb.__version__,
        'shap_version': shap.__version__,

        'filtering': filtering_stats,

        'xgb_params': XGB_PARAMS,

        'reverse_modeling': {
            'description': 'Predicting environmental conditions from PFAM profile',
            'n_targets': len(reverse_df),
            'n_features': filtering_stats['n_pfam_filtered'],
            'mean_r2_test': float(reverse_df['r2_test'].mean()),
            'max_r2_test': float(reverse_df['r2_test'].max()),
            'best_target': reverse_df.iloc[0]['env_var'],
            'targets_r2_above_0.3': int((reverse_df['r2_test'] > 0.3).sum()),
            'targets_r2_above_0.2': int((reverse_df['r2_test'] > 0.2).sum()),
        },

        'forward_modeling': {
            'description': 'Predicting PFAM abundances from environmental variables',
            'n_targets': len(forward_df),
            'n_features': len(env_cols),
            'mean_r2_test': float(forward_df['r2_test'].mean()),
            'max_r2_test': float(forward_df['r2_test'].max()),
            'best_target': forward_df.iloc[0]['pfam'],
            'targets_r2_above_0.3': int((forward_df['r2_test'] > 0.3).sum()),
        }
    }

    with open(REPORTS_DIR / f"xgboost_shap_summary_{TIMESTAMP}.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: xgboost_shap_summary_{TIMESTAMP}.json")

def main():
    print("=" * 70)
    print("PHASE 8: XGBOOST + SHAP ANALYSIS (FULL PFAM DOMAIN SET)")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"XGBoost version: {xgb.__version__}")
    print(f"SHAP version: {shap.__version__}")
    print("=" * 70)

    # Load and filter data
    env_df, pfam_df, filtering_stats = load_and_filter_data(
        prevalence_threshold=PREVALENCE_THRESHOLD
    )

    # Apply CLR transformation
    pfam_clr = apply_clr_transform(pfam_df)

    env_cols = env_df.columns.tolist()
    pfam_cols = pfam_clr.columns.tolist()

    # Run reverse modeling (PFAM → Environment)
    reverse_df, shap_reverse, pfam_feature_names = run_reverse_modeling_with_shap(
        pfam_clr, env_df, env_cols
    )

    # Run forward modeling (Environment → PFAM)
    forward_df, shap_forward = run_forward_modeling_with_shap(
        env_df, pfam_clr, env_cols, pfam_cols, n_targets=100
    )

    # Generate SHAP plots
    plot_shap_summary(shap_reverse, pfam_feature_names, 'shap_reverse', 'reverse')
    plot_shap_summary(shap_forward, env_cols, 'shap_forward', 'forward')

    # Save results
    save_results(reverse_df, forward_df, filtering_stats,
                 shap_reverse, shap_forward, pfam_cols, env_cols)

    print("\n" + "=" * 70)
    print("PHASE 8 COMPLETE")
    print("=" * 70)

    return reverse_df, forward_df

if __name__ == "__main__":
    reverse_df, forward_df = main()
