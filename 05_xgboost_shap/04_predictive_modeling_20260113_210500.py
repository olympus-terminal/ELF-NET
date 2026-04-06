#!/usr/bin/env python3
"""
Phase 4: Predictive Modeling (Environment → PFAM).

This script builds models to predict PFAM domain abundances from
environmental variables using XGBoost and Random Forest.

Provenance:
  Input: Processed data from Phase 1
  Date: 2026-01-13
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import json

# Data integrity enforcement (CLAUDE.md requirement)
from DataIntegrityGuard import enforce_data_integrity
enforce_data_integrity()

# Paths
BASE_DIR = Path("/media/drn/External1/TARA-Oceans/03_analyses/env_pfam_manifold")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def load_processed_data():
    """Load processed data from Phase 1."""
    print("=" * 70)
    print("LOADING PROCESSED DATA")
    print("=" * 70)

    # Find the most recent processed files
    env_files = sorted(DATA_DIR.glob("env_matrix_*.npy"))
    pfam_files = sorted(DATA_DIR.glob("pfam_matrix_*.npy"))

    if not env_files or not pfam_files:
        raise FileNotFoundError("Processed data not found. Run Phase 1 first.")

    # Load most recent
    env_matrix = np.load(env_files[-1])
    pfam_matrix = np.load(pfam_files[-1])

    # Load column names
    env_cols_file = sorted(DATA_DIR.glob("env_columns_*.txt"))[-1]
    pfam_cols_file = sorted(DATA_DIR.glob("pfam_columns_*.txt"))[-1]

    with open(env_cols_file) as f:
        env_cols = [line.strip() for line in f]

    with open(pfam_cols_file) as f:
        pfam_cols = [line.strip() for line in f]

    print(f"  Environmental matrix: {env_matrix.shape}")
    print(f"  PFAM matrix: {pfam_matrix.shape}")
    print(f"  Environmental variables: {len(env_cols)}")
    print(f"  PFAM domains: {len(pfam_cols)}")

    return env_matrix, pfam_matrix, env_cols, pfam_cols

def select_top_pfam_targets(pfam_matrix, pfam_cols, top_n=100):
    """Select top N most variable PFAM domains as prediction targets."""
    print("\n" + "=" * 70)
    print(f"SELECTING TOP {top_n} PFAM TARGETS")
    print("=" * 70)

    # Calculate variance for each PFAM domain
    variances = np.var(pfam_matrix, axis=0)
    top_indices = np.argsort(variances)[-top_n:]

    pfam_subset = pfam_matrix[:, top_indices]
    pfam_cols_subset = [pfam_cols[i] for i in top_indices]

    print(f"  Selected {top_n} most variable PFAM domains")
    print(f"  Variance range: [{variances[top_indices].min():.4f}, {variances[top_indices].max():.4f}]")

    return pfam_subset, pfam_cols_subset, top_indices

def train_random_forest(X_train, X_test, y_train, y_test, target_name):
    """Train Random Forest for a single target."""
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Predictions
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    # Metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return {
        'model': rf,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test,
        'feature_importance': rf.feature_importances_
    }

def train_xgboost(X_train, X_test, y_train, y_test, target_name):
    """Train XGBoost for a single target."""
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)

    # Predictions
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)

    # Metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return {
        'model': xgb_model,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test,
        'feature_importance': xgb_model.feature_importances_
    }

def run_predictive_models(env_matrix, pfam_subset, pfam_cols_subset, env_cols):
    """Train predictive models for each PFAM target."""
    print("\n" + "=" * 70)
    print("TRAINING PREDICTIVE MODELS (Environment → PFAM)")
    print("=" * 70)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        env_matrix, pfam_subset,
        test_size=0.2, random_state=42
    )

    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Targets: {y_train.shape[1]}")

    rf_results = []
    xgb_results = []

    n_targets = len(pfam_cols_subset)

    for i, pfam_name in enumerate(pfam_cols_subset):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"\n  Processing target {i+1}/{n_targets}: {pfam_name}...")

        # Random Forest
        rf_res = train_random_forest(
            X_train, X_test,
            y_train[:, i], y_test[:, i],
            pfam_name
        )
        rf_results.append({
            'pfam': pfam_name,
            'r2_train': rf_res['r2_train'],
            'r2_test': rf_res['r2_test'],
            'rmse_test': rf_res['rmse_test'],
            'feature_importance': rf_res['feature_importance']
        })

        # XGBoost
        xgb_res = train_xgboost(
            X_train, X_test,
            y_train[:, i], y_test[:, i],
            pfam_name
        )
        xgb_results.append({
            'pfam': pfam_name,
            'r2_train': xgb_res['r2_train'],
            'r2_test': xgb_res['r2_test'],
            'rmse_test': xgb_res['rmse_test'],
            'feature_importance': xgb_res['feature_importance']
        })

    # Convert to DataFrames
    rf_df = pd.DataFrame(rf_results)
    xgb_df = pd.DataFrame(xgb_results)

    print("\n  Random Forest Results (top 10 by R²):")
    print(rf_df.nlargest(10, 'r2_test')[['pfam', 'r2_train', 'r2_test', 'rmse_test']].to_string(index=False))

    print("\n  XGBoost Results (top 10 by R²):")
    print(xgb_df.nlargest(10, 'r2_test')[['pfam', 'r2_train', 'r2_test', 'rmse_test']].to_string(index=False))

    return rf_df, xgb_df, env_cols

def plot_model_comparison(rf_df, xgb_df, output_path):
    """Plot comparison of model performance."""
    print("\n" + "=" * 70)
    print("CREATING MODEL COMPARISON PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # R² distribution comparison
    ax = axes[0]
    ax.hist(rf_df['r2_test'], bins=30, alpha=0.7, label='Random Forest', color='blue')
    ax.hist(xgb_df['r2_test'], bins=30, alpha=0.7, label='XGBoost', color='orange')
    ax.set_xlabel('Test R²')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Test R² Scores')
    ax.legend()

    # R² scatter plot (RF vs XGBoost)
    ax = axes[1]
    ax.scatter(rf_df['r2_test'], xgb_df['r2_test'], alpha=0.5, s=10)
    ax.plot([0, 1], [0, 1], 'r--', label='y=x')
    ax.set_xlabel('Random Forest R²')
    ax.set_ylabel('XGBoost R²')
    ax.set_title('Model Comparison: RF vs XGBoost')
    ax.legend()

    # Top 20 most predictable PFAM domains
    ax = axes[2]
    top_rf = rf_df.nlargest(20, 'r2_test')
    y_pos = range(len(top_rf))
    ax.barh(y_pos, top_rf['r2_test'], color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p.split('.')[0] for p in top_rf['pfam']], fontsize=8)
    ax.set_xlabel('Test R²')
    ax.set_title('Top 20 Most Predictable PFAM Domains (RF)')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def plot_feature_importance(rf_df, env_cols, output_path):
    """Plot aggregated feature importance across all targets."""
    print("\n  Creating feature importance plot...")

    # Aggregate feature importance across all targets
    importance_matrix = np.array([r['feature_importance'] for _, r in rf_df.iterrows()])
    mean_importance = importance_matrix.mean(axis=0)

    # Sort by importance
    sorted_idx = np.argsort(mean_importance)[::-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = range(len(env_cols))
    ax.bar(x_pos, mean_importance[sorted_idx], color='steelblue')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([env_cols[i] for i in sorted_idx], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Environmental Variable')
    ax.set_ylabel('Mean Feature Importance')
    ax.set_title('Aggregated Feature Importance (Random Forest)\nfor Predicting PFAM Abundances')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def save_modeling_results(rf_df, xgb_df, env_cols):
    """Save all modeling results."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save results DataFrames (without numpy arrays)
    rf_df_save = rf_df.drop(columns=['feature_importance'])
    xgb_df_save = xgb_df.drop(columns=['feature_importance'])

    rf_df_save.to_csv(REPORTS_DIR / f"rf_results_{TIMESTAMP}.csv", index=False)
    xgb_df_save.to_csv(REPORTS_DIR / f"xgb_results_{TIMESTAMP}.csv", index=False)

    print(f"  Saved Random Forest results")
    print(f"  Saved XGBoost results")

    # Save summary
    summary = {
        'timestamp': TIMESTAMP,
        'n_targets': int(len(rf_df)),
        'n_features': int(len(env_cols)),
        'rf_mean_r2_test': float(rf_df['r2_test'].mean()),
        'rf_max_r2_test': float(rf_df['r2_test'].max()),
        'rf_targets_r2_above_0.3': int((rf_df['r2_test'] > 0.3).sum()),
        'xgb_mean_r2_test': float(xgb_df['r2_test'].mean()),
        'xgb_max_r2_test': float(xgb_df['r2_test'].max()),
        'xgb_targets_r2_above_0.3': int((xgb_df['r2_test'] > 0.3).sum())
    }

    with open(REPORTS_DIR / f"modeling_summary_{TIMESTAMP}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved summary statistics")

def main():
    print("=" * 70)
    print("PHASE 4: PREDICTIVE MODELING (Environment → PFAM)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load data
    env_matrix, pfam_matrix, env_cols, pfam_cols = load_processed_data()

    # Select top PFAM targets
    pfam_subset, pfam_cols_subset, top_indices = select_top_pfam_targets(
        pfam_matrix, pfam_cols, top_n=100
    )

    # Train models
    rf_df, xgb_df, env_cols = run_predictive_models(
        env_matrix, pfam_subset, pfam_cols_subset, env_cols
    )

    # Plot model comparison
    plot_model_comparison(
        rf_df, xgb_df,
        FIGURES_DIR / f"model_comparison_{TIMESTAMP}.png"
    )

    # Plot feature importance
    plot_feature_importance(
        rf_df, env_cols,
        FIGURES_DIR / f"feature_importance_{TIMESTAMP}.png"
    )

    # Save results
    save_modeling_results(rf_df, xgb_df, env_cols)

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)

    return rf_df, xgb_df

if __name__ == "__main__":
    rf_df, xgb_df = main()
