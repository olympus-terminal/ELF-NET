#!/usr/bin/env python3
"""
Phase 5: Reverse Modeling (PFAM → Environment).

This script builds models to predict environmental conditions from
PFAM domain abundances - i.e., inferring habitat from genomic signatures.

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
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

def reduce_pfam_dimensionality(pfam_matrix, n_components=50):
    """Reduce PFAM dimensionality using PCA for more efficient modeling."""
    print("\n" + "=" * 70)
    print(f"REDUCING PFAM DIMENSIONS TO {n_components} PCs")
    print("=" * 70)

    pca = PCA(n_components=n_components, random_state=42)
    pfam_pca = pca.fit_transform(pfam_matrix)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  PCA components: {n_components}")
    print(f"  Total variance explained: {explained_var*100:.2f}%")

    return pfam_pca, pca

def train_environment_predictor(X_train, X_test, y_train, y_test, env_name, model_type='rf'):
    """Train a model to predict an environmental variable from PFAM features."""

    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    else:  # xgboost
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)

    return {
        'model': model,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test,
        'mae_test': mae_test,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }

def run_reverse_models(pfam_pca, env_matrix, env_cols):
    """Train models to predict each environmental variable from PFAM features."""
    print("\n" + "=" * 70)
    print("TRAINING REVERSE MODELS (PFAM → Environment)")
    print("=" * 70)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        pfam_pca, env_matrix,
        test_size=0.2, random_state=42
    )

    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features (PCs): {X_train.shape[1]}")
    print(f"  Targets (Env vars): {y_train.shape[1]}")

    rf_results = []
    xgb_results = []

    for i, env_name in enumerate(env_cols):
        print(f"\n  Processing: {env_name}...")

        # Random Forest
        rf_res = train_environment_predictor(
            X_train, X_test,
            y_train[:, i], y_test[:, i],
            env_name, model_type='rf'
        )
        rf_results.append({
            'env_var': env_name,
            'r2_train': rf_res['r2_train'],
            'r2_test': rf_res['r2_test'],
            'rmse_test': rf_res['rmse_test'],
            'mae_test': rf_res['mae_test'],
            'y_test': rf_res['y_test'],
            'y_pred_test': rf_res['y_pred_test']
        })

        # XGBoost
        xgb_res = train_environment_predictor(
            X_train, X_test,
            y_train[:, i], y_test[:, i],
            env_name, model_type='xgb'
        )
        xgb_results.append({
            'env_var': env_name,
            'r2_train': xgb_res['r2_train'],
            'r2_test': xgb_res['r2_test'],
            'rmse_test': xgb_res['rmse_test'],
            'mae_test': xgb_res['mae_test']
        })

    # Convert to DataFrames (excluding arrays)
    rf_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_test', 'y_pred_test']}
                          for r in rf_results])
    xgb_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_test', 'y_pred_test']}
                           for r in xgb_results])

    print("\n" + "=" * 70)
    print("REVERSE MODELING RESULTS")
    print("=" * 70)
    print("\n  Random Forest Results (sorted by R²):")
    print(rf_df.sort_values('r2_test', ascending=False)[['env_var', 'r2_train', 'r2_test', 'rmse_test']].to_string(index=False))

    print("\n  XGBoost Results (sorted by R²):")
    print(xgb_df.sort_values('r2_test', ascending=False)[['env_var', 'r2_train', 'r2_test', 'rmse_test']].to_string(index=False))

    return rf_df, xgb_df, rf_results

def plot_reverse_model_results(rf_df, xgb_df, rf_results, env_cols, output_path):
    """Plot reverse modeling results."""
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # R² comparison bar plot
    ax = axes[0, 0]
    x = range(len(env_cols))
    width = 0.35
    rf_r2 = rf_df.sort_values('r2_test', ascending=False)['r2_test'].values
    xgb_r2 = xgb_df.sort_values('r2_test', ascending=False)['r2_test'].values
    env_sorted = rf_df.sort_values('r2_test', ascending=False)['env_var'].values

    ax.barh([i - width/2 for i in x], rf_r2, width, label='Random Forest', color='steelblue')
    ax.barh([i + width/2 for i in x], xgb_r2, width, label='XGBoost', color='darkorange')
    ax.set_yticks(x)
    ax.set_yticklabels(env_sorted, fontsize=8)
    ax.set_xlabel('Test R²')
    ax.set_title('Environmental Variable Predictability from PFAM Profile')
    ax.legend()
    ax.invert_yaxis()

    # RF vs XGBoost scatter
    ax = axes[0, 1]
    ax.scatter(rf_df['r2_test'], xgb_df['r2_test'], s=50, alpha=0.7)
    for i, name in enumerate(rf_df['env_var']):
        if rf_df.iloc[i]['r2_test'] > 0.2 or xgb_df.iloc[i]['r2_test'] > 0.2:
            ax.annotate(name, (rf_df.iloc[i]['r2_test'], xgb_df.iloc[i]['r2_test']),
                       fontsize=7, alpha=0.8)
    ax.plot([0, 0.8], [0, 0.8], 'r--', alpha=0.5)
    ax.set_xlabel('Random Forest R²')
    ax.set_ylabel('XGBoost R²')
    ax.set_title('Model Comparison: RF vs XGBoost')

    # Predicted vs Actual for best-predicted variable
    best_idx = rf_df['r2_test'].idxmax()
    best_result = rf_results[best_idx]
    best_env = rf_df.iloc[best_idx]['env_var']

    ax = axes[1, 0]
    ax.scatter(best_result['y_test'], best_result['y_pred_test'], alpha=0.5, s=10)
    ax.plot([best_result['y_test'].min(), best_result['y_test'].max()],
            [best_result['y_test'].min(), best_result['y_test'].max()], 'r--', label='Perfect')
    ax.set_xlabel(f'Actual {best_env} (standardized)')
    ax.set_ylabel(f'Predicted {best_env}')
    ax.set_title(f'Best Predicted: {best_env} (R²={rf_df.iloc[best_idx]["r2_test"]:.3f})')
    ax.legend()

    # Prediction error distribution
    ax = axes[1, 1]
    errors = best_result['y_test'] - best_result['y_pred_test']
    ax.hist(errors, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', label='Zero error')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Count')
    ax.set_title(f'Prediction Error Distribution: {best_env}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def save_reverse_modeling_results(rf_df, xgb_df):
    """Save reverse modeling results."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    rf_df.to_csv(REPORTS_DIR / f"reverse_rf_results_{TIMESTAMP}.csv", index=False)
    xgb_df.to_csv(REPORTS_DIR / f"reverse_xgb_results_{TIMESTAMP}.csv", index=False)

    print(f"  Saved RF results to reports/")
    print(f"  Saved XGBoost results to reports/")

    # Summary
    summary = {
        'timestamp': TIMESTAMP,
        'n_env_targets': int(len(rf_df)),
        'rf_mean_r2': float(rf_df['r2_test'].mean()),
        'rf_max_r2': float(rf_df['r2_test'].max()),
        'rf_best_var': rf_df.loc[rf_df['r2_test'].idxmax(), 'env_var'],
        'rf_vars_r2_above_0.2': int((rf_df['r2_test'] > 0.2).sum()),
        'xgb_mean_r2': float(xgb_df['r2_test'].mean()),
        'xgb_max_r2': float(xgb_df['r2_test'].max()),
        'xgb_best_var': xgb_df.loc[xgb_df['r2_test'].idxmax(), 'env_var']
    }

    with open(REPORTS_DIR / f"reverse_modeling_summary_{TIMESTAMP}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved summary")

def main():
    print("=" * 70)
    print("PHASE 5: REVERSE MODELING (PFAM → Environment)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load data
    env_matrix, pfam_matrix, env_cols, pfam_cols = load_processed_data()

    # Reduce PFAM dimensions
    pfam_pca, pca_model = reduce_pfam_dimensionality(pfam_matrix, n_components=50)

    # Train reverse models
    rf_df, xgb_df, rf_results = run_reverse_models(pfam_pca, env_matrix, env_cols)

    # Plot results
    plot_reverse_model_results(
        rf_df, xgb_df, rf_results, env_cols,
        FIGURES_DIR / f"reverse_model_results_{TIMESTAMP}.png"
    )

    # Save results
    save_reverse_modeling_results(rf_df, xgb_df)

    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE")
    print("=" * 70)

    return rf_df, xgb_df

if __name__ == "__main__":
    rf_df, xgb_df = main()
