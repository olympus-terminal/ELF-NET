#!/usr/bin/env python3
"""
Phase 8: XGBoost Modeling with SHAP Interpretation for algaGPT.

This script builds XGBoost models for both forward (Env->PFAM) and reverse
(PFAM->Env) modeling, with SHAP-based feature importance analysis.

Key improvements:
- Full PFAM domain set (not just top 100)
- SHAP (SHapley Additive exPlanations) for interpretability
- Documented prevalence filtering
- Publication-quality SHAP visualizations

Provenance:
  Input: data/*.npy from Phase 1
  Output: reports/xgboost_*.csv, figures/shap_*.pdf
  Date: 2026-01-22
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Try imports
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("ERROR: xgboost not installed. This script requires XGBoost.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: shap not installed. SHAP analysis will be skipped.")

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import json
import sys
import joblib

# Data integrity enforcement
sys.path.insert(0, str(Path(__file__).parent))
from DataIntegrityGuard import enforce_data_integrity
enforce_data_integrity()

# ============================================================================
# FIGURE PROTOCOL
# ============================================================================

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 5
mpl.rcParams['axes.linewidth'] = 0.25
mpl.rcParams['xtick.major.width'] = 0.25
mpl.rcParams['ytick.major.width'] = 0.25

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

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
SHAP_SAMPLE_SIZE = 500  # Subsample for SHAP computation
TOP_FEATURES_PLOT = 25  # Features to show in plots

TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_processed_data():
    """Load processed data from Phase 1."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    env_files = sorted(DATA_DIR.glob("env_matrix_*.npy"))
    pfam_files = sorted(DATA_DIR.glob("pfam_matrix_*.npy"))

    if not env_files or not pfam_files:
        raise FileNotFoundError("Processed data not found. Run Phase 1 first.")

    env_matrix = np.load(env_files[-1])
    pfam_matrix = np.load(pfam_files[-1])

    env_cols_file = sorted(DATA_DIR.glob("env_columns_*.txt"))[-1]
    pfam_cols_file = sorted(DATA_DIR.glob("pfam_columns_*.txt"))[-1]

    with open(env_cols_file) as f:
        env_cols = [line.strip() for line in f]

    with open(pfam_cols_file) as f:
        pfam_cols = [line.strip() for line in f]

    print(f"  Env: {env_matrix.shape}, PFAM: {pfam_matrix.shape}")

    return env_matrix, pfam_matrix, env_cols, pfam_cols

def train_xgboost(X_train, X_test, y_train, y_test, target_name):
    """Train XGBoost model."""
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return {
        'target': target_name,
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'model': model
    }

def compute_shap_values(model, X, feature_names, sample_size=SHAP_SAMPLE_SIZE):
    """Compute SHAP values for interpretation."""
    if not SHAP_AVAILABLE:
        return None, None, None

    # Subsample for efficiency
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
        idx = np.arange(len(X))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    return shap_values, mean_abs_shap, X_sample

def run_reverse_modeling_with_shap(pfam_matrix, env_matrix, pfam_cols, env_cols):
    """Reverse modeling: PFAM -> Environment with SHAP."""
    print("\n" + "=" * 70)
    print("REVERSE MODELING: PFAM -> Environment (with SHAP)")
    print("=" * 70)

    # Standardize environment
    scaler = StandardScaler()
    env_scaled = scaler.fit_transform(env_matrix)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        pfam_matrix, env_scaled, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Features: {pfam_matrix.shape[1]:,}, Targets: {len(env_cols)}")

    results = []
    shap_results = {}
    trained_models = {}  # Store trained models for serialization

    for i, env_var in enumerate(env_cols):
        print(f"  [{i+1}/{len(env_cols)}] {env_var}...", end=" ", flush=True)

        # Skip if target has NaN or no variance
        y_tr = y_train[:, i]
        y_te = y_test[:, i]
        if np.isnan(y_tr).any() or np.isnan(y_te).any() or np.std(y_tr) < 1e-10:
            print("SKIPPED (NaN or no variance)")
            results.append({
                'env_var': env_var,
                'r2_train': np.nan,
                'r2_test': np.nan,
                'rmse_test': np.nan
            })
            continue

        res = train_xgboost(X_train, X_test, y_tr, y_te, env_var)
        results.append({
            'env_var': env_var,
            'r2_train': res['r2_train'],
            'r2_test': res['r2_test'],
            'rmse_test': res['rmse_test']
        })

        # Store trained model
        trained_models[env_var] = res['model']

        print(f"R2={res['r2_test']:.3f}")

        # SHAP for well-predicted targets
        if res['r2_test'] > 0.2 and SHAP_AVAILABLE:
            shap_vals, mean_shap, X_shap = compute_shap_values(
                res['model'], X_test, pfam_cols
            )
            if shap_vals is not None:
                shap_results[env_var] = {
                    'shap_values': shap_vals,
                    'mean_abs_shap': mean_shap,
                    'X_shap': X_shap,
                    'r2': res['r2_test']
                }

    results_df = pd.DataFrame(results).sort_values('r2_test', ascending=False)
    return results_df, shap_results, trained_models, scaler

def run_forward_modeling_with_shap(env_matrix, pfam_matrix, env_cols, pfam_cols, n_targets=50):
    """Forward modeling: Environment -> PFAM with SHAP."""
    print("\n" + "=" * 70)
    print(f"FORWARD MODELING: Environment -> PFAM (top {n_targets} targets)")
    print("=" * 70)

    # Standardize environment
    scaler = StandardScaler()
    X = scaler.fit_transform(env_matrix)

    # Select most variable PFAMs
    pfam_var = np.var(pfam_matrix, axis=0)
    top_idx = np.argsort(pfam_var)[-n_targets:]
    y = pfam_matrix[:, top_idx]
    target_names = [pfam_cols[i] for i in top_idx]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Features: {X.shape[1]}, Targets: {n_targets}")

    results = []
    shap_results = {}
    trained_models = {}  # Store trained models for serialization

    for i, pfam_name in enumerate(target_names):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_targets}]...", end=" ", flush=True)

        res = train_xgboost(X_train, X_test, y_train[:, i], y_test[:, i], pfam_name)
        results.append({
            'pfam': pfam_name,
            'r2_train': res['r2_train'],
            'r2_test': res['r2_test'],
            'rmse_test': res['rmse_test']
        })

        # Store trained model
        trained_models[pfam_name] = res['model']

        # SHAP for best targets
        if res['r2_test'] > 0.3 and len(shap_results) < 5 and SHAP_AVAILABLE:
            shap_vals, mean_shap, X_shap = compute_shap_values(res['model'], X_test, env_cols)
            if shap_vals is not None:
                shap_results[pfam_name] = {
                    'shap_values': shap_vals,
                    'mean_abs_shap': mean_shap,
                    'r2': res['r2_test']
                }

    print("done")
    results_df = pd.DataFrame(results).sort_values('r2_test', ascending=False)

    # Return metadata for model reproducibility
    model_metadata = {
        'target_names': target_names,
        'target_indices': top_idx.tolist()
    }

    return results_df, shap_results, trained_models, scaler, model_metadata

def save_figure(fig, name):
    """Save as PDF and SVG."""
    for fmt in ['pdf', 'svg']:
        fig.savefig(FIGURES_DIR / f"{name}_{TIMESTAMP}.{fmt}",
                   format=fmt, bbox_inches='tight', transparent=True, edgecolor='none')
    print(f"  Saved: {name}_{TIMESTAMP}.pdf/.svg")

def plot_shap_summary(shap_results, feature_names, output_prefix, direction):
    """Create SHAP summary visualization."""
    print(f"\n  Creating {direction} SHAP plots...")

    if not shap_results:
        print(f"    No models with R2 > threshold for SHAP")
        return

    # Aggregate importance across targets
    n_features = len(feature_names)
    agg_importance = np.zeros(n_features)

    for target, data in shap_results.items():
        if len(data['mean_abs_shap']) == n_features:
            agg_importance += data['mean_abs_shap']

    if agg_importance.sum() == 0:
        print(f"    No valid SHAP values")
        return

    agg_importance /= len(shap_results)

    # Top features
    top_idx = np.argsort(agg_importance)[-TOP_FEATURES_PLOT:]

    fig, ax = plt.subplots(figsize=(5, 6))

    y_pos = np.arange(len(top_idx))
    ax.barh(y_pos, agg_importance[top_idx], color='steelblue', height=0.7, edgecolor='none')
    ax.set_yticks(y_pos)

    # Clean feature names
    labels = []
    for i in top_idx:
        name = feature_names[i]
        if name.startswith('PF'):
            name = name.split('.')[0]  # Remove version
        labels.append(name)

    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'Feature Importance ({direction.upper()} modeling)\n'
                f'Averaged across {len(shap_results)} targets', fontsize=6)

    plt.tight_layout()
    save_figure(fig, f'shap_{output_prefix}_importance')
    plt.close()

def plot_results_overview(reverse_df, forward_df):
    """Create overview figure of modeling results."""
    print("\n  Creating results overview...")

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    # Reverse modeling R2 distribution
    ax = axes[0]
    ax.hist(reverse_df['r2_test'], bins=20, color='steelblue', edgecolor='none', alpha=0.8)
    ax.axvline(0.2, color='red', ls='--', lw=0.5)
    ax.set_xlabel('Test R2')
    ax.set_ylabel('Count')
    ax.set_title('Reverse: PFAM->Env', fontsize=6)
    ax.text(-0.12, 1.05, 'A', transform=ax.transAxes, fontweight='bold', fontsize=7)

    # Forward modeling R2 distribution
    ax = axes[1]
    ax.hist(forward_df['r2_test'], bins=20, color='darkorange', edgecolor='none', alpha=0.8)
    ax.axvline(0.3, color='red', ls='--', lw=0.5)
    ax.set_xlabel('Test R2')
    ax.set_ylabel('Count')
    ax.set_title('Forward: Env->PFAM', fontsize=6)
    ax.text(-0.12, 1.05, 'B', transform=ax.transAxes, fontweight='bold', fontsize=7)

    # Top reverse predictions
    ax = axes[2]
    top_rev = reverse_df.head(15)
    y_pos = range(len(top_rev))
    ax.barh(y_pos, top_rev['r2_test'], color='steelblue', height=0.7, edgecolor='none')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_rev['env_var'], fontsize=5)
    ax.set_xlabel('Test R2')
    ax.set_title('Top Predictable Env Vars', fontsize=6)
    ax.invert_yaxis()
    ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, fontweight='bold', fontsize=7)

    plt.tight_layout()
    save_figure(fig, 'xgboost_shap_overview')
    plt.close()

def save_models(reverse_models, reverse_scaler, forward_models, forward_scaler,
                forward_metadata, pfam_cols, env_cols):
    """Save trained XGBoost models and scalers for reproducibility."""
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)

    MODELS_DIR.mkdir(exist_ok=True)

    # Save reverse models (PFAM -> Env)
    reverse_bundle = {
        'models': reverse_models,
        'scaler': reverse_scaler,
        'feature_names': pfam_cols,
        'target_names': env_cols,
        'direction': 'reverse (PFAM -> Environment)',
        'xgb_params': XGB_PARAMS,
        'timestamp': TIMESTAMP
    }
    reverse_path = MODELS_DIR / f"xgboost_reverse_models_{TIMESTAMP}.joblib"
    joblib.dump(reverse_bundle, reverse_path)
    print(f"  Saved: {reverse_path.name} ({len(reverse_models)} models)")

    # Save forward models (Env -> PFAM)
    forward_bundle = {
        'models': forward_models,
        'scaler': forward_scaler,
        'feature_names': env_cols,
        'target_names': forward_metadata['target_names'],
        'target_indices': forward_metadata['target_indices'],
        'direction': 'forward (Environment -> PFAM)',
        'xgb_params': XGB_PARAMS,
        'timestamp': TIMESTAMP
    }
    forward_path = MODELS_DIR / f"xgboost_forward_models_{TIMESTAMP}.joblib"
    joblib.dump(forward_bundle, forward_path)
    print(f"  Saved: {forward_path.name} ({len(forward_models)} models)")

    # Write model manifest
    manifest = {
        'timestamp': TIMESTAMP,
        'reverse_models': {
            'file': reverse_path.name,
            'n_models': len(reverse_models),
            'features': len(pfam_cols),
            'targets': list(reverse_models.keys())
        },
        'forward_models': {
            'file': forward_path.name,
            'n_models': len(forward_models),
            'features': len(env_cols),
            'targets': list(forward_models.keys())
        },
        'xgb_params': XGB_PARAMS,
        'usage': 'Use joblib.load() to load model bundles'
    }
    manifest_path = MODELS_DIR / f"model_manifest_{TIMESTAMP}.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {manifest_path.name}")

    return reverse_path, forward_path

def save_results(reverse_df, forward_df, shap_reverse, shap_forward, pfam_cols, env_cols):
    """Save all results."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # DataFrames
    reverse_df.to_csv(REPORTS_DIR / f"xgboost_reverse_full_{TIMESTAMP}.csv", index=False)
    forward_df.to_csv(REPORTS_DIR / f"xgboost_forward_full_{TIMESTAMP}.csv", index=False)
    print(f"  Saved model results")

    # SHAP importance
    if shap_reverse:
        shap_rows = []
        for env_var, data in shap_reverse.items():
            for i, (feat, shap_val) in enumerate(zip(pfam_cols, data['mean_abs_shap'])):
                shap_rows.append({
                    'target': env_var,
                    'feature': feat,
                    'mean_abs_shap': shap_val
                })
        pd.DataFrame(shap_rows).to_csv(
            REPORTS_DIR / f"shap_reverse_importance_{TIMESTAMP}.csv", index=False
        )
        print(f"  Saved SHAP reverse importance")

    # Summary
    summary = {
        'timestamp': TIMESTAMP,
        'xgboost_params': XGB_PARAMS,
        'reverse_modeling': {
            'n_targets': len(reverse_df),
            'n_features': len(pfam_cols),
            'mean_r2': float(reverse_df['r2_test'].mean()),
            'max_r2': float(reverse_df['r2_test'].max()),
            'best_target': reverse_df.iloc[0]['env_var'],
            'n_r2_above_0.2': int((reverse_df['r2_test'] > 0.2).sum()),
        },
        'forward_modeling': {
            'n_targets': len(forward_df),
            'n_features': len(env_cols),
            'mean_r2': float(forward_df['r2_test'].mean()),
            'max_r2': float(forward_df['r2_test'].max()),
            'best_target': forward_df.iloc[0]['pfam'],
            'n_r2_above_0.3': int((forward_df['r2_test'] > 0.3).sum()),
        },
        'shap': {
            'reverse_targets_analyzed': list(shap_reverse.keys()) if shap_reverse else [],
            'forward_targets_analyzed': list(shap_forward.keys()) if shap_forward else []
        }
    }

    with open(REPORTS_DIR / f"xgboost_shap_summary_{TIMESTAMP}.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved summary")

    # Print summary
    print(f"\n  Summary:")
    print(f"    Reverse: mean R2={summary['reverse_modeling']['mean_r2']:.3f}, "
          f"max={summary['reverse_modeling']['max_r2']:.3f} ({summary['reverse_modeling']['best_target']})")
    print(f"    Forward: mean R2={summary['forward_modeling']['mean_r2']:.3f}, "
          f"max={summary['forward_modeling']['max_r2']:.3f}")

def main():
    if not XGB_AVAILABLE:
        print("ERROR: XGBoost not available. Cannot proceed.")
        return None, None

    print("=" * 70)
    print("PHASE 8: XGBOOST + SHAP ANALYSIS (algaGPT)")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"XGBoost: {xgb.__version__}")
    if SHAP_AVAILABLE:
        print(f"SHAP: {shap.__version__}")
    print("=" * 70)

    # Load data
    env_matrix, pfam_matrix, env_cols, pfam_cols = load_processed_data()

    # Reverse modeling
    reverse_df, shap_reverse, reverse_models, reverse_scaler = run_reverse_modeling_with_shap(
        pfam_matrix, env_matrix, pfam_cols, env_cols
    )

    # Forward modeling
    forward_df, shap_forward, forward_models, forward_scaler, forward_metadata = run_forward_modeling_with_shap(
        env_matrix, pfam_matrix, env_cols, pfam_cols, n_targets=100
    )

    # Save trained models (supplementary data)
    save_models(
        reverse_models, reverse_scaler,
        forward_models, forward_scaler,
        forward_metadata, pfam_cols, env_cols
    )

    # Visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    plot_shap_summary(shap_reverse, pfam_cols, 'reverse', 'reverse')
    plot_shap_summary(shap_forward, env_cols, 'forward', 'forward')
    plot_results_overview(reverse_df, forward_df)

    # Save results
    save_results(reverse_df, forward_df, shap_reverse, shap_forward, pfam_cols, env_cols)

    print("\n" + "=" * 70)
    print("PHASE 8 COMPLETE")
    print("=" * 70)

    return reverse_df, forward_df

if __name__ == "__main__":
    reverse_df, forward_df = main()
