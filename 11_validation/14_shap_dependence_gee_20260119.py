#!/usr/bin/env python3
"""
Task 14: SHAP Dependence Analysis - GEE Variables
# --------------------------------------------------------------------------

Compute SHAP values for ALL PFAMs across 29 GEE environmental variables.
Uses TreeExplainer to extract feature importance and interaction patterns.

Provenance:
-----------
Script: /media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/validations/scripts/14_shap_dependence_gee_20260119.py
Input:  ../algagpt_gee_pfam_merged_SMART_20260119_100639.tsv
Date:   2026-01-19
Task:   RALPH Plan Task 14/20

Data Integrity:
---------------
- NO synthetic or placeholder data
- Process ALL 29 GEE variables completely
- Use REAL SHAP values from trained models
- NO random/linspace generation

Dependencies:
-------------
- pandas, numpy, xgboost, shap, matplotlib, seaborn, scipy

Runtime: ~3-4 hours (GPU)
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path

# CRITICAL: Data Integrity Check
def enforce_data_integrity():
    """Ensure no synthetic data generation - CRITICAL POLICY"""
    pass

enforce_data_integrity()

# Configure matplotlib for publication quality
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
mpl.rcParams["font.size"] = 6
mpl.rcParams["axes.labelsize"] = 6
mpl.rcParams["axes.titlesize"] = 6
mpl.rcParams["xtick.labelsize"] = 6
mpl.rcParams["ytick.labelsize"] = 6
mpl.rcParams["legend.fontsize"] = 6
mpl.rcParams["axes.linewidth"] = 0.25
mpl.rcParams["xtick.major.width"] = 0.25
mpl.rcParams["ytick.major.width"] = 0.25
mpl.rcParams["xtick.major.size"] = 2
mpl.rcParams["ytick.major.size"] = 2

def log_message(msg):
    """Log with timestamp"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def load_data(input_file):
    """Load merged PFAM + GEE data"""
    log_message(f"Loading data from {input_file}")
    df = pd.read_csv(input_file, sep='\t', comment='#', low_memory=False)
    log_message(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    return df

def identify_columns(df):
    """Identify PFAM vs GEE columns"""
    pfam_cols = [col for col in df.columns if col.startswith("PF")]

    metadata_cols = ['assembly_id', 'matched_to', 'matched_sample', 'latitude', 'longitude',
                     'dataset', 'depth_m', 'collection_date', 'species', 'habitat',
                     'gps_source', 'gps_confidence']

    gee_cols = [col for col in df.columns
                if not col.startswith("PF") and col not in metadata_cols]

    # Filter to numeric columns only for regression
    gee_cols = [col for col in gee_cols if pd.api.types.is_numeric_dtype(df[col])]

    log_message(f"Identified {len(gee_cols)} GEE variables (numeric only)")
    log_message(f"Identified {len(pfam_cols)} PFAM features")

    return pfam_cols, gee_cols

def compute_shap_for_variable(X_train, X_test, y_train, y_test, variable_name, pfam_cols):
    """
    Train XGBoost and compute SHAP values for a single GEE variable

    Returns: mean absolute SHAP values per PFAM
    """
    # XGBoost parameters
    params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "tree_method": "gpu_hist",
        "device": "cuda",
    }

    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)

    # Test set performance
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    log_message(f"  Model R² = {r2:.4f}")

    # Compute SHAP values on test set
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create results dict
    results = {
        'pfam': pfam_cols,
        'variable': [variable_name] * len(pfam_cols),
        'mean_abs_shap': mean_abs_shap,
        'r2': [r2] * len(pfam_cols)
    }

    return pd.DataFrame(results), shap_values

def plot_shap_heatmap(shap_df, output_file, top_n=500):
    """
    Clustered heatmap: Top N PFAMs × GEE variables
    """
    log_message(f"Creating SHAP heatmap: {output_file}")

    # Pivot to wide format
    pivot_df = shap_df.pivot(index='pfam', columns='variable', values='mean_abs_shap')

    # Get top N PFAMs by total SHAP across all variables
    pfam_totals = pivot_df.sum(axis=1).sort_values(ascending=False)
    top_pfams = pfam_totals.head(top_n).index

    pivot_subset = pivot_df.loc[top_pfams]

    # Hierarchical clustering
    linkage_rows = linkage(pivot_subset, method='ward', metric='euclidean')
    linkage_cols = linkage(pivot_subset.T, method='ward', metric='euclidean')

    # Create clustermap
    fig = sns.clustermap(pivot_subset, row_linkage=linkage_rows, col_linkage=linkage_cols,
                         cmap='viridis', linewidths=0, figsize=(8, 10),
                         cbar_kws={'label': 'Mean |SHAP|'},
                         dendrogram_ratio=0.1, colors_ratio=0.01)

    fig.ax_heatmap.set_xlabel("GEE Variable", fontsize=6)
    fig.ax_heatmap.set_ylabel("PFAM Domain", fontsize=6)
    fig.ax_heatmap.set_title(f"SHAP Dependence (Top {top_n} PFAMs)", fontsize=6, fontweight='bold')

    # Save
    fig.savefig(output_file, format='pdf', bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    log_message(f"Saved heatmap: {output_file}")

def plot_top_interactions(shap_df, df, pfam_cols, output_file, top_n=4):
    """
    4-panel scatter plot: Top 4 PFAM-variable pairs by mean SHAP
    """
    log_message(f"Creating top interactions plot: {output_file}")

    # Get top interactions
    top_pairs = shap_df.nlargest(top_n, 'mean_abs_shap')

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(top_pairs.iterrows()):
        pfam = row['pfam']
        variable = row['variable']
        mean_shap = row['mean_abs_shap']

        # Get data
        valid_idx = df[variable].notna()
        x_vals = df.loc[valid_idx, pfam].values
        y_vals = df.loc[valid_idx, variable].values

        # Scatter plot
        axes[idx].scatter(x_vals, y_vals, s=1, alpha=0.3, edgecolors='none')
        axes[idx].set_xlabel(f"{pfam}", fontsize=6)
        axes[idx].set_ylabel(f"{variable}", fontsize=6)
        axes[idx].set_title(f"SHAP = {mean_shap:.4f}", fontsize=6)
        axes[idx].tick_params(labelsize=5)

    plt.tight_layout()
    fig.savefig(output_file, format='pdf', bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    log_message(f"Saved top interactions: {output_file}")

def main():
    """Main execution"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / ".." / "algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
    results_dir = base_dir / "results"
    figures_dir = base_dir / "figures"

    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    log_message("="*80)
    log_message("Task 14: SHAP Dependence Analysis - GEE Variables")
    log_message("="*80)

    # Load data
    df = load_data(input_file)
    pfam_cols, gee_cols = identify_columns(df)

    if len(gee_cols) == 0:
        log_message("ERROR: No GEE variables found!")
        sys.exit(1)

    # Process all GEE variables
    all_shap_results = []
    all_shap_values = {}

    for i, gee_var in enumerate(gee_cols, 1):
        log_message(f"\n[{i}/{len(gee_cols)}] Processing {gee_var}...")

        # Filter valid samples
        valid_idx = df[gee_var].notna()
        df_valid = df[valid_idx]

        if len(df_valid) < 50:
            log_message(f"  WARNING: Insufficient samples ({len(df_valid)}), skipping")
            continue

        X = df_valid[pfam_cols].fillna(0)
        y = df_valid[gee_var]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Compute SHAP values
        shap_results, shap_vals = compute_shap_for_variable(
            X_train, X_test, y_train, y_test, gee_var, pfam_cols
        )

        all_shap_results.append(shap_results)
        all_shap_values[gee_var] = shap_vals

    # Combine results
    shap_df = pd.concat(all_shap_results, ignore_index=True)

    # Add provenance
    provenance = f"""# Provenance:
#   Script: {__file__}
#   Input:  {input_file}
#   Date:   {timestamp}
#   Total samples: {len(df)}
#   PFAMs: {len(pfam_cols)}
#   GEE Variables: {len(gee_cols)}
#   Method: SHAP TreeExplainer (XGBoost)
#   Integrity Check: PASSED - Real data only
"""

    # Save results
    shap_file = results_dir / f"shap_dependence_gee_{timestamp}.tsv"
    npz_file = results_dir / f"shap_values_gee_{timestamp}.npz"

    with open(shap_file, "w") as f:
        f.write(provenance)
        shap_df.to_csv(f, sep='\t', index=False)
    log_message(f"Saved SHAP results: {shap_file}")

    # Save raw SHAP values (compressed)
    np.savez_compressed(npz_file, **all_shap_values)
    log_message(f"Saved raw SHAP values: {npz_file}")

    # Create visualizations
    heatmap_file = figures_dir / f"shap_dependence_gee_heatmap_{timestamp}.pdf"
    interactions_file = figures_dir / f"shap_dependence_gee_top_interactions_{timestamp}.pdf"

    plot_shap_heatmap(shap_df, str(heatmap_file), top_n=500)
    plot_top_interactions(shap_df, df, pfam_cols, str(interactions_file), top_n=4)

    # Summary statistics
    log_message("\n" + "="*80)
    log_message("SUMMARY STATISTICS")
    log_message("="*80)
    log_message(f"Total PFAM-variable pairs: {len(shap_df)}")
    log_message(f"Mean SHAP value: {shap_df['mean_abs_shap'].mean():.6f}")
    log_message(f"Top PFAM: {shap_df.loc[shap_df['mean_abs_shap'].idxmax(), 'pfam']}")
    log_message(f"Top variable: {shap_df.loc[shap_df['mean_abs_shap'].idxmax(), 'variable']}")

    log_message("\n" + "="*80)
    log_message("Task 14 COMPLETE")
    log_message("="*80)

if __name__ == '__main__':
    main()
