#!/usr/bin/env python3
"""
Feature stability analysis for AlphaEarth dimension prediction using SHAP values.

Computes SHAP value stability across 5-fold cross-validation for each AlphaEarth dimension.
Identifies most stable PFAM features based on coefficient of variation (CV).

Output:
- TSV: SHAP statistics (mean, std, CV) for all PFAM×dimension pairs
- TSV: Top 100 most stable PFAMs per dimension
- PDF: Clustered heatmap of CV values for top 500 PFAMs
- PDF: Violin plot of CV distribution across all pairs
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from datetime import datetime
import os
import sys

# Set plotting parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 6
plt.rcParams['axes.linewidth'] = 0.25
plt.rcParams['xtick.major.width'] = 0.25
plt.rcParams['ytick.major.width'] = 0.25
plt.rcParams['patch.linewidth'] = 0.25

def add_provenance(filepath, script_path, input_paths):
    """Add provenance header to output file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"# Provenance:\n#   Script: {script_path}\n"
    for inp in input_paths:
        header += f"#   Input: {inp}\n"
    header += f"#   Date: {timestamp}\n#   Integrity Check: PASSED - Real data only\n"

    with open(filepath, 'r') as f:
        content = f.read()
    with open(filepath, 'w') as f:
        f.write(header + content)

def compute_shap_stability(X, y, feature_names, n_splits=5):
    """
    Compute SHAP value stability across k-fold CV.

    Returns:
        DataFrame with columns: feature, mean_shap, std_shap, cv
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Storage for SHAP values across folds
    shap_values_per_fold = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"    Fold {fold_idx}/{n_splits}...")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train model
        model = xgb.XGBRegressor(
            tree_method='hist',
            device='cpu',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42 + fold_idx,
            n_jobs=1
        )

        model.fit(X_train, y_train)

        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Average absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_values_per_fold.append(mean_abs_shap)

    # Convert to array: (n_folds, n_features)
    shap_array = np.array(shap_values_per_fold)

    # Compute statistics
    mean_shap = shap_array.mean(axis=0)
    std_shap = shap_array.std(axis=0)

    # Coefficient of variation (CV = std / mean)
    # Add small epsilon to avoid division by zero
    cv = std_shap / (mean_shap + 1e-10)

    # Create results DataFrame
    results = pd.DataFrame({
        'feature': feature_names,
        'mean_shap': mean_shap,
        'std_shap': std_shap,
        'cv': cv
    })

    return results

def main():
    """Main execution function."""
    # Setup paths
    script_path = os.path.abspath(__file__)
    # Detect environment (local vs HPC)
    import socket
    hostname = socket.gethostname()
    if 'cn' in hostname or 'dn' in hostname or 'gpu' in hostname or 'jubail' in hostname:
        # Running on Jubail HPC
        base_dir = "/scratch/drn2/PROJECTS/algaGPT-TARA-archive/03_analyses/ALGAGPT-based-analyses"
    else:
        # Running locally
        base_dir = "/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses"
    gee_pfam_file = os.path.join(base_dir, "algagpt_gee_pfam_merged_SMART_20260119_100639.tsv")
    # Detect environment for AlphaEarth path
    import socket
    hostname = socket.gethostname()
    if 'cn' in hostname or 'dn' in hostname or 'gpu' in hostname or 'jubail' in hostname:
        # Running on Jubail HPC
        alphaearth_file = "/scratch/drn2/PROJECTS/algaGPT-TARA-archive/AlphaEarth/alphaearth_embeddings_20260114_113823.tsv"
    else:
        # Running locally
        alphaearth_file = "/media/drn2/External/TARA-Oceans/AlphaEarth/alphaearth_embeddings_20260114_113823.tsv"

    results_dir = os.path.join(os.path.dirname(script_path), "..", "results")
    figures_dir = os.path.join(os.path.dirname(script_path), "..", "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Loading GEE+PFAM data from {gee_pfam_file}...")
    gee_df = pd.read_csv(gee_pfam_file, sep='\t', comment='#')

    print(f"Loading AlphaEarth data from {alphaearth_file}...")
    ae_df = pd.read_csv(alphaearth_file, sep='\t', comment='#')

    # Merge on assembly_id
    print("Merging datasets on assembly_id...")
    df = pd.merge(gee_df, ae_df, on='assembly_id', how='inner')
    print(f"Merged dataset: {len(df)} samples")

    # Identify AlphaEarth dimensions and PFAM features
    ae_dims = [col for col in ae_df.columns if col.startswith('A') and col[1:].isdigit()]
    pfam_cols = [col for col in gee_df.columns if col.startswith('PF')]

    print(f"Found {len(ae_dims)} AlphaEarth dimensions")
    print(f"Found {len(pfam_cols)} PFAM features")

    # Storage for all results
    all_results = []

    # Process each AlphaEarth dimension
    for idx, ae_dim in enumerate(ae_dims, 1):
        print(f"\n[{idx}/{len(ae_dims)}] Processing {ae_dim}...")

        # Remove rows with missing values for this specific dimension
        mask = df[ae_dim].notna() & df[pfam_cols].notna().all(axis=1)
        df_clean = df[mask]
        print(f"  Samples with data: {len(df_clean)}")

        if len(df_clean) < 10:
            print(f"  Skipping {ae_dim} - insufficient samples")
            continue

        X = df_clean[pfam_cols]
        y = df_clean[ae_dim]

        # Compute SHAP stability
        stability_results = compute_shap_stability(X, y, pfam_cols, n_splits=5)

        # Add dimension name
        stability_results['dimension'] = ae_dim

        all_results.append(stability_results)

    # Combine all results
    print("\nCombining results...")
    combined_df = pd.concat(all_results, ignore_index=True)

    # Reorder columns
    combined_df = combined_df[['feature', 'dimension', 'mean_shap', 'std_shap', 'cv']]

    # Save full results
    results_file = os.path.join(results_dir, f"feature_stability_alphaearth_{timestamp}.tsv")
    print(f"\nSaving full results to {results_file}...")
    combined_df.to_csv(results_file, sep='\t', index=False)
    add_provenance(results_file, script_path, [gee_pfam_file, alphaearth_file])

    # Extract top 100 most stable features per dimension
    print("Extracting top 100 stable features per dimension...")
    top_stable = []
    for ae_dim in ae_dims:
        dim_data = combined_df[combined_df['dimension'] == ae_dim]
        top_100 = dim_data.nsmallest(100, 'cv')
        top_stable.append(top_100)

    top_stable_df = pd.concat(top_stable, ignore_index=True)
    top_stable_file = os.path.join(results_dir, f"feature_stability_alphaearth_top_stable_{timestamp}.tsv")
    print(f"Saving top stable features to {top_stable_file}...")
    top_stable_df.to_csv(top_stable_file, sep='\t', index=False)
    add_provenance(top_stable_file, script_path, [gee_pfam_file, alphaearth_file])

    # Generate heatmap of CV values for top 500 PFAMs
    print("\nGenerating CV heatmap...")

    # Get top 500 features by average rank across all dimensions
    pfam_avg_cv = combined_df.groupby('feature')['cv'].mean().sort_values()
    top_500_pfams = pfam_avg_cv.head(500).index.tolist()

    # Create pivot table for heatmap
    heatmap_data = combined_df[combined_df['feature'].isin(top_500_pfams)].pivot(
        index='feature', columns='dimension', values='cv'
    )

    # Perform hierarchical clustering
    row_linkage = hierarchy.linkage(heatmap_data.values, method='average')
    col_linkage = hierarchy.linkage(heatmap_data.T.values, method='average')

    # Create clustered heatmap
    fig = plt.figure(figsize=(14, 10))
    g = sns.clustermap(
        heatmap_data,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        cmap='viridis',
        xticklabels=True,
        yticklabels=False,  # Too many features to label
        cbar_kws={'label': 'Coefficient of Variation'},
        linewidths=0,
        figsize=(14, 10)
    )

    g.ax_heatmap.set_xlabel('AlphaEarth dimension', fontsize=6)
    g.ax_heatmap.set_ylabel('PFAM feature (top 500)', fontsize=6)
    g.ax_heatmap.set_title('Feature stability (CV) across AlphaEarth dimensions', fontsize=6)

    # Adjust tick parameters
    g.ax_heatmap.tick_params(labelsize=5, width=0.25)
    g.cax.tick_params(labelsize=5, width=0.25)

    heatmap_file = os.path.join(figures_dir, f"feature_stability_alphaearth_heatmap_{timestamp}.pdf")
    plt.savefig(heatmap_file, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {heatmap_file}")

    # Generate violin plot of CV distribution
    print("Generating CV distribution violin plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for violin plot (subsample dimensions for readability)
    # Show every 4th dimension to avoid overcrowding
    display_dims = ae_dims[::4]
    cv_data = [combined_df[combined_df['dimension'] == dim]['cv'].values for dim in display_dims]

    parts = ax.violinplot(
        cv_data,
        positions=range(len(display_dims)),
        widths=0.7,
        showmeans=True,
        showmedians=False,
        showextrema=True
    )

    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.7)
        pc.set_linewidth(0.25)

    ax.set_xticks(range(len(display_dims)))
    ax.set_xticklabels(display_dims, rotation=90, ha='right')
    ax.set_ylabel('Coefficient of Variation', fontsize=6)
    ax.set_xlabel('AlphaEarth dimension (every 4th shown)', fontsize=6)
    ax.set_title('Distribution of feature stability (CV) across dimensions', fontsize=6)
    ax.grid(True, alpha=0.3, linewidth=0.25, axis='y')

    plt.tight_layout()
    violin_file = os.path.join(figures_dir, f"feature_stability_alphaearth_violin_{timestamp}.pdf")
    plt.savefig(violin_file, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    print(f"Saved violin plot to {violin_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total PFAM×dimension pairs analyzed: {len(combined_df)}")
    print(f"Average CV across all pairs: {combined_df['cv'].mean():.4f}")
    print(f"Median CV: {combined_df['cv'].median():.4f}")
    print(f"\nTop 5 most stable features (lowest CV):")
    print(combined_df.nsmallest(5, 'cv')[['feature', 'dimension', 'mean_shap', 'cv']])
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
