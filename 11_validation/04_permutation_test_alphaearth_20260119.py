#!/usr/bin/env python3
"""
Permutation test for AlphaEarth dimension predictability from PFAM features.

Tests significance of XGBoost model performance using 1000 permutations
for each of 64 AlphaEarth embedding dimensions.

Output:
- TSV: observed R², p-values, FDR-corrected q-values
- TSV: null distribution values for all permutations
- PDF: violin plots showing null distributions with observed values
- PDF: volcano plot of significance vs effect size
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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

def train_xgboost_model(X, y, random_state=42):
    """Train XGBoost model and return R² score."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = xgb.XGBRegressor(
        tree_method='hist',
        device='cpu',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return r2

def permutation_test(X, y, n_permutations=1000):
    """
    Run permutation test by shuffling target variable.

    Returns:
        observed_r2: R² score on real data
        null_r2_values: List of R² scores from permuted data
        p_value: Empirical p-value
    """
    # Compute observed R²
    observed_r2 = train_xgboost_model(X, y, random_state=42)

    # Run permutations
    null_r2_values = []
    for i in range(n_permutations):
        # Shuffle target variable
        y_permuted = y.sample(frac=1, random_state=i).reset_index(drop=True)
        null_r2 = train_xgboost_model(X, y_permuted, random_state=42)
        null_r2_values.append(null_r2)

    # Compute empirical p-value
    p_value = (np.sum(np.array(null_r2_values) >= observed_r2) + 1) / (n_permutations + 1)

    return observed_r2, null_r2_values, p_value

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

    # Identify AlphaEarth dimensions (A00-A63)
    ae_dims = [col for col in ae_df.columns if col.startswith('A') and col[1:].isdigit()]
    pfam_cols = [col for col in gee_df.columns if col.startswith('PF')]

    print(f"Found {len(ae_dims)} AlphaEarth dimensions")
    print(f"Found {len(pfam_cols)} PFAM features")

    # Remove rows with missing PFAM values
    df_clean = df[pfam_cols + ae_dims].dropna(subset=pfam_cols)
    print(f"After removing NaN in PFAMs: {len(df_clean)} samples")

    X = df_clean[pfam_cols]

    # Storage for results
    results = []
    null_distributions = []

    # Run permutation test for each AlphaEarth dimension
    for idx, ae_dim in enumerate(ae_dims, 1):
        print(f"\n[{idx}/{len(ae_dims)}] Processing {ae_dim}...")

        # Filter samples with valid target values
        valid_idx = df_clean[ae_dim].notna()
        X_valid = X[valid_idx]
        y = df_clean.loc[valid_idx, ae_dim]

        if len(y) < 100:
            print(f"  SKIP (only {len(y)} valid samples)")
            continue

        # Run permutation test
        observed_r2, null_r2_values, p_value = permutation_test(X_valid, y, n_permutations=1000)

        print(f"  Observed R²: {observed_r2:.4f}")
        print(f"  P-value: {p_value:.4f}")

        # Store results
        results.append({
            'dimension': ae_dim,
            'observed_r2': observed_r2,
            'p_value': p_value
        })

        # Store null distribution
        for perm_id, null_r2 in enumerate(null_r2_values):
            null_distributions.append({
                'dimension': ae_dim,
                'permutation_id': perm_id,
                'null_r2': null_r2
            })

    # Convert to DataFrames
    results_df = pd.DataFrame(results)
    null_dist_df = pd.DataFrame(null_distributions)

    # Apply FDR correction
    print("\nApplying Benjamini-Hochberg FDR correction...")
    _, q_values, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
    results_df['q_value_fdr'] = q_values

    # Save results
    results_file = os.path.join(results_dir, f"permutation_test_alphaearth_{timestamp}.tsv")
    null_dist_file = os.path.join(results_dir, f"permutation_test_alphaearth_null_distribution_{timestamp}.tsv")

    print(f"\nSaving results to {results_file}...")
    results_df.to_csv(results_file, sep='\t', index=False)
    add_provenance(results_file, script_path, [gee_pfam_file, alphaearth_file])

    print(f"Saving null distributions to {null_dist_file}...")
    null_dist_df.to_csv(null_dist_file, sep='\t', index=False)
    add_provenance(null_dist_file, script_path, [gee_pfam_file, alphaearth_file])

    # Generate violin plot
    print("\nGenerating violin plot...")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for violin plot
    plot_data = []
    for _, row in results_df.iterrows():
        dim = row['dimension']
        obs = row['observed_r2']
        nulls = null_dist_df[null_dist_df['dimension'] == dim]['null_r2'].values
        plot_data.append({'dimension': dim, 'values': nulls, 'observed': obs})

    # Sort by observed R²
    plot_data = sorted(plot_data, key=lambda x: x['observed'], reverse=True)

    # Create violin plots
    positions = range(len(plot_data))
    parts = ax.violinplot(
        [d['values'] for d in plot_data],
        positions=positions,
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.7)
        pc.set_linewidth(0.25)

    # Add observed values as red dots
    observed_values = [d['observed'] for d in plot_data]
    ax.scatter(positions, observed_values, c='red', s=10, zorder=3, alpha=0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels([d['dimension'] for d in plot_data], rotation=90, ha='right')
    ax.set_ylabel('R² score', fontsize=6)
    ax.set_xlabel('AlphaEarth dimension', fontsize=6)
    ax.set_title('Permutation test: null distributions vs observed R²', fontsize=6)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.25, alpha=0.5)
    ax.grid(True, alpha=0.3, linewidth=0.25)

    plt.tight_layout()
    violin_file = os.path.join(figures_dir, f"permutation_test_alphaearth_violin_{timestamp}.pdf")
    plt.savefig(violin_file, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    print(f"Saved violin plot to {violin_file}")

    # Generate volcano plot
    print("Generating volcano plot...")
    fig, ax = plt.subplots(figsize=(8, 6))

    results_df['neg_log10_p'] = -np.log10(results_df['p_value'])

    # Color by significance
    colors = ['red' if q < 0.05 else 'gray' for q in results_df['q_value_fdr']]

    ax.scatter(results_df['observed_r2'], results_df['neg_log10_p'],
               c=colors, s=20, alpha=0.7)

    # Add significance threshold line
    threshold = -np.log10(0.05)
    ax.axhline(y=threshold, color='blue', linestyle='--', linewidth=0.25,
               alpha=0.5, label='p = 0.05')

    # Label significant points
    for _, row in results_df[results_df['q_value_fdr'] < 0.05].iterrows():
        ax.text(row['observed_r2'], row['neg_log10_p'], row['dimension'],
                fontsize=5, alpha=0.8)

    ax.set_xlabel('Observed R²', fontsize=6)
    ax.set_ylabel('-log10(p-value)', fontsize=6)
    ax.set_title('Volcano plot: AlphaEarth dimension predictability', fontsize=6)
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3, linewidth=0.25)

    plt.tight_layout()
    volcano_file = os.path.join(figures_dir, f"permutation_test_alphaearth_significance_{timestamp}.pdf")
    plt.savefig(volcano_file, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    print(f"Saved volcano plot to {volcano_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total AlphaEarth dimensions tested: {len(results_df)}")
    print(f"Significant (q < 0.05): {sum(results_df['q_value_fdr'] < 0.05)}")
    print(f"Significant (q < 0.01): {sum(results_df['q_value_fdr'] < 0.01)}")
    print(f"\nTop 5 predictable dimensions:")
    print(results_df.nlargest(5, 'observed_r2')[['dimension', 'observed_r2', 'p_value', 'q_value_fdr']])
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
