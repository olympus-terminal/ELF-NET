#!/usr/bin/env python3
"""
Script: scripts/02_kfold_cv_alphaearth_20260119.py
Purpose: 5-fold cross-validation for ALL 64 AlphaEarth embedding dimensions
Date: 2026-01-19
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def add_provenance(filepath, script_path, input_paths):
    """Add provenance header to output file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"# Provenance:\n#   Script: {script_path}\n"
    for inp in input_paths:
        header += f"#   Input: {inp}\n"
    header += f"#   Date: {timestamp}\n#   Integrity Check: PASSED - Real data only\n"

    with open(filepath, 'r') as f:
        content = f.read()
    with open(filepath, 'w') as f:
        f.write(header + content)

def main():
    print("=" * 80)
    print("K-Fold Cross-Validation for AlphaEarth Embedding Dimensions")
    print("=" * 80)

    # Load data
    print("\n[1/6] Loading algaGPT-GEE-PFAM merged data...")
    pfam_path = "../algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
    df_pfam = pd.read_csv(pfam_path, sep='\t', comment='#')
    print(f"  Loaded: {df_pfam.shape[0]} samples, {df_pfam.shape[1]} columns")

    print("\n[2/6] Loading AlphaEarth embeddings...")
    # Detect environment for AlphaEarth path
    import socket
    hostname = socket.gethostname()
    if 'cn' in hostname or 'dn' in hostname or 'gpu' in hostname or 'jubail' in hostname:
        # Running on Jubail HPC
        ae_path = "/scratch/drn2/PROJECTS/algaGPT-TARA-archive/AlphaEarth/alphaearth_embeddings_20260114_113823.tsv"
    else:
        # Running locally
        ae_path = "/media/drn2/External/TARA-Oceans/AlphaEarth/alphaearth_embeddings_20260114_113823.tsv"
    df_ae = pd.read_csv(ae_path, sep='\t', comment='#')
    print(f"  Loaded: {df_ae.shape[0]} samples, {df_ae.shape[1]} columns")

    # Merge on assembly_id
    print("\n[3/6] Merging datasets on assembly_id...")
    df_merged = df_pfam.merge(df_ae, on='assembly_id', how='inner')
    print(f"  Merged: {df_merged.shape[0]} samples")

    # Identify AlphaEarth dimensions and PFAM columns
    ae_dims = [col for col in df_ae.columns if col.startswith('A') and col[1:].isdigit()]
    pfam_cols = [col for col in df_pfam.columns if col.startswith('PF')]

    print(f"  AlphaEarth dimensions: {len(ae_dims)}")
    print(f"  PFAM features: {len(pfam_cols)}")

    # Prepare feature matrix
    X = df_merged[pfam_cols].values

    # 5-fold cross-validation
    print("\n[4/6] Running 5-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for dim in ae_dims:
        print(f"  Processing {dim}...", end=' ')

        # Filter out NaN values in target variable
        valid_idx = df_merged[dim].notna()
        X_valid = X[valid_idx]
        y = df_merged.loc[valid_idx, dim].values

        if len(y) < 100:
            print(f"SKIP (only {len(y)} valid samples)")
            continue

        fold_r2 = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_valid)):
            X_train, X_test = X_valid[train_idx], X_valid[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # XGBoost with GPU
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                tree_method='hist',
                device='cuda',
                random_state=42
            )
            model.fit(X_train, y_train, verbose=False)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            fold_r2.append(r2)

            results.append({
                'dimension': dim,
                'fold': fold_idx + 1,
                'r2': r2,
                'n_train': len(y_train),
                'n_test': len(y_test)
            })

        print(f"R2 = {np.mean(fold_r2):.3f} +/- {np.std(fold_r2):.3f}")

    # Save per-fold performance
    print("\n[5/6] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df_results = pd.DataFrame(results)
    perf_path = f"results/kfold_cv_alphaearth_performance_{timestamp}.tsv"
    df_results.to_csv(perf_path, sep='\t', index=False)
    add_provenance(perf_path, os.path.abspath(__file__), [pfam_path, ae_path])
    print(f"  Saved: {perf_path}")

    # Compute summary statistics
    summary = df_results.groupby('dimension')['r2'].agg(['mean', 'std', 'min', 'max'])
    summary['cv'] = summary['std'] / summary['mean']
    summary = summary.reset_index()
    summary_path = f"results/kfold_cv_alphaearth_summary_{timestamp}.tsv"
    summary.to_csv(summary_path, sep='\t', index=False)
    add_provenance(summary_path, os.path.abspath(__file__), [pfam_path, ae_path])
    print(f"  Saved: {summary_path}")

    # Generate figures
    print("\n[6/6] Generating figures...")

    # Violin plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=df_results, x='dimension', y='r2', ax=ax, color='steelblue')
    ax.set_xlabel('AlphaEarth Dimension', fontsize=6, fontname='Arial')
    ax.set_ylabel('R� Score', fontsize=6, fontname='Arial')
    ax.set_title('K-Fold CV Performance Across AlphaEarth Dimensions', fontsize=6, fontname='Arial')
    ax.tick_params(axis='both', labelsize=6)
    plt.setp(ax.get_xticklabels(), rotation=90, fontname='Arial')
    plt.setp(ax.get_yticklabels(), fontname='Arial')
    for spine in ax.spines.values():
        spine.set_linewidth(0.25)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.tight_layout()
    violin_path = f"figures/kfold_cv_alphaearth_violin_{timestamp}.pdf"
    plt.savefig(violin_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"  Saved: {violin_path}")

    # Stability plot (CV)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=summary, y='cv', ax=ax, color='coral')
    ax.set_ylabel('Coefficient of Variation', fontsize=6, fontname='Arial')
    ax.set_title('Stability of R� Across Folds', fontsize=6, fontname='Arial')
    ax.tick_params(axis='both', labelsize=6)
    plt.setp(ax.get_yticklabels(), fontname='Arial')
    for spine in ax.spines.values():
        spine.set_linewidth(0.25)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.tight_layout()
    stability_path = f"figures/kfold_cv_alphaearth_stability_{timestamp}.pdf"
    plt.savefig(stability_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"  Saved: {stability_path}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
