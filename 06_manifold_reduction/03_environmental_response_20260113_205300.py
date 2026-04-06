#!/usr/bin/env python3
"""
Phase 3: Environmental Response Screening.

This script performs univariate and multivariate correlation analysis
between environmental variables and PFAM domain abundances.

Provenance:
  Input: Processed data from Phase 1
  Date: 2026-01-13
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Data integrity enforcement (CLAUDE.md requirement)
from DataIntegrityGuard import enforce_data_integrity
enforce_data_integrity()

# Paths
BASE_DIR = Path("/media/drn/External1/TARA-Oceans/03_analyses/env_pfam_manifold")
DATA_DIR = BASE_DIR / "data"
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

def compute_correlations(env_matrix, pfam_matrix, env_cols, pfam_cols):
    """Compute Spearman correlations between env vars and PFAM domains."""
    print("\n" + "=" * 70)
    print("COMPUTING CORRELATIONS")
    print("=" * 70)

    n_env = env_matrix.shape[1]
    n_pfam = pfam_matrix.shape[1]

    print(f"  Computing {n_env} x {n_pfam} = {n_env * n_pfam:,} correlations...")

    # Initialize correlation and p-value matrices
    corr_matrix = np.zeros((n_env, n_pfam))
    pval_matrix = np.zeros((n_env, n_pfam))

    for i in range(n_env):
        if (i + 1) % 5 == 0:
            print(f"    Processing env var {i+1}/{n_env}...")

        env_var = env_matrix[:, i]

        # Skip if all NaN or constant
        if np.isnan(env_var).all() or np.std(env_var[~np.isnan(env_var)]) == 0:
            corr_matrix[i, :] = np.nan
            pval_matrix[i, :] = np.nan
            continue

        for j in range(n_pfam):
            pfam_var = pfam_matrix[:, j]

            # Get valid pairs
            valid = ~np.isnan(env_var) & ~np.isnan(pfam_var)
            if valid.sum() < 10:
                corr_matrix[i, j] = np.nan
                pval_matrix[i, j] = np.nan
                continue

            rho, pval = spearmanr(env_var[valid], pfam_var[valid])
            corr_matrix[i, j] = rho
            pval_matrix[i, j] = pval

    print(f"  Done. Non-NaN correlations: {(~np.isnan(corr_matrix)).sum():,}")

    return corr_matrix, pval_matrix

def apply_fdr_correction(pval_matrix, alpha=0.05):
    """Apply FDR correction for multiple testing."""
    print("\n" + "=" * 70)
    print("APPLYING FDR CORRECTION")
    print("=" * 70)

    # Flatten, removing NaN
    pvals_flat = pval_matrix.flatten()
    valid_mask = ~np.isnan(pvals_flat)
    pvals_valid = pvals_flat[valid_mask]

    print(f"  Total p-values: {len(pvals_flat):,}")
    print(f"  Valid p-values: {len(pvals_valid):,}")

    # Apply Benjamini-Hochberg
    reject, qvals, _, _ = multipletests(pvals_valid, alpha=alpha, method='fdr_bh')

    # Reconstruct q-value matrix
    qval_matrix = np.full_like(pval_matrix, np.nan)
    qval_matrix.flat[valid_mask] = qvals

    significant_matrix = np.full_like(pval_matrix, False, dtype=bool)
    significant_matrix.flat[valid_mask] = reject

    n_significant = reject.sum()
    print(f"  Significant after FDR (q<{alpha}): {n_significant:,} ({100*n_significant/len(pvals_valid):.2f}%)")

    return qval_matrix, significant_matrix

def identify_top_correlations(corr_matrix, pval_matrix, qval_matrix, env_cols, pfam_cols, top_n=100):
    """Identify top correlations."""
    print("\n" + "=" * 70)
    print(f"IDENTIFYING TOP {top_n} CORRELATIONS")
    print("=" * 70)

    results = []

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            rho = corr_matrix[i, j]
            pval = pval_matrix[i, j]
            qval = qval_matrix[i, j]

            if not np.isnan(rho):
                results.append({
                    'env_var': env_cols[i],
                    'pfam': pfam_cols[j],
                    'rho': rho,
                    'pvalue': pval,
                    'qvalue': qval,
                    'abs_rho': abs(rho)
                })

    df = pd.DataFrame(results)
    df = df.sort_values('abs_rho', ascending=False)

    print(f"\n  Top 20 correlations:")
    print(df.head(20)[['env_var', 'pfam', 'rho', 'qvalue']].to_string(index=False))

    return df

def plot_correlation_heatmap(corr_matrix, env_cols, pfam_cols, top_n_pfam=50, output_path=None):
    """Plot heatmap of top correlated PFAM domains."""
    print("\n" + "=" * 70)
    print("CREATING CORRELATION HEATMAP")
    print("=" * 70)

    # Find top variable PFAM domains (by max absolute correlation)
    max_abs_corr = np.nanmax(np.abs(corr_matrix), axis=0)
    top_pfam_idx = np.argsort(max_abs_corr)[-top_n_pfam:]

    # Subset correlation matrix
    corr_subset = corr_matrix[:, top_pfam_idx]
    pfam_subset = [pfam_cols[i] for i in top_pfam_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))

    sns.heatmap(
        corr_subset,
        xticklabels=[p.split('.')[0] for p in pfam_subset],
        yticklabels=env_cols,
        cmap='RdBu_r',
        center=0,
        vmin=-0.5,
        vmax=0.5,
        ax=ax
    )

    ax.set_title(f'Spearman Correlations: Environmental Variables vs Top {top_n_pfam} PFAM Domains')
    ax.set_xlabel('PFAM Domain')
    ax.set_ylabel('Environmental Variable')

    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()

def plot_top_correlations(top_corr_df, n_plots=9, output_path=None):
    """Plot scatter plots of top correlations."""
    print("\n  Creating scatter plots for top correlations...")

    # This requires original data, not matrices
    # Skip for now, placeholder
    pass

def save_correlation_results(corr_matrix, pval_matrix, qval_matrix, top_corr_df, env_cols, pfam_cols):
    """Save all correlation results."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save correlation matrix
    corr_df = pd.DataFrame(corr_matrix, index=env_cols, columns=pfam_cols)
    corr_df.to_csv(REPORTS_DIR / f"correlation_matrix_{TIMESTAMP}.csv")
    print(f"  Saved correlation matrix")

    # Save top correlations
    top_corr_df.to_csv(REPORTS_DIR / f"top_correlations_{TIMESTAMP}.csv", index=False)
    print(f"  Saved top correlations ({len(top_corr_df)} entries)")

    # Summary statistics - convert numpy types to Python native types
    summary = {
        'timestamp': TIMESTAMP,
        'n_env_vars': int(len(env_cols)),
        'n_pfam_domains': int(len(pfam_cols)),
        'n_tests': int((~np.isnan(corr_matrix)).sum()),
        'n_significant_q05': int((qval_matrix < 0.05).sum()) if qval_matrix is not None else 0,
        'n_significant_q01': int((qval_matrix < 0.01).sum()) if qval_matrix is not None else 0,
        'max_abs_correlation': float(np.nanmax(np.abs(corr_matrix))),
        'mean_abs_correlation': float(np.nanmean(np.abs(corr_matrix)))
    }

    import json
    with open(REPORTS_DIR / f"correlation_summary_{TIMESTAMP}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved summary statistics")

def main():
    print("=" * 70)
    print("PHASE 3: ENVIRONMENTAL RESPONSE SCREENING")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load data
    env_matrix, pfam_matrix, env_cols, pfam_cols = load_processed_data()

    # Compute correlations
    corr_matrix, pval_matrix = compute_correlations(env_matrix, pfam_matrix, env_cols, pfam_cols)

    # Apply FDR correction
    qval_matrix, significant_matrix = apply_fdr_correction(pval_matrix)

    # Identify top correlations
    top_corr_df = identify_top_correlations(corr_matrix, pval_matrix, qval_matrix, env_cols, pfam_cols)

    # Plot heatmap
    plot_correlation_heatmap(
        corr_matrix, env_cols, pfam_cols,
        output_path=FIGURES_DIR / f"correlation_heatmap_{TIMESTAMP}.png"
    )

    # Save results
    save_correlation_results(corr_matrix, pval_matrix, qval_matrix, top_corr_df, env_cols, pfam_cols)

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)

    return corr_matrix, top_corr_df

if __name__ == "__main__":
    corr_matrix, top_corr_df = main()
