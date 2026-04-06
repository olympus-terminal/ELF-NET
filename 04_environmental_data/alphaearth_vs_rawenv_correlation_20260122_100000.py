#!/usr/bin/env python3
"""
AlphaEarth vs Raw Environmental Variables Correlation Analysis

Purpose: Compute correlations between AlphaEarth embeddings (64 dims) and
         raw environmental variables (29 vars) to assess redundancy.

Inputs:
    - AlphaEarth embeddings: PythiaTIfreeLA4SR_TARA/alphaearth_embeddings_*.tsv
    - GEE raw env vars: ALGAGPT-based-analyses/algagpt_gee_pfam_merged_SMART_*.tsv

Output:
    - Correlation matrix (64 x 29)
    - Summary statistics
    - Heatmap visualization

Date: 2026-01-22
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
from datetime import datetime
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = "/media/drn2/External/TARA-Oceans"
OUTPUT_DIR = f"{BASE_DIR}/03_analyses"

# Input files
ALPHAEARTH_FILE = f"{BASE_DIR}/PythiaTIfreeLA4SR_TARA/alphaearth_embeddings_20260114_113823.tsv"
GEE_FILE = f"{BASE_DIR}/03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"

# Raw environmental variable columns (29 total)
RAW_ENV_COLS = [
    'air_temp_mean_c', 'air_temp_max_c', 'air_temp_min_c', 'air_temp_range_c',
    'precip_mean_mm', 'solar_rad_mj_m2', 'elevation_m', 'bathymetry_m',
    'distance_to_coast_km', 'sst_mean_c', 'sst_max_c', 'sst_min_c', 'sst_range_c',
    'chl_mean_mg_m3', 'chl_max_mg_m3', 'chl_min_mg_m3', 'nflh_mean', 'poc_mean_mg_m3',
    'modis_sst_mean_c', 'rrs_412', 'rrs_443', 'rrs_469', 'rrs_488', 'rrs_531',
    'rrs_547', 'rrs_555', 'rrs_645', 'rrs_667', 'rrs_678'
]

# AlphaEarth dimension columns (64 total)
ALPHAEARTH_COLS = [f'A{i:02d}' for i in range(64)]

def load_alphaearth_embeddings(filepath):
    """Load AlphaEarth embeddings, skipping comment lines."""
    print(f"Loading AlphaEarth embeddings from: {filepath}")

    # Read file, skip comment lines
    df = pd.read_csv(filepath, sep='\t', comment='#')

    print(f"  Loaded {len(df)} samples")
    print(f"  Columns: {list(df.columns[:5])} ... {list(df.columns[-3:])}")

    return df

def load_gee_data(filepath):
    """Load GEE merged data with raw environmental variables."""
    print(f"Loading GEE data from: {filepath}")

    # Read file, skip comment lines
    df = pd.read_csv(filepath, sep='\t', comment='#')

    print(f"  Loaded {len(df)} samples")

    # Check which env columns exist
    existing_env_cols = [c for c in RAW_ENV_COLS if c in df.columns]
    missing_env_cols = [c for c in RAW_ENV_COLS if c not in df.columns]

    print(f"  Found {len(existing_env_cols)}/{len(RAW_ENV_COLS)} env columns")
    if missing_env_cols:
        print(f"  Missing: {missing_env_cols}")

    return df, existing_env_cols

def merge_datasets(alphaearth_df, gee_df):
    """Merge AlphaEarth and GEE data by assembly_id."""
    print("\nMerging datasets...")

    # Standardize column names for merging
    if 'assembly_id' in alphaearth_df.columns:
        merge_col = 'assembly_id'
    elif 'sample_id' in alphaearth_df.columns:
        alphaearth_df = alphaearth_df.rename(columns={'sample_id': 'assembly_id'})
        merge_col = 'assembly_id'
    else:
        # First column is likely the ID
        first_col = alphaearth_df.columns[0]
        alphaearth_df = alphaearth_df.rename(columns={first_col: 'assembly_id'})
        merge_col = 'assembly_id'

    # Merge on assembly_id
    merged = pd.merge(
        alphaearth_df,
        gee_df,
        on='assembly_id',
        how='inner'
    )

    print(f"  AlphaEarth samples: {len(alphaearth_df)}")
    print(f"  GEE samples: {len(gee_df)}")
    print(f"  Merged samples: {len(merged)}")

    return merged

def compute_correlations(df, alphaearth_cols, env_cols):
    """Compute Spearman correlations between AlphaEarth dims and env vars."""
    print(f"\nComputing correlations: {len(alphaearth_cols)} x {len(env_cols)} = {len(alphaearth_cols) * len(env_cols)} tests")

    results = []

    for alpha_col in alphaearth_cols:
        if alpha_col not in df.columns:
            continue

        for env_col in env_cols:
            if env_col not in df.columns:
                continue

            # Get paired non-null values
            mask = df[alpha_col].notna() & df[env_col].notna()
            x = df.loc[mask, alpha_col].values
            y = df.loc[mask, env_col].values

            if len(x) < 10:  # Skip if too few samples
                continue

            # Spearman correlation
            rho, pval = stats.spearmanr(x, y)

            results.append({
                'alphaearth_dim': alpha_col,
                'env_var': env_col,
                'spearman_rho': rho,
                'pvalue': pval,
                'n_samples': len(x)
            })

    corr_df = pd.DataFrame(results)
    print(f"  Computed {len(corr_df)} correlations")

    return corr_df

def apply_fdr_correction(corr_df):
    """Apply Benjamini-Hochberg FDR correction."""
    print("\nApplying FDR correction...")

    # FDR correction
    reject, qvals, _, _ = multipletests(corr_df['pvalue'], method='fdr_bh')
    corr_df['qvalue'] = qvals
    corr_df['significant_fdr05'] = qvals < 0.05
    corr_df['significant_fdr01'] = qvals < 0.01

    n_sig_05 = corr_df['significant_fdr05'].sum()
    n_sig_01 = corr_df['significant_fdr01'].sum()

    print(f"  Significant (FDR < 0.05): {n_sig_05} ({100*n_sig_05/len(corr_df):.1f}%)")
    print(f"  Significant (FDR < 0.01): {n_sig_01} ({100*n_sig_01/len(corr_df):.1f}%)")

    return corr_df

def create_correlation_matrix(corr_df):
    """Create pivot table for correlation matrix."""
    matrix = corr_df.pivot(
        index='alphaearth_dim',
        columns='env_var',
        values='spearman_rho'
    )
    return matrix

def summarize_results(corr_df, matrix):
    """Generate summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY: AlphaEarth vs Raw Environmental Variables Correlations")
    print("="*70)

    # Overall statistics
    print(f"\nTotal correlations computed: {len(corr_df)}")
    print(f"Significant (FDR < 0.05): {corr_df['significant_fdr05'].sum()} ({100*corr_df['significant_fdr05'].mean():.1f}%)")
    print(f"Significant (FDR < 0.01): {corr_df['significant_fdr01'].sum()} ({100*corr_df['significant_fdr01'].mean():.1f}%)")

    # Correlation magnitude
    print(f"\nCorrelation magnitude:")
    print(f"  Max |rho|: {corr_df['spearman_rho'].abs().max():.4f}")
    print(f"  Mean |rho|: {corr_df['spearman_rho'].abs().mean():.4f}")
    print(f"  Median |rho|: {corr_df['spearman_rho'].abs().median():.4f}")

    # Top 10 strongest correlations
    print("\nTop 10 strongest correlations:")
    top10 = corr_df.nlargest(10, 'spearman_rho', keep='first')[['alphaearth_dim', 'env_var', 'spearman_rho', 'qvalue']]
    print(top10.to_string(index=False))

    # Bottom 10 (strongest negative)
    print("\nTop 10 strongest negative correlations:")
    bottom10 = corr_df.nsmallest(10, 'spearman_rho', keep='first')[['alphaearth_dim', 'env_var', 'spearman_rho', 'qvalue']]
    print(bottom10.to_string(index=False))

    # Per-dimension summary
    dim_summary = corr_df.groupby('alphaearth_dim').agg({
        'spearman_rho': ['mean', lambda x: x.abs().max()],
        'significant_fdr05': 'sum'
    }).round(4)
    dim_summary.columns = ['mean_rho', 'max_abs_rho', 'n_significant']
    dim_summary = dim_summary.sort_values('max_abs_rho', ascending=False)

    print("\nTop 10 AlphaEarth dimensions by max correlation:")
    print(dim_summary.head(10).to_string())

    # Per-env-var summary
    env_summary = corr_df.groupby('env_var').agg({
        'spearman_rho': ['mean', lambda x: x.abs().max()],
        'significant_fdr05': 'sum'
    }).round(4)
    env_summary.columns = ['mean_rho', 'max_abs_rho', 'n_significant']
    env_summary = env_summary.sort_values('max_abs_rho', ascending=False)

    print("\nTop 10 env variables by max correlation with AlphaEarth:")
    print(env_summary.head(10).to_string())

    # Redundancy assessment
    print("\n" + "="*70)
    print("REDUNDANCY ASSESSMENT")
    print("="*70)

    high_corr = (corr_df['spearman_rho'].abs() > 0.5).sum()
    moderate_corr = ((corr_df['spearman_rho'].abs() > 0.3) & (corr_df['spearman_rho'].abs() <= 0.5)).sum()
    low_corr = (corr_df['spearman_rho'].abs() <= 0.3).sum()

    print(f"\nCorrelation strength distribution:")
    print(f"  High (|rho| > 0.5):     {high_corr} ({100*high_corr/len(corr_df):.1f}%)")
    print(f"  Moderate (0.3-0.5):    {moderate_corr} ({100*moderate_corr/len(corr_df):.1f}%)")
    print(f"  Low (|rho| <= 0.3):    {low_corr} ({100*low_corr/len(corr_df):.1f}%)")

    if high_corr > len(corr_df) * 0.1:
        print("\n  --> CONCLUSION: SUBSTANTIAL REDUNDANCY detected")
        print("      AlphaEarth captures similar information to raw env vars")
        print("      Ternary CCA may not add much value")
    elif moderate_corr > len(corr_df) * 0.3:
        print("\n  --> CONCLUSION: MODERATE OVERLAP detected")
        print("      AlphaEarth partially overlaps with raw env vars")
        print("      Consider concatenation (Option B) over ternary CCA")
    else:
        print("\n  --> CONCLUSION: LIMITED REDUNDANCY")
        print("      AlphaEarth captures different information")
        print("      Ternary CCA may reveal additional structure")

    return dim_summary, env_summary

def save_results(corr_df, matrix, dim_summary, env_summary):
    """Save all results to files."""
    print("\nSaving results...")

    # Full correlation table
    out_full = f"{OUTPUT_DIR}/alphaearth_rawenv_correlations_{TIMESTAMP}_full.tsv"
    corr_df.to_csv(out_full, sep='\t', index=False)
    print(f"  Full correlations: {out_full}")

    # Correlation matrix
    out_matrix = f"{OUTPUT_DIR}/alphaearth_rawenv_correlations_{TIMESTAMP}_matrix.tsv"
    matrix.to_csv(out_matrix, sep='\t')
    print(f"  Correlation matrix: {out_matrix}")

    # Significant only
    sig_df = corr_df[corr_df['significant_fdr05']].sort_values('spearman_rho', ascending=False)
    out_sig = f"{OUTPUT_DIR}/alphaearth_rawenv_correlations_{TIMESTAMP}_significant.tsv"
    sig_df.to_csv(out_sig, sep='\t', index=False)
    print(f"  Significant only: {out_sig}")

    # Summary
    out_summary = f"{OUTPUT_DIR}/alphaearth_rawenv_correlations_{TIMESTAMP}_summary.txt"
    with open(out_summary, 'w') as f:
        f.write("AlphaEarth vs Raw Environmental Variables Correlation Summary\n")
        f.write(f"Date: {TIMESTAMP}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total correlations: {len(corr_df)}\n")
        f.write(f"Significant (FDR < 0.05): {corr_df['significant_fdr05'].sum()}\n")
        f.write(f"Max |rho|: {corr_df['spearman_rho'].abs().max():.4f}\n")
        f.write(f"Mean |rho|: {corr_df['spearman_rho'].abs().mean():.4f}\n")
        f.write("\n\nTop AlphaEarth dims by max correlation:\n")
        f.write(dim_summary.head(10).to_string())
        f.write("\n\nTop env vars by max correlation with AlphaEarth:\n")
        f.write(env_summary.head(10).to_string())
    print(f"  Summary: {out_summary}")

    return out_full, out_matrix, out_sig, out_summary

def main():
    """Main execution."""
    print("="*70)
    print("AlphaEarth vs Raw Environmental Variables Correlation Analysis")
    print("="*70)
    print(f"Timestamp: {TIMESTAMP}")

    # Load data
    alphaearth_df = load_alphaearth_embeddings(ALPHAEARTH_FILE)
    gee_df, existing_env_cols = load_gee_data(GEE_FILE)

    # Merge datasets
    merged_df = merge_datasets(alphaearth_df, gee_df)

    if len(merged_df) < 50:
        print(f"\nERROR: Only {len(merged_df)} samples after merge - insufficient for correlation analysis")
        print("Need to match samples by GPS coordinates instead of assembly_id")
        return

    # Compute correlations
    corr_df = compute_correlations(merged_df, ALPHAEARTH_COLS, existing_env_cols)

    if len(corr_df) == 0:
        print("\nERROR: No valid correlations computed")
        return

    # FDR correction
    corr_df = apply_fdr_correction(corr_df)

    # Create matrix
    matrix = create_correlation_matrix(corr_df)

    # Summarize
    dim_summary, env_summary = summarize_results(corr_df, matrix)

    # Save results
    save_results(corr_df, matrix, dim_summary, env_summary)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
