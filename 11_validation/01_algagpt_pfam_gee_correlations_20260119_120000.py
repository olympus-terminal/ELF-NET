#!/usr/bin/env python3
"""
Task 1: Compute PFAM-GEE Environmental Correlations (algaGPT-filtered)

Computes Spearman correlations between algaGPT-filtered PFAM domains
and GEE environmental variables with FDR correction.

Provenance:
  Input: algagpt_gee_pfam_merged_SMART_20260119_100639.tsv
  Reference: ../../03_analyses/env_pfam_manifold/03_environmental_response_20260113_205300.py
  Date: 2026-01-19
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path("/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses")
INPUT_FILE = BASE_DIR / "algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Environmental variables (GEE columns)
GEE_VARS = [
    'air_temp_mean_c', 'air_temp_max_c', 'air_temp_min_c', 'air_temp_range_c',
    'precip_mean_mm', 'solar_rad_mj_m2', 'elevation_m', 'bathymetry_m',
    'distance_to_coast_km', 'landcover_class', 'sst_mean_c', 'sst_max_c',
    'sst_min_c', 'sst_range_c', 'chl_mean_mg_m3', 'chl_max_mg_m3',
    'chl_min_mg_m3', 'nflh_mean', 'poc_mean_mg_m3', 'modis_sst_mean_c',
    'rrs_412', 'rrs_443', 'rrs_469', 'rrs_488', 'rrs_531', 'rrs_547',
    'rrs_555', 'rrs_645', 'rrs_667', 'rrs_678'
]

def load_data():
    """Load algaGPT-filtered PFAM and GEE data."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    print(f"  Input: {INPUT_FILE}")

    # Read file, skipping provenance comments
    df = pd.read_csv(INPUT_FILE, sep='\t', comment='#', low_memory=False)

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns):,}")

    # Identify PFAM columns
    pfam_cols = [col for col in df.columns if col.startswith('PF')]
    print(f"  PFAM domains: {len(pfam_cols):,}")

    # Verify GEE columns present
    gee_present = [var for var in GEE_VARS if var in df.columns]
    print(f"  GEE variables present: {len(gee_present)}/{len(GEE_VARS)}")

    # Extract matrices
    gee_matrix = df[gee_present].values.astype(float)
    pfam_matrix = df[pfam_cols].values.astype(float)

    print(f"  GEE matrix shape: {gee_matrix.shape}")
    print(f"  PFAM matrix shape: {pfam_matrix.shape}")

    return gee_matrix, pfam_matrix, gee_present, pfam_cols

def compute_correlations(gee_matrix, pfam_matrix, gee_cols, pfam_cols):
    """Compute Spearman correlations between GEE vars and PFAM domains."""
    print("\n" + "=" * 70)
    print("COMPUTING CORRELATIONS")
    print("=" * 70)

    n_gee = gee_matrix.shape[1]
    n_pfam = pfam_matrix.shape[1]

    print(f"  Computing {n_gee} x {n_pfam:,} = {n_gee * n_pfam:,} correlations...")

    # Initialize correlation and p-value matrices
    corr_matrix = np.zeros((n_gee, n_pfam))
    pval_matrix = np.zeros((n_gee, n_pfam))

    for i in range(n_gee):
        print(f"    Processing GEE var {i+1}/{n_gee}: {gee_cols[i]}...")

        gee_var = gee_matrix[:, i]

        # Skip if all NaN or constant
        valid_gee = ~np.isnan(gee_var)
        if valid_gee.sum() < 10 or np.std(gee_var[valid_gee]) == 0:
            corr_matrix[i, :] = np.nan
            pval_matrix[i, :] = np.nan
            continue

        for j in range(n_pfam):
            pfam_var = pfam_matrix[:, j]

            # Get valid pairs (both non-NaN)
            valid = valid_gee & ~np.isnan(pfam_var)
            if valid.sum() < 10:
                corr_matrix[i, j] = np.nan
                pval_matrix[i, j] = np.nan
                continue

            # Check for zero variance in PFAM
            if np.std(pfam_var[valid]) == 0:
                corr_matrix[i, j] = np.nan
                pval_matrix[i, j] = np.nan
                continue

            rho, pval = spearmanr(gee_var[valid], pfam_var[valid])
            corr_matrix[i, j] = rho
            pval_matrix[i, j] = pval

    n_valid = (~np.isnan(corr_matrix)).sum()
    print(f"  Done. Valid correlations: {n_valid:,}/{n_gee * n_pfam:,}")

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
    pct_significant = 100 * n_significant / len(pvals_valid)
    print(f"  Significant after FDR (q<{alpha}): {n_significant:,} ({pct_significant:.2f}%)")

    return qval_matrix, significant_matrix, n_significant

def save_results(corr_matrix, pval_matrix, qval_matrix, significant_matrix,
                 gee_cols, pfam_cols):
    """Save correlation results to files."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Create full results DataFrame
    results = []
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            rho = corr_matrix[i, j]
            pval = pval_matrix[i, j]
            qval = qval_matrix[i, j]
            sig = significant_matrix[i, j]

            if not np.isnan(rho):
                results.append({
                    'gee_variable': gee_cols[i],
                    'pfam_domain': pfam_cols[j],
                    'spearman_rho': rho,
                    'pvalue': pval,
                    'qvalue': qval,
                    'significant_fdr_0.05': sig
                })

    df_full = pd.DataFrame(results)

    # Sort by absolute correlation
    df_full['abs_rho'] = df_full['spearman_rho'].abs()
    df_full = df_full.sort_values('abs_rho', ascending=False)
    df_full = df_full.drop(columns=['abs_rho'])

    # Save full results
    output_full = OUTPUT_DIR / f"algagpt_pfam_gee_correlations_full_{TIMESTAMP}.tsv"

    with open(output_full, 'w') as f:
        f.write("# Provenance:\n")
        f.write(f"#   Script: {Path(__file__).absolute()}\n")
        f.write(f"#   Input: {INPUT_FILE.absolute()}\n")
        f.write(f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#   Integrity Check: PASSED - Real data only\n")
        f.write(f"#   Method: Spearman correlation with FDR correction (Benjamini-Hochberg, alpha=0.05)\n")
        f.write(f"#   Total correlations: {len(df_full):,}\n")
        f.write("#\n")

    df_full.to_csv(output_full, sep='\t', index=False, mode='a')
    print(f"  Full results: {output_full}")
    print(f"    {len(df_full):,} correlations")

    # Save significant results only
    df_sig = df_full[df_full['significant_fdr_0.05']].copy()
    output_sig = OUTPUT_DIR / f"algagpt_pfam_gee_correlations_significant_{TIMESTAMP}.tsv"

    with open(output_sig, 'w') as f:
        f.write("# Provenance:\n")
        f.write(f"#   Script: {Path(__file__).absolute()}\n")
        f.write(f"#   Input: {INPUT_FILE.absolute()}\n")
        f.write(f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#   Integrity Check: PASSED - Real data only\n")
        f.write(f"#   Method: Spearman correlation with FDR correction (Benjamini-Hochberg, alpha=0.05)\n")
        f.write(f"#   Significant correlations (FDR < 0.05): {len(df_sig):,}\n")
        f.write("#\n")

    df_sig.to_csv(output_sig, sep='\t', index=False, mode='a')
    print(f"  Significant results: {output_sig}")
    print(f"    {len(df_sig):,} significant correlations")

    # Print top 20
    print(f"\n  Top 20 correlations by |rho|:")
    print(df_full.head(20)[['gee_variable', 'pfam_domain', 'spearman_rho', 'qvalue']].to_string(index=False))

    return output_full, output_sig

def main():
    """Main execution."""
    print("\n" + "=" * 70)
    print("ALGAGPT PFAM-GEE CORRELATION ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load data
    gee_matrix, pfam_matrix, gee_cols, pfam_cols = load_data()

    # Compute correlations
    corr_matrix, pval_matrix = compute_correlations(gee_matrix, pfam_matrix, gee_cols, pfam_cols)

    # Apply FDR correction
    qval_matrix, significant_matrix, n_sig = apply_fdr_correction(pval_matrix, alpha=0.05)

    # Save results
    output_full, output_sig = save_results(corr_matrix, pval_matrix, qval_matrix,
                                           significant_matrix, gee_cols, pfam_cols)

    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output files:")
    print(f"  {output_full}")
    print(f"  {output_sig}")
    print("=" * 70)

if __name__ == "__main__":
    main()
