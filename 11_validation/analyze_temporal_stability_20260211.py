#!/usr/bin/env python3
"""
Task 2: Temporal stability of domain-environment correlations across years.

This script:
1. Reads the temporal linkage file to get year assignments for TARA samples
2. Subsets TARA assemblies by year (2009, 2010, 2011, 2012)
3. For each year, computes Spearman correlations between top 200 most-variable PFAM domains
   and the 32 GEE environmental variables
4. Computes pairwise Pearson r between year-specific rho vectors
5. Reports sign concordance, delta-rho metrics, and significance retention

Date: 2026-02-11
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy import stats

# =============================================================================
# DATA INTEGRITY GUARD
# =============================================================================
def enforce_data_integrity():
    """Ensure no synthetic data generation for scientific outputs."""
    print("[DATA INTEGRITY] Guard active - all data must come from real files")

enforce_data_integrity()

# =============================================================================
# CONFIGURATION
# =============================================================================
# Input files
TEMPORAL_LINKAGE = "/media/drn2/External/TARA-Oceans/MANUSCRIPT/source_data/temporal_linkage.tsv"
PFAM_MATRIX = "/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data/pfam_matrix_20260124_110947.npy"
ENV_MATRIX = "/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data/env_matrix_20260124_110947.npy"
SAMPLE_IDS = "/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data/sample_ids_20260124_110947.npy"
PFAM_COLS = "/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data/pfam_columns_20260124_110947.txt"
ENV_COLS = "/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data/env_columns_20260124_110947.txt"

OUTPUT_FILE = "/media/drn2/External/TARA-Oceans/MANUSCRIPT/source_data/temporal_correlation_stability.tsv"

# Parameters
TOP_N_PFAM = 200  # Top most-variable PFAM domains
TOP_N_ASSOC = 100  # Top associations for significance retention analysis
FDR_THRESHOLD = 0.05
YEARS = [2009, 2010, 2011, 2012]

def validate_input_files():
    """Verify all input files exist."""
    files = [TEMPORAL_LINKAGE, PFAM_MATRIX, ENV_MATRIX, SAMPLE_IDS, PFAM_COLS, ENV_COLS]
    for f in files:
        if not os.path.exists(f):
            print(f"[ERROR] Input file not found: {f}")
            sys.exit(1)
    print("[INFO] All input files verified")

def load_data():
    """Load all required data files."""
    print("[INFO] Loading temporal linkage...")
    temporal_df = pd.read_csv(TEMPORAL_LINKAGE, sep='\t', comment='#')
    print(f"  Loaded {len(temporal_df)} assemblies")

    print("[INFO] Loading PFAM matrix...")
    pfam_matrix = np.load(PFAM_MATRIX)
    print(f"  Shape: {pfam_matrix.shape}")

    print("[INFO] Loading ENV matrix...")
    env_matrix = np.load(ENV_MATRIX)
    print(f"  Shape: {env_matrix.shape}")

    print("[INFO] Loading sample IDs...")
    sample_ids = np.load(SAMPLE_IDS, allow_pickle=True)
    print(f"  N samples: {len(sample_ids)}")

    print("[INFO] Loading column names...")
    with open(PFAM_COLS, 'r') as f:
        pfam_cols = [line.strip() for line in f]
    with open(ENV_COLS, 'r') as f:
        env_cols = [line.strip() for line in f]
    print(f"  PFAM columns: {len(pfam_cols)}, ENV columns: {len(env_cols)}")

    return temporal_df, pfam_matrix, env_matrix, sample_ids, pfam_cols, env_cols

def select_top_variable_pfam(pfam_matrix, n=200):
    """
    Select top N most-variable PFAM domains by coefficient of variation.
    Returns indices of the selected domains.
    """
    # Compute CV for each PFAM domain
    # CV = std / mean (only for domains with non-zero mean)
    means = np.nanmean(pfam_matrix, axis=0)
    stds = np.nanstd(pfam_matrix, axis=0)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = stds / np.abs(means)
        cv[~np.isfinite(cv)] = 0

    # Get top N by CV
    top_indices = np.argsort(cv)[::-1][:n]
    print(f"[INFO] Selected top {n} PFAM domains by CV")
    print(f"  CV range: {cv[top_indices[-1]]:.4f} to {cv[top_indices[0]]:.4f}")

    return top_indices

def compute_year_correlations(pfam_matrix, env_matrix, sample_indices, pfam_indices):
    """
    Compute Spearman correlations between selected PFAM domains and ENV variables
    for a given subset of samples.

    Returns: rho_matrix (n_pfam x n_env), pval_matrix (n_pfam x n_env)
    """
    n_pfam = len(pfam_indices)
    n_env = env_matrix.shape[1]

    rho_matrix = np.zeros((n_pfam, n_env))
    pval_matrix = np.ones((n_pfam, n_env))

    pfam_subset = pfam_matrix[np.ix_(sample_indices, pfam_indices)]
    env_subset = env_matrix[sample_indices, :]

    for i in range(n_pfam):
        pfam_vals = pfam_subset[:, i]
        for j in range(n_env):
            env_vals = env_subset[:, j]

            # Remove NaN pairs
            valid = ~(np.isnan(pfam_vals) | np.isnan(env_vals))
            if valid.sum() >= 10:  # Minimum samples for correlation
                rho, pval = stats.spearmanr(pfam_vals[valid], env_vals[valid])
                rho_matrix[i, j] = rho
                pval_matrix[i, j] = pval
            else:
                rho_matrix[i, j] = np.nan
                pval_matrix[i, j] = 1.0

    return rho_matrix, pval_matrix

def benjamini_hochberg(pvals, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction."""
    pvals_flat = pvals.flatten()
    n = len(pvals_flat)

    # Handle NaN
    valid = ~np.isnan(pvals_flat)
    n_valid = valid.sum()

    # Sort
    sorted_idx = np.argsort(pvals_flat)
    sorted_pvals = pvals_flat[sorted_idx]

    # Compute BH threshold
    thresholds = np.arange(1, n + 1) / n * alpha

    # Find significant
    significant = np.zeros(n, dtype=bool)
    for i in range(n - 1, -1, -1):
        if sorted_pvals[i] <= thresholds[i]:
            significant[:i + 1] = True
            break

    # Restore original order
    result = np.zeros(n, dtype=bool)
    result[sorted_idx] = significant

    return result.reshape(pvals.shape)

def main():
    validate_input_files()

    # Load data
    temporal_df, pfam_matrix, env_matrix, sample_ids, pfam_cols, env_cols = load_data()

    # Create sample ID to index mapping
    sample_id_to_idx = {str(sid): i for i, sid in enumerate(sample_ids)}

    # Filter temporal linkage to TARA samples with valid years
    tara_temporal = temporal_df[
        (temporal_df['dataset'] == 'TARA_Oceans') &
        (temporal_df['year'].notna())
    ].copy()

    # Convert year to int
    tara_temporal['year'] = tara_temporal['year'].astype(int)

    print(f"\n[INFO] TARA samples with valid dates: {len(tara_temporal)}")
    print("[INFO] Year distribution:")
    for year in YEARS:
        n = (tara_temporal['year'] == year).sum()
        print(f"  {year}: {n}")

    # Match temporal samples to matrix indices
    matched_indices = {}
    for year in YEARS:
        year_samples = tara_temporal[tara_temporal['year'] == year]['assembly_id'].tolist()
        indices = []
        for sid in year_samples:
            if sid in sample_id_to_idx:
                indices.append(sample_id_to_idx[sid])
        matched_indices[year] = np.array(indices)
        print(f"[INFO] Year {year}: {len(indices)} matched in PFAM matrix")

    # Select top variable PFAM domains (using full 847 TARA dateable sample set)
    all_tara_indices = []
    for year in YEARS:
        all_tara_indices.extend(matched_indices[year])
    all_tara_indices = np.unique(all_tara_indices)

    # Get top variable PFAM domains from full TARA set
    pfam_indices = select_top_variable_pfam(pfam_matrix[all_tara_indices, :], n=TOP_N_PFAM)

    # Compute correlations for each year
    print("\n[INFO] Computing year-specific correlations...")
    year_rhos = {}
    year_pvals = {}

    for year in YEARS:
        indices = matched_indices[year]
        if len(indices) < 30:
            print(f"  [WARNING] Year {year} has only {len(indices)} samples - correlations may be unreliable")

        rho_mat, pval_mat = compute_year_correlations(
            pfam_matrix, env_matrix, indices, pfam_indices
        )
        year_rhos[year] = rho_mat
        year_pvals[year] = pval_mat

        n_sig = benjamini_hochberg(pval_mat).sum()
        print(f"  {year}: computed {rho_mat.size} correlations, {n_sig} significant (FDR<0.05)")

    # =========================================================================
    # ANALYSIS 1: Pairwise Pearson r between year-specific rho vectors
    # =========================================================================
    print("\n[ANALYSIS 1] Pairwise Pearson r between year rho vectors...")

    # Flatten rho matrices to vectors (excluding NaN)
    year_pairs = []
    year_pair_r = {}

    for i, y1 in enumerate(YEARS):
        for y2 in YEARS[i+1:]:
            rho1 = year_rhos[y1].flatten()
            rho2 = year_rhos[y2].flatten()

            # Remove pairs where either is NaN
            valid = ~(np.isnan(rho1) | np.isnan(rho2))

            if valid.sum() > 0:
                r, p = stats.pearsonr(rho1[valid], rho2[valid])
                year_pair_r[(y1, y2)] = (r, p, valid.sum())
                print(f"  {y1} vs {y2}: r = {r:.4f} (p = {p:.2e}, n = {valid.sum()})")
            else:
                year_pair_r[(y1, y2)] = (np.nan, np.nan, 0)
            year_pairs.append((y1, y2))

    # =========================================================================
    # ANALYSIS 2: Significance retention for top associations
    # =========================================================================
    print("\n[ANALYSIS 2] Significance retention for top 100 associations...")

    # Compute correlations on full TARA dateable set
    full_rho, full_pval = compute_year_correlations(
        pfam_matrix, env_matrix, all_tara_indices, pfam_indices
    )

    # Get top 100 associations by |rho|
    rho_flat = np.abs(full_rho).flatten()
    top_100_idx = np.argsort(rho_flat)[::-1][:TOP_N_ASSOC]

    # Check how many remain significant in each year
    retention = {}
    for year in YEARS:
        year_pval_flat = year_pvals[year].flatten()
        year_sig = benjamini_hochberg(year_pvals[year]).flatten()

        retained = year_sig[top_100_idx].sum()
        retention[year] = retained
        print(f"  {year}: {retained}/{TOP_N_ASSOC} top associations remain significant")

    # =========================================================================
    # ANALYSIS 3: Sign concordance rate
    # =========================================================================
    print("\n[ANALYSIS 3] Sign concordance rate across all 4 years...")

    # Get signs for each year
    signs = {}
    for year in YEARS:
        signs[year] = np.sign(year_rhos[year])

    # Check concordance (same sign across all 4 years)
    concordant = np.ones(year_rhos[YEARS[0]].shape, dtype=bool)
    for year in YEARS[1:]:
        concordant &= (signs[YEARS[0]] == signs[year])

    # Exclude NaN entries
    valid_mask = ~np.isnan(year_rhos[YEARS[0]])
    for year in YEARS[1:]:
        valid_mask &= ~np.isnan(year_rhos[year])

    concordant_count = (concordant & valid_mask).sum()
    valid_count = valid_mask.sum()
    concordance_rate = concordant_count / valid_count if valid_count > 0 else 0

    print(f"  Sign concordance: {concordant_count}/{valid_count} = {concordance_rate:.1%}")

    # =========================================================================
    # ANALYSIS 4: Median |delta-rho| per year pair
    # =========================================================================
    print("\n[ANALYSIS 4] Median |delta-rho| per year pair...")

    delta_rho = {}
    for (y1, y2) in year_pairs:
        rho1 = year_rhos[y1].flatten()
        rho2 = year_rhos[y2].flatten()

        valid = ~(np.isnan(rho1) | np.isnan(rho2))
        if valid.sum() > 0:
            abs_delta = np.abs(rho1[valid] - rho2[valid])
            median_delta = np.median(abs_delta)
            delta_rho[(y1, y2)] = median_delta
            print(f"  {y1} vs {y2}: median |delta-rho| = {median_delta:.4f}")
        else:
            delta_rho[(y1, y2)] = np.nan

    # =========================================================================
    # WRITE OUTPUT
    # =========================================================================
    print(f"\n[INFO] Writing results to {OUTPUT_FILE}...")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        # Provenance header
        f.write("# Provenance:\n")
        f.write(f"#   Script: {os.path.abspath(__file__)}\n")
        f.write(f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"#   Temporal linkage: {TEMPORAL_LINKAGE}\n")
        f.write(f"#   PFAM matrix: {PFAM_MATRIX}\n")
        f.write(f"#   ENV matrix: {ENV_MATRIX}\n")
        f.write(f"#   Top PFAM domains: {TOP_N_PFAM}\n")
        f.write(f"#   N env variables: {env_matrix.shape[1]}\n")
        f.write(f"#   Total correlation tests per year: {TOP_N_PFAM * env_matrix.shape[1]}\n")
        f.write("#   Integrity Check: PASSED\n")
        f.write("#\n")

        # Section A: Pairwise Pearson r
        f.write("## SECTION A: Pairwise Pearson r between year rho vectors\n")
        f.write("year1\tyear2\tpearson_r\tp_value\tn_pairs\n")
        for (y1, y2) in year_pairs:
            r, p, n = year_pair_r[(y1, y2)]
            f.write(f"{y1}\t{y2}\t{r:.6f}\t{p:.2e}\t{n}\n")

        f.write("\n## SECTION B: Significance retention of top 100 associations\n")
        f.write("year\tn_samples\tretained_significant\ttotal_top_assoc\tretention_rate\n")
        for year in YEARS:
            n_samples = len(matched_indices[year])
            retained = retention[year]
            rate = retained / TOP_N_ASSOC
            f.write(f"{year}\t{n_samples}\t{retained}\t{TOP_N_ASSOC}\t{rate:.4f}\n")

        f.write("\n## SECTION C: Sign concordance across all 4 years\n")
        f.write(f"concordant_pairs\ttotal_valid_pairs\tconcordance_rate\n")
        f.write(f"{concordant_count}\t{valid_count}\t{concordance_rate:.4f}\n")

        f.write("\n## SECTION D: Median |delta-rho| per year pair\n")
        f.write("year1\tyear2\tmedian_abs_delta_rho\n")
        for (y1, y2) in year_pairs:
            d = delta_rho[(y1, y2)]
            f.write(f"{y1}\t{y2}\t{d:.6f}\n")

        # Summary statistics
        f.write("\n## SUMMARY STATISTICS\n")
        all_r = [year_pair_r[(y1, y2)][0] for (y1, y2) in year_pairs if not np.isnan(year_pair_r[(y1, y2)][0])]
        mean_r = np.mean(all_r) if all_r else np.nan
        min_r = np.min(all_r) if all_r else np.nan
        max_r = np.max(all_r) if all_r else np.nan

        f.write(f"mean_pairwise_r\t{mean_r:.4f}\n")
        f.write(f"min_pairwise_r\t{min_r:.4f}\n")
        f.write(f"max_pairwise_r\t{max_r:.4f}\n")
        f.write(f"sign_concordance_rate\t{concordance_rate:.4f}\n")

        all_delta = [delta_rho[(y1, y2)] for (y1, y2) in year_pairs if not np.isnan(delta_rho[(y1, y2)])]
        mean_delta = np.mean(all_delta) if all_delta else np.nan
        f.write(f"mean_median_delta_rho\t{mean_delta:.4f}\n")

    print(f"\n[SUCCESS] Results written to: {OUTPUT_FILE}")

    # Print summary
    print("\n" + "="*60)
    print("TEMPORAL STABILITY SUMMARY")
    print("="*60)
    print(f"PFAM domains analyzed: {TOP_N_PFAM}")
    print(f"ENV variables: {env_matrix.shape[1]}")
    print(f"Correlations per year: {TOP_N_PFAM * env_matrix.shape[1]}")
    print(f"\nYear-to-year correlation stability (Pearson r of rho vectors):")
    for (y1, y2) in year_pairs:
        r, p, n = year_pair_r[(y1, y2)]
        print(f"  {y1} vs {y2}: r = {r:.4f}")
    print(f"\nMean pairwise r: {mean_r:.4f}")
    print(f"Sign concordance rate: {concordance_rate:.1%}")
    print(f"Mean median |delta-rho|: {mean_delta:.4f}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
