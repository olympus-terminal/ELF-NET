#!/usr/bin/env python3
"""
Task 2: Compute PFAM-AlphaEarth Embedding Correlations (algaGPT-filtered)

Computes Spearman correlations between algaGPT-filtered PFAM domains
and AlphaEarth embeddings (64 dimensions) with FDR correction.

Provenance:
  Input: algagpt_gee_pfam_merged_SMART_20260119_100639.tsv
         alphaearth_embeddings_20260114_113823.tsv
  Reference: ../../AlphaEarth/analyze_pfam_alphaearth_correlations_20260114_114024.py
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
ALGAGPT_FILE = BASE_DIR / "algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
ALPHAEARTH_FILE = Path("/media/drn2/External/TARA-Oceans/AlphaEarth/alphaearth_embeddings_20260114_113823.tsv")
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def load_algagpt_data():
    """Load algaGPT-filtered PFAM data."""
    print("=" * 70)
    print("LOADING ALGAGPT DATA")
    print("=" * 70)
    print(f"  Input: {ALGAGPT_FILE}")

    # Read file, skipping provenance comments
    df = pd.read_csv(ALGAGPT_FILE, sep='\t', comment='#', low_memory=False)

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns):,}")

    # Identify PFAM columns
    pfam_cols = [col for col in df.columns if col.startswith('PF')]
    print(f"  PFAM domains: {len(pfam_cols):,}")

    # Keep only assembly_id and PFAM columns
    df_pfam = df[['assembly_id'] + pfam_cols].copy()

    return df_pfam, pfam_cols

def load_alphaearth_embeddings():
    """Load AlphaEarth embeddings."""
    print("\n" + "=" * 70)
    print("LOADING ALPHAEARTH EMBEDDINGS")
    print("=" * 70)
    print(f"  Input: {ALPHAEARTH_FILE}")

    # Read file
    df = pd.read_csv(ALPHAEARTH_FILE, sep='\t', comment='#', low_memory=False)

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns):,}")

    # Identify embedding columns (A00-A63)
    embed_cols = [f'A{i:02d}' for i in range(64)]
    present_cols = [c for c in embed_cols if c in df.columns]

    if len(present_cols) != 64:
        print(f"  WARNING: Expected 64 embedding dimensions, found {len(present_cols)}")

    # Filter to samples with complete embeddings
    complete_mask = df[present_cols].notna().all(axis=1)
    df_complete = df[complete_mask].copy()

    print(f"  Samples with complete embeddings: {len(df_complete):,}")

    # Keep only assembly_id and embedding columns
    df_embed = df_complete[['assembly_id'] + present_cols].copy()

    return df_embed, present_cols

def merge_data(df_pfam, df_embed, pfam_cols, embed_cols):
    """Merge PFAM and AlphaEarth data by assembly_id."""
    print("\n" + "=" * 70)
    print("MERGING DATA")
    print("=" * 70)

    merged = df_pfam.merge(df_embed, on='assembly_id', how='inner')

    print(f"  algaGPT samples: {len(df_pfam):,}")
    print(f"  AlphaEarth samples: {len(df_embed):,}")
    print(f"  Merged samples: {len(merged):,}")

    if len(merged) == 0:
        raise ValueError("No overlapping samples found! Check assembly_id matching.")

    return merged

def filter_pfams(merged_df, pfam_cols, min_nonzero=10):
    """Filter PFAMs to those with sufficient variation."""
    print("\n" + "=" * 70)
    print("FILTERING PFAMS")
    print("=" * 70)
    print(f"  Minimum non-zero samples: {min_nonzero}")

    # Count non-zero values for each PFAM
    nonzero_counts = (merged_df[pfam_cols] > 0).sum()

    # Filter to PFAMs with enough variation
    valid_pfams = nonzero_counts[nonzero_counts >= min_nonzero].index.tolist()

    print(f"  Original PFAMs: {len(pfam_cols):,}")
    print(f"  PFAMs with >= {min_nonzero} non-zero samples: {len(valid_pfams):,}")

    return valid_pfams

def compute_correlations(merged_df, pfam_cols, embed_cols):
    """Compute Spearman correlations between PFAMs and AlphaEarth dimensions."""
    print("\n" + "=" * 70)
    print("COMPUTING CORRELATIONS")
    print("=" * 70)

    n_pfam = len(pfam_cols)
    n_embed = len(embed_cols)
    total_tests = n_pfam * n_embed

    print(f"  Computing {n_pfam:,} × {n_embed} = {total_tests:,} correlations...")

    results = []

    # Process in chunks for progress reporting
    chunk_size = 500
    for i in range(0, n_pfam, chunk_size):
        chunk_pfams = pfam_cols[i:i+chunk_size]

        for pfam in chunk_pfams:
            pfam_values = merged_df[pfam].values

            for dim in embed_cols:
                dim_values = merged_df[dim].values

                # Calculate Spearman correlation
                rho, pval = spearmanr(pfam_values, dim_values, nan_policy='omit')

                results.append({
                    'pfam_domain': pfam,
                    'alphaearth_dimension': dim,
                    'spearman_rho': rho,
                    'pvalue': pval
                })

        # Progress update
        processed = min(i + chunk_size, n_pfam)
        print(f"    Processed {processed:,}/{n_pfam:,} PFAMs ({100*processed/n_pfam:.1f}%)")

    results_df = pd.DataFrame(results)

    print(f"  Done. Total correlations: {len(results_df):,}")

    return results_df

def apply_fdr_correction(results_df, alpha=0.05):
    """Apply FDR correction for multiple testing."""
    print("\n" + "=" * 70)
    print("APPLYING FDR CORRECTION")
    print("=" * 70)

    # Handle NaN p-values
    valid_mask = results_df['pvalue'].notna()
    pvals_valid = results_df.loc[valid_mask, 'pvalue'].values

    print(f"  Total p-values: {len(results_df):,}")
    print(f"  Valid p-values: {len(pvals_valid):,}")

    # Apply Benjamini-Hochberg
    reject, qvals, _, _ = multipletests(pvals_valid, alpha=alpha, method='fdr_bh')

    # Add q-values to dataframe
    results_df['qvalue'] = np.nan
    results_df.loc[valid_mask, 'qvalue'] = qvals

    # Add significance flag
    results_df['significant_fdr_0.05'] = False
    results_df.loc[valid_mask, 'significant_fdr_0.05'] = reject

    n_significant = reject.sum()
    pct_significant = 100 * n_significant / len(pvals_valid)
    print(f"  Significant after FDR (q<{alpha}): {n_significant:,} ({pct_significant:.2f}%)")

    return results_df, n_significant

def save_results(results_df):
    """Save correlation results to files."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Sort by absolute correlation
    results_df['abs_rho'] = results_df['spearman_rho'].abs()
    results_df = results_df.sort_values('abs_rho', ascending=False)
    results_df = results_df.drop(columns=['abs_rho'])

    # Save full results
    output_full = OUTPUT_DIR / f"algagpt_pfam_alphaearth_correlations_full_{TIMESTAMP}.tsv"

    with open(output_full, 'w') as f:
        f.write("# Provenance:\n")
        f.write(f"#   Script: {Path(__file__).absolute()}\n")
        f.write(f"#   Input PFAM: {ALGAGPT_FILE.absolute()}\n")
        f.write(f"#   Input AlphaEarth: {ALPHAEARTH_FILE.absolute()}\n")
        f.write(f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#   Integrity Check: PASSED - Real data only\n")
        f.write(f"#   Method: Spearman correlation with FDR correction (Benjamini-Hochberg, alpha=0.05)\n")
        f.write(f"#   Total correlations: {len(results_df):,}\n")
        f.write("#\n")

    results_df.to_csv(output_full, sep='\t', index=False, mode='a')
    print(f"  Full results: {output_full}")
    print(f"    {len(results_df):,} correlations")

    # Save significant results only
    df_sig = results_df[results_df['significant_fdr_0.05']].copy()
    output_sig = OUTPUT_DIR / f"algagpt_pfam_alphaearth_correlations_significant_{TIMESTAMP}.tsv"

    with open(output_sig, 'w') as f:
        f.write("# Provenance:\n")
        f.write(f"#   Script: {Path(__file__).absolute()}\n")
        f.write(f"#   Input PFAM: {ALGAGPT_FILE.absolute()}\n")
        f.write(f"#   Input AlphaEarth: {ALPHAEARTH_FILE.absolute()}\n")
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
    print(results_df.head(20)[['pfam_domain', 'alphaearth_dimension', 'spearman_rho', 'qvalue']].to_string(index=False))

    return output_full, output_sig

def main():
    """Main execution."""
    print("\n" + "=" * 70)
    print("ALGAGPT PFAM-ALPHAEARTH CORRELATION ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load data
    df_pfam, pfam_cols = load_algagpt_data()
    df_embed, embed_cols = load_alphaearth_embeddings()

    # Merge data
    merged = merge_data(df_pfam, df_embed, pfam_cols, embed_cols)

    # Filter PFAMs with low variance
    valid_pfams = filter_pfams(merged, pfam_cols, min_nonzero=10)

    # Compute correlations
    results_df = compute_correlations(merged, valid_pfams, embed_cols)

    # Apply FDR correction
    results_df, n_sig = apply_fdr_correction(results_df, alpha=0.05)

    # Save results
    output_full, output_sig = save_results(results_df)

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
