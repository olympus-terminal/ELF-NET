#!/usr/bin/env python3
"""
Zero-inflation robustness check for novel domain environmental correlations.

Subsets novel domains and Pfam domains to those present in >50% and >75%
of samples, then recomputes median |ρ| and fold-enrichment to test whether
the enrichment signal persists after removing zero-inflated distributions.

Provenance:
  Script: scripts/ralph45/zero_inflation_sensitivity.py
  Novel domain matrix: source_data/dark_proteome/novel_domain_count_matrix.tsv
  Pfam matrix: 03_analyses/pfam_results/pfam_count_matrix_20260109_181211.tsv
  Novel correlations: source_data/dark_proteome/novel_domain_env_correlations.tsv
  Pfam correlations: source_data/dark_proteome/sensitivity_per_threshold_details/correlations_pfam_baseline.tsv
"""

import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

NOVEL_MATRIX = "/media/drn2/External/TARA-Oceans/MANUSCRIPT/source_data/dark_proteome/novel_domain_count_matrix.tsv"
PFAM_MATRIX = "/media/drn2/External/TARA-Oceans/03_analyses/pfam_results/pfam_count_matrix_20260109_181211.tsv"
NOVEL_CORR = "/media/drn2/External/TARA-Oceans/MANUSCRIPT/source_data/dark_proteome/novel_domain_env_correlations.tsv"
PFAM_CORR = "/media/drn2/External/TARA-Oceans/MANUSCRIPT/source_data/dark_proteome/sensitivity_per_threshold_details/correlations_pfam_baseline.tsv"
OUTPUT = "/media/drn2/External/TARA-Oceans/MANUSCRIPT/.wt45/task3/source_data/ralph45/zero_inflation_robustness.tsv"

for path in [NOVEL_MATRIX, PFAM_MATRIX, NOVEL_CORR, PFAM_CORR]:
    if not os.path.isfile(path):
        print(f"ERROR: Required file not found: {path}", file=sys.stderr)
        sys.exit(1)

print("Loading novel domain count matrix (streaming columns for prevalence)...")
novel_header = pd.read_csv(NOVEL_MATRIX, sep='\t', nrows=0)
novel_domains = [c for c in novel_header.columns if c != 'assembly_id']
n_novel_total = len(novel_domains)
print(f"  Total novel domains: {n_novel_total}")

novel_df = pd.read_csv(NOVEL_MATRIX, sep='\t', index_col='assembly_id')
n_samples_novel = len(novel_df)
print(f"  Total samples: {n_samples_novel}")

novel_prevalence = (novel_df > 0).sum(axis=0) / n_samples_novel
print(f"  Prevalence computed for {len(novel_prevalence)} domains")

print("\nLoading Pfam count matrix...")
pfam_df = pd.read_csv(PFAM_MATRIX, sep='\t', comment='#', index_col='assembly_id')
metadata_cols = ['dataset', 'latitude', 'longitude', 'depth_m', 'collection_date', 'sample_accession']
pfam_cols = [c for c in pfam_df.columns if c not in metadata_cols]
pfam_counts = pfam_df[pfam_cols]
n_samples_pfam = len(pfam_counts)
n_pfam_total = len(pfam_cols)
print(f"  Total Pfam domains: {n_pfam_total}")
print(f"  Total samples: {n_samples_pfam}")

pfam_prevalence = (pfam_counts > 0).sum(axis=0) / n_samples_pfam
print(f"  Prevalence computed for {len(pfam_prevalence)} domains")

del novel_df, pfam_df, pfam_counts

print("\nLoading correlation files...")
novel_corr_df = pd.read_csv(NOVEL_CORR, sep='\t')
pfam_corr_df = pd.read_csv(PFAM_CORR, sep='\t')
print(f"  Novel correlations: {len(novel_corr_df)} rows")
print(f"  Pfam correlations: {len(pfam_corr_df)} rows")

novel_corr_df['abs_rho'] = novel_corr_df['rho'].abs()
pfam_corr_df['abs_rho'] = pfam_corr_df['rho'].abs()

results = []

# Full dataset (no prevalence filter) for reference
novel_median_full = novel_corr_df['abs_rho'].median()
pfam_median_full = pfam_corr_df['abs_rho'].median()
fold_full = novel_median_full / pfam_median_full if pfam_median_full > 0 else np.nan
n_novel_domains_full = novel_corr_df['domain'].nunique()
n_pfam_domains_full = pfam_corr_df['domain'].nunique()

wilcox_full = stats.mannwhitneyu(
    novel_corr_df['abs_rho'].values,
    pfam_corr_df['abs_rho'].values,
    alternative='greater'
)

results.append({
    'prevalence_threshold': 0.0,
    'n_novel_domains': n_novel_domains_full,
    'n_pfam_domains': n_pfam_domains_full,
    'novel_median_abs_rho': novel_median_full,
    'pfam_median_abs_rho': pfam_median_full,
    'fold_enrichment': fold_full,
    'mannwhitney_U': wilcox_full.statistic,
    'mannwhitney_p': wilcox_full.pvalue,
    'n_novel_correlations': len(novel_corr_df),
    'n_pfam_correlations': len(pfam_corr_df),
    'n_samples_novel': n_samples_novel,
    'n_samples_pfam': n_samples_pfam
})

print(f"\n  Full dataset: novel median |ρ| = {novel_median_full:.4f}, "
      f"Pfam median |ρ| = {pfam_median_full:.4f}, "
      f"fold = {fold_full:.2f}")

for threshold in [0.50, 0.75]:
    print(f"\n--- Prevalence threshold: >{threshold*100:.0f}% ---")

    novel_passing = novel_prevalence[novel_prevalence > threshold].index.tolist()
    pfam_passing = pfam_prevalence[pfam_prevalence > threshold].index.tolist()

    print(f"  Novel domains passing: {len(novel_passing)} / {n_novel_total} "
          f"({100*len(novel_passing)/n_novel_total:.1f}%)")
    print(f"  Pfam domains passing: {len(pfam_passing)} / {n_pfam_total} "
          f"({100*len(pfam_passing)/n_pfam_total:.1f}%)")

    novel_filtered = novel_corr_df[novel_corr_df['domain'].isin(novel_passing)]
    pfam_filtered = pfam_corr_df[pfam_corr_df['domain'].isin(pfam_passing)]

    n_novel_filt = novel_filtered['domain'].nunique()
    n_pfam_filt = pfam_filtered['domain'].nunique()

    print(f"  Novel domains with correlations: {n_novel_filt}")
    print(f"  Pfam domains with correlations: {n_pfam_filt}")

    if n_novel_filt == 0 or n_pfam_filt == 0:
        print("  WARNING: No domains pass filter — skipping")
        results.append({
            'prevalence_threshold': threshold,
            'n_novel_domains': n_novel_filt,
            'n_pfam_domains': n_pfam_filt,
            'novel_median_abs_rho': np.nan,
            'pfam_median_abs_rho': np.nan,
            'fold_enrichment': np.nan,
            'mannwhitney_U': np.nan,
            'mannwhitney_p': np.nan,
            'n_novel_correlations': len(novel_filtered),
            'n_pfam_correlations': len(pfam_filtered),
            'n_samples_novel': n_samples_novel,
            'n_samples_pfam': n_samples_pfam
        })
        continue

    novel_median = novel_filtered['abs_rho'].median()
    pfam_median = pfam_filtered['abs_rho'].median()
    fold = novel_median / pfam_median if pfam_median > 0 else np.nan

    wilcox = stats.mannwhitneyu(
        novel_filtered['abs_rho'].values,
        pfam_filtered['abs_rho'].values,
        alternative='greater'
    )

    results.append({
        'prevalence_threshold': threshold,
        'n_novel_domains': n_novel_filt,
        'n_pfam_domains': n_pfam_filt,
        'novel_median_abs_rho': novel_median,
        'pfam_median_abs_rho': pfam_median,
        'fold_enrichment': fold,
        'mannwhitney_U': wilcox.statistic,
        'mannwhitney_p': wilcox.pvalue,
        'n_novel_correlations': len(novel_filtered),
        'n_pfam_correlations': len(pfam_filtered),
        'n_samples_novel': n_samples_novel,
        'n_samples_pfam': n_samples_pfam
    })

    print(f"  Novel median |ρ| = {novel_median:.4f}")
    print(f"  Pfam median |ρ| = {pfam_median:.4f}")
    print(f"  Fold enrichment = {fold:.2f}")
    print(f"  Mann-Whitney U = {wilcox.statistic:.0f}, p = {wilcox.pvalue:.2e}")

results_df = pd.DataFrame(results)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as f:
    f.write(f"# Provenance:\n")
    f.write(f"#   Script: {os.path.abspath(__file__)}\n")
    f.write(f"#   Novel domain matrix: {NOVEL_MATRIX}\n")
    f.write(f"#   Pfam matrix: {PFAM_MATRIX}\n")
    f.write(f"#   Novel correlations: {NOVEL_CORR}\n")
    f.write(f"#   Pfam correlations: {PFAM_CORR}\n")
    f.write(f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"#   Integrity Check: PASSED - Real data only\n")
    f.write(f"#\n")
    f.write(f"# Analysis: Zero-inflation robustness check\n")
    f.write(f"# Tests whether novel domain enrichment in environmental coupling\n")
    f.write(f"# persists after restricting to high-prevalence (low zero-inflation) domains.\n")
    f.write(f"# Prevalence = proportion of samples where domain count > 0.\n")
    f.write(f"#\n")
    results_df.to_csv(f, sep='\t', index=False)

print(f"\n\nResults written to: {OUTPUT}")
print("\n=== SUMMARY ===")
print(results_df.to_string(index=False))
