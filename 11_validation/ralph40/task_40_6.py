#!/usr/bin/env python3
"""
Ralph40 Task 40.6: Prevalence-matched novel vs Pfam coupling control.

Reviewer concern (R1.5): "A control analysis comparing coupling strength of
novel domains with matched prevalence distributions to Pfam domains."

The dark proteome analysis reports novel domains show 2.3x higher median |rho|
coupling to environment compared to Pfam domains (0.162 vs 0.070). Reviewer
questions whether this enrichment is an artifact of prevalence differences.

Action: Match novel domains to Pfam domains by prevalence (bin matching).
Compare median |rho| between matched sets. Test if 2.3x enrichment persists.

Data sources:
  - Novel domain correlations: pre-computed Spearman rho (n=1,523 samples,
    18 env variables, CLR-normalized, FDR-corrected)
  - Novel domain prevalence: from novel_domain_count_matrix.tsv (2,044 samples)
  - Pfam correlations: computed here using the CCA pipeline's pre-processed
    data (1,809 samples, CLR-normalized PFAM matrix, 32 env variables)
  - Pfam prevalence: computed from the same 1,809-sample dataset

Provenance:
  Input:  novel_domain_env_correlations.tsv, novel_domain_count_matrix.tsv,
          env_pfam_manifold/data/*.npy (CCA pipeline pre-processed data)
  Output: source_data/ralph40/prevalence_matched_coupling.tsv
  Date:   2026-04-08
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, mannwhitneyu
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# ============================================================================
# Data integrity enforcement
# ============================================================================
BASE_DIR = Path("/media/drn2/External/TARA-Oceans")
MANUSCRIPT_DIR = Path("/media/drn2/External/TARA-Oceans/MANUSCRIPT/.wt/a3")
CCA_DATA_DIR = BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "env_pfam_manifold" / "data"

sys.path.insert(0, str(BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "env_pfam_manifold"))
from DataIntegrityGuard import enforce_data_integrity, validate_input_source, create_provenance_header
enforce_data_integrity()

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# Input files
# ============================================================================
NOVEL_CORR_FILE = BASE_DIR / "MANUSCRIPT" / "source_data" / "dark_proteome" / "novel_domain_env_correlations.tsv"
NOVEL_COUNT_FILE = BASE_DIR / "MANUSCRIPT" / "source_data" / "dark_proteome" / "novel_domain_count_matrix.tsv"

# CCA pipeline pre-processed data (matched samples with env + Pfam)
PFAM_CLR_FILE = CCA_DATA_DIR / "pfam_matrix_20260122_101559.npy"
PFAM_RAW_FILE = CCA_DATA_DIR / "pfam_raw_20260122_101559.npy"
ENV_MATRIX_FILE = CCA_DATA_DIR / "env_matrix_20260122_101559.npy"
PFAM_COLS_FILE = CCA_DATA_DIR / "pfam_columns_20260122_101559.txt"
ENV_COLS_FILE = CCA_DATA_DIR / "env_columns_20260122_101559.txt"
SAMPLE_IDS_FILE = CCA_DATA_DIR / "sample_ids_20260122_101559.npy"

for f in [NOVEL_CORR_FILE, NOVEL_COUNT_FILE, PFAM_CLR_FILE, PFAM_RAW_FILE,
          ENV_MATRIX_FILE, PFAM_COLS_FILE, ENV_COLS_FILE, SAMPLE_IDS_FILE]:
    validate_input_source(f)
    print(f"  Validated: {f.name}")

# Environmental variables used in novel domain correlations (18 variables)
NOVEL_ENV_VARS = [
    'air_temp_mean_c', 'air_temp_max_c', 'air_temp_min_c', 'air_temp_range_c',
    'sst_mean_c', 'sst_max_c', 'sst_min_c', 'sst_range_c', 'modis_sst_mean_c',
    'chl_mean_mg_m3', 'chl_max_mg_m3', 'chl_min_mg_m3',
    'poc_mean_mg_m3', 'nflh_mean',
    'bathymetry_m', 'distance_to_coast_km',
    'precip_mean_mm', 'solar_rad_mj_m2',
]

# ============================================================================
# Step 1: Load novel domain correlations (pre-computed)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: LOADING NOVEL DOMAIN CORRELATIONS")
print("=" * 70)

novel_corr = pd.read_csv(NOVEL_CORR_FILE, sep='\t')
print(f"  Novel correlations: {len(novel_corr)} rows")
print(f"  Novel domains: {novel_corr['domain'].nunique()}")
print(f"  Env variables: {novel_corr['env_variable'].nunique()}")

# Compute per-domain summary: median |rho| across all env variables
novel_summary = novel_corr.groupby('domain').agg(
    median_abs_rho=('rho', lambda x: np.median(np.abs(x))),
    n_env_tested=('env_variable', 'count'),
    n_significant=('significant', 'sum'),
).reset_index()

print(f"  Novel domain summary: {len(novel_summary)} domains")
print(f"  Overall novel median |rho|: {novel_summary['median_abs_rho'].median():.4f}")

# ============================================================================
# Step 2: Load novel domain count matrix to compute prevalence
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: COMPUTING NOVEL DOMAIN PREVALENCE")
print("=" * 70)

novel_counts = pd.read_csv(NOVEL_COUNT_FILE, sep='\t', index_col=0)
print(f"  Novel count matrix: {novel_counts.shape}")

# Prevalence = number of samples where domain count > 0
novel_prevalence = (novel_counts > 0).sum(axis=0)
novel_prevalence.name = 'prevalence'
n_novel_samples = len(novel_counts)
print(f"  Total samples in novel matrix: {n_novel_samples}")
print(f"  Novel prevalence range: {novel_prevalence.min()} - {novel_prevalence.max()}")
print(f"  Novel prevalence median: {novel_prevalence.median():.0f}")

# Merge prevalence with novel summary
novel_prev_df = novel_prevalence.reset_index()
novel_prev_df.columns = ['domain', 'prevalence_count']
novel_summary = novel_summary.merge(novel_prev_df, on='domain', how='left')
novel_summary['prevalence_frac'] = novel_summary['prevalence_count'] / n_novel_samples

print(f"  Novel domains with prevalence: {novel_summary['prevalence_count'].notna().sum()}")
print(f"  Novel prevalence fraction range: {novel_summary['prevalence_frac'].min():.4f} - {novel_summary['prevalence_frac'].max():.4f}")

# Free memory
del novel_counts

# ============================================================================
# Step 3: Load CCA pipeline Pfam and env data
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: LOADING PFAM DATA (CCA PIPELINE)")
print("=" * 70)

pfam_clr = np.load(PFAM_CLR_FILE)
pfam_raw = np.load(PFAM_RAW_FILE)
env_matrix = np.load(ENV_MATRIX_FILE)
sample_ids = np.load(SAMPLE_IDS_FILE, allow_pickle=True)

with open(PFAM_COLS_FILE) as f:
    pfam_cols = [line.strip() for line in f]

with open(ENV_COLS_FILE) as f:
    env_cols = [line.strip() for line in f]

n_pfam_samples = len(sample_ids)
n_pfam_domains = len(pfam_cols)

print(f"  PFAM CLR matrix: {pfam_clr.shape}")
print(f"  PFAM raw matrix: {pfam_raw.shape}")
print(f"  Env matrix: {env_matrix.shape}")
print(f"  Samples: {n_pfam_samples}")
print(f"  Pfam domains: {n_pfam_domains}")
print(f"  Env variables: {len(env_cols)}")

# Identify the 18 env variable indices that match the novel domain pipeline
env_var_indices = []
env_var_names = []
for var in NOVEL_ENV_VARS:
    if var in env_cols:
        env_var_indices.append(env_cols.index(var))
        env_var_names.append(var)
print(f"  Matching env variables: {len(env_var_names)}/{len(NOVEL_ENV_VARS)}")

# ============================================================================
# Step 4: Compute Pfam prevalence
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: COMPUTING PFAM PREVALENCE")
print("=" * 70)

pfam_prevalence = (pfam_raw > 0).sum(axis=0)
print(f"  Pfam prevalence range: {pfam_prevalence.min()} - {pfam_prevalence.max()}")
print(f"  Pfam prevalence median: {np.median(pfam_prevalence):.0f}")

pfam_prev_frac = pfam_prevalence / n_pfam_samples

# ============================================================================
# Step 5: Compute Pfam-environment Spearman correlations
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: COMPUTING PFAM-ENVIRONMENT SPEARMAN CORRELATIONS")
print("=" * 70)

n_env = len(env_var_names)
print(f"  Computing {n_pfam_domains} x {n_env} = {n_pfam_domains * n_env} correlations...")
print(f"  Using CLR-transformed PFAM data (same as novel domain pipeline)")

pfam_rho_results = []
for j_idx, j in enumerate(env_var_indices):
    env_var = env_var_names[j_idx]
    env_col = env_matrix[:, j]
    valid = ~np.isnan(env_col)
    n_valid = int(valid.sum())

    if n_valid < 20:
        print(f"  Skipping {env_var}: only {n_valid} valid samples")
        continue

    for i in range(n_pfam_domains):
        domain_col = pfam_clr[valid, i]
        rho, pval = spearmanr(domain_col, env_col[valid])
        pfam_rho_results.append({
            'domain': pfam_cols[i],
            'env_variable': env_var,
            'rho': rho,
            'p_value': pval,
            'n_samples': n_valid,
        })

    if (j_idx + 1) % 5 == 0:
        print(f"  Processed {j_idx+1}/{n_env} env variables...")

pfam_corr = pd.DataFrame(pfam_rho_results)
print(f"  Total Pfam correlations: {len(pfam_corr)}")

# FDR correction (Benjamini-Hochberg)
from statsmodels.stats.multitest import multipletests
reject, qvals, _, _ = multipletests(pfam_corr['p_value'].values, method='fdr_bh')
pfam_corr['q_value'] = qvals
pfam_corr['significant'] = reject

print(f"  Pfam significant (q<0.05): {pfam_corr['significant'].sum()}")
print(f"  Pfam median |rho|: {np.median(np.abs(pfam_corr['rho'])):.4f}")

# Per-domain summary
pfam_summary = pfam_corr.groupby('domain').agg(
    median_abs_rho=('rho', lambda x: np.median(np.abs(x))),
    n_env_tested=('env_variable', 'count'),
    n_significant=('significant', 'sum'),
).reset_index()

# Build prevalence lookup and merge properly
pfam_prev_df = pd.DataFrame({
    'domain': pfam_cols,
    'prevalence_count': pfam_prevalence,
    'prevalence_frac': pfam_prev_frac,
})
pfam_summary = pfam_summary.merge(pfam_prev_df, on='domain', how='left')

print(f"  Pfam domain summary: {len(pfam_summary)} domains")
print(f"  Overall Pfam median |rho|: {pfam_summary['median_abs_rho'].median():.4f}")

# ============================================================================
# Step 6: Prevalence-matched comparison
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: PREVALENCE-MATCHED COMPARISON")
print("=" * 70)

# First, report unmatched comparison
unmatched_novel_median = novel_summary['median_abs_rho'].median()
unmatched_pfam_median = pfam_summary['median_abs_rho'].median()
unmatched_ratio = unmatched_novel_median / unmatched_pfam_median
print(f"\n  UNMATCHED comparison:")
print(f"    Novel median |rho|: {unmatched_novel_median:.4f}")
print(f"    Pfam median |rho|:  {unmatched_pfam_median:.4f}")
print(f"    Ratio: {unmatched_ratio:.2f}x")

# Use prevalence fraction for matching (accounts for different sample sizes)
# Bin-matching approach: create prevalence bins using unique edges
n_bins = 20
combined_prev = np.concatenate([
    novel_summary['prevalence_frac'].dropna().values,
    pfam_summary['prevalence_frac'].dropna().values,
])
bin_edges = np.percentile(combined_prev, np.linspace(0, 100, n_bins + 1))

# Deduplicate bin edges
bin_edges = np.unique(bin_edges)
bin_edges[0] = 0
bin_edges[-1] = max(bin_edges[-1], 1.01)
n_bins_actual = len(bin_edges) - 1
print(f"\n  Using {n_bins_actual} prevalence bins (after dedup)")

novel_summary['prev_bin'] = pd.cut(novel_summary['prevalence_frac'], bins=bin_edges,
                                    labels=False, include_lowest=True)
pfam_summary['prev_bin'] = pd.cut(pfam_summary['prevalence_frac'], bins=bin_edges,
                                   labels=False, include_lowest=True)

# For each bin, sample min(n_novel, n_pfam) from each set
matched_novel_rhos = []
matched_pfam_rhos = []
bin_stats = []

np.random.seed(42)

for b in range(n_bins_actual):
    novel_in_bin = novel_summary[novel_summary['prev_bin'] == b]
    pfam_in_bin = pfam_summary[pfam_summary['prev_bin'] == b]

    n_novel_b = len(novel_in_bin)
    n_pfam_b = len(pfam_in_bin)

    if n_novel_b == 0 or n_pfam_b == 0:
        continue

    n_match = min(n_novel_b, n_pfam_b)

    # Sample without replacement
    novel_sample = novel_in_bin.sample(n=n_match, random_state=42)
    pfam_sample = pfam_in_bin.sample(n=n_match, random_state=42)

    matched_novel_rhos.extend(novel_sample['median_abs_rho'].values)
    matched_pfam_rhos.extend(pfam_sample['median_abs_rho'].values)

    bin_stats.append({
        'bin': b,
        'prev_lo': round(bin_edges[b], 6),
        'prev_hi': round(bin_edges[b + 1], 6),
        'n_novel': n_novel_b,
        'n_pfam': n_pfam_b,
        'n_matched': n_match,
        'novel_median_abs_rho': round(np.median(novel_sample['median_abs_rho'].values), 6),
        'pfam_median_abs_rho': round(np.median(pfam_sample['median_abs_rho'].values), 6),
    })

matched_novel_rhos = np.array(matched_novel_rhos)
matched_pfam_rhos = np.array(matched_pfam_rhos)

total_matched = len(matched_novel_rhos)
matched_novel_median = np.median(matched_novel_rhos)
matched_pfam_median = np.median(matched_pfam_rhos)
matched_ratio = matched_novel_median / matched_pfam_median if matched_pfam_median > 0 else np.inf

print(f"\n  PREVALENCE-MATCHED comparison ({total_matched} domains each):")
print(f"    Novel median |rho|: {matched_novel_median:.4f}")
print(f"    Pfam median |rho|:  {matched_pfam_median:.4f}")
print(f"    Ratio: {matched_ratio:.2f}x")

# Statistical test: Mann-Whitney U
u_stat, u_pval = mannwhitneyu(matched_novel_rhos, matched_pfam_rhos, alternative='greater')
print(f"    Mann-Whitney U: {u_stat:.0f}, p = {u_pval:.2e}")

# Print bin-level details
print(f"\n  Bin-level details:")
print(f"  {'Bin':<4} {'Prev_lo':<10} {'Prev_hi':<10} {'N_novel':<10} {'N_pfam':<10} {'N_match':<10} {'Novel_|rho|':<12} {'Pfam_|rho|':<12}")
print(f"  {'-'*78}")
for bs in bin_stats:
    print(f"  {bs['bin']:<4} {bs['prev_lo']:<10.4f} {bs['prev_hi']:<10.4f} {bs['n_novel']:<10} {bs['n_pfam']:<10} {bs['n_matched']:<10} {bs['novel_median_abs_rho']:<12.4f} {bs['pfam_median_abs_rho']:<12.4f}")

print(f"\n  Total matched domains: {total_matched} per group")

# ============================================================================
# Step 7: Save results
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: SAVING RESULTS")
print("=" * 70)

output_path = MANUSCRIPT_DIR / "source_data" / "ralph40" / "prevalence_matched_coupling.tsv"
output_path.parent.mkdir(parents=True, exist_ok=True)

# Create main results summary
summary_rows = [
    {'metric': 'n_novel_domains', 'value': str(len(novel_summary))},
    {'metric': 'n_pfam_domains', 'value': str(len(pfam_summary))},
    {'metric': 'n_novel_samples', 'value': str(n_novel_samples)},
    {'metric': 'n_pfam_samples', 'value': str(n_pfam_samples)},
    {'metric': 'n_env_variables', 'value': str(n_env)},
    {'metric': 'n_prevalence_bins', 'value': str(n_bins_actual)},
    {'metric': 'n_matched_per_group', 'value': str(total_matched)},
    {'metric': 'unmatched_novel_median_abs_rho', 'value': f"{unmatched_novel_median:.6f}"},
    {'metric': 'unmatched_pfam_median_abs_rho', 'value': f"{unmatched_pfam_median:.6f}"},
    {'metric': 'unmatched_ratio', 'value': f"{unmatched_ratio:.4f}"},
    {'metric': 'matched_novel_median_abs_rho', 'value': f"{matched_novel_median:.6f}"},
    {'metric': 'matched_pfam_median_abs_rho', 'value': f"{matched_pfam_median:.6f}"},
    {'metric': 'matched_ratio', 'value': f"{matched_ratio:.4f}"},
    {'metric': 'mannwhitney_U', 'value': f"{u_stat:.0f}"},
    {'metric': 'mannwhitney_p', 'value': f"{u_pval:.2e}"},
]

summary_df = pd.DataFrame(summary_rows)

provenance = create_provenance_header(
    script_path=__file__,
    input_path=str(NOVEL_CORR_FILE),
    output_description="Prevalence-matched novel vs Pfam domain-environment coupling comparison"
)

with open(output_path, 'w') as f:
    for line in provenance.strip().split('\n'):
        f.write(line + '\n')
    f.write('#\n')
    f.write('# SUMMARY\n')
    summary_df.to_csv(f, sep='\t', index=False)
    f.write('#\n')
    f.write('# BIN-LEVEL DETAILS\n')
    pd.DataFrame(bin_stats).to_csv(f, sep='\t', index=False)

print(f"  Saved: {output_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY FOR MANUSCRIPT")
print("=" * 70)
print(f"\n  Unmatched: novel/Pfam median |rho| = {unmatched_novel_median:.4f} / {unmatched_pfam_median:.4f} = {unmatched_ratio:.1f}x")
print(f"  Prevalence-matched ({total_matched} domains/group, {n_bins_actual} bins):")
print(f"    novel/Pfam = {matched_novel_median:.4f} / {matched_pfam_median:.4f} = {matched_ratio:.1f}x")
print(f"  Mann-Whitney p = {u_pval:.2e}")

if matched_ratio > 1.5:
    print(f"\n  CONCLUSION: The enrichment persists after prevalence matching (ratio = {matched_ratio:.1f}x).")
    print(f"  The coupling enrichment is NOT an artifact of prevalence differences alone.")
else:
    print(f"\n  CONCLUSION: The enrichment is substantially reduced after prevalence matching (ratio = {matched_ratio:.1f}x).")
    print(f"  Prevalence differences may partly explain the coupling enrichment.")

print(f"\n{'='*70}")
print(f"TASK 40.6 COMPLETE — {datetime.now().isoformat()}")
print(f"{'='*70}")
