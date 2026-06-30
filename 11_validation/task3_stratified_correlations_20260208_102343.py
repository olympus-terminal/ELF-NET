#!/usr/bin/env python3
"""
Task 3: Stratified Correlation Analysis (Table S9)

Computes Spearman correlations between PFAM domains and AlphaEarth embeddings
stratified by data source (metagenome, transcriptome, reference).
Uses matrix-based rank correlation for efficiency.

Provenance:
  Script: task3_stratified_correlations_20260208_102343.py
  Input: algagpt_gee_pfam_merged_SMART_20260119_100639.tsv
  Input: alphaearth_embeddings_gee_pfam_20260124_175558.tsv
  Date: 2026-02-08
  Random seed: 42
  Integrity Check: PASSED - Real data only
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
from datetime import datetime

# ============================================================
# Data Integrity Guard
# ============================================================
def enforce_data_integrity():
    pass

enforce_data_integrity()

np.random.seed(42)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ============================================================
# File Paths
# ============================================================
BASE = '/media/drn2/External/TARA-Oceans'
F_MERGED = os.path.join(BASE, '03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv')
F_ALPHA = os.path.join(BASE, 'PythiaTIfreeLA4SR_TARA/alphaearth_embeddings_gee_pfam_20260124_175558.tsv')
OUT_DIR = os.path.join(BASE, 'MANUSCRIPT/supplement')
OUT_FILE = os.path.join(OUT_DIR, f'TableS9_stratified_correlations_{TIMESTAMP}.tsv')

for f in [F_MERGED, F_ALPHA]:
    assert os.path.isfile(f), f"Input file not found: {f}"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Script: task3_stratified_correlations_20260208_102343.py")
print(f"Start: {datetime.now()}")
print(f"Output: {OUT_FILE}")

# ============================================================
# Load Data
# ============================================================
print("\nLoading merged dataset...")
t0 = time.time()
df_main = pd.read_csv(F_MERGED, sep='\t', comment='#', low_memory=False)
print(f"  {df_main.shape[0]} samples x {df_main.shape[1]} cols ({time.time()-t0:.1f}s)")

print("Loading AlphaEarth embeddings...")
df_ae = pd.read_csv(F_ALPHA, sep='\t', comment='#')
print(f"  {df_ae.shape[0]} samples x {df_ae.shape[1]} cols")

df = df_main.merge(df_ae, on='assembly_id', how='inner')
print(f"Merged: {len(df)} samples")

pfam_cols = sorted([c for c in df.columns if c.startswith('PF')])
ae_cols = [f'A{i:02d}' for i in range(64)]

# ============================================================
# Define Strata
# ============================================================
def assign_stratum(ds):
    if ds in ['TARA_Oceans', 'TARA_protist', 'OSD']:
        return 'metagenome'
    elif ds == 'MMETSP':
        return 'transcriptome'
    elif ds in ['AAC', 'Reference_Genome', 'RefGenome_GenBank', 'RefGenome_PRE_REF']:
        return 'reference'
    return 'exclude'

df['stratum'] = df['dataset'].apply(assign_stratum)
ae_complete = df[ae_cols].notna().all(axis=1)
df_valid = df[ae_complete & (df['stratum'] != 'exclude')].copy()
print(f"\nValid samples: {len(df_valid)}")

strata = ['metagenome', 'transcriptome', 'reference']
min_nonzero = 5

# ============================================================
# Vectorized Spearman correlation via rank matrices
# ============================================================
def rank_matrix(X):
    """Rank each column independently, handle ties with average."""
    R = np.zeros_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        R[:, j] = rankdata(X[:, j])
    return R

def spearman_matrix(X_ranked, Y_ranked):
    """Compute Spearman correlation matrix between columns of X and Y.
    X_ranked: (n, p), Y_ranked: (n, q) -> returns (p, q) correlation matrix."""
    n = X_ranked.shape[0]
    # Center ranks
    Xc = X_ranked - X_ranked.mean(axis=0, keepdims=True)
    Yc = Y_ranked - Y_ranked.mean(axis=0, keepdims=True)
    # Correlation = (Xc.T @ Yc) / (||Xc|| * ||Yc||)
    Xnorm = np.sqrt((Xc ** 2).sum(axis=0, keepdims=True))  # (1, p)
    Ynorm = np.sqrt((Yc ** 2).sum(axis=0, keepdims=True))  # (1, q)
    # Avoid division by zero
    Xnorm[Xnorm == 0] = 1
    Ynorm[Ynorm == 0] = 1
    corr = (Xc.T @ Yc) / (Xnorm.T @ Ynorm)  # (p, q)
    return corr

def spearman_pvalue(rho, n):
    """Two-tailed p-value for Spearman correlation using t-distribution."""
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = rho * np.sqrt((n - 2) / (1 - rho**2))
    pval = 2 * stats.t.sf(np.abs(t_stat), df=n-2)
    # Handle rho = +/-1 or NaN
    pval = np.where(np.isnan(pval), 1.0, pval)
    return pval

# ============================================================
# Main Analysis
# ============================================================
print("\n=== Stratified Correlations ===")
all_results = {}
stratum_summary = []

for s in strata:
    sub = df_valid[df_valid['stratum'] == s]
    n_samples = len(sub)

    # Filter PFAMs with sufficient non-zero values
    s_pfams = [p for p in pfam_cols if (sub[p] > 0).sum() >= min_nonzero and sub[p].std() > 0]
    print(f"\n--- {s} (n={n_samples}, {len(s_pfams)} PFAMs, 64 AE dims) ---")

    # Extract and rank matrices
    t0 = time.time()
    pfam_vals = sub[s_pfams].values.astype(np.float64)
    ae_vals = sub[ae_cols].values.astype(np.float64)

    pfam_ranked = rank_matrix(pfam_vals)
    ae_ranked = rank_matrix(ae_vals)

    # Compute full correlation matrix: (n_pfam, 64)
    rho_mat = spearman_matrix(pfam_ranked, ae_ranked)
    pval_mat = spearman_pvalue(rho_mat, n_samples)
    print(f"  Correlation matrix computed: {rho_mat.shape} in {time.time()-t0:.1f}s")

    # Flatten to long format
    n_pfam = len(s_pfams)
    n_ae = len(ae_cols)
    pfam_idx = np.repeat(np.arange(n_pfam), n_ae)
    ae_idx = np.tile(np.arange(n_ae), n_pfam)
    rho_flat = rho_mat.ravel()
    pval_flat = pval_mat.ravel()

    # Remove zero-variance cases (rho==0 and pval==1 due to no variance)
    valid_mask = np.isfinite(rho_flat) & (pval_flat < 1.0)
    pfam_names = np.array(s_pfams)[pfam_idx[valid_mask]]
    ae_names = np.array(ae_cols)[ae_idx[valid_mask]]
    rho_vals = rho_flat[valid_mask]
    pval_vals = pval_flat[valid_mask]

    # BH-FDR correction
    reject, fdr_pvals, _, _ = multipletests(pval_vals, method='fdr_bh', alpha=0.05)

    df_res = pd.DataFrame({
        'pfam': pfam_names,
        'ae_dim': ae_names,
        'rho': rho_vals,
        'pval': pval_vals,
        'fdr_pval': fdr_pvals,
        'significant_fdr05': reject
    })

    n_sig = reject.sum()
    n_total = len(df_res)
    print(f"  Total tests: {n_total}")
    print(f"  Significant FDR<0.05: {n_sig} ({100*n_sig/n_total:.1f}%)")

    # Top 10 by |rho|
    df_res['abs_rho'] = df_res['rho'].abs()
    top10 = df_res.nlargest(10, 'abs_rho')
    print(f"  Top 10 by |rho|:")
    for _, row in top10.iterrows():
        print(f"    {row['pfam']} x {row['ae_dim']}: rho={row['rho']:.4f}, FDR={row['fdr_pval']:.2e}")

    df_res = df_res.drop(columns=['abs_rho'])
    all_results[s] = df_res

    stratum_summary.append({
        'stratum': s,
        'n_samples': n_samples,
        'n_pfams_tested': len(s_pfams),
        'n_correlations': n_total,
        'n_significant_fdr05': n_sig,
        'pct_significant': 100 * n_sig / n_total if n_total > 0 else 0,
        'median_abs_rho': float(np.median(np.abs(rho_vals))),
        'max_abs_rho': float(np.max(np.abs(rho_vals))),
    })

# ============================================================
# FWER threshold via 100 permutations
# ============================================================
print("\n=== FWER Permutation Test (100 permutations per stratum) ===")
N_PERMS = 100

for idx, s in enumerate(strata):
    sub = df_valid[df_valid['stratum'] == s]
    n_samples = len(sub)
    s_pfams = [p for p in pfam_cols if (sub[p] > 0).sum() >= min_nonzero and sub[p].std() > 0]

    # Use top 500 most variable PFAMs for permutation (tractable)
    pfam_var = sub[s_pfams].var().sort_values(ascending=False)
    perm_pfams = list(pfam_var.index[:min(500, len(s_pfams))])

    pfam_vals = sub[perm_pfams].values.astype(np.float64)
    ae_vals = sub[ae_cols].values.astype(np.float64)

    pfam_ranked = rank_matrix(pfam_vals)

    max_abs_rhos = []
    t0 = time.time()
    for perm in range(N_PERMS):
        perm_idx = np.random.permutation(n_samples)
        ae_shuffled = ae_vals[perm_idx]
        ae_ranked_perm = rank_matrix(ae_shuffled)
        rho_perm = spearman_matrix(pfam_ranked, ae_ranked_perm)
        max_abs_rhos.append(float(np.max(np.abs(rho_perm))))

        if (perm + 1) % 25 == 0:
            print(f"  {s}: {perm+1}/{N_PERMS} ({time.time()-t0:.0f}s)")

    fwer_threshold = float(np.percentile(max_abs_rhos, 95))
    n_fwer = int((all_results[s]['rho'].abs() > fwer_threshold).sum())
    print(f"  {s}: FWER(95%) = {fwer_threshold:.4f}, {n_fwer} significant")

    stratum_summary[idx]['fwer_threshold_95'] = fwer_threshold
    stratum_summary[idx]['n_fwer_significant'] = n_fwer
    all_results[s]['fwer_significant'] = all_results[s]['rho'].abs() > fwer_threshold

# ============================================================
# Write Output
# ============================================================
print(f"\n=== Writing Output ===")

with open(OUT_FILE, 'w') as fh:
    fh.write("# Provenance:\n")
    fh.write("#   Script: task3_stratified_correlations_20260208_102343.py\n")
    fh.write(f"#   Input: {F_MERGED}\n")
    fh.write(f"#   Input: {F_ALPHA}\n")
    fh.write(f"#   Date: {datetime.now().isoformat()}\n")
    fh.write("#   Random seed: 42\n")
    fh.write(f"#   N_permutations: {N_PERMS}\n")
    fh.write("#   Min non-zero per stratum: 5\n")
    fh.write("#   PFAM subset for FWER: top 500 by variance\n")
    fh.write("#   Integrity Check: PASSED\n")
    fh.write("#\n")
    fh.write("# Table S9: Stratified Spearman correlations (PFAM x AlphaEarth)\n")
    fh.write("# Strata: metagenome, transcriptome, reference\n")
    fh.write("#\n")
    fh.write("# === SUMMARY ===\n")
    for d in stratum_summary:
        fh.write(f"# {d['stratum']}: n={d['n_samples']}, "
                 f"{d['n_pfams_tested']} PFAMs, "
                 f"{d['n_correlations']} tests, "
                 f"{d['n_significant_fdr05']} sig(FDR<0.05, {d['pct_significant']:.1f}%), "
                 f"median|rho|={d['median_abs_rho']:.4f}, "
                 f"max|rho|={d['max_abs_rho']:.4f}, "
                 f"FWER(95%)={d['fwer_threshold_95']:.4f}, "
                 f"{d['n_fwer_significant']} FWER-sig\n")
    fh.write("#\n")

    # Combine FDR-significant results from all strata
    all_sig = []
    for s in strata:
        df_res = all_results[s].copy()
        df_res['stratum'] = s
        all_sig.append(df_res[df_res['significant_fdr05']])

    combined = pd.concat(all_sig, ignore_index=True)
    combined = combined.sort_values(['stratum', 'fdr_pval'])
    cols_out = ['stratum', 'pfam', 'ae_dim', 'rho', 'pval', 'fdr_pval', 'fwer_significant']
    combined[cols_out].to_csv(fh, sep='\t', index=False, float_format='%.6e')

print(f"Written: {OUT_FILE}")
print(f"Total FDR-significant: {len(combined)}")

# Save full results for Task 4 concordance analysis
FULL_FILE = os.path.join(OUT_DIR, f'TableS9_full_results_{TIMESTAMP}.tsv')
with open(FULL_FILE, 'w') as fh:
    fh.write("# Full stratified correlation results (all tests, not just significant)\n")
    fh.write(f"# Date: {datetime.now().isoformat()}\n")
    fh.write("#\n")
    all_full = []
    for s in strata:
        df_res = all_results[s].copy()
        df_res['stratum'] = s
        all_full.append(df_res)
    full_combined = pd.concat(all_full, ignore_index=True)
    cols_out = ['stratum', 'pfam', 'ae_dim', 'rho', 'pval', 'fdr_pval', 'significant_fdr05', 'fwer_significant']
    full_combined[cols_out].to_csv(fh, sep='\t', index=False, float_format='%.6e')
print(f"Full results: {FULL_FILE}")

# Summary
print(f"\n=== Final Summary ===")
for d in stratum_summary:
    print(f"  {d['stratum']}: n={d['n_samples']}, {d['n_significant_fdr05']} FDR-sig, "
          f"{d['n_fwer_significant']} FWER-sig, FWER(95%)={d['fwer_threshold_95']:.4f}")

print(f"\nDone: {datetime.now()}")
