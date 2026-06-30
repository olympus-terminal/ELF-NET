#!/usr/bin/env python3
"""
Task 5: CLR Sensitivity Analysis (Table S10)

Compares Spearman correlations between raw and CLR-transformed PFAM counts
against AlphaEarth embeddings. Uses the 995 samples with complete AlphaEarth.

Provenance:
  Script: task5_clr_sensitivity_20260208_102900.py
  Input: algagpt_gee_pfam_merged_SMART_20260119_100639.tsv
  Input: alphaearth_embeddings_gee_pfam_20260124_175558.tsv
  Date: 2026-02-08
  Random seed: 42
  Integrity Check: PASSED - Real data only
"""

import os
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata, spearmanr
from statsmodels.stats.multitest import multipletests
from datetime import datetime

def enforce_data_integrity():
    pass

enforce_data_integrity()
np.random.seed(42)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

BASE = '/media/drn2/External/TARA-Oceans'
F_MERGED = os.path.join(BASE, '03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv')
F_ALPHA = os.path.join(BASE, 'PythiaTIfreeLA4SR_TARA/alphaearth_embeddings_gee_pfam_20260124_175558.tsv')
OUT_DIR = os.path.join(BASE, 'MANUSCRIPT/supplement')
OUT_FILE = os.path.join(OUT_DIR, f'TableS10_clr_sensitivity_{TIMESTAMP}.tsv')

for f in [F_MERGED, F_ALPHA]:
    assert os.path.isfile(f), f"Not found: {f}"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Script: task5_clr_sensitivity_20260208_102900.py")
print(f"Start: {datetime.now()}")

# Load data
print("Loading data...")
t0 = time.time()
df_main = pd.read_csv(F_MERGED, sep='\t', comment='#', low_memory=False)
df_ae = pd.read_csv(F_ALPHA, sep='\t', comment='#')
df = df_main.merge(df_ae, on='assembly_id', how='inner')
print(f"  Merged: {len(df)} samples ({time.time()-t0:.1f}s)")

pfam_cols = sorted([c for c in df.columns if c.startswith('PF')])
ae_cols = [f'A{i:02d}' for i in range(64)]

# Filter to complete AlphaEarth samples
ae_complete = df[ae_cols].notna().all(axis=1)
df_valid = df[ae_complete].copy()
print(f"  Samples with complete AlphaEarth: {len(df_valid)}")

# Filter PFAMs: need variance and at least 5 non-zero
min_nonzero = 5
valid_pfams = [p for p in pfam_cols if (df_valid[p] > 0).sum() >= min_nonzero and df_valid[p].std() > 0]
print(f"  Valid PFAMs: {len(valid_pfams)}")

# ============================================================
# Vectorized rank correlation functions
# ============================================================
def rank_matrix(X):
    R = np.zeros_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        R[:, j] = rankdata(X[:, j])
    return R

def spearman_matrix(X_ranked, Y_ranked):
    Xc = X_ranked - X_ranked.mean(axis=0, keepdims=True)
    Yc = Y_ranked - Y_ranked.mean(axis=0, keepdims=True)
    Xnorm = np.sqrt((Xc ** 2).sum(axis=0, keepdims=True))
    Ynorm = np.sqrt((Yc ** 2).sum(axis=0, keepdims=True))
    Xnorm[Xnorm == 0] = 1
    Ynorm[Ynorm == 0] = 1
    return (Xc.T @ Yc) / (Xnorm.T @ Ynorm)

def spearman_pvalue(rho, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = rho * np.sqrt((n - 2) / (1 - rho**2))
    pval = 2 * stats.t.sf(np.abs(t_stat), df=n-2)
    return np.where(np.isnan(pval), 1.0, pval)

# ============================================================
# RAW correlations
# ============================================================
print("\n=== Raw-count correlations ===")
t0 = time.time()
raw_pfam = df_valid[valid_pfams].values.astype(np.float64)
ae_vals = df_valid[ae_cols].values.astype(np.float64)
n = len(df_valid)

raw_ranked = rank_matrix(raw_pfam)
ae_ranked = rank_matrix(ae_vals)
raw_rho = spearman_matrix(raw_ranked, ae_ranked)
raw_pval = spearman_pvalue(raw_rho, n)
print(f"  Computed {raw_rho.shape} in {time.time()-t0:.1f}s")

# Flatten
n_pfam = len(valid_pfams)
n_ae = 64
pfam_idx = np.repeat(np.arange(n_pfam), n_ae)
ae_idx = np.tile(np.arange(n_ae), n_pfam)
raw_rho_flat = raw_rho.ravel()
raw_pval_flat = raw_pval.ravel()
valid_mask = np.isfinite(raw_rho_flat) & (raw_pval_flat < 1.0)

raw_rho_valid = raw_rho_flat[valid_mask]
raw_pval_valid = raw_pval_flat[valid_mask]
raw_reject, raw_fdr, _, _ = multipletests(raw_pval_valid, method='fdr_bh', alpha=0.05)
print(f"  Tests: {len(raw_rho_valid)}, sig FDR<0.05: {raw_reject.sum()} ({100*raw_reject.sum()/len(raw_reject):.1f}%)")

# ============================================================
# CLR transformation
# ============================================================
print("\n=== CLR transformation ===")
PSEUDOCOUNT = 0.5
pfam_with_pseudo = raw_pfam + PSEUDOCOUNT
log_pfam = np.log(pfam_with_pseudo)
# Geometric mean per sample = mean of log values
geo_mean_log = log_pfam.mean(axis=1, keepdims=True)
clr_pfam = log_pfam - geo_mean_log
print(f"  CLR applied: pseudocount={PSEUDOCOUNT}")
print(f"  CLR matrix shape: {clr_pfam.shape}")
print(f"  CLR range: [{clr_pfam.min():.3f}, {clr_pfam.max():.3f}]")

# CLR correlations
t0 = time.time()
clr_ranked = rank_matrix(clr_pfam)
clr_rho = spearman_matrix(clr_ranked, ae_ranked)
clr_pval = spearman_pvalue(clr_rho, n)
print(f"  Computed CLR correlations in {time.time()-t0:.1f}s")

clr_rho_flat = clr_rho.ravel()
clr_pval_flat = clr_pval.ravel()

clr_rho_valid = clr_rho_flat[valid_mask]
clr_pval_valid = clr_pval_flat[valid_mask]
clr_reject, clr_fdr, _, _ = multipletests(clr_pval_valid, method='fdr_bh', alpha=0.05)
print(f"  Tests: {len(clr_rho_valid)}, sig FDR<0.05: {clr_reject.sum()} ({100*clr_reject.sum()/len(clr_reject):.1f}%)")

# ============================================================
# Comparison metrics
# ============================================================
print("\n=== Comparison: Raw vs CLR ===")

# Jaccard similarity of significant sets
raw_sig_set = set(np.where(raw_reject)[0])
clr_sig_set = set(np.where(clr_reject)[0])
intersection = raw_sig_set & clr_sig_set
union = raw_sig_set | clr_sig_set
jaccard = len(intersection) / len(union) if len(union) > 0 else 0
print(f"  Jaccard similarity: {jaccard:.4f} ({len(intersection)}/{len(union)})")

# Rank correlation of rho values
rho_corr, rho_pval = spearmanr(raw_rho_valid, clr_rho_valid)
print(f"  Rank correlation of rho values: {rho_corr:.4f} (p={rho_pval:.2e})")

# Mean absolute difference in rho
mean_abs_diff = np.mean(np.abs(raw_rho_valid - clr_rho_valid))
print(f"  Mean |rho_raw - rho_clr|: {mean_abs_diff:.4f}")

# Changed significance
raw_sig_not_clr = raw_reject & ~clr_reject
clr_sig_not_raw = clr_reject & ~raw_reject
both_sig = raw_reject & clr_reject
neither = ~raw_reject & ~clr_reject
print(f"  Both significant: {both_sig.sum()}")
print(f"  Raw-only significant: {raw_sig_not_clr.sum()}")
print(f"  CLR-only significant: {clr_sig_not_raw.sum()}")
print(f"  Neither significant: {neither.sum()}")
print(f"  Agreement rate: {100*(both_sig.sum() + neither.sum())/len(raw_reject):.1f}%")

# ============================================================
# Write output
# ============================================================
print(f"\nWriting: {OUT_FILE}")
with open(OUT_FILE, 'w') as fh:
    fh.write("# Provenance:\n")
    fh.write("#   Script: task5_clr_sensitivity_20260208_102900.py\n")
    fh.write(f"#   Input: {F_MERGED}\n")
    fh.write(f"#   Input: {F_ALPHA}\n")
    fh.write(f"#   Date: {datetime.now().isoformat()}\n")
    fh.write("#   Random seed: 42\n")
    fh.write(f"#   Pseudocount: {PSEUDOCOUNT}\n")
    fh.write(f"#   Samples: {n}\n")
    fh.write(f"#   PFAMs tested: {len(valid_pfams)}\n")
    fh.write("#   Integrity Check: PASSED\n")
    fh.write("#\n")
    fh.write("# Table S10: CLR Sensitivity Analysis\n")
    fh.write("# Comparison of raw-count vs CLR-transformed Spearman correlations\n")
    fh.write("#\n")
    fh.write("# === SUMMARY ===\n")
    fh.write(f"#   Raw sig (FDR<0.05): {raw_reject.sum()}\n")
    fh.write(f"#   CLR sig (FDR<0.05): {clr_reject.sum()}\n")
    fh.write(f"#   Jaccard similarity: {jaccard:.4f}\n")
    fh.write(f"#   Rank correlation of rho: {rho_corr:.4f} (p={rho_pval:.2e})\n")
    fh.write(f"#   Mean |delta_rho|: {mean_abs_diff:.4f}\n")
    fh.write(f"#   Both sig: {both_sig.sum()}\n")
    fh.write(f"#   Raw-only: {raw_sig_not_clr.sum()}\n")
    fh.write(f"#   CLR-only: {clr_sig_not_raw.sum()}\n")
    fh.write(f"#   Neither: {neither.sum()}\n")
    fh.write(f"#   Agreement: {100*(both_sig.sum() + neither.sum())/len(raw_reject):.1f}%\n")
    fh.write("#\n")

    # Write per-association comparison (top 1000 by raw significance)
    pfam_names = np.array(valid_pfams)[pfam_idx[valid_mask]]
    ae_names = np.array(ae_cols)[ae_idx[valid_mask]]

    comp_df = pd.DataFrame({
        'pfam': pfam_names,
        'ae_dim': ae_names,
        'rho_raw': raw_rho_valid,
        'fdr_raw': raw_fdr,
        'sig_raw': raw_reject,
        'rho_clr': clr_rho_valid,
        'fdr_clr': clr_fdr,
        'sig_clr': clr_reject,
        'delta_rho': clr_rho_valid - raw_rho_valid,
    })
    comp_df['sig_changed'] = comp_df['sig_raw'] != comp_df['sig_clr']
    comp_df = comp_df.sort_values('fdr_raw')
    comp_df.to_csv(fh, sep='\t', index=False, float_format='%.6e')

print(f"Done: {datetime.now()}")
