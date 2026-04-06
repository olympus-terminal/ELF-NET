#!/usr/bin/env python3
"""
14_bootstrap_cca_loadings.py — Bootstrap stability of CCA loadings.

1,000 bootstrap resamples of the full CCA pipeline:
  - Draw n samples with replacement from X_domain and X_env
  - Apply PCA(100) to the domain matrix
  - Fit CCA(10) between PCA-reduced domains and environment
  - Back-project CCA weights through PCA to get PFAM-level loadings
  - Record top-20 CC1 domain ranks, loading magnitudes, and sign

Output TSV with columns:
  pfam_id, mean_loading, ci_lower, ci_upper, sign_consistency,
  top20_frequency, median_rank

Provenance: ralph32 task 10 — bootstrap CCA loading stability analysis.
"""

import json
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path resolution — HPC vs local
# ---------------------------------------------------------------------------

def get_paths():
    """Detect HPC vs local environment, return (data_dir, output_dir)."""
    hpc_data = Path('/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/kan_cca')
    local_data = Path(__file__).resolve().parent.parent.parent / 'kan_cca_results'

    if hpc_data.exists():
        data_dir = hpc_data
        output_dir = Path('/scratch/drn2/PROJECTS/TARA-LA4SR/MANUSCRIPT/kan_cca_results')
    else:
        data_dir = local_data
        output_dir = local_data

    output_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, output_dir

# ---------------------------------------------------------------------------
# Core bootstrap
# ---------------------------------------------------------------------------

N_BOOTSTRAP = 1000
N_PCA = 100
N_CCA = 10
TOP_K = 20
SEED = 42

def pca_reduce(X, n_components):
    """Center X, apply PCA, return (scores, components, mean).

    components: (n_components, p) — rows are principal axes.
    scores: (n, n_components) — projected data.
    """
    mu = X.mean(axis=0)
    Xc = X - mu
    # Economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = U[:, :n_components] * S[:n_components]
    components = Vt[:n_components, :]  # (n_components, p)
    return scores, components, mu

def cca(X, Y, n_components):
    """Standard CCA via SVD of cross-covariance.

    Returns:
      wx: (px, n_components) — X weights
      wy: (py, n_components) — Y weights
      corrs: (n_components,) — canonical correlations
    """
    n = X.shape[0]
    assert Y.shape[0] == n

    # Center
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    # Covariance matrices (regularized)
    reg = 1e-8
    Cxx = (Xc.T @ Xc) / (n - 1) + reg * np.eye(Xc.shape[1])
    Cyy = (Yc.T @ Yc) / (n - 1) + reg * np.eye(Yc.shape[1])
    Cxy = (Xc.T @ Yc) / (n - 1)

    # Whitening via Cholesky
    Lx = np.linalg.cholesky(Cxx)
    Ly = np.linalg.cholesky(Cyy)

    # Whitened cross-covariance
    Lx_inv = np.linalg.solve(Lx, np.eye(Lx.shape[0]))
    Ly_inv = np.linalg.solve(Ly, np.eye(Ly.shape[0]))
    M = Lx_inv @ Cxy @ Ly_inv.T

    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    nc = min(n_components, len(S))
    corrs = S[:nc]

    # Back to original space
    wx = np.linalg.solve(Lx.T, U[:, :nc])
    wy = np.linalg.solve(Ly.T, Vt[:nc, :].T)

    return wx, wy, corrs

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{ts}] 14_bootstrap_cca_loadings.py starting on {socket.gethostname()}")

    data_dir, output_dir = get_paths()
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load data
    X_domain = np.load(data_dir / 'X_domain.npy')
    X_env = np.load(data_dir / 'X_env.npy')
    with open(data_dir / 'feature_names.json') as f:
        feature_names = json.load(f)

    n, p = X_domain.shape
    _, q = X_env.shape
    domain_names = feature_names['domain_cols']
    assert len(domain_names) == p, f"Feature names mismatch: {len(domain_names)} vs {p}"

    print(f"Data: {n} samples, {p} domain features, {q} env features")
    print(f"Bootstrap: {N_BOOTSTRAP} resamples, PCA({N_PCA}), CCA({N_CCA})")
    print()

    # Storage for PFAM-level CC1 loadings across bootstrap resamples
    all_loadings = np.zeros((N_BOOTSTRAP, p), dtype=np.float64)
    # Storage for rank of each PFAM feature in CC1 (by |loading|)
    all_ranks = np.zeros((N_BOOTSTRAP, p), dtype=np.int32)
    # Track which features appear in top-K
    top_k_counts = np.zeros(p, dtype=np.int32)

    rng = np.random.RandomState(SEED)

    for b in range(N_BOOTSTRAP):
        if b % 100 == 0:
            print(f"  Bootstrap {b}/{N_BOOTSTRAP}...")

        # Resample with replacement
        idx = rng.choice(n, size=n, replace=True)
        X_d_boot = X_domain[idx]
        X_e_boot = X_env[idx]

        # PCA reduction on domain matrix
        scores, components, _ = pca_reduce(X_d_boot, N_PCA)
        # components: (N_PCA, p)

        # CCA between PCA scores and environment
        wx, wy, corrs = cca(scores, X_e_boot, N_CCA)
        # wx: (N_PCA, N_CCA) — weights in PCA space

        # Back-project CC1 weights to PFAM space:
        # PFAM loading = PCA_components^T @ CCA_weight_CC1
        # components is (N_PCA, p), wx[:,0] is (N_PCA,)
        pfam_loadings = components.T @ wx[:, 0]  # (p,)

        all_loadings[b] = pfam_loadings

        # Rank by |loading| (rank 1 = highest)
        abs_loadings = np.abs(pfam_loadings)
        ranks = np.argsort(-abs_loadings) + 1  # argsort gives indices; we want rank
        # Convert index-order to feature-rank:
        rank_of_feature = np.empty(p, dtype=np.int32)
        rank_of_feature[np.argsort(-abs_loadings)] = np.arange(1, p + 1)
        all_ranks[b] = rank_of_feature

        # Track top-K
        top_k_idx = np.argsort(-abs_loadings)[:TOP_K]
        top_k_counts[top_k_idx] += 1

    print(f"Bootstrap complete. Computing summary statistics...")

    # Summary statistics
    mean_loading = all_loadings.mean(axis=0)
    ci_lower = np.percentile(all_loadings, 2.5, axis=0)
    ci_upper = np.percentile(all_loadings, 97.5, axis=0)

    # Sign consistency: fraction of resamples with same sign as mean
    mean_sign = np.sign(mean_loading)
    # For each feature, count how often bootstrap sign matches mean sign
    sign_match = (np.sign(all_loadings) == mean_sign[np.newaxis, :]).mean(axis=0)
    # Handle zero mean (set consistency to 0.5)
    sign_match[mean_loading == 0] = 0.5

    # Median rank
    median_rank = np.median(all_ranks, axis=0)

    # Top-K frequency (fraction of bootstrap resamples where feature is in top-K)
    top_k_freq = top_k_counts / N_BOOTSTRAP

    # Build output DataFrame
    import pandas as pd

    results = pd.DataFrame({
        'pfam_id': domain_names,
        'mean_loading': mean_loading,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'sign_consistency': sign_match,
        'top20_frequency': top_k_freq,
        'median_rank': median_rank.astype(int),
    })

    # Sort by top20_frequency (descending), then by |mean_loading| (descending)
    results = results.sort_values(
        ['top20_frequency', 'mean_loading'],
        ascending=[False, False],
        key=lambda col: col.abs() if col.name == 'mean_loading' else col
    ).reset_index(drop=True)

    out_path = output_dir / f'bootstrap_cca_loadings_{ts}.tsv'
    # Write provenance header
    with open(out_path, 'w') as f:
        f.write(f"# Bootstrap CCA loading stability analysis\n")
        f.write(f"# Generated: {ts} on {socket.gethostname()}\n")
        f.write(f"# Script: 14_bootstrap_cca_loadings.py (ralph32 task 10)\n")
        f.write(f"# Input: {data_dir}/X_domain.npy ({n} x {p}), X_env.npy ({n} x {q})\n")
        f.write(f"# Bootstrap: {N_BOOTSTRAP} resamples, PCA({N_PCA}), CCA({N_CCA}), top-{TOP_K}\n")
        f.write(f"# Seed: {SEED}\n")
    results.to_csv(out_path, sep='\t', index=False, mode='a')
    print(f"Saved: {out_path} ({len(results)} features)")

    # Print top-20 most stable features
    print(f"\n=== Top-{TOP_K} Most Stable CC1 Loadings ===")
    print(f"{'PFAM':<15} {'Mean Loading':>12} {'95% CI':>20} {'Sign Cons':>10} "
          f"{'Top20 Freq':>10} {'Med Rank':>9}")
    for _, row in results.head(TOP_K).iterrows():
        print(f"{row['pfam_id']:<15} {row['mean_loading']:>12.6f} "
              f"[{row['ci_lower']:>8.6f}, {row['ci_upper']:>8.6f}] "
              f"{row['sign_consistency']:>10.3f} "
              f"{row['top20_frequency']:>10.3f} "
              f"{int(row['median_rank']):>9d}")

    # Summary statistics
    n_stable = (results['top20_frequency'] >= 0.5).sum()
    n_very_stable = (results['top20_frequency'] >= 0.8).sum()
    n_sign_consistent = (results['sign_consistency'] >= 0.95).sum()
    print(f"\n=== Summary ===")
    print(f"Features in top-{TOP_K} in >=50% of resamples: {n_stable}")
    print(f"Features in top-{TOP_K} in >=80% of resamples: {n_very_stable}")
    print(f"Features with >=95% sign consistency: {n_sign_consistent}")
    print(f"Timestamp: {ts}")

if __name__ == '__main__':
    main()
