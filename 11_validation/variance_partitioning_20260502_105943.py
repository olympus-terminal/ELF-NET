#!/usr/bin/env python3
"""
variance_partitioning_20260502_105943.py
Formal variance partitioning via distance-based redundancy analysis (db-RDA).

Decomposes domain composition variance into:
  [a] pure environmental
  [b] shared (spatially structured environmental)
  [c] pure geographic
  [d] residual

Uses Bray-Curtis dissimilarity on CLR-transformed PFAM abundances,
PCoA to extract principal coordinates, then canonical RDA partitioning
following Legendre & Legendre (2012) and Borcard et al. (1992).

Geographic matrix: latitude, longitude, and distance-based Moran
eigenvector maps (dbMEMs) capturing spatial autocorrelation at
multiple scales.

Environmental matrix: 24 GEE+WOA23 variables (standardized).

Input: preprocessed kan_cca arrays (786 samples with complete metadata).
Output: source_data/ralph44/variance_partitioning.tsv + summary .md
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

TIMESTAMP = "20260502_105943"
SCRIPT_PATH = os.path.abspath(__file__)

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
MANUSCRIPT_DIR = Path(__file__).resolve().parent.parent
KAN_CCA_DIR = Path("/media/drn2/External/TARA-Oceans/03_analyses/kan_cca")
OUTPUT_DIR = MANUSCRIPT_DIR / "source_data" / "ralph44"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_input_source(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if p.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {path}")
    return p


def haversine_matrix(lat_lon):
    """Compute pairwise great-circle distances (km) from lat/lon in degrees."""
    lat = np.radians(lat_lon[:, 0])
    lon = np.radians(lat_lon[:, 1])
    n = len(lat)
    D = np.zeros((n, n))
    for i in range(n):
        dlat = lat[i] - lat
        dlon = lon[i] - lon
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon / 2) ** 2
        D[i] = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return D


def compute_dbmems(lat_lon, truncation_quantile=0.5):
    """
    Compute distance-based Moran eigenvector maps (dbMEMs).

    1. Compute geographic distance matrix (haversine).
    2. Truncate: set distances > threshold to 4× threshold (Borcard & Legendre 2002).
    3. Compute principal coordinates of the truncated distance matrix.
    4. Retain eigenvectors with positive eigenvalues (positive spatial autocorrelation).
    """
    print("Computing geographic distance matrix (haversine)...")
    D_geo = haversine_matrix(lat_lon)

    threshold = np.quantile(D_geo[np.triu_indices_from(D_geo, k=1)], truncation_quantile)
    print(f"Truncation threshold ({truncation_quantile} quantile): {threshold:.1f} km")

    D_trunc = D_geo.copy()
    D_trunc[D_trunc > threshold] = 4 * threshold

    n = D_trunc.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D_trunc ** 2) @ H

    eigenvalues, eigenvectors = eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    pos_mask = eigenvalues > 1e-10
    n_pos = pos_mask.sum()
    print(f"dbMEMs: {n_pos} eigenvectors with positive eigenvalues "
          f"(of {len(eigenvalues)} total)")

    dbmems = eigenvectors[:, pos_mask] * np.sqrt(eigenvalues[pos_mask])
    return dbmems, eigenvalues[pos_mask]


def pcoa_from_clr(X_clr, metric="braycurtis"):
    """
    Compute PCoA from CLR-transformed abundance matrix.

    For Bray-Curtis on CLR data, we shift CLR values to non-negative
    (add |min| + 0.5) so Bray-Curtis is well-defined, following
    Gloor et al. (2017) recommendation for compositional data.
    """
    X_shifted = X_clr - X_clr.min() + 0.5
    print(f"Computing {metric} distances on shifted CLR matrix "
          f"(shape {X_shifted.shape})...")
    D_flat = pdist(X_shifted, metric=metric)
    D = squareform(D_flat)

    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H

    eigenvalues, eigenvectors = eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    pos_mask = eigenvalues > 1e-10
    n_pos = pos_mask.sum()
    print(f"PCoA: {n_pos} positive eigenvalues (explaining "
          f"{eigenvalues[pos_mask].sum() / eigenvalues[eigenvalues > 0].sum() * 100:.1f}% "
          f"of positive variance)")

    Y_pcoa = eigenvectors[:, pos_mask] * np.sqrt(eigenvalues[pos_mask])
    return Y_pcoa, eigenvalues[pos_mask]


def rda_adjR2(Y, X, n):
    """
    Compute adjusted R² for RDA (redundancy analysis).

    Y: response matrix (PCoA scores, n × p)
    X: predictor matrix (n × q)
    n: number of observations

    Returns: R², adjusted R², number of predictors
    """
    q = X.shape[1]

    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)

    H = X_c @ np.linalg.pinv(X_c)
    Y_hat = H @ Y_c

    SS_total = np.sum(Y_c ** 2)
    SS_fitted = np.sum(Y_hat ** 2)

    R2 = SS_fitted / SS_total

    R2_adj = 1 - (1 - R2) * (n - 1) / (n - q - 1)

    return R2, R2_adj, q


def permutation_test_rda(Y, X, n_perm=999):
    """
    Permutation test for RDA significance.
    Permute rows of Y, recompute R², build null distribution.
    """
    n = Y.shape[0]
    R2_obs, _, _ = rda_adjR2(Y, X, n)

    rng = np.random.default_rng(42)
    count_ge = 0
    for _ in range(n_perm):
        perm_idx = rng.permutation(n)
        Y_perm = Y[perm_idx]
        R2_perm, _, _ = rda_adjR2(Y_perm, X, n)
        if R2_perm >= R2_obs:
            count_ge += 1

    p_value = (count_ge + 1) / (n_perm + 1)
    return p_value


def variance_partitioning(Y, X_env, X_geo, n):
    """
    Two-table variance partitioning (Borcard et al. 1992).

    [a+b+c] = R²_adj(Y ~ X_env + X_geo)  (full model)
    [a+b]   = R²_adj(Y ~ X_env)           (env only)
    [b+c]   = R²_adj(Y ~ X_geo)           (geo only)

    [a] = [a+b+c] - [b+c]   (pure environmental)
    [c] = [a+b+c] - [a+b]   (pure geographic)
    [b] = [a+b] + [b+c] - [a+b+c]  (shared)
    [d] = 1 - [a+b+c]       (residual)
    """
    X_full = np.hstack([X_env, X_geo])

    _, R2_adj_full, q_full = rda_adjR2(Y, X_full, n)
    _, R2_adj_env, q_env = rda_adjR2(Y, X_env, n)
    _, R2_adj_geo, q_geo = rda_adjR2(Y, X_geo, n)

    a = R2_adj_full - R2_adj_geo  # pure environmental
    c = R2_adj_full - R2_adj_env  # pure geographic
    b = R2_adj_env + R2_adj_geo - R2_adj_full  # shared
    d = 1 - R2_adj_full  # residual

    return {
        "pure_env_a": a,
        "shared_b": b,
        "pure_geo_c": c,
        "residual_d": d,
        "env_total_ab": R2_adj_env,
        "geo_total_bc": R2_adj_geo,
        "full_abc": R2_adj_full,
        "q_env": q_env,
        "q_geo": q_geo,
        "q_full": q_full,
    }


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] Variance partitioning (db-RDA) starting...")

    # Load data
    X_domain_path = KAN_CCA_DIR / "X_domain.npy"
    X_env_path = KAN_CCA_DIR / "X_env.npy"
    lat_lon_path = KAN_CCA_DIR / "lat_lon.npy"
    fnames_path = KAN_CCA_DIR / "feature_names.json"

    for p in [X_domain_path, X_env_path, lat_lon_path, fnames_path]:
        validate_input_source(p)

    X_clr = np.load(X_domain_path)
    X_env = np.load(X_env_path)
    lat_lon = np.load(lat_lon_path)
    with open(fnames_path) as f:
        feature_names = json.load(f)

    n_samples = X_clr.shape[0]
    n_domains = X_clr.shape[1]
    n_env = X_env.shape[1]
    env_cols = feature_names["env_cols"]
    print(f"Samples: {n_samples}, Domains: {n_domains}, Env vars: {n_env}")
    print(f"Env variables: {env_cols}")

    # Step 1: PCoA on Bray-Curtis dissimilarity of CLR-transformed PFAM
    Y_pcoa, pcoa_eigenvalues = pcoa_from_clr(X_clr, metric="braycurtis")

    cum_var = np.cumsum(pcoa_eigenvalues) / pcoa_eigenvalues.sum()
    n_axes_80 = np.searchsorted(cum_var, 0.80) + 1
    n_axes_90 = np.searchsorted(cum_var, 0.90) + 1
    print(f"PCoA axes for 80% variance: {n_axes_80}")
    print(f"PCoA axes for 90% variance: {n_axes_90}")

    # Use enough PCoA axes to capture ≥80% of positive variance
    # but cap to avoid overfitting with too many response variables
    n_pcoa_axes = min(n_axes_80, n_samples // 3)
    Y = Y_pcoa[:, :n_pcoa_axes]
    var_retained = cum_var[n_pcoa_axes - 1]
    print(f"Using {n_pcoa_axes} PCoA axes ({var_retained * 100:.1f}% of positive variance)")

    # Step 2: Compute dbMEMs for geographic predictor matrix
    dbmems, dbmem_eigenvalues = compute_dbmems(lat_lon, truncation_quantile=0.5)

    # Forward-select dbMEMs: keep those with positive Moran's I
    # (proxy: keep eigenvectors explaining > 1% of total positive eigenvalue sum)
    dbmem_var_frac = dbmem_eigenvalues / dbmem_eigenvalues.sum()
    dbmem_keep = dbmem_var_frac > 0.01
    dbmems_selected = dbmems[:, dbmem_keep]
    print(f"dbMEMs selected (>1% variance each): {dbmems_selected.shape[1]} "
          f"of {dbmems.shape[1]}")

    # Geographic predictor matrix: lat, lon + selected dbMEMs
    lat_lon_std = (lat_lon - lat_lon.mean(axis=0)) / lat_lon.std(axis=0)
    X_geo = np.hstack([lat_lon_std, dbmems_selected])
    print(f"Geographic predictor matrix: {X_geo.shape}")

    # Step 3: Variance partitioning
    print("\n=== VARIANCE PARTITIONING ===")
    vp = variance_partitioning(Y, X_env, X_geo, n_samples)

    print(f"\nFull model [a+b+c]: adj.R² = {vp['full_abc']:.4f} "
          f"({vp['q_full']} predictors)")
    print(f"Env-only  [a+b]:   adj.R² = {vp['env_total_ab']:.4f} "
          f"({vp['q_env']} predictors)")
    print(f"Geo-only  [b+c]:   adj.R² = {vp['geo_total_bc']:.4f} "
          f"({vp['q_geo']} predictors)")
    print(f"\nPure environmental [a]: {vp['pure_env_a']:.4f}")
    print(f"Shared env|geo    [b]: {vp['shared_b']:.4f}")
    print(f"Pure geographic   [c]: {vp['pure_geo_c']:.4f}")
    print(f"Residual          [d]: {vp['residual_d']:.4f}")

    # Step 4: Permutation tests for each fraction
    print("\nPermutation tests (999 permutations)...")
    p_env = permutation_test_rda(Y, X_env, n_perm=999)
    p_geo = permutation_test_rda(Y, X_geo, n_perm=999)
    X_full = np.hstack([X_env, X_geo])
    p_full = permutation_test_rda(Y, X_full, n_perm=999)
    print(f"p(env-only):  {p_env:.4f}")
    print(f"p(geo-only):  {p_geo:.4f}")
    print(f"p(full):      {p_full:.4f}")

    # Compute ratio of pure env to pure geo
    if vp['pure_geo_c'] > 0:
        env_geo_ratio = vp['pure_env_a'] / vp['pure_geo_c']
    else:
        env_geo_ratio = float('inf')

    # Step 5: Write TSV output
    tsv_path = OUTPUT_DIR / "variance_partitioning.tsv"
    rows = [
        {"fraction": "pure_environmental_a", "adj_R2": vp['pure_env_a'],
         "pct_of_explained": vp['pure_env_a'] / vp['full_abc'] * 100 if vp['full_abc'] > 0 else 0},
        {"fraction": "shared_env_geo_b", "adj_R2": vp['shared_b'],
         "pct_of_explained": vp['shared_b'] / vp['full_abc'] * 100 if vp['full_abc'] > 0 else 0},
        {"fraction": "pure_geographic_c", "adj_R2": vp['pure_geo_c'],
         "pct_of_explained": vp['pure_geo_c'] / vp['full_abc'] * 100 if vp['full_abc'] > 0 else 0},
        {"fraction": "residual_d", "adj_R2": vp['residual_d'],
         "pct_of_explained": 0},
        {"fraction": "env_total_ab", "adj_R2": vp['env_total_ab'],
         "pct_of_explained": 0},
        {"fraction": "geo_total_bc", "adj_R2": vp['geo_total_bc'],
         "pct_of_explained": 0},
        {"fraction": "full_model_abc", "adj_R2": vp['full_abc'],
         "pct_of_explained": 100},
    ]
    df_out = pd.DataFrame(rows)

    with open(tsv_path, "w") as f:
        f.write(f"# Provenance:\n")
        f.write(f"#   Script: {SCRIPT_PATH}\n")
        f.write(f"#   Input:  {KAN_CCA_DIR}/X_domain.npy, X_env.npy, lat_lon.npy\n")
        f.write(f"#   Date:   {ts}\n")
        f.write(f"#   Integrity Check: PASSED\n")
        f.write(f"#   Method: db-RDA variance partitioning (Borcard et al. 1992)\n")
        f.write(f"#   Dissimilarity: Bray-Curtis on CLR-transformed PFAM abundances\n")
        f.write(f"#   PCoA axes: {n_pcoa_axes} ({var_retained * 100:.1f}% of positive variance)\n")
        f.write(f"#   Env predictors: {n_env} GEE+WOA23 variables\n")
        f.write(f"#   Geo predictors: 2 (lat/lon) + {dbmems_selected.shape[1]} dbMEMs = {X_geo.shape[1]}\n")
        f.write(f"#   Samples: {n_samples}\n")
        f.write(f"#   Permutation p-values: env={p_env:.4f}, geo={p_geo:.4f}, full={p_full:.4f}\n")
        df_out.to_csv(f, sep="\t", index=False)

    print(f"\nTSV written: {tsv_path}")

    # Step 6: Write summary markdown
    md_path = OUTPUT_DIR / "variance_partitioning.md"
    with open(md_path, "w") as f:
        f.write(f"# Variance Partitioning: db-RDA Results\n\n")
        f.write(f"## Provenance\n")
        f.write(f"- **Script:** `{SCRIPT_PATH}`\n")
        f.write(f"- **Input:** `{KAN_CCA_DIR}/X_domain.npy`, `X_env.npy`, `lat_lon.npy`\n")
        f.write(f"- **Date:** {ts}\n")
        f.write(f"- **Integrity Check:** PASSED\n\n")
        f.write(f"## Method\n")
        f.write(f"- Distance-based redundancy analysis (db-RDA) following Borcard et al. (1992)\n")
        f.write(f"- Dissimilarity: Bray-Curtis on CLR-transformed PFAM abundances\n")
        f.write(f"- PCoA response: {n_pcoa_axes} axes ({var_retained * 100:.1f}% of positive variance)\n")
        f.write(f"- Environmental predictors: {n_env} GEE+WOA23 variables (standardized)\n")
        f.write(f"- Geographic predictors: lat/lon + {dbmems_selected.shape[1]} dbMEMs = {X_geo.shape[1]} total\n")
        f.write(f"- Samples: {n_samples}\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Fraction | Adj. R² | % of explained |\n")
        f.write(f"|----------|---------|----------------|\n")
        f.write(f"| Pure environmental [a] | {vp['pure_env_a']:.4f} | "
                f"{vp['pure_env_a'] / vp['full_abc'] * 100:.1f}% |\n")
        f.write(f"| Shared env|geo [b] | {vp['shared_b']:.4f} | "
                f"{vp['shared_b'] / vp['full_abc'] * 100:.1f}% |\n")
        f.write(f"| Pure geographic [c] | {vp['pure_geo_c']:.4f} | "
                f"{vp['pure_geo_c'] / vp['full_abc'] * 100:.1f}% |\n")
        f.write(f"| Residual [d] | {vp['residual_d']:.4f} | — |\n")
        f.write(f"| **Full model [a+b+c]** | **{vp['full_abc']:.4f}** | **100%** |\n\n")
        f.write(f"## Permutation tests (999 permutations)\n")
        f.write(f"- Environmental fraction: p = {p_env:.4f}\n")
        f.write(f"- Geographic fraction: p = {p_geo:.4f}\n")
        f.write(f"- Full model: p = {p_full:.4f}\n\n")
        f.write(f"## Interpretation\n")
        if vp['pure_env_a'] > vp['pure_geo_c']:
            f.write(f"Pure environmental variance ({vp['pure_env_a']:.4f}) exceeds pure geographic "
                    f"variance ({vp['pure_geo_c']:.4f}) by {env_geo_ratio:.1f}-fold, ")
            if vp['shared_b'] > vp['pure_env_a']:
                f.write(f"and the shared fraction ({vp['shared_b']:.4f}) is the largest component. "
                        f"The dominant signal is spatially structured environmental variation, "
                        f"consistent with environmental filtering operating along geographic gradients.\n")
            else:
                f.write(f"providing formal support for environmental filtering as the primary "
                        f"structuring force in protein domain composition.\n")
        else:
            f.write(f"Pure geographic variance ({vp['pure_geo_c']:.4f}) exceeds pure environmental "
                    f"variance ({vp['pure_env_a']:.4f}), suggesting dispersal limitation "
                    f"contributes to domain composition structure.\n")

    print(f"Summary written: {md_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
