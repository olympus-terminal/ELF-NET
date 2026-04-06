#!/usr/bin/env python3
"""
04_sparse_cca_cv.py — Sparse CCA with cross-validated sparsity selection.

Grid search over L1 constraints (c_u, c_v) using 10-fold spatial block CV.
For each (c_u, c_v) pair, fit sparse CCA on train, evaluate canonical
correlation on held-out test fold.

At optimal sparsity, extract top-12 components and save selected features.
"""

import json
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sparse_cca import sparse_cca

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_base_dir():
    scratch = Path('/scratch/drn2/PROJECTS/TARA-LA4SR')
    archive = Path('/archive/drn2/TARA-Oceans')
    return scratch, archive

BASE_DIR, ARCHIVE_DIR = get_base_dir()
OUTPUT_DIR = BASE_DIR / '03_analyses' / 'kan_cca'

def evaluate_on_test(X_train, Y_train, X_test, Y_test, c_u, c_v, n_components=3):
    """Fit sparse CCA on train, evaluate canonical correlation on test.

    Returns test correlations for each component.
    """
    U, V, train_corrs = sparse_cca(
        X_train, Y_train, c_u, c_v, n_components=n_components)

    test_corrs = []
    for k in range(n_components):
        Xu_test = X_test @ U[:, k]
        Yv_test = Y_test @ V[:, k]
        xu_std = np.std(Xu_test)
        yv_std = np.std(Yv_test)
        if xu_std > 1e-15 and yv_std > 1e-15:
            c = np.corrcoef(Xu_test, Yv_test)[0, 1]
        else:
            c = 0.0
        test_corrs.append(c)

    n_nonzero_u = np.sum(np.abs(U[:, 0]) > 1e-10)
    n_nonzero_v = np.sum(np.abs(V[:, 0]) > 1e-10)

    return np.array(test_corrs), np.array(train_corrs), n_nonzero_u, n_nonzero_v, U, V

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{ts}] 04_sparse_cca_cv.py starting on {socket.gethostname()}")

    # Load preprocessed data
    X_domain = np.load(OUTPUT_DIR / 'X_domain.npy')
    X_env = np.load(OUTPUT_DIR / 'X_env.npy')
    with open(OUTPUT_DIR / 'fold_indices.json') as f:
        folds = json.load(f)
    with open(OUTPUT_DIR / 'feature_names.json') as f:
        feature_names = json.load(f)

    n, p = X_domain.shape
    _, q = X_env.shape
    n_folds = len(folds)
    n_components = 12

    print(f"Data: {n} samples, {p} domain features, {q} env features")
    print(f"Folds: {n_folds}")
    print(f"sqrt(p) = {np.sqrt(p):.2f}, sqrt(q) = {np.sqrt(q):.2f}")

    # Grid of L1 constraints
    # c_u in [1, sqrt(p)] — sqrt(9466)~97.3
    # For domain: use fractions of sqrt(p)
    c_u_grid = [2.0, 5.0, 10.0, 20.0, 50.0, np.sqrt(p)]
    # c_v in [1, sqrt(q)] — sqrt(19)~4.36
    c_v_grid = [1.5, 2.0, 2.5, 3.0, np.sqrt(q)]

    print(f"c_u grid: {[f'{c:.1f}' for c in c_u_grid]}")
    print(f"c_v grid: {[f'{c:.1f}' for c in c_v_grid]}")
    print(f"Total grid points: {len(c_u_grid) * len(c_v_grid)}")
    print()

    # Grid search
    results = []
    best_mean_corr = -np.inf
    best_params = None

    for c_u in c_u_grid:
        for c_v in c_v_grid:
            fold_test_corrs = []
            fold_train_corrs = []
            fold_nnz_u = []
            fold_nnz_v = []

            for fold_i, fold in enumerate(folds):
                train_idx = np.array(fold['train'])
                test_idx = np.array(fold['test'])

                X_tr = X_domain[train_idx]
                Y_tr = X_env[train_idx]
                X_te = X_domain[test_idx]
                Y_te = X_env[test_idx]

                test_corrs, train_corrs, nnz_u, nnz_v, _, _ = evaluate_on_test(
                    X_tr, Y_tr, X_te, Y_te, c_u, c_v, n_components)

                fold_test_corrs.append(test_corrs)
                fold_train_corrs.append(train_corrs)
                fold_nnz_u.append(nnz_u)
                fold_nnz_v.append(nnz_v)

            fold_test_corrs = np.array(fold_test_corrs)  # (n_folds, n_components)
            fold_train_corrs = np.array(fold_train_corrs)

            # Mean first-component test correlation
            mean_test_c1 = np.mean(fold_test_corrs[:, 0])
            std_test_c1 = np.std(fold_test_corrs[:, 0])
            mean_train_c1 = np.mean(fold_train_corrs[:, 0])

            result = {
                'c_u': c_u,
                'c_v': c_v,
                'mean_test_corr_c1': mean_test_c1,
                'std_test_corr_c1': std_test_c1,
                'mean_train_corr_c1': mean_train_c1,
                'mean_nnz_u': np.mean(fold_nnz_u),
                'mean_nnz_v': np.mean(fold_nnz_v),
            }
            # Also store per-component means
            for k in range(n_components):
                result[f'mean_test_corr_c{k+1}'] = np.mean(fold_test_corrs[:, k])
                result[f'std_test_corr_c{k+1}'] = np.std(fold_test_corrs[:, k])

            results.append(result)

            print(f"c_u={c_u:6.1f}, c_v={c_v:4.1f}: "
                  f"test_c1={mean_test_c1:.4f}±{std_test_c1:.4f}, "
                  f"train_c1={mean_train_c1:.4f}, "
                  f"nnz_u={np.mean(fold_nnz_u):.0f}, nnz_v={np.mean(fold_nnz_v):.0f}")

            if mean_test_c1 > best_mean_corr:
                best_mean_corr = mean_test_c1
                best_params = (c_u, c_v)

    print(f"\nBest: c_u={best_params[0]:.1f}, c_v={best_params[1]:.1f}, "
          f"mean_test_c1={best_mean_corr:.4f}")

    # Save grid results
    import pandas as pd
    results_df = pd.DataFrame(results)
    grid_path = OUTPUT_DIR / f'sparse_cca_cv_results_{ts}.tsv'
    results_df.to_csv(grid_path, sep='\t', index=False)
    print(f"Saved grid results: {grid_path}")

    # Refit at optimal sparsity on full data
    print(f"\nRefitting at optimal sparsity on full data...")
    c_u_opt, c_v_opt = best_params
    U_full, V_full, corrs_full = sparse_cca(
        X_domain, X_env, c_u_opt, c_v_opt, n_components=n_components)

    print(f"Full-data correlations: {corrs_full}")

    # Extract selected features
    feature_rows = []
    for k in range(n_components):
        # Domain features
        u_k = U_full[:, k]
        nonzero_domain = np.where(np.abs(u_k) > 1e-10)[0]
        for idx in nonzero_domain:
            feature_rows.append({
                'component': k + 1,
                'feature_set': 'domain',
                'feature_name': feature_names['domain_cols'][idx],
                'feature_index': int(idx),
                'weight': u_k[idx],
                'abs_weight': abs(u_k[idx]),
            })

        # Env features
        v_k = V_full[:, k]
        nonzero_env = np.where(np.abs(v_k) > 1e-10)[0]
        for idx in nonzero_env:
            feature_rows.append({
                'component': k + 1,
                'feature_set': 'env',
                'feature_name': feature_names['env_cols'][idx],
                'feature_index': int(idx),
                'weight': v_k[idx],
                'abs_weight': abs(v_k[idx]),
            })

    features_df = pd.DataFrame(feature_rows)
    features_path = OUTPUT_DIR / f'sparse_cca_features_{ts}.tsv'
    features_df.to_csv(features_path, sep='\t', index=False)
    print(f"Saved features: {features_path} ({len(feature_rows)} rows)")

    # Save selected indices for downstream use
    selected = {
        'c_u_opt': float(c_u_opt),
        'c_v_opt': float(c_v_opt),
        'correlations_full': corrs_full.tolist(),
        'domain_indices': {},
        'env_indices': {},
    }
    for k in range(n_components):
        u_k = U_full[:, k]
        v_k = V_full[:, k]
        selected['domain_indices'][f'component_{k+1}'] = \
            np.where(np.abs(u_k) > 1e-10)[0].tolist()
        selected['env_indices'][f'component_{k+1}'] = \
            np.where(np.abs(v_k) > 1e-10)[0].tolist()

    indices_path = OUTPUT_DIR / 'sparse_cca_selected_indices.json'
    with open(indices_path, 'w') as f:
        json.dump(selected, f, indent=2)
    print(f"Saved indices: {indices_path}")

    # Also save per-fold results for each component at optimal params
    # (needed for downstream comparison with KAN-CCA)
    fold_results = []
    for fold_i, fold in enumerate(folds):
        train_idx = np.array(fold['train'])
        test_idx = np.array(fold['test'])

        X_tr = X_domain[train_idx]
        Y_tr = X_env[train_idx]
        X_te = X_domain[test_idx]
        Y_te = X_env[test_idx]

        test_corrs, train_corrs, nnz_u, nnz_v, _, _ = evaluate_on_test(
            X_tr, Y_tr, X_te, Y_te, c_u_opt, c_v_opt, n_components)

        for k in range(n_components):
            fold_results.append({
                'fold': fold_i,
                'component': k + 1,
                'linear_test_corr': test_corrs[k],
                'linear_train_corr': train_corrs[k],
                'nnz_u': nnz_u,
                'nnz_v': nnz_v,
            })

    fold_df = pd.DataFrame(fold_results)
    fold_path = OUTPUT_DIR / f'sparse_cca_fold_results_{ts}.tsv'
    fold_df.to_csv(fold_path, sep='\t', index=False)
    print(f"Saved per-fold results: {fold_path}")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Grid: {len(c_u_grid)} x {len(c_v_grid)} = {len(results)} parameter pairs")
    print(f"Best params: c_u={c_u_opt:.1f}, c_v={c_v_opt:.1f}")
    print(f"Best mean test corr (comp 1): {best_mean_corr:.4f}")
    print(f"Full-data correlations: {corrs_full}")
    for k in range(n_components):
        n_dom = len(selected['domain_indices'][f'component_{k+1}'])
        n_env = len(selected['env_indices'][f'component_{k+1}'])
        print(f"  Component {k+1}: {n_dom} domain + {n_env} env features")
    print(f"Timestamp: {ts}")

if __name__ == '__main__':
    main()
