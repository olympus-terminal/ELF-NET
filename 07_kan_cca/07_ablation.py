#!/usr/bin/env python3
"""
07_ablation.py — Ablation study: sparsity level vs KAN-CCA performance.

Run KAN-CCA at multiple sparsity levels (top-k features) with 10-fold
spatial block CV. Compares both linear CCA and KAN-CCA at each level.
"""

import json
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sparse_cca import sparse_cca
from kan_cca_model import train_kan_cca, evaluate_kan_cca

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_base_dir():
    scratch = Path('/scratch/drn2/PROJECTS/TARA-LA4SR')
    archive = Path('/archive/drn2/TARA-Oceans')
    return scratch, archive

BASE_DIR, ARCHIVE_DIR = get_base_dir()
OUTPUT_DIR = BASE_DIR / '03_analyses' / 'kan_cca'

N_COMPONENTS = 12
HIDDEN_DIM = 64
N_EPOCHS = 200
LR = 1e-3
PATIENCE = 20

# Ablation levels: number of top domain features to use
# (env features always all 19 — they're already sparse)
ABLATION_LEVELS = [6, 13, 50, 100, 500]

def standardize(X_train, X_test):
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mu) / std, (X_test - mu) / std

def get_top_k_domain_indices(X_domain, X_env, k, c_v=3.0):
    """Get top-k domain features by absolute sparse CCA weight on full data."""
    # Use sparse CCA with c_u tuned for ~k features
    # For simplicity, fit on full data and pick top-k by weight magnitude
    p = X_domain.shape[1]
    c_u = min(np.sqrt(p), max(1.1, k * 0.3))

    U, V, corrs = sparse_cca(X_domain, X_env, c_u, c_v, n_components=1)
    weights = np.abs(U[:, 0])
    top_indices = np.argsort(weights)[-k:]
    return sorted(top_indices.tolist())

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{ts}] 07_ablation.py starting on {socket.gethostname()}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    X_domain = np.load(OUTPUT_DIR / 'X_domain.npy')
    X_env = np.load(OUTPUT_DIR / 'X_env.npy')
    with open(OUTPUT_DIR / 'fold_indices.json') as f:
        folds = json.load(f)
    with open(OUTPUT_DIR / 'sparse_cca_selected_indices.json') as f:
        selected = json.load(f)

    c_v_opt = selected['c_v_opt']
    n_folds = len(folds)

    print(f"Data: {X_domain.shape[0]} samples, {X_domain.shape[1]} domains, {X_env.shape[1]} env")
    print(f"Ablation levels: {ABLATION_LEVELS}")

    results = []

    for k in ABLATION_LEVELS:
        print(f"\n=== Ablation: top-{k} domain features ===")

        # Get top-k domain indices
        if k >= X_domain.shape[1]:
            domain_idx = list(range(X_domain.shape[1]))
        else:
            domain_idx = get_top_k_domain_indices(X_domain, X_env, k, c_v_opt)
        print(f"  Selected {len(domain_idx)} domain features")

        X_d_sub = X_domain[:, domain_idx]

        for fold_i, fold in enumerate(folds):
            train_idx = np.array(fold['train'])
            test_idx = np.array(fold['test'])

            X_d_tr, X_d_te = standardize(X_d_sub[train_idx], X_d_sub[test_idx])
            X_e_tr, X_e_te = standardize(X_env[train_idx], X_env[test_idx])

            # Linear CCA (sparse, with appropriate c_u)
            c_u_lin = min(np.sqrt(len(domain_idx)), max(1.1, len(domain_idx) * 0.3))
            U, V, train_corrs = sparse_cca(X_d_tr, X_e_tr, c_u_lin, c_v_opt, N_COMPONENTS)
            linear_test_corrs = []
            for comp in range(N_COMPONENTS):
                Xu = X_d_te @ U[:, comp]
                Yv = X_e_te @ V[:, comp]
                if np.std(Xu) > 1e-10 and np.std(Yv) > 1e-10:
                    linear_test_corrs.append(np.corrcoef(Xu, Yv)[0, 1])
                else:
                    linear_test_corrs.append(0.0)

            # KAN-CCA
            model, history = train_kan_cca(
                X_d_tr, X_e_tr, n_components=N_COMPONENTS,
                hidden_dim=min(HIDDEN_DIM, max(8, len(domain_idx))),
                n_epochs=N_EPOCHS, lr=LR, patience=PATIENCE, device=device)

            kan_test_corrs = evaluate_kan_cca(model, X_d_te, X_e_te, device)

            for comp in range(N_COMPONENTS):
                results.append({
                    'k_domain': len(domain_idx),
                    'fold': fold_i,
                    'component': comp + 1,
                    'linear_test_corr': linear_test_corrs[comp],
                    'kan_test_corr': kan_test_corrs[comp],
                })

        # Quick summary for this level
        level_results = [r for r in results if r['k_domain'] == len(domain_idx)]
        for comp in range(1, N_COMPONENTS + 1):
            comp_data = [r for r in level_results if r['component'] == comp]
            lin_mean = np.mean([r['linear_test_corr'] for r in comp_data])
            kan_mean = np.mean([r['kan_test_corr'] for r in comp_data])
            print(f"  Comp {comp}: linear={lin_mean:.4f}, KAN={kan_mean:.4f}")

    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / f'kan_cca_ablation_{ts}.tsv'
    df.to_csv(out_path, sep='\t', index=False)
    print(f"\nSaved: {out_path}")

    # Summary table
    print(f"\n=== ABLATION SUMMARY (Component 1) ===")
    print(f"{'k_domain':>8s} {'Linear':>10s} {'KAN':>10s} {'Delta':>8s}")
    for k in sorted(df['k_domain'].unique()):
        c1 = df[(df['k_domain'] == k) & (df['component'] == 1)]
        lin = c1['linear_test_corr'].mean()
        kan = c1['kan_test_corr'].mean()
        print(f"{k:8d} {lin:10.4f} {kan:10.4f} {kan-lin:+8.4f}")

    print(f"\nTimestamp: {ts}")

if __name__ == '__main__':
    main()
