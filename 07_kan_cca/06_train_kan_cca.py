#!/usr/bin/env python3
"""
06_train_kan_cca.py — Train KAN-CCA with 10-fold spatial block CV.

For each fold:
  1. Select sparse feature subsets (union of all component indices)
  2. Standardize train/test
  3. Train KAN-CCA
  4. Compute linear CCA on same folds for comparison
  5. Extract spline activations from best model
  6. Record per-epoch train/test correlations

Output:
  - kan_cca_cv_results_TIMESTAMP.tsv: per-fold, per-component comparison
  - kan_cca_spline_activations_TIMESTAMP.tsv: spline parameters for interpretability
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
from kan_cca_model import KANCCAModel, train_kan_cca, evaluate_kan_cca

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
WEIGHT_DECAY = 1e-4
LAMBDA_L1 = 1e-3
LAMBDA_SMOOTH = 1e-3
PATIENCE = 20

def standardize(X_train, X_test):
    """Standardize test using train statistics."""
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mu) / std, (X_test - mu) / std

def linear_cca_on_fold(X_d_train, X_e_train, X_d_test, X_e_test, c_u, c_v, n_comp):
    """Fit sparse linear CCA on train, evaluate on test."""
    U, V, train_corrs = sparse_cca(X_d_train, X_e_train, c_u, c_v, n_comp)

    test_corrs = []
    for k in range(n_comp):
        Xu = X_d_test @ U[:, k]
        Yv = X_e_test @ V[:, k]
        if np.std(Xu) > 1e-10 and np.std(Yv) > 1e-10:
            c = np.corrcoef(Xu, Yv)[0, 1]
        else:
            c = 0.0
        test_corrs.append(c)

    return np.array(train_corrs), np.array(test_corrs)

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{ts}] 06_train_kan_cca.py starting on {socket.gethostname()}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    X_domain = np.load(OUTPUT_DIR / 'X_domain.npy')
    X_env = np.load(OUTPUT_DIR / 'X_env.npy')
    with open(OUTPUT_DIR / 'fold_indices.json') as f:
        folds = json.load(f)
    with open(OUTPUT_DIR / 'sparse_cca_selected_indices.json') as f:
        selected = json.load(f)
    with open(OUTPUT_DIR / 'feature_names.json') as f:
        feature_names = json.load(f)

    n_folds = len(folds)
    c_u_opt = selected['c_u_opt']
    c_v_opt = selected['c_v_opt']

    # Get union of selected feature indices across all components
    domain_idx = sorted(set(
        idx for comp in selected['domain_indices'].values() for idx in comp
    ))
    env_idx = sorted(set(
        idx for comp in selected['env_indices'].values() for idx in comp
    ))

    print(f"Data: {X_domain.shape[0]} samples")
    print(f"Sparse domain features (union): {len(domain_idx)}")
    print(f"Sparse env features (union): {len(env_idx)}")
    print(f"Optimal sparsity: c_u={c_u_opt}, c_v={c_v_opt}")
    print(f"KAN params: hidden={HIDDEN_DIM}, epochs={N_EPOCHS}, lr={LR}, patience={PATIENCE}")
    print()

    # Get feature names for the selected indices
    domain_feature_names = [feature_names['domain_cols'][i] for i in domain_idx]
    env_feature_names = [feature_names['env_cols'][i] for i in env_idx]
    print(f"Selected domain features: {domain_feature_names}")
    print(f"Selected env features: {env_feature_names}")
    print()

    # Subset to sparse features
    X_d_sparse = X_domain[:, domain_idx]
    X_e_sparse = X_env[:, env_idx]

    results = []
    all_spline_data = []

    for fold_i, fold in enumerate(folds):
        print(f"--- Fold {fold_i} ---")
        train_idx = np.array(fold['train'])
        test_idx = np.array(fold['test'])

        # Split and standardize
        X_d_tr, X_d_te = standardize(X_d_sparse[train_idx], X_d_sparse[test_idx])
        X_e_tr, X_e_te = standardize(X_e_sparse[train_idx], X_e_sparse[test_idx])

        # --- Linear CCA ---
        # Use full feature sets for linear CCA (same sparsity params)
        X_d_tr_full, X_d_te_full = standardize(
            X_domain[train_idx], X_domain[test_idx])
        X_e_tr_full, X_e_te_full = standardize(
            X_env[train_idx], X_env[test_idx])

        linear_train_corrs, linear_test_corrs = linear_cca_on_fold(
            X_d_tr_full, X_e_tr_full, X_d_te_full, X_e_te_full,
            c_u_opt, c_v_opt, N_COMPONENTS)

        print(f"  Linear CCA test corrs: {linear_test_corrs}")

        # --- KAN-CCA ---
        model, history = train_kan_cca(
            X_d_tr, X_e_tr,
            n_components=N_COMPONENTS,
            hidden_dim=HIDDEN_DIM,
            n_epochs=N_EPOCHS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            lambda_l1=LAMBDA_L1,
            lambda_smooth=LAMBDA_SMOOTH,
            patience=PATIENCE,
            device=device,
        )

        # Evaluate on train and test
        kan_train_corrs = evaluate_kan_cca(model, X_d_tr, X_e_tr, device)
        kan_test_corrs = evaluate_kan_cca(model, X_d_te, X_e_te, device)
        best_epoch = len(history)

        print(f"  KAN-CCA test corrs: {kan_test_corrs} (best epoch: {best_epoch})")

        # Store results
        for k in range(N_COMPONENTS):
            results.append({
                'fold': fold_i,
                'component': k + 1,
                'linear_train_corr': linear_train_corrs[k],
                'linear_test_corr': linear_test_corrs[k],
                'kan_train_corr': kan_train_corrs[k],
                'kan_test_corr': kan_test_corrs[k],
                'n_epochs': len(history),
                'best_epoch': best_epoch,
                'n_domain_features': len(domain_idx),
                'n_env_features': len(env_idx),
            })

        # Extract spline activations from best model (first layer only)
        model.cpu()
        splines = model.extract_first_layer_splines(x_range=(-2, 2), n_points=100)

        # Save domain first-layer spline activations for top features
        for branch_name, branch_splines in splines.items():
            feat_names = domain_feature_names if branch_name == 'domain' else env_feature_names
            for (i, j), (x_vals, y_vals) in branch_splines.items():
                if j == 0:  # only first output unit for compactness
                    all_spline_data.append({
                        'fold': fold_i,
                        'branch': branch_name,
                        'input_idx': i,
                        'feature_name': feat_names[i] if i < len(feat_names) else f'hidden_{i}',
                        'output_idx': j,
                        'x_vals': ','.join(f'{v:.4f}' for v in x_vals),
                        'y_vals': ','.join(f'{v:.4f}' for v in y_vals),
                    })

    # Save results
    import pandas as pd

    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / f'kan_cca_cv_results_{ts}.tsv'
    results_df.to_csv(results_path, sep='\t', index=False)
    print(f"\nSaved CV results: {results_path}")

    spline_df = pd.DataFrame(all_spline_data)
    spline_path = OUTPUT_DIR / f'kan_cca_spline_activations_{ts}.tsv'
    spline_df.to_csv(spline_path, sep='\t', index=False)
    print(f"Saved spline activations: {spline_path}")

    # Summary
    print(f"\n=== SUMMARY ===")
    for k in range(N_COMPONENTS):
        comp_data = results_df[results_df['component'] == k + 1]
        lin_mean = comp_data['linear_test_corr'].mean()
        lin_std = comp_data['linear_test_corr'].std()
        kan_mean = comp_data['kan_test_corr'].mean()
        kan_std = comp_data['kan_test_corr'].std()
        improvement = (kan_mean - lin_mean) / abs(lin_mean) * 100 if abs(lin_mean) > 1e-6 else 0
        print(f"Component {k+1}: Linear={lin_mean:.4f}±{lin_std:.4f}, "
              f"KAN={kan_mean:.4f}±{kan_std:.4f}, "
              f"improvement={improvement:+.1f}%")
    print(f"Timestamp: {ts}")

if __name__ == '__main__':
    main()
