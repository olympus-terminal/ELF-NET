#!/usr/bin/env python3
"""
13_temporal_holdout.py — Temporal holdout validation for KAN-CCA.

Train on samples from 2009-2011, test on 2012-2013+.
Compares linear sparse CCA vs KAN-CCA on the temporal split to test
whether the nonlinear CC2-CC3 advantage persists across sampling periods.

Steps:
  1. Load preprocessed data (X_domain, X_env, sample_ids)
  2. Load merged TSV, extract collection_date per sample
  3. Parse years, define train (2009-2011) and test (2012-2013+)
  4. Apply same sparse feature selection as main analysis
  5. Fit linear sparse CCA on train, evaluate on test
  6. Fit KAN-CCA on train, evaluate on test (3 random seeds)
  7. Report per-component canonical correlations

Output:
  - temporal_holdout_TIMESTAMP.tsv: per-seed, per-component comparison
"""

import json
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
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

def find_data(relative_path):
    for base in [BASE_DIR, ARCHIVE_DIR]:
        p = base / relative_path
        if p.exists():
            return p
    raise FileNotFoundError(f"Not found in scratch or archive: {relative_path}")

# Hyperparameters (same as 06_train_kan_cca.py)
N_COMPONENTS = 3
HIDDEN_DIM = 32
N_EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_L1 = 1e-3
LAMBDA_SMOOTH = 1e-3
PATIENCE = 20
N_SEEDS = 3  # multiple seeds to assess variance

def standardize(X_train, X_test):
    """Standardize test using train statistics."""
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mu) / std, (X_test - mu) / std

def get_collection_years(sample_ids):
    """Load collection dates from merged TSV and parse years."""
    data_path = find_data(
        '03_analyses/ALGAGPT-based-analyses/'
        'algagpt_gee_pfam_merged_SMART_20260119_100639.tsv'
    )
    print(f"Loading merged TSV for dates: {data_path}")
    df = pd.read_csv(data_path, sep='\t', comment='#', low_memory=False,
                     usecols=['assembly_id', 'collection_date'])

    # Build assembly_id -> year mapping
    id_to_year = {}
    for _, row in df.iterrows():
        aid = row['assembly_id']
        date_str = row.get('collection_date', '')
        if pd.isna(date_str) or date_str == '':
            continue
        try:
            # Try parsing various date formats
            date_str = str(date_str).strip()
            if len(date_str) >= 4:
                year = int(date_str[:4])
                if 2000 <= year <= 2030:
                    id_to_year[aid] = year
        except (ValueError, TypeError):
            continue

    # Map sample_ids to years
    years = np.array([id_to_year.get(sid, -1) for sid in sample_ids])
    return years

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{ts}] 13_temporal_holdout.py starting on {socket.gethostname()}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. Load preprocessed data
    X_domain = np.load(OUTPUT_DIR / 'X_domain.npy')
    X_env = np.load(OUTPUT_DIR / 'X_env.npy')
    sample_ids = np.load(OUTPUT_DIR / 'sample_ids.npy', allow_pickle=True)
    with open(OUTPUT_DIR / 'sparse_cca_selected_indices.json') as f:
        selected = json.load(f)
    with open(OUTPUT_DIR / 'feature_names.json') as f:
        feature_names = json.load(f)

    print(f"Data: {X_domain.shape[0]} samples, "
          f"{X_domain.shape[1]} domains, {X_env.shape[1]} env features")

    c_u_opt = selected['c_u_opt']
    c_v_opt = selected['c_v_opt']

    # Sparse feature indices (same as main analysis)
    domain_idx = sorted(set(
        idx for comp in selected['domain_indices'].values() for idx in comp
    ))
    env_idx = sorted(set(
        idx for comp in selected['env_indices'].values() for idx in comp
    ))
    domain_feature_names = [feature_names['domain_cols'][i] for i in domain_idx]
    env_feature_names = [feature_names['env_cols'][i] for i in env_idx]

    print(f"Sparse features: {len(domain_idx)} domain, {len(env_idx)} env")
    print(f"Optimal sparsity: c_u={c_u_opt}, c_v={c_v_opt}")

    # 2. Get collection years
    years = get_collection_years(sample_ids)
    valid_years = years[years > 0]
    print(f"\nYear distribution (samples with valid dates: {len(valid_years)}/{len(years)}):")
    for yr in sorted(np.unique(valid_years)):
        print(f"  {yr}: {(valid_years == yr).sum()} samples")

    # 3. Define temporal split
    train_years = {2009, 2010, 2011}
    test_years = {2012, 2013, 2014, 2015}  # include all later years

    train_mask = np.isin(years, list(train_years))
    test_mask = np.isin(years, list(test_years))

    n_train = train_mask.sum()
    n_test = test_mask.sum()
    n_no_date = (years < 0).sum()

    print(f"\nTemporal split:")
    print(f"  Train ({sorted(train_years)}): {n_train} samples")
    print(f"  Test ({sorted(test_years)}): {n_test} samples")
    print(f"  No date / excluded: {n_no_date} samples")

    if n_train < 50 or n_test < 50:
        print(f"WARNING: Very small split sizes. Results may be unreliable.")
        if n_train < 10 or n_test < 10:
            print(f"ERROR: Insufficient samples for temporal holdout. Aborting.")
            sys.exit(1)

    # 4. Subset sparse features
    X_d_sparse = X_domain[:, domain_idx]
    X_e_sparse = X_env[:, env_idx]

    # Split
    X_d_train_raw = X_d_sparse[train_mask]
    X_d_test_raw = X_d_sparse[test_mask]
    X_e_train_raw = X_e_sparse[train_mask]
    X_e_test_raw = X_e_sparse[test_mask]

    # Standardize using train statistics
    X_d_train, X_d_test = standardize(X_d_train_raw, X_d_test_raw)
    X_e_train, X_e_test = standardize(X_e_train_raw, X_e_test_raw)

    # Also prepare full feature sets for linear CCA
    X_d_train_full, X_d_test_full = standardize(
        X_domain[train_mask], X_domain[test_mask])
    X_e_train_full, X_e_test_full = standardize(
        X_env[train_mask], X_env[test_mask])

    print(f"\nTrain shape: domain {X_d_train.shape}, env {X_e_train.shape}")
    print(f"Test shape: domain {X_d_test.shape}, env {X_e_test.shape}")

    # 5. Linear sparse CCA
    print(f"\n--- Linear Sparse CCA ---")
    U, V, lin_train_corrs = sparse_cca(
        X_d_train_full, X_e_train_full, c_u_opt, c_v_opt, N_COMPONENTS)

    lin_test_corrs = []
    for k in range(N_COMPONENTS):
        Xu = X_d_test_full @ U[:, k]
        Yv = X_e_test_full @ V[:, k]
        if np.std(Xu) > 1e-10 and np.std(Yv) > 1e-10:
            c = np.corrcoef(Xu, Yv)[0, 1]
        else:
            c = 0.0
        lin_test_corrs.append(c)

    lin_test_corrs = np.array(lin_test_corrs)
    lin_train_corrs = np.array(lin_train_corrs)

    print(f"  Train correlations: {lin_train_corrs}")
    print(f"  Test correlations: {lin_test_corrs}")

    # 6. KAN-CCA (multiple seeds)
    results = []

    # Record linear results (seed=0 convention)
    for k in range(N_COMPONENTS):
        results.append({
            'method': 'linear_sparse_cca',
            'seed': 0,
            'component': k + 1,
            'train_corr': lin_train_corrs[k],
            'test_corr': lin_test_corrs[k],
            'n_train': n_train,
            'n_test': n_test,
            'train_years': str(sorted(train_years)),
            'test_years': str(sorted(test_years)),
            'n_domain_features': len(domain_idx),
            'n_env_features': len(env_idx),
        })

    for seed in range(N_SEEDS):
        print(f"\n--- KAN-CCA (seed={seed}) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model, history = train_kan_cca(
            X_d_train, X_e_train,
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

        kan_train_corrs = evaluate_kan_cca(model, X_d_train, X_e_train, device)
        kan_test_corrs = evaluate_kan_cca(model, X_d_test, X_e_test, device)

        print(f"  Train correlations: {kan_train_corrs}")
        print(f"  Test correlations: {kan_test_corrs}")
        print(f"  Epochs trained: {len(history)}")

        for k in range(N_COMPONENTS):
            results.append({
                'method': 'kan_cca',
                'seed': seed,
                'component': k + 1,
                'train_corr': kan_train_corrs[k],
                'test_corr': kan_test_corrs[k],
                'n_train': n_train,
                'n_test': n_test,
                'train_years': str(sorted(train_years)),
                'test_years': str(sorted(test_years)),
                'n_domain_features': len(domain_idx),
                'n_env_features': len(env_idx),
            })

    # 7. Save results
    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / f'temporal_holdout_{ts}.tsv'
    results_df.to_csv(results_path, sep='\t', index=False)
    print(f"\nSaved: {results_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"TEMPORAL HOLDOUT SUMMARY")
    print(f"{'='*60}")
    print(f"Train: {n_train} samples ({sorted(train_years)})")
    print(f"Test: {n_test} samples ({sorted(test_years)})")
    print()

    for k in range(N_COMPONENTS):
        lin_test = lin_test_corrs[k]
        kan_rows = results_df[(results_df['method'] == 'kan_cca') &
                              (results_df['component'] == k + 1)]
        kan_mean = kan_rows['test_corr'].mean()
        kan_std = kan_rows['test_corr'].std()
        delta = kan_mean - lin_test
        pct = delta / abs(lin_test) * 100 if abs(lin_test) > 1e-6 else 0

        print(f"Component {k+1}:")
        print(f"  Linear test: {lin_test:.4f}")
        print(f"  KAN test:    {kan_mean:.4f} ± {kan_std:.4f} (n={N_SEEDS} seeds)")
        print(f"  Delta:       {delta:+.4f} ({pct:+.1f}%)")
        print()

    print(f"Timestamp: {ts}")

if __name__ == '__main__':
    main()
