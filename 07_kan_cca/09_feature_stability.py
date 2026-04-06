#!/usr/bin/env python3
"""
09_feature_stability.py — Feature stability analysis across CV folds.

For each fold, fit sparse CCA at optimal (c_u, c_v) on the training set.
Compute which features are selected (nonzero weights) per fold.
Report pairwise Jaccard similarity and consensus features.
"""

import json
import socket
import sys
from datetime import datetime
from pathlib import Path
from itertools import combinations

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

N_COMPONENTS = 3

def jaccard(set_a, set_b):
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{ts}] 09_feature_stability.py on {socket.gethostname()}")

    # Load data
    X_domain = np.load(OUTPUT_DIR / 'X_domain.npy')
    X_env = np.load(OUTPUT_DIR / 'X_env.npy')
    with open(OUTPUT_DIR / 'fold_indices.json') as f:
        folds = json.load(f)
    with open(OUTPUT_DIR / 'sparse_cca_selected_indices.json') as f:
        selected = json.load(f)
    with open(OUTPUT_DIR / 'feature_names.json') as f:
        feature_names = json.load(f)

    c_u_opt = selected['c_u_opt']
    c_v_opt = selected['c_v_opt']
    n_folds = len(folds)

    print(f"Data: {X_domain.shape[0]} samples, {X_domain.shape[1]} domains, {X_env.shape[1]} env")
    print(f"Optimal: c_u={c_u_opt}, c_v={c_v_opt}")

    # Per-fold feature selection
    fold_domain_sets = {k: [] for k in range(1, N_COMPONENTS + 1)}
    fold_env_sets = {k: [] for k in range(1, N_COMPONENTS + 1)}

    for fold_i, fold in enumerate(folds):
        train_idx = np.array(fold['train'])
        X_d_tr = X_domain[train_idx]
        X_e_tr = X_env[train_idx]

        U, V, corrs = sparse_cca(X_d_tr, X_e_tr, c_u_opt, c_v_opt, N_COMPONENTS)

        for k in range(N_COMPONENTS):
            dom_idx = set(np.where(np.abs(U[:, k]) > 1e-10)[0].tolist())
            env_idx = set(np.where(np.abs(V[:, k]) > 1e-10)[0].tolist())
            fold_domain_sets[k + 1].append(dom_idx)
            fold_env_sets[k + 1].append(env_idx)

        print(f"  Fold {fold_i}: domains=[{','.join(str(len(s)) for s in [fold_domain_sets[k+1][-1] for k in range(N_COMPONENTS)])}], "
              f"env=[{','.join(str(len(s)) for s in [fold_env_sets[k+1][-1] for k in range(N_COMPONENTS)])}]")

    # Pairwise Jaccard similarity
    print(f"\n=== Pairwise Jaccard Similarity ===")
    for comp in range(1, N_COMPONENTS + 1):
        dom_jaccards = []
        env_jaccards = []
        for i, j in combinations(range(n_folds), 2):
            dom_jaccards.append(jaccard(fold_domain_sets[comp][i], fold_domain_sets[comp][j]))
            env_jaccards.append(jaccard(fold_env_sets[comp][i], fold_env_sets[comp][j]))

        print(f"Component {comp}:")
        print(f"  Domain Jaccard: {np.mean(dom_jaccards):.4f} ± {np.std(dom_jaccards):.4f} "
              f"(range [{np.min(dom_jaccards):.4f}, {np.max(dom_jaccards):.4f}])")
        print(f"  Env Jaccard:    {np.mean(env_jaccards):.4f} ± {np.std(env_jaccards):.4f} "
              f"(range [{np.min(env_jaccards):.4f}, {np.max(env_jaccards):.4f}])")

    # Consensus features (present in >=7/10 folds)
    print(f"\n=== Consensus Features (>=7/10 folds) ===")
    for comp in range(1, N_COMPONENTS + 1):
        # Count frequency of each domain feature
        all_dom = {}
        for s in fold_domain_sets[comp]:
            for idx in s:
                all_dom[idx] = all_dom.get(idx, 0) + 1

        all_env = {}
        for s in fold_env_sets[comp]:
            for idx in s:
                all_env[idx] = all_env.get(idx, 0) + 1

        consensus_dom = {idx: cnt for idx, cnt in all_dom.items() if cnt >= 7}
        consensus_env = {idx: cnt for idx, cnt in all_env.items() if cnt >= 7}

        print(f"Component {comp}:")
        print(f"  Domain consensus ({len(consensus_dom)}):")
        for idx, cnt in sorted(consensus_dom.items(), key=lambda x: -x[1]):
            name = feature_names['domain_cols'][idx] if idx < len(feature_names['domain_cols']) else f'idx_{idx}'
            print(f"    {name}: {cnt}/10 folds")

        print(f"  Env consensus ({len(consensus_env)}):")
        for idx, cnt in sorted(consensus_env.items(), key=lambda x: -x[1]):
            name = feature_names['env_cols'][idx] if idx < len(feature_names['env_cols']) else f'idx_{idx}'
            print(f"    {name}: {cnt}/10 folds")

    # Most stable features (>=8/10 folds)
    print(f"\n=== Most Stable Features (>=8/10 folds) ===")
    for comp in range(1, N_COMPONENTS + 1):
        all_dom = {}
        for s in fold_domain_sets[comp]:
            for idx in s:
                all_dom[idx] = all_dom.get(idx, 0) + 1

        stable_dom = {idx: cnt for idx, cnt in all_dom.items() if cnt >= 8}
        print(f"Component {comp}: {len(stable_dom)} stable domain features")
        for idx, cnt in sorted(stable_dom.items(), key=lambda x: -x[1]):
            name = feature_names['domain_cols'][idx]
            print(f"    {name}: {cnt}/10 folds")

    print(f"\nTimestamp: {ts}")

if __name__ == '__main__':
    main()
