#!/usr/bin/env python3
"""
17_te_partial_correlations_dino.py — Partial Spearman correlations between
TE domains and SST, controlling for 5 Form II dinoflagellate lineage counts.

Compares zero-order Spearman ρ(TE, SST) to partial ρ(TE, SST | 5 dino lineages).
If partial correlations persist, TE–SST coupling is not fully explained by
dinoflagellate taxonomic sorting.

Method: Partial Spearman correlation via rank-then-regress approach.
For each pair (X=TE, Y=SST), controlling for Z=[5 dino lineages]:
  1. Rank-transform X, Y, Z
  2. Compute residuals: X_res = X_rank - Z @ (Z^T Z)^{-1} Z^T X_rank
  3. Compute residuals: Y_res = Y_rank - Z @ (Z^T Z)^{-1} Z^T Y_rank
  4. Partial Spearman ρ = Pearson(X_res, Y_res)

Inputs:
  - PFAM matrix: data/env_to_pfam_algagpt_20260125_190454.npz
  - RuBisCO counts: omen-work/rubisco_merged_formI_formII_20260223.tsv

Output:
  - source_data/te_partial_correlations_dino.tsv

Provenance: ralph34 task 4
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MANUSCRIPT = Path(__file__).resolve().parent.parent.parent
PFAM_NPZ = MANUSCRIPT / 'data' / 'env_to_pfam_algagpt_20260125_190454.npz'
RUBISCO_TSV = MANUSCRIPT / 'omen-work' / 'rubisco_merged_formI_formII_20260223.tsv'
OUTPUT = MANUSCRIPT / 'source_data' / 'te_partial_correlations_dino.tsv'

# ---------------------------------------------------------------------------
# Domain definitions
# ---------------------------------------------------------------------------
TE_DOMAINS = {
    'PF20209': 'DUF6570',
    'PF14214': 'Helitron_like_N',
    'PF00589': 'Phage_integrase',
    'PF01609': 'DDE_Tnp_1',
    'PF17921': 'Integrase_H2C2',
    'PF00665': 'rve',
    'PF13358': 'DDE_3',
    'PF17919': 'RT_RNaseH_2',
    'PF00075': 'RNase_H',
    'PF00078': 'RVT_1',
    'PF07727': 'RVT_2',
}

DINO_LINEAGES = [
    'symbiodiniaceae', 'peridiniales', 'gonyaulacales',
    'prorocentrales', 'chromerida',
]

# Environmental variables to test
ENV_TARGETS = ['sst_mean_c', 'bathymetry_m', 'air_temp_mean_c', 'chl_mean_mg_m3']

def partial_spearman(x, y, z):
    """
    Compute partial Spearman correlation between x and y controlling for z.

    Uses rank-then-partial approach:
    1. Rank-transform x, y, and each column of z
    2. Regress ranks of x and y on ranks of z
    3. Correlate residuals

    Parameters:
        x: array (n,) — variable 1
        y: array (n,) — variable 2
        z: array (n, k) — covariates

    Returns:
        rho_partial: float
        p_value: float
    """
    n = len(x)
    k = z.shape[1] if z.ndim > 1 else 1

    # Rank transform
    x_rank = stats.rankdata(x)
    y_rank = stats.rankdata(y)
    if z.ndim == 1:
        z_rank = stats.rankdata(z).reshape(-1, 1)
    else:
        z_rank = np.column_stack([stats.rankdata(z[:, j]) for j in range(z.shape[1])])

    # Add intercept
    Z = np.column_stack([np.ones(n), z_rank])

    # OLS residuals
    try:
        beta_x = np.linalg.lstsq(Z, x_rank, rcond=None)[0]
        beta_y = np.linalg.lstsq(Z, y_rank, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    x_res = x_rank - Z @ beta_x
    y_res = y_rank - Z @ beta_y

    # Partial correlation = Pearson of residuals
    rho, p = stats.pearsonr(x_res, y_res)

    # Adjust p-value for degrees of freedom
    # df = n - k - 2 (controlling for k covariates)
    df = n - k - 2
    if df <= 0:
        return rho, np.nan

    t_stat = rho * np.sqrt(df / (1 - rho ** 2 + 1e-12))
    p_adjusted = 2 * stats.t.sf(abs(t_stat), df)

    return rho, p_adjusted

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{ts}] 17_te_partial_correlations_dino.py starting")

    # 1. Load PFAM + ENV data
    data = np.load(PFAM_NPZ, allow_pickle=True)
    pfam_cols = data['output_feature_names']
    env_cols = data['input_feature_names']
    all_pfam = np.vstack([data['y_train'], data['y_val'], data['y_test']])
    all_env = np.vstack([data['X_train'], data['X_val'], data['X_test']])
    all_ids = np.concatenate([data['train_ids'], data['val_ids'], data['test_ids']])
    print(f"PFAM: {all_pfam.shape}, ENV: {all_env.shape}")

    # Map PFAM accession → index
    acc_to_idx = {}
    for i, col in enumerate(pfam_cols):
        acc = str(col).split('.')[0]
        if acc in TE_DOMAINS:
            acc_to_idx[acc] = i

    # Map env name → index
    env_to_idx = {str(name): i for i, name in enumerate(env_cols)}

    # Build sample ID map
    pfam_id_map = {str(sid): i for i, sid in enumerate(all_ids)}

    # 2. Load RuBisCO lineage data
    rub = pd.read_csv(RUBISCO_TSV, sep='\t', comment='#')
    rub['sample_id_norm'] = rub['sample_id'].str.replace(r'\.fa\.aa$', '', regex=True)
    rub['sample_id_norm'] = rub['sample_id_norm'].str.replace(r'\.aa$', '', regex=True)

    # 3. Match samples (need: PFAM row, ENV row, RuBisCO row all present)
    matched_indices = []
    matched_rub_rows = []
    for _, row in rub.iterrows():
        norm_id = row['sample_id_norm']
        if norm_id in pfam_id_map:
            idx = pfam_id_map[norm_id]
            matched_indices.append(idx)
            matched_rub_rows.append(row)

    n_matched = len(matched_indices)
    print(f"Matched samples: {n_matched}")

    pfam_matched = all_pfam[matched_indices]
    env_matched = all_env[matched_indices]
    rub_matched = pd.DataFrame(matched_rub_rows).reset_index(drop=True)

    # 4. Filter to samples with valid SST (non-NaN)
    sst_idx = env_to_idx['sst_mean_c']
    valid_sst = ~np.isnan(env_matched[:, sst_idx]) & (env_matched[:, sst_idx] != 0)
    print(f"Samples with valid SST: {valid_sst.sum()}")

    # 5. Build dino lineage covariate matrix
    dino_z = np.column_stack([
        rub_matched[lin].values.astype(float) for lin in DINO_LINEAGES
    ])

    # 6. Compute zero-order and partial correlations
    results = []
    for env_name in ENV_TARGETS:
        if env_name not in env_to_idx:
            print(f"WARNING: {env_name} not found in ENV features")
            continue

        env_idx = env_to_idx[env_name]
        env_vals = env_matched[:, env_idx]

        # Valid mask (non-NaN env)
        valid = ~np.isnan(env_vals) & np.isfinite(env_vals)
        n_valid = valid.sum()
        if n_valid < 50:
            print(f"  {env_name}: too few valid samples ({n_valid})")
            continue

        print(f"\n--- {env_name} (n={n_valid}) ---")

        for acc in sorted(TE_DOMAINS.keys()):
            te_vals = pfam_matched[:, acc_to_idx[acc]]

            # Zero-order Spearman
            rho_zero, p_zero = stats.spearmanr(te_vals[valid], env_vals[valid])

            # Partial Spearman controlling for 5 dino lineages
            rho_partial, p_partial = partial_spearman(
                te_vals[valid], env_vals[valid], dino_z[valid]
            )

            # Change in correlation
            delta_rho = rho_partial - rho_zero
            pct_retained = (rho_partial / rho_zero * 100) if rho_zero != 0 else np.nan

            results.append({
                'te_domain': acc,
                'te_name': TE_DOMAINS[acc],
                'env_variable': env_name,
                'n_samples': n_valid,
                'rho_zero_order': rho_zero,
                'p_zero_order': p_zero,
                'rho_partial_5dino': rho_partial,
                'p_partial_5dino': p_partial,
                'delta_rho': delta_rho,
                'pct_retained': pct_retained,
            })

            sign = '✓' if (rho_partial != 0 and np.sign(rho_partial) == np.sign(rho_zero)) else '✗'
            sig = '*' if (p_partial is not None and not np.isnan(p_partial) and p_partial < 0.05) else ''
            print(f"  {acc} ({TE_DOMAINS[acc]:20s}): "
                  f"ρ₀ = {rho_zero:+.3f} → ρ_partial = {rho_partial:+.3f}{sig} "
                  f"({pct_retained:.0f}% retained) {sign}")

    results_df = pd.DataFrame(results)

    # 7. Summary
    print(f"\n{'='*80}")
    print("PARTIAL CORRELATION SUMMARY")
    print(f"{'='*80}")

    # Focus on SST
    sst_results = results_df[results_df['env_variable'] == 'sst_mean_c']
    if len(sst_results) > 0:
        n_sig_zero = (sst_results['p_zero_order'] < 0.05).sum()
        n_sig_partial = (sst_results['p_partial_5dino'] < 0.05).sum()
        n_sign_preserved = (
            np.sign(sst_results['rho_zero_order']) == np.sign(sst_results['rho_partial_5dino'])
        ).sum()
        mean_pct = sst_results['pct_retained'].mean()

        print(f"\nSST correlations:")
        print(f"  Zero-order significant: {n_sig_zero}/11")
        print(f"  Partial significant: {n_sig_partial}/11")
        print(f"  Sign preserved: {n_sign_preserved}/11")
        print(f"  Mean % retained: {mean_pct:.1f}%")

    # Bonferroni correction across all tests
    n_tests = len(results_df)
    results_df['p_bonferroni'] = np.minimum(results_df['p_partial_5dino'] * n_tests, 1.0)
    results_df['significant_bonf'] = results_df['p_bonferroni'] < 0.05

    n_bonf = results_df['significant_bonf'].sum()
    print(f"\nAll env variables:")
    print(f"  Total tests: {n_tests}")
    print(f"  Significant after Bonferroni: {n_bonf}")

    # 8. Save
    header = (
        f"# Provenance:\n"
        f"#   Script: scripts/kan_cca/17_te_partial_correlations_dino.py\n"
        f"#   PFAM source: data/env_to_pfam_algagpt_20260125_190454.npz\n"
        f"#   RuBisCO source: omen-work/rubisco_merged_formI_formII_20260223.tsv\n"
        f"#   Date: {datetime.now().isoformat()}\n"
        f"#   Matched samples: {n_matched}\n"
        f"#   Covariates: {', '.join(DINO_LINEAGES)}\n"
        f"#   Method: Partial Spearman (rank-then-regress)\n"
        f"#\n"
    )
    with open(OUTPUT, 'w') as f:
        f.write(header)
        results_df.to_csv(f, sep='\t', index=False)

    print(f"\nResults saved to: {OUTPUT}")
    print("Done.")

if __name__ == '__main__':
    main()
