#!/usr/bin/env python3
"""
15_te_rubisco_covariation.py — Compute TE–RuBisCO lineage covariation.

For each of the 11 top TE domains (from Table S2 forward model) and each
of the 15 RuBisCO lineage counts, compute Spearman ρ across matched samples.

Inputs:
  - PFAM matrix: data/env_to_pfam_algagpt_20260125_190454.npz
  - RuBisCO counts: omen-work/rubisco_merged_formI_formII_20260223.tsv

Output:
  - source_data/te_rubisco_covariation.tsv

Provenance: ralph34 task 2
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
OUTPUT = MANUSCRIPT / 'source_data' / 'te_rubisco_covariation.tsv'

# ---------------------------------------------------------------------------
# 11 TE domains from Table S2 (18 most environment-predictable)
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

# RuBisCO lineage columns (all 15 from merged file)
RUBISCO_LINEAGES = [
    'formI_green', 'formI_red', 'formII',
    'mamiellophyceae', 'prasinophyceae', 'pyramimonadales',
    'chlorellaceae', 'trebouxiophyceae', 'scenedesmaceae',
    'pelagophyceae', 'bolidophyceae', 'haptophyta', 'cryptophyta',
    'symbiodiniaceae', 'peridiniales', 'gonyaulacales',
    'prorocentrales', 'chromerida',
]

# Form II dinoflagellate lineages specifically
DINO_LINEAGES = [
    'symbiodiniaceae', 'peridiniales', 'gonyaulacales',
    'prorocentrales', 'chromerida',
]

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{ts}] 15_te_rubisco_covariation.py starting")

    # 1. Load PFAM data
    print(f"Loading PFAM: {PFAM_NPZ}")
    data = np.load(PFAM_NPZ, allow_pickle=True)
    pfam_cols = data['output_feature_names']
    all_pfam = np.vstack([data['y_train'], data['y_val'], data['y_test']])
    all_ids = np.concatenate([data['train_ids'], data['val_ids'], data['test_ids']])
    print(f"  PFAM matrix: {all_pfam.shape[0]} samples × {all_pfam.shape[1]} domains")

    # Map versioned PFAM column names to accession
    acc_to_idx = {}
    for i, col in enumerate(pfam_cols):
        acc = str(col).split('.')[0]
        if acc in TE_DOMAINS:
            acc_to_idx[acc] = i

    found = sorted(acc_to_idx.keys())
    missing = sorted(set(TE_DOMAINS.keys()) - set(found))
    print(f"  Found {len(found)}/11 TE domains in PFAM matrix")
    if missing:
        print(f"  Missing: {missing}")
        sys.exit(1)

    # Build PFAM sample ID → row index map
    pfam_id_map = {str(sid): i for i, sid in enumerate(all_ids)}

    # 2. Load RuBisCO data
    print(f"Loading RuBisCO: {RUBISCO_TSV}")
    rub = pd.read_csv(RUBISCO_TSV, sep='\t', comment='#')
    print(f"  RuBisCO: {len(rub)} samples × {len(rub.columns)} columns")

    # Normalize sample IDs: strip .fa.aa / .aa suffix
    rub['sample_id_norm'] = rub['sample_id'].str.replace(r'\.fa\.aa$', '', regex=True)
    rub['sample_id_norm'] = rub['sample_id_norm'].str.replace(r'\.aa$', '', regex=True)

    # 3. Match samples
    matched_pfam_rows = []
    matched_rub_rows = []
    for _, row in rub.iterrows():
        norm_id = row['sample_id_norm']
        if norm_id in pfam_id_map:
            matched_pfam_rows.append(pfam_id_map[norm_id])
            matched_rub_rows.append(row)

    n_matched = len(matched_pfam_rows)
    print(f"  Matched samples: {n_matched}")
    if n_matched < 100:
        print("ERROR: Too few matched samples")
        sys.exit(1)

    # Extract matched TE abundances
    pfam_matched = all_pfam[matched_pfam_rows]
    rub_matched = pd.DataFrame(matched_rub_rows).reset_index(drop=True)

    # 4. Compute Spearman correlations: 11 TE domains × 18 RuBisCO lineages
    results = []
    for acc in sorted(TE_DOMAINS.keys()):
        te_vals = pfam_matched[:, acc_to_idx[acc]]
        for lineage in RUBISCO_LINEAGES:
            rub_vals = rub_matched[lineage].values.astype(float)

            # Spearman correlation
            rho, pval = stats.spearmanr(te_vals, rub_vals)

            results.append({
                'te_domain': acc,
                'te_name': TE_DOMAINS[acc],
                'rubisco_lineage': lineage,
                'is_dino_lineage': lineage in DINO_LINEAGES,
                'spearman_rho': rho,
                'p_value': pval,
                'n_samples': n_matched,
                'n_nonzero_lineage': int((rub_vals > 0).sum()),
            })

    results_df = pd.DataFrame(results)

    # Bonferroni correction
    n_tests = len(results_df)
    results_df['p_bonferroni'] = np.minimum(results_df['p_value'] * n_tests, 1.0)
    results_df['significant_bonf'] = results_df['p_bonferroni'] < 0.05

    # 5. Summary statistics
    print(f"\n{'='*80}")
    print("TE–RuBisCO SPEARMAN CORRELATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {n_tests} (11 TE × {len(RUBISCO_LINEAGES)} lineages)")
    print(f"Significant (Bonferroni): {results_df['significant_bonf'].sum()}")

    # High correlations (|ρ| > 0.5)
    high = results_df[results_df['spearman_rho'].abs() > 0.5]
    print(f"\nPairs with |ρ| > 0.5: {len(high)}")
    if len(high) > 0:
        for _, row in high.sort_values('spearman_rho', key=abs, ascending=False).iterrows():
            print(f"  {row['te_domain']} ({row['te_name']}) × {row['rubisco_lineage']}: "
                  f"ρ = {row['spearman_rho']:.3f}, p = {row['p_value']:.2e}")

    # Moderate correlations (|ρ| > 0.3)
    moderate = results_df[results_df['spearman_rho'].abs() > 0.3]
    print(f"\nPairs with |ρ| > 0.3: {len(moderate)}")

    # Dinoflagellate lineage correlations specifically
    dino_results = results_df[results_df['is_dino_lineage']]
    print(f"\n--- Dinoflagellate lineage correlations ---")
    print(f"Pairs with |ρ| > 0.3: {(dino_results['spearman_rho'].abs() > 0.3).sum()}")
    print(f"Pairs with |ρ| > 0.5: {(dino_results['spearman_rho'].abs() > 0.5).sum()}")

    # Mean |ρ| by lineage type
    for lineage in RUBISCO_LINEAGES:
        subset = results_df[results_df['rubisco_lineage'] == lineage]
        mean_abs_rho = subset['spearman_rho'].abs().mean()
        max_rho = subset.loc[subset['spearman_rho'].abs().idxmax()]
        print(f"  {lineage:20s}: mean|ρ| = {mean_abs_rho:.3f}, "
              f"max|ρ| = {abs(max_rho['spearman_rho']):.3f} ({max_rho['te_domain']})")

    # 6. Save results
    header = (
        f"# Provenance:\n"
        f"#   Script: scripts/kan_cca/15_te_rubisco_covariation.py\n"
        f"#   PFAM source: data/env_to_pfam_algagpt_20260125_190454.npz\n"
        f"#   RuBisCO source: omen-work/rubisco_merged_formI_formII_20260223.tsv\n"
        f"#   Date: {datetime.now().isoformat()}\n"
        f"#   Matched samples: {n_matched}\n"
        f"#   Tests: {n_tests}\n"
        f"#\n"
    )
    with open(OUTPUT, 'w') as f:
        f.write(header)
        results_df.to_csv(f, sep='\t', index=False)

    print(f"\nResults saved to: {OUTPUT}")
    print("Done.")

if __name__ == '__main__':
    main()
