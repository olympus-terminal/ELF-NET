#!/usr/bin/env python3
"""
16_te_variance_within_dino.py — Within-dinoflagellate TE variance analysis.

Among samples where any Form II RuBisCO was detected (dinoflagellate-positive),
compute coefficient of variation (CV) of each TE domain's abundance.
Compare TE domain CVs to non-TE domain CVs using Mann-Whitney test.

If TE domains show significantly higher within-dino CV, this suggests TE
variation beyond what taxonomic composition alone predicts.

Inputs:
  - PFAM matrix: data/env_to_pfam_algagpt_20260125_190454.npz
  - RuBisCO counts: omen-work/rubisco_merged_formI_formII_20260223.tsv

Output:
  - source_data/te_variance_within_dino.tsv

Provenance: ralph34 task 3
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
OUTPUT = MANUSCRIPT / 'source_data' / 'te_variance_within_dino.tsv'

# ---------------------------------------------------------------------------
# Domain definitions (same as task 2)
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

# 7 non-TE domains from Table S2 top 18
NON_TE_TABLE_S2 = {
    'PF05970': 'PIF1',
    'PF19028': 'TSP1_spondin',
    'PF13604': 'AAA_30',
    'PF13245': 'AAA_19',
    'PF11999': 'Ice_binding',
    'PF00063': 'Myosin_head',
    'PF00858': 'ASC',
}

def compute_cv(arr):
    """Compute coefficient of variation; returns NaN if mean == 0."""
    mean = np.mean(arr)
    if mean == 0:
        return np.nan
    return np.std(arr) / mean

def compute_mad(arr):
    """Compute median absolute deviation (robust spread for CLR data)."""
    return np.median(np.abs(arr - np.median(arr)))

def compute_iqr(arr):
    """Compute interquartile range."""
    return np.percentile(arr, 75) - np.percentile(arr, 25)

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{ts}] 16_te_variance_within_dino.py starting")

    # 1. Load PFAM data
    data = np.load(PFAM_NPZ, allow_pickle=True)
    pfam_cols = data['output_feature_names']
    all_pfam = np.vstack([data['y_train'], data['y_val'], data['y_test']])
    all_ids = np.concatenate([data['train_ids'], data['val_ids'], data['test_ids']])
    print(f"PFAM: {all_pfam.shape[0]} samples × {all_pfam.shape[1]} domains")

    # Map PFAM accession → column index
    acc_to_idx = {}
    for i, col in enumerate(pfam_cols):
        acc = str(col).split('.')[0]
        acc_to_idx[acc] = i

    # Verify TE domains exist
    for acc in TE_DOMAINS:
        if acc not in acc_to_idx:
            print(f"ERROR: {acc} not found in PFAM matrix")
            sys.exit(1)

    # Build PFAM sample ID → row index map
    pfam_id_map = {str(sid): i for i, sid in enumerate(all_ids)}

    # 2. Load RuBisCO data and identify Form II-positive samples
    rub = pd.read_csv(RUBISCO_TSV, sep='\t', comment='#')
    rub['sample_id_norm'] = rub['sample_id'].str.replace(r'\.fa\.aa$', '', regex=True)
    rub['sample_id_norm'] = rub['sample_id_norm'].str.replace(r'\.aa$', '', regex=True)

    # Form II positive = any Form II RuBisCO detected
    rub_formII = rub[rub['formII'] > 0].copy()
    print(f"Form II-positive samples in RuBisCO data: {len(rub_formII)}")

    # Match with PFAM matrix
    dino_pfam_rows = []
    for _, row in rub_formII.iterrows():
        norm_id = row['sample_id_norm']
        if norm_id in pfam_id_map:
            dino_pfam_rows.append(pfam_id_map[norm_id])

    n_dino = len(dino_pfam_rows)
    print(f"Form II-positive samples matched to PFAM: {n_dino}")

    # Also get all non-dino samples for comparison
    rub_no_formII = rub[rub['formII'] == 0].copy()
    nondino_pfam_rows = []
    for _, row in rub_no_formII.iterrows():
        norm_id = row['sample_id_norm']
        if norm_id in pfam_id_map:
            nondino_pfam_rows.append(pfam_id_map[norm_id])

    n_nondino = len(nondino_pfam_rows)
    print(f"Form II-negative samples matched to PFAM: {n_nondino}")

    # Extract subsets
    pfam_dino = all_pfam[dino_pfam_rows]
    pfam_nondino = all_pfam[nondino_pfam_rows]

    # 3. Compute CVs for TE domains within dino-positive samples
    all_domains = {**TE_DOMAINS, **NON_TE_TABLE_S2}
    results = []

    for acc, name in sorted(all_domains.items()):
        idx = acc_to_idx[acc]
        is_te = acc in TE_DOMAINS

        # Within dino-positive samples
        vals_dino = pfam_dino[:, idx]
        cv_dino = compute_cv(vals_dino)
        mean_dino = np.mean(vals_dino)
        std_dino = np.std(vals_dino)
        mad_dino = compute_mad(vals_dino)
        iqr_dino = compute_iqr(vals_dino)

        # Within dino-negative samples
        vals_nondino = pfam_nondino[:, idx]
        cv_nondino = compute_cv(vals_nondino)
        mean_nondino = np.mean(vals_nondino)
        std_nondino = np.std(vals_nondino)
        mad_nondino = compute_mad(vals_nondino)
        iqr_nondino = compute_iqr(vals_nondino)

        # All samples
        vals_all = all_pfam[:, idx]
        cv_all = compute_cv(vals_all)

        results.append({
            'pfam_accession': acc,
            'pfam_name': name,
            'is_te': is_te,
            'n_dino_positive': n_dino,
            'n_dino_negative': n_nondino,
            'mean_dino_pos': mean_dino,
            'std_dino_pos': std_dino,
            'cv_dino_pos': cv_dino,
            'mad_dino_pos': mad_dino,
            'iqr_dino_pos': iqr_dino,
            'mean_dino_neg': mean_nondino,
            'std_dino_neg': std_nondino,
            'cv_dino_neg': cv_nondino,
            'mad_dino_neg': mad_nondino,
            'iqr_dino_neg': iqr_nondino,
            'cv_all_samples': cv_all,
        })

    results_df = pd.DataFrame(results)

    # 4. Compare TE vs non-TE CVs within dino-positive samples
    te_cvs = results_df[results_df['is_te']]['cv_dino_pos'].dropna()
    non_te_cvs = results_df[~results_df['is_te']]['cv_dino_pos'].dropna()

    print(f"\n{'='*80}")
    print("WITHIN-DINOFLAGELLATE CV COMPARISON")
    print(f"{'='*80}")
    print(f"TE domains (n={len(te_cvs)}): median CV = {te_cvs.median():.3f}, "
          f"mean CV = {te_cvs.mean():.3f}")
    print(f"Non-TE domains (n={len(non_te_cvs)}): median CV = {non_te_cvs.median():.3f}, "
          f"mean CV = {non_te_cvs.mean():.3f}")

    if len(te_cvs) > 0 and len(non_te_cvs) > 0:
        u_stat, mw_p = stats.mannwhitneyu(te_cvs, non_te_cvs, alternative='two-sided')
        print(f"Mann-Whitney U = {u_stat:.1f}, p = {mw_p:.4f}")

        # Also one-sided: are TE CVs higher?
        _, mw_p_greater = stats.mannwhitneyu(te_cvs, non_te_cvs, alternative='greater')
        print(f"Mann-Whitney (TE > non-TE): p = {mw_p_greater:.4f}")
    else:
        mw_p = np.nan
        mw_p_greater = np.nan

    # 5. Also compare using ALL non-TE domains (not just Table S2)
    # Note: PFAM matrix contains CLR-transformed values (mean ~0), so CV is
    # unreliable. Use standard deviation as the primary variance metric.
    print(f"\n--- Broader comparison: SD within dino-positive samples (all domains) ---")
    all_sds = []
    for i in range(all_pfam.shape[1]):
        vals = pfam_dino[:, i]
        sd = np.std(vals)
        acc = str(pfam_cols[i]).split('.')[0]
        all_sds.append({
            'acc': acc,
            'sd': sd,
            'is_te': acc in TE_DOMAINS,
        })

    all_sds_df = pd.DataFrame(all_sds)
    te_broad = all_sds_df[all_sds_df['is_te']]['sd']
    non_te_broad = all_sds_df[~all_sds_df['is_te']]['sd']

    print(f"TE domains (n={len(te_broad)}): median SD = {te_broad.median():.3f}, "
          f"mean SD = {te_broad.mean():.3f}")
    print(f"All other domains (n={len(non_te_broad)}): median SD = {non_te_broad.median():.3f}, "
          f"mean SD = {non_te_broad.mean():.3f}")

    if len(te_broad) > 0 and len(non_te_broad) > 0:
        u_broad, p_broad = stats.mannwhitneyu(te_broad, non_te_broad, alternative='two-sided')
        _, p_broad_greater = stats.mannwhitneyu(te_broad, non_te_broad, alternative='greater')
        print(f"Mann-Whitney (TE vs all others, two-sided): U = {u_broad:.0f}, p = {p_broad:.4e}")
        print(f"Mann-Whitney (TE > all others, one-sided): p = {p_broad_greater:.4e}")

        # Percentile rank of TE domains by SD
        all_sd_sorted = sorted(all_sds_df['sd'])
        print(f"\nTE domain percentile ranks (by SD within dino-positive samples):")
        for _, row in results_df[results_df['is_te']].iterrows():
            sd = row['std_dino_pos']
            pct = np.searchsorted(all_sd_sorted, sd) / len(all_sd_sorted) * 100
            print(f"  {row['pfam_accession']} ({row['pfam_name']}): "
                  f"SD = {sd:.3f} (percentile {pct:.1f}%)")
    else:
        p_broad = np.nan
        p_broad_greater = np.nan

    # 6. Compare TE vs non-TE Table S2 using SD (more appropriate for CLR)
    te_sds = results_df[results_df['is_te']]['std_dino_pos']
    non_te_sds = results_df[~results_df['is_te']]['std_dino_pos']
    print(f"\n--- Table S2 comparison (SD) ---")
    print(f"TE domains (n={len(te_sds)}): median SD = {te_sds.median():.3f}, "
          f"mean SD = {te_sds.mean():.3f}")
    print(f"Non-TE domains (n={len(non_te_sds)}): median SD = {non_te_sds.median():.3f}, "
          f"mean SD = {non_te_sds.mean():.3f}")
    if len(te_sds) > 0 and len(non_te_sds) > 0:
        u_s2, p_s2 = stats.mannwhitneyu(te_sds, non_te_sds, alternative='two-sided')
        _, p_s2_greater = stats.mannwhitneyu(te_sds, non_te_sds, alternative='greater')
        print(f"Mann-Whitney (TE vs non-TE, two-sided): U = {u_s2:.1f}, p = {p_s2:.4f}")
        print(f"Mann-Whitney (TE > non-TE, one-sided): p = {p_s2_greater:.4f}")

    # 6. Per-domain details
    print(f"\n--- Per-domain CVs (Table S2 domains) ---")
    for _, row in results_df.sort_values('cv_dino_pos', ascending=False).iterrows():
        te_tag = "TE" if row['is_te'] else "non-TE"
        print(f"  {row['pfam_accession']:8s} ({row['pfam_name']:20s}) [{te_tag}]: "
              f"CV_dino={row['cv_dino_pos']:.3f}, CV_nondino={row['cv_dino_neg']:.3f}, "
              f"CV_all={row['cv_all_samples']:.3f}")

    # 7. Save results
    header = (
        f"# Provenance:\n"
        f"#   Script: scripts/kan_cca/16_te_variance_within_dino.py\n"
        f"#   PFAM source: data/env_to_pfam_algagpt_20260125_190454.npz\n"
        f"#   RuBisCO source: omen-work/rubisco_merged_formI_formII_20260223.tsv\n"
        f"#   Date: {datetime.now().isoformat()}\n"
        f"#   Form II-positive samples: {n_dino}\n"
        f"#   Form II-negative samples: {n_nondino}\n"
        f"#   Mann-Whitney (TE vs non-TE Table S2, two-sided): p = {mw_p:.4e}\n"
        f"#   Mann-Whitney (TE vs all other domains, two-sided): p = {p_broad:.4e}\n"
        f"#\n"
    )
    with open(OUTPUT, 'w') as f:
        f.write(header)
        results_df.to_csv(f, sep='\t', index=False)

    print(f"\nResults saved to: {OUTPUT}")
    print("Done.")

if __name__ == '__main__':
    main()
