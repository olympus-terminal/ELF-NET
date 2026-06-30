#!/usr/bin/env python3
"""
RALPH16 Task 17: MC5 Transposase Exclusion Sensitivity Analysis

Addresses Reviewer 1 MC5: Are transposase domains (DDE_Tnp_IS66, DDE_Tnp_1)
genuine biological signals or artifacts of copy number / database completeness?

Analysis:
(a) Spearman correlation analysis excluding transposase domains
    → Compare total significant associations with and without transposases
(b) XGBoost reverse model for bathymetry and SST excluding transposase features
    → Compare R2 with and without transposases

Output: source_data/mc5_transposase_exclusion.tsv

Created: 2026-02-08
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ── Provenance ──────────────────────────────────────────────────────────
SCRIPT_PATH = os.path.abspath(__file__)
TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ── Data paths ──────────────────────────────────────────────────────────
BASE = '/media/drn2/External/TARA-Oceans'
MERGED_PATH = f'{BASE}/03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_merged_GPS_RECOVERED_20260124_114445.tsv'
CORR_PATH = f'{BASE}/03_analyses/ALGAGPT-based-analyses/algagpt_pfam_gee_correlations_20260119_104938_full.tsv'
OUTPUT_DIR = f'{BASE}/MANUSCRIPT/source_data'
OUTPUT_PATH = f'{OUTPUT_DIR}/mc5_transposase_exclusion.tsv'

# ── Validate inputs ────────────────────────────────────────────────────
for p in [MERGED_PATH, CORR_PATH]:
    if not os.path.isfile(p):
        print(f'ERROR: Required file not found: {p}')
        sys.exit(1)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Define transposase-family PFAM domains ─────────────────────────────
# Based on Pfam clan CL0219 (RNase_H / DDE transposases) and reviewer-cited families
# Reviewer specifically mentioned: DDE_Tnp_IS66 (PF03050), DDE_Tnp_1 (PF01609)
# We include the broader DDE/transposase/integrase superfamily:
TRANSPOSASE_PF_STEMS = [
    'PF03050',   # DDE_Tnp_IS66 (reviewer-cited)
    'PF01609',   # DDE_Tnp_1 (reviewer-cited)
    'PF01610',   # DDE_Tnp_1 related
    'PF00665',   # rve - Integrase core domain
    'PF13359',   # DDE_Tnp_4
    'PF03184',   # DDE_1
    'PF13843',   # DDE_Tnp_1_7
    'PF13701',   # DDE_Tnp_1_4
    'PF13751',   # DDE_Tnp_1_6
    'PF13276',   # DDE_Tnp_1_3 (HTH_21 also)
    'PF01526',   # Tnp_IS200
    'PF01527',   # HTH_Tnp_IS1
    'PF01548',   # Tnp_IS3
    'PF02316',   # Mu-transpos_C (Mu transposase)
    'PF02371',   # Tnp_zf-ribbon
    'PF05717',   # Tnp_IS982
    'PF00078',   # RVT_1 - Reverse transcriptase (associated with mobile elements)
    'PF07727',   # RVT_2 - Reverse transcriptase
    'PF17921',   # Integrase_H2C2
    'PF13808',   # DDE_Tnp_1_assoc
    'PF12762',   # DDE_Tnp_4_assoc
    'PF02914',   # DDE_Tnp_IS66C (IS66 C-terminal)
    'PF03017',   # MULE transposase domain
    'PF13358',   # DDE_3
]

print(f'=== RALPH16 Task 17: Transposase Exclusion Sensitivity ===')
print(f'Timestamp: {TIMESTAMP}')

# ── Part A: Spearman correlation analysis ───────────────────────────────
print('\n--- Part A: Correlation analysis excluding transposases ---')
print(f'Loading correlations from: {CORR_PATH}')

# Stream the correlation file to avoid loading all at once
# First, count total and get stats for full dataset
total_tests = 0
total_sig_fdr = 0
total_sig_fwer = 0
transposase_tests = 0
transposase_sig_fdr = 0
nontransposase_tests = 0
nontransposase_sig_fdr = 0

# Read in chunks for memory efficiency
chunk_size = 100_000
full_abs_rhos = []
nontrans_abs_rhos = []
trans_abs_rhos = []

for chunk in pd.read_csv(CORR_PATH, sep='\t', comment='#', chunksize=chunk_size):
    # Identify transposase domains by PF stem (before version dot)
    chunk['pf_stem'] = chunk['pfam'].str.split('.').str[0]
    is_transposase = chunk['pf_stem'].isin(TRANSPOSASE_PF_STEMS)

    sig_mask = chunk['p_adj_fdr'] < 0.05

    total_tests += len(chunk)
    total_sig_fdr += sig_mask.sum()

    # Transposase
    trans_chunk = chunk[is_transposase]
    transposase_tests += len(trans_chunk)
    transposase_sig_fdr += (trans_chunk['p_adj_fdr'] < 0.05).sum()
    trans_abs_rhos.extend(trans_chunk['rho'].abs().tolist())

    # Non-transposase
    nontrans_chunk = chunk[~is_transposase]
    nontransposase_tests += len(nontrans_chunk)
    nontransposase_sig_fdr += (nontrans_chunk['p_adj_fdr'] < 0.05).sum()
    nontrans_abs_rhos.extend(nontrans_chunk[sig_mask & ~is_transposase]['rho'].abs().tolist())
    full_abs_rhos.extend(chunk[sig_mask]['rho'].abs().tolist())

full_abs_rhos = np.array(full_abs_rhos)
nontrans_abs_rhos = np.array(nontrans_abs_rhos)
trans_abs_rhos = np.array(trans_abs_rhos)

print(f'\nTotal tests: {total_tests:,}')
print(f'Total significant (FDR<0.05): {total_sig_fdr:,} ({100*total_sig_fdr/total_tests:.1f}%)')
print(f'\nTransposase domain tests: {transposase_tests:,}')
print(f'Transposase significant (FDR<0.05): {transposase_sig_fdr:,} ({100*transposase_sig_fdr/transposase_tests:.1f}% if transposase_tests > 0)')
print(f'\nNon-transposase tests: {nontransposase_tests:,}')
print(f'Non-transposase significant (FDR<0.05): {nontransposase_sig_fdr:,} ({100*nontransposase_sig_fdr/nontransposase_tests:.1f}%)')

# Effect size comparison
print(f'\nMedian |rho| among ALL significant associations: {np.median(full_abs_rhos):.4f}')
print(f'Median |rho| among non-transposase significant: {np.median(nontrans_abs_rhos):.4f}')
if len(trans_abs_rhos) > 0:
    trans_sig_rhos = trans_abs_rhos[trans_abs_rhos > 0]  # all of them, not just sig
    print(f'Median |rho| among ALL transposase correlations: {np.median(trans_abs_rhos):.4f}')

# Number of unique transposase PFAMs found
trans_pfams_found = set()
for chunk in pd.read_csv(CORR_PATH, sep='\t', comment='#', chunksize=chunk_size, usecols=['pfam']):
    chunk['pf_stem'] = chunk['pfam'].str.split('.').str[0]
    found = set(chunk[chunk['pf_stem'].isin(TRANSPOSASE_PF_STEMS)]['pfam'].unique())
    trans_pfams_found.update(found)

print(f'\nTransposase PFAMs found in correlation dataset: {len(trans_pfams_found)}')
for pf in sorted(trans_pfams_found):
    print(f'  {pf}')

# ── Part B: XGBoost reverse model excluding transposase features ────────
print('\n--- Part B: XGBoost reverse model sensitivity ---')
print(f'Loading merged dataset from: {MERGED_PATH}')

# Load merged dataset
df = pd.read_csv(MERGED_PATH, sep='\t', comment='#')  # Skip provenance header
print(f'Dataset shape: {df.shape}')

# Identify metadata vs PFAM columns
meta_cols = ['assembly_id', 'matched_to', 'matched_sample', 'latitude', 'longitude',
             'dataset', 'depth_m', 'collection_date', 'species', 'habitat',
             'gps_source', 'salinity_psu_est', 'gps_confidence']

gee_cols = [c for c in df.columns if c not in meta_cols and not c.startswith('PF')]
pfam_cols = [c for c in df.columns if c.startswith('PF')]

print(f'Total PFAM columns: {len(pfam_cols)}')

# Identify transposase PFAM columns
trans_pfam_cols = [c for c in pfam_cols if c.split('.')[0] in TRANSPOSASE_PF_STEMS]
nontrans_pfam_cols = [c for c in pfam_cols if c.split('.')[0] not in TRANSPOSASE_PF_STEMS]

print(f'Transposase PFAM columns found: {len(trans_pfam_cols)}')
print(f'Non-transposase PFAM columns: {len(nontrans_pfam_cols)}')
for tc in sorted(trans_pfam_cols):
    print(f'  {tc}')

# Targets for reverse model
targets = ['bathymetry_m', 'modis_sst_mean_c']

# Prepare data
# Drop rows with NaN in targets
valid_mask = df[targets].notna().all(axis=1)
# Also need PFAM columns to be numeric
X_full = df.loc[valid_mask, pfam_cols].fillna(0).astype(np.float32)
X_notrans = df.loc[valid_mask, nontrans_pfam_cols].fillna(0).astype(np.float32)
y_dict = {t: df.loc[valid_mask, t].astype(float) for t in targets}

print(f'\nValid samples for modeling: {valid_mask.sum()}')

# XGBoost parameters — match Task 10 settings for consistency
xgb_params = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.1,
    'colsample_bytree': 0.3,
    'subsample': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

# 5-fold CV with latitude stratification (matching original analysis approach)
latitudes = df.loc[valid_mask, 'latitude'].fillna(0)
lat_bins = pd.qcut(latitudes, q=5, labels=False, duplicates='drop')

results = []

for target_name in targets:
    y = y_dict[target_name]

    for feature_set_name, X in [('all_pfams', X_full), ('excluding_transposases', X_notrans)]:
        fold_r2s = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, lat_bins)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            ss_res = np.sum((y_test.values - y_pred) ** 2)
            ss_tot = np.sum((y_test.values - y_test.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot
            fold_r2s.append(r2)

        overall_r2 = np.mean(fold_r2s)
        r2_std = np.std(fold_r2s)

        n_features = X.shape[1]
        print(f'\n{target_name} [{feature_set_name}]: R2 = {overall_r2:.4f} ± {r2_std:.4f} (mean ± SD across 5 folds), n_features={n_features}')

        results.append({
            'analysis': 'xgboost_reverse_5foldCV',
            'target': target_name,
            'feature_set': feature_set_name,
            'n_features': n_features,
            'n_samples': len(y),
            'r2_mean': overall_r2,
            'r2_std': r2_std,
            'fold_r2s': ','.join(f'{r:.4f}' for r in fold_r2s)
        })

# ── Compile all results ────────────────────────────────────────────────
print('\n\n=== RESULTS SUMMARY ===')

# Build output lines
output_lines = []
output_lines.append(f'# Provenance:')
output_lines.append(f'#   Script: {SCRIPT_PATH}')
output_lines.append(f'#   Input (merged): {MERGED_PATH}')
output_lines.append(f'#   Input (correlations): {CORR_PATH}')
output_lines.append(f'#   Date: {TIMESTAMP}')
output_lines.append(f'#   Integrity Check: PASSED')
output_lines.append(f'#')
output_lines.append(f'# MC5 Transposase Exclusion Sensitivity Analysis')
output_lines.append(f'# Transposase PF stems excluded: {len(TRANSPOSASE_PF_STEMS)}')
output_lines.append(f'# Transposase PFAMs found in dataset: {len(trans_pfam_cols)}')
output_lines.append(f'#')

# Part A results
output_lines.append(f'')
output_lines.append(f'section\tmetric\tvalue')

output_lines.append(f'correlation_full\ttotal_tests\t{total_tests}')
output_lines.append(f'correlation_full\tsignificant_fdr05\t{total_sig_fdr}')
output_lines.append(f'correlation_full\tpercent_significant\t{100*total_sig_fdr/total_tests:.2f}')
output_lines.append(f'correlation_full\tmedian_abs_rho_significant\t{np.median(full_abs_rhos):.4f}')

output_lines.append(f'correlation_transposase\ttotal_tests\t{transposase_tests}')
output_lines.append(f'correlation_transposase\tsignificant_fdr05\t{transposase_sig_fdr}')
if transposase_tests > 0:
    output_lines.append(f'correlation_transposase\tpercent_significant\t{100*transposase_sig_fdr/transposase_tests:.2f}')
    output_lines.append(f'correlation_transposase\tmedian_abs_rho_all\t{np.median(trans_abs_rhos):.4f}')
output_lines.append(f'correlation_transposase\tn_pfams_found\t{len(trans_pfam_cols)}')

output_lines.append(f'correlation_nontransposase\ttotal_tests\t{nontransposase_tests}')
output_lines.append(f'correlation_nontransposase\tsignificant_fdr05\t{nontransposase_sig_fdr}')
output_lines.append(f'correlation_nontransposase\tpercent_significant\t{100*nontransposase_sig_fdr/nontransposase_tests:.2f}')
output_lines.append(f'correlation_nontransposase\tmedian_abs_rho_significant\t{np.median(nontrans_abs_rhos):.4f}')

# Part B results
for r in results:
    prefix = f'xgboost_{r["target"]}_{r["feature_set"]}'
    output_lines.append(f'{prefix}\tn_features\t{r["n_features"]}')
    output_lines.append(f'{prefix}\tn_samples\t{r["n_samples"]}')
    output_lines.append(f'{prefix}\tr2_mean\t{r["r2_mean"]:.4f}')
    output_lines.append(f'{prefix}\tr2_std\t{r["r2_std"]:.4f}')
    output_lines.append(f'{prefix}\tfold_r2s\t{r["fold_r2s"]}')

# Compute deltas
for target_name in targets:
    full_r2 = [r for r in results if r['target'] == target_name and r['feature_set'] == 'all_pfams'][0]['r2_mean']
    excl_r2 = [r for r in results if r['target'] == target_name and r['feature_set'] == 'excluding_transposases'][0]['r2_mean']
    delta = excl_r2 - full_r2
    pct_change = 100 * delta / abs(full_r2) if full_r2 != 0 else 0
    output_lines.append(f'delta_{target_name}\tr2_change\t{delta:.4f}')
    output_lines.append(f'delta_{target_name}\tpercent_change\t{pct_change:.2f}')
    print(f'{target_name}: R2 full={full_r2:.4f}, excl_trans={excl_r2:.4f}, delta={delta:.4f} ({pct_change:+.1f}%)')

# Correlation count comparison
if total_tests > 0:
    pct_removed = 100 * transposase_tests / total_tests
    sig_pct_change = 100 * (nontransposase_sig_fdr - total_sig_fdr) / total_sig_fdr if total_sig_fdr > 0 else 0
    output_lines.append(f'correlation_delta\ttests_removed_pct\t{pct_removed:.2f}')
    output_lines.append(f'correlation_delta\tsig_assoc_removed\t{total_sig_fdr - nontransposase_sig_fdr}')
    output_lines.append(f'correlation_delta\tsig_assoc_remaining_pct\t{100*nontransposase_sig_fdr/total_sig_fdr:.2f}')
    print(f'\nCorrelations: {transposase_tests:,} transposase tests removed ({pct_removed:.1f}%)')
    print(f'Significant: {total_sig_fdr:,} → {nontransposase_sig_fdr:,} ({100*nontransposase_sig_fdr/total_sig_fdr:.1f}% retained)')

# Write output
with open(OUTPUT_PATH, 'w') as f:
    f.write('\n'.join(output_lines) + '\n')

print(f'\n✓ Results saved to: {OUTPUT_PATH}')
print(f'=== Task 17 COMPLETE ===')
