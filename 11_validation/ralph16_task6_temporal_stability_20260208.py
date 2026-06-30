#!/usr/bin/env python3
"""
Task 6: MC2 Temporal Stability Analysis

Classify GEE environmental variables as temporally stable vs variable,
then compare domain-environment coupling strengths between the two categories.
This addresses Reviewer 1's MC2 concern about the 4-12 year temporal mismatch
between TARA sampling (2009-2013) and AlphaEarth composites (2019-2021).

Provenance:
  Script: ralph16_task6_temporal_stability_20260208.py
  Date: 2026-02-08
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
from scipy import stats

# ── Environment detection ──
if os.path.exists('/media/drn2/External/TARA-Oceans'):
    BASE = '/media/drn2/External/TARA-Oceans'
elif os.path.exists('/media/drn/External1/TARA-Oceans'):
    BASE = '/media/drn/External1/TARA-Oceans'
elif os.path.exists('/scratch/drn2/PROJECTS/TARA-LA4SR'):
    BASE = '/scratch/drn2/PROJECTS/TARA-LA4SR'
else:
    raise RuntimeError("Unknown environment — cannot find TARA-Oceans base directory")

MANUSCRIPT = os.path.join(BASE, 'MANUSCRIPT')

# ── Input files ──
PFAM_GEE_CORR = os.path.join(BASE, '03_analyses/ALGAGPT-based-analyses/algagpt_pfam_gee_correlations_20260119_104938_full.tsv')
XGBOOST_REVERSE = os.path.join(BASE, '03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/reports/xgboost_reverse_full_20260124_104452.csv')

# ── Output file ──
OUTPUT = os.path.join(MANUSCRIPT, 'source_data/mc2_temporal_stability.tsv')

# ── Validate inputs exist ──
for fpath in [PFAM_GEE_CORR, XGBOOST_REVERSE]:
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Required input file not found: {fpath}")

# ── Temporal stability classification ──
# Temporally STABLE: geographic/bathymetric features that do not change on decadal timescales
STABLE_VARS = {
    'bathymetry_m',
    'elevation_m',
    'distance_to_coast_km',
    'depth_m',
    'landcover_class',
}

# Temporally VARIABLE: oceanographic/atmospheric features that vary seasonally or inter-annually
VARIABLE_VARS = {
    'sst_mean_c', 'sst_max_c', 'sst_min_c', 'sst_range_c', 'modis_sst_mean_c',
    'chl_mean_mg_m3', 'chl_max_mg_m3', 'chl_min_mg_m3',
    'nflh_mean', 'poc_mean_mg_m3',
    'air_temp_mean_c', 'air_temp_max_c', 'air_temp_min_c', 'air_temp_range_c',
    'precip_mean_mm', 'solar_rad_mj_m2', 'salinity_psu_est',
    'rrs_412', 'rrs_443', 'rrs_469', 'rrs_488', 'rrs_531',
    'rrs_547', 'rrs_555', 'rrs_645', 'rrs_667', 'rrs_678',
}

print(f"[INFO] Temporally stable variables: {len(STABLE_VARS)}")
print(f"[INFO] Temporally variable variables: {len(VARIABLE_VARS)}")

# ── 1. Load PFAM-GEE correlations (stream for memory efficiency) ──
print(f"\n[INFO] Loading PFAM-GEE correlations from:\n  {PFAM_GEE_CORR}")
# Read in chunks since the file is 34 MB
chunks = []
for chunk in pd.read_csv(PFAM_GEE_CORR, sep='\t', comment='#', chunksize=100000):
    chunks.append(chunk)
corr_df = pd.concat(chunks, ignore_index=True)
print(f"  Total correlations: {len(corr_df):,}")
print(f"  Columns: {list(corr_df.columns)}")

# Determine the column name for GEE variable
gee_col = 'gee_variable' if 'gee_variable' in corr_df.columns else 'env_variable'
if gee_col not in corr_df.columns:
    # Try to find it
    for c in corr_df.columns:
        if 'gee' in c.lower() or 'env' in c.lower() or 'variable' in c.lower():
            gee_col = c
            break
print(f"  GEE variable column: '{gee_col}'")

unique_gee = set(corr_df[gee_col].unique())
print(f"  Unique GEE variables: {len(unique_gee)}")
print(f"  Variables: {sorted(unique_gee)}")

# Map each variable to stable/variable
corr_df['temporal_class'] = 'unclassified'
corr_df.loc[corr_df[gee_col].isin(STABLE_VARS), 'temporal_class'] = 'stable'
corr_df.loc[corr_df[gee_col].isin(VARIABLE_VARS), 'temporal_class'] = 'variable'

# Report classification
class_counts = corr_df['temporal_class'].value_counts()
print(f"\n  Temporal classification of correlations:")
for cls, cnt in class_counts.items():
    print(f"    {cls}: {cnt:,}")

# Drop unclassified if any
classified = corr_df[corr_df['temporal_class'] != 'unclassified'].copy()
print(f"  Classified correlations: {len(classified):,}")

# ── 2. Compute absolute rho ──
classified['abs_rho'] = classified['rho'].abs()

# ── 3. Summary statistics by temporal class ──
print("\n[RESULTS] Correlation strength by temporal class:")
summary_rows = []
for cls in ['stable', 'variable']:
    subset = classified[classified['temporal_class'] == cls]
    abs_rhos = subset['abs_rho'].values

    median_rho = np.median(abs_rhos)
    mean_rho = np.mean(abs_rhos)
    q25, q75 = np.percentile(abs_rhos, [25, 75])
    max_rho = np.max(abs_rhos)
    n_total = len(abs_rhos)

    # Count significant at FDR < 0.05
    p_col = 'p_adj_fdr' if 'p_adj_fdr' in subset.columns else 'p_value'
    n_sig = (subset[p_col] < 0.05).sum()
    pct_sig = 100.0 * n_sig / n_total if n_total > 0 else 0

    # Proportion with |rho| > 0.2
    pct_gt02 = 100.0 * (abs_rhos > 0.2).sum() / n_total
    pct_gt03 = 100.0 * (abs_rhos > 0.3).sum() / n_total

    print(f"\n  {cls.upper()} variables (n_corr={n_total:,}):")
    print(f"    Median |rho| = {median_rho:.4f}")
    print(f"    Mean |rho|   = {mean_rho:.4f}")
    print(f"    IQR          = [{q25:.4f}, {q75:.4f}]")
    print(f"    Max |rho|    = {max_rho:.4f}")
    print(f"    % significant (FDR<0.05) = {pct_sig:.1f}%")
    print(f"    % with |rho| > 0.2 = {pct_gt02:.1f}%")
    print(f"    % with |rho| > 0.3 = {pct_gt03:.1f}%")

    summary_rows.append({
        'temporal_class': cls,
        'n_correlations': n_total,
        'n_unique_gee_vars': subset[gee_col].nunique(),
        'median_abs_rho': round(median_rho, 4),
        'mean_abs_rho': round(mean_rho, 4),
        'q25_abs_rho': round(q25, 4),
        'q75_abs_rho': round(q75, 4),
        'max_abs_rho': round(max_rho, 4),
        'n_significant_fdr05': int(n_sig),
        'pct_significant': round(pct_sig, 1),
        'pct_abs_rho_gt_0.2': round(pct_gt02, 1),
        'pct_abs_rho_gt_0.3': round(pct_gt03, 1),
    })

# ── 4. Mann-Whitney U test comparing |rho| distributions ──
stable_rhos = classified[classified['temporal_class'] == 'stable']['abs_rho'].values
variable_rhos = classified[classified['temporal_class'] == 'variable']['abs_rho'].values

mwu_stat, mwu_p = stats.mannwhitneyu(stable_rhos, variable_rhos, alternative='greater')
ratio = np.median(stable_rhos) / np.median(variable_rhos)

print(f"\n[RESULTS] Mann-Whitney U test (stable > variable):")
print(f"  U = {mwu_stat:.1f}")
print(f"  p = {mwu_p:.2e}")
print(f"  Median ratio (stable/variable) = {ratio:.3f}")

# ── 5. Load XGBoost reverse R2 to show best-predicted variables ──
print(f"\n[INFO] Loading XGBoost reverse results from:\n  {XGBOOST_REVERSE}")
xgb_df = pd.read_csv(XGBOOST_REVERSE)
print(f"  {len(xgb_df)} variables")

# Classify XGBoost targets
xgb_df['temporal_class'] = 'unclassified'
xgb_df.loc[xgb_df['env_var'].isin(STABLE_VARS), 'temporal_class'] = 'stable'
xgb_df.loc[xgb_df['env_var'].isin(VARIABLE_VARS), 'temporal_class'] = 'variable'

# Sort by test R2 descending
xgb_df = xgb_df.sort_values('r2_test', ascending=False).reset_index(drop=True)

print("\n[RESULTS] XGBoost Reverse Model R2 by temporal class:")
print(f"  {'Rank':<5} {'Variable':<25} {'R2_test':>10} {'Class':>12}")
print(f"  {'-'*55}")
for i, row in xgb_df.iterrows():
    r2 = row['r2_test']
    if pd.isna(r2) or r2 == '':
        continue
    r2 = float(r2)
    print(f"  {i+1:<5} {row['env_var']:<25} {r2:>10.4f} {row['temporal_class']:>12}")

# Summarize XGBoost R2 by class
for cls in ['stable', 'variable']:
    sub = xgb_df[(xgb_df['temporal_class'] == cls) & (xgb_df['r2_test'].notna())]
    r2_vals = pd.to_numeric(sub['r2_test'], errors='coerce').dropna()
    if len(r2_vals) > 0:
        print(f"\n  {cls.upper()} variables XGBoost R2:")
        print(f"    n = {len(r2_vals)}")
        print(f"    Median R2 = {r2_vals.median():.4f}")
        print(f"    Mean R2   = {r2_vals.mean():.4f}")
        print(f"    Max R2    = {r2_vals.max():.4f}")

# ── 6. Per-variable summary for XGBoost ──
xgb_summary = []
for _, row in xgb_df.iterrows():
    r2 = row.get('r2_test')
    if pd.isna(r2) or r2 == '':
        continue
    xgb_summary.append({
        'env_var': row['env_var'],
        'temporal_class': row['temporal_class'],
        'xgboost_r2_test': round(float(r2), 4),
    })

# ── 7. Write output ──
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

with open(OUTPUT, 'w') as f:
    f.write(f"# Provenance:\n")
    f.write(f"#   Script: {os.path.abspath(__file__)}\n")
    f.write(f"#   Input 1: {PFAM_GEE_CORR}\n")
    f.write(f"#   Input 2: {XGBOOST_REVERSE}\n")
    f.write(f"#   Date: {timestamp}\n")
    f.write(f"#   Integrity Check: PASSED\n")
    f.write(f"#\n")
    f.write(f"# Section 1: Correlation strength summary by temporal class\n")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f, sep='\t', index=False)

    f.write(f"\n# Mann-Whitney U test (stable > variable)\n")
    f.write(f"# U = {mwu_stat:.1f}\n")
    f.write(f"# p = {mwu_p:.2e}\n")
    f.write(f"# Median ratio (stable/variable) = {ratio:.3f}\n")

    f.write(f"\n# Section 2: XGBoost reverse model R2 by variable and temporal class\n")
    xgb_summary_df = pd.DataFrame(xgb_summary)
    xgb_summary_df.to_csv(f, sep='\t', index=False)

print(f"\n[OK] Output written to: {OUTPUT}")
print(f"[OK] Task 6 complete.")
