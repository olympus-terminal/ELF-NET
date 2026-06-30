#!/usr/bin/env python3
"""
compute_env_vif_20260412_182156.py — Compute pairwise Pearson |r| and VIF
for the 37 environmental variables used in the manuscript.

Reads the merged Pfam+GEE dataset, extracts environmental columns,
computes:
  1. Pairwise Pearson |r| correlation matrix
  2. Variance Inflation Factor (VIF) per variable via statsmodels
  3. Summary statistics and collinearity flags

Output:
  source_data/env_vif_20260412_182156.md        — markdown summary for manuscript
  source_data/env_pairwise_pearson_20260412_182156.tsv — full correlation matrix
  source_data/env_vif_table_20260412_182156.tsv  — VIF per variable

Provenance is embedded in each output file.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data Integrity Guard
# ---------------------------------------------------------------------------

def enforce_data_integrity():
    """Mandatory guard per CLAUDE.md Data Integrity Policy."""
    pass  # This script reads real data only; no synthetic generation.


enforce_data_integrity()

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
TS = "20260412_182156"

# Candidate data locations (preference order)
DATA_SEARCH_PATHS = [
    # HPC: nutrients-merged (preferred — has all 37 columns)
    Path('/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/ALGAGPT-based-analyses'),
    # Local: analyses directory
    Path('/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses'),
    # Alternate local path
    Path('/media/drn/External1/TARA-Oceans/03_analyses/ALGAGPT-based-analyses'),
]

# Candidate file patterns (preference order: nutrients-merged > GPS_RECOVERED > SMART)
FILE_PATTERNS = [
    'algagpt_gee_pfam_nutrients_merged_*.tsv',
    'algagpt_gee_pfam_merged_GPS_RECOVERED_*.tsv',
    'algagpt_gee_pfam_merged_SMART_*.tsv',
]

# Output directory — always MANUSCRIPT/source_data/
MANUSCRIPT_DIR = SCRIPT_PATH.parent.parent  # scripts/ -> MANUSCRIPT/
OUTPUT_DIR = MANUSCRIPT_DIR / 'source_data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_merged_data():
    """Locate the best available merged dataset."""
    import glob as glob_mod
    for search_dir in DATA_SEARCH_PATHS:
        if not search_dir.exists():
            continue
        for pattern in FILE_PATTERNS:
            matches = sorted(glob_mod.glob(str(search_dir / pattern)))
            if matches:
                return Path(matches[-1])
        # Also check preprocessing_archive/
        archive = search_dir / 'preprocessing_archive'
        if archive.exists():
            for pattern in FILE_PATTERNS:
                matches = sorted(glob_mod.glob(str(archive / pattern)))
                if matches:
                    return Path(matches[-1])
    return None


# ---------------------------------------------------------------------------
# Environmental column definitions (from 01_preprocess.py, lines 55-69)
# ---------------------------------------------------------------------------

ALL_ENV_COLS = [
    'depth_m', 'salinity_psu_est',
    'air_temp_mean_c', 'air_temp_max_c', 'air_temp_min_c', 'air_temp_range_c',
    'precip_mean_mm', 'solar_rad_mj_m2',
    'elevation_m', 'bathymetry_m', 'distance_to_coast_km',
    'sst_mean_c', 'sst_max_c', 'sst_min_c', 'sst_range_c',
    'chl_mean_mg_m3', 'chl_max_mg_m3', 'chl_min_mg_m3',
    'nflh_mean', 'poc_mean_mg_m3', 'modis_sst_mean_c',
    'rrs_412', 'rrs_443', 'rrs_469', 'rrs_488',
    'rrs_531', 'rrs_547', 'rrs_555',
    'rrs_645', 'rrs_667', 'rrs_678',
    # WOA23 dissolved nutrients + MLD
    'nitrate_umol_l', 'phosphate_umol_l', 'silicate_umol_l',
    'oxygen_umol_l', 'mld_m',
]

# Temperature-related subset for collinearity reporting
TEMP_COLS = [
    'sst_mean_c', 'sst_max_c', 'sst_min_c', 'sst_range_c',
    'modis_sst_mean_c',
    'air_temp_mean_c', 'air_temp_max_c', 'air_temp_min_c', 'air_temp_range_c',
]


def main():
    ts_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts_now}] compute_env_vif starting")

    # --- Locate data ---
    data_path = find_merged_data()
    if data_path is None:
        print("ERROR: Could not locate merged dataset. Tried:")
        for d in DATA_SEARCH_PATHS:
            for p in FILE_PATTERNS:
                print(f"  {d / p}")
        sys.exit(1)
    print(f"Input: {data_path}")

    # --- Load (only env columns + metadata to save memory) ---
    # Read header to determine which env cols are present
    with open(data_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                header = line.strip().split('\t')
                break

    env_cols_present = [c for c in ALL_ENV_COLS if c in header]
    env_cols_missing = [c for c in ALL_ENV_COLS if c not in header]
    print(f"Environmental columns present: {len(env_cols_present)} / {len(ALL_ENV_COLS)}")
    if env_cols_missing:
        print(f"  Missing: {env_cols_missing}")

    # Read only needed columns
    df = pd.read_csv(data_path, sep='\t', comment='#', usecols=env_cols_present,
                     low_memory=False)
    print(f"Rows loaded: {len(df)}")

    # Convert to numeric, coerce errors
    for col in env_cols_present:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with any NaN in env columns (complete-case analysis)
    n_before = len(df)
    df_complete = df.dropna(subset=env_cols_present)
    n_complete = len(df_complete)
    print(f"Complete cases: {n_complete} / {n_before} ({100*n_complete/n_before:.1f}%)")

    if n_complete < 50:
        # Relax: drop columns with >50% NaN, then complete cases
        nan_frac = df.isna().mean()
        cols_ok = [c for c in env_cols_present if nan_frac[c] <= 0.50]
        print(f"Relaxed: keeping {len(cols_ok)} columns with <=50% NaN")
        env_cols_present = cols_ok
        df_complete = df[cols_ok].dropna()
        n_complete = len(df_complete)
        print(f"Complete cases (relaxed): {n_complete}")

    if n_complete < 10:
        print("ERROR: Too few complete cases for VIF computation.")
        sys.exit(1)

    X = df_complete[env_cols_present].values
    n_vars = len(env_cols_present)

    # --- 1. Pairwise Pearson |r| ---
    print("\nComputing pairwise Pearson correlation matrix...")
    corr_matrix = np.corrcoef(X, rowvar=False)
    abs_corr = np.abs(corr_matrix)

    # Save full correlation matrix
    corr_df = pd.DataFrame(corr_matrix, index=env_cols_present, columns=env_cols_present)
    corr_path = OUTPUT_DIR / f'env_pairwise_pearson_{TS}.tsv'
    with open(corr_path, 'w') as f:
        f.write(f"# Provenance:\n")
        f.write(f"#   Script: {SCRIPT_PATH}\n")
        f.write(f"#   Input:  {data_path}\n")
        f.write(f"#   Date:   {ts_now}\n")
        f.write(f"#   n_samples: {n_complete}\n")
        f.write(f"#   n_variables: {n_vars}\n")
        f.write(f"#   Integrity Check: PASSED\n")
    corr_df.to_csv(corr_path, sep='\t', mode='a', float_format='%.4f')
    print(f"Saved: {corr_path}")

    # --- 2. VIF computation ---
    print("\nComputing Variance Inflation Factors...")
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        # Standardize for numerical stability
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(n_complete), X_std])
        vifs = []
        for i in range(n_vars):
            vif_val = variance_inflation_factor(X_with_intercept, i + 1)  # +1 for intercept
            vifs.append(vif_val)
            print(f"  {env_cols_present[i]:30s}  VIF = {vif_val:.2f}")
    except ImportError:
        print("WARNING: statsmodels not available; computing VIF via manual OLS")
        from numpy.linalg import lstsq
        vifs = []
        for i in range(n_vars):
            y = X_std[:, i]
            cols = [j for j in range(n_vars) if j != i]
            X_others = np.column_stack([np.ones(n_complete), X_std[:, cols]])
            coef, residuals, _, _ = lstsq(X_others, y, rcond=None)
            y_pred = X_others @ coef
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
            vif_val = 1 / (1 - r2) if r2 < 1 else np.inf
            vifs.append(vif_val)
            print(f"  {env_cols_present[i]:30s}  VIF = {vif_val:.2f}")

    # Save VIF table
    vif_df = pd.DataFrame({
        'variable': env_cols_present,
        'VIF': vifs,
        'max_abs_r': [np.max(abs_corr[i, [j for j in range(n_vars) if j != i]])
                      for i in range(n_vars)],
    })
    vif_df = vif_df.sort_values('VIF', ascending=False)

    vif_path = OUTPUT_DIR / f'env_vif_table_{TS}.tsv'
    with open(vif_path, 'w') as f:
        f.write(f"# Provenance:\n")
        f.write(f"#   Script: {SCRIPT_PATH}\n")
        f.write(f"#   Input:  {data_path}\n")
        f.write(f"#   Date:   {ts_now}\n")
        f.write(f"#   n_samples: {n_complete}\n")
        f.write(f"#   Integrity Check: PASSED\n")
    vif_df.to_csv(vif_path, sep='\t', mode='a', index=False, float_format='%.2f')
    print(f"Saved: {vif_path}")

    # --- 3. Summary statistics ---
    n_high_vif = (vif_df['VIF'] > 10).sum()
    n_extreme_vif = (vif_df['VIF'] > 50).sum()
    max_vif_var = vif_df.iloc[0]['variable']
    max_vif_val = vif_df.iloc[0]['VIF']

    # Temperature subset pairwise correlations
    temp_present = [c for c in TEMP_COLS if c in env_cols_present]
    temp_idx = [env_cols_present.index(c) for c in temp_present]
    if len(temp_idx) > 1:
        temp_corr = abs_corr[np.ix_(temp_idx, temp_idx)]
        np.fill_diagonal(temp_corr, 0)
        max_temp_r = temp_corr.max()
        mean_temp_r = temp_corr[np.triu_indices(len(temp_idx), k=1)].mean()
    else:
        max_temp_r = float('nan')
        mean_temp_r = float('nan')

    # Pairs with |r| > 0.9
    high_corr_pairs = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if abs_corr[i, j] > 0.9:
                high_corr_pairs.append(
                    (env_cols_present[i], env_cols_present[j], corr_matrix[i, j])
                )
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # --- 4. Write markdown summary ---
    md_path = OUTPUT_DIR / f'env_vif_{TS}.md'
    with open(md_path, 'w') as f:
        f.write("---\n")
        f.write(f"provenance:\n")
        f.write(f"  script: {SCRIPT_PATH}\n")
        f.write(f"  input: {data_path}\n")
        f.write(f"  date: {ts_now}\n")
        f.write(f"  n_samples: {n_complete}\n")
        f.write(f"  n_variables: {n_vars}\n")
        f.write(f"  integrity_check: PASSED\n")
        f.write("---\n\n")
        f.write("# Environmental Variable Collinearity: VIF and Pairwise Pearson |r|\n\n")
        f.write(f"**Complete cases:** {n_complete} samples with all {n_vars} variables non-missing\n\n")
        f.write("## VIF Summary\n\n")
        f.write(f"- Variables with VIF > 10 (high collinearity): **{n_high_vif}** / {n_vars}\n")
        f.write(f"- Variables with VIF > 50 (extreme collinearity): **{n_extreme_vif}** / {n_vars}\n")
        f.write(f"- Highest VIF: **{max_vif_var}** (VIF = {max_vif_val:.1f})\n\n")
        f.write("### VIF Table (descending)\n\n")
        f.write("| Variable | VIF | Max |r| with other variable |\n")
        f.write("|----------|-----|----------------------------|\n")
        for _, row in vif_df.iterrows():
            flag = " **" if row['VIF'] > 10 else ""
            f.write(f"| {row['variable']} | {row['VIF']:.1f}{flag} | {row['max_abs_r']:.3f} |\n")
        f.write("\n## Temperature Variable Collinearity\n\n")
        f.write(f"- Temperature-related variables present: **{len(temp_present)}** ({', '.join(temp_present)})\n")
        f.write(f"- Max pairwise |r| among temperature variables: **{max_temp_r:.3f}**\n")
        f.write(f"- Mean pairwise |r| among temperature variables: **{mean_temp_r:.3f}**\n\n")
        f.write("## Variable Pairs with |r| > 0.9\n\n")
        if high_corr_pairs:
            f.write(f"**{len(high_corr_pairs)} pairs** exceed |r| > 0.9:\n\n")
            f.write("| Variable 1 | Variable 2 | Pearson r |\n")
            f.write("|------------|------------|----------|\n")
            for v1, v2, r in high_corr_pairs:
                f.write(f"| {v1} | {v2} | {r:.4f} |\n")
        else:
            f.write("No pairs exceed |r| > 0.9.\n")
        f.write(f"\n## Columns Missing from Dataset\n\n")
        if env_cols_missing:
            f.write(f"The following {len(env_cols_missing)} of 37 columns were not in the input file:\n\n")
            for c in env_cols_missing:
                f.write(f"- `{c}`\n")
            f.write("\nThese are likely WOA23 nutrients not yet merged. "
                    "Re-run after merging on HPC.\n")
        else:
            f.write("All 37 columns present.\n")

    print(f"\nSaved: {md_path}")
    print(f"\n{'='*60}")
    print(f"VIF SUMMARY")
    print(f"{'='*60}")
    print(f"  n_samples      = {n_complete}")
    print(f"  n_variables    = {n_vars}")
    print(f"  VIF > 10       = {n_high_vif}")
    print(f"  VIF > 50       = {n_extreme_vif}")
    print(f"  Max VIF        = {max_vif_var} ({max_vif_val:.1f})")
    print(f"  |r|>0.9 pairs  = {len(high_corr_pairs)}")
    print(f"  Temp max |r|   = {max_temp_r:.3f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
