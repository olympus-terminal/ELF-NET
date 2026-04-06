#!/usr/bin/env python3
"""
01_preprocess.py — Preprocess merged Pfam+GEE dataset for KAN-CCA analysis.

Steps:
  1. Load merged TSV (Pfam domain counts + GEE environmental variables)
  2. Identify Pfam (PF*) and environmental columns
  3. Drop env columns with >50% NaN; filter SST outliers
  4. Prevalence-filter Pfam domains (>=5% nonzero samples)
  5. CLR-transform Pfam counts: log(x + 0.5) - row geometric mean
  6. Drop rows with any NaN in either feature set
  7. Standardize environmental variables (zero mean, unit variance)
  8. Save preprocessed arrays to output directory

Output files (in OUTPUT_DIR):
  - X_domain.npy       : CLR-transformed Pfam matrix (n_samples x n_domains)
  - X_env.npy          : Standardized environmental matrix (n_samples x n_env)
  - sample_ids.npy     : Assembly IDs for each row
  - feature_names.json : {"domain_cols": [...], "env_cols": [...]}
  - lat_lon.npy        : Latitude/longitude pairs (n_samples x 2)
"""

import json
import socket
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_base_dir():
    scratch = Path('/scratch/drn2/PROJECTS/TARA-LA4SR')
    archive = Path('/archive/drn2/TARA-Oceans')
    return scratch, archive

BASE_DIR, ARCHIVE_DIR = get_base_dir()
OUTPUT_DIR = BASE_DIR / '03_analyses' / 'kan_cca'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def find_data(relative_path):
    for base in [BASE_DIR, ARCHIVE_DIR]:
        p = base / relative_path
        if p.exists():
            return p
    raise FileNotFoundError(f"Not found in scratch or archive: {relative_path}")

# ---------------------------------------------------------------------------
# Environmental column definitions — all candidate columns
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
    # WOA23 dissolved nutrients + MLD (added 2026-03-20)
    'nitrate_umol_l', 'phosphate_umol_l', 'silicate_umol_l',
    'oxygen_umol_l', 'mld_m',
]

# Physical limits for outlier filtering
PHYSICAL_BOUNDS = {
    'sst_mean_c': (-5, 40),
    'sst_max_c': (-5, 45),
    'sst_min_c': (-5, 40),
    'sst_range_c': (0, 30),
    'modis_sst_mean_c': (-5, 40),
    'bathymetry_m': (-11000, 0),  # ocean depths are negative
    'distance_to_coast_km': (0, 20000),
    'chl_mean_mg_m3': (0, 100),
    'chl_max_mg_m3': (0, 200),
    'chl_min_mg_m3': (0, 100),
    'poc_mean_mg_m3': (0, 10000),
    # WOA23 dissolved nutrients + MLD bounds
    'nitrate_umol_l': (0, 50),       # µmol/L; max ~45 in deep upwelling
    'phosphate_umol_l': (0, 5),      # µmol/L; max ~3.5 in deep water
    'silicate_umol_l': (0, 200),     # µmol/L; max ~180 in deep Pacific
    'oxygen_umol_l': (0, 400),       # µmol/L; max ~380 in cold polar
    'mld_m': (0, 1500),              # meters; max ~1000 in Labrador Sea
}

MAX_NAN_FRACTION = 0.50  # drop columns with >50% NaN

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{ts}] 01_preprocess.py starting on {socket.gethostname()}")

    # 1. Load merged TSV (prefer nutrients-merged if available)
    import glob as glob_mod
    data_path = None
    for pattern in [
        str(BASE_DIR / '03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_nutrients_merged_*.tsv'),
        str(BASE_DIR / '03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_merged_GPS_RECOVERED_*.tsv'),
        str(BASE_DIR / '03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_merged_SMART_*.tsv'),
        str(BASE_DIR / '03_analyses/algagpt_gee_pfam_merged_SMART_*.tsv'),
    ]:
        found = sorted(glob_mod.glob(pattern))
        if found:
            data_path = Path(found[-1])
            break
    if data_path is None:
        data_path = find_data(
            '03_analyses/ALGAGPT-based-analyses/'
            'algagpt_gee_pfam_merged_GPS_RECOVERED_20260124_114445.tsv'
        )
    print(f"Loading: {data_path}")
    df = pd.read_csv(data_path, sep='\t', comment='#', low_memory=False)
    n_raw = len(df)
    print(f"Raw shape: {df.shape}")

    # 2. Identify Pfam columns
    pfam_cols = [c for c in df.columns if c.startswith('PF')]
    print(f"Pfam columns: {len(pfam_cols)}")

    # 3. Process environmental columns
    env_cols_present = [c for c in ALL_ENV_COLS if c in df.columns]
    print(f"Candidate env columns: {len(env_cols_present)}")

    # Convert to numeric
    env_df = df[env_cols_present].copy()
    for col in env_cols_present:
        env_df[col] = pd.to_numeric(env_df[col], errors='coerce')

    # Apply physical bounds — set outliers to NaN
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col in env_df.columns:
            mask = (env_df[col] < lo) | (env_df[col] > hi)
            n_outlier = mask.sum()
            if n_outlier > 0:
                print(f"  {col}: {n_outlier} outliers set to NaN (bounds [{lo}, {hi}])")
                env_df.loc[mask, col] = np.nan

    # Drop columns with >50% NaN
    nan_frac = env_df.isna().mean()
    keep_cols = nan_frac[nan_frac <= MAX_NAN_FRACTION].index.tolist()
    drop_cols = nan_frac[nan_frac > MAX_NAN_FRACTION].index.tolist()
    print(f"Env columns dropped (>{MAX_NAN_FRACTION*100:.0f}% NaN): {drop_cols}")
    print(f"Env columns kept: {len(keep_cols)}: {keep_cols}")
    env_df = env_df[keep_cols]

    # 4. Prevalence filter — keep Pfam domains present in >=5% of samples
    pfam_data = df[pfam_cols].fillna(0)
    prevalence = (pfam_data > 0).mean(axis=0)
    threshold = 0.05
    keep_mask = prevalence >= threshold
    pfam_cols_filtered = list(prevalence[keep_mask].index)
    print(f"Prevalence filter (>={threshold*100:.0f}%): {len(pfam_cols)} -> {len(pfam_cols_filtered)} domains")

    # 5. CLR transform: log(x + 0.5) - row geometric mean of log(x + 0.5)
    X_pfam = pfam_data[pfam_cols_filtered].values.astype(np.float64)
    log_X = np.log(X_pfam + 0.5)
    row_geom_mean = log_X.mean(axis=1, keepdims=True)
    X_clr = log_X - row_geom_mean
    print(f"CLR-transformed domain matrix: {X_clr.shape}")

    # 6. Drop rows with NaN in lat/lon or env
    X_env_raw = env_df.values.astype(np.float64)
    lat_lon_raw = df[['latitude', 'longitude']].values.astype(np.float64)

    nan_domain = np.any(np.isnan(X_clr), axis=1)
    nan_env = np.any(np.isnan(X_env_raw), axis=1)
    nan_latlon = np.any(np.isnan(lat_lon_raw), axis=1)
    nan_any = nan_domain | nan_env | nan_latlon
    keep_rows = ~nan_any
    print(f"Rows with NaN: domain={nan_domain.sum()}, env={nan_env.sum()}, "
          f"latlon={nan_latlon.sum()}, any={nan_any.sum()}")
    print(f"Keeping {keep_rows.sum()} / {n_raw} rows ({keep_rows.sum()/n_raw*100:.1f}%)")

    X_clr = X_clr[keep_rows]
    X_env_raw = X_env_raw[keep_rows]
    sample_ids = df['assembly_id'].values[keep_rows]
    lat_lon = lat_lon_raw[keep_rows]

    # 7. Standardize env (zero mean, unit variance)
    env_mean = X_env_raw.mean(axis=0)
    env_std = X_env_raw.std(axis=0)
    env_std[env_std == 0] = 1.0
    X_env_std = (X_env_raw - env_mean) / env_std

    print(f"Standardized env: mean range [{X_env_std.mean(axis=0).min():.6f}, "
          f"{X_env_std.mean(axis=0).max():.6f}]")
    print(f"Standardized env: std range [{X_env_std.std(axis=0).min():.6f}, "
          f"{X_env_std.std(axis=0).max():.6f}]")

    # 8. Save outputs
    np.save(OUTPUT_DIR / 'X_domain.npy', X_clr)
    np.save(OUTPUT_DIR / 'X_env.npy', X_env_std)
    np.save(OUTPUT_DIR / 'sample_ids.npy', sample_ids)
    np.save(OUTPUT_DIR / 'lat_lon.npy', lat_lon)

    feature_names = {
        'domain_cols': pfam_cols_filtered,
        'env_cols': keep_cols,
    }
    with open(OUTPUT_DIR / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Samples: {X_clr.shape[0]} (from {n_raw} raw)")
    print(f"Domain features (CLR): {X_clr.shape[1]}")
    print(f"Env features (std): {X_env_std.shape[1]}")
    print(f"Env column names: {keep_cols}")
    print(f"Lat/Lon: {lat_lon.shape}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files saved:")
    for name in ['X_domain.npy', 'X_env.npy', 'sample_ids.npy', 'lat_lon.npy',
                  'feature_names.json']:
        p = OUTPUT_DIR / name
        sz = p.stat().st_size / 1024
        print(f"  {name}: {sz:.1f} KB")
    print(f"Timestamp: {ts}")

if __name__ == '__main__':
    main()
