#!/usr/bin/env python3
"""
01_extract_nutrients.py — Extract WOA23 nutrient values at TARA sample coordinates.

Reads WOA23 annual climatology NetCDFs (nitrate, phosphate, silicate, dissolved
oxygen) and de Boyer Montégut MLD climatology, then extracts surface values at
each GPS-mapped sample coordinate using nearest-neighbor interpolation.

Input:
    - WOA23 NetCDFs in {BASE}/03_analyses/woa23_nutrients/
    - GPS coordinates from merged dataset:
      {BASE}/03_analyses/ALGAGPT-based-analyses/
      algagpt_gee_pfam_merged_GPS_RECOVERED_20260124_114445.tsv

Output:
    - {BASE}/03_analyses/woa23_nutrients/nutrients_at_sample_coordinates.tsv
      Columns: assembly_id, latitude, longitude, nitrate_umol_l,
               phosphate_umol_l, silicate_umol_l, oxygen_umol_l, mld_m

Usage:
    python3 scripts/nutrients/01_extract_nutrients.py

Author: TARA-OMEN analysis pipeline
Date: 2026-03-20
"""

import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError:
    print("ERROR: xarray required. Install with: pip install xarray netCDF4")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_base_dir():
    hostname = socket.gethostname()
    if any(x in hostname for x in ['cn', 'gpu', 'dn', 'jubail', 'login', 'fast']):
        return Path("/scratch/drn2/PROJECTS/TARA-LA4SR")
    else:
        # Local development — use relative data directory
        script_dir = Path(__file__).resolve().parent
        return script_dir.parent.parent / "data"

BASE = get_base_dir()
WOA_DIR = BASE / "03_analyses" / "woa23_nutrients" if "scratch" in str(BASE) else BASE / "woa23_nutrients"

# ---------------------------------------------------------------------------
# WOA23 variable mapping
# ---------------------------------------------------------------------------

# WOA23 NetCDF variable names (inside the files)
WOA_FILES = {
    'nitrate_umol_l': {
        'filename': 'woa23_all_n00_01.nc',
        'varname': 'n_an',  # WOA23 analyzed mean field
    },
    'phosphate_umol_l': {
        'filename': 'woa23_all_p00_01.nc',
        'varname': 'p_an',
    },
    'silicate_umol_l': {
        'filename': 'woa23_all_i00_01.nc',
        'varname': 'i_an',
    },
    'oxygen_umol_l': {
        'filename': 'woa23_all_o00_01.nc',
        'varname': 'o_an',
    },
}

MLD_FILE = None  # Will be auto-detected from WOA_DIR

def load_gps_coordinates():
    """Load GPS coordinates from the merged GEE+PFAM dataset."""
    if "scratch" in str(BASE):
        # Try multiple known locations on HPC
        candidates = []
        for pattern in [
            BASE / "03_analyses" / "ALGAGPT-based-analyses" / "algagpt_gee_pfam_merged_GPS_RECOVERED_*.tsv",
            BASE / "03_analyses" / "ALGAGPT-based-analyses" / "algagpt_gee_pfam_merged_SMART_*.tsv",
            BASE / "03_analyses" / "algagpt_gee_pfam_merged_SMART_*.tsv",
            BASE / "03_analyses" / "algagpt_gee_pfam_merged_*.tsv",
            BASE / "algagpt_gee_pfam_merged_*.tsv",
        ]:
            import glob as glob_mod
            found = sorted(glob_mod.glob(str(pattern)))
            candidates.extend(found)
        if candidates:
            merged_path = Path(candidates[0])
        else:
            print("ERROR: Cannot find merged GPS dataset on HPC")
            sys.exit(1)
    else:
        # For local dev, try common locations
        import glob as glob_mod
        candidates = sorted(glob_mod.glob(str(BASE / "**" / "algagpt_gee_pfam_merged_*.tsv")))
        if not candidates:
            candidates = sorted(glob_mod.glob("**/algagpt_gee_pfam_merged_*.tsv"))
        if candidates:
            merged_path = Path(candidates[0])
        else:
            print("ERROR: Cannot find merged GPS dataset")
            sys.exit(1)

    print(f"Loading GPS coordinates from: {merged_path}")
    df = pd.read_csv(merged_path, sep='\t', comment='#', low_memory=False)

    # Identify assembly ID and GPS columns
    id_col = 'assembly_id'
    lat_col = next((c for c in df.columns if c.lower() in ('latitude', 'lat')), None)
    lon_col = next((c for c in df.columns if c.lower() in ('longitude', 'lon', 'lng')), None)

    if lat_col is None or lon_col is None:
        print(f"Available columns: {list(df.columns[:20])}")
        print("ERROR: Cannot find latitude/longitude columns")
        sys.exit(1)

    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')

    # Keep only rows with valid GPS
    gps = df[[id_col, lat_col, lon_col]].dropna(subset=[lat_col, lon_col])
    gps = gps.rename(columns={lat_col: 'latitude', lon_col: 'longitude'})
    gps = gps.drop_duplicates(subset=[id_col])

    print(f"  {len(gps)} samples with valid GPS coordinates")
    return gps

def extract_woa23_at_points(nc_path, varname, lats, lons):
    """
    Extract values from a WOA23 NetCDF at given lat/lon points.

    Uses nearest-neighbor lookup on the 1° grid.
    WOA23 files have dimensions: (depth, lat, lon) for the analyzed field.
    We take the surface layer (depth index 0).
    """
    ds = xr.open_dataset(nc_path, decode_times=False)

    # The analyzed mean field
    if varname not in ds:
        # Try common alternatives
        candidates = [v for v in ds.data_vars if '_an' in v or '_mn' in v]
        if candidates:
            varname = candidates[0]
            print(f"  Using variable: {varname}")
        else:
            print(f"  ERROR: Variable {varname} not found. Available: {list(ds.data_vars)}")
            ds.close()
            return np.full(len(lats), np.nan)

    data = ds[varname]

    # Select surface depth (index 0)
    if 'depth' in data.dims:
        data = data.isel(depth=0)
    elif 'level' in data.dims:
        data = data.isel(level=0)

    # If there's a time dimension, select first (annual mean)
    if 'time' in data.dims:
        data = data.isel(time=0)

    # Extract at each point using nearest-neighbor
    values = np.full(len(lats), np.nan)
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        try:
            val = float(data.sel(lat=lat, lon=lon, method='nearest').values)
            if not np.isnan(val) and val < 1e10:  # WOA fill value check
                values[i] = val
        except (KeyError, ValueError):
            pass

    ds.close()
    return values

def extract_mld_at_points(nc_path, lats, lons):
    """
    Extract MLD values from de Boyer Montégut climatology.

    The file contains monthly climatology; we compute annual mean MLD
    for each coordinate.
    """
    ds = xr.open_dataset(nc_path, decode_times=False)

    # Identify the MLD variable
    mld_candidates = [v for v in ds.data_vars
                      if 'mld' in v.lower() or 'mix' in v.lower()]
    if not mld_candidates:
        # Use the first non-coordinate variable
        mld_candidates = [v for v in ds.data_vars
                          if v not in ('lat', 'lon', 'time', 'mask')]

    if not mld_candidates:
        print(f"  ERROR: No MLD variable found. Available: {list(ds.data_vars)}")
        ds.close()
        return np.full(len(lats), np.nan)

    varname = mld_candidates[0]
    print(f"  MLD variable: {varname}, dims: {ds[varname].dims}")
    data = ds[varname]

    # Compute annual mean if monthly
    if 'time' in data.dims and data.sizes['time'] > 1:
        data = data.mean(dim='time', skipna=True)
    elif 'month' in data.dims:
        data = data.mean(dim='month', skipna=True)

    # Identify lat/lon dimension names
    lat_dim = next((d for d in data.dims if d in ('lat', 'latitude', 'y')), None)
    lon_dim = next((d for d in data.dims if d in ('lon', 'longitude', 'x')), None)

    if lat_dim is None or lon_dim is None:
        print(f"  ERROR: Cannot identify lat/lon dims: {data.dims}")
        ds.close()
        return np.full(len(lats), np.nan)

    values = np.full(len(lats), np.nan)
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        try:
            val = float(data.sel({lat_dim: lat, lon_dim: lon}, method='nearest').values)
            if not np.isnan(val) and val < 1e10:
                values[i] = val
        except (KeyError, ValueError):
            pass

    ds.close()
    return values

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{ts}] 01_extract_nutrients.py starting on {socket.gethostname()}")
    print(f"WOA23 directory: {WOA_DIR}")

    # Load GPS coordinates
    gps = load_gps_coordinates()
    lats = gps['latitude'].values
    lons = gps['longitude'].values

    # Extract each WOA23 variable
    result = gps[['assembly_id', 'latitude', 'longitude']].copy()

    for col_name, info in WOA_FILES.items():
        nc_path = WOA_DIR / info['filename']
        if not nc_path.exists():
            print(f"[SKIP] {col_name}: {nc_path} not found")
            result[col_name] = np.nan
            continue

        print(f"Extracting {col_name} from {nc_path.name}...")
        values = extract_woa23_at_points(nc_path, info['varname'], lats, lons)
        result[col_name] = values

        n_valid = np.sum(~np.isnan(values))
        pct = 100.0 * n_valid / len(values)
        print(f"  {n_valid}/{len(values)} valid ({pct:.1f}%)")

        # Sanity check ranges
        valid = values[~np.isnan(values)]
        if len(valid) > 0:
            print(f"  Range: [{valid.min():.2f}, {valid.max():.2f}], "
                  f"median: {np.median(valid):.2f}")

    # Extract MLD — auto-detect the NetCDF file
    mld_candidates = sorted(WOA_DIR.glob('mld*.nc')) + sorted(WOA_DIR.glob('**/mld*.nc'))
    mld_path = mld_candidates[0] if mld_candidates else WOA_DIR / 'mld_DT02_c1m_reg2.0.nc'
    if mld_path.exists():
        print(f"Extracting MLD from {mld_path.name}...")
        mld_values = extract_mld_at_points(mld_path, lats, lons)
        result['mld_m'] = mld_values

        n_valid = np.sum(~np.isnan(mld_values))
        pct = 100.0 * n_valid / len(mld_values)
        print(f"  {n_valid}/{len(mld_values)} valid ({pct:.1f}%)")

        valid = mld_values[~np.isnan(mld_values)]
        if len(valid) > 0:
            print(f"  Range: [{valid.min():.1f}, {valid.max():.1f}] m, "
                  f"median: {np.median(valid):.1f} m")
    else:
        print(f"[SKIP] MLD: {mld_path} not found")
        result['mld_m'] = np.nan

    # Write output
    out_path = WOA_DIR / "nutrients_at_sample_coordinates.tsv"
    header_lines = [
        f"# Nutrient extraction at TARA sample coordinates",
        f"# Generated: {datetime.now().isoformat()}",
        f"# Host: {socket.gethostname()}",
        f"# Sources: WOA23 annual climatology (1° grid, surface), "
        f"de Boyer Montégut MLD climatology",
        f"# N samples: {len(result)}",
    ]

    with open(out_path, 'w') as f:
        for line in header_lines:
            f.write(line + '\n')
        result.to_csv(f, sep='\t', index=False)

    print(f"\nOutput: {out_path}")

    # Coverage summary
    print("\n=== Coverage Summary ===")
    nutrient_cols = ['nitrate_umol_l', 'phosphate_umol_l', 'silicate_umol_l',
                     'oxygen_umol_l', 'mld_m']
    for col in nutrient_cols:
        if col in result.columns:
            n_valid = result[col].notna().sum()
            pct = 100.0 * n_valid / len(result)
            print(f"  {col}: {n_valid}/{len(result)} ({pct:.1f}%)")

    # Complete coverage (all 5 nutrients non-NaN)
    present_cols = [c for c in nutrient_cols if c in result.columns]
    all_valid = result[present_cols].notna().all(axis=1).sum()
    pct_all = 100.0 * all_valid / len(result)
    print(f"  All nutrients valid: {all_valid}/{len(result)} ({pct_all:.1f}%)")

    # Latitudinal sanity check
    print("\n=== Latitudinal Gradient Check (nitrate) ===")
    if 'nitrate_umol_l' in result.columns:
        tropical = result[(result['latitude'].abs() < 23.5) &
                          result['nitrate_umol_l'].notna()]
        highlatN = result[(result['latitude'] > 45) &
                          result['nitrate_umol_l'].notna()]
        highlatS = result[(result['latitude'] < -45) &
                          result['nitrate_umol_l'].notna()]

        if len(tropical) > 0:
            print(f"  Tropical (<23.5°): median NO₃ = "
                  f"{tropical['nitrate_umol_l'].median():.2f} µmol/L "
                  f"(n={len(tropical)})")
        if len(highlatN) > 0:
            print(f"  High-lat N (>45°): median NO₃ = "
                  f"{highlatN['nitrate_umol_l'].median():.2f} µmol/L "
                  f"(n={len(highlatN)})")
        if len(highlatS) > 0:
            print(f"  High-lat S (<-45°): median NO₃ = "
                  f"{highlatS['nitrate_umol_l'].median():.2f} µmol/L "
                  f"(n={len(highlatS)})")

if __name__ == '__main__':
    main()
