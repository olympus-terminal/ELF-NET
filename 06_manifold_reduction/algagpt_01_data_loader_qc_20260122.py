#!/usr/bin/env python3
"""
Phase 1: Data Loading and Quality Control for algaGPT PFAM Analysis.

This script loads the algaGPT merged PFAM+environment data and prepares it
for downstream manifold learning and predictive modeling analyses.

Key steps:
1. Load merged TSV with PFAM abundances and environmental variables
2. Separate metadata, coordinates, environmental vars, and PFAM domains
3. Apply prevalence-based filtering to PFAM domains
4. Handle missing values and apply CLR transformation
5. Save processed arrays for subsequent phases

Provenance:
  Input: algagpt_gee_pfam_merged_SMART_20260119_100639.tsv
  Output: data/*.npy, data/*.txt
  Date: 2026-01-22
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import sys

# Data integrity enforcement
sys.path.insert(0, str(Path(__file__).parent))
from DataIntegrityGuard import enforce_data_integrity, validate_input_source
enforce_data_integrity()

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

# Input file
INPUT_FILE = BASE_DIR.parent / "algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Filtering parameters
PREVALENCE_THRESHOLD = 0.05  # Domain must be present in >= 5% of samples
MIN_SAMPLES_WITH_COORDS = 100  # Minimum samples with valid GPS

# Metadata columns (non-numeric, not used in analysis)
METADATA_COLS = [
    'assembly_id', 'matched_to', 'matched_sample', 'dataset',
    'collection_date', 'species', 'habitat', 'gps_source', 'gps_confidence'
]

# Coordinate columns
COORD_COLS = ['latitude', 'longitude']

# Environmental variable columns (all numeric env/satellite data)
ENV_COLS = [
    'depth_m', 'salinity_psu_est',
    'air_temp_mean_c', 'air_temp_max_c', 'air_temp_min_c', 'air_temp_range_c',
    'precip_mean_mm', 'solar_rad_mj_m2', 'elevation_m',
    'bathymetry_m', 'distance_to_coast_km', 'landcover_class',
    'sst_mean_c', 'sst_max_c', 'sst_min_c', 'sst_range_c',
    'chl_mean_mg_m3', 'chl_max_mg_m3', 'chl_min_mg_m3',
    'nflh_mean', 'poc_mean_mg_m3', 'modis_sst_mean_c',
    'rrs_412', 'rrs_443', 'rrs_469', 'rrs_488',
    'rrs_531', 'rrs_547', 'rrs_555', 'rrs_645', 'rrs_667', 'rrs_678'
]

def load_raw_data():
    """Load the merged algaGPT PFAM+environment TSV."""
    print("=" * 70)
    print("LOADING RAW DATA")
    print("=" * 70)

    validate_input_source(INPUT_FILE)
    print(f"  Input: {INPUT_FILE.name}")
    print(f"  Size: {INPUT_FILE.stat().st_size / 1e6:.1f} MB")

    # Read TSV, skipping comment lines
    df = pd.read_csv(INPUT_FILE, sep='\t', comment='#', low_memory=False)

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns):,}")

    return df

def identify_column_types(df):
    """Identify and categorize columns."""
    print("\n" + "=" * 70)
    print("IDENTIFYING COLUMN TYPES")
    print("=" * 70)

    all_cols = df.columns.tolist()

    # PFAM columns start with 'PF'
    pfam_cols = [c for c in all_cols if c.startswith('PF')]

    # Environmental columns that exist in dataframe
    env_cols_present = [c for c in ENV_COLS if c in all_cols]

    # Coordinate columns
    coord_cols_present = [c for c in COORD_COLS if c in all_cols]

    # Metadata columns
    meta_cols_present = [c for c in METADATA_COLS if c in all_cols]

    print(f"  Metadata columns: {len(meta_cols_present)}")
    print(f"  Coordinate columns: {len(coord_cols_present)}")
    print(f"  Environmental columns: {len(env_cols_present)}")
    print(f"  PFAM domains: {len(pfam_cols):,}")

    return {
        'metadata': meta_cols_present,
        'coordinates': coord_cols_present,
        'environmental': env_cols_present,
        'pfam': pfam_cols
    }

def filter_samples_with_coordinates(df, col_types):
    """Filter to samples with valid GPS coordinates."""
    print("\n" + "=" * 70)
    print("FILTERING SAMPLES WITH COORDINATES")
    print("=" * 70)

    coord_cols = col_types['coordinates']

    # Check for valid coordinates
    has_lat = df['latitude'].notna() & (df['latitude'] != 0)
    has_lon = df['longitude'].notna() & (df['longitude'] != 0)
    valid_coords = has_lat & has_lon

    # Filter to reasonable lat/lon ranges
    valid_lat = (df['latitude'] >= -90) & (df['latitude'] <= 90)
    valid_lon = (df['longitude'] >= -180) & (df['longitude'] <= 180)
    valid_range = valid_lat & valid_lon

    mask = valid_coords & valid_range

    print(f"  Total samples: {len(df):,}")
    print(f"  With valid coordinates: {mask.sum():,}")
    print(f"  Removed: {(~mask).sum():,}")

    if mask.sum() < MIN_SAMPLES_WITH_COORDS:
        print(f"  WARNING: Only {mask.sum()} samples with coordinates (< {MIN_SAMPLES_WITH_COORDS})")

    return df[mask].copy(), mask

def filter_pfam_by_prevalence(df, pfam_cols, threshold=PREVALENCE_THRESHOLD):
    """Filter PFAM domains by prevalence threshold."""
    print("\n" + "=" * 70)
    print("FILTERING PFAM BY PREVALENCE")
    print("=" * 70)

    print(f"  Threshold: >= {threshold*100:.1f}% of samples")
    print(f"  Min samples: {int(threshold * len(df)):,}")

    # Calculate prevalence (fraction of samples where domain count > 0)
    pfam_data = df[pfam_cols].values.astype(float)
    prevalence = (pfam_data > 0).mean(axis=0)

    # Apply filter
    mask = prevalence >= threshold
    passing_cols = [pfam_cols[i] for i in range(len(pfam_cols)) if mask[i]]

    print(f"  Original PFAM domains: {len(pfam_cols):,}")
    print(f"  Passing filter: {len(passing_cols):,} ({100*len(passing_cols)/len(pfam_cols):.1f}%)")
    print(f"  Removed: {len(pfam_cols) - len(passing_cols):,}")

    # Prevalence statistics
    print(f"\n  Prevalence distribution:")
    print(f"    Min: {prevalence.min()*100:.2f}%")
    print(f"    Max: {prevalence.max()*100:.2f}%")
    print(f"    Mean: {prevalence.mean()*100:.2f}%")
    print(f"    Median: {np.median(prevalence)*100:.2f}%")

    return passing_cols, prevalence, mask

def prepare_environmental_matrix(df, env_cols):
    """Prepare environmental matrix with missing value handling."""
    print("\n" + "=" * 70)
    print("PREPARING ENVIRONMENTAL MATRIX")
    print("=" * 70)

    env_data = df[env_cols].copy()

    # Convert to numeric, coercing errors
    for col in env_cols:
        env_data[col] = pd.to_numeric(env_data[col], errors='coerce')

    # Report missing values
    missing = env_data.isna().sum()
    missing_pct = (missing / len(env_data) * 100).round(1)

    print(f"  Environmental variables: {len(env_cols)}")
    print(f"\n  Missing value summary:")
    for col, pct in sorted(missing_pct.items(), key=lambda x: -x[1])[:10]:
        if pct > 0:
            print(f"    {col}: {pct}%")

    # Fill missing with median
    env_filled = env_data.fillna(env_data.median())

    # Standardize (z-score)
    env_mean = env_filled.mean()
    env_std = env_filled.std()
    env_std = env_std.replace(0, 1)  # Avoid division by zero
    env_standardized = (env_filled - env_mean) / env_std

    print(f"\n  Applied z-score standardization")

    return env_standardized.values, env_cols

def prepare_pfam_matrix(df, pfam_cols):
    """Prepare PFAM matrix with CLR transformation."""
    print("\n" + "=" * 70)
    print("PREPARING PFAM MATRIX")
    print("=" * 70)

    pfam_data = df[pfam_cols].values.astype(float)

    print(f"  PFAM domains: {len(pfam_cols):,}")
    print(f"  Samples: {pfam_data.shape[0]:,}")

    # Replace NaN with 0
    pfam_data = np.nan_to_num(pfam_data, nan=0.0)

    # Summary statistics
    total_counts = pfam_data.sum(axis=1)
    print(f"\n  Total counts per sample:")
    print(f"    Min: {total_counts.min():.0f}")
    print(f"    Max: {total_counts.max():.0f}")
    print(f"    Mean: {total_counts.mean():.0f}")
    print(f"    Median: {np.median(total_counts):.0f}")

    # Apply CLR transformation
    print(f"\n  Applying CLR transformation...")
    pfam_clr = apply_clr(pfam_data)

    print(f"  CLR range: [{pfam_clr.min():.3f}, {pfam_clr.max():.3f}]")

    return pfam_data, pfam_clr, pfam_cols

def apply_clr(counts):
    """
    Apply Centered Log-Ratio transformation.

    CLR(x) = log(x / geometric_mean(x))

    Handles zeros by adding pseudocount of 1.
    """
    # Add pseudocount
    counts_pseudo = counts + 1

    # Log transform
    log_counts = np.log(counts_pseudo)

    # Subtract geometric mean (= mean of log) per sample
    geo_mean = log_counts.mean(axis=1, keepdims=True)
    clr = log_counts - geo_mean

    return clr

def save_processed_data(df, col_types, env_matrix, pfam_raw, pfam_clr,
                        filtered_pfam_cols, env_cols):
    """Save all processed data for downstream phases."""
    print("\n" + "=" * 70)
    print("SAVING PROCESSED DATA")
    print("=" * 70)

    # Coordinates
    coords = df[col_types['coordinates']].values
    np.save(DATA_DIR / f"coordinates_{TIMESTAMP}.npy", coords)
    print(f"  Saved coordinates: {coords.shape}")

    # Sample IDs
    sample_ids = df['assembly_id'].values
    np.save(DATA_DIR / f"sample_ids_{TIMESTAMP}.npy", sample_ids)
    print(f"  Saved sample IDs: {len(sample_ids)}")

    # Environmental matrix
    np.save(DATA_DIR / f"env_matrix_{TIMESTAMP}.npy", env_matrix)
    print(f"  Saved env matrix: {env_matrix.shape}")

    # PFAM matrices (raw counts and CLR-transformed)
    np.save(DATA_DIR / f"pfam_raw_{TIMESTAMP}.npy", pfam_raw)
    np.save(DATA_DIR / f"pfam_matrix_{TIMESTAMP}.npy", pfam_clr)
    print(f"  Saved PFAM raw: {pfam_raw.shape}")
    print(f"  Saved PFAM CLR: {pfam_clr.shape}")

    # Column names
    with open(DATA_DIR / f"env_columns_{TIMESTAMP}.txt", 'w') as f:
        f.write('\n'.join(env_cols))
    print(f"  Saved env columns: {len(env_cols)}")

    with open(DATA_DIR / f"pfam_columns_{TIMESTAMP}.txt", 'w') as f:
        f.write('\n'.join(filtered_pfam_cols))
    print(f"  Saved PFAM columns: {len(filtered_pfam_cols)}")

    # Also save as CSV for easy reloading (parquet requires pyarrow)
    # Use more efficient method to avoid fragmentation warning
    try:
        pfam_df = pd.DataFrame(pfam_clr, columns=filtered_pfam_cols)
        processed_df = pd.concat([
            df[['assembly_id'] + col_types['coordinates'] + env_cols].reset_index(drop=True),
            pfam_df
        ], axis=1)
        processed_df.to_parquet(DATA_DIR / f"processed_env_pfam_{TIMESTAMP}.parquet")
        print(f"  Saved parquet: processed_env_pfam_{TIMESTAMP}.parquet")
    except ImportError:
        print(f"  Skipped parquet (pyarrow not installed) - numpy arrays are sufficient")

    return True

def save_qc_report(df, col_types, filtered_pfam_cols, prevalence, env_cols):
    """Save QC summary report."""
    print("\n" + "=" * 70)
    print("SAVING QC REPORT")
    print("=" * 70)

    qc_report = {
        'timestamp': TIMESTAMP,
        'input_file': str(INPUT_FILE),
        'samples': {
            'total': int(len(df)),
            'with_coordinates': int(len(df)),
        },
        'pfam': {
            'total_domains': int(len(col_types['pfam'])),
            'after_prevalence_filter': int(len(filtered_pfam_cols)),
            'prevalence_threshold': float(PREVALENCE_THRESHOLD),
            'prevalence_stats': {
                'min': float(prevalence.min()),
                'max': float(prevalence.max()),
                'mean': float(prevalence.mean()),
                'median': float(np.median(prevalence)),
            }
        },
        'environmental': {
            'n_variables': int(len(env_cols)),
            'variables': env_cols
        },
        'output_files': {
            'coordinates': f"coordinates_{TIMESTAMP}.npy",
            'sample_ids': f"sample_ids_{TIMESTAMP}.npy",
            'env_matrix': f"env_matrix_{TIMESTAMP}.npy",
            'pfam_raw': f"pfam_raw_{TIMESTAMP}.npy",
            'pfam_clr': f"pfam_matrix_{TIMESTAMP}.npy",
            'parquet': f"processed_env_pfam_{TIMESTAMP}.parquet"
        }
    }

    with open(REPORTS_DIR / f"qc_report_{TIMESTAMP}.json", 'w') as f:
        json.dump(qc_report, f, indent=2)

    print(f"  Saved: qc_report_{TIMESTAMP}.json")

    return qc_report

def main():
    print("=" * 70)
    print("PHASE 1: DATA LOADING AND QC (algaGPT)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load raw data
    df = load_raw_data()

    # Identify column types
    col_types = identify_column_types(df)

    # Filter to samples with valid coordinates
    df_filtered, coord_mask = filter_samples_with_coordinates(df, col_types)

    # Filter PFAM by prevalence
    filtered_pfam_cols, prevalence, pfam_mask = filter_pfam_by_prevalence(
        df_filtered, col_types['pfam'], PREVALENCE_THRESHOLD
    )

    # Prepare environmental matrix
    env_matrix, env_cols = prepare_environmental_matrix(
        df_filtered, col_types['environmental']
    )

    # Prepare PFAM matrices
    pfam_raw, pfam_clr, _ = prepare_pfam_matrix(df_filtered, filtered_pfam_cols)

    # Save processed data
    save_processed_data(
        df_filtered, col_types, env_matrix, pfam_raw, pfam_clr,
        filtered_pfam_cols, env_cols
    )

    # Save QC report
    qc_report = save_qc_report(
        df_filtered, col_types, filtered_pfam_cols, prevalence, env_cols
    )

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Samples: {len(df_filtered):,}")
    print(f"  PFAM domains (filtered): {len(filtered_pfam_cols):,}")
    print(f"  Environmental variables: {len(env_cols)}")
    print(f"\nOutput directory: {DATA_DIR}")

    return df_filtered, env_matrix, pfam_clr

if __name__ == "__main__":
    df, env_matrix, pfam_clr = main()
