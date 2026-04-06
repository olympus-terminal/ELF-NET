#!/usr/bin/env python3
"""
Phase 1: Data Loading and Quality Control for Environmental-PFAM Analysis.

This script loads the merged GEE+PFAM data, performs quality control,
normalizes the data, and prepares it for downstream analysis.

Provenance:
  Input: gee_pfam_merged_v7_20260113_201639.tsv
  Date: 2026-01-13
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Data integrity enforcement (CLAUDE.md requirement)
from DataIntegrityGuard import enforce_data_integrity, validate_input_source
enforce_data_integrity()

# Paths
BASE_DIR = Path("/media/drn/External1/TARA-Oceans/03_analyses/env_pfam_manifold")
DATA_DIR = BASE_DIR / "data"
INPUT_FILE = Path("/media/drn/External1/TARA-Oceans/GoogleEarthEngine/gee_pfam_merged_v7_20260113_201639.tsv")

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def load_data():
    """Load the merged GEE+PFAM data."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # Skip comment lines
    df = pd.read_csv(INPUT_FILE, sep='\t', comment='#')
    print(f"  Loaded {len(df)} samples, {len(df.columns)} columns")

    return df

def identify_column_types(df):
    """Identify metadata, environmental, and PFAM columns."""
    print("\n" + "=" * 70)
    print("IDENTIFYING COLUMN TYPES")
    print("=" * 70)

    # Metadata columns (identifiers, categorical)
    metadata_cols = ['assembly_id', 'matched_sample', 'dataset', 'depth_m',
                     'collection_date', 'species', 'habitat', 'gps_source',
                     'gps_confidence', 'landcover_class']
    metadata_cols = [c for c in metadata_cols if c in df.columns]

    # PFAM columns (start with PF)
    pfam_cols = [c for c in df.columns if c.startswith('PF')]

    # Environmental columns (everything else that's numeric)
    env_cols = [c for c in df.columns
                if c not in metadata_cols
                and c not in pfam_cols
                and c not in ['latitude', 'longitude']]

    # Coordinate columns
    coord_cols = ['latitude', 'longitude']

    print(f"  Metadata columns: {len(metadata_cols)}")
    print(f"  Coordinate columns: {len(coord_cols)}")
    print(f"  Environmental columns: {len(env_cols)}")
    print(f"  PFAM columns: {len(pfam_cols)}")

    print(f"\n  Environmental variables:")
    for col in env_cols:
        print(f"    - {col}")

    return metadata_cols, coord_cols, env_cols, pfam_cols

def clean_environmental_data(df, env_cols):
    """Clean and validate environmental data."""
    print("\n" + "=" * 70)
    print("CLEANING ENVIRONMENTAL DATA")
    print("=" * 70)

    env_stats = []
    cols_to_drop = []

    for col in env_cols:
        # Convert to numeric, coercing errors
        df[col] = pd.to_numeric(df[col], errors='coerce')

        n_valid = df[col].notna().sum()
        n_missing = df[col].isna().sum()
        pct_missing = 100 * n_missing / len(df)

        if n_valid > 0:
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
        else:
            mean_val = std_val = min_val = max_val = np.nan

        env_stats.append({
            'variable': col,
            'n_valid': n_valid,
            'n_missing': n_missing,
            'pct_missing': pct_missing,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val
        })

        if pct_missing > 50:
            print(f"  WARNING: {col} has {pct_missing:.1f}% missing values")

        # Mark columns with 100% missing for removal
        if pct_missing >= 100.0:
            cols_to_drop.append(col)
            print(f"  DROPPING: {col} (100% missing)")

    # Remove columns with 100% missing
    if cols_to_drop:
        env_cols_filtered = [c for c in env_cols if c not in cols_to_drop]
        print(f"\n  Dropped {len(cols_to_drop)} columns with 100% missing data")
    else:
        env_cols_filtered = env_cols

    env_stats_df = pd.DataFrame(env_stats)
    print(f"\n  Environmental variable summary:")
    print(env_stats_df.to_string())

    return df, env_stats_df, env_cols_filtered

def filter_pfam_domains(df, pfam_cols, min_prevalence=0.05, min_variance_pct=0.01):
    """Filter PFAM domains based on prevalence and variance."""
    print("\n" + "=" * 70)
    print("FILTERING PFAM DOMAINS")
    print("=" * 70)

    # Convert PFAM columns to numeric
    for col in pfam_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    pfam_data = df[pfam_cols].values

    # Calculate prevalence (fraction of samples with non-zero counts)
    prevalence = (pfam_data > 0).mean(axis=0)

    # Calculate variance
    variance = pfam_data.var(axis=0)
    total_variance = variance.sum()
    variance_pct = variance / total_variance

    # Filter criteria
    keep_prevalence = prevalence >= min_prevalence
    keep_variance = variance_pct >= min_variance_pct
    keep_mask = keep_prevalence | keep_variance  # Keep if either criterion met

    filtered_pfam_cols = [c for c, k in zip(pfam_cols, keep_mask) if k]

    print(f"  Original PFAM domains: {len(pfam_cols)}")
    print(f"  After prevalence filter (>={min_prevalence*100:.0f}%): {keep_prevalence.sum()}")
    print(f"  After variance filter (>={min_variance_pct*100:.2f}% var): {keep_variance.sum()}")
    print(f"  Final filtered domains: {len(filtered_pfam_cols)}")

    # Summary stats for filtered domains
    filtered_data = df[filtered_pfam_cols].values
    print(f"\n  Filtered PFAM matrix shape: {filtered_data.shape}")
    print(f"  Total counts: {filtered_data.sum():.0f}")
    print(f"  Non-zero entries: {(filtered_data > 0).sum()} ({100*(filtered_data > 0).mean():.1f}%)")

    return df, filtered_pfam_cols

def normalize_pfam_counts(df, pfam_cols, method='clr'):
    """Normalize PFAM counts using CLR or other methods."""
    print("\n" + "=" * 70)
    print(f"NORMALIZING PFAM COUNTS (method={method})")
    print("=" * 70)

    pfam_data = df[pfam_cols].values.astype(float)

    if method == 'clr':
        # Centered log-ratio transformation
        # Add pseudocount to avoid log(0)
        pseudocount = 0.5
        pfam_data_pseudo = pfam_data + pseudocount

        # Geometric mean per sample
        log_data = np.log(pfam_data_pseudo)
        geom_mean = log_data.mean(axis=1, keepdims=True)

        # CLR transform
        pfam_normalized = log_data - geom_mean

    elif method == 'tpm':
        # Transcripts per million (actually counts per million here)
        row_sums = pfam_data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        pfam_normalized = (pfam_data / row_sums) * 1e6

    elif method == 'log':
        # Simple log transformation
        pfam_normalized = np.log1p(pfam_data)

    else:
        pfam_normalized = pfam_data

    # Create normalized column names
    norm_cols = [f"{c}_norm" for c in pfam_cols]

    # Add to dataframe
    for i, col in enumerate(norm_cols):
        df[col] = pfam_normalized[:, i]

    print(f"  Normalized {len(pfam_cols)} PFAM domains")
    print(f"  Value range: [{pfam_normalized.min():.3f}, {pfam_normalized.max():.3f}]")
    print(f"  Mean: {pfam_normalized.mean():.3f}, Std: {pfam_normalized.std():.3f}")

    return df, norm_cols, pfam_normalized

def prepare_analysis_matrices(df, env_cols, pfam_norm_cols, coord_cols):
    """Prepare clean matrices for analysis."""
    print("\n" + "=" * 70)
    print("PREPARING ANALYSIS MATRICES")
    print("=" * 70)

    # Environmental matrix - fill missing with median
    env_data = df[env_cols].copy()
    for col in env_cols:
        if env_data[col].isna().any():
            median_val = env_data[col].median()
            # If all values are NaN, use 0
            if pd.isna(median_val):
                median_val = 0
            env_data[col] = env_data[col].fillna(median_val)

    # Replace any remaining inf/nan with 0
    env_data = env_data.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Standardize environmental variables
    scaler = StandardScaler()
    env_scaled = scaler.fit_transform(env_data.values)

    # Handle any NaN/inf from scaling
    env_scaled = np.nan_to_num(env_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    env_scaled_df = pd.DataFrame(env_scaled, columns=env_cols, index=df.index)

    # PFAM matrix (already normalized)
    pfam_data = df[pfam_norm_cols].values

    # Coordinates
    coords = df[coord_cols].values

    # Sample IDs
    sample_ids = df['assembly_id'].values

    print(f"  Environmental matrix: {env_scaled.shape}")
    print(f"  PFAM matrix: {pfam_data.shape}")
    print(f"  Coordinates: {coords.shape}")
    print(f"  Samples: {len(sample_ids)}")

    # Check for any remaining NaN/Inf
    env_valid = np.isfinite(env_scaled).all()
    pfam_valid = np.isfinite(pfam_data).all()
    print(f"\n  Environmental data valid: {env_valid}")
    print(f"  PFAM data valid: {pfam_valid}")

    return env_scaled_df, pfam_data, coords, sample_ids, scaler

def save_processed_data(df, env_scaled_df, pfam_data, pfam_cols, coords, sample_ids, env_stats_df, scaler):
    """Save all processed data for downstream analysis."""
    print("\n" + "=" * 70)
    print("SAVING PROCESSED DATA")
    print("=" * 70)

    # Convert problematic columns to appropriate types for parquet
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric first
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            if numeric_col.notna().sum() > 0 and numeric_col.notna().sum() == df[col].notna().sum():
                df[col] = numeric_col
            else:
                # Keep as string
                df[col] = df[col].astype(str)

    # Save processed dataframe
    output_file = DATA_DIR / f"processed_env_pfam_{TIMESTAMP}.parquet"
    df.to_parquet(output_file, index=False)
    print(f"  Saved full dataframe: {output_file}")

    # Save matrices as numpy arrays
    np.save(DATA_DIR / f"env_matrix_{TIMESTAMP}.npy", env_scaled_df.values)
    np.save(DATA_DIR / f"pfam_matrix_{TIMESTAMP}.npy", pfam_data)
    np.save(DATA_DIR / f"coordinates_{TIMESTAMP}.npy", coords)
    np.save(DATA_DIR / f"sample_ids_{TIMESTAMP}.npy", sample_ids)

    # Save column names
    with open(DATA_DIR / f"env_columns_{TIMESTAMP}.txt", 'w') as f:
        f.write('\n'.join(env_scaled_df.columns))

    with open(DATA_DIR / f"pfam_columns_{TIMESTAMP}.txt", 'w') as f:
        f.write('\n'.join(pfam_cols))

    # Save environmental stats
    env_stats_df.to_csv(DATA_DIR / f"env_stats_{TIMESTAMP}.csv", index=False)

    # Save metadata for reproducibility
    metadata = {
        'timestamp': TIMESTAMP,
        'input_file': str(INPUT_FILE),
        'n_samples': len(sample_ids),
        'n_env_vars': env_scaled_df.shape[1],
        'n_pfam_domains': pfam_data.shape[1],
        'normalization': 'clr',
        'env_scaling': 'standard'
    }

    import json
    with open(DATA_DIR / f"processing_metadata_{TIMESTAMP}.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved all matrices and metadata to {DATA_DIR}")

    return output_file

def main():
    print("=" * 70)
    print("PHASE 1: DATA LOADING AND QUALITY CONTROL")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load data
    df = load_data()

    # Identify column types
    metadata_cols, coord_cols, env_cols, pfam_cols = identify_column_types(df)

    # Clean environmental data
    df, env_stats_df, env_cols_filtered = clean_environmental_data(df, env_cols)

    # Update env_cols to filtered version
    env_cols = env_cols_filtered

    # Filter PFAM domains
    df, filtered_pfam_cols = filter_pfam_domains(df, pfam_cols)

    # Normalize PFAM counts
    df, pfam_norm_cols, pfam_normalized = normalize_pfam_counts(df, filtered_pfam_cols)

    # Prepare analysis matrices
    env_scaled_df, pfam_data, coords, sample_ids, scaler = prepare_analysis_matrices(
        df, env_cols, pfam_norm_cols, coord_cols
    )

    # Save processed data
    output_file = save_processed_data(
        df, env_scaled_df, pfam_data, filtered_pfam_cols,
        coords, sample_ids, env_stats_df, scaler
    )

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)

    return df, env_scaled_df, pfam_data, coords, sample_ids

if __name__ == "__main__":
    df, env_scaled, pfam_data, coords, sample_ids = main()
