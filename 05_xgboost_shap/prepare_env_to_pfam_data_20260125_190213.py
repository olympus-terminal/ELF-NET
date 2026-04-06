#!/usr/bin/env python3
"""
# --------------------------------------------------------------------------
Data Preparation: Environment → Pfam Prediction Model
# --------------------------------------------------------------------------

Purpose: Prepare training data for predicting Pfam domain profiles from
         environmental features (GEE variables + AlphaEarth embeddings).

Design:
  - Input:  GEE environmental vars (30 dims) + AlphaEarth embeddings (64 dims)
            = 94 dimensions total
  - Output: Pfam CLR-normalized domain profiles (~9600 dims)
  - Missing AlphaEarth = 0 (implicitly encodes "open ocean")
  - Missing GEE values = 0 (informative signal)

Creates datasets for BOTH filtering strategies:
  1. Pythia-LA4SR filtering
  2. AlgaGPT SMART filtering

Author: TARA-LA4SR Analysis Pipeline
Date: 2026-01-25
# --------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import json
import socket
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Environment Detection
# =============================================================================

def get_base_dir() -> Path:
    """Detect environment and return appropriate base directory."""
    hostname = socket.gethostname()
    if any(x in hostname for x in ['cn', 'gpu', 'dn', 'jubail', 'login', 'fast']):
        return Path("/scratch/drn2/PROJECTS/TARA-LA4SR")
    else:
        return Path("/media/drn2/External/TARA-Oceans")

# =============================================================================
# Data Loading
# =============================================================================

def load_gee_pfam_merged(filepath: Path) -> pd.DataFrame:
    """Load GEE+Pfam merged file, skipping provenance header."""
    print(f"Loading: {filepath.name}")

    # Read file, skip comment lines
    df = pd.read_csv(filepath, sep='\t', comment='#')

    print(f"  Loaded {len(df)} samples")
    return df

def load_alphaearth_embeddings(filepath: Path) -> pd.DataFrame:
    """Load AlphaEarth embeddings file."""
    print(f"Loading AlphaEarth: {filepath.name}")

    # Read file, skip comment lines
    df = pd.read_csv(filepath, sep='\t', comment='#')

    # AlphaEarth columns are A00-A63
    ae_cols = [f'A{i:02d}' for i in range(64)]

    # Check how many have complete embeddings
    has_ae = df[ae_cols].notna().all(axis=1) & (df[ae_cols] != '').all(axis=1)
    print(f"  Samples with AlphaEarth: {has_ae.sum()} / {len(df)}")

    return df

def identify_columns(df: pd.DataFrame) -> Dict[str, list]:
    """Identify GEE, Pfam, and metadata columns."""
    all_cols = df.columns.tolist()

    # Metadata columns (first ~12 columns before environmental data)
    metadata_cols = ['assembly_id', 'matched_to', 'matched_sample', 'latitude',
                     'longitude', 'dataset', 'depth_m', 'collection_date',
                     'species', 'habitat', 'gps_source', 'salinity_psu_est',
                     'gps_confidence']
    metadata_cols = [c for c in metadata_cols if c in all_cols]

    # Pfam columns (start with 'PF')
    pfam_cols = [c for c in all_cols if c.startswith('PF')]

    # GEE columns (everything else that's numeric and not metadata/pfam)
    gee_candidate_cols = [
        'air_temp_mean_c', 'air_temp_max_c', 'air_temp_min_c', 'air_temp_range_c',
        'precip_mean_mm', 'solar_rad_mj_m2', 'elevation_m', 'bathymetry_m',
        'distance_to_coast_km', 'landcover_class', 'sst_mean_c', 'sst_max_c',
        'sst_min_c', 'sst_range_c', 'chl_mean_mg_m3', 'chl_max_mg_m3',
        'chl_min_mg_m3', 'nflh_mean', 'poc_mean_mg_m3', 'modis_sst_mean_c',
        'rrs_412', 'rrs_443', 'rrs_469', 'rrs_488', 'rrs_531', 'rrs_547',
        'rrs_555', 'rrs_645', 'rrs_667', 'rrs_678',
        # WOA23 dissolved nutrients + MLD (added 2026-03-20)
        'nitrate_umol_l', 'phosphate_umol_l', 'silicate_umol_l',
        'oxygen_umol_l', 'mld_m',
    ]
    gee_cols = [c for c in gee_candidate_cols if c in all_cols]

    print(f"  Metadata columns: {len(metadata_cols)}")
    print(f"  GEE columns: {len(gee_cols)}")
    print(f"  Pfam columns: {len(pfam_cols)}")

    return {
        'metadata': metadata_cols,
        'gee': gee_cols,
        'pfam': pfam_cols
    }

# =============================================================================
# Feature Preparation
# =============================================================================

def prepare_features(df: pd.DataFrame,
                     ae_df: Optional[pd.DataFrame],
                     col_info: Dict[str, list]) -> Tuple[np.ndarray, np.ndarray, list, list, list]:
    """
    Prepare input features (GEE + AlphaEarth) and target (Pfam CLR).

    Returns:
        X: Input features (n_samples, n_env_features)
        y: Target Pfam profiles (n_samples, n_pfam_features)
        sample_ids: List of assembly IDs
        input_feature_names: Names of input features
        output_feature_names: Names of Pfam columns
    """
    sample_ids = df['assembly_id'].tolist()

    # Extract GEE features
    gee_cols = col_info['gee']
    X_gee = df[gee_cols].copy()

    # Convert to numeric, fill missing with 0
    for col in gee_cols:
        X_gee[col] = pd.to_numeric(X_gee[col], errors='coerce')
    X_gee = X_gee.fillna(0).values

    print(f"  GEE features shape: {X_gee.shape}")

    # Extract AlphaEarth features (if available)
    ae_cols = [f'A{i:02d}' for i in range(64)]

    if ae_df is not None:
        # Merge AlphaEarth by assembly_id
        ae_subset = ae_df[['assembly_id'] + ae_cols].copy()

        # Create mapping from assembly_id to AlphaEarth
        ae_dict = {}
        for _, row in ae_subset.iterrows():
            aid = row['assembly_id']
            vals = row[ae_cols].values
            # Check if all values are valid numbers
            try:
                vals = np.array([float(v) if pd.notna(v) and v != '' else 0.0 for v in vals])
            except:
                vals = np.zeros(64)
            ae_dict[aid] = vals

        # Build AlphaEarth matrix for our samples
        X_ae = np.zeros((len(sample_ids), 64))
        ae_count = 0
        for i, sid in enumerate(sample_ids):
            if sid in ae_dict:
                vals = ae_dict[sid]
                if not np.all(vals == 0):
                    X_ae[i] = vals
                    ae_count += 1

        print(f"  AlphaEarth features shape: {X_ae.shape}")
        print(f"  Samples with AlphaEarth: {ae_count} / {len(sample_ids)}")
    else:
        X_ae = np.zeros((len(sample_ids), 64))
        print(f"  AlphaEarth features: None (all zeros)")

    # Combine GEE + AlphaEarth
    X = np.hstack([X_gee, X_ae])
    input_feature_names = gee_cols + ae_cols

    print(f"  Combined input shape: {X.shape}")

    # Extract Pfam targets
    pfam_cols = col_info['pfam']
    y_raw = df[pfam_cols].copy()

    # Convert to numeric
    for col in pfam_cols:
        y_raw[col] = pd.to_numeric(y_raw[col], errors='coerce')
    y_raw = y_raw.fillna(0).values

    # Apply CLR transformation
    y = clr_transform(y_raw)

    print(f"  Pfam target shape: {y.shape}")

    return X, y, sample_ids, input_feature_names, pfam_cols

def clr_transform(X: np.ndarray, pseudo_count: float = 1e-6) -> np.ndarray:
    """Apply Centered Log-Ratio transformation."""
    X_pos = X + pseudo_count
    log_X = np.log(X_pos)
    geometric_mean = np.mean(log_X, axis=1, keepdims=True)
    return log_X - geometric_mean

# =============================================================================
# Data Splitting
# =============================================================================

def random_split(n_samples: int,
                 train_frac: float = 0.70,
                 val_frac: float = 0.15,
                 seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random train/val/test split."""
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)

    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx

# =============================================================================
# Normalization
# =============================================================================

def normalize_features(X_train: np.ndarray,
                       X_val: np.ndarray,
                       X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Z-score normalize input features using training set statistics."""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1  # Avoid division by zero

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    stats = {'mean': mean, 'std': std}
    return X_train_norm, X_val_norm, X_test_norm, stats

# =============================================================================
# Main Processing
# =============================================================================

def process_dataset(name: str,
                    gee_pfam_path: Path,
                    ae_path: Path,
                    output_dir: Path,
                    timestamp: str) -> Dict:
    """Process a single dataset (Pythia or AlgaGPT)."""

    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    # Load data
    df = load_gee_pfam_merged(gee_pfam_path)
    ae_df = load_alphaearth_embeddings(ae_path) if ae_path.exists() else None

    # Identify columns
    col_info = identify_columns(df)

    # Prepare features
    X, y, sample_ids, input_names, output_names = prepare_features(df, ae_df, col_info)

    # Random split
    train_idx, val_idx, test_idx = random_split(len(sample_ids))

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
    print(f"  Test:  {len(test_idx)}")

    # Split data
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # Normalize inputs
    X_train, X_val, X_test, norm_stats = normalize_features(X_train, X_val, X_test)

    # Get sample IDs for each split
    train_ids = [sample_ids[i] for i in train_idx]
    val_ids = [sample_ids[i] for i in val_idx]
    test_ids = [sample_ids[i] for i in test_idx]

    # Save
    output_file = output_dir / f"env_to_pfam_{name}_{timestamp}.npz"

    np.savez_compressed(
        output_file,
        # Input features (Environment)
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        # Target (Pfam CLR)
        y_train=y_train.astype(np.float32),
        y_val=y_val.astype(np.float32),
        y_test=y_test.astype(np.float32),
        # Normalization stats
        input_mean=norm_stats['mean'].astype(np.float32),
        input_std=norm_stats['std'].astype(np.float32),
        # Feature names
        input_feature_names=np.array(input_names),
        output_feature_names=np.array(output_names),
        # Sample IDs
        train_ids=np.array(train_ids),
        val_ids=np.array(val_ids),
        test_ids=np.array(test_ids),
    )

    print(f"\nSaved: {output_file}")

    # Summary stats
    stats = {
        'name': name,
        'n_samples': len(sample_ids),
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'n_input_features': X.shape[1],
        'n_gee_features': len(col_info['gee']),
        'n_alphaearth_features': 64,
        'n_output_features': y.shape[1],
        'input_file': str(gee_pfam_path),
        'output_file': str(output_file),
    }

    return stats

def main():
    """Main entry point."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("="*60)
    print("Environment → Pfam Data Preparation")
    print("="*60)
    print(f"Timestamp: {timestamp}")

    # Setup paths
    base_dir = get_base_dir()
    print(f"Base directory: {base_dir}")

    # Output directory
    output_dir = base_dir / "MANUSCRIPT" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data files
    pythia_gee_pfam = base_dir / "03_analyses/ALGAGPT-based-analyses/preprocessing_archive/gee_pfam_merged_v7_20260113_201639.tsv"
    algagpt_gee_pfam = base_dir / "03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
    alphaearth_file = base_dir / "PythiaTIfreeLA4SR_TARA/alphaearth_embeddings_gee_pfam_20260124_175558.tsv"

    # Check files exist
    for f, name in [(pythia_gee_pfam, "Pythia GEE+Pfam"),
                    (algagpt_gee_pfam, "AlgaGPT GEE+Pfam"),
                    (alphaearth_file, "AlphaEarth embeddings")]:
        if not f.exists():
            print(f"WARNING: {name} not found: {f}")

    results = []

    # Process Pythia-LA4SR dataset
    if pythia_gee_pfam.exists():
        stats = process_dataset(
            name="pythia",
            gee_pfam_path=pythia_gee_pfam,
            ae_path=alphaearth_file,
            output_dir=output_dir,
            timestamp=timestamp
        )
        results.append(stats)

    # Process AlgaGPT SMART dataset
    if algagpt_gee_pfam.exists():
        stats = process_dataset(
            name="algagpt",
            gee_pfam_path=algagpt_gee_pfam,
            ae_path=alphaearth_file,
            output_dir=output_dir,
            timestamp=timestamp
        )
        results.append(stats)

    # Save summary
    summary_file = output_dir / f"env_to_pfam_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for r in results:
        print(f"\n{r['name'].upper()}:")
        print(f"  Samples: {r['n_samples']} (train={r['n_train']}, val={r['n_val']}, test={r['n_test']})")
        print(f"  Input dims: {r['n_input_features']} (GEE={r['n_gee_features']}, AE={r['n_alphaearth_features']})")
        print(f"  Output dims: {r['n_output_features']} (Pfam CLR)")

    print(f"\nSummary saved: {summary_file}")

if __name__ == "__main__":
    main()
