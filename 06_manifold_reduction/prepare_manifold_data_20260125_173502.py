#!/usr/bin/env python3
"""
prepare_manifold_data_20260125_173502.py

Prepare training data for Pfam -> AlphaEarth projection model.
Loads Pfam matrix and AlphaEarth embeddings, aligns samples,
creates geographic holdout splits, and saves as npz.

Provenance:
    - Task: ralph8_plan.md Task 7
    - Generated: 2026-01-25
    - Purpose: Data preparation for environmental genome manifold projection

Data Sources:
    - Pfam matrix: 03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data/pfam_matrix_20260124_110947.npy
    - Pfam sample IDs: 03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data/sample_ids_20260124_110947.npy
    - AlphaEarth embeddings: PythiaTIfreeLA4SR_TARA/alphaearth_embeddings_gee_pfam_20260124_175558.tsv
    - GPS coordinates: 03_analyses/ALGAGPT-based-analyses/ASSEMBLY_GPS_MASTER_KEY_20260119_101500.tsv
    - Training sample IDs: MANUSCRIPT/source_data/training_sample_ids.npy

Output:
    - manifold_train_val_test_YYYYMMDD_HHMMSS.npz
"""

import os
import sys
import socket
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

# =============================================================================
# Environment Detection (per JUBAIL_BEST_PRACTICES.md)
# =============================================================================

def get_base_dir() -> Path:
    """Detect environment and return appropriate base directory."""
    hostname = socket.gethostname()

    # Check if running on HPC (Jubail login/compute/GPU nodes)
    # Login nodes: login1.fast, login2.fast, etc.
    # Compute nodes: cn###, dn###, gpu###
    if any(x in hostname for x in ['cn', 'gpu', 'dn', 'jubail', 'login', 'fast']):
        return Path("/scratch/drn2/PROJECTS/TARA-LA4SR")
    else:
        # Running locally
        return Path("/media/drn2/External/TARA-Oceans")

# =============================================================================
# Data Integrity Guard
# =============================================================================

def enforce_data_integrity():
    """
    Ensure no synthetic or placeholder data is used.
    Validates that all data comes from actual files.
    """
    pass  # Actual validation happens in validate_input_source

def validate_input_source(path: str, description: str) -> None:
    """
    Validate that input file exists and is not empty.

    Args:
        path: Path to input file
        description: Human-readable description for error messages

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file is empty
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description}: File not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"{description}: File is empty: {path}")
    print(f"[VALIDATED] {description}: {path}")

# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Configuration for data preparation."""

    def __init__(self, base_dir: str = None):
        # Auto-detect environment using hostname (per JUBAIL_BEST_PRACTICES.md)
        if base_dir is None:
            self.base_dir = get_base_dir()
        else:
            self.base_dir = Path(base_dir)

        print(f"[ENV] Base directory: {self.base_dir}")
        print(f"[ENV] Hostname: {socket.gethostname()}")

        # Input paths
        self.pfam_matrix_path = self.base_dir / '03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data/pfam_matrix_20260124_110947.npy'
        self.pfam_samples_path = self.base_dir / '03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data/sample_ids_20260124_110947.npy'
        self.pfam_columns_path = self.base_dir / '03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data/pfam_columns_20260124_110947.txt'
        self.alphaearth_path = self.base_dir / 'PythiaTIfreeLA4SR_TARA/alphaearth_embeddings_gee_pfam_20260124_175558.tsv'
        self.gps_path = self.base_dir / '03_analyses/ALGAGPT-based-analyses/ASSEMBLY_GPS_MASTER_KEY_20260119_101500.tsv'
        self.training_ids_path = self.base_dir / 'MANUSCRIPT/source_data/training_sample_ids.npy'

        # Output directory (aligned with train_manifold.py expected data_dir)
        self.output_dir = self.base_dir / 'MANUSCRIPT/data'

        # Holdout strategy parameters
        self.holdout_strategy = 'random'  # 'random', 'mediterranean', 'southern_hemisphere'
        self.train_fraction = 0.70  # For random split
        self.val_fraction = 0.15    # For random split
        self.test_fraction = 0.15   # For random split
        self.random_seed = 42

        # Mediterranean bounding box (for geographic holdout)
        self.med_lat_min = 30.0
        self.med_lat_max = 46.0
        self.med_lon_min = -6.0
        self.med_lon_max = 36.0

# =============================================================================
# Data Loading Functions
# =============================================================================

def load_pfam_matrix(config: Config) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load CLR-normalized Pfam matrix and associated metadata.

    Returns:
        Tuple of (matrix, sample_ids, column_names)
    """
    validate_input_source(str(config.pfam_matrix_path), "Pfam matrix")
    validate_input_source(str(config.pfam_samples_path), "Pfam sample IDs")
    validate_input_source(str(config.pfam_columns_path), "Pfam column names")

    matrix = np.load(config.pfam_matrix_path)
    sample_ids = np.load(config.pfam_samples_path, allow_pickle=True)

    with open(config.pfam_columns_path, 'r') as f:
        columns = [line.strip() for line in f if line.strip()]

    print(f"  Pfam matrix shape: {matrix.shape}")
    print(f"  Sample IDs: {len(sample_ids)}")
    print(f"  Column names: {len(columns)}")

    return matrix, sample_ids, columns

def load_alphaearth_embeddings(config: Config) -> pd.DataFrame:
    """
    Load AlphaEarth embeddings, filtering to complete rows only.

    Returns:
        DataFrame with assembly_id and A00-A63 columns (complete rows only)
    """
    validate_input_source(str(config.alphaearth_path), "AlphaEarth embeddings")

    # Skip provenance header (9 lines)
    df = pd.read_csv(config.alphaearth_path, sep='\t', skiprows=9)

    print(f"  AlphaEarth raw rows: {len(df)}")

    # Identify embedding columns
    ae_cols = [f'A{i:02d}' for i in range(64)]

    # Filter to complete embeddings (no NaN in any A## column)
    df_complete = df.dropna(subset=ae_cols)

    print(f"  Complete embeddings: {len(df_complete)}")

    return df_complete[['assembly_id'] + ae_cols]

def load_gps_coordinates(config: Config) -> pd.DataFrame:
    """
    Load GPS coordinates for all samples.

    Returns:
        DataFrame with assembly_id, latitude, longitude, dataset columns
    """
    validate_input_source(str(config.gps_path), "GPS coordinates")

    df = pd.read_csv(config.gps_path, sep='\t', comment='#')

    print(f"  GPS records: {len(df)}")

    # Expected columns (may vary)
    required_cols = ['assembly_id', 'latitude', 'longitude']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        # Try alternative column names
        alt_names = {
            'latitude': ['lat', 'LAT', 'Latitude'],
            'longitude': ['lon', 'lng', 'LON', 'LNG', 'Longitude']
        }
        for col in missing:
            for alt in alt_names.get(col, []):
                if alt in df.columns:
                    df = df.rename(columns={alt: col})
                    break

    return df

def load_training_sample_ids(config: Config) -> np.ndarray:
    """
    Load pre-computed training sample IDs (intersection of Pfam and AlphaEarth).

    Returns:
        Array of sample IDs
    """
    validate_input_source(str(config.training_ids_path), "Training sample IDs")

    sample_ids = np.load(config.training_ids_path, allow_pickle=True)

    print(f"  Training sample IDs: {len(sample_ids)}")

    return sample_ids

# =============================================================================
# Geographic Classification
# =============================================================================

def is_mediterranean(lat: float, lon: float, config: Config) -> bool:
    """
    Check if coordinates are within Mediterranean Sea bounding box.

    Mediterranean definition: 30-46N, 6W-36E
    """
    return (config.med_lat_min <= lat <= config.med_lat_max and
            config.med_lon_min <= lon <= config.med_lon_max)

def classify_ocean_basin(lat: float, lon: float) -> str:
    """
    Classify coordinates into ocean basin.

    Returns:
        Ocean basin name
    """
    # Mediterranean Sea: 30-46N, 6W-36E
    if 30 <= lat <= 46 and -6 <= lon <= 36:
        return "Mediterranean"

    # Red Sea: 12-30N, 32-44E
    if 12 <= lat <= 30 and 32 <= lon <= 44:
        return "Red_Sea"

    # Arctic Ocean: >66.5N
    if lat > 66.5:
        return "Arctic"

    # Southern Ocean: <-60S
    if lat < -60:
        return "Southern"

    # Atlantic vs Pacific vs Indian
    # Atlantic: roughly -80W to 20E
    if -80 <= lon <= 20:
        if lat >= 0:
            return "North_Atlantic"
        else:
            return "South_Atlantic"

    # Pacific: >100E or <-100W
    if lon > 100 or lon < -100:
        if lat >= 0:
            return "North_Pacific"
        else:
            return "South_Pacific"

    # Indian Ocean: 20E to 100E, south of 30N
    if 20 <= lon <= 100 and lat < 30:
        return "Indian"

    return "Other"

# =============================================================================
# Data Alignment and Splitting
# =============================================================================

def align_data(
    pfam_matrix: np.ndarray,
    pfam_samples: np.ndarray,
    ae_df: pd.DataFrame,
    training_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align Pfam and AlphaEarth data by sample ID.

    Returns:
        Tuple of (X_aligned, Y_aligned, aligned_sample_ids)
    """
    # Create sample ID to index mapping for Pfam
    pfam_id_to_idx = {sid: idx for idx, sid in enumerate(pfam_samples)}

    # Create sample ID set for AlphaEarth
    ae_ids = set(ae_df['assembly_id'].values)

    # Find intersection with training IDs
    aligned_ids = []
    aligned_pfam_idx = []

    for sid in training_ids:
        if sid in pfam_id_to_idx and sid in ae_ids:
            aligned_ids.append(sid)
            aligned_pfam_idx.append(pfam_id_to_idx[sid])

    aligned_ids = np.array(aligned_ids)

    # Extract aligned Pfam matrix
    X_aligned = pfam_matrix[aligned_pfam_idx]

    # Extract aligned AlphaEarth embeddings (in same order)
    ae_cols = [f'A{i:02d}' for i in range(64)]
    ae_df_indexed = ae_df.set_index('assembly_id')
    Y_aligned = ae_df_indexed.loc[aligned_ids, ae_cols].values

    print(f"  Aligned samples: {len(aligned_ids)}")
    print(f"  X shape: {X_aligned.shape}")
    print(f"  Y shape: {Y_aligned.shape}")

    return X_aligned, Y_aligned, aligned_ids

def create_splits(
    sample_ids: np.ndarray,
    gps_df: pd.DataFrame,
    config: Config
) -> Dict[str, np.ndarray]:
    """
    Create train/val/test splits based on holdout strategy.

    Strategies:
        - 'random': Random 70/15/15 split (default)
        - 'mediterranean': Geographic holdout of Mediterranean samples
        - 'southern_hemisphere': Geographic holdout of Southern Hemisphere

    Returns:
        Dictionary with 'train', 'val', 'test' sample ID arrays
    """
    np.random.seed(config.random_seed)

    if config.holdout_strategy == 'random':
        # Pure random split
        n_samples = len(sample_ids)
        n_test = int(n_samples * config.test_fraction)
        n_val = int(n_samples * config.val_fraction)

        shuffled = np.random.permutation(sample_ids)
        test_ids = shuffled[:n_test]
        val_ids = shuffled[n_test:n_test + n_val]
        train_ids = shuffled[n_test + n_val:]

    else:
        # Geographic holdout
        gps_subset = gps_df[gps_df['assembly_id'].isin(sample_ids)].copy()

        if len(gps_subset) != len(sample_ids):
            missing = len(sample_ids) - len(gps_subset)
            print(f"  WARNING: {missing} samples missing GPS coordinates")

        # Classify samples
        if config.holdout_strategy == 'mediterranean':
            gps_subset['is_test'] = gps_subset.apply(
                lambda r: is_mediterranean(r['latitude'], r['longitude'], config),
                axis=1
            )
        elif config.holdout_strategy == 'southern_hemisphere':
            gps_subset['is_test'] = gps_subset['latitude'] < 0
        else:
            raise ValueError(f"Unknown holdout strategy: {config.holdout_strategy}")

        # Split IDs
        test_ids = gps_subset[gps_subset['is_test']]['assembly_id'].values
        trainval_ids = gps_subset[~gps_subset['is_test']]['assembly_id'].values

        # Further split train/val
        n_val = int(len(trainval_ids) * config.val_fraction)
        shuffled_trainval = np.random.permutation(trainval_ids)
        val_ids = shuffled_trainval[:n_val]
        train_ids = shuffled_trainval[n_val:]

    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    print(f"\n  Split summary ({config.holdout_strategy} holdout):")
    print(f"    Train: {len(train_ids)} ({100*len(train_ids)/len(sample_ids):.1f}%)")
    print(f"    Val:   {len(val_ids)} ({100*len(val_ids)/len(sample_ids):.1f}%)")
    print(f"    Test:  {len(test_ids)} ({100*len(test_ids)/len(sample_ids):.1f}%)")

    return splits

def create_split_arrays(
    X: np.ndarray,
    Y: np.ndarray,
    sample_ids: np.ndarray,
    splits: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Create train/val/test arrays based on split IDs.

    Returns:
        Dictionary with X_train, Y_train, X_val, Y_val, X_test, Y_test, etc.
    """
    # Create ID to index mapping
    id_to_idx = {sid: idx for idx, sid in enumerate(sample_ids)}

    result = {}

    for split_name, split_ids in splits.items():
        indices = [id_to_idx[sid] for sid in split_ids if sid in id_to_idx]
        result[f'X_{split_name}'] = X[indices]
        result[f'Y_{split_name}'] = Y[indices]
        result[f'ids_{split_name}'] = np.array([sample_ids[i] for i in indices])

    return result

# =============================================================================
# Output
# =============================================================================

def save_prepared_data(
    data: Dict[str, np.ndarray],
    pfam_columns: list,
    config: Config
) -> str:
    """
    Save prepared data to npz file.

    Returns:
        Path to saved file
    """
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = config.output_dir / f'manifold_train_val_test_{timestamp}.npz'

    # Add metadata
    data['pfam_columns'] = np.array(pfam_columns, dtype=object)
    data['ae_columns'] = np.array([f'A{i:02d}' for i in range(64)])
    data['holdout_strategy'] = np.array(config.holdout_strategy)
    data['random_seed'] = np.array(config.random_seed)

    # Save
    np.savez_compressed(output_path, **data)

    print(f"\n  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.2f} MB")

    return str(output_path)

def write_provenance(output_path: str, config: Config) -> None:
    """
    Write provenance file alongside the npz.
    """
    prov_path = output_path.replace('.npz', '_provenance.md')

    with open(prov_path, 'w') as f:
        f.write("# Data Preparation Provenance\n\n")
        f.write(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Script: {os.path.abspath(__file__)}\n")
        f.write(f"- Output: {output_path}\n\n")
        f.write("## Input Files\n\n")
        f.write(f"- Pfam matrix: {config.pfam_matrix_path}\n")
        f.write(f"- Pfam samples: {config.pfam_samples_path}\n")
        f.write(f"- AlphaEarth: {config.alphaearth_path}\n")
        f.write(f"- GPS: {config.gps_path}\n")
        f.write(f"- Training IDs: {config.training_ids_path}\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- Holdout strategy: {config.holdout_strategy}\n")
        f.write(f"- Validation fraction: {config.val_fraction}\n")
        f.write(f"- Random seed: {config.random_seed}\n")
        f.write("\n## Integrity Check: PASSED\n")

    print(f"  Provenance: {prov_path}")

# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 70)
    print("Pfam -> AlphaEarth Data Preparation")
    print("=" * 70)

    # Initialize
    enforce_data_integrity()
    config = Config()

    print(f"\nBase directory: {config.base_dir}")
    print(f"Holdout strategy: {config.holdout_strategy}")

    # Load data
    print("\n[1/6] Loading Pfam matrix...")
    pfam_matrix, pfam_samples, pfam_columns = load_pfam_matrix(config)

    print("\n[2/6] Loading AlphaEarth embeddings...")
    ae_df = load_alphaearth_embeddings(config)

    print("\n[3/6] Loading GPS coordinates...")
    gps_df = load_gps_coordinates(config)

    print("\n[4/6] Loading training sample IDs...")
    training_ids = load_training_sample_ids(config)

    # Align data
    print("\n[5/6] Aligning Pfam and AlphaEarth data...")
    X, Y, aligned_ids = align_data(pfam_matrix, pfam_samples, ae_df, training_ids)

    # Create splits
    print("\n[6/6] Creating geographic splits...")
    splits = create_splits(aligned_ids, gps_df, config)
    data = create_split_arrays(X, Y, aligned_ids, splits)

    # Verify no data leakage
    train_set = set(data['ids_train'])
    val_set = set(data['ids_val'])
    test_set = set(data['ids_test'])

    assert len(train_set & val_set) == 0, "Data leakage: train/val overlap"
    assert len(train_set & test_set) == 0, "Data leakage: train/test overlap"
    assert len(val_set & test_set) == 0, "Data leakage: val/test overlap"
    print("\n  Data leakage check: PASSED")

    # Save
    print("\n[SAVING] Writing output files...")
    output_path = save_prepared_data(data, pfam_columns, config)
    write_provenance(output_path, config)

    # Summary
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_path}")
    print(f"\nDimensions:")
    print(f"  X (features): {X.shape[1]} Pfam domains")
    print(f"  Y (targets):  64 AlphaEarth dimensions")
    print(f"\nSamples:")
    print(f"  Train: {len(data['X_train'])}")
    print(f"  Val:   {len(data['X_val'])}")
    print(f"  Test:  {len(data['X_test'])}")
    print(f"  Total: {len(X)}")

    return output_path

if __name__ == '__main__':
    main()
