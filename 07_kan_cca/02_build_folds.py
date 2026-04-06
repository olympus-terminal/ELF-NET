#!/usr/bin/env python3
"""
02_build_folds.py — Build spatial block cross-validation folds.

Creates 10-fold spatial block CV using 2-degree grid cells as groups.
No grid cell appears in both train and test within any fold.

Input:
  - lat_lon.npy from kan_cca output dir

Output:
  - fold_indices.json: list of 10 dicts with 'train' and 'test' index arrays
"""

import json
import socket
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_base_dir():
    scratch = Path('/scratch/drn2/PROJECTS/TARA-LA4SR')
    archive = Path('/archive/drn2/TARA-Oceans')
    return scratch, archive

BASE_DIR, ARCHIVE_DIR = get_base_dir()
OUTPUT_DIR = BASE_DIR / '03_analyses' / 'kan_cca'

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{ts}] 02_build_folds.py starting on {socket.gethostname()}")

    # 1. Load lat/lon
    lat_lon = np.load(OUTPUT_DIR / 'lat_lon.npy')
    n_samples = lat_lon.shape[0]
    print(f"Loaded lat_lon: {lat_lon.shape}")

    # 2. Create 2-degree grid blocks
    lat_grid = np.floor(lat_lon[:, 0] / 2) * 2
    lon_grid = np.floor(lat_lon[:, 1] / 2) * 2
    # Unique grid cell label: combine lat and lon into a string label
    grid_labels = np.array([f"{la:.0f}_{lo:.0f}" for la, lo in zip(lat_grid, lon_grid)])

    unique_cells = np.unique(grid_labels)
    n_cells = len(unique_cells)
    print(f"Grid cells (2-degree): {n_cells}")

    # Cell size distribution
    cell_sizes = [np.sum(grid_labels == c) for c in unique_cells]
    print(f"Cell sizes: min={min(cell_sizes)}, max={max(cell_sizes)}, "
          f"median={np.median(cell_sizes):.0f}, mean={np.mean(cell_sizes):.1f}")

    # 3. GroupKFold — 10 folds
    n_splits = 10
    gkf = GroupKFold(n_splits=n_splits)
    folds = []

    # GroupKFold needs X, y — use dummy arrays
    dummy_X = np.zeros(n_samples)
    dummy_y = np.zeros(n_samples)

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(dummy_X, dummy_y, groups=grid_labels)):
        folds.append({
            'train': train_idx.tolist(),
            'test': test_idx.tolist(),
        })
        print(f"  Fold {fold_i}: train={len(train_idx)}, test={len(test_idx)}")

    # 4. Verify no spatial leakage
    print("\nVerifying no spatial leakage...")
    leakage_found = False
    for fold_i, fold in enumerate(folds):
        train_cells = set(grid_labels[fold['train']])
        test_cells = set(grid_labels[fold['test']])
        overlap = train_cells & test_cells
        if overlap:
            print(f"  LEAKAGE in fold {fold_i}: {len(overlap)} shared cells!")
            leakage_found = True
        else:
            print(f"  Fold {fold_i}: OK — {len(train_cells)} train cells, "
                  f"{len(test_cells)} test cells, 0 overlap")

    if leakage_found:
        raise RuntimeError("Spatial leakage detected!")

    print("No spatial leakage detected.")

    # 5. Save fold indices
    out_path = OUTPUT_DIR / 'fold_indices.json'
    with open(out_path, 'w') as f:
        json.dump(folds, f)
    print(f"\nSaved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Summary statistics
    train_sizes = [len(f['train']) for f in folds]
    test_sizes = [len(f['test']) for f in folds]
    print(f"\n=== SUMMARY ===")
    print(f"Samples: {n_samples}")
    print(f"Grid cells: {n_cells}")
    print(f"Folds: {n_splits}")
    print(f"Train sizes: {min(train_sizes)}-{max(train_sizes)} "
          f"(mean {np.mean(train_sizes):.0f})")
    print(f"Test sizes: {min(test_sizes)}-{max(test_sizes)} "
          f"(mean {np.mean(test_sizes):.0f})")
    print(f"Timestamp: {ts}")

if __name__ == '__main__':
    main()
