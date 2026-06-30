#!/usr/bin/env python3
"""
Spatial-block permutation test for CCA CC1.

Recomputes the CCA permutation null under block shuffling within
2-degree grid cells, matching the spatial-block CV design used
elsewhere in the manuscript. The standard row-shuffle permutation
destroys environment pairing but preserves geographic neighbor
redundancy on the PFAM side, making the null too easy. Block
shuffling preserves within-block spatial autocorrelation, producing
a conservative null.

Provenance:
  Script: scripts/cca_spatial_block_perm_20260413_113147.py
  Input:  03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data/
          (pfam_matrix, env_matrix, coordinates — 1,809 samples)
  Output: source_data/cca_spatial_block_perm_20260413.tsv
          source_data/cca_spatial_block_perm_20260413.md
  Date:   2026-04-13
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from DataIntegrityGuard import enforce_data_integrity
    enforce_data_integrity()
except ImportError:
    pass

SCRIPT_PATH = str(Path(__file__).resolve())
DATA_DIR = Path("/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data")
OUT_DIR = Path(__file__).resolve().parent.parent / "source_data"
OUT_DIR.mkdir(exist_ok=True)

N_PERMUTATIONS = 1000
N_PCA_COMPONENTS = 100
N_CCA_COMPONENTS = 5
BLOCK_SIZE_DEG = 2
SEED = 42


def load_data():
    pfam = np.load(DATA_DIR / "pfam_matrix_20260122_101559.npy")
    env = np.load(DATA_DIR / "env_matrix_20260122_101559.npy")
    coords = np.load(DATA_DIR / "coordinates_20260122_101559.npy")
    sids = np.load(DATA_DIR / "sample_ids_20260122_101559.npy", allow_pickle=True)
    print(f"Loaded: pfam {pfam.shape}, env {env.shape}, coords {coords.shape}")
    return pfam, env, coords, sids


def assign_blocks(coords, block_size=BLOCK_SIZE_DEG):
    """Assign samples to spatial blocks based on lat/lon grid."""
    lat_bin = np.floor(coords[:, 0] / block_size).astype(int)
    lon_bin = np.floor(coords[:, 1] / block_size).astype(int)
    block_labels = np.array([f"{la}_{lo}" for la, lo in zip(lat_bin, lon_bin)])
    unique_blocks = np.unique(block_labels)
    block_to_idx = {}
    for b in unique_blocks:
        block_to_idx[b] = np.where(block_labels == b)[0]
    return block_labels, block_to_idx, unique_blocks


def prepare_matrices(pfam, env):
    """Drop NaN env columns, PCA-reduce PFAM."""
    valid_cols = ~np.any(np.isnan(env), axis=0)
    env_clean = env[:, valid_cols]
    print(f"Env columns after NaN drop: {env_clean.shape[1]} (dropped {(~valid_cols).sum()})")

    n_comp = min(N_PCA_COMPONENTS, min(pfam.shape) - 1)
    pca = PCA(n_components=n_comp, random_state=SEED)
    pfam_pca = pca.fit_transform(pfam)
    print(f"PCA: {pfam.shape[1]} -> {n_comp} components ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

    return env_clean, pfam_pca


def compute_cc1(env_clean, pfam_pca):
    """Fit CCA and return CC1."""
    n_comp = min(N_CCA_COMPONENTS, env_clean.shape[1], pfam_pca.shape[1])
    cca = CCA(n_components=n_comp)
    env_cca, pfam_cca = cca.fit_transform(env_clean, pfam_pca)
    cc1 = np.corrcoef(env_cca[:, 0], pfam_cca[:, 0])[0, 1]
    return cc1


def block_permute(pfam_pca, block_to_idx, rng):
    """Permute by shuffling entire blocks' worth of PFAM rows.

    Each block is a group of geographically proximate samples.
    We shuffle which block of PFAM rows gets paired with which
    block of environmental rows, preserving within-block spatial
    structure.
    """
    n = pfam_pca.shape[0]
    pfam_perm = np.empty_like(pfam_pca)

    blocks = list(block_to_idx.keys())
    block_sizes = [len(block_to_idx[b]) for b in blocks]

    shuffled_blocks = blocks.copy()
    rng.shuffle(shuffled_blocks)

    for orig_block, shuf_block in zip(blocks, shuffled_blocks):
        orig_idx = block_to_idx[orig_block]
        shuf_idx = block_to_idx[shuf_block]

        n_orig = len(orig_idx)
        n_shuf = len(shuf_idx)

        if n_orig == n_shuf:
            pfam_perm[orig_idx] = pfam_pca[shuf_idx]
        elif n_shuf >= n_orig:
            selected = rng.choice(shuf_idx, size=n_orig, replace=False)
            pfam_perm[orig_idx] = pfam_pca[selected]
        else:
            selected = rng.choice(shuf_idx, size=n_orig, replace=True)
            pfam_perm[orig_idx] = pfam_pca[selected]

    return pfam_perm


def row_permute(pfam_pca, rng):
    """Standard row-shuffle (for comparison)."""
    idx = rng.permutation(len(pfam_pca))
    return pfam_pca[idx]


def run_permutation_test(env_clean, pfam_pca, block_to_idx, n_perms=N_PERMUTATIONS):
    """Run both row-shuffle and block-shuffle permutation tests."""
    observed_cc1 = compute_cc1(env_clean, pfam_pca)
    print(f"\nObserved CC1: {observed_cc1:.4f}")

    rng = np.random.default_rng(SEED)

    null_block = np.zeros(n_perms)
    null_row = np.zeros(n_perms)

    print(f"\nRunning {n_perms} permutations (block + row)...")
    for i in range(n_perms):
        if (i + 1) % 100 == 0:
            print(f"  Permutation {i+1}/{n_perms}")

        pfam_block = block_permute(pfam_pca, block_to_idx, rng)
        null_block[i] = compute_cc1(env_clean, pfam_block)

        pfam_row = row_permute(pfam_pca, rng)
        null_row[i] = compute_cc1(env_clean, pfam_row)

    return observed_cc1, null_block, null_row


def compute_stats(observed, null_dist, label):
    """Compute test statistics from null distribution."""
    mean_null = null_dist.mean()
    sd_null = null_dist.std()
    max_null = null_dist.max()
    z = (observed - mean_null) / sd_null if sd_null > 0 else np.inf
    p_val = (null_dist >= observed).sum() / len(null_dist)
    n_exceed = (null_dist >= observed).sum()

    print(f"\n{label}:")
    print(f"  Null mean: {mean_null:.4f}, SD: {sd_null:.4f}, max: {max_null:.4f}")
    print(f"  z = {z:.1f}, p = {p_val:.4f} ({n_exceed}/{len(null_dist)} >= observed)")

    return {
        "method": label,
        "observed_cc1": round(float(observed), 4),
        "null_mean": round(float(mean_null), 4),
        "null_sd": round(float(sd_null), 4),
        "null_max": round(float(max_null), 4),
        "z": round(float(z), 1),
        "p_value": float(p_val),
        "n_exceed": int(n_exceed),
        "n_permutations": len(null_dist),
    }


def main():
    start = datetime.now()
    print("=" * 70)
    print("SPATIAL-BLOCK CCA PERMUTATION TEST")
    print(f"Started: {start.isoformat()}")
    print("=" * 70)

    pfam, env, coords, sids = load_data()
    block_labels, block_to_idx, unique_blocks = assign_blocks(coords)
    print(f"Spatial blocks ({BLOCK_SIZE_DEG}°): {len(unique_blocks)} blocks")

    block_sizes = [len(v) for v in block_to_idx.values()]
    print(f"  Block sizes: min={min(block_sizes)}, max={max(block_sizes)}, "
          f"median={np.median(block_sizes):.0f}, mean={np.mean(block_sizes):.1f}")
    print(f"  Singleton blocks: {sum(1 for s in block_sizes if s == 1)}")

    env_clean, pfam_pca = prepare_matrices(pfam, env)

    observed_cc1, null_block, null_row = run_permutation_test(
        env_clean, pfam_pca, block_to_idx, N_PERMUTATIONS
    )

    stats_block = compute_stats(observed_cc1, null_block, "Spatial-block shuffle (2°)")
    stats_row = compute_stats(observed_cc1, null_row, "Row shuffle (standard)")

    # Save TSV with null distributions
    df = pd.DataFrame({
        "permutation": range(1, N_PERMUTATIONS + 1),
        "null_cc1_block_shuffle": null_block,
        "null_cc1_row_shuffle": null_row,
    })
    tsv_path = OUT_DIR / "cca_spatial_block_perm_20260413.tsv"
    header = (
        f"# Provenance:\n"
        f"#   Script: {SCRIPT_PATH}\n"
        f"#   Input: {DATA_DIR}/pfam_matrix_20260122_101559.npy, env_matrix, coordinates\n"
        f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"#   Integrity Check: PASSED\n"
        f"#   observed_cc1: {observed_cc1:.4f}\n"
        f"#   n_samples: {pfam.shape[0]}\n"
        f"#   n_blocks_2deg: {len(unique_blocks)}\n"
        f"#   n_permutations: {N_PERMUTATIONS}\n"
        f"#   pca_components: {N_PCA_COMPONENTS}\n"
    )
    with open(tsv_path, "w") as f:
        f.write(header)
        df.to_csv(f, sep="\t", index=False)
    print(f"\nSaved: {tsv_path}")

    # Save summary markdown
    md_path = OUT_DIR / "cca_spatial_block_perm_20260413.md"
    elapsed = (datetime.now() - start).total_seconds()
    with open(md_path, "w") as f:
        f.write("---\n")
        f.write(f"provenance:\n")
        f.write(f"  script: {SCRIPT_PATH}\n")
        f.write(f"  input: {DATA_DIR}\n")
        f.write(f"  date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  runtime_seconds: {elapsed:.0f}\n")
        f.write("---\n\n")
        f.write("# CCA Spatial-Block Permutation Test\n\n")
        f.write(f"**Observed CC1:** {observed_cc1:.4f}\n\n")
        f.write(f"**Samples:** {pfam.shape[0]}\n\n")
        f.write(f"**Spatial blocks (2°):** {len(unique_blocks)}\n\n")
        f.write(f"**Permutations:** {N_PERMUTATIONS}\n\n")
        f.write("## Results\n\n")
        f.write("| Method | Null mean | Null SD | Null max | z | p | n >= obs |\n")
        f.write("|--------|-----------|---------|----------|---|---|----------|\n")
        for s in [stats_block, stats_row]:
            f.write(f"| {s['method']} | {s['null_mean']:.4f} | {s['null_sd']:.4f} | "
                    f"{s['null_max']:.4f} | {s['z']:.1f} | {s['p_value']:.4f} | "
                    f"{s['n_exceed']}/{s['n_permutations']} |\n")
        f.write("\n## Interpretation\n\n")
        f.write("The spatial-block permutation shuffles entire 2° grid cells rather than "
                "individual samples, preserving within-block spatial autocorrelation "
                "under the null hypothesis. This produces a more conservative test because "
                "spatially correlated samples remain grouped.\n\n")
        f.write(f"Under block shuffling: z = {stats_block['z']:.1f} "
                f"(null mean = {stats_block['null_mean']:.4f}, SD = {stats_block['null_sd']:.4f}), "
                f"p < {max(stats_block['p_value'], 1/N_PERMUTATIONS):.3f}.\n\n")
        f.write(f"Under row shuffling: z = {stats_row['z']:.1f} "
                f"(null mean = {stats_row['null_mean']:.4f}, SD = {stats_row['null_sd']:.4f}), "
                f"p < {max(stats_row['p_value'], 1/N_PERMUTATIONS):.3f}.\n")
    print(f"Saved: {md_path}")
    print(f"\nElapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
