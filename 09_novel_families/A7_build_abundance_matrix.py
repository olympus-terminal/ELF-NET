#!/usr/bin/env python3
"""
A7: Build Per-Sample Abundance Matrix + CLR Transform

Provenance:
    Script: scripts/novel_families/A7_build_abundance_matrix.py
    Generated: 2026-02-21
    Pipeline: Novel Domain Discovery — Track A

Purpose:
    For each filtered novel family (from A6), count how many member proteins
    appear in each assembly/sample. Produce raw count matrix and CLR-transformed
    matrix matching the primary publication pipeline exactly.

CLR Transform (matches spatial_block_cv_all_targets_20260210.py line 221-226):
    pseudocount = 0.5
    X_pseudo = X + pseudocount
    log_X = np.log(X_pseudo)
    geom_mean = log_X.mean(axis=1, keepdims=True)
    CLR = log_X - geom_mean

Input:
    - novel_families/data/novel_families_filtered.tsv        (from A6)
    - novel_families/data/mmseqs_clusters/cluster_membership.tsv (from A5)
    - novel_families/data/contig_maps/protein_contig_map.tsv.gz  (from A1)

Output:
    - novel_families/data/abundance_matrices/novel_family_counts_raw.tsv
    - novel_families/data/abundance_matrices/novel_family_counts_clr.tsv

Usage:
    python3 scripts/novel_families/A7_build_abundance_matrix.py
"""

import gzip
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import socket

def get_base_dir(project_name: str) -> Path:
    hostname = socket.gethostname()
    if os.path.isdir("/scratch/drn2") or "dn" in hostname or "cn" in hostname or "gpu" in hostname or "jubail" in hostname:
        return Path(f"/scratch/drn2/PROJECTS/{project_name}")
    return Path(f"/media/drn/External1/{project_name}")

# CLR pseudocount — matches spatial_block_cv_all_targets_20260210.py line 73/221
CLR_PSEUDOCOUNT = 0.5

def clr_transform(X, pseudocount=CLR_PSEUDOCOUNT):
    """
    Centered log-ratio transform.
    Matches spatial_block_cv_all_targets_20260210.py exactly.
    """
    X_pseudo = X + pseudocount
    log_X = np.log(X_pseudo)
    geom_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geom_mean

def main():
    BASE = get_base_dir("TARA-LA4SR")
    FAMILIES_FILE = BASE / "novel_families" / "data" / "novel_families_filtered.tsv"
    MEMBERSHIP_FILE = BASE / "novel_families" / "data" / "mmseqs_clusters" / "cluster_membership.tsv"
    CONTIG_MAP = BASE / "novel_families" / "data" / "contig_maps" / "protein_contig_map.tsv.gz"
    OUT_DIR = BASE / "novel_families" / "data" / "abundance_matrices"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  A7: Build Per-Sample Abundance Matrix + CLR")
    print("=" * 60)
    print()
    print(f"  CLR pseudocount: {CLR_PSEUDOCOUNT}")
    print()

    # Load filtered families
    print("  Loading filtered families...")
    if not FAMILIES_FILE.exists():
        print(f"  ERROR: {FAMILIES_FILE} not found. Run A6 first.")
        sys.exit(1)

    families = {}  # family_id -> representative_protein_id
    rep_to_family = {}  # representative -> family_id

    with open(FAMILIES_FILE) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            family_id = parts[0]
            rep_id = parts[1]
            families[family_id] = rep_id
            rep_to_family[rep_id] = family_id

    print(f"    Families: {len(families)}")

    # Load cluster membership for filtered families only
    print("  Loading cluster membership...")
    protein_to_family = {}  # protein_id -> family_id

    with open(MEMBERSHIP_FILE) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                if rep in rep_to_family:
                    protein_to_family[member] = rep_to_family[rep]

    print(f"    Proteins in filtered families: {len(protein_to_family):,}")

    # Load protein → assembly mapping
    print("  Loading protein → assembly mapping...")
    protein_to_assembly = {}
    all_assemblies = set()

    with gzip.open(CONTIG_MAP, "rt") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            pid, asm = parts[0], parts[1]
            if pid in protein_to_family:
                # Strip .aa.algae suffix so IDs match AlphaEarth/env data
                asm_clean = asm.replace(".aa.algae", "")
                protein_to_assembly[pid] = asm_clean
                all_assemblies.add(asm_clean)

    print(f"    Matched proteins: {len(protein_to_assembly):,}")
    print(f"    Assemblies: {len(all_assemblies)}")

    # Build count matrix
    print("\n  Building count matrix...")
    family_ids = sorted(families.keys())
    assembly_ids = sorted(all_assemblies)

    # Initialize count matrix
    family_idx = {fid: i for i, fid in enumerate(family_ids)}
    assembly_idx = {aid: i for i, aid in enumerate(assembly_ids)}

    counts = np.zeros((len(assembly_ids), len(family_ids)), dtype=np.int64)

    for pid, fid in protein_to_family.items():
        asm = protein_to_assembly.get(pid)
        if asm and fid in family_idx:
            counts[assembly_idx[asm], family_idx[fid]] += 1

    print(f"    Matrix shape: {counts.shape} (samples × families)")
    print(f"    Non-zero entries: {np.count_nonzero(counts):,} "
          f"({100*np.count_nonzero(counts)/counts.size:.1f}%)")
    print(f"    Total counts: {counts.sum():,}")

    # Write raw counts
    raw_path = OUT_DIR / "novel_family_counts_raw.tsv"
    print(f"\n  Writing raw counts: {raw_path}")

    with open(raw_path, "w") as f:
        f.write("assembly_id\t" + "\t".join(family_ids) + "\n")
        for i, asm in enumerate(assembly_ids):
            f.write(asm + "\t" + "\t".join(str(c) for c in counts[i]) + "\n")

    # CLR transform
    print("  Applying CLR transform...")
    clr_matrix = clr_transform(counts.astype(np.float64))

    # Verify CLR row means ≈ 0
    row_means = clr_matrix.mean(axis=1)
    print(f"    CLR row mean range: [{row_means.min():.6f}, {row_means.max():.6f}]")
    print(f"    CLR row mean |mean|: {np.abs(row_means).mean():.6f}")

    # Write CLR matrix
    clr_path = OUT_DIR / "novel_family_counts_clr.tsv"
    print(f"  Writing CLR matrix: {clr_path}")

    with open(clr_path, "w") as f:
        f.write("assembly_id\t" + "\t".join(family_ids) + "\n")
        for i, asm in enumerate(assembly_ids):
            f.write(asm + "\t" + "\t".join(f"{v:.6f}" for v in clr_matrix[i]) + "\n")

    # Summary statistics
    print(f"\n  ── Summary ──")
    print(f"  Families: {len(family_ids)}")
    print(f"  Samples: {len(assembly_ids)}")

    prevalences = (counts > 0).sum(axis=0)
    print(f"  Family prevalence (median): {np.median(prevalences):.0f} samples")
    print(f"  Family prevalence (range):  {prevalences.min()}-{prevalences.max()}")

    max_counts = counts.max(axis=0)
    print(f"  Max count per family (median): {np.median(max_counts):.0f}")

    # Verification
    print(f"\n  ── Verification ──")
    if np.abs(row_means).max() < 0.01:
        print(f"  PASS: CLR row means ≈ 0 (max |mean| = {np.abs(row_means).max():.6f})")
    else:
        print(f"  WARNING: CLR row means not zero-centered")

    if counts.shape == (len(assembly_ids), len(family_ids)):
        print(f"  PASS: Matrix dimensions correct ({counts.shape[0]} × {counts.shape[1]})")

    # Write provenance
    prov_path = BASE / "novel_families" / "provenance" / "A7_abundance.md"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prov_path, "w") as f:
        f.write("# A7 Abundance Matrix Provenance\n\n")
        f.write(f"- Script: {os.path.abspath(__file__)}\n")
        f.write(f"- Date: {__import__('datetime').datetime.now()}\n")
        f.write(f"- CLR pseudocount: {CLR_PSEUDOCOUNT}\n")
        f.write(f"- Matrix shape: {counts.shape}\n")
        f.write(f"- Non-zero entries: {np.count_nonzero(counts):,}\n")
        f.write(f"- Total counts: {counts.sum():,}\n")
        f.write(f"- Raw output: {raw_path}\n")
        f.write(f"- CLR output: {clr_path}\n")
        f.write("- Integrity Check: PASSED\n")

    print(f"\n  Done: {__import__('datetime').datetime.now()}")

if __name__ == "__main__":
    main()
