#!/usr/bin/env python3
"""
B3: Dark Proteome Abundance Matrix + CLR (same as A7)

Provenance:
    Script: scripts/novel_families/B3_dark_abundance.py
    Generated: 2026-02-21
    Pipeline: Novel Domain Discovery — Track B
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

CLR_PSEUDOCOUNT = 0.5  # match spatial_block_cv

def clr_transform(X, pseudocount=CLR_PSEUDOCOUNT):
    X_pseudo = X + pseudocount
    log_X = np.log(X_pseudo)
    geom_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geom_mean

def main():
    BASE = get_base_dir("TARA-LA4SR")
    FAMILIES_FILE = BASE / "novel_families" / "data" / "dark_families_filtered.tsv"
    MEMBERSHIP = BASE / "novel_families" / "data" / "mmseqs_clusters_dark" / "cluster_membership.tsv"
    CONTIG_MAP = BASE / "novel_families" / "data" / "contig_maps" / "protein_contig_map.tsv.gz"
    OUT_DIR = BASE / "novel_families" / "data" / "abundance_matrices"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  B3: Dark Proteome Abundance Matrix + CLR")
    print("=" * 60)

    if not FAMILIES_FILE.exists():
        print(f"  ERROR: {FAMILIES_FILE} not found. Run B2 first.")
        sys.exit(1)

    # Load families
    families = {}
    rep_to_family = {}
    with open(FAMILIES_FILE) as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            families[parts[0]] = parts[1]
            rep_to_family[parts[1]] = parts[0]

    print(f"  Families: {len(families)}")

    # Load membership
    protein_to_family = {}
    with open(MEMBERSHIP) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] in rep_to_family:
                protein_to_family[parts[1]] = rep_to_family[parts[0]]

    # Load protein → assembly
    protein_to_assembly = {}
    all_assemblies = set()
    with gzip.open(CONTIG_MAP, "rt") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in protein_to_family:
                protein_to_assembly[parts[0]] = parts[1]
                all_assemblies.add(parts[1])

    # Build matrix
    family_ids = sorted(families.keys())
    assembly_ids = sorted(all_assemblies)
    family_idx = {f: i for i, f in enumerate(family_ids)}
    assembly_idx = {a: i for i, a in enumerate(assembly_ids)}

    counts = np.zeros((len(assembly_ids), len(family_ids)), dtype=np.int64)
    for pid, fid in protein_to_family.items():
        asm = protein_to_assembly.get(pid)
        if asm and fid in family_idx:
            counts[assembly_idx[asm], family_idx[fid]] += 1

    print(f"  Matrix: {counts.shape}")

    # Write raw
    raw_path = OUT_DIR / "dark_family_counts_raw.tsv"
    with open(raw_path, "w") as f:
        f.write("assembly_id\t" + "\t".join(family_ids) + "\n")
        for i, asm in enumerate(assembly_ids):
            f.write(asm + "\t" + "\t".join(str(c) for c in counts[i]) + "\n")

    # CLR
    clr_matrix = clr_transform(counts.astype(np.float64))
    clr_path = OUT_DIR / "dark_family_counts_clr.tsv"
    with open(clr_path, "w") as f:
        f.write("assembly_id\t" + "\t".join(family_ids) + "\n")
        for i, asm in enumerate(assembly_ids):
            f.write(asm + "\t" + "\t".join(f"{v:.6f}" for v in clr_matrix[i]) + "\n")

    print(f"  Raw: {raw_path}")
    print(f"  CLR: {clr_path}")
    print(f"  Done: {__import__('datetime').datetime.now()}")

if __name__ == "__main__":
    main()
