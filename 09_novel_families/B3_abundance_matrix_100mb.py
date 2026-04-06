#!/usr/bin/env python3
"""
B3_100mb: Abundance Matrix + CLR for 100MB Subset Families

Provenance:
    Script: scripts/novel_families/B3_abundance_matrix_100mb.py
    Generated: 2026-02-24
    Pipeline: Novel Families — 100MB Subset Analysis

Purpose:
    Build per-assembly count matrix for filtered families and apply CLR
    transform (pseudocount=0.5). Processes both 30% and 50% tiers.

Input:
    - {subset}/families_filtered_{tier}.tsv       (from B2_100mb)
    - {subset}/clusters_{tier}/membership.tsv     (from B1_100mb)
    - {subset}/protein_to_assembly.tsv            (from B0_100mb)

Output:
    - {subset}/abundance_raw_{tier}.tsv
    - {subset}/abundance_clr_{tier}.tsv

Usage:
    python3 B3_abundance_matrix_100mb.py --subset dark
    python3 B3_abundance_matrix_100mb.py --subset annotated
"""

import argparse
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

CLR_PSEUDOCOUNT = 0.5

def clr_transform(X, pseudocount=CLR_PSEUDOCOUNT):
    """Centered log-ratio transform."""
    X_pseudo = X + pseudocount
    log_X = np.log(X_pseudo)
    geom_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geom_mean

def build_abundance(families_file, membership_file, assembly_map_file, out_dir, tier):
    """Build abundance matrix for one clustering tier."""

    # Load families (family_id -> representative)
    families = {}
    rep_to_family = {}
    with open(families_file) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            fam_id = parts[0]
            rep = parts[1]
            families[fam_id] = rep
            rep_to_family[rep] = fam_id

    if not families:
        print(f"    No families in {families_file} — skipping")
        return

    # Load membership: map each protein to its family (if rep is in filtered set)
    protein_to_family = {}
    with open(membership_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] in rep_to_family:
                protein_to_family[parts[1]] = rep_to_family[parts[0]]

    # Load protein -> assembly
    protein_to_assembly = {}
    with open(assembly_map_file) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] in protein_to_family:
                protein_to_assembly[parts[0]] = parts[1]

    # Build count matrix
    all_assemblies = sorted(set(protein_to_assembly.values()))
    family_ids = sorted(families.keys())
    family_idx = {f: i for i, f in enumerate(family_ids)}
    assembly_idx = {a: i for i, a in enumerate(all_assemblies)}

    counts = np.zeros((len(all_assemblies), len(family_ids)), dtype=np.int64)
    for pid, fid in protein_to_family.items():
        asm = protein_to_assembly.get(pid)
        if asm and fid in family_idx:
            counts[assembly_idx[asm], family_idx[fid]] += 1

    print(f"    Matrix: {counts.shape[0]} assemblies x {counts.shape[1]} families")
    print(f"    Non-zero entries: {(counts > 0).sum():,}")

    # Write raw counts
    raw_path = out_dir / f"abundance_raw_{tier}.tsv"
    with open(raw_path, "w") as f:
        f.write("assembly_id\t" + "\t".join(family_ids) + "\n")
        for i, asm in enumerate(all_assemblies):
            f.write(asm + "\t" + "\t".join(str(c) for c in counts[i]) + "\n")

    # CLR transform
    clr_matrix = clr_transform(counts.astype(np.float64))
    clr_path = out_dir / f"abundance_clr_{tier}.tsv"
    with open(clr_path, "w") as f:
        f.write("assembly_id\t" + "\t".join(family_ids) + "\n")
        for i, asm in enumerate(all_assemblies):
            f.write(asm + "\t" + "\t".join(f"{v:.6f}" for v in clr_matrix[i]) + "\n")

    print(f"    Raw: {raw_path}")
    print(f"    CLR: {clr_path}")

    return counts.shape

def main():
    parser = argparse.ArgumentParser(
        description="Build abundance matrices for 100MB subset families")
    parser.add_argument("--subset", required=True, choices=["dark", "annotated"])
    args = parser.parse_args()

    BASE = get_base_dir("TARA-LA4SR")
    KOUROSH_DIR = BASE / "03_analyses" / "novel_domains" / "kourosh_subsets"
    SUBSET_DIR = KOUROSH_DIR / args.subset

    ASSEMBLY_MAP = SUBSET_DIR / "protein_to_assembly.tsv"

    print("=" * 60)
    print(f"  B3_100mb: Abundance Matrix + CLR — {args.subset}")
    print("=" * 60)

    if not ASSEMBLY_MAP.exists():
        print(f"  ERROR: {ASSEMBLY_MAP} not found. Run B0_100mb first.")
        sys.exit(1)

    for tier in ["30", "50"]:
        families_file = SUBSET_DIR / f"families_filtered_{tier}.tsv"
        membership_file = SUBSET_DIR / f"clusters_{tier}" / "membership.tsv"

        if not families_file.exists():
            print(f"\n  Tier {tier}%: families file not found — skipping")
            continue
        if not membership_file.exists():
            print(f"\n  Tier {tier}%: membership file not found — skipping")
            continue

        print(f"\n  Processing tier {tier}%...")
        build_abundance(families_file, membership_file, ASSEMBLY_MAP,
                        SUBSET_DIR, tier)

    print(f"\n  Done: {__import__('datetime').datetime.now()}")

if __name__ == "__main__":
    main()
