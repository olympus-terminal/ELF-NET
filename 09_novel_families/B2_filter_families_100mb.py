#!/usr/bin/env python3
"""
B2_100mb: Filter Families from 100MB Subset Clusters

Provenance:
    Script: scripts/novel_families/B2_filter_families_100mb.py
    Generated: 2026-02-24
    Pipeline: Novel Families — 100MB Subset Analysis

Purpose:
    Filter MMseqs2 clusters from B1_100mb by minimum member count and
    minimum distinct assemblies. Lowered thresholds for subset scale
    (default: >=5 members, >=3 assemblies).

Input:
    - {subset}/clusters_{tier}/membership.tsv  (from B1_100mb)
    - {subset}/protein_to_assembly.tsv         (from B0_100mb)

Output:
    - {subset}/families_filtered_30.tsv
    - {subset}/families_filtered_50.tsv

Usage:
    python3 B2_filter_families_100mb.py --subset dark
    python3 B2_filter_families_100mb.py --subset annotated --min-members 10 --min-assemblies 5
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import socket

def get_base_dir(project_name: str) -> Path:
    hostname = socket.gethostname()
    if os.path.isdir("/scratch/drn2") or "dn" in hostname or "cn" in hostname or "gpu" in hostname or "jubail" in hostname:
        return Path(f"/scratch/drn2/PROJECTS/{project_name}")
    return Path(f"/media/drn/External1/{project_name}")

def filter_clusters(membership_file, protein_to_assembly, min_members, min_assemblies,
                    family_prefix):
    """Filter clusters by size and assembly diversity."""
    # Load clusters: rep -> [member1, member2, ...]
    clusters = defaultdict(list)
    with open(membership_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                clusters[parts[0]].append(parts[1])

    total_clusters = len(clusters)
    total_proteins = sum(len(m) for m in clusters.values())

    # Size filter
    size_ok = {r: m for r, m in clusters.items() if len(m) >= min_members}

    # Assembly diversity filter
    results = []
    for rep, members in sorted(size_ok.items(), key=lambda x: -len(x[1])):
        assemblies = set()
        for pid in members:
            asm = protein_to_assembly.get(pid)
            if asm:
                assemblies.add(asm)

        if len(assemblies) >= min_assemblies:
            results.append({
                "family_id": f"{family_prefix}_{len(results)+1:04d}",
                "representative": rep,
                "n_members": len(members),
                "n_assemblies": len(assemblies),
            })

    return results, total_clusters, total_proteins

def main():
    parser = argparse.ArgumentParser(
        description="Filter 100MB subset clusters into families")
    parser.add_argument("--subset", required=True, choices=["dark", "annotated"])
    parser.add_argument("--min-members", type=int, default=5,
                        help="Minimum cluster members (default: 5)")
    parser.add_argument("--min-assemblies", type=int, default=3,
                        help="Minimum distinct assemblies (default: 3)")
    args = parser.parse_args()

    BASE = get_base_dir("TARA-LA4SR")
    KOUROSH_DIR = BASE / "03_analyses" / "novel_domains" / "kourosh_subsets"
    SUBSET_DIR = KOUROSH_DIR / args.subset

    ASSEMBLY_MAP = SUBSET_DIR / "protein_to_assembly.tsv"

    print("=" * 60)
    print(f"  B2_100mb: Filter Families — {args.subset}")
    print("=" * 60)
    print()
    print(f"  Min members:    {args.min_members}")
    print(f"  Min assemblies: {args.min_assemblies}")
    print()

    if not ASSEMBLY_MAP.exists():
        print(f"  ERROR: {ASSEMBLY_MAP} not found. Run B0_100mb first.")
        sys.exit(1)

    # Load protein -> assembly mapping
    print("  Loading protein -> assembly map...")
    protein_to_assembly = {}
    with open(ASSEMBLY_MAP) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                protein_to_assembly[parts[0]] = parts[1]
    print(f"    Mapped proteins: {len(protein_to_assembly):,}")

    # Process both clustering tiers
    for tier in ["30", "50"]:
        membership_file = SUBSET_DIR / f"clusters_{tier}" / "membership.tsv"
        if not membership_file.exists():
            print(f"\n  WARNING: {membership_file} not found — skipping tier {tier}%")
            continue

        prefix = f"SF{tier}" if args.subset == "dark" else f"AF{tier}"
        results, n_clusters, n_proteins = filter_clusters(
            membership_file, protein_to_assembly,
            args.min_members, args.min_assemblies, prefix)

        # Write output
        out_file = SUBSET_DIR / f"families_filtered_{tier}.tsv"
        with open(out_file, "w") as f:
            cols = ["family_id", "representative", "n_members", "n_assemblies"]
            f.write("\t".join(cols) + "\n")
            for r in results:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")

        print(f"\n  Tier {tier}%:")
        print(f"    Total clusters:  {n_clusters:,} ({n_proteins:,} proteins)")
        print(f"    Families passing filters: {len(results)}")
        if results:
            sizes = [r["n_members"] for r in results]
            print(f"    Largest family: {max(sizes)} members")
            print(f"    Median family:  {sorted(sizes)[len(sizes)//2]} members")
        print(f"    Output: {out_file}")

    print(f"\n  Done: {__import__('datetime').datetime.now()}")

if __name__ == "__main__":
    main()
