#!/usr/bin/env python3
"""
B2: Filter Dark Proteome Families (same thresholds as A6)

Provenance:
    Script: scripts/novel_families/B2_filter_dark_families.py
    Generated: 2026-02-21
    Pipeline: Novel Domain Discovery — Track B

Purpose:
    Filter MMseqs2 clusters of full dark proteome using same thresholds
    as Track A: ≥50 members, ≥10 distinct samples.

Input:
    - novel_families/data/mmseqs_clusters_dark/cluster_membership.tsv (B1)
    - novel_families/data/contig_maps/protein_contig_map.tsv.gz       (A1)

Output:
    - novel_families/data/dark_families_filtered.tsv
"""

import gzip
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

def main():
    BASE = get_base_dir("TARA-LA4SR")
    MEMBERSHIP = BASE / "novel_families" / "data" / "mmseqs_clusters_dark" / "cluster_membership.tsv"
    CONTIG_MAP = BASE / "novel_families" / "data" / "contig_maps" / "protein_contig_map.tsv.gz"
    BASIN_FILE = BASE / "03_analyses" / "WorldModelApp" / "data" / "ocean_basin_assignments.tsv"
    OUT_DIR = BASE / "novel_families" / "data"

    MIN_MEMBERS = 50
    MIN_SAMPLES = 10

    print("=" * 60)
    print("  B2: Filter Dark Proteome Families")
    print("=" * 60)
    print()

    if not MEMBERSHIP.exists():
        print(f"  ERROR: {MEMBERSHIP} not found. Run B1 first.")
        sys.exit(1)

    # Load clusters
    print("  Loading cluster membership...")
    clusters = defaultdict(list)
    with open(MEMBERSHIP) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                clusters[parts[0]].append(parts[1])

    print(f"    Total clusters: {len(clusters):,}")

    # Size filter
    size_filtered = {r: m for r, m in clusters.items() if len(m) >= MIN_MEMBERS}
    print(f"    After size filter (>={MIN_MEMBERS}): {len(size_filtered):,}")

    # Load protein → assembly
    print("  Loading protein → assembly mapping...")
    protein_to_assembly = {}
    with gzip.open(CONTIG_MAP, "rt") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            protein_to_assembly[parts[0]] = parts[1]

    # Basin map
    basin_map = {}
    if BASIN_FILE.exists():
        with open(BASIN_FILE) as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    basin_map[parts[0]] = parts[1]

    # Filter by sample count
    results = []
    for rep, members in sorted(size_filtered.items(), key=lambda x: -len(x[1])):
        assemblies = set()
        basins = set()
        for m in members:
            asm = protein_to_assembly.get(m)
            if asm:
                assemblies.add(asm)
                b = basin_map.get(asm)
                if b:
                    basins.add(b)

        if len(assemblies) >= MIN_SAMPLES:
            results.append({
                "family_id": f"DF_{len(results)+1:04d}",
                "representative_protein_id": rep,
                "n_members": len(members),
                "n_samples": len(assemblies),
                "n_basins": len(basins),
            })

    print(f"    After sample filter (>={MIN_SAMPLES}): {len(results)}")

    # Write output
    output_path = OUT_DIR / "dark_families_filtered.tsv"
    with open(output_path, "w") as f:
        cols = ["family_id", "representative_protein_id", "n_members",
                "n_samples", "n_basins"]
        f.write("\t".join(cols) + "\n")
        for r in results:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    print(f"\n  Output: {output_path}")
    print(f"  Dark families: {len(results)}")
    print(f"  Done: {__import__('datetime').datetime.now()}")

if __name__ == "__main__":
    main()
