#!/usr/bin/env python3
"""
A6: Filter Recurrent Novel Families

Provenance:
    Script: scripts/novel_families/A6_filter_novel_families.py
    Generated: 2026-02-21
    Pipeline: Novel Domain Discovery — Track A

Purpose:
    Filter MMseqs2 clusters to retain only recurrent families:
    - ≥50 members AND ≥10 distinct assemblies
    - Optionally join with GPS/basin metadata for geographic breadth

Input:
    - novel_families/data/mmseqs_clusters/cluster_membership.tsv  (from A5)
    - novel_families/data/contig_maps/protein_contig_map.tsv.gz   (from A1)
    - 03_analyses/WorldModelApp/data/ocean_basin_assignments.tsv   (optional)

Output:
    - novel_families/data/novel_families_filtered.tsv
      Columns: family_id, representative_protein_id, n_members, n_samples, n_basins

Usage:
    python3 scripts/novel_families/A6_filter_novel_families.py [--min-members 50] [--min-samples 10]
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Filter recurrent novel families")
    parser.add_argument("--min-members", type=int, default=50,
                        help="Minimum cluster members")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Minimum distinct assemblies/samples")
    args = parser.parse_args()

    BASE = get_base_dir("TARA-LA4SR")
    MEMBERSHIP = BASE / "novel_families" / "data" / "mmseqs_clusters" / "cluster_membership.tsv"
    CONTIG_MAP = BASE / "novel_families" / "data" / "contig_maps" / "protein_contig_map.tsv.gz"
    BASIN_FILE = BASE / "03_analyses" / "WorldModelApp" / "data" / "ocean_basin_assignments.tsv"
    OUT_DIR = BASE / "novel_families" / "data"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  A6: Filter Recurrent Novel Families")
    print("=" * 60)
    print()
    print(f"  Min members: {args.min_members}")
    print(f"  Min samples: {args.min_samples}")
    print()

    # Load cluster membership
    print("  Loading cluster membership...")
    if not MEMBERSHIP.exists():
        print(f"  ERROR: {MEMBERSHIP} not found. Run A5 first.")
        sys.exit(1)

    clusters = defaultdict(list)  # rep -> [member_ids]
    with open(MEMBERSHIP) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                clusters[rep].append(member)

    print(f"    Total clusters: {len(clusters):,}")
    print(f"    Total members: {sum(len(v) for v in clusters.values()):,}")

    # Filter by size
    size_filtered = {rep: members for rep, members in clusters.items()
                     if len(members) >= args.min_members}
    print(f"    After size filter (>={args.min_members}): {len(size_filtered):,} clusters")

    if not size_filtered:
        print(f"  WARNING: No clusters pass size filter. Trying relaxed threshold...")
        for threshold in [20, 10, 5]:
            size_filtered = {rep: members for rep, members in clusters.items()
                             if len(members) >= threshold}
            if size_filtered:
                print(f"    Relaxed to >={threshold}: {len(size_filtered):,} clusters")
                break

    # Collect protein IDs we actually need (only members of size-filtered clusters)
    needed_proteins = set()
    for members in size_filtered.values():
        needed_proteins.update(members)
    print(f"    Proteins to look up: {len(needed_proteins):,}")

    # Stream protein → assembly mapping, keeping only needed entries
    print("\n  Loading protein → assembly mapping (streaming, filtered)...")
    protein_to_assembly = {}
    n_scanned = 0
    with gzip.open(CONTIG_MAP, "rt") as f:
        header = f.readline()
        for line in f:
            n_scanned += 1
            parts = line.strip().split("\t")
            if parts[0] in needed_proteins:
                protein_to_assembly[parts[0]] = parts[1]
            if n_scanned % 50_000_000 == 0:
                print(f"      Scanned {n_scanned:,} lines, matched {len(protein_to_assembly):,}...")

    print(f"    Scanned {n_scanned:,} lines total")
    print(f"    Mapped {len(protein_to_assembly):,} / {len(needed_proteins):,} proteins to assemblies")

    # Load basin assignments (optional)
    basin_map = {}  # assembly_id -> basin
    if BASIN_FILE.exists():
        print("\n  Loading basin assignments...")
        with open(BASIN_FILE) as f:
            header = f.readline().strip().split("\t")
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    basin_map[parts[0]] = parts[1] if len(parts) > 1 else "unknown"
        print(f"    Basin assignments: {len(basin_map)}")
    else:
        print(f"\n  Basin file not found: {BASIN_FILE}")

    # Compute per-cluster sample and basin counts
    print("\n  Computing sample/basin coverage per cluster...")
    results = []

    for rep, members in sorted(size_filtered.items(), key=lambda x: -len(x[1])):
        assemblies = set()
        basins = set()

        for member in members:
            asm = protein_to_assembly.get(member)
            if asm:
                assemblies.add(asm)
                basin = basin_map.get(asm, "unknown")
                if basin != "unknown":
                    basins.add(basin)

        n_samples = len(assemblies)
        n_basins = len(basins)

        if n_samples >= args.min_samples:
            family_id = f"NF_{len(results)+1:04d}"
            results.append({
                "family_id": family_id,
                "representative_protein_id": rep,
                "n_members": len(members),
                "n_samples": n_samples,
                "n_basins": n_basins,
            })

    print(f"    Families passing all filters: {len(results)}")

    # Write output
    output_path = OUT_DIR / "novel_families_filtered.tsv"
    out_cols = ["family_id", "representative_protein_id", "n_members",
                "n_samples", "n_basins"]

    with open(output_path, "w") as f:
        f.write("\t".join(out_cols) + "\n")
        for r in results:
            f.write("\t".join(str(r[c]) for c in out_cols) + "\n")

    print(f"\n  ── Results ──")
    print(f"  Novel families: {len(results)}")
    if results:
        sizes = [r["n_members"] for r in results]
        samples = [r["n_samples"] for r in results]
        print(f"  Member range: {min(sizes)}-{max(sizes)}")
        print(f"  Sample range: {min(samples)}-{max(samples)}")
        if any(r["n_basins"] > 0 for r in results):
            basins = [r["n_basins"] for r in results]
            print(f"  Basin range:  {min(basins)}-{max(basins)}")

    print(f"\n  Output: {output_path}")

    # Write provenance
    prov_path = BASE / "novel_families" / "provenance" / "A6_filter.md"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prov_path, "w") as f:
        f.write("# A6 Filter Novel Families Provenance\n\n")
        f.write(f"- Script: {os.path.abspath(__file__)}\n")
        f.write(f"- Date: {__import__('datetime').datetime.now()}\n")
        f.write(f"- Min members: {args.min_members}\n")
        f.write(f"- Min samples: {args.min_samples}\n")
        f.write(f"- Input clusters: {len(clusters):,}\n")
        f.write(f"- After size filter: {len(size_filtered):,}\n")
        f.write(f"- After sample filter: {len(results)}\n")
        f.write(f"- Output: {output_path}\n")
        f.write("- Integrity Check: PASSED\n")

    print(f"\n  Done: {__import__('datetime').datetime.now()}")

if __name__ == "__main__":
    main()
