#!/usr/bin/env python3
"""
B0_100mb: Build protein -> assembly mapping for Kourosh 100MB subsets.

Provenance:
    Script: scripts/novel_families/B0_build_assembly_map_100mb.py
    Generated: 2026-02-24
    Pipeline: Novel Families — 100MB Subset Analysis

Purpose:
    Extract protein->assembly mapping for a given 100MB subset FASTA by
    streaming through the full protein_contig_map.tsv.gz (from A1) and
    keeping only matching protein IDs.

Input:
    - 100MB subset FASTA (dark or annotated)
    - novel_families/data/contig_maps/protein_contig_map.tsv.gz (from A1)

Output:
    - {outdir}/protein_to_assembly.tsv  (protein_id<TAB>assembly_id)

Usage:
    python3 B0_build_assembly_map_100mb.py --subset dark
    python3 B0_build_assembly_map_100mb.py --subset annotated
"""

import argparse
import gzip
import os
import sys
import time
from pathlib import Path

import socket

def get_base_dir(project_name: str) -> Path:
    hostname = socket.gethostname()
    if os.path.isdir("/scratch/drn2") or "dn" in hostname or "cn" in hostname or "gpu" in hostname or "jubail" in hostname:
        return Path(f"/scratch/drn2/PROJECTS/{project_name}")
    return Path(f"/media/drn/External1/{project_name}")

def main():
    parser = argparse.ArgumentParser(
        description="Build protein->assembly map for 100MB subsets")
    parser.add_argument("--subset", required=True, choices=["dark", "annotated"],
                        help="Which subset to process")
    args = parser.parse_args()

    BASE = get_base_dir("TARA-LA4SR")
    KOUROSH_DIR = BASE / "03_analyses" / "novel_domains" / "kourosh_subsets"
    CONTIG_MAP = BASE / "novel_families" / "data" / "contig_maps" / "protein_contig_map.tsv.gz"

    if args.subset == "dark":
        FASTA = KOUROSH_DIR / "dark_proteome_100mb.fa"
    else:
        FASTA = KOUROSH_DIR / "annotated_proteome_100mb.fa"

    OUT_DIR = KOUROSH_DIR / args.subset
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE = OUT_DIR / "protein_to_assembly.tsv"

    print("=" * 60)
    print(f"  B0_100mb: Build Assembly Map — {args.subset}")
    print("=" * 60)
    print()
    print(f"  FASTA:      {FASTA}")
    print(f"  Contig map: {CONTIG_MAP}")
    print(f"  Output:     {OUT_FILE}")
    print()

    # --- Validate inputs ---
    if not FASTA.exists():
        print(f"  ERROR: Subset FASTA not found: {FASTA}")
        sys.exit(1)

    if not CONTIG_MAP.exists():
        print(f"  ERROR: protein_contig_map.tsv.gz not found: {CONTIG_MAP}")
        print("  Run A1_build_contig_map.py first.")
        sys.exit(1)

    # --- Load protein IDs from FASTA ---
    t0 = time.time()
    print("  Loading protein IDs from FASTA...")
    protein_ids = set()
    with open(FASTA) as f:
        for line in f:
            if line.startswith(">"):
                pid = line[1:].strip().split()[0]
                protein_ids.add(pid)

    print(f"    Proteins in subset: {len(protein_ids):,}")
    print(f"    Time: {time.time() - t0:.1f}s")

    # --- Stream through contig map, keep matches ---
    t1 = time.time()
    print("  Streaming protein_contig_map.tsv.gz...")
    matched = 0
    total_lines = 0

    with gzip.open(CONTIG_MAP, "rt") as fin, open(OUT_FILE, "w") as fout:
        fout.write("protein_id\tassembly_id\n")

        header = fin.readline()  # skip header
        for line in fin:
            total_lines += 1
            parts = line.split("\t", 3)  # only need first 2 columns
            pid = parts[0]
            if pid in protein_ids:
                assembly_id = parts[1]
                fout.write(f"{pid}\t{assembly_id}\n")
                matched += 1

            if total_lines % 50_000_000 == 0:
                elapsed = time.time() - t1
                print(f"    Scanned {total_lines / 1e6:.0f}M lines, "
                      f"matched {matched:,} ({elapsed:.0f}s)")

    elapsed = time.time() - t1
    coverage = 100 * matched / len(protein_ids) if protein_ids else 0

    print()
    print(f"  Results:")
    print(f"    Lines scanned:   {total_lines:,}")
    print(f"    Proteins matched: {matched:,} / {len(protein_ids):,} ({coverage:.1f}%)")
    print(f"    Time: {elapsed:.0f}s")
    print()

    if coverage < 50:
        print(f"  WARNING: Low coverage ({coverage:.1f}%). Check that A1 contig map "
              f"covers the same assemblies as the subset.")
    elif coverage < 85:
        print(f"  NOTE: Coverage is {coverage:.1f}% — some proteins may use "
              f"non-standard headers not in the contig map.")
    else:
        print(f"  OK: Good coverage ({coverage:.1f}%)")

    print(f"\n  Output: {OUT_FILE}")
    print(f"  Done: {time.time() - t0:.0f}s total")

if __name__ == "__main__":
    main()
