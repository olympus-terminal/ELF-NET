#!/usr/bin/env python3
"""
B5_100mb: Export Full Cluster Membership for LM Comparison

Provenance:
    Script: scripts/novel_families/B5_export_membership_100mb.py
    Generated: 2026-02-24
    Pipeline: Novel Families — 100MB Subset Analysis

Purpose:
    Export ALL protein->family assignments from MMseqs2 clustering, including
    singletons. This is the key deliverable for comparing our clustering
    results with Kourosh's LM-based domain discovery.

    Every protein in the input FASTA gets a row. Singletons (clusters of
    size 1) are labeled "SINGLETON". Non-singleton clusters use their
    representative protein ID as the family label.

Input:
    - {subset}/clusters_{tier}/membership.tsv  (from B1_100mb)
    - Original 100MB FASTA (for completeness check)

Output:
    - {subset}/membership_{tier}pct.tsv
      Columns: protein_id, family_id, cluster_size
    - {subset}/family_summary_{tier}.tsv
      Columns: family_id, representative, n_members

Usage:
    python3 B5_export_membership_100mb.py --subset dark
    python3 B5_export_membership_100mb.py --subset annotated
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

def export_tier(subset_dir, tier, fasta_ids):
    """Export membership for one clustering tier."""
    membership_file = subset_dir / f"clusters_{tier}" / "membership.tsv"
    if not membership_file.exists():
        print(f"    WARNING: {membership_file} not found — skipping tier {tier}%")
        return

    # Load clusters: rep -> [members]
    clusters = defaultdict(list)
    with open(membership_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                clusters[parts[0]].append(parts[1])

    # Build reverse map: protein -> (rep, cluster_size)
    protein_to_cluster = {}
    for rep, members in clusters.items():
        size = len(members)
        for pid in members:
            protein_to_cluster[pid] = (rep, size)

    # Write full membership
    membership_path = subset_dir / f"membership_{tier}pct.tsv"
    n_singletons = 0
    n_clustered = 0
    n_missing = 0

    with open(membership_path, "w") as f:
        f.write("protein_id\tfamily_id\tcluster_size\n")
        for pid in sorted(fasta_ids):
            if pid in protein_to_cluster:
                rep, size = protein_to_cluster[pid]
                if size == 1:
                    f.write(f"{pid}\tSINGLETON\t1\n")
                    n_singletons += 1
                else:
                    f.write(f"{pid}\t{rep}\t{size}\n")
                    n_clustered += 1
            else:
                # Protein not in any cluster output (shouldn't happen with
                # mmseqs cluster, but handle gracefully)
                f.write(f"{pid}\tUNMAPPED\t0\n")
                n_missing += 1

    # Write family summary (non-singleton clusters)
    summary_path = subset_dir / f"family_summary_{tier}.tsv"
    with open(summary_path, "w") as f:
        f.write("family_id\trepresentative\tn_members\n")
        for rep in sorted(clusters.keys(), key=lambda r: -len(clusters[r])):
            size = len(clusters[rep])
            if size > 1:
                f.write(f"{rep}\t{rep}\t{size}\n")

    n_families = sum(1 for m in clusters.values() if len(m) > 1)

    print(f"    Tier {tier}%:")
    print(f"      Clustered proteins: {n_clustered:,}")
    print(f"      Singletons:         {n_singletons:,}")
    if n_missing:
        print(f"      Unmapped:           {n_missing:,}")
    print(f"      Non-singleton families: {n_families:,}")
    print(f"      Membership: {membership_path}")
    print(f"      Summary:    {summary_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Export full cluster membership for LM comparison")
    parser.add_argument("--subset", required=True, choices=["dark", "annotated"])
    args = parser.parse_args()

    BASE = get_base_dir("TARA-LA4SR")
    KOUROSH_DIR = BASE / "03_analyses" / "novel_domains" / "kourosh_subsets"
    SUBSET_DIR = KOUROSH_DIR / args.subset

    if args.subset == "dark":
        FASTA = KOUROSH_DIR / "dark_proteome_100mb.fa"
    else:
        FASTA = KOUROSH_DIR / "annotated_proteome_100mb.fa"

    print("=" * 60)
    print(f"  B5_100mb: Export Membership — {args.subset}")
    print("=" * 60)
    print()

    if not FASTA.exists():
        print(f"  ERROR: FASTA not found: {FASTA}")
        sys.exit(1)

    # Load all protein IDs from FASTA
    print("  Loading protein IDs from FASTA...")
    fasta_ids = set()
    with open(FASTA) as f:
        for line in f:
            if line.startswith(">"):
                pid = line[1:].strip().split()[0]
                fasta_ids.add(pid)
    print(f"    Proteins in FASTA: {len(fasta_ids):,}")
    print()

    for tier in ["30", "50"]:
        export_tier(SUBSET_DIR, tier, fasta_ids)
        print()

    print(f"  Done: {__import__('datetime').datetime.now()}")

if __name__ == "__main__":
    main()
