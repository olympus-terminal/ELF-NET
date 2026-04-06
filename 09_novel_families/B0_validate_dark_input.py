#!/usr/bin/env python3
"""
B0: Validate Dark Proteome Input

Provenance:
    Script: scripts/novel_families/B0_validate_dark_input.py
    Generated: 2026-02-21
    Pipeline: Novel Domain Discovery — Track B

Purpose:
    Validate dark proteome FASTA from the dark proteome agent:
    - No overlap with Pfam-annotated proteins
    - Reasonable fraction (30-60% of total)
    - Assembly coverage across samples

Trigger: Run after dark proteome agent delivers its output.

Input:
    - 03_analyses/novel_domains/dark_proteome_filtered.fa  (from dark proteome pipeline)
    - novel_families/data/contig_maps/protein_annotation_status.tsv.gz  (from A2)

Output:
    - novel_families/data/track_b/dark_proteome_validated.tsv
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

    # Possible dark proteome input paths
    DARK_PATHS = [
        BASE / "03_analyses" / "novel_domains" / "dark_proteome_filtered.fa",
        BASE / "03_analyses" / "novel_domains" / "dark_proteome_concat.fa",
        BASE / "novel_families" / "data" / "track_b" / "dark_proteome_input.fasta",
    ]

    ANNOT_FILE = (BASE / "novel_families" / "data" / "contig_maps" /
                  "protein_annotation_status.tsv.gz")
    OUT_DIR = BASE / "novel_families" / "data" / "track_b"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  B0: Validate Dark Proteome Input")
    print("=" * 60)
    print()

    # Find dark proteome FASTA
    dark_path = None
    for p in DARK_PATHS:
        if p.exists():
            dark_path = p
            break

    if dark_path is None:
        print("  ERROR: No dark proteome FASTA found at any expected path:")
        for p in DARK_PATHS:
            print(f"    {p}")
        print("\n  Track B is staged — waiting for dark proteome agent output.")
        sys.exit(1)

    print(f"  Dark proteome FASTA: {dark_path}")

    # Read dark protein IDs
    print("  Reading dark protein IDs...")
    dark_ids = set()
    dark_assembly_counts = defaultdict(int)
    n_seqs = 0

    with open(dark_path) as f:
        for line in f:
            if line.startswith(">"):
                protein_id = line.strip().lstrip(">").split()[0]
                dark_ids.add(protein_id)
                n_seqs += 1
                # Try to extract assembly from protein ID
                parts = protein_id.split("_")
                if len(parts) >= 2:
                    # Heuristic: assembly might be prefix before last numeric part
                    assembly_guess = "_".join(parts[:-1])
                    dark_assembly_counts[assembly_guess] += 1

    print(f"    Dark proteins: {len(dark_ids):,}")
    print(f"    Assembly coverage (approx): {len(dark_assembly_counts):,}")

    # Load annotation status to check for overlap
    if ANNOT_FILE.exists():
        print("  Checking overlap with Pfam-annotated proteins...")
        annotated_ids = set()
        total_proteins = 0

        with gzip.open(ANNOT_FILE, "rt") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split("\t")
                total_proteins += 1
                if parts[7] == "annotated":  # annotation_status column
                    annotated_ids.add(parts[0])

        overlap = dark_ids & annotated_ids
        print(f"    Total proteins in annotation map: {total_proteins:,}")
        print(f"    Pfam-annotated: {len(annotated_ids):,}")
        print(f"    Overlap (dark ∩ annotated): {len(overlap)}")

        if len(overlap) > 0:
            pct_overlap = 100 * len(overlap) / len(dark_ids)
            print(f"    WARNING: {pct_overlap:.2f}% overlap with annotated proteins")
            if pct_overlap > 5:
                print(f"    FAIL: Overlap exceeds 5% — investigate data pipeline")
            else:
                print(f"    PASS: Overlap < 5% — acceptable")
        else:
            print(f"    PASS: No overlap with Pfam-annotated proteins")

        # Dark fraction
        dark_fraction = len(dark_ids) / total_proteins * 100
        print(f"\n    Dark fraction: {dark_fraction:.1f}%")
        if 30 <= dark_fraction <= 60:
            print(f"    PASS: Dark fraction in expected range (30-60%)")
        else:
            print(f"    WARNING: Dark fraction outside expected range")
    else:
        print("  WARNING: Annotation status file not found — skipping overlap check")

    # Write validation report
    report_path = OUT_DIR / "dark_proteome_validated.tsv"
    with open(report_path, "w") as f:
        f.write("metric\tvalue\n")
        f.write(f"dark_fasta_path\t{dark_path}\n")
        f.write(f"n_dark_proteins\t{len(dark_ids)}\n")
        f.write(f"n_assemblies_approx\t{len(dark_assembly_counts)}\n")
        if ANNOT_FILE.exists():
            f.write(f"n_total_proteins\t{total_proteins}\n")
            f.write(f"n_annotated\t{len(annotated_ids)}\n")
            f.write(f"n_overlap\t{len(overlap)}\n")
            f.write(f"dark_fraction_pct\t{dark_fraction:.2f}\n")

    print(f"\n  Report: {report_path}")
    print(f"  Done: {__import__('datetime').datetime.now()}")

if __name__ == "__main__":
    main()
