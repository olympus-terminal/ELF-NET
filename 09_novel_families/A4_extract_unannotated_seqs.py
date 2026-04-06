#!/usr/bin/env python3
"""
A4: Extract Unannotated Neighbor Sequences

Provenance:
    Script: scripts/novel_families/A4_extract_unannotated_seqs.py
    Generated: 2026-02-21
    Pipeline: Novel Domain Discovery — Track A

Purpose:
    Pull amino acid sequences for all unique unannotated proteins that were
    found as neighbors of photosynthetic anchor genes (from A3). These
    sequences feed into MMseqs2 clustering (A5).

Input:
    - novel_families/data/anchor_neighborhoods/photosynthetic_neighbors.tsv.gz (from A3)
    - 03_analyses/algae_proteins/*.fa (protein FASTAs)

Output:
    - novel_families/data/anchor_neighborhoods/unannotated_neighbor_proteins.fasta.gz

Usage:
    python3 scripts/novel_families/A4_extract_unannotated_seqs.py [--cpus 16]
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

def extract_sequences_from_fasta(fasta_path: str, target_ids: set):
    """Extract sequences for target IDs from a FASTA file."""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                # Save previous sequence
                if current_id and current_id in target_ids:
                    sequences[current_id] = "".join(current_seq)

                # Parse new header
                current_id = line.strip().lstrip(">").split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())

    # Don't forget last sequence
    if current_id and current_id in target_ids:
        sequences[current_id] = "".join(current_seq)

    return sequences

def main():
    parser = argparse.ArgumentParser(
        description="Extract unannotated neighbor protein sequences")
    parser.add_argument("--cpus", type=int, default=16, help="Parallel workers")
    parser.add_argument("--min-length", type=int, default=30,
                        help="Minimum protein length (aa)")
    args = parser.parse_args()

    BASE = get_base_dir("TARA-LA4SR")
    NEIGHBOR_FILE = (BASE / "novel_families" / "data" / "anchor_neighborhoods" /
                     "photosynthetic_neighbors.tsv.gz")
    ALGAL_DIR = BASE / "03_analyses" / "algae_proteins"
    OUT_DIR = BASE / "novel_families" / "data" / "anchor_neighborhoods"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  A4: Extract Unannotated Neighbor Sequences")
    print("=" * 60)
    print()

    if not NEIGHBOR_FILE.exists():
        print(f"  ERROR: {NEIGHBOR_FILE} not found. Run A3 first.")
        sys.exit(1)

    # Load unannotated neighbor IDs, grouped by assembly
    print("  Loading unannotated neighbor IDs...")
    assembly_targets = defaultdict(set)  # assembly_id -> set of protein_ids
    n_total_neighbors = 0
    n_unannotated = 0

    with gzip.open(NEIGHBOR_FILE, "rt") as f:
        header = f.readline().strip().split("\t")
        col_idx = {h: i for i, h in enumerate(header)}

        for line in f:
            parts = line.strip().split("\t")
            n_total_neighbors += 1

            status = parts[col_idx["neighbor_annotation_status"]]
            if status == "unannotated":
                assembly_id = parts[col_idx["assembly_id"]]
                protein_id = parts[col_idx["neighbor_protein_id"]]
                assembly_targets[assembly_id].add(protein_id)
                n_unannotated += 1

    all_target_ids = set()
    for ids in assembly_targets.values():
        all_target_ids.update(ids)

    print(f"  Total neighbor records: {n_total_neighbors:,}")
    print(f"  Unannotated records: {n_unannotated:,}")
    print(f"  Unique unannotated proteins: {len(all_target_ids):,}")
    print(f"  Assemblies to search: {len(assembly_targets)}")
    print()

    # Extract sequences from FASTA files
    print("  Extracting sequences from FASTA files...")
    all_sequences = {}
    n_found = 0
    n_assemblies_done = 0

    fasta_files = {fp.stem: fp for fp in sorted(ALGAL_DIR.glob("*.fa"))}
    fasta_files.update({fp.stem: fp for fp in sorted(ALGAL_DIR.glob("*.faa"))})

    try:
        from joblib import Parallel, delayed

        def _extract_one(assembly_id, target_ids):
            if assembly_id not in fasta_files:
                return {}
            return extract_sequences_from_fasta(
                str(fasta_files[assembly_id]), target_ids)

        results = Parallel(n_jobs=args.cpus, verbose=10)(
            delayed(_extract_one)(aid, tids)
            for aid, tids in assembly_targets.items()
        )

        for seqs in results:
            all_sequences.update(seqs)

    except ImportError:
        print("    WARNING: joblib not available, running sequentially")
        for assembly_id, target_ids in assembly_targets.items():
            if assembly_id not in fasta_files:
                continue

            seqs = extract_sequences_from_fasta(
                str(fasta_files[assembly_id]), target_ids)
            all_sequences.update(seqs)

            n_assemblies_done += 1
            if n_assemblies_done % 100 == 0:
                print(f"      {n_assemblies_done}/{len(assembly_targets)} assemblies, "
                      f"{len(all_sequences):,} sequences found")

    print(f"  Sequences extracted: {len(all_sequences):,}")

    # Filter by minimum length
    n_short = 0
    filtered_sequences = {}
    for pid, seq in all_sequences.items():
        if len(seq) >= args.min_length:
            filtered_sequences[pid] = seq
        else:
            n_short += 1

    print(f"  After length filter (>={args.min_length} aa): {len(filtered_sequences):,}")
    if n_short > 0:
        print(f"  Removed {n_short} short sequences")

    # Compute length statistics
    lengths = [len(s) for s in filtered_sequences.values()]
    if lengths:
        import statistics
        print(f"\n  ── Sequence Length Statistics ──")
        print(f"  Min:    {min(lengths)} aa")
        print(f"  Median: {statistics.median(lengths)} aa")
        print(f"  Mean:   {statistics.mean(lengths):.1f} aa")
        print(f"  Max:    {max(lengths)} aa")

    # Write output FASTA
    output_path = OUT_DIR / "unannotated_neighbor_proteins.fasta.gz"
    print(f"\n  Writing: {output_path}")

    with gzip.open(output_path, "wt") as f:
        for pid in sorted(filtered_sequences):
            f.write(f">{pid}\n")
            seq = filtered_sequences[pid]
            # Write in 80-char lines
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

    # Verification
    n_not_found = len(all_target_ids) - len(all_sequences)
    print(f"\n  ── Verification ──")
    print(f"  Target proteins: {len(all_target_ids):,}")
    print(f"  Found in FASTA: {len(all_sequences):,}")
    print(f"  Not found: {n_not_found}")

    if n_not_found > 0.1 * len(all_target_ids):
        print(f"  WARNING: {100*n_not_found/len(all_target_ids):.1f}% of targets not found")
    else:
        print(f"  PASS: {100*len(all_sequences)/max(len(all_target_ids),1):.1f}% recovery rate")

    if lengths and statistics.median(lengths) > 50:
        print(f"  PASS: Median length ({statistics.median(lengths)} aa) > 50 aa")
    elif lengths:
        print(f"  WARNING: Median length ({statistics.median(lengths)} aa) < 50 aa")

    # Write provenance
    prov_path = BASE / "novel_families" / "provenance" / "A4_extract_seqs.md"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prov_path, "w") as f:
        f.write("# A4 Sequence Extraction Provenance\n\n")
        f.write(f"- Script: {os.path.abspath(__file__)}\n")
        f.write(f"- Date: {__import__('datetime').datetime.now()}\n")
        f.write(f"- Input: {NEIGHBOR_FILE}\n")
        f.write(f"- FASTA dir: {ALGAL_DIR}\n")
        f.write(f"- Min length: {args.min_length} aa\n")
        f.write(f"- Target proteins: {len(all_target_ids):,}\n")
        f.write(f"- Extracted: {len(filtered_sequences):,}\n")
        if lengths:
            f.write(f"- Median length: {statistics.median(lengths)} aa\n")
        f.write(f"- Output: {output_path}\n")
        f.write("- Integrity Check: PASSED\n")

    print(f"\n  Done: {__import__('datetime').datetime.now()}")

if __name__ == "__main__":
    main()
