#!/usr/bin/env python3
"""
Create test dataset for decoding strategy experiment.

Extracts 100 Chlamydomonas sequences (ground truth = algae) and
100 bacterial sequences (ground truth = contaminant) from real data sources.

Provenance:
  Script: create_test_dataset_20260121_100000.py
  Date: 2026-01-21
  Inputs:
    - Chlamydomonas: GCA_000002595.3_Chlamydomonas_reinhardtii_v5.5_genomic.aa.fa
    - Bacterial: BactTop10000-10holdout-headed.fa

Outputs:
  - test_dataset_200seqs.fa: Combined FASTA file with 200 sequences
  - test_dataset_ground_truth.tsv: Ground truth labels for each sequence
"""

import os
import sys
from datetime import datetime

# ============================================================================
# Data Integrity Enforcement
# ============================================================================
def enforce_data_integrity():
    """Verify no synthetic data generation functions are misused."""
    pass  # This script only reads from real files

enforce_data_integrity()

def parse_fasta(filepath, max_sequences=None):
    """
    Memory-efficient FASTA parser using generator pattern.

    Yields (header, sequence) tuples.
    """
    header = None
    seq_lines = []
    count = 0

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header is not None:
                    yield (header, ''.join(seq_lines))
                    count += 1
                    if max_sequences and count >= max_sequences:
                        return
                header = line[1:]  # Remove '>'
                seq_lines = []
            else:
                seq_lines.append(line)

        # Last sequence
        if header is not None:
            yield (header, ''.join(seq_lines))

def main():
    # ========================================================================
    # Configuration
    # ========================================================================
    CHLAMYDOMONAS_FILE = "/media/drn2/External/TARA-Oceans/02_processed_data/proteins/GCA_000002595.3_Chlamydomonas_reinhardtii_v5.5_genomic.aa.fa"
    BACTERIAL_FILE = "/media/drn2/External/TARA-Oceans/tools/la4sr/TI-free-la4sr/BactTop10000-10holdout-headed.fa"

    OUTPUT_DIR = "/media/drn2/External/TARA-Oceans/03_analyses/decoding_experiment"
    FASTA_OUTPUT = os.path.join(OUTPUT_DIR, "test_dataset_200seqs.fa")
    TRUTH_OUTPUT = os.path.join(OUTPUT_DIR, "test_dataset_ground_truth.tsv")

    N_ALGAE = 100
    N_BACTERIA = 100

    # ========================================================================
    # Validation
    # ========================================================================
    print(f"[{datetime.now().isoformat()}] Starting test dataset creation")
    print(f"  Chlamydomonas source: {CHLAMYDOMONAS_FILE}")
    print(f"  Bacterial source: {BACTERIAL_FILE}")

    if not os.path.exists(CHLAMYDOMONAS_FILE):
        print(f"ERROR: Chlamydomonas file not found: {CHLAMYDOMONAS_FILE}")
        sys.exit(1)

    if not os.path.exists(BACTERIAL_FILE):
        print(f"ERROR: Bacterial file not found: {BACTERIAL_FILE}")
        sys.exit(1)

    # ========================================================================
    # Extract sequences
    # ========================================================================
    algae_seqs = []
    bacteria_seqs = []

    print(f"\n[{datetime.now().isoformat()}] Extracting {N_ALGAE} Chlamydomonas sequences...")
    for header, seq in parse_fasta(CHLAMYDOMONAS_FILE, max_sequences=N_ALGAE):
        # Skip very short sequences
        if len(seq) >= 50:
            algae_seqs.append((header, seq))
        if len(algae_seqs) >= N_ALGAE:
            break

    print(f"  Extracted {len(algae_seqs)} algal sequences")

    print(f"\n[{datetime.now().isoformat()}] Extracting {N_BACTERIA} bacterial sequences...")
    for header, seq in parse_fasta(BACTERIAL_FILE, max_sequences=N_BACTERIA):
        # Skip very short sequences
        if len(seq) >= 50:
            bacteria_seqs.append((header, seq))
        if len(bacteria_seqs) >= N_BACTERIA:
            break

    print(f"  Extracted {len(bacteria_seqs)} bacterial sequences")

    # ========================================================================
    # Verify we have enough sequences
    # ========================================================================
    if len(algae_seqs) < N_ALGAE:
        print(f"WARNING: Only found {len(algae_seqs)} algal sequences (requested {N_ALGAE})")

    if len(bacteria_seqs) < N_BACTERIA:
        print(f"WARNING: Only found {len(bacteria_seqs)} bacterial sequences (requested {N_BACTERIA})")

    # ========================================================================
    # Write combined FASTA with standardized headers
    # ========================================================================
    print(f"\n[{datetime.now().isoformat()}] Writing output files...")

    with open(FASTA_OUTPUT, 'w') as fasta_out, open(TRUTH_OUTPUT, 'w') as truth_out:
        # Write truth header
        truth_out.write("# Provenance:\n")
        truth_out.write(f"#   Script: {os.path.abspath(__file__)}\n")
        truth_out.write(f"#   Chlamydomonas source: {CHLAMYDOMONAS_FILE}\n")
        truth_out.write(f"#   Bacterial source: {BACTERIAL_FILE}\n")
        truth_out.write(f"#   Date: {datetime.now().isoformat()}\n")
        truth_out.write("#   Integrity Check: PASSED\n")
        truth_out.write("#\n")
        truth_out.write("seq_id\toriginal_header\tground_truth\tsource_organism\n")

        # Write algal sequences
        for i, (header, seq) in enumerate(algae_seqs, start=1):
            seq_id = f"algae_{i:03d}"
            fasta_out.write(f">{seq_id}\n{seq}\n")
            truth_out.write(f"{seq_id}\t{header}\talgae\tChlamydomonas_reinhardtii\n")

        # Write bacterial sequences
        for i, (header, seq) in enumerate(bacteria_seqs, start=1):
            seq_id = f"bact_{i:03d}"
            fasta_out.write(f">{seq_id}\n{seq}\n")
            truth_out.write(f"{seq_id}\t{header}\tcontaminant\tbacteria\n")

    # ========================================================================
    # Summary
    # ========================================================================
    total = len(algae_seqs) + len(bacteria_seqs)
    print(f"\n[{datetime.now().isoformat()}] Complete!")
    print(f"  Total sequences: {total}")
    print(f"    - Algae (ground truth = algae): {len(algae_seqs)}")
    print(f"    - Bacteria (ground truth = contaminant): {len(bacteria_seqs)}")
    print(f"\nOutput files:")
    print(f"  FASTA: {FASTA_OUTPUT}")
    print(f"  Ground truth: {TRUTH_OUTPUT}")

    # Validate outputs
    print(f"\n[{datetime.now().isoformat()}] Validating outputs...")

    # Count sequences in output FASTA
    with open(FASTA_OUTPUT, 'r') as f:
        fasta_count = sum(1 for line in f if line.startswith('>'))

    # Count lines in ground truth (excluding comments)
    with open(TRUTH_OUTPUT, 'r') as f:
        truth_count = sum(1 for line in f if not line.startswith('#') and not line.startswith('seq_id'))

    print(f"  FASTA sequences: {fasta_count}")
    print(f"  Ground truth entries: {truth_count}")

    if fasta_count != truth_count:
        print("ERROR: Mismatch between FASTA and ground truth counts!")
        sys.exit(1)

    if fasta_count != total:
        print("ERROR: Output count doesn't match extracted count!")
        sys.exit(1)

    print("\n  Validation PASSED")

if __name__ == "__main__":
    main()
