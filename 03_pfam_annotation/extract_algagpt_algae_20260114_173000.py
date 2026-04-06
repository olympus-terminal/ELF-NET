#!/usr/bin/env python3
"""
Extract algae-classified sequences from algaGPT results.

Reads algaGPT classification TSV files and extracts sequences labeled as 'algae'
from the original protein FASTA files.

Usage: python extract_algagpt_algae_20260114_173000.py <algagpt_tsv> <input_fasta> <output_fasta>

Provenance:
  Script: extract_algagpt_algae_20260114_173000.py
  Date: 2026-01-14
"""

import sys
import os
from datetime import datetime

def extract_algae_sequences(algagpt_tsv, input_fasta, output_fasta):
    """Extract sequences classified as algae."""

    # Read algae sequence IDs from algaGPT output
    algae_ids = set()
    total_seqs = 0

    with open(algagpt_tsv, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_seqs += 1
            # Format: seqid\t|label|>algae or seqid\t|label|>conta
            parts = line.split('\t')
            if len(parts) >= 2 and '>algae' in parts[1]:
                seq_id = parts[0]
                algae_ids.add(seq_id)

    print(f"  Total sequences in algaGPT: {total_seqs}", file=sys.stderr)
    print(f"  Algae sequences: {len(algae_ids)} ({len(algae_ids)/total_seqs*100:.1f}%)", file=sys.stderr)

    # Extract sequences from FASTA
    extracted = 0
    current_id = None
    current_seq = []
    write_seq = False

    with open(input_fasta, 'r') as fin, open(output_fasta, 'w') as fout:
        for line in fin:
            if line.startswith('>'):
                # Write previous sequence if it was algae
                if write_seq and current_seq:
                    fout.write(f">{current_id}\n")
                    fout.write(''.join(current_seq))
                    extracted += 1

                # Parse new header
                header = line[1:].strip()
                current_id = header.split()[0]  # Get first word as ID
                current_seq = []
                write_seq = current_id in algae_ids
            else:
                if write_seq:
                    current_seq.append(line)

        # Don't forget last sequence
        if write_seq and current_seq:
            fout.write(f">{current_id}\n")
            fout.write(''.join(current_seq))
            extracted += 1

    print(f"  Extracted sequences: {extracted}", file=sys.stderr)
    return extracted, len(algae_ids)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <algagpt_tsv> <input_fasta> <output_fasta>")
        sys.exit(1)

    algagpt_tsv = sys.argv[1]
    input_fasta = sys.argv[2]
    output_fasta = sys.argv[3]

    print(f"Processing: {os.path.basename(input_fasta)}", file=sys.stderr)
    extract_algae_sequences(algagpt_tsv, input_fasta, output_fasta)
