#!/usr/bin/env python3
"""
RuBisCO search on FULL (unfiltered) protein files.

This searches ALL proteins, not just LA4SR-classified "algal" sequences.
rbcL is definitively algal - its presence validates algae in the sample
independent of LA4SR classification.

Usage:
    python rubisco_full_proteins_20260114.py <sample_id> <protein_file> <hmm_dir> <output_dir>
"""

import os
import sys
import subprocess
import tempfile
from collections import defaultdict

# E-value thresholds
BROAD_EVALUE = 1e-7      # Standard Pfam threshold for detection
LINEAGE_EVALUE = 1e-10   # More stringent for lineage assignment

def run_hmmsearch(hmm_file, fasta_file, evalue=BROAD_EVALUE):
    """Run hmmsearch and return dict of {seq_id: score}."""
    hits = {}
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tbl', delete=False) as tbl:
            tbl_file = tbl.name
        subprocess.run(
            ['hmmsearch', '--tblout', tbl_file, '-E', str(evalue), hmm_file, fasta_file],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800
        )
        with open(tbl_file) as f:
            for line in f:
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 6:
                        seq_id = parts[0]
                        score = float(parts[5])
                        # Keep best score per sequence
                        if seq_id not in hits or score > hits[seq_id]:
                            hits[seq_id] = score
        os.unlink(tbl_file)
    except Exception as e:
        print(f"Warning: hmmsearch failed: {e}", file=sys.stderr)
    return hits

def main():
    if len(sys.argv) != 5:
        print("Usage: python rubisco_full_proteins_20260114.py <sample_id> <protein_file> <hmm_dir> <output_dir>")
        sys.exit(1)

    sample_id = sys.argv[1]
    protein_file = sys.argv[2]
    hmm_dir = sys.argv[3]
    output_dir = sys.argv[4]

    if not os.path.exists(protein_file):
        print(f"ERROR: Protein file not found: {protein_file}", file=sys.stderr)
        sys.exit(1)

    # Count total sequences
    n_total = 0
    with open(protein_file) as f:
        for line in f:
            if line.startswith('>'):
                n_total += 1

    results = {'sample_id': sample_id, 'n_total': n_total}

    # Broad search with green and red HMMs
    green_hmm = os.path.join(hmm_dir, 'green_rbcL.hmm')
    red_hmm = os.path.join(hmm_dir, 'red_rbcL.hmm')

    green_hits = run_hmmsearch(green_hmm, protein_file, BROAD_EVALUE)
    red_hits = run_hmmsearch(red_hmm, protein_file, BROAD_EVALUE)

    results['green_rbcL'] = len(green_hits)
    results['red_rbcL'] = len(red_hits)

    # All RuBisCO hits (union, taking best score)
    all_rubisco = {}
    for seq_id, score in green_hits.items():
        all_rubisco[seq_id] = ('green', score)
    for seq_id, score in red_hits.items():
        if seq_id not in all_rubisco or score > all_rubisco[seq_id][1]:
            all_rubisco[seq_id] = ('red', score)

    results['total_rubisco'] = len(all_rubisco)

    # If we have RuBisCO hits, do lineage-specific search
    lineage_counts = defaultdict(int)

    if all_rubisco:
        # Extract just RuBisCO sequences for detailed lineage search
        rubisco_seqs = set(all_rubisco.keys())

        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp:
            rubisco_fasta = tmp.name

        # Extract sequences
        with open(protein_file) as fin, open(rubisco_fasta, 'w') as fout:
            write_seq = False
            for line in fin:
                if line.startswith('>'):
                    seq_id = line[1:].split()[0]
                    write_seq = seq_id in rubisco_seqs
                if write_seq:
                    fout.write(line)

        # Lineage-specific HMMs
        lineage_hmms = [
            'mamiellophyceae_rbcL.hmm',
            'prasinophyceae_rbcL.hmm',
            'pyramimonadales_rbcL.hmm',
            'chlorellaceae_rbcL.hmm',
            'trebouxiophyceae_rbcL.hmm',
            'scenedesmaceae_rbcL.hmm',
            'pelagophyceae_rbcL.hmm',
            'bolidophyceae_rbcL.hmm',
            'haptophyta_rbcL.hmm',
            'cryptophyta_rbcL.hmm',
        ]

        # Track best lineage per sequence
        seq_scores = defaultdict(dict)

        for hmm_name in lineage_hmms:
            hmm_path = os.path.join(hmm_dir, hmm_name)
            if not os.path.exists(hmm_path):
                continue

            lineage = hmm_name.replace('_rbcL.hmm', '')
            hits = run_hmmsearch(hmm_path, rubisco_fasta, LINEAGE_EVALUE)

            for seq_id, score in hits.items():
                seq_scores[seq_id][lineage] = score

        # Assign best lineage per sequence
        for seq_id, scores in seq_scores.items():
            if scores:
                best_lineage = max(scores, key=scores.get)
                lineage_counts[best_lineage] += 1

        os.unlink(rubisco_fasta)

    # Add lineage counts to results
    for lineage in ['mamiellophyceae', 'prasinophyceae', 'pyramimonadales',
                   'chlorellaceae', 'trebouxiophyceae', 'scenedesmaceae',
                   'pelagophyceae', 'bolidophyceae', 'haptophyta', 'cryptophyta']:
        results[lineage] = lineage_counts.get(lineage, 0)

    # Write results
    output_file = os.path.join(output_dir, f"{sample_id}_rubisco.tsv")
    columns = ['sample_id', 'n_total', 'total_rubisco', 'green_rbcL', 'red_rbcL',
               'mamiellophyceae', 'prasinophyceae', 'pyramimonadales',
               'chlorellaceae', 'trebouxiophyceae', 'scenedesmaceae',
               'pelagophyceae', 'bolidophyceae', 'haptophyta', 'cryptophyta']

    with open(output_file, 'w') as f:
        values = [str(results.get(col, 0)) for col in columns]
        f.write('\t'.join(values) + '\n')

    print(f"{sample_id}: {n_total} total proteins, {results.get('total_rubisco', 0)} RuBisCO")

if __name__ == "__main__":
    main()
