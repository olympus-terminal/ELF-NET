#!/usr/bin/env python3
"""
RuBisCO search per sample - E-value 1e-7 version.

CHANGES from 20260106 version:
- E-value for broad detection: 1e-50 -> 1e-7
- This should catch divergent rbcL sequences we were missing

For each metagenome:
1. Load algal-classified sequence IDs
2. Extract those sequences
3. Run hmmsearch against all RuBisCO HMMs
4. Record hits per lineage per sample

Usage:
    python rubisco_per_sample_20260114_e7.py <sample_id> <classification_file> <protein_file> <hmm_dir> <output_dir>
"""

import os
import sys
import subprocess
import tempfile
from collections import defaultdict

# CHANGED: E-value threshold for broad detection
BROAD_EVALUE = 1e-7  # Was 1e-50, now 1e-7 (standard Pfam threshold)
LINEAGE_EVALUE = 1e-10  # Keep lineage assignment stringent

def load_algal_ids(classification_file):
    """Load sequence IDs classified as algal (@@@)."""
    algal_ids = set()
    with open(classification_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and '@@@' in parts[1]:
                algal_ids.add(parts[0])
    return algal_ids

def extract_sequences(protein_file, seq_ids, output_file):
    """Extract sequences matching IDs to temp file."""
    count = 0
    with open(protein_file) as fin, open(output_file, 'w') as fout:
        write_seq = False
        for line in fin:
            if line.startswith('>'):
                seq_id = line[1:].split()[0]
                write_seq = seq_id in seq_ids
                if write_seq:
                    count += 1
            if write_seq:
                fout.write(line)
    return count

def run_hmmsearch(hmm_file, fasta_file, evalue=BROAD_EVALUE):
    """Run hmmsearch and return set of hit sequence IDs."""
    hits = set()
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tbl', delete=False) as tbl:
            tbl_file = tbl.name
        subprocess.run(
            ['hmmsearch', '--tblout', tbl_file, '-E', str(evalue), hmm_file, fasta_file],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600
        )
        with open(tbl_file) as f:
            for line in f:
                if line and not line.startswith('#'):
                    parts = line.split()
                    if parts:
                        hits.add(parts[0])
        os.unlink(tbl_file)
    except Exception as e:
        print(f"Warning: hmmsearch failed: {e}", file=sys.stderr)
    return hits

def main():
    if len(sys.argv) != 6:
        print("Usage: python rubisco_per_sample_20260114_e7.py <sample_id> <classification_file> <protein_file> <hmm_dir> <output_dir>")
        sys.exit(1)

    sample_id = sys.argv[1]
    classification_file = sys.argv[2]
    protein_file = sys.argv[3]
    hmm_dir = sys.argv[4]
    output_dir = sys.argv[5]

    # Load algal IDs
    algal_ids = load_algal_ids(classification_file)
    n_algal = len(algal_ids)

    if n_algal == 0:
        # No algal sequences, write empty result
        with open(os.path.join(output_dir, f"{sample_id}_rubisco.tsv"), 'w') as f:
            f.write(f"{sample_id}\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\n")
        return

    # Extract algal sequences to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp:
        tmp_fasta = tmp.name

    try:
        n_extracted = extract_sequences(protein_file, algal_ids, tmp_fasta)

        # Search with each HMM
        results = {'sample_id': sample_id, 'n_algal': n_algal}

        # Broad search first - NOW WITH E=1e-7
        green_hmm = os.path.join(hmm_dir, 'green_rbcL.hmm')
        red_hmm = os.path.join(hmm_dir, 'red_rbcL.hmm')

        green_hits = run_hmmsearch(green_hmm, tmp_fasta, BROAD_EVALUE)
        red_hits = run_hmmsearch(red_hmm, tmp_fasta, BROAD_EVALUE)

        results['green_rbcL'] = len(green_hits)
        results['red_rbcL'] = len(red_hits)

        # All RuBisCO hits (union)
        all_rubisco = green_hits | red_hits
        results['total_rubisco'] = len(all_rubisco)

        # If we have RuBisCO hits, do lineage-specific search
        if all_rubisco:
            # Extract just RuBisCO sequences for detailed search
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp2:
                rubisco_fasta = tmp2.name

            extract_sequences(tmp_fasta, all_rubisco, rubisco_fasta)

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
            seq_scores = defaultdict(dict)  # seq_id -> {lineage: score}

            for hmm_name in lineage_hmms:
                hmm_path = os.path.join(hmm_dir, hmm_name)
                if not os.path.exists(hmm_path):
                    continue

                lineage = hmm_name.replace('_rbcL.hmm', '')

                # Run with tblout to get scores
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbl', delete=False) as tbl:
                        lineage_tbl = tbl.name
                    subprocess.run(
                        ['hmmsearch', '--tblout', lineage_tbl, '-E', str(LINEAGE_EVALUE), hmm_path, rubisco_fasta],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300
                    )
                    with open(lineage_tbl) as f:
                        for line in f:
                            if line and not line.startswith('#'):
                                parts = line.split()
                                if len(parts) >= 6:
                                    seq_id = parts[0]
                                    score = float(parts[5])
                                    seq_scores[seq_id][lineage] = score
                    os.unlink(lineage_tbl)
                except Exception as e:
                    print(f"Warning: {hmm_name} search failed: {e}", file=sys.stderr)

            # Assign best lineage per sequence
            lineage_counts = defaultdict(int)
            for seq_id, scores in seq_scores.items():
                if scores:
                    best_lineage = max(scores, key=scores.get)
                    lineage_counts[best_lineage] += 1

            # Add to results
            for lineage in ['mamiellophyceae', 'prasinophyceae', 'pyramimonadales',
                           'chlorellaceae', 'trebouxiophyceae', 'scenedesmaceae',
                           'pelagophyceae', 'bolidophyceae', 'haptophyta', 'cryptophyta']:
                results[lineage] = lineage_counts.get(lineage, 0)

            os.unlink(rubisco_fasta)

        # Write results
        output_file = os.path.join(output_dir, f"{sample_id}_rubisco.tsv")
        columns = ['sample_id', 'n_algal', 'total_rubisco', 'green_rbcL', 'red_rbcL',
                   'mamiellophyceae', 'prasinophyceae', 'pyramimonadales',
                   'chlorellaceae', 'trebouxiophyceae', 'scenedesmaceae',
                   'pelagophyceae', 'bolidophyceae', 'haptophyta', 'cryptophyta']

        with open(output_file, 'w') as f:
            values = [str(results.get(col, 0)) for col in columns]
            f.write('\t'.join(values) + '\n')

        print(f"{sample_id}: {n_algal} algal, {results.get('total_rubisco', 0)} RuBisCO")

    finally:
        if os.path.exists(tmp_fasta):
            os.unlink(tmp_fasta)

if __name__ == "__main__":
    main()
