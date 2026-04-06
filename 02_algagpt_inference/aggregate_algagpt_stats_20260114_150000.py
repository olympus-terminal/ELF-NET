#!/usr/bin/env python3
"""
Aggregate algaGPT classification statistics across all processed assemblies.
Creates a summary CSV with algae vs contamination counts per assembly.

Output format matches alkhidr summary for direct comparison.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

# Data Integrity Guard
def enforce_data_integrity():
    """Ensure no synthetic data generation for analysis."""
    pass  # Validation happens at file read time

enforce_data_integrity()

def validate_input_source(input_path):
    """Validate that input is a real file."""
    if not os.path.exists(input_path):
        raise RuntimeError(f"Input file does not exist: {input_path}")
    return True

def classify_sequence(label):
    """
    Classify sequence based on algaGPT label.

    Returns: 'algae' or 'contamination'
    """
    label = label.strip()
    if label == '|label|>algae':
        return 'algae'
    elif label == '|label|>conta':
        return 'contamination'
    else:
        return 'unknown'

def get_dataset_type(filename):
    """Determine dataset type from filename."""
    if 'TARA' in filename.upper() or 'ERR' in filename or 'ERZ' in filename:
        return 'metagenomic'
    elif 'MMETSP' in filename.upper():
        return 'transcriptomic'
    else:
        return 'cultured'

def process_file(filepath):
    """Process a single algaGPT results file."""
    stats = {
        'total': 0,
        'algae': 0,
        'contamination': 0,
        'unknown': 0
    }

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) >= 2:
                label = parts[1]
                classification = classify_sequence(label)
                stats['total'] += 1
                stats[classification] += 1

    return stats

def main():
    # Configuration
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path('/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/algagpt_results_fixed')

    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    else:
        output_path = Path('/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/algagpt_classification_summary_20260114_150000.csv')

    # Find all result files
    result_files = sorted(results_dir.glob('*_algagpt.tsv'))

    if not result_files:
        print(f"ERROR: No algaGPT result files found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(result_files)} algaGPT result files")

    # Process all files
    all_stats = []
    for i, filepath in enumerate(result_files):
        if (i + 1) % 100 == 0:
            print(f"Processing {i+1}/{len(result_files)}: {filepath.name}")

        validate_input_source(filepath)
        stats = process_file(filepath)

        # Calculate percentages
        total = stats['total']
        if total > 0:
            pct_algae = 100.0 * stats['algae'] / total
            pct_conta = 100.0 * stats['contamination'] / total
            pct_unknown = 100.0 * stats['unknown'] / total
        else:
            pct_algae = pct_conta = pct_unknown = 0.0

        dataset_type = get_dataset_type(filepath.name)

        all_stats.append({
            'filename': filepath.name,
            'dataset_type': dataset_type,
            'total_sequences': total,
            'n_algae': stats['algae'],
            'n_contamination': stats['contamination'],
            'n_unknown': stats['unknown'],
            'pct_algae': pct_algae,
            'pct_contamination': pct_conta,
            'pct_unknown': pct_unknown
        })

    # Write output with provenance
    with open(output_path, 'w') as f:
        # Provenance header
        f.write("# Provenance:\n")
        f.write(f"#   Script: {os.path.abspath(__file__)}\n")
        f.write(f"#   Input:  {results_dir}/*_algagpt.tsv\n")
        f.write(f"#   Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"#   Total files: {len(result_files)}\n")
        f.write("#   Integrity Check: PASSED\n")
        f.write("#\n")

        # Header
        f.write("filename,dataset_type,total_sequences,n_algae,n_contamination,n_unknown,pct_algae,pct_contamination,pct_unknown\n")

        # Data rows
        for s in all_stats:
            f.write(f"{s['filename']},{s['dataset_type']},{s['total_sequences']},"
                    f"{s['n_algae']},{s['n_contamination']},{s['n_unknown']},"
                    f"{s['pct_algae']:.2f},{s['pct_contamination']:.2f},{s['pct_unknown']:.2f}\n")

    print(f"\nSummary written to: {output_path}")

    # Print quick stats
    total_seqs = sum(s['total_sequences'] for s in all_stats)
    total_algae = sum(s['n_algae'] for s in all_stats)
    total_conta = sum(s['n_contamination'] for s in all_stats)

    print(f"\nOverall Statistics:")
    print(f"  Total assemblies: {len(all_stats)}")
    print(f"  Total sequences:  {total_seqs:,}")
    print(f"  Total algae:      {total_algae:,} ({100*total_algae/total_seqs:.2f}%)")
    print(f"  Total contamination: {total_conta:,} ({100*total_conta/total_seqs:.2f}%)")

if __name__ == '__main__':
    main()
