#!/usr/bin/env python3
"""
Compute verified statistics from LA4SR classification summary.
Reads the actual CSV file and outputs statistics.
"""

import csv
import sys
from pathlib import Path

input_file = Path("/media/drn/External1/TARA-Oceans/03_analyses/algagpt_classification_summary_20260114_150000.csv")

# Initialize counters
total_files = 0
total_sequences = 0
total_algae = 0
total_contamination = 0
total_unknown = 0
pct_algae_sum = 0.0

# Dataset type specific stats
dataset_stats = {}

# Read and process file
with open(input_file, 'r') as f:
    # Skip comment lines at the beginning
    for line in f:
        if not line.startswith('#'):
            # This is the header line
            break

    # Now read data rows
    reader = csv.DictReader(f, fieldnames=['filename', 'dataset_type', 'total_sequences',
                                            'n_algae', 'n_contamination', 'n_unknown',
                                            'pct_algae', 'pct_contamination', 'pct_unknown'])

    for row in reader:
        total_files += 1
        seqs = int(row['total_sequences'])
        alg = int(row['n_algae'])
        cont = int(row['n_contamination'])
        unk = int(row['n_unknown'])
        pct = float(row['pct_algae'])

        total_sequences += seqs
        total_algae += alg
        total_contamination += cont
        total_unknown += unk
        pct_algae_sum += pct

        # Track dataset types with detailed stats
        dt = row['dataset_type']
        if dt not in dataset_stats:
            dataset_stats[dt] = {'count': 0, 'sequences': 0, 'algae': 0, 'pct_sum': 0.0}
        dataset_stats[dt]['count'] += 1
        dataset_stats[dt]['sequences'] += seqs
        dataset_stats[dt]['algae'] += alg
        dataset_stats[dt]['pct_sum'] += pct

# Calculate derived statistics
mean_pct_algae = pct_algae_sum / total_files if total_files > 0 else 0
overall_pct_algae = (total_algae / total_sequences * 100) if total_sequences > 0 else 0

print("=" * 60)
print("LA4SR Classification Summary Statistics")
print("=" * 60)
print(f"Source file: {input_file}")
print()
print("OVERALL COUNTS:")
print(f"  Total files processed: {total_files:,}")
print(f"  Total sequences: {total_sequences:,}")
print(f"  Total algal sequences: {total_algae:,}")
print(f"  Total contamination sequences: {total_contamination:,}")
print(f"  Total unknown sequences: {total_unknown:,}")
print()
print("OVERALL PERCENTAGES:")
print(f"  Mean algal percentage (per file): {mean_pct_algae:.2f}%")
print(f"  Overall algal percentage (global): {overall_pct_algae:.2f}%")
print()
print("BY DATASET TYPE:")
for dt, stats in sorted(dataset_stats.items()):
    dt_mean_pct = stats['pct_sum'] / stats['count'] if stats['count'] > 0 else 0
    dt_overall_pct = (stats['algae'] / stats['sequences'] * 100) if stats['sequences'] > 0 else 0
    print(f"  {dt}:")
    print(f"    Files: {stats['count']:,}")
    print(f"    Sequences: {stats['sequences']:,}")
    print(f"    Algal: {stats['algae']:,}")
    print(f"    Mean pct_algae: {dt_mean_pct:.2f}%")
    print(f"    Overall pct: {dt_overall_pct:.2f}%")
print("=" * 60)

# Write markdown output to source_data
output_md = Path("/media/drn/External1/TARA-Oceans/MANUSCRIPT/source_data/la4sr_validation_stats.md")
output_md.parent.mkdir(exist_ok=True)
with open(output_md, 'w') as f:
    f.write("# LA4SR Classification Validation Statistics\n\n")
    f.write("## Provenance\n\n")
    f.write(f"- **Source file:** `{input_file}`\n")
    f.write(f"- **Generated:** 2026-01-14\n")
    f.write(f"- **Script:** `compute_la4sr_stats.py`\n\n")
    f.write("## Overall Statistics\n\n")
    f.write(f"| Metric | Value |\n")
    f.write(f"|--------|-------|\n")
    f.write(f"| Total files processed | {total_files:,} |\n")
    f.write(f"| Total sequences | {total_sequences:,} |\n")
    f.write(f"| Total algal sequences | {total_algae:,} |\n")
    f.write(f"| Total contamination sequences | {total_contamination:,} |\n")
    f.write(f"| Total unknown sequences | {total_unknown:,} |\n")
    f.write(f"| Mean algal % (per file) | {mean_pct_algae:.2f}% |\n")
    f.write(f"| Overall algal % (global) | {overall_pct_algae:.2f}% |\n\n")
    f.write("## By Dataset Type\n\n")
    for dt, stats in sorted(dataset_stats.items()):
        dt_mean_pct = stats['pct_sum'] / stats['count'] if stats['count'] > 0 else 0
        dt_overall_pct = (stats['algae'] / stats['sequences'] * 100) if stats['sequences'] > 0 else 0
        f.write(f"### {dt.capitalize()}\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Files | {stats['count']:,} |\n")
        f.write(f"| Sequences | {stats['sequences']:,} |\n")
        f.write(f"| Algal sequences | {stats['algae']:,} |\n")
        f.write(f"| Mean algal % (per file) | {dt_mean_pct:.2f}% |\n")
        f.write(f"| Overall algal % | {dt_overall_pct:.2f}% |\n\n")

print(f"\nMarkdown stats written to: {output_md}")
