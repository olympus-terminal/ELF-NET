#!/usr/bin/env python3
"""
Audit RuBisCO statistics from rubisco_all_samples_20260114.tsv

Provenance:
  Input: /media/drn2/External/TARA-Oceans/03_analyses/rubisco_all_samples_20260114.tsv
  Input: /media/drn2/External/TARA-Oceans/03_analyses/algagpt_classification_summary_20260114_150000.csv
  Date: 2026-02-09
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# Read the RuBisCO data
print("=" * 80)
print("RUBISCO DATA AUDIT")
print("=" * 80)

rubisco_file = "/media/drn2/External/TARA-Oceans/03_analyses/rubisco_all_samples_20260114.tsv"
print(f"\nReading: {rubisco_file}")

df = pd.read_csv(rubisco_file, sep='\t')

print(f"\nFile has {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

# Task 1: Overall statistics
print("\n" + "=" * 80)
print("TASK 1: OVERALL STATISTICS")
print("=" * 80)

total_samples = df['sample_id'].nunique()
samples_with_rubisco = df[df['total_rubisco'] > 0]['sample_id'].nunique()
total_rubisco_sequences = df['total_rubisco'].sum()

print(f"\nTotal unique samples: {total_samples}")
print(f"Samples with at least one RuBisCO: {samples_with_rubisco}")
print(f"Total RuBisCO sequences across all samples: {total_rubisco_sequences}")

# Task 1: Per-lineage statistics
print("\n" + "=" * 80)
print("PER-LINEAGE STATISTICS")
print("=" * 80)

# The lineage columns are individual columns, not a single 'lineage' column
lineage_columns = ['mamiellophyceae', 'prasinophyceae', 'pyramimonadales',
                   'chlorellaceae', 'trebouxiophyceae', 'scenedesmaceae',
                   'pelagophyceae', 'bolidophyceae', 'haptophyta', 'cryptophyta']

lineage_stats = []
for lineage in lineage_columns:
    df_lineage = df[df[lineage] > 0]
    num_samples = df_lineage['sample_id'].nunique()
    total_sequences = df_lineage[lineage].sum()
    lineage_stats.append({
        'lineage': lineage.capitalize(),
        'num_samples': num_samples,
        'total_sequences': total_sequences
    })

lineage_df = pd.DataFrame(lineage_stats)
lineage_df = lineage_df.sort_values('num_samples', ascending=False)

print("\nLineage statistics (sorted by number of samples):")
print(lineage_df.to_string(index=False))

# Task 2: Check TARA metagenome assemblies (MGYA prefix)
print("\n" + "=" * 80)
print("TASK 2: TARA METAGENOME ASSEMBLIES")
print("=" * 80)

classification_file = "/media/drn2/External/TARA-Oceans/03_analyses/algagpt_classification_summary_20260114_150000.csv"
print(f"\nReading: {classification_file}")

class_df = pd.read_csv(classification_file, comment='#')
print(f"\nTotal entries in classification file: {len(class_df)}")
print(f"Columns: {list(class_df.columns)}")

# Count MGYA entries (it's in filename column)
mgya_entries = class_df[class_df['filename'].str.contains('MGYA', na=False)]
print(f"\nEntries containing 'MGYA' in filename: {len(mgya_entries)}")

# Task 4: Verify georeferenced vs TARA assembly numbers
print("\n" + "=" * 80)
print("TASK 4: GEOREFERENCED vs TARA ASSEMBLIES")
print("=" * 80)

# Check if there's a column indicating georeferencing or TARA status
print("\nChecking for georeferencing indicators...")
print(f"Sample ID examples:")
print(df['sample_id'].head(20).tolist())

# Separate TARA (MGYA) from other samples
tara_samples = df[df['sample_id'].str.contains('MGYA', na=False)]
georeferenced_samples = df[~df['sample_id'].str.contains('MGYA', na=False)]

print(f"\nSamples with 'MGYA' (TARA assemblies): {tara_samples['sample_id'].nunique()}")
print(f"RuBisCO sequences in TARA assemblies: {tara_samples['total_rubisco'].sum()}")

print(f"\nSamples without 'MGYA' (possibly georeferenced): {georeferenced_samples['sample_id'].nunique()}")
print(f"RuBisCO sequences in non-MGYA samples: {georeferenced_samples['total_rubisco'].sum()}")

# Task 7: Statistics for TARA assemblies (1,203 claim)
print("\n" + "=" * 80)
print("TASK 7: TARA ASSEMBLY STATISTICS")
print("=" * 80)

tara_with_rubisco = tara_samples[tara_samples['total_rubisco'] > 0]
tara_total = tara_samples['sample_id'].nunique()
tara_with_rubisco_count = tara_with_rubisco['sample_id'].nunique()

print(f"\nTotal TARA assemblies (MGYA samples): {tara_total}")
print(f"TARA assemblies with at least one RuBisCO: {tara_with_rubisco_count}")
print(f"Percentage: {100 * tara_with_rubisco_count / tara_total:.1f}%")

# Per-sample statistics for TARA
tara_per_sample = tara_samples.groupby('sample_id')['total_rubisco'].sum()
print(f"\nRuBisCO sequences per TARA sample:")
print(f"  Median: {tara_per_sample.median():.1f}")
print(f"  Mean: {tara_per_sample.mean():.1f}")
print(f"  Range: {tara_per_sample.min()}-{tara_per_sample.max()}")
print(f"  Std dev: {tara_per_sample.std():.1f}")

# Distribution bins
bins_21_50 = ((tara_per_sample >= 21) & (tara_per_sample <= 50)).sum()
bins_over_50 = (tara_per_sample > 50).sum()

print(f"\nDistribution of sequences per TARA sample:")
print(f"  21-50 sequences: {bins_21_50} samples ({100 * bins_21_50 / tara_total:.1f}%)")
print(f"  >50 sequences: {bins_over_50} samples ({100 * bins_over_50 / tara_total:.1f}%)")

# Task 6: Verify specific lineage claims
print("\n" + "=" * 80)
print("TASK 6: VERIFY MANUSCRIPT LINEAGE CLAIMS")
print("=" * 80)

manuscript_claims = {
    'Haptophyta': (934, 6305),
    'Chlorellaceae': (918, 4856),
    'Bolidophyceae': (769, 2279),
    'Mamiellophyceae': (564, 1872),
    'Pelagophyceae': (457, 735),
    'Pyramimonadales': (286, None),
    'Cryptophyta': (269, None),
    'Prasinophyceae sensu lato': (248, None),
    'Scenedesmaceae': (135, None),
    'Trebouxiophyceae': (35, None)
}

print("\nComparing manuscript claims to actual data:")
print(f"{'Lineage':<30} {'Claimed Samples':<20} {'Actual Samples':<20} {'Match?':<10} {'Claimed Seqs':<15} {'Actual Seqs':<15} {'Match?'}")
print("-" * 140)

for lineage, (claimed_samples, claimed_seqs) in manuscript_claims.items():
    # Need to match lineage names properly
    lineage_lower = lineage.lower().replace(' sensu lato', '')
    actual_row = lineage_df[lineage_df['lineage'].str.lower().str.replace('aceae', 'aceae') == lineage_lower]
    if len(actual_row) == 0:
        # Try harder to match
        matching_rows = lineage_df[lineage_df['lineage'].str.lower().str.contains(lineage_lower[:5])]
        if len(matching_rows) == 0:
            actual_samples = 0
            actual_seqs = 0
        else:
            actual_samples = int(matching_rows['num_samples'].iloc[0])
            actual_seqs = int(matching_rows['total_sequences'].iloc[0])
    else:
        actual_samples = int(actual_row['num_samples'].iloc[0])
        actual_seqs = int(actual_row['total_sequences'].iloc[0])

    sample_match = "YES" if actual_samples == claimed_samples else "NO"

    if claimed_seqs is None:
        seq_match = "N/A"
        claimed_seqs_str = "N/A"
        actual_seqs_str = str(actual_seqs)
    else:
        seq_match = "YES" if actual_seqs == claimed_seqs else "NO"
        claimed_seqs_str = str(claimed_seqs)
        actual_seqs_str = str(actual_seqs)

    print(f"{lineage:<30} {claimed_samples:<20} {actual_samples:<20} {sample_match:<10} {claimed_seqs_str:<15} {actual_seqs_str:<15} {seq_match}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nKey findings:")
print(f"1. Total samples: {total_samples}")
print(f"2. Samples with RuBisCO: {samples_with_rubisco}")
print(f"3. Total RuBisCO sequences: {total_rubisco_sequences}")
print(f"4. TARA assemblies: {tara_total}")
print(f"5. RuBisCO in TARA: {tara_samples['total_rubisco'].sum()}")
print(f"6. Non-MGYA samples: {georeferenced_samples['sample_id'].nunique()}")
print(f"7. RuBisCO in non-MGYA: {georeferenced_samples['total_rubisco'].sum()}")
