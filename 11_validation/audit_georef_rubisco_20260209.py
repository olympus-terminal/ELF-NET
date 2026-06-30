#!/usr/bin/env python3
"""
Verify georeferenced RuBisCO claim

Provenance:
  Input: /media/drn2/External/TARA-Oceans/03_analyses/rubisco_all_samples_20260114.tsv
  Input: /media/drn2/External/TARA-Oceans/01_raw_data/metadata/ALL_assemblies_GPS_mapping.tsv
  Date: 2026-02-09
"""

import pandas as pd

print("=" * 80)
print("GEOREFERENCED RUBISCO VERIFICATION")
print("=" * 80)

# Read RuBisCO data
rubisco_file = "/media/drn2/External/TARA-Oceans/03_analyses/rubisco_all_samples_20260114.tsv"
print(f"\nReading RuBisCO data: {rubisco_file}")
rubisco_df = pd.read_csv(rubisco_file, sep='\t')

# Read GPS mapping
gps_file = "/media/drn2/External/TARA-Oceans/01_raw_data/metadata/ALL_assemblies_GPS_mapping.tsv"
print(f"Reading GPS mapping: {gps_file}")
gps_df = pd.read_csv(gps_file, sep='\t', comment='#')

print(f"\nGPS mapping columns: {list(gps_df.columns)}")
print(f"GPS mapping rows: {len(gps_df)}")
print("\nFirst few GPS rows:")
print(gps_df.head())

# Clean sample IDs for matching (GPS has assembly_id, not sample_id)
rubisco_df['sample_clean'] = rubisco_df['sample_id'].str.replace('.aa$', '', regex=True).str.replace('.fa', '')
gps_df['sample_clean'] = gps_df['assembly_id']

# Check if GPS column exists and has valid lat/lon
if 'latitude' in gps_df.columns and 'longitude' in gps_df.columns:
    print("\nFound latitude/longitude columns")
    gps_valid = gps_df[(gps_df['latitude'].notna()) & (gps_df['longitude'].notna())]
    print(f"Samples with valid GPS coordinates: {len(gps_valid)}")

    # Merge with RuBisCO data
    merged = rubisco_df.merge(gps_valid[['sample_clean', 'latitude', 'longitude']],
                              on='sample_clean', how='inner')

    print(f"\nRuBisCO samples with GPS coordinates: {len(merged)}")
    print(f"Total RuBisCO sequences in georeferenced samples: {merged['total_rubisco'].sum()}")

    # Also check TARA specifically
    tara_georef = merged[merged['sample_id'].str.contains('MGYA', na=False)]
    print(f"\nTARA samples with GPS coordinates: {len(tara_georef)}")
    print(f"RuBisCO sequences in georeferenced TARA samples: {tara_georef['total_rubisco'].sum()}")
else:
    print("\nNo latitude/longitude columns found in GPS mapping file")
    print(f"Available columns: {list(gps_df.columns)}")

# Check the 21,501 taxa claim
print("\n" + "=" * 80)
print("VERIFY TAXA ESTIMATE CLAIM")
print("=" * 80)

tara_rubisco = rubisco_df[rubisco_df['sample_id'].str.contains('MGYA', na=False)]['total_rubisco'].sum()
mmetsp_count = 677  # from file count
refgenome_count = 193  # from file count

print(f"\nComponents of ~21,500 estimate:")
print(f"  TARA metagenome assemblies: {tara_rubisco:,}")
print(f"  MMETSP transcriptomes: {mmetsp_count:,}")
print(f"  Reference genomes: {refgenome_count:,}")
print(f"  Total: {tara_rubisco + mmetsp_count + refgenome_count:,}")

print("\nManuscript claimed: 20,592 + 676 + 233 = 21,501")
print(f"Actual calculation: {tara_rubisco:,} + {mmetsp_count} + {refgenome_count} = {tara_rubisco + mmetsp_count + refgenome_count:,}")

# Distribution verification
print("\n" + "=" * 80)
print("VERIFY DISTRIBUTION CLAIMS")
print("=" * 80)

tara_samples = rubisco_df[rubisco_df['sample_id'].str.contains('MGYA', na=False)]
tara_per_sample = tara_samples.groupby('sample_id')['total_rubisco'].sum()

print(f"\nMedian claimed: 13, Actual: {tara_per_sample.median():.1f}")
print(f"Mean claimed: 19.3, Actual: {tara_per_sample.mean():.1f}")
print(f"Range claimed: 1-113, Actual: {tara_per_sample.min()}-{tara_per_sample.max()}")
print(f"Std dev claimed: 18.3, Actual: {tara_per_sample.std():.1f}")

total_tara = len(tara_per_sample)
count_21_50 = ((tara_per_sample >= 21) & (tara_per_sample <= 50)).sum()
count_over_50 = (tara_per_sample > 50).sum()

print(f"\n21-50 sequences claimed: 30.0%, Actual: {100 * count_21_50 / total_tara:.1f}%")
print(f">50 sequences claimed: 6.7%, Actual: {100 * count_over_50 / total_tara:.1f}%")
