#!/usr/bin/env python3
"""
Geographic breadth vs environmental coupling analysis.

For each novel domain AND each Pfam domain, compute:
  (a) number of unique GPS-mapped stations detected
  (b) number of ocean basins detected
Then correlate these breadth metrics with coupling strength (median |ρ|).

Expected: negative correlation — narrower-range domains have higher |ρ|.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime

NOVEL_MATRIX = "/media/drn2/External/TARA-Oceans/MANUSCRIPT/source_data/dark_proteome/novel_domain_count_matrix.tsv"
NOVEL_CORR = "/media/drn2/External/TARA-Oceans/MANUSCRIPT/source_data/dark_proteome/novel_domain_env_correlations.tsv"
PFAM_CORR = "/media/drn2/External/TARA-Oceans/MANUSCRIPT/source_data/dark_proteome/sensitivity_per_threshold_details/correlations_pfam_baseline.tsv"
PFAM_MERGED = "/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
GPS_KEY = "/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/ASSEMBLY_GPS_MASTER_KEY_20260119_101500.tsv"
OUTPUT = "/media/drn2/External/TARA-Oceans/MANUSCRIPT/.wt45/task4/source_data/ralph45/geographic_breadth_vs_coupling.tsv"


def classify_ocean_basin(lat, lon):
    """Classify a coordinate into an ocean basin."""
    if lat > 66.5:
        return "Arctic"
    if lat < -60:
        return "Southern"
    if 30 <= lat <= 46 and -6 <= lon <= 36:
        return "Mediterranean"
    if 12 <= lat <= 30 and 32 <= lon <= 44:
        return "Red_Sea"
    if 20 <= lat <= 30 and 20 <= lon <= 100:
        return "Indian"
    if -60 <= lat < 20 and 20 <= lon <= 145:
        return "Indian"
    if lon > 100 or lon < -100:
        return "Pacific"
    if -80 <= lon <= 0:
        return "Atlantic"
    return "Other"


def load_gps_with_basins():
    """Load GPS key and compute basin assignments."""
    gps = pd.read_csv(GPS_KEY, sep='\t', comment='#')
    gps = gps.dropna(subset=['latitude', 'longitude'])
    gps['basin'] = gps.apply(lambda r: classify_ocean_basin(r['latitude'], r['longitude']), axis=1)
    gps['station_id'] = gps['latitude'].round(2).astype(str) + '_' + gps['longitude'].round(2).astype(str)
    return gps


def compute_novel_domain_breadth(gps):
    """Compute geographic breadth for each novel domain from count matrix."""
    print("Loading novel domain count matrix...")
    # Read in chunks to manage memory
    novel_df = pd.read_csv(NOVEL_MATRIX, sep='\t', low_memory=False)

    assembly_ids_matrix = novel_df['assembly_id'].values
    # Strip .aa suffix for matching
    assembly_ids_clean = [a.replace('.aa', '') for a in assembly_ids_matrix]

    # Build mapping: clean assembly_id -> (station_id, basin)
    gps_lookup = {}
    for _, row in gps.iterrows():
        gps_lookup[row['assembly_id']] = (row['station_id'], row['basin'])

    # Map matrix rows to station/basin
    row_station = []
    row_basin = []
    valid_rows = []
    for i, aid in enumerate(assembly_ids_clean):
        if aid in gps_lookup:
            s, b = gps_lookup[aid]
            row_station.append(s)
            row_basin.append(b)
            valid_rows.append(i)

    print(f"  Matched {len(valid_rows)}/{len(assembly_ids_clean)} assemblies to GPS")

    # Get domain columns
    domain_cols = [c for c in novel_df.columns if c != 'assembly_id']

    # For each domain, count unique stations and basins where domain is present (count > 0)
    breadth_data = []
    matrix = novel_df.iloc[valid_rows][domain_cols].values
    stations = np.array(row_station)
    basins = np.array(row_basin)

    print(f"  Computing breadth for {len(domain_cols)} novel domains...")
    for j, domain in enumerate(domain_cols):
        if j % 5000 == 0:
            print(f"    {j}/{len(domain_cols)}")
        col = matrix[:, j]
        present = col > 0
        if present.sum() == 0:
            continue
        n_stations = len(set(stations[present]))
        n_basins = len(set(basins[present]))
        n_samples = int(present.sum())
        breadth_data.append({
            'domain': domain,
            'domain_type': 'novel',
            'n_stations': n_stations,
            'n_basins': n_basins,
            'n_samples_present': n_samples,
            'prevalence': n_samples / len(valid_rows)
        })

    del novel_df, matrix
    return pd.DataFrame(breadth_data)


def compute_pfam_breadth(gps):
    """Compute geographic breadth for each Pfam domain from merged dataset."""
    print("Loading Pfam merged dataset...")
    pfam_df = pd.read_csv(PFAM_MERGED, sep='\t', comment='#', low_memory=False)

    # Identify Pfam columns (start with PF)
    pfam_cols = [c for c in pfam_df.columns if c.startswith('PF')]
    print(f"  Found {len(pfam_cols)} Pfam domains")

    # Match assembly IDs to GPS
    assembly_ids = pfam_df['assembly_id'].values
    gps_lookup = {}
    for _, row in gps.iterrows():
        gps_lookup[row['assembly_id']] = (row['station_id'], row['basin'])

    valid_rows = []
    row_station = []
    row_basin = []
    for i, aid in enumerate(assembly_ids):
        # Try with and without .aa suffix
        clean = str(aid).replace('.aa', '')
        if clean in gps_lookup:
            s, b = gps_lookup[clean]
            row_station.append(s)
            row_basin.append(b)
            valid_rows.append(i)
        elif aid in gps_lookup:
            s, b = gps_lookup[aid]
            row_station.append(s)
            row_basin.append(b)
            valid_rows.append(i)

    print(f"  Matched {len(valid_rows)}/{len(assembly_ids)} assemblies to GPS")

    matrix = pfam_df.iloc[valid_rows][pfam_cols].values
    stations = np.array(row_station)
    basins = np.array(row_basin)

    breadth_data = []
    print(f"  Computing breadth for {len(pfam_cols)} Pfam domains...")
    for j, domain in enumerate(pfam_cols):
        if j % 5000 == 0:
            print(f"    {j}/{len(pfam_cols)}")
        col = matrix[:, j]
        present = col > 0
        if present.sum() == 0:
            continue
        n_stations = len(set(stations[present]))
        n_basins = len(set(basins[present]))
        n_samples = int(present.sum())
        breadth_data.append({
            'domain': domain,
            'domain_type': 'pfam',
            'n_stations': n_stations,
            'n_basins': n_basins,
            'n_samples_present': n_samples,
            'prevalence': n_samples / len(valid_rows)
        })

    del pfam_df, matrix
    return pd.DataFrame(breadth_data)


def load_coupling_strength():
    """Load per-domain median |ρ| from correlation files."""
    print("Loading novel domain correlations...")
    novel_corr = pd.read_csv(NOVEL_CORR, sep='\t')
    novel_corr['abs_rho'] = novel_corr['rho'].abs()
    novel_median = novel_corr.groupby('domain')['abs_rho'].median().reset_index()
    novel_median.columns = ['domain', 'median_abs_rho']

    print("Loading Pfam domain correlations...")
    pfam_corr = pd.read_csv(PFAM_CORR, sep='\t')
    pfam_corr['abs_rho'] = pfam_corr['rho'].abs()
    pfam_median = pfam_corr.groupby('domain')['abs_rho'].median().reset_index()
    pfam_median.columns = ['domain', 'median_abs_rho']

    return pd.concat([novel_median, pfam_median], ignore_index=True)


def main():
    print(f"=== Geographic Breadth vs Environmental Coupling ===")
    print(f"Started: {datetime.now()}")

    # Step 1: Load GPS and compute basins
    print("\n[1/4] Loading GPS coordinates and classifying basins...")
    gps = load_gps_with_basins()
    print(f"  {len(gps)} samples with GPS, {gps['basin'].nunique()} basins")

    # Step 2: Compute breadth for novel domains
    print("\n[2/4] Computing novel domain geographic breadth...")
    novel_breadth = compute_novel_domain_breadth(gps)
    print(f"  {len(novel_breadth)} novel domains with breadth data")

    # Step 3: Compute breadth for Pfam domains
    print("\n[3/4] Computing Pfam domain geographic breadth...")
    pfam_breadth = compute_pfam_breadth(gps)
    print(f"  {len(pfam_breadth)} Pfam domains with breadth data")

    # Step 4: Load coupling strength
    print("\n[4/4] Loading coupling strength (median |ρ|)...")
    coupling = load_coupling_strength()
    print(f"  {len(coupling)} domains with coupling data")

    # Merge breadth + coupling
    all_breadth = pd.concat([novel_breadth, pfam_breadth], ignore_index=True)
    merged = all_breadth.merge(coupling, on='domain', how='inner')
    print(f"\n  Merged: {len(merged)} domains with both breadth and coupling data")
    print(f"  Novel: {(merged['domain_type'] == 'novel').sum()}")
    print(f"  Pfam: {(merged['domain_type'] == 'pfam').sum()}")

    # Compute correlations: breadth vs coupling
    print("\n=== Results ===")

    results = {}
    for dtype in ['novel', 'pfam', 'all']:
        subset = merged if dtype == 'all' else merged[merged['domain_type'] == dtype]
        if len(subset) < 10:
            continue

        for breadth_metric in ['n_stations', 'n_basins']:
            rho, p = stats.spearmanr(subset[breadth_metric], subset['median_abs_rho'])
            key = f"{dtype}_{breadth_metric}"
            results[key] = {'rho': rho, 'p': p, 'n': len(subset)}
            print(f"  {dtype:6s} | {breadth_metric:10s} vs median|ρ|: Spearman ρ = {rho:.4f}, p = {p:.2e}, n = {len(subset)}")

    # Also compute for prevalence-controlled subsets (>50% prevalence)
    print("\n--- Prevalence-controlled (>50% of GPS-mapped samples) ---")
    for dtype in ['novel', 'pfam', 'all']:
        subset = merged if dtype == 'all' else merged[merged['domain_type'] == dtype]
        subset_high = subset[subset['prevalence'] > 0.5]
        if len(subset_high) < 10:
            print(f"  {dtype:6s} | <10 domains with >50% prevalence, skipping")
            continue

        for breadth_metric in ['n_stations', 'n_basins']:
            rho, p = stats.spearmanr(subset_high[breadth_metric], subset_high['median_abs_rho'])
            key = f"{dtype}_{breadth_metric}_prev50"
            results[key] = {'rho': rho, 'p': p, 'n': len(subset_high)}
            print(f"  {dtype:6s} | {breadth_metric:10s} vs median|ρ| (prev>50%): ρ = {rho:.4f}, p = {p:.2e}, n = {len(subset_high)}")

    # Save full output
    merged.to_csv(OUTPUT, sep='\t', index=False)
    print(f"\nFull results saved to: {OUTPUT}")
    print(f"Finished: {datetime.now()}")

    # Save summary statistics
    summary_path = OUTPUT.replace('.tsv', '_summary.tsv')
    summary_rows = []
    for key, val in results.items():
        parts = key.split('_', 1)
        summary_rows.append({
            'comparison': key,
            'spearman_rho': val['rho'],
            'p_value': val['p'],
            'n_domains': val['n']
        })
    pd.DataFrame(summary_rows).to_csv(summary_path, sep='\t', index=False)
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
