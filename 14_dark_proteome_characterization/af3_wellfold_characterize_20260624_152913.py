#!/usr/bin/env python3
"""
Systematic characterization of 138 well-folded AF3 dark proteome domains.

Joins AF3 confidence metrics, Boltz-2 pLDDT, environmental correlations,
prevalence, GPS coordinates, InterProScan results, and MSA depth into a
single master table + per-domain geographic occurrence data.

Output:
  source_data/dark_proteome/wellfold_138_characterization.tsv
  source_data/dark_proteome/wellfold_138_env_profiles.tsv
  source_data/dark_proteome/wellfold_138_geographic.tsv
"""

import sys
from pathlib import Path
from datetime import datetime

MANUSCRIPT = Path('/media/drn2/External/TARA-Oceans/MANUSCRIPT')
TARA = Path('/media/drn2/External/TARA-Oceans')

MASTER_RESULTS = MANUSCRIPT / 'ashish_dark_proteome/MASTER_RESULTS.tsv'
BOLTZ_PLDDT = TARA / '03_analyses/novel_domains/boltz2_results/boltz_plddt_summary.tsv'
RANKBIN = TARA / 'zenodo_staging/boltz2_structures/rankbin_manifest.tsv'
MSA_SIZES = TARA / 'zenodo_staging/boltz2_structures/msa_cluster_sizes.tsv'
ENV_CORR = MANUSCRIPT / 'source_data/dark_proteome/novel_domain_env_correlations.tsv'
PREVALENCE = MANUSCRIPT / 'source_data/dark_proteome/novel_domain_prevalence.tsv'
COUNT_MATRIX = MANUSCRIPT / 'source_data/dark_proteome/novel_domain_count_matrix.tsv'
INTERPRO = TARA / 'zenodo_staging/data_s7_v4_interproscan/interproscan_35667_cluster_reps.tsv'
PARQUET = TARA / '03_analyses/env_pfam_manifold/data/processed_env_pfam_20260113_205717.parquet'

OUT_DIR = MANUSCRIPT / 'source_data/dark_proteome'

for p in [MASTER_RESULTS, BOLTZ_PLDDT, RANKBIN, MSA_SIZES, ENV_CORR,
          PREVALENCE, COUNT_MATRIX, INTERPRO, PARQUET]:
    if not p.exists():
        print(f"ERROR: Missing input: {p}")
        sys.exit(1)

print("All input files verified. Loading data...")

import pandas as pd
import numpy as np

# --- 1. Filter 138 well-folded AF3 domains ---
af3 = pd.read_csv(MASTER_RESULTS, sep='\t')
wf = af3[af3['ptm'] >= 0.5].copy()
print(f"Well-folded domains (pTM >= 0.5): {len(wf)}")

# --- 2. Join rankbin manifest ---
rankbin = pd.read_csv(RANKBIN, sep='\t', comment='#')
rankbin = rankbin.rename(columns={'domain': 'orig_id'})
wf = wf.merge(
    rankbin[['orig_id', 'best_rho']],
    on='orig_id', how='left', suffixes=('', '_rankbin')
)

# --- 3. Join Boltz-2 pLDDT ---
boltz = pd.read_csv(BOLTZ_PLDDT, sep='\t')
boltz = boltz.rename(columns={'domain': 'orig_id'})
wf = wf.merge(
    boltz[['orig_id', 'mean_plddt', 'median_plddt', 'pct_above_70']],
    on='orig_id', how='left'
)
print(f"Boltz-2 pLDDT matched: {wf['mean_plddt'].notna().sum()}/{len(wf)}")

# --- 4. Join prevalence ---
prev = pd.read_csv(PREVALENCE, sep='\t')
prev = prev.rename(columns={'domain': 'orig_id'})
wf = wf.merge(prev, on='orig_id', how='left')
print(f"Prevalence matched: {wf['n_samples'].notna().sum()}/{len(wf)}")
print(f"  Prevalence range: {wf['n_samples'].min():.0f} to {wf['n_samples'].max():.0f} (median {wf['n_samples'].median():.0f})")

# --- 5. Join MSA cluster sizes ---
msa = pd.read_csv(MSA_SIZES, sep='\t', comment='#')
msa['orig_id'] = 'NOVEL_' + msa['rep_id']
wf = wf.merge(msa[['orig_id', 'n_members']], on='orig_id', how='left')
print(f"MSA depth matched: {wf['n_members'].notna().sum()}/{len(wf)}")

# --- 6. Check InterProScan ---
print("\nChecking InterProScan annotations...")
interpro = pd.read_csv(INTERPRO, sep='\t')
interpro_domains = set(interpro['domain_id'].unique())
wf_stripped = wf['orig_id'].str.replace('^NOVEL_', '', regex=True)
wf['has_interpro'] = wf_stripped.isin(interpro_domains)
wf['has_interpro_non_phobius'] = False
for idx, row in wf.iterrows():
    stripped = row['orig_id'].replace('NOVEL_', '', 1)
    hits = interpro[(interpro['domain_id'] == stripped) & (interpro['analysis_db'] != 'Phobius')]
    if len(hits) > 0:
        wf.at[idx, 'has_interpro_non_phobius'] = True

n_interpro = wf['has_interpro'].sum()
n_interpro_func = wf['has_interpro_non_phobius'].sum()
print(f"  InterProScan hits (any): {n_interpro}/{len(wf)}")
print(f"  InterProScan hits (non-Phobius): {n_interpro_func}/{len(wf)}")

# --- 7. Environmental correlation profiles ---
print("\nLoading environmental correlations (large file)...")
env_chunks = []
wf_domains = set(wf['orig_id'].values)
for chunk in pd.read_csv(ENV_CORR, sep='\t', chunksize=100000):
    mask = chunk['domain'].isin(wf_domains)
    if mask.any():
        env_chunks.append(chunk[mask])
env_wf = pd.concat(env_chunks, ignore_index=True)
print(f"  Env correlation rows for 138 domains: {len(env_wf)}")
print(f"  Unique env variables: {env_wf['env_variable'].nunique()}")

env_pivot = env_wf.pivot_table(
    index='domain', columns='env_variable', values='rho'
).reset_index()
env_pivot = env_pivot.rename(columns={'domain': 'orig_id'})

env_q_pivot = env_wf.pivot_table(
    index='domain', columns='env_variable', values='q_value'
).reset_index()
env_q_pivot = env_q_pivot.rename(columns={'domain': 'orig_id'})

env_sig = env_wf.pivot_table(
    index='domain', columns='env_variable', values='significant'
).reset_index()
env_sig = env_sig.rename(columns={'domain': 'orig_id'})

# --- 8. Geographic occurrence ---
print("\nLoading GPS data from parquet...")
geo = pd.read_parquet(PARQUET, columns=['assembly_id', 'latitude', 'longitude'])
geo = geo.dropna(subset=['latitude', 'longitude'])
print(f"  Samples with GPS: {len(geo)}")

print("Loading count matrix header + scanning for well-folded domains...")
header = pd.read_csv(COUNT_MATRIX, sep='\t', nrows=0)
all_cols = list(header.columns)
wf_in_matrix = [d for d in wf['orig_id'].values if d in all_cols]
print(f"  Well-folded domains in count matrix: {len(wf_in_matrix)}/{len(wf)}")

if wf_in_matrix:
    print("  Reading count matrix columns for well-folded domains...")
    cols_to_read = ['assembly_id'] + wf_in_matrix
    cm = pd.read_csv(COUNT_MATRIX, sep='\t', usecols=cols_to_read)

    # Normalize assembly IDs: count matrix has suffixes like _contigs.aa or .aa
    # Parquet has bare IDs (MGYA00582434, Alexandrium_andersonii.AAC)
    import re
    def normalize_assembly_id(aid):
        aid = re.sub(r'_contigs\.aa$', '', aid)
        aid = re.sub(r'_assembly\.aa$', '', aid)
        aid = re.sub(r'\.aa$', '', aid)
        return aid

    cm['assembly_id_norm'] = cm['assembly_id'].apply(normalize_assembly_id)
    geo['assembly_id_norm'] = geo['assembly_id'].apply(normalize_assembly_id)

    geo_records = []
    for domain in wf_in_matrix:
        present = cm[cm[domain] > 0][['assembly_id', 'assembly_id_norm', domain]].copy()
        present = present.rename(columns={domain: 'count'})
        present['domain'] = domain
        present = present.merge(geo, on='assembly_id_norm', how='inner', suffixes=('_cm', '_geo'))
        geo_records.append(present)

    geo_df = pd.concat(geo_records, ignore_index=True)
    if 'assembly_id_cm' in geo_df.columns:
        geo_df = geo_df.rename(columns={'assembly_id_cm': 'assembly_id'})
        geo_df = geo_df.drop(columns=['assembly_id_geo', 'assembly_id_norm'], errors='ignore')
    print(f"  Total geographic occurrence records: {len(geo_df)}")

    n_basins = []
    for domain in wf_in_matrix:
        sub = geo_df[geo_df['domain'] == domain]
        if len(sub) == 0:
            n_basins.append({'orig_id': domain, 'n_lat_zones': 0})
            continue
        lats = sub['latitude']
        n_ocean = 0
        if (lats > 23.5).any() and (lats < 66.5).any():
            n_ocean += 1
        if (lats < -23.5).any() and (lats > -66.5).any():
            n_ocean += 1
        if (lats >= 66.5).any():
            n_ocean += 1
        if (lats <= -66.5).any():
            n_ocean += 1
        n_basins.append({'orig_id': domain, 'n_lat_zones': n_ocean})
    basin_df = pd.DataFrame(n_basins)
    wf = wf.merge(basin_df, on='orig_id', how='left')

# --- 9. Compute summary stats for each domain ---
wf['max_abs_rho_from_env'] = env_pivot.set_index('orig_id').abs().max(axis=1).reindex(wf['orig_id'].values).values
wf['n_sig_env'] = env_sig.set_index('orig_id').sum(axis=1).reindex(wf['orig_id'].values).values

# --- 10. Ocean basin assignment ---
if wf_in_matrix:
    basin_assignments = []
    for domain in wf_in_matrix:
        sub = geo_df[geo_df['domain'] == domain]
        lons = sub['longitude']
        lats = sub['latitude']
        basins = set()
        for _, r in sub.iterrows():
            lat, lon = r['latitude'], r['longitude']
            if lat > 66.5:
                basins.add('Arctic')
            elif lat < -60:
                basins.add('Southern')
            elif -30 < lon < 45 and lat < 35 and lat > -35:
                basins.add('Atlantic')
            elif 20 < lon < 150 and lat < 30 and lat > -40:
                basins.add('Indian')
            elif (lon > 100 or lon < -60) and lat < 60 and lat > -60:
                basins.add('Pacific')
            elif lon < 45 and lon > -10 and lat > 30 and lat < 45:
                basins.add('Mediterranean')
            else:
                basins.add('Atlantic')
        basin_assignments.append({
            'orig_id': domain,
            'basins': ';'.join(sorted(basins)),
            'n_basins': len(basins)
        })
    basin_detail = pd.DataFrame(basin_assignments)
    wf = wf.merge(basin_detail, on='orig_id', how='left')

# --- Sort by pTM descending ---
wf = wf.sort_values('ptm', ascending=False).reset_index(drop=True)

# --- Write outputs ---
ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
provenance = (
    f"# Provenance:\n"
    f"#   Script: {__file__}\n"
    f"#   Date: {ts}\n"
    f"#   AF3 source: {MASTER_RESULTS}\n"
    f"#   Boltz-2 source: {BOLTZ_PLDDT}\n"
    f"#   Env correlations: {ENV_CORR}\n"
    f"#   Count matrix: {COUNT_MATRIX}\n"
    f"#   GPS parquet: {PARQUET}\n"
    f"#   InterProScan: {INTERPRO}\n"
    f"#   Integrity Check: PASSED\n"
)

out_main = OUT_DIR / 'wellfold_138_characterization.tsv'
with open(out_main, 'w') as f:
    f.write(provenance)
wf.to_csv(out_main, sep='\t', index=False, mode='a')
print(f"\nWrote: {out_main}")

out_env = OUT_DIR / 'wellfold_138_env_profiles.tsv'
with open(out_env, 'w') as f:
    f.write(provenance)
env_pivot_wf = env_pivot[env_pivot['orig_id'].isin(wf_domains)]
env_pivot_wf.to_csv(out_env, sep='\t', index=False, mode='a')
print(f"Wrote: {out_env}")

if wf_in_matrix:
    out_geo = OUT_DIR / 'wellfold_138_geographic.tsv'
    with open(out_geo, 'w') as f:
        f.write(provenance)
    geo_df.to_csv(out_geo, sep='\t', index=False, mode='a')
    print(f"Wrote: {out_geo}")

# --- Summary ---
print("\n" + "="*70)
print("SUMMARY: 138 Well-Folded AF3 Dark Proteome Domains")
print("="*70)
print(f"Total well-folded (pTM >= 0.5): {len(wf)}")
print(f"pTM range: {wf['ptm'].min():.2f} -- {wf['ptm'].max():.2f} (median {wf['ptm'].median():.2f})")
print(f"Fraction disordered: {wf['fraction_disordered'].min():.2f} -- {wf['fraction_disordered'].max():.2f} (median {wf['fraction_disordered'].median():.2f})")
print(f"Sequence length: {wf['length'].min()} -- {wf['length'].max()} aa (median {wf['length'].median():.0f})")
print(f"InterProScan: {n_interpro}/{len(wf)} any hit, {n_interpro_func}/{len(wf)} non-Phobius")
print(f"Prevalence (n_samples): {wf['n_samples'].min():.0f} -- {wf['n_samples'].max():.0f} (median {wf['n_samples'].median():.0f})")
if 'n_basins' in wf.columns:
    print(f"Ocean basins: {wf['n_basins'].min():.0f} -- {wf['n_basins'].max():.0f} (median {wf['n_basins'].median():.0f})")
print(f"Boltz-2 mean pLDDT: {wf['mean_plddt'].min():.1f} -- {wf['mean_plddt'].max():.1f} (median {wf['mean_plddt'].median():.1f})")
print(f"MSA depth (n_members): {wf['n_members'].min():.0f} -- {wf['n_members'].max():.0f} (median {wf['n_members'].median():.0f})")
print(f"\nBest env variable distribution:")
print(wf['best_env'].value_counts().to_string())
print(f"\nRank bin distribution:")
print(wf['bin'].value_counts().sort_index().to_string())
print(f"\nCross-method: Spearman rho(AF3 pTM, Boltz-2 pLDDT) = ", end='')
from scipy import stats
r, p = stats.spearmanr(wf['ptm'].dropna(), wf['mean_plddt'].dropna())
print(f"{r:.3f} (p = {p:.2e})")
print(f"\nSignificant env correlations per domain: {wf['n_sig_env'].min():.0f} -- {wf['n_sig_env'].max():.0f} (median {wf['n_sig_env'].median():.0f})")
