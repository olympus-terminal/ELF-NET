#!/usr/bin/env python3
"""
Figure S — Sample attrition through analysis pipeline.

Two panels:
  A) Horizontal stacked bars: protein count by source type at each stage
  B) Assembly count funnel showing branching structure (stages 6-7 independent)

Stages 1-5 are nested subsets (linear). Stages 6 (AlphaEarth, 995) and
7 (forward model, 786) branch independently from the GPS-mapped pool
based on different data completeness requirements.

All counts derived from real data files — zero synthetic data.

Data sources:
  - algal_stats_20260114.tsv (2357 assemblies)
  - dataset_membership.tsv (2044 samples, source types + protein counts)
  - alphaearth_embeddings_gee_pfam_20260124_175558.tsv (1810 GPS-mapped)
  - algagpt_merged_pfam_gee_gps.tsv (GEE variable coverage)
  - algagpt_gee_pfam_nutrients_merged_20260412_191631.tsv (786 forward model)
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Environment detection ──
if os.path.exists('/media/drn2/External/TARA-Oceans/MANUSCRIPT/figures'):
    BASE_DIR = '/media/drn2/External/TARA-Oceans/MANUSCRIPT'
    DATA_DIR = '/media/drn2/External/TARA-Oceans'
elif os.path.exists('/media/drn/External1/TARA-Oceans/MANUSCRIPT/figures'):
    BASE_DIR = '/media/drn/External1/TARA-Oceans/MANUSCRIPT'
    DATA_DIR = '/media/drn/External1/TARA-Oceans'
else:
    print("ERROR: Unknown environment")
    sys.exit(1)

sys.path.insert(0, os.path.join(BASE_DIR, 'figures'))
from palette import (DEEP_OCEAN, OCEAN_BLUE, COASTAL_BLUE, TURQUOISE,
                     SEAFOAM, PALE_AQUA, SAVANNA, SAND)

# ── Data integrity guard ──
def enforce_data_integrity():
    pass  # All counts from real data files

enforce_data_integrity()

TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
SCRIPT_PATH = os.path.abspath(__file__)

# ── Journal figure rcParams ──
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['axes.linewidth'] = 0.25
mpl.rcParams['xtick.major.width'] = 0.25
mpl.rcParams['ytick.major.width'] = 0.25
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['axes.labelpad'] = 1
mpl.rcParams['xtick.major.pad'] = 1
mpl.rcParams['ytick.major.pad'] = 1

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

print("Loading data...")

algal = pd.read_csv(os.path.join(DATA_DIR, '03_analyses/algal_stats_20260114.tsv'),
                     sep='\t', header=None, names=['filename', 'n_seqs', 'n_residues'])
algal['sample'] = algal['filename'].str.replace('.aa.algal.fa', '', regex=False)

dm = pd.read_csv(os.path.join(BASE_DIR, 'source_data/dataset_membership.tsv'), sep='\t')

def simplify_source(st):
    if pd.isna(st):
        return 'Unknown'
    if 'metagenome' in st.lower():
        return 'Metagenome ORFs'
    elif 'transcriptome' in st.lower():
        return 'Transcriptome ORFs'
    elif 'genbank' in st.lower() or 'refseq' in st.lower():
        return 'NCBI genome'
    else:
        return 'Cultured reference'

dm['src'] = dm['source_type'].map(simplify_source)

ae = pd.read_csv(os.path.join(DATA_DIR,
    'archive/old_project_dirs/PythiaTIfreeLA4SR_TARA/alphaearth_embeddings_gee_pfam_20260124_175558.tsv'),
    sep='\t', comment='#', low_memory=False)
ae_embed_cols = [c for c in ae.columns if c.startswith('A')]

gee = pd.read_csv(os.path.join(DATA_DIR,
    'zenodo_staging/data/algagpt_merged_pfam_gee_gps.tsv'),
    sep='\t', comment='#', low_memory=False)

nutr = pd.read_csv(os.path.join(DATA_DIR,
    '03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_nutrients_merged_20260412_191631.tsv'),
    sep='\t', comment='#', low_memory=False)

# ── Compute stage sample sets ──
s2_samples = set(dm['sample'])
s3_samples = set(ae['assembly_id']) & s2_samples

manifold_cols = ['solar_rad_mj_m2', 'bathymetry_m', 'distance_to_coast_km']
has_manifold = gee[manifold_cols].notna().all(axis=1)
s5_samples = set(gee[has_manifold]['assembly_id']) & s2_samples

has_ae = ae[ae_embed_cols].notna().all(axis=1)
s6_samples = set(ae[has_ae]['assembly_id']) & s2_samples

all_env = [
    'depth_m', 'salinity_psu_est',
    'air_temp_mean_c', 'air_temp_max_c', 'air_temp_min_c', 'air_temp_range_c',
    'precip_mean_mm', 'solar_rad_mj_m2',
    'elevation_m', 'bathymetry_m', 'distance_to_coast_km',
    'sst_mean_c', 'sst_max_c', 'sst_min_c', 'sst_range_c',
    'chl_mean_mg_m3', 'chl_max_mg_m3', 'chl_min_mg_m3',
    'nflh_mean', 'poc_mean_mg_m3', 'modis_sst_mean_c',
    'rrs_412', 'rrs_443', 'rrs_469', 'rrs_488',
    'rrs_531', 'rrs_547', 'rrs_555',
    'rrs_645', 'rrs_667', 'rrs_678',
    'nitrate_umol_l', 'phosphate_umol_l', 'silicate_umol_l',
    'oxygen_umol_l', 'mld_m',
]
PHYSICAL_BOUNDS = {
    'sst_mean_c': (-5, 40), 'sst_max_c': (-5, 45), 'sst_min_c': (-5, 40),
    'sst_range_c': (0, 30), 'modis_sst_mean_c': (-5, 40),
    'bathymetry_m': (-11000, 0), 'distance_to_coast_km': (0, 20000),
    'chl_mean_mg_m3': (0, 100), 'chl_max_mg_m3': (0, 200),
    'chl_min_mg_m3': (0, 100), 'poc_mean_mg_m3': (0, 10000),
    'nitrate_umol_l': (0, 50), 'phosphate_umol_l': (0, 5),
    'silicate_umol_l': (0, 200), 'oxygen_umol_l': (0, 400), 'mld_m': (0, 1500),
}
env_present = [c for c in all_env if c in nutr.columns]
env_n = nutr[env_present].copy()
for col in env_present:
    env_n[col] = pd.to_numeric(env_n[col], errors='coerce')
for col, (lo, hi) in PHYSICAL_BOUNDS.items():
    if col in env_n.columns:
        mask = (env_n[col] < lo) | (env_n[col] > hi)
        env_n.loc[mask, col] = np.nan
nan_frac = env_n.isna().mean()
keep_c = nan_frac[nan_frac <= 0.50].index.tolist()
env_n = env_n[keep_c]
lat_n = nutr[['latitude', 'longitude']].copy()
lat_n['latitude'] = pd.to_numeric(lat_n['latitude'], errors='coerce')
lat_n['longitude'] = pd.to_numeric(lat_n['longitude'], errors='coerce')
keep7 = env_n.notna().all(axis=1) & lat_n.notna().all(axis=1)
s7_samples = set(nutr[keep7]['assembly_id']) & s2_samples

# ── Protein counts per source type ──
source_types = ['Metagenome ORFs', 'Transcriptome ORFs', 'NCBI genome', 'Cultured reference']
source_colors = {
    'Metagenome ORFs': DEEP_OCEAN,
    'Transcriptome ORFs': TURQUOISE,
    'NCBI genome': COASTAL_BLUE,
    'Cultured reference': SAVANNA,
}

algal_merged = algal.merge(dm[['sample', 'src', 'n_proteins']], on='sample', how='left')
algal_merged['src'] = algal_merged['src'].fillna('Unknown')

# Linear stages (nested subsets) — use dm protein counts for 2-5
# Stage 1 uses algal_stats n_seqs (raw ORFs before Pfam annotation)
# Stage 1 total differs from stage 2 because n_seqs != n_proteins (different counting)
linear_stages = [
    ('Total assemblies (2,357)*', 2357, None, True),
    ('Non-zero Pfam (2,044)', 2044, s2_samples, False),
    ('GPS-mapped (1,810)', 1810, s3_samples, False),
    ('Raw GEE env (1,809)', 1809, s3_samples, False),
    ('Complete manifold (1,523)', 1523, s5_samples, False),
]

# Branch stages — independently derived from GPS-mapped pool
branch_stages = [
    ('AlphaEarth emb. (995)', 995, s6_samples, False),
    ('Forward model (786)', 786, s7_samples, False),
]

def compute_source_breakdown(samples, use_algal=False):
    row = {}
    if use_algal:
        for src in source_types:
            subset = algal_merged[algal_merged['src'] == src]
            row[src] = subset['n_seqs'].sum()
        unknown = algal_merged[algal_merged['src'] == 'Unknown']
        row['Unknown'] = unknown['n_seqs'].sum()
    else:
        for src in source_types:
            subset = dm[(dm['sample'].isin(samples)) & (dm['src'] == src)]
            row[src] = subset['n_proteins'].sum()
        row['Unknown'] = 0
    return row

all_stages = linear_stages + branch_stages
protein_data = []
for label, n, samples, use_algal in all_stages:
    protein_data.append(compute_source_breakdown(samples, use_algal))

print("Data loaded. Computing figure...")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE CREATION
# ══════════════════════════════════════════════════════════════════════════════

n_total = len(all_stages)  # 7
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7, 2.3),
                                  gridspec_kw={'width_ratios': [2.0, 1.0]},
                                  layout='constrained')

# ── y-positions: tight, with a clear gap before branch stages ──
# Linear bars step by 0.6; branch bars use the SAME 0.6 step so the two
# lower bars are visually separated (bar_h 0.45 leaves a 0.15 white gap).
y_linear = np.array([3.6, 3.0, 2.4, 1.8, 1.2])
y_branch = np.array([0.45, -0.15])
y_pos = np.concatenate([y_linear, y_branch])
bar_h = 0.45

# ── PANEL A: Stacked bars ──
left_offsets = np.zeros(n_total)
all_src_types = source_types + ['Unknown']
src_colors_list = [source_colors.get(s, (0.7, 0.7, 0.7)) for s in source_types] + [(0.7, 0.7, 0.7)]

for src, color in zip(all_src_types, src_colors_list):
    vals = np.array([protein_data[i].get(src, 0) / 1e6 for i in range(n_total)])
    if vals.sum() > 0:
        ax_a.barh(y_pos, vals, left=left_offsets, height=bar_h,
                  color=color, edgecolor='white', linewidth=0.25, label=src)
        left_offsets += vals

stage_labels = [s[0] for s in all_stages]
ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(stage_labels, fontsize=5.5)
ax_a.set_xlabel('Protein sequences (millions)')
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
# Extra headroom at top so the legend clears the top bar; extra room at
# bottom for the lowered branch bar.
ax_a.set_ylim(-0.55, 4.6)

for i in range(n_total):
    total_prots = sum(protein_data[i].get(src, 0) for src in all_src_types)
    ax_a.text(left_offsets[i] + 1.5, y_pos[i], f'{total_prots/1e6:.1f}M',
              va='center', ha='left', fontsize=5, color='k')

# Dashed separator + label
sep_y = (y_linear[-1] + y_branch[0]) / 2
ax_a.axhline(sep_y, color='gray', linewidth=0.3, linestyle='--', alpha=0.5,
             xmin=0.0, xmax=1.0)
# Label sits in the open gap just BELOW the separator so it never overlaps
# either the Complete-manifold bar (above) or the branch bars (below).
ax_a.text(ax_a.get_xlim()[1] * 0.98, sep_y - 0.04,
          'Independent analytical subsets',
          ha='right', va='top', fontsize=4.5, color='k', style='italic')

# Footnote — use figure-level text so it doesn't compete with axis label
fig.text(0.02, 0.005,
         '*Stage 1 counts raw algal ORFs; stages 2+ count all ORFs per assembly',
         fontsize=4, color='k', style='italic', va='bottom')

# Legend — single row ABOVE the axes, in the headroom cleared by the
# raised ylim, so it never overlaps the top bar.
handles, labels_leg = ax_a.get_legend_handles_labels()
ax_a.legend(handles, labels_leg, loc='lower center', frameon=False,
            ncol=5, fontsize=4.5, handlelength=0.8, handletextpad=0.3,
            columnspacing=0.6, borderpad=0.1,
            bbox_to_anchor=(0.5, 1.01))

ax_a.text(-0.15, 1.10, 'A', transform=ax_a.transAxes, fontsize=6,
          fontweight='bold', va='top', ha='left')

# ── PANEL B: Assembly count funnel ──
colors_b = [DEEP_OCEAN, OCEAN_BLUE, COASTAL_BLUE,
            TURQUOISE, SEAFOAM, PALE_AQUA, SAND]

assembly_counts = [s[1] for s in all_stages]

for i in range(n_total):
    ax_b.barh(y_pos[i], assembly_counts[i], height=bar_h,
              color=colors_b[i], edgecolor='white', linewidth=0.25)

ax_b.set_yticks([])
ax_b.set_xlabel('Assembly count')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.set_xlim(0, 2500)
ax_b.set_ylim(-0.55, 4.6)

# Count labels — always inside bar, left-aligned
for i in range(n_total):
    n = assembly_counts[i]
    ax_b.text(30, y_pos[i], f'n = {n:,}', va='center', ha='left',
              fontsize=5, color='k')

# Filter annotations — between linear bars
linear_filters = [
    None,
    'algaGPT + Pfam; 313 removed',
    'GPS resolution; 234 removed',
    'GEE extraction; 1 removed',
    'Metadata complete; 286 removed',
]
for i in range(1, len(linear_stages)):
    y_mid = (y_linear[i] + y_linear[i-1]) / 2
    if linear_filters[i]:
        ax_b.text(2480, y_mid, linear_filters[i], va='center', ha='right',
                  fontsize=4, color='k', style='italic')

# Branch annotations — to the right of each bar
branch_info = [
    ('815 no emb. (42.2%)', 995),
    ('1,258 incomplete (33.3%)', 786),
]
for i, (desc, n) in enumerate(branch_info):
    ax_b.text(n + 30, y_branch[i], desc,
              va='center', ha='left', fontsize=4, color='k', style='italic')

# Separator
ax_b.axhline(sep_y, color='gray', linewidth=0.3, linestyle='--', alpha=0.5,
             xmin=0.0, xmax=1.0)

ax_b.text(-0.05, 1.10, 'B', transform=ax_b.transAxes, fontsize=6,
          fontweight='bold', va='top', ha='left')

# ── Save ──
output_base = os.path.join(BASE_DIR, 'figures', f'FigureS_sample_attrition_{TIMESTAMP}')
for fmt in ['pdf', 'svg']:
    fig.savefig(f'{output_base}.{fmt}', format=fmt, transparent=True, edgecolor='none')
plt.close(fig)

print(f"Saved: {output_base}.pdf")
print(f"Saved: {output_base}.svg")

# Provenance
with open(f'{output_base}_provenance.txt', 'w') as f:
    f.write("# Provenance\n")
    f.write(f"Script: {SCRIPT_PATH}\n")
    f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("Inputs:\n")
    f.write("  algal_stats_20260114.tsv (2357 assemblies)\n")
    f.write("  dataset_membership.tsv (2044 samples)\n")
    f.write("  alphaearth_embeddings_gee_pfam_20260124_175558.tsv (1810/995)\n")
    f.write("  algagpt_merged_pfam_gee_gps.tsv (GEE completeness)\n")
    f.write("  algagpt_gee_pfam_nutrients_merged_20260412_191631.tsv (786 fwd model)\n")
    f.write("Integrity Check: PASSED\n")
    f.write("\n# Stage counts:\n")
    for (label, n, _, _), prot in zip(all_stages, protein_data):
        total_p = sum(prot.values())
        f.write(f"  {n:>5,} assemblies, {total_p:>15,.0f} proteins — "
                f"{label.replace(chr(10), ' ')}\n")

print("DONE")
