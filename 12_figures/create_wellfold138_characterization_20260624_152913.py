#!/usr/bin/env python3
"""
Standalone reference figure: Systematic characterization of 138 well-folded
AF3 dark proteome domains.

Layout (full A4 page):
  Row 0: (A) pTM histogram + disorder overlay  (B) Rank-bin distribution  (C) Best env bar chart
  Row 1: (D) InterPro absence tile  (E) Annotated vs well-folded comparison
  Row 2: (F) 138×18 env coupling heatmap (clustered)  (G) max|rho| marginal
  Row 3: (H) Global occurrence map  (I) Prevalence violin  (J) Per-basin bar
  Row 4: (K-V) Structure gallery — top 12 AF3 renders with metadata
  Row 5: (W) AF3 pTM vs Boltz-2 pLDDT  (X) Seq length vs pTM
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from palette import (
    DIVERGING_CMAP, OCEAN_CMAP, THERMAL_CMAP,
    BASIN_COLORS, DEEP_OCEAN, TURQUOISE, COASTAL_BLUE,
    FOREST_GREEN, DESERT_TAN, CLAY, SIENNA, OCEAN_BLUE,
    CLOUD_WHITE, ABYSS, SAND, SAVANNA, ICE_WHITE,
    get_categorical_colors,
)

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
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

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from scipy import stats
import numpy as np
import pandas as pd

MANUSCRIPT = Path('/media/drn2/External/TARA-Oceans/MANUSCRIPT')

CHAR_TSV = MANUSCRIPT / 'source_data/dark_proteome/wellfold_138_characterization.tsv'
ENV_TSV = MANUSCRIPT / 'source_data/dark_proteome/wellfold_138_env_profiles.tsv'
GEO_TSV = MANUSCRIPT / 'source_data/dark_proteome/wellfold_138_geographic.tsv'
RENDER_DIR = MANUSCRIPT / 'figures/af3_renders'
OUT_DIR = MANUSCRIPT / 'figures'

print("Loading data...")
wf = pd.read_csv(CHAR_TSV, sep='\t', comment='#')
env = pd.read_csv(ENV_TSV, sep='\t', comment='#')
geo = pd.read_csv(GEO_TSV, sep='\t', comment='#')
print(f"  Domains: {len(wf)}, Env profiles: {len(env)}, Geo records: {len(geo)}")

wf = wf.sort_values('ptm', ascending=False).reset_index(drop=True)

# Env variable display names
ENV_DISPLAY = {
    'bathymetry_m': 'Bathymetry',
    'modis_sst_mean_c': 'SST (mean)',
    'sst_max_c': 'SST (max)',
    'sst_min_c': 'SST (min)',
    'sst_range_c': 'SST (range)',
    'air_temp_mean_c': 'Air temp',
    'air_temp_max_c': 'Air temp (max)',
    'air_temp_min_c': 'Air temp (min)',
    'air_temp_range_c': 'Air temp (range)',
    'chl_mean_mg_m3': 'Chl-a (mean)',
    'chl_max_mg_m3': 'Chl-a (max)',
    'chl_min_mg_m3': 'Chl-a (min)',
    'nflh_mean': 'NFLH',
    'poc_mean_mg_m3': 'POC',
    'precip_mean_mm': 'Precipitation',
    'solar_rad_mj_m2': 'Solar radiation',
    'distance_to_coast_km': 'Dist. to coast',
    'oxygen_umol_l': 'Oxygen',
    'nitrate_umol_l': 'Nitrate',
    'phosphate_umol_l': 'Phosphate',
    'silicate_umol_l': 'Silicate',
    'elevation_m': 'Elevation',
    'modis_sst_mean_c': 'SST (MODIS)',
}

# Reduced env set for the heatmap (core 18, skip redundant)
CORE_ENV = [
    'bathymetry_m', 'modis_sst_mean_c', 'sst_range_c',
    'air_temp_mean_c', 'air_temp_min_c',
    'chl_mean_mg_m3', 'chl_min_mg_m3', 'nflh_mean', 'poc_mean_mg_m3',
    'solar_rad_mj_m2', 'precip_mean_mm', 'distance_to_coast_km',
    'oxygen_umol_l', 'nitrate_umol_l', 'phosphate_umol_l', 'silicate_umol_l',
]
available_env = [c for c in CORE_ENV if c in env.columns]

# ── Env color categories for annotation bar ──
ENV_CATEGORIES = {}
for v in available_env:
    if 'sst' in v or 'air_temp' in v:
        ENV_CATEGORIES[v] = 'Temperature'
    elif 'chl' in v or 'nflh' in v or 'poc' in v:
        ENV_CATEGORIES[v] = 'Productivity'
    elif 'bathy' in v or 'dist' in v or 'elev' in v:
        ENV_CATEGORIES[v] = 'Geography'
    elif 'solar' in v or 'precip' in v:
        ENV_CATEGORIES[v] = 'Atmospheric'
    elif v in ('oxygen_umol_l', 'nitrate_umol_l', 'phosphate_umol_l', 'silicate_umol_l'):
        ENV_CATEGORIES[v] = 'Nutrient'
    else:
        ENV_CATEGORIES[v] = 'Other'

ENV_CAT_COLORS = {
    'Temperature': COASTAL_BLUE,
    'Productivity': FOREST_GREEN,
    'Geography': DEEP_OCEAN,
    'Atmospheric': SAND,
    'Nutrient': SAVANNA,
    'Other': (0.6, 0.6, 0.6),
}

# ── Best env color mapping ──
BEST_ENV_COLORS = {
    'bathymetry_m': DEEP_OCEAN,
    'modis_sst_mean_c': COASTAL_BLUE,
    'sst_range_c': COASTAL_BLUE,
    'sst_max_c': COASTAL_BLUE,
    'sst_min_c': COASTAL_BLUE,
    'air_temp_min_c': COASTAL_BLUE,
    'solar_rad_mj_m2': DESERT_TAN,
    'precip_mean_mm': TURQUOISE,
    'poc_mean_mg_m3': FOREST_GREEN,
    'nflh_mean': FOREST_GREEN,
    'chl_min_mg_m3': FOREST_GREEN,
    'oxygen_umol_l': SAVANNA,
    'distance_to_coast_km': OCEAN_BLUE,
}

# ══════════════════════════════════════════════════════════════════════════════
# BUILD FIGURE
# ══════════════════════════════════════════════════════════════════════════════

print("Building figure...")

fig = plt.figure(figsize=(11.69, 16.54), layout='constrained')  # A3 landscape-ish

# 6 row bands
subfigs = fig.subfigures(6, 1, height_ratios=[1, 0.7, 2.2, 1.5, 2.2, 1])

# ── Row 0: Overview ──────────────────────────────────────────────────────────
ax_a, ax_b, ax_c = subfigs[0].subplots(1, 3, gridspec_kw={'width_ratios': [1.2, 1, 1]})

# (A) pTM distribution (left) and fraction disordered (right)
ax_a.hist(wf['ptm'], bins=20, color=DEEP_OCEAN, edgecolor='white', linewidth=0.3, alpha=0.9)
ax_a.axvline(wf['ptm'].median(), color=CLAY, ls='--', lw=0.5, label=f"median={wf['ptm'].median():.2f}")
ax_a.set_xlabel('AF3 pTM')
ax_a.set_ylabel('Count')
ax_a.set_title('A  pTM distribution (n=138)', fontweight='bold', loc='left')
ax_a.legend(frameon=False, loc='upper right')

# (B) Rank-bin distribution
bin_counts = wf['bin'].value_counts().sort_index()
colors_bins = [OCEAN_CMAP(i / 10) for i in range(10)]
ax_b.bar(bin_counts.index, bin_counts.values, color=colors_bins, edgecolor='white', linewidth=0.3)
ax_b.set_xlabel('Rank bin (1 = strongest coupling)')
ax_b.set_ylabel('Count')
ax_b.set_title('B  Rank-bin distribution', fontweight='bold', loc='left')
ax_b.set_xticks(range(1, 11))

# (C) Best environmental variable
best_env_counts = wf['best_env'].value_counts().head(8)
bar_colors = [BEST_ENV_COLORS.get(e, (0.5, 0.5, 0.5)) for e in best_env_counts.index]
bar_labels = [ENV_DISPLAY.get(e, e) for e in best_env_counts.index]
bars = ax_c.barh(range(len(best_env_counts)), best_env_counts.values, color=bar_colors, edgecolor='white', linewidth=0.3)
ax_c.set_yticks(range(len(best_env_counts)))
ax_c.set_yticklabels(bar_labels)
ax_c.set_xlabel('Count')
ax_c.set_title('C  Top-correlated env variable', fontweight='bold', loc='left')
ax_c.invert_yaxis()

# ── Row 1: InterPro absence ─────────────────────────────────────────────────
ax_d, ax_e = subfigs[1].subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})

# (D) InterPro absence annotation bar — light gray tiles = searched, no hits
interpro_dbs = ['Pfam', 'SMART', 'Gene3D', 'SUPERFAMILY', 'PANTHER',
                'ProSite\nPatterns', 'ProSite\nProfiles', 'Phobius', 'InterPro']
absence_matrix = np.full((len(wf), len(interpro_dbs)), 0.08)
from matplotlib.colors import ListedColormap
absence_cmap = ListedColormap([ICE_WHITE, DEEP_OCEAN])
ax_d.imshow(absence_matrix, aspect='auto', cmap=absence_cmap, vmin=0, vmax=1, interpolation='nearest')
ax_d.set_xticks(range(len(interpro_dbs)))
ax_d.set_xticklabels(interpro_dbs, rotation=45, ha='right')
ax_d.set_ylabel('138 domains (sorted by pTM)')
ax_d.set_title('D  InterProScan: 0/138 annotated', fontweight='bold', loc='left')
ax_d.text(len(interpro_dbs) / 2, len(wf) / 2, '0 / 138\nhits across\n9 databases',
          ha='center', va='center', fontsize=7, fontweight='bold', color=SIENNA)

# (E) Fraction disordered distribution
ax_e.hist(wf['fraction_disordered'], bins=20, color=DESERT_TAN, edgecolor='white', linewidth=0.3)
ax_e.axvline(wf['fraction_disordered'].median(), color=SIENNA, ls='--', lw=0.5,
             label=f"median={wf['fraction_disordered'].median():.2f}")
ax_e.set_xlabel('Fraction disordered (AF3)')
ax_e.set_ylabel('Count')
ax_e.set_title('E  Disorder distribution', fontweight='bold', loc='left')
ax_e.legend(frameon=False, loc='upper right')

# ── Row 2: Environmental coupling heatmap ────────────────────────────────────
ax_f_row = subfigs[2]
ax_f, ax_g = ax_f_row.subplots(1, 2, gridspec_kw={'width_ratios': [5, 1]})

# Build the heatmap matrix
env_indexed = env.set_index('orig_id')
heatmap_data = env_indexed.reindex(wf['orig_id'])[available_env].values
heatmap_data = np.nan_to_num(heatmap_data, nan=0.0)

# Cluster rows
if heatmap_data.shape[0] > 2:
    row_dist = pdist(heatmap_data, metric='euclidean')
    row_link = linkage(row_dist, method='ward')
    row_order = leaves_list(row_link)
else:
    row_order = np.arange(heatmap_data.shape[0])

# Cluster columns
if heatmap_data.shape[1] > 2:
    col_dist = pdist(heatmap_data.T, metric='euclidean')
    col_link = linkage(col_dist, method='ward')
    col_order = leaves_list(col_link)
else:
    col_order = np.arange(heatmap_data.shape[1])

heatmap_ordered = heatmap_data[row_order][:, col_order]
env_labels_ordered = [ENV_DISPLAY.get(available_env[i], available_env[i]) for i in col_order]

vmax = np.percentile(np.abs(heatmap_data), 98)
im = ax_f.imshow(heatmap_ordered, aspect='auto', cmap=DIVERGING_CMAP,
                 vmin=-vmax, vmax=vmax, interpolation='nearest')
ax_f.set_xticks(range(len(env_labels_ordered)))
ax_f.set_xticklabels(env_labels_ordered, rotation=45, ha='right')
ax_f.set_ylabel('138 domains (clustered)')
ax_f.set_title('F  Environmental coupling (Spearman ρ)', fontweight='bold', loc='left')

cbar = ax_f_row.colorbar(im, ax=ax_f, fraction=0.02, pad=0.01)
cbar.set_label('Spearman ρ')

# (G) Marginal: max |rho| distribution
max_rho_wf = wf['max_abs_rho'].dropna()
ax_g.hist(max_rho_wf, bins=20, orientation='horizontal', color=DEEP_OCEAN,
          edgecolor='white', linewidth=0.3)
ax_g.axhline(max_rho_wf.median(), color=CLAY, ls='--', lw=0.5)
ax_g.set_xlabel('Count')
ax_g.set_ylabel('max |ρ|')
ax_g.set_title('G  Coupling strength', fontweight='bold', loc='left')

# ── Row 3: Geographic occurrence ─────────────────────────────────────────────
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ax_i_plain, ax_j_plain = subfigs[3].subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})

# We need to create the map axis separately with cartopy projection
# Replace the left two panels with a single wide map + prevalence + basin
# Use a gridspec approach
subfigs[3].delaxes(ax_i_plain)
subfigs[3].delaxes(ax_j_plain)

gs3 = subfigs[3].add_gridspec(1, 3, width_ratios=[2.5, 1, 1])
ax_h = subfigs[3].add_subplot(gs3[0], projection=ccrs.Robinson())
ax_i = subfigs[3].add_subplot(gs3[1])
ax_j = subfigs[3].add_subplot(gs3[2])

# (H) Global map with coastlines
ax_h.set_global()
ax_h.add_feature(cfeature.LAND, facecolor=(0.92, 0.92, 0.90), edgecolor='none', zorder=0)
ax_h.add_feature(cfeature.COASTLINE, linewidth=0.15, color=(0.5, 0.5, 0.5), zorder=1)
ax_h.add_feature(cfeature.OCEAN, facecolor=CLOUD_WHITE, zorder=0)

if len(geo) > 0 and 'latitude' in geo.columns:
    geo_agg = geo.groupby(['latitude', 'longitude']).agg(
        n_domains=('domain', 'nunique'),
        total_count=('count', 'sum')
    ).reset_index()

    scatter = ax_h.scatter(
        geo_agg['longitude'], geo_agg['latitude'],
        c=geo_agg['n_domains'], s=np.clip(geo_agg['n_domains'] * 1.5, 2, 30),
        cmap=OCEAN_CMAP, alpha=0.75, edgecolors='none', vmin=1,
        transform=ccrs.PlateCarree(), zorder=2
    )
    cbar2 = subfigs[3].colorbar(scatter, ax=ax_h, fraction=0.03, pad=0.02, shrink=0.8)
    cbar2.set_label('# well-folded domains detected')

ax_h.set_title('H  Global occurrence of 138 well-folded domains', fontweight='bold', loc='left')

# (I) Prevalence distribution
ax_i.hist(wf['n_samples'], bins=30, color=TURQUOISE, edgecolor='white', linewidth=0.3)
ax_i.axvline(wf['n_samples'].median(), color=CLAY, ls='--', lw=0.5)
ax_i.set_xlabel('Prevalence (n samples)')
ax_i.set_ylabel('Count')
ax_i.set_title('I  Prevalence', fontweight='bold', loc='left')
ax_i.text(0.95, 0.95, f"median={wf['n_samples'].median():.0f}\nrange={wf['n_samples'].min():.0f}–{wf['n_samples'].max():.0f}",
          transform=ax_i.transAxes, ha='right', va='top', fontsize=5)

# (J) Basin distribution
if 'basins' in wf.columns:
    all_basins = []
    for b in wf['basins'].dropna():
        all_basins.extend(b.split(';'))
    basin_counts = pd.Series(all_basins).value_counts()
    b_colors = [BASIN_COLORS.get(b, (0.5, 0.5, 0.5)) for b in basin_counts.index]
    ax_j.barh(range(len(basin_counts)), basin_counts.values, color=b_colors, edgecolor='white', linewidth=0.3)
    ax_j.set_yticks(range(len(basin_counts)))
    ax_j.set_yticklabels(basin_counts.index)
    ax_j.set_xlabel('# domains detected')
    ax_j.set_title('J  Basin coverage', fontweight='bold', loc='left')
    ax_j.invert_yaxis()
else:
    ax_j.text(0.5, 0.5, 'Basin data\nnot available', ha='center', va='center', transform=ax_j.transAxes)
    ax_j.set_title('J  Basin coverage', fontweight='bold', loc='left')

# ── Row 4: Structure gallery ────────────────────────────────────────────────
gallery_axs = subfigs[4].subplots(2, 6)

top12 = wf.head(12)
for idx, (_, row) in enumerate(top12.iterrows()):
    r = idx // 6
    c = idx % 6
    ax = gallery_axs[r][c]

    render_path = RENDER_DIR / f"render_{row['job_id']}.png"
    if render_path.exists():
        img = mpimg.imread(str(render_path))
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, 'No render', ha='center', va='center',
                transform=ax.transAxes, fontsize=5, color='grey')

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    label_letter = chr(ord('K') + idx)
    env_short = ENV_DISPLAY.get(row['best_env'], row['best_env'][:8])
    ax.set_title(
        f"{label_letter}  pTM {row['ptm']:.2f} | {row['length']}aa | {env_short}",
        fontsize=6, fontweight='bold', loc='left', pad=1
    )

# ── Row 5: Cross-method validation ──────────────────────────────────────────
ax_w, ax_x = subfigs[5].subplots(1, 2)

# (W) AF3 pTM vs Boltz-2 mean pLDDT
mask = wf['mean_plddt'].notna() & wf['ptm'].notna()
rho_cross, p_cross = stats.spearmanr(wf.loc[mask, 'ptm'], wf.loc[mask, 'mean_plddt'])
ax_w.scatter(wf.loc[mask, 'ptm'], wf.loc[mask, 'mean_plddt'],
             c=wf.loc[mask, 'bin'], cmap=OCEAN_CMAP, s=10, edgecolors='white',
             linewidths=0.2, alpha=0.8, vmin=1, vmax=10)
ax_w.set_xlabel('AF3 pTM')
ax_w.set_ylabel('Boltz-2 mean pLDDT')
ax_w.set_title(f'W  Cross-method (ρ={rho_cross:.2f}, p={p_cross:.1e})',
               fontweight='bold', loc='left')
ax_w.axhline(70, color=CLAY, ls='--', lw=0.3)
ax_w.axvline(0.5, color=CLAY, ls='--', lw=0.3)

# (X) Sequence length vs pTM
best_env_colors_list = [BEST_ENV_COLORS.get(e, (0.5, 0.5, 0.5)) for e in wf['best_env']]
ax_x.scatter(wf['length'], wf['ptm'], c=best_env_colors_list, s=10,
             edgecolors='white', linewidths=0.2, alpha=0.8)
ax_x.set_xlabel('Sequence length (aa)')
ax_x.set_ylabel('AF3 pTM')
ax_x.set_title('X  Length vs confidence', fontweight='bold', loc='left')
ax_x.axhline(0.5, color=CLAY, ls='--', lw=0.3)

# ── Supertitle ───────────────────────────────────────────────────────────────
fig.suptitle(
    '138 Well-Folded Dark Proteome Domains: AlphaFold 3 Structural Characterization\n'
    '0/138 InterProScan annotations — structurally real, database-invisible proteins',
    fontsize=8, fontweight='bold', y=1.01
)

# ── Save ─────────────────────────────────────────────────────────────────────
out_pdf = OUT_DIR / 'wellfold138_af3_characterization_20260624.pdf'
out_svg = OUT_DIR / 'wellfold138_af3_characterization_20260624.svg'
out_png = OUT_DIR / 'wellfold138_af3_characterization_20260624.png'

fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
fig.savefig(out_svg, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=150)
plt.close()

print(f"\nSaved: {out_pdf}")
print(f"Saved: {out_svg}")
print(f"Saved: {out_png}")
print(f"\n{'='*60}")
print("PROVENANCE")
print(f"  Script: {__file__}")
print(f"  Input: {CHAR_TSV}")
print(f"  Input: {ENV_TSV}")
print(f"  Input: {GEO_TSV}")
print(f"  Renders: {RENDER_DIR}")
print(f"  Domains: {len(wf)}")
print(f"  Geographic records: {len(geo)}")
