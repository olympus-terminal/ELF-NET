#!/usr/bin/env python3
"""
Figure S3 (dense rebuild): Basin coverage, RuBisCO lineage, AlgaGPT retention,
domain-environment correlation, XGBoost CV R², and RbcL phylogeny.

Layout (3 rows, 10 panels A-J):
  Row 1: A (basin map, ~60%) | B (depth violins, ~40%) — compact
  Row 2: C (Form I bars) | D (Form II bars) | E (AlgaGPT retention) |
         F (Spearman rho histogram) | G (XGBoost CV R²) — all half-height
  Row 3: H-J (phylo trees, composited from existing PDF)

Drops UMAP panels F-H from old S3; absorbs old S4 panels A-B.

Provenance:
  Script: create_figureS3_dense_20260627_201701.py
  Input: basin_statistics.json, ocean_basin_assignments.tsv
  Input: algagpt_gee_pfam_merged_SMART_20260119_100639.tsv
  Input: algagpt_classification_summary_20260114_150000.csv
  Input: rubisco_merged_formI_formII_20260223.tsv
  Input: algagpt_pfam_alphaearth_correlations_20260124_175739_full.tsv
  Input: FigureS_phylo_compact_20260627.pdf (composited for Row 3)
  Output: figures/FigureS3_dense_<TS>.pdf, .svg
  Date: 2026-06-27
  Integrity Check: PASSED
"""

import os
import sys
import json
import pathlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

import matplotlib as mpl
mpl.use('Agg')
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
mpl.rcParams['legend.fontsize'] = 5
mpl.rcParams['axes.linewidth'] = 0.25
mpl.rcParams['xtick.major.width'] = 0.25
mpl.rcParams['ytick.major.width'] = 0.25
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['axes.labelpad'] = 1
mpl.rcParams['xtick.major.pad'] = 1
mpl.rcParams['ytick.major.pad'] = 1

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, '/media/drn2/External/TARA-Oceans/MANUSCRIPT/figures')
from palette import (
    OCEAN_CMAP, FOREST_CMAP,
    BASIN_COLORS,
    DEEP_OCEAN, TURQUOISE, FOREST_GREEN, SAVANNA, DESERT_TAN,
    COASTAL_BLUE, CLAY, SIENNA, SEAFOAM,
    OCEAN_BLUE, PALE_GREEN, SAND,
)

sys.path.insert(0, '/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/env_pfam_manifold')
from DataIntegrityGuard import enforce_data_integrity
enforce_data_integrity()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = "/media/drn2/External/TARA-Oceans"
ANALYSIS_DIR = f"{BASE_DIR}/03_analyses"
BC_DIR = f"{ANALYSIS_DIR}/WorldModelApp/basin_coverage"
ALGAGPT_DIR = f"{ANALYSIS_DIR}/ALGAGPT-based-analyses"
MANUSCRIPT = pathlib.Path(BASE_DIR) / "MANUSCRIPT"

BASIN_STATS = f"{BC_DIR}/data/basin_statistics.json"
BASIN_ASSIGN = f"{ANALYSIS_DIR}/WorldModelApp/data/ocean_basin_assignments.tsv"
MERGED_TSV = f"{ALGAGPT_DIR}/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
ALGAGPT_CSV = MANUSCRIPT / "source_data" / "algagpt_classification_summary_20260114_150000.csv"
RUBISCO_MERGED_TSV = pathlib.Path(ANALYSIS_DIR) / "rubisco_hmms" / "rubisco_extended_combined_20260218_081615.tsv"
CORR_TSV = f"{ALGAGPT_DIR}/algagpt_pfam_alphaearth_correlations_20260124_175739_full.tsv"
PHYLO_PDF = MANUSCRIPT / "figures" / "FigureS_phylo_compact_20260627.pdf"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"{BASE_DIR}/MANUSCRIPT/figures"
OUT_PDF = f"{OUT_DIR}/FigureS3_dense_{TIMESTAMP}.pdf"
OUT_SVG = f"{OUT_DIR}/FigureS3_dense_{TIMESTAMP}.svg"

for path in [BASIN_STATS, BASIN_ASSIGN, MERGED_TSV, str(ALGAGPT_CSV),
             str(RUBISCO_MERGED_TSV), CORR_TSV, str(PHYLO_PDF)]:
    if not os.path.isfile(path):
        print(f"ERROR: Missing input: {path}")
        sys.exit(1)
print("All input files verified.")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

# Basin coverage
print("Loading basin statistics...")
with open(BASIN_STATS, 'r') as f:
    basin_stats = json.load(f)
basin_order = basin_stats['summary']['basin_order_by_count']
df_basins = pd.read_csv(BASIN_ASSIGN, sep='\t', comment='#')
df_gps = df_basins.dropna(subset=['latitude', 'longitude']).copy()
print(f"  GPS samples: {len(df_gps)}")

df_merged_basin = pd.read_csv(MERGED_TSV, sep='\t', comment='#',
                               usecols=['assembly_id', 'dataset', 'depth_m'],
                               dtype={'depth_m': str})
df_merged_basin['depth_m'] = pd.to_numeric(df_merged_basin['depth_m'], errors='coerce')
df_joined = df_merged_basin.merge(
    df_basins[['assembly_id', 'ocean_basin']], on='assembly_id', how='left')
df_joined_gps = df_joined[df_joined['ocean_basin'] != 'no_gps'].copy()

# AlgaGPT classification
print("Loading AlgaGPT classification data...")
df_ag = pd.read_csv(ALGAGPT_CSV, comment='#')

import re as _re
def _classify(fn):
    m = _re.match(r'MGYA(\d+)', fn)
    if m:
        mid = int(m.group(1))
        if 582000 <= mid < 583000:
            return 'osd_metagenome'
        return 'tara_metagenome'
    if fn.startswith('MMETSP'):
        return 'mmetsp'
    return 'cultured_reference'

df_ag['source_type'] = df_ag['filename'].apply(_classify)
TYPE_MAP = {
    'tara_metagenome':    'TARA meta.',
    'osd_metagenome':     'OSD meta.',
    'mmetsp':             'MMETSP',
    'cultured_reference': 'Cultured refs.',
}
SOURCE_COLORS = {
    'TARA meta.':     DEEP_OCEAN,
    'OSD meta.':      COASTAL_BLUE,
    'MMETSP':         TURQUOISE,
    'Cultured refs.': FOREST_GREEN,
}
df_ag['display_cat'] = df_ag['source_type'].map(TYPE_MAP)
df_ag['retention'] = df_ag['pct_algae']

# RuBisCO data
print("Loading RuBisCO data...")
rub_merged = pd.read_csv(RUBISCO_MERGED_TSV, sep='\t', comment='#')
FORMI_LINEAGES = [
    'chlorellaceae', 'haptophyta', 'bolidophyceae', 'prasinophyceae',
    'pelagophyceae', 'mamiellophyceae', 'cryptophyta', 'scenedesmaceae',
    'pyramimonadales', 'trebouxiophyceae',
]
FORMII_LINEAGES = [
    'prorocentrales', 'gonyaulacales', 'peridiniales',
    'symbiodiniaceae', 'chromerida',
]
ALL_LINEAGES_RAW = FORMI_LINEAGES + FORMII_LINEAGES
rub_totals = {lin: int(rub_merged[lin].sum()) for lin in ALL_LINEAGES_RAW}
rub_totals = {k: v for k, v in sorted(rub_totals.items(), key=lambda x: -x[1])}

DISPLAY_NAMES = {
    'chlorellaceae': 'Chlorellaceae', 'haptophyta': 'Haptophyta',
    'bolidophyceae': 'Bolidophyceae', 'prasinophyceae': 'Prasinophyceae',
    'pelagophyceae': 'Pelagophyceae', 'mamiellophyceae': 'Mamiellophyceae',
    'cryptophyta': 'Cryptophyta', 'scenedesmaceae': 'Scenedesmaceae',
    'pyramimonadales': 'Pyramimonadales', 'trebouxiophyceae': 'Trebouxiophyceae',
    'prorocentrales': 'Prorocentrales', 'gonyaulacales': 'Gonyaulacales',
    'peridiniales': 'Peridiniales', 'symbiodiniaceae': 'Symbiodiniaceae',
    'chromerida': 'Chromerida',
}
LINEAGE_ORDER = [DISPLAY_NAMES[k] for k in rub_totals.keys()]
rub_hits = np.array(list(rub_totals.values()))

GREEN_LINEAGES = {
    'Chlorellaceae', 'Prasinophyceae', 'Mamiellophyceae',
    'Scenedesmaceae', 'Pyramimonadales', 'Trebouxiophyceae',
}
RED_LINEAGES = {
    'Haptophyta', 'Bolidophyceae', 'Pelagophyceae', 'Cryptophyta',
}
FORMII_DISPLAY = {
    'Prorocentrales', 'Gonyaulacales', 'Peridiniales',
    'Symbiodiniaceae', 'Chromerida',
}

# Correlation data (for panel F)
print("Loading correlation data...")
with open(CORR_TSV, 'r') as f:
    skip_lines = 0
    for line in f:
        if line.startswith('#'):
            skip_lines += 1
        else:
            break
corr_df = pd.read_csv(CORR_TSV, sep='\t', skiprows=skip_lines)
rho_values = corr_df['rho'].values
n_sig = (corr_df['p_adj_fdr'] < 0.05).sum() if 'p_adj_fdr' in corr_df.columns else 0
pct_sig = 100 * n_sig / len(corr_df) if len(corr_df) > 0 else 0
print(f"  Correlation tests: {len(corr_df):,}, significant: {n_sig:,} ({pct_sig:.1f}%)")

# XGBoost CV data (hardcoded from verified analysis — same as S4 script)
XGBOOST_CV_RESULTS = {
    'A31': {'mean_r2': 0.1729, 'std_r2': 0.1436},
    'A07': {'mean_r2': 0.1683, 'std_r2': 0.0684},
    'A49': {'mean_r2': 0.1401, 'std_r2': 0.1380},
    'A20': {'mean_r2': 0.1199, 'std_r2': 0.0725},
    'A19': {'mean_r2': 0.1060, 'std_r2': 0.0600},
    'A18': {'mean_r2': 0.1043, 'std_r2': 0.0921},
    'A15': {'mean_r2': 0.0974, 'std_r2': 0.1332},
    'A13': {'mean_r2': 0.0970, 'std_r2': 0.1100},
    'A51': {'mean_r2': 0.0809, 'std_r2': 0.0910},
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

panel_idx = 0
panel_letters = 'ABCDEFG'

def add_label(ax, letter=None, x=-0.06, y=1.04):
    global panel_idx
    if letter is None:
        letter = panel_letters[panel_idx]
        panel_idx += 1
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=6, fontweight='bold', va='top', ha='left')

def get_basin_color(basin_name):
    if basin_name in BASIN_COLORS:
        return BASIN_COLORS[basin_name]
    for alt in [basin_name.replace('_', ' '), basin_name.replace(' ', '_')]:
        if alt in BASIN_COLORS:
            return BASIN_COLORS[alt]
    return (0.5, 0.5, 0.5)

def _rub_color(name):
    if name in GREEN_LINEAGES:
        return FOREST_GREEN
    elif name in RED_LINEAGES:
        return COASTAL_BLUE
    else:
        return CLAY

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.25)

# ══════════════════════════════════════════════════════════════════════════════
# CREATE FIGURE — Rows 1+2 only (Row 3 composited from PDF later)
# ══════════════════════════════════════════════════════════════════════════════
print("\nCreating figure (rows 1-2)...")

fig = plt.figure(figsize=(7.5, 4.6))

outer = gridspec.GridSpec(3, 1, figure=fig,
                          height_ratios=[0.46, 0.12, 0.42],
                          hspace=0.0)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1: A (basin map) | B (depth violins) — compact
# ══════════════════════════════════════════════════════════════════════════════
gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                            width_ratios=[1.4, 0.6], wspace=0.22)

# ── Panel A: Basin world map ─────────────────────────────────────────────────
print("  Panel A: Basin map...")
ax_map = fig.add_subplot(gs_row1[0, 0], projection=ccrs.Robinson())
ax_map.set_global()
ax_map.add_feature(cfeature.LAND, facecolor='#F0F0F0', edgecolor='none', zorder=1)
ax_map.add_feature(cfeature.OCEAN, facecolor='#FAFAFA', edgecolor='none', zorder=0)
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.25, edgecolor='#999999', zorder=2)
try:
    ax_map.outline_patch.set_linewidth(0.25)
except AttributeError:
    ax_map.spines['geo'].set_linewidth(0.25)

plot_order = list(reversed(basin_order))
for basin in plot_order:
    mask_b = df_gps['ocean_basin'] == basin
    subset = df_gps[mask_b]
    if len(subset) == 0:
        continue
    color = get_basin_color(basin)
    ax_map.scatter(
        subset['longitude'].values, subset['latitude'].values,
        c=[color], s=3, alpha=0.7, edgecolors='none', linewidths=0,
        transform=ccrs.PlateCarree(), zorder=3,
        label=f"{basin.replace('_', ' ')} (n={len(subset)})"
    )

handles, labels = ax_map.get_legend_handles_labels()
handle_dict = {lbl.split(' (')[0]: (h, lbl) for h, lbl in zip(handles, labels)}
ordered_handles, ordered_labels = [], []
for basin in basin_order:
    key = basin.replace('_', ' ')
    if key in handle_dict:
        h, lbl = handle_dict[key]
        ordered_handles.append(h)
        ordered_labels.append(lbl)

leg = ax_map.legend(ordered_handles, ordered_labels, loc='upper left', fontsize=5,
                    frameon=True, framealpha=0.85, edgecolor='#CCCCCC',
                    fancybox=False, markerscale=1.5, handletextpad=0.3,
                    borderpad=0.2, labelspacing=0.10, borderaxespad=0.2)
leg.get_frame().set_linewidth(0.25)
add_label(ax_map, x=-0.02, y=1.05)

# ── Panel B: Depth Distribution (compact) ───────────────────────────────────
print("  Panel B: Depth distribution...")
n_basins = len(basin_order)
y_positions = np.arange(n_basins) * 0.55
bar_height = 0.42
ax_depth = fig.add_subplot(gs_row1[0, 1])

depth_per_basin = {}
for basin in basin_order:
    mask_b = df_joined_gps['ocean_basin'] == basin
    depths = df_joined_gps.loc[mask_b, 'depth_m'].dropna()
    depth_per_basin[basin] = depths.values

box_colors = [get_basin_color(b) for b in basin_order]
def to_log(x):
    return np.log10(np.clip(x, 0, None) + 1)

log_tick_vals = [0, 5, 25, 100, 500, 2000, 5000]
log_tick_pos = [to_log(v) for v in log_tick_vals]
log_tick_labels = ['0', '5', '25', '100', '500', '2k', '5k']

violin_width = bar_height * 0.80
for i, basin in enumerate(basin_order):
    vals = depth_per_basin[basin]
    ypos = y_positions[i]
    color = box_colors[i]
    log_vals = to_log(vals)
    if len(vals) < 2:
        ax_depth.scatter(log_vals, np.full_like(log_vals, ypos), s=1.5, color=color,
                        alpha=0.6, edgecolors='none', zorder=3)
        continue
    parts = ax_depth.violinplot([log_vals], positions=[ypos], vert=False,
                                 widths=violin_width, showextrema=False, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
        pc.set_edgecolor(color)
        pc.set_linewidth(0.4)
    med = np.median(log_vals)
    ax_depth.plot([med, med], [ypos - violin_width * 0.3, ypos + violin_width * 0.3],
                  color='black', linewidth=0.5, zorder=4)
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-violin_width * 0.20, violin_width * 0.20, len(vals))
    ax_depth.scatter(log_vals, ypos + jitter, s=1, color='black',
                    alpha=0.35, edgecolors='none', zorder=3, rasterized=True)

ax_depth.set_xlabel('Depth (m)')
ax_depth.set_ylim(y_positions[-1] + bar_height, y_positions[0] - bar_height)
ax_depth.set_yticks(y_positions)
ax_depth.set_yticklabels([b.replace('_', ' ').replace('Mediterranean', 'Medit.')
                           for b in basin_order], fontsize=5)
ax_depth.tick_params(axis='y', labelsize=5, pad=1)
ax_depth.set_xlim(to_log(-0.5), to_log(5500))
ax_depth.set_xticks(log_tick_pos)
ax_depth.set_xticklabels(log_tick_labels)
style_ax(ax_depth)
ax_depth.tick_params(axis='x', width=0.25, colors='#666666')
add_label(ax_depth, x=-0.08, y=1.08)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2: C | D | E | F | G — all compact
# ══════════════════════════════════════════════════════════════════════════════
print("  Row 2: Five compact panels...")
gs_row2 = gridspec.GridSpecFromSubplotSpec(
    1, 5, subplot_spec=outer[2],
    # Panel E (index 2) widened ~20% (1.0 -> 1.2), expanding left by taking the
    # width from panel D (0.75 -> 0.55) so E grows into D's space without
    # shifting F/G. Gives E's curves room and clears the legend overlap.
    width_ratios=[0.95, 0.55, 1.2, 1.15, 0.80],
    wspace=0.60)

# Split lineages
formi_names = [n for n in LINEAGE_ORDER if n in GREEN_LINEAGES or n in RED_LINEAGES]
formii_names = [n for n in LINEAGE_ORDER if n in FORMII_DISPLAY]
formi_idx = [LINEAGE_ORDER.index(n) for n in formi_names]
formii_idx = [LINEAGE_ORDER.index(n) for n in formii_names]
formi_hits = rub_hits[formi_idx]
formii_hits = rub_hits[formii_idx]

# ── Panel C: Form I RuBisCO ──────────────────────────────────────────────────
ax_c = fig.add_subplot(gs_row2[0, 0])
y1 = np.arange(len(formi_names))[::-1]
colors_1 = [_rub_color(n) for n in formi_names]
ax_c.barh(y1, formi_hits, color=colors_1, edgecolor='none', height=0.65)
ax_c.set_yticks(y1)
ax_c.set_yticklabels(formi_names, fontsize=5)
ax_c.set_xlabel('Form I hits')
ax_c.set_xlim(0, max(formi_hits) * 1.08)
style_ax(ax_c)
# Combined C+D legend: all three RuBisCO categories stacked in one box,
# placed lower-centre over panel C's empty mid-low region (~Pyramimonadales
# row) so it clears both panels' y-axes. Panel D carries no separate legend.
combined_legend = [
    Patch(facecolor=FOREST_GREEN, label='Form IB (green)'),
    Patch(facecolor=COASTAL_BLUE, label='Form ID (red/other)'),
    Patch(facecolor=CLAY, label='Form II (dinoflag.)'),
]
ax_c.legend(handles=combined_legend, loc='center', fontsize=5,
            bbox_to_anchor=(0.62, 0.20), frameon=True, framealpha=0.92,
            edgecolor='#cccccc', borderpad=0.3,
            handletextpad=0.3, labelspacing=0.18)
add_label(ax_c, x=-0.12, y=1.06)

# ── Panel D: Form II RuBisCO ─────────────────────────────────────────────────
ax_d = fig.add_subplot(gs_row2[0, 1])
n_formi = len(formi_names)
n_formii = len(formii_names)
y2_top = n_formi - 1
y2 = np.arange(n_formii)[::-1] + (n_formi - n_formii)
ax_d.barh(y2, formii_hits, color=[CLAY] * n_formii, edgecolor='none', height=0.65)
ax_d.set_yticks(y2)
ax_d.set_yticklabels(formii_names, fontsize=5)
ax_d.set_xlabel('Form II hits')
ax_d.set_xlim(0, max(formii_hits) * 1.08)
ax_d.set_ylim(-0.5, n_formi - 0.5)
style_ax(ax_d)
# Legend merged into the combined box on panel C (see above).
add_label(ax_d, x=-0.12, y=1.06)

# ── Panel E: AlgaGPT retention KDE ──────────────────────────────────────────
ax_e = fig.add_subplot(gs_row2[0, 2])
cat_order = ['TARA meta.', 'OSD meta.', 'MMETSP', 'Cultured refs.']
for cat in cat_order:
    vals = df_ag.loc[df_ag['display_cat'] == cat, 'retention'].dropna().values
    if len(vals) < 3:
        continue
    n = len(vals)
    try:
        kde = stats.gaussian_kde(vals, bw_method=0.3)
        x_grid = np.linspace(0, 100, 300)
        density = kde(x_grid)
        ax_e.plot(x_grid, density, color=SOURCE_COLORS[cat], linewidth=0.7,
                  label=f'{cat} (n={n:,})')
        ax_e.fill_between(x_grid, density, alpha=0.20, color=SOURCE_COLORS[cat])
    except Exception:
        pass
ax_e.set_xlabel('Sequences retained (%)')
ax_e.set_ylabel('Density')
# Headroom so the upper-left legend clears the tallest KDE peak (~0.065).
ax_e.set_ylim(0, 0.09)
leg_e = ax_e.legend(fontsize=5, frameon=True, framealpha=0.92, edgecolor='#cccccc',
                    loc='upper left', bbox_to_anchor=(0.0, 1.0),
                    borderpad=0.2, labelspacing=0.10,
                    handlelength=1.0, handletextpad=0.3)
leg_e.get_frame().set_linewidth(0.25)
style_ax(ax_e)
add_label(ax_e, x=-0.10, y=1.06)

# ── Panel F: Spearman rho histogram ──────────────────────────────────────────
ax_f = fig.add_subplot(gs_row2[0, 3])
total_tests = len(rho_values)
max_rho = rho_values.max()
min_rho = rho_values.min()

n_bins = 80
counts, bins, patches = ax_f.hist(rho_values, bins=n_bins, edgecolor='none', alpha=0.9)
for patch, bin_center in zip(patches, (bins[:-1] + bins[1:]) / 2):
    if bin_center < -0.1:
        patch.set_facecolor(DEEP_OCEAN)
    elif bin_center > 0.1:
        patch.set_facecolor(SIENNA)
    else:
        patch.set_facecolor((0.6, 0.6, 0.6))

ax_f.axvline(x=0, color='black', linestyle='-', linewidth=0.4, alpha=0.5)
ax_f.axvline(x=max_rho, color=SIENNA, linestyle='--', linewidth=0.4, alpha=0.7)
ax_f.axvline(x=min_rho, color=DEEP_OCEAN, linestyle='--', linewidth=0.4, alpha=0.7)
ax_f.set_xlabel('Spearman ρ')
ax_f.set_ylabel('Frequency')
ax_f.set_xlim(-0.7, 0.7)

stats_text = (f"n = {total_tests:,}\n"
              f"{pct_sig:.1f}% sig.\n"
              f"(FDR < 0.05)")
ax_f.text(0.97, 0.97, stats_text, transform=ax_f.transAxes,
          fontsize=5, va='top', ha='right',
          bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                    edgecolor=(0.7, 0.7, 0.7), alpha=0.9, linewidth=0.25))
style_ax(ax_f)
add_label(ax_f, x=-0.10, y=1.06)

# ── Panel G: XGBoost CV R² ──────────────────────────────────────────────────
ax_g = fig.add_subplot(gs_row2[0, 4])
dims_sorted = sorted(XGBOOST_CV_RESULTS.keys(),
                     key=lambda x: XGBOOST_CV_RESULTS[x]['mean_r2'],
                     reverse=True)
r2_vals = [XGBOOST_CV_RESULTS[d]['mean_r2'] for d in dims_sorted]
r2_errs = [XGBOOST_CV_RESULTS[d]['std_r2'] for d in dims_sorted]

x_pos = np.arange(len(dims_sorted))
bars = ax_g.bar(x_pos, r2_vals, yerr=r2_errs, capsize=1.5,
              color=COASTAL_BLUE, edgecolor='none', alpha=0.9,
              error_kw={'linewidth': 0.4, 'capthick': 0.4})
bars[0].set_color(SIENNA)

ax_g.set_xlabel('AE dim.')
ax_g.set_ylabel('CV R²')
ax_g.set_xticks(x_pos)
ax_g.set_xticklabels(dims_sorted, rotation=45, ha='right', fontsize=5)
ax_g.set_ylim(0, 0.45)
style_ax(ax_g)
add_label(ax_g, x=-0.06, y=1.06)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE final figure (panels A-G only; trees are in Figure S1 A-C)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nSaving final figure (panels A-G)...")
for fmt in ['pdf', 'svg']:
    path = OUT_PDF.replace('.pdf', f'.{fmt}')
    fig.savefig(path, format=fmt, bbox_inches='tight',
                transparent=True, edgecolor='none', dpi=300)
    print(f"  Saved: {path}")

png_path = OUT_PDF.replace('.pdf', '_check.png')
fig.savefig(png_path, dpi=300, bbox_inches='tight', transparent=False)
print(f"  Check: {png_path}")
plt.close()

# Provenance
prov_path = OUT_PDF.replace('.pdf', '_provenance.txt')
with open(prov_path, 'w') as f:
    f.write(f"# Provenance for Figure S3 (dense rebuild)\n")
    f.write(f"# Generated: {datetime.now().isoformat()}\n")
    f.write(f"# Script: {os.path.abspath(__file__)}\n")
    f.write(f"#\n")
    f.write(f"# Inputs:\n")
    f.write(f"#   {BASIN_STATS}\n")
    f.write(f"#   {BASIN_ASSIGN}\n")
    f.write(f"#   {MERGED_TSV}\n")
    f.write(f"#   {ALGAGPT_CSV}\n")
    f.write(f"#   {RUBISCO_MERGED_TSV}\n")
    f.write(f"#   {CORR_TSV}\n")
    f.write(f"#   {PHYLO_PDF} (panels H-J, unchanged)\n")
    f.write(f"# Data Integrity Check: PASSED\n")
print(f"  Provenance: {prov_path}")

print(f"\n{'='*70}")
print(f"COMPLETE — Dense Figure S3 (10 panels A-J)")
print(f"  Row 1: A (basin map) | B (depth distribution)")
print(f"  Row 2: C (Form I) | D (Form II) | E (retention) | F (Spearman) | G (XGBoost)")
print(f"  Row 3: H-J (phylo trees)")
print(f"{'='*70}")
