#!/usr/bin/env python3
"""
Ralph25 Task 6: Composite Figure S9 — Extended CCA Analysis

Multi-panel supplementary figure combining results from tasks 1-5:
  (A) PFAM loading bar chart for CC1 top-20 domains
  (B) PFAM loading bar chart for CC2 top-10 and CC3 top-10
  (C) World map with CC1 env-side scores (Robinson projection)
  (D) Cross-validated shrinkage plot (observed bars + CV violin overlay)
  (E) GCCA pairwise correlation comparison

Provenance:
  Input: source_data/cca_pfam_loadings_20260212_100221.tsv (task 1)
  Input: source_data/cca_cc2_cc5_summary_20260212_100532.tsv (task 2)
  Input: source_data/cca_shrinkage_cv_20260212_101811.tsv (task 4)
  Input: source_data/cca_shrinkage_cv_distribution_20260212_101811.tsv (task 4)
  Input: source_data/gcca_ternary_comparison_20260212_102910.tsv (task 5)
  Input: CCA env embedding + GPS from merged dataset (task 3)
  Output: figures/FigureS9_cca_extended_*.pdf, *.svg
  Date: 2026-02-12
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from datetime import datetime
import sys

# ============================================================================
# Data integrity enforcement
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
MANUSCRIPT_DIR = SCRIPT_DIR.parent
BASE_DIR = Path("/media/drn2/External/TARA-Oceans")
CCA_DATA_DIR = BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "env_pfam_manifold" / "data"
SOURCE_DATA_DIR = MANUSCRIPT_DIR / "source_data"

sys.path.insert(0, str(BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "env_pfam_manifold"))
from DataIntegrityGuard import enforce_data_integrity, validate_input_source, create_provenance_header
enforce_data_integrity()

# Import palette
sys.path.insert(0, str(SCRIPT_DIR))
from palette import (
    get_diverging_cmap, get_sequential_cmap,
    DEEP_OCEAN, OCEAN_BLUE, COASTAL_BLUE, TURQUOISE, SEAFOAM, PALE_AQUA,
    FOREST_GREEN, SAVANNA, DESERT_TAN, CLAY, SIENNA,
    CLOUD_WHITE, ABYSS, SAND,
    EARTH_CATEGORICAL, OCEAN_CMAP, DIVERGING_CMAP,
)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# FIGURE_PROTOCOL.md: matplotlib settings
# ============================================================================
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

# ============================================================================
# LOAD ALL DATA
# ============================================================================
print("=" * 70)
print("LOADING DATA FOR COMPOSITE FIGURE S9")
print("=" * 70)

# --- Task 1: PFAM loadings ---
pfam_loadings_path = SOURCE_DATA_DIR / "cca_pfam_loadings_20260212_100221.tsv"
validate_input_source(pfam_loadings_path)
pfam_loadings = pd.read_csv(pfam_loadings_path, sep='\t', comment='#')
print(f"  PFAM loadings: {pfam_loadings.shape[0]} rows, {pfam_loadings.columns.tolist()}")

# --- Task 2: CC summary ---
cc_summary_path = SOURCE_DATA_DIR / "cca_cc2_cc5_summary_20260212_100532.tsv"
validate_input_source(cc_summary_path)
cc_summary = pd.read_csv(cc_summary_path, sep='\t', comment='#')
print(f"  CC summary: {cc_summary.shape[0]} rows")

# --- Task 4: Shrinkage ---
shrinkage_path = SOURCE_DATA_DIR / "cca_shrinkage_cv_20260212_101811.tsv"
validate_input_source(shrinkage_path)
shrinkage = pd.read_csv(shrinkage_path, sep='\t', comment='#')
print(f"  Shrinkage summary: {shrinkage.shape[0]} rows")

shrinkage_dist_path = SOURCE_DATA_DIR / "cca_shrinkage_cv_distribution_20260212_101811.tsv"
validate_input_source(shrinkage_dist_path)
shrinkage_dist = pd.read_csv(shrinkage_dist_path, sep='\t', comment='#')
print(f"  Shrinkage distribution: {shrinkage_dist.shape}")

# --- Task 5: GCCA comparison ---
gcca_comp_path = SOURCE_DATA_DIR / "gcca_ternary_comparison_20260212_102910.tsv"
validate_input_source(gcca_comp_path)
gcca_comp = pd.read_csv(gcca_comp_path, sep='\t', comment='#')
print(f"  GCCA comparison: {gcca_comp.shape}")

# --- Task 3: CCA geography data (recompute from original data) ---
env_emb_path = CCA_DATA_DIR / "cca_env_embedding_20260122_104244.npy"
sample_ids_path = CCA_DATA_DIR / "sample_ids_20260122_101559.npy"
merged_path = BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"

validate_input_source(env_emb_path)
validate_input_source(sample_ids_path)
validate_input_source(merged_path)

env_embedding = np.load(env_emb_path)
sample_ids = np.load(sample_ids_path, allow_pickle=True)
cc1_scores = env_embedding[:, 0]
print(f"  CCA env embedding: {env_embedding.shape}")
print(f"  CC1 range: [{cc1_scores.min():.4f}, {cc1_scores.max():.4f}]")

merged = pd.read_csv(merged_path, sep='\t', comment='#',
                     usecols=['assembly_id', 'latitude', 'longitude'])
gps_lookup = merged.set_index('assembly_id')[['latitude', 'longitude']]
lats = np.array([gps_lookup.loc[sid, 'latitude'] for sid in sample_ids])
lons = np.array([gps_lookup.loc[sid, 'longitude'] for sid in sample_ids])
print(f"  GPS matched: {len(lats)} samples")

# ============================================================================
# PREPARE PANEL DATA
# ============================================================================
print("\n" + "=" * 70)
print("PREPARING PANEL DATA")
print("=" * 70)

# Panel A: CC1 top 20 PFAM domains
cc1_data = pfam_loadings[pfam_loadings['canonical_component'] == 'CC1'].head(20).copy()
cc1_data = cc1_data.sort_values('cca_pca_loading', ascending=True)  # negative values first for horizontal bars
# Strip version from PFAM IDs for display
cc1_data['pfam_short'] = cc1_data['pfam_domain'].str.replace(r'\.\d+$', '', regex=True)
print(f"  Panel A: CC1 top {len(cc1_data)} PFAM domains")

# Panel B: CC2 top 10 + CC3 top 10
cc2_data = pfam_loadings[pfam_loadings['canonical_component'] == 'CC2'].head(10).copy()
cc2_data = cc2_data.sort_values('cca_pca_loading', ascending=True)
cc2_data['pfam_short'] = cc2_data['pfam_domain'].str.replace(r'\.\d+$', '', regex=True)

cc3_data = pfam_loadings[pfam_loadings['canonical_component'] == 'CC3'].head(10).copy()
cc3_data = cc3_data.sort_values('cca_pca_loading', ascending=True)
cc3_data['pfam_short'] = cc3_data['pfam_domain'].str.replace(r'\.\d+$', '', regex=True)
print(f"  Panel B: CC2 top {len(cc2_data)}, CC3 top {len(cc3_data)} PFAM domains")

# Panel C: world map data already loaded above
sort_idx = np.argsort(np.abs(cc1_scores))
lats_sorted = lats[sort_idx]
lons_sorted = lons[sort_idx]
cc1_sorted = cc1_scores[sort_idx]
vmax = max(abs(cc1_scores.min()), abs(cc1_scores.max()))
vmin = -vmax
print(f"  Panel C: {len(cc1_scores)} samples, color limits [{vmin:.4f}, {vmax:.4f}]")

# Panel D: shrinkage data already loaded
print(f"  Panel D: {len(shrinkage)} CC components, {shrinkage_dist.shape[0]} CV iterations")

# Panel E: GCCA comparison data
gcca_ccs = gcca_comp[gcca_comp['component'] != 'mean'].copy()
print(f"  Panel E: {len(gcca_ccs)} CC components × {len(gcca_comp.columns)-1} analyses")

# ============================================================================
# CREATE COMPOSITE FIGURE
# ============================================================================
print("\n" + "=" * 70)
print("CREATING COMPOSITE FIGURE S9")
print("=" * 70)

fig = plt.figure(figsize=(7.5, 10.0))

# Layout: 3 rows
# Row 1: (A) CC1 PFAM bar chart | (B) CC2 + CC3 PFAM bar charts (stacked)
# Row 2: (C) World map (full width)
# Row 3: (D) Shrinkage plot | (E) GCCA comparison

outer_gs = gridspec.GridSpec(3, 1, figure=fig,
                             height_ratios=[3.0, 2.5, 2.5],
                             hspace=0.30)

# Row 1: A and B side by side
row1_gs = outer_gs[0].subgridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.45)

# Row 2: C (map) — full width
# Row 3: D and E side by side
row3_gs = outer_gs[2].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.35)

# ── Panel A: CC1 PFAM loadings (horizontal bar chart) ──
ax_a = fig.add_subplot(row1_gs[0, 0])

bar_colors_a = [DEEP_OCEAN if v < 0 else SIENNA for v in cc1_data['cca_pca_loading']]
ax_a.barh(range(len(cc1_data)), cc1_data['cca_pca_loading'].values,
          color=bar_colors_a, edgecolor='none', height=0.7)
ax_a.set_yticks(range(len(cc1_data)))
ax_a.set_yticklabels(cc1_data['pfam_short'].values, fontsize=5)
ax_a.set_xlabel('CCA loading (through PCA)', fontsize=6)
ax_a.set_title('CC1 top-20 PFAM domain loadings', fontsize=6, fontweight='bold')
ax_a.axvline(0, color='black', linewidth=0.25, zorder=0)
ax_a.invert_yaxis()

# Add Spearman r annotations to the right of zero line
for i, (_, row) in enumerate(cc1_data.iterrows()):
    r = row['spearman_r_vs_cc_score']
    ax_a.text(0.002, i, f'r={r:.2f}',
              fontsize=4.5, va='center', ha='left',
              color=(0.4, 0.4, 0.4))

# Spines
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# Panel label
ax_a.text(-0.15, 1.05, 'A', transform=ax_a.transAxes,
          fontsize=8, fontweight='bold', va='top')

# ── Panel B: CC2 + CC3 PFAM loadings (stacked) ──
b_gs = row1_gs[0, 1].subgridspec(2, 1, hspace=0.40)

# B top: CC2
ax_b1 = fig.add_subplot(b_gs[0])
bar_colors_b1 = [DEEP_OCEAN if v < 0 else SIENNA for v in cc2_data['cca_pca_loading']]
ax_b1.barh(range(len(cc2_data)), cc2_data['cca_pca_loading'].values,
           color=bar_colors_b1, edgecolor='none', height=0.7)
ax_b1.set_yticks(range(len(cc2_data)))
ax_b1.set_yticklabels(cc2_data['pfam_short'].values, fontsize=5)
ax_b1.set_title('CC2 top-10 PFAM loadings', fontsize=6, fontweight='bold')
ax_b1.axvline(0, color='black', linewidth=0.25, zorder=0)
ax_b1.invert_yaxis()
ax_b1.spines['top'].set_visible(False)
ax_b1.spines['right'].set_visible(False)

# B bottom: CC3
ax_b2 = fig.add_subplot(b_gs[1])
bar_colors_b2 = [DEEP_OCEAN if v < 0 else SIENNA for v in cc3_data['cca_pca_loading']]
ax_b2.barh(range(len(cc3_data)), cc3_data['cca_pca_loading'].values,
           color=bar_colors_b2, edgecolor='none', height=0.7)
ax_b2.set_yticks(range(len(cc3_data)))
ax_b2.set_yticklabels(cc3_data['pfam_short'].values, fontsize=5)
ax_b2.set_xlabel('CCA loading (through PCA)', fontsize=6)
ax_b2.set_title('CC3 top-10 PFAM loadings', fontsize=6, fontweight='bold')
ax_b2.axvline(0, color='black', linewidth=0.25, zorder=0)
ax_b2.invert_yaxis()
ax_b2.spines['top'].set_visible(False)
ax_b2.spines['right'].set_visible(False)

# Panel label
ax_b1.text(-0.15, 1.10, 'B', transform=ax_b1.transAxes,
           fontsize=8, fontweight='bold', va='top')

# ── Panel C: World map with CC1 scores ──
ax_c = fig.add_subplot(outer_gs[1], projection=ccrs.Robinson())

cmap_div = get_diverging_cmap('ocean_land')

ax_c.add_feature(cfeature.LAND, facecolor=(0.92, 0.92, 0.92),
                 edgecolor='none', zorder=0)
ax_c.add_feature(cfeature.OCEAN, facecolor=(0.97, 0.97, 0.98),
                 edgecolor='none', zorder=0)
ax_c.add_feature(cfeature.COASTLINE, linewidth=0.15,
                 edgecolor=(0.5, 0.5, 0.5), zorder=1)

sc = ax_c.scatter(lons_sorted, lats_sorted,
                  c=cc1_sorted, cmap=cmap_div,
                  s=3, alpha=0.85,
                  vmin=vmin, vmax=vmax,
                  linewidth=0.1, edgecolor=(0.3, 0.3, 0.3),
                  transform=ccrs.PlateCarree(),
                  zorder=2)

gl = ax_c.gridlines(draw_labels=False, linewidth=0.15,
                    color=(0.7, 0.7, 0.7), alpha=0.5,
                    linestyle='--')

ax_c.set_title('CC1 env-side scores: bathymetry-temperature gradient (n = 1,809)',
               fontsize=6, fontweight='bold', pad=2)

# Colorbar for map
# Get the position of ax_c after layout
cax_c = fig.add_axes([0.88, 0.395, 0.012, 0.18])
cbar_c = plt.colorbar(sc, cax=cax_c, orientation='vertical')
cbar_c.set_label('CC1 score', fontsize=6)
cbar_c.ax.tick_params(labelsize=5, width=0.25, length=2)
cbar_c.outline.set_linewidth(0.25)

# Panel label
ax_c.text(-0.02, 1.05, 'C', transform=ax_c.transAxes,
          fontsize=8, fontweight='bold', va='top')

# ── Panel D: Cross-validated shrinkage ──
ax_d = fig.add_subplot(row3_gs[0, 0])

cc_labels = [f'CC{i+1}' for i in range(10)]
observed_ccs = shrinkage['observed_correlation'].values
cv_medians = shrinkage['cv_median'].values
shrinkage_pcts = shrinkage['shrinkage_pct'].values

# Distribution data for violins
cv_dist_cols = [f'CC{i+1}' for i in range(10)]
cv_dist_data = [shrinkage_dist[col].values for col in cv_dist_cols]

x = np.arange(10)
bar_width = 0.35

# Observed bars
bars_obs = ax_d.bar(x, observed_ccs, width=bar_width, label='Observed',
                    color=COASTAL_BLUE, edgecolor='none', alpha=0.9)

# CV violin overlay (shifted right)
vp = ax_d.violinplot(cv_dist_data, positions=x + bar_width,
                     showmeans=False, showmedians=True, showextrema=False,
                     widths=bar_width * 1.5)

# Style violins
for body in vp['bodies']:
    body.set_facecolor(DEEP_OCEAN)
    body.set_edgecolor('none')
    body.set_alpha(0.6)
vp['cmedians'].set_color('white')
vp['cmedians'].set_linewidth(0.5)

# Shrinkage percentage labels
for i, pct in enumerate(shrinkage_pcts):
    ax_d.text(x[i] + bar_width / 2, observed_ccs[i] + 0.02,
              f'{pct:.0f}%', fontsize=4.5, ha='center', va='bottom',
              color=SIENNA, fontweight='bold')

ax_d.set_xticks(x + bar_width / 2)
ax_d.set_xticklabels(cc_labels, fontsize=5)
ax_d.set_ylabel('Canonical correlation', fontsize=6)
ax_d.set_title('Observed vs. cross-validated CCs (100 split-half iterations)',
               fontsize=6, fontweight='bold')
ax_d.set_ylim(0, 1.0)
ax_d.legend(loc='upper right', frameon=False, fontsize=5,
            labels=['Observed', 'CV distribution'])
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)

# Panel label
ax_d.text(-0.12, 1.05, 'D', transform=ax_d.transAxes,
          fontsize=8, fontweight='bold', va='top')

# ── Panel E: GCCA pairwise correlation comparison ──
ax_e = fig.add_subplot(row3_gs[0, 1])

# Columns to plot
analysis_cols = ['gcca_3view_n995', 'env_pfam_2view_n995',
                 'alpha_pfam_2view_n995', 'alpha_env_2view_n995',
                 'env_pfam_2view_n1809']
analysis_labels = ['GCCA\n3-view\n(n=995)', 'Env×PFAM\n2-view\n(n=995)',
                   'AE×PFAM\n2-view\n(n=995)', 'AE×Env\n2-view\n(n=995)',
                   'Env×PFAM\n2-view\n(n=1809)']
analysis_colors = [FOREST_GREEN, COASTAL_BLUE, TURQUOISE, DESERT_TAN, DEEP_OCEAN]

x_e = np.arange(10)  # CC1-CC10
width_e = 0.15

for j, (col, label, color) in enumerate(zip(analysis_cols, analysis_labels, analysis_colors)):
    vals = gcca_ccs[col].values
    offset = (j - 2) * width_e
    ax_e.bar(x_e + offset, vals, width=width_e, label=label.replace('\n', ' '),
             color=color, edgecolor='none', alpha=0.85)

ax_e.set_xticks(x_e)
ax_e.set_xticklabels([f'CC{i+1}' for i in range(10)], fontsize=5)
ax_e.set_ylabel('Canonical correlation', fontsize=6)
ax_e.set_title('CCA vs. GCCA: pairwise comparison', fontsize=6, fontweight='bold')
ax_e.set_ylim(0, 1.05)
ax_e.legend(loc='upper right', frameon=False, fontsize=4.5, ncol=1,
            handlelength=1.0, handletextpad=0.3, labelspacing=0.3)
ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)

# Panel label
ax_e.text(-0.12, 1.05, 'E', transform=ax_e.transAxes,
          fontsize=8, fontweight='bold', va='top')

# ============================================================================
# SAVE OUTPUTS
# ============================================================================
print("\n" + "=" * 70)
print("SAVING COMPOSITE FIGURE S9")
print("=" * 70)

for fmt in ['pdf', 'svg']:
    filename = SCRIPT_DIR / f"FigureS9_cca_extended_{TIMESTAMP}.{fmt}"
    fig.savefig(filename, format=fmt,
                bbox_inches='tight',
                transparent=True,
                edgecolor='none',
                dpi=300)
    print(f"  Saved: {filename}")

plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Panel A: CC1 top-20 PFAM domains (all negative loadings)")
print(f"    Top domain: {cc1_data.iloc[-1]['pfam_short']} (loading={cc1_data.iloc[-1]['cca_pca_loading']:.4f})")
print(f"  Panel B: CC2 top-10 (all positive) + CC3 top-10 (mixed sign)")
print(f"  Panel C: World map with {len(cc1_scores)} samples, CC1 range [{cc1_scores.min():.4f}, {cc1_scores.max():.4f}]")
print(f"  Panel D: Shrinkage — CC1={shrinkage_pcts[0]:.1f}%, mean={shrinkage_pcts.mean():.1f}%")
print(f"  Panel E: 5 CCA/GCCA analyses × 10 CCs")
print(f"    AE×Env CC1={gcca_ccs['alpha_env_2view_n995'].iloc[0]:.4f} (highest pairwise)")
print(f"    GCCA CC1={gcca_ccs['gcca_3view_n995'].iloc[0]:.4f} (lowest — mean optimization)")

print(f"\n{'=' * 70}")
print(f"TASK 6 COMPLETE — {datetime.now().isoformat()}")
print(f"{'=' * 70}")
