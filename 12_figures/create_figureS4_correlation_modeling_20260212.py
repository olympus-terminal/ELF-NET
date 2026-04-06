#!/usr/bin/env python3
"""
Merged Figure S4: Domain-environment correlation and modeling performance

Combines old Figure S4 (correlation analysis, 2 panels) and old Figure S6
(bootstrap CIs, 1 panel) into a single 3-panel figure.

Layout:
  Row 1: (A) Spearman correlation distribution | (B) XGBoost CV R2 by dimension
  Row 2: (C) Bootstrap CI forest plot for reverse model (full width)

Provenance:
  Input: algagpt_pfam_alphaearth_correlations_20260124_175739_full.tsv (correlation data)
  Input: supplement/FigureS7_bootstrap_ci_20260208_105414.tsv (bootstrap CIs)
  Output: figures/FigureS4_correlation_modeling_*.pdf, *.svg
  Date: 2026-02-12
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
import sys

# ============================================================================
# Data integrity enforcement
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
MANUSCRIPT_DIR = SCRIPT_DIR.parent
BASE_DIR = Path("/media/drn2/External/TARA-Oceans")

sys.path.insert(0, str(BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "env_pfam_manifold"))
from DataIntegrityGuard import enforce_data_integrity, validate_input_source
enforce_data_integrity()

# Import palette
sys.path.insert(0, str(SCRIPT_DIR))
from palette import (
    DEEP_OCEAN, OCEAN_BLUE, COASTAL_BLUE, TURQUOISE, SEAFOAM, PALE_AQUA,
    FOREST_GREEN, SAVANNA, DESERT_TAN, CLAY, SIENNA,
    CLOUD_WHITE, ABYSS, SAND,
    ENV_CATEGORY_COLORS,
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
print("LOADING DATA FOR MERGED FIGURE S4")
print("=" * 70)

# --- Correlation data ---
corr_file = BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "algagpt_pfam_alphaearth_correlations_20260124_175739_full.tsv"
validate_input_source(corr_file)

# Read with comment skipping
with open(corr_file, 'r') as f:
    skip_lines = 0
    for line in f:
        if line.startswith('#'):
            skip_lines += 1
        else:
            break

corr_df = pd.read_csv(corr_file, sep='\t', skiprows=skip_lines)
print(f"  Correlation data: {len(corr_df):,} records")
rho_values = corr_df['rho'].values
print(f"  Rho range: [{rho_values.min():.4f}, {rho_values.max():.4f}]")

# Count significant
if 'p_adj_fdr' in corr_df.columns:
    n_sig = (corr_df['p_adj_fdr'] < 0.05).sum()
    pct_sig = 100 * n_sig / len(corr_df)
    print(f"  Significant (FDR<0.05): {n_sig:,} ({pct_sig:.1f}%)")
else:
    n_sig = 0
    pct_sig = 0

# XGBoost CV results — these are documented verified values from source_data/alphaearth_stats.md
# Read from the original analysis
XGBOOST_CV_RESULTS = {
    'A31': {'mean_r2': 0.4916, 'std_r2': 0.0604},
    'A20': {'mean_r2': 0.3838, 'std_r2': 0.0844},
    'A25': {'mean_r2': 0.3597, 'std_r2': 0.0784},
    'A14': {'mean_r2': 0.3548, 'std_r2': 0.0822},
    'A21': {'mean_r2': 0.3129, 'std_r2': 0.0652},
    'A63': {'mean_r2': 0.2893, 'std_r2': 0.0622},
    'A52': {'mean_r2': 0.2546, 'std_r2': 0.0932},
    'A49': {'mean_r2': 0.2400, 'std_r2': 0.0700},
    'A00': {'mean_r2': 0.0975, 'std_r2': 0.0524},
}
print(f"  XGBoost CV: {len(XGBOOST_CV_RESULTS)} dimensions")

# --- Bootstrap CI data ---
bootstrap_file = MANUSCRIPT_DIR / "supplement" / "FigureS7_bootstrap_ci_20260208_105414.tsv"
validate_input_source(bootstrap_file)
boot_df = pd.read_csv(bootstrap_file, sep='\t', comment='#')
print(f"  Bootstrap CI data: {len(boot_df)} targets")

# ============================================================================
# PREPARE PANEL DATA
# ============================================================================
print("\n" + "=" * 70)
print("PREPARING PANEL DATA")
print("=" * 70)

# Panel A: Correlation distribution
total_tests = len(rho_values)
max_rho = rho_values.max()
min_rho = rho_values.min()
print(f"  Panel A: {total_tests:,} tests, range [{min_rho:.4f}, {max_rho:.4f}]")

# Panel B: XGBoost R2 sorted
dims_sorted = sorted(XGBOOST_CV_RESULTS.keys(),
                     key=lambda x: XGBOOST_CV_RESULTS[x]['mean_r2'],
                     reverse=True)
r2_vals = [XGBOOST_CV_RESULTS[d]['mean_r2'] for d in dims_sorted]
r2_errs = [XGBOOST_CV_RESULTS[d]['std_r2'] for d in dims_sorted]
print(f"  Panel B: {len(dims_sorted)} dimensions, best={dims_sorted[0]} (R2={r2_vals[0]:.3f})")

# Panel C: Bootstrap forest plot
boot_df_sorted = boot_df.sort_values('r2_point', ascending=True).reset_index(drop=True)

# Category assignment for coloring
env_categories = {
    'modis_sst_mean_c': 'Temperature', 'sst_mean_c': 'Temperature',
    'sst_max_c': 'Temperature', 'sst_min_c': 'Temperature',
    'sst_range_c': 'Temperature',
    'air_temp_mean_c': 'Atmospheric', 'air_temp_max_c': 'Atmospheric',
    'air_temp_min_c': 'Atmospheric', 'air_temp_range_c': 'Atmospheric',
    'bathymetry_m': 'Bathymetry', 'elevation_m': 'Bathymetry',
    'distance_to_coast_km': 'Bathymetry', 'depth_m': 'Bathymetry',
    'solar_rad_mj_m2': 'Atmospheric',
    'nflh_mean': 'Productivity', 'chl_mean_mg_m3': 'Productivity',
    'chl_max_mg_m3': 'Productivity', 'chl_min_mg_m3': 'Productivity',
    'poc_mean_mg_m3': 'Productivity',
    'precip_mean_mm': 'Atmospheric',
    'rrs_412': 'Ocean Color', 'rrs_443': 'Ocean Color',
    'rrs_469': 'Ocean Color', 'rrs_488': 'Ocean Color',
    'rrs_531': 'Ocean Color', 'rrs_547': 'Ocean Color',
    'rrs_555': 'Ocean Color', 'rrs_645': 'Ocean Color',
    'rrs_667': 'Ocean Color', 'rrs_678': 'Ocean Color',
    'nitrate_umol_l': 'Nutrient', 'phosphate_umol_l': 'Nutrient',
    'silicate_umol_l': 'Nutrient', 'oxygen_umol_l': 'Nutrient',
    'mld_m': 'Nutrient', 'salinity_psu_est': 'Salinity',
    'landcover_class': 'Atmospheric',
}

nice_labels = {
    'modis_sst_mean_c': 'MODIS SST', 'sst_mean_c': 'SST (mean)',
    'sst_max_c': 'SST (max)', 'sst_min_c': 'SST (min)',
    'sst_range_c': 'SST (range)',
    'air_temp_mean_c': 'Air T (mean)', 'air_temp_max_c': 'Air T (max)',
    'air_temp_min_c': 'Air T (min)', 'air_temp_range_c': 'Air T (range)',
    'bathymetry_m': 'Bathymetry', 'elevation_m': 'Elevation',
    'distance_to_coast_km': 'Dist. coast',
    'solar_rad_mj_m2': 'Solar radiation',
    'nflh_mean': 'NFLH', 'chl_mean_mg_m3': 'Chl-a (mean)',
    'chl_max_mg_m3': 'Chl-a (max)', 'chl_min_mg_m3': 'Chl-a (min)',
    'poc_mean_mg_m3': 'POC', 'precip_mean_mm': 'Precipitation',
    'rrs_412': 'Rrs 412', 'rrs_443': 'Rrs 443',
    'rrs_469': 'Rrs 469', 'rrs_488': 'Rrs 488',
    'rrs_531': 'Rrs 531', 'rrs_547': 'Rrs 547',
    'rrs_555': 'Rrs 555', 'rrs_645': 'Rrs 645',
    'rrs_667': 'Rrs 667', 'rrs_678': 'Rrs 678',
    'nitrate_umol_l': 'Nitrate', 'phosphate_umol_l': 'Phosphate',
    'silicate_umol_l': 'Silicate', 'oxygen_umol_l': 'Diss. O₂',
    'mld_m': 'MLD', 'salinity_psu_est': 'Salinity',
    'landcover_class': 'Land cover', 'depth_m': 'Depth',
}

boot_df_sorted['category'] = boot_df_sorted['env_target'].map(env_categories)
boot_df_sorted['color'] = boot_df_sorted['category'].map(ENV_CATEGORY_COLORS)
boot_df_sorted['label'] = boot_df_sorted['env_target'].map(nice_labels)
# Fill any missing labels/colors
boot_df_sorted['label'] = boot_df_sorted.apply(
    lambda r: r['label'] if pd.notna(r['label']) else r['env_target'], axis=1)
boot_df_sorted['color'] = boot_df_sorted['color'].apply(
    lambda c: c if isinstance(c, tuple) else (0.5, 0.5, 0.5))

# Clip CIs for display
boot_df_sorted['ci_low_clipped'] = boot_df_sorted['ci_95_low'].clip(lower=-0.5)
boot_df_sorted['ci_high_clipped'] = boot_df_sorted['ci_95_high'].clip(upper=1.0)

n_targets = len(boot_df_sorted)
print(f"  Panel C: {n_targets} targets, best={boot_df_sorted.iloc[-1]['env_target']} (R2={boot_df_sorted.iloc[-1]['r2_point']:.3f})")

# ============================================================================
# CREATE COMPOSITE FIGURE
# ============================================================================
print("\n" + "=" * 70)
print("CREATING MERGED FIGURE S4")
print("=" * 70)

# Figure with 2 rows: top row has A+B side by side, bottom has C (forest plot)
fig_height = 3.0 + 0.16 * n_targets + 0.8  # top panels + forest plot
fig = plt.figure(figsize=(7.5, fig_height))

outer_gs = gridspec.GridSpec(2, 1, figure=fig,
                             height_ratios=[2.5, 0.16 * n_targets + 0.8],
                             hspace=0.25)

# ── Row 1: (A) Correlation distribution | (B) XGBoost CV R2 ──
row1_gs = outer_gs[0].subgridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.35)

# Panel A: Spearman correlation distribution
ax_a = fig.add_subplot(row1_gs[0, 0])
n_bins = 80
counts, bins, patches = ax_a.hist(rho_values, bins=n_bins, edgecolor='none', alpha=0.9)

# Color bins by sign
for patch, bin_center in zip(patches, (bins[:-1] + bins[1:]) / 2):
    if bin_center < -0.1:
        patch.set_facecolor(DEEP_OCEAN)
    elif bin_center > 0.1:
        patch.set_facecolor(SIENNA)
    else:
        patch.set_facecolor((0.6, 0.6, 0.6))

ax_a.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax_a.axvline(x=max_rho, color=SIENNA, linestyle='--', linewidth=0.5, alpha=0.7)
ax_a.axvline(x=min_rho, color=DEEP_OCEAN, linestyle='--', linewidth=0.5, alpha=0.7)
ax_a.set_xlabel('Spearman correlation (rho)')
ax_a.set_ylabel('Frequency')
ax_a.set_xlim(-0.7, 0.7)

stats_text = (f"n = {total_tests:,} tests\n"
              f"{pct_sig:.1f}% significant\n"
              f"(FDR < 0.05)\n"
              f"rho range: [{min_rho:.2f}, {max_rho:.2f}]")
ax_a.text(0.97, 0.97, stats_text, transform=ax_a.transAxes,
          fontsize=5, va='top', ha='right',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor=(0.7, 0.7, 0.7), alpha=0.9, linewidth=0.25))
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.text(-0.12, 1.05, 'A', transform=ax_a.transAxes, fontsize=8, fontweight='bold', va='top')

# Panel B: XGBoost CV R2
ax_b = fig.add_subplot(row1_gs[0, 1])
x_pos = np.arange(len(dims_sorted))
bars = ax_b.bar(x_pos, r2_vals, yerr=r2_errs, capsize=2,
                color=COASTAL_BLUE, edgecolor='none', alpha=0.9,
                error_kw={'linewidth': 0.5, 'capthick': 0.5})
bars[0].set_color(SIENNA)  # Highlight top performer

ax_b.set_xlabel('AlphaEarth dimension')
ax_b.set_ylabel('Cross-validation R2')
ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(dims_sorted, rotation=45, ha='right')
ax_b.set_ylim(0, 0.65)
ax_b.axhline(y=0.3, color=(0.6, 0.6, 0.6), linestyle=':', linewidth=0.5, alpha=0.7)

ax_b.annotate(f'R2 = {r2_vals[0]:.2f}',
              xy=(0, r2_vals[0]),
              xytext=(1.5, r2_vals[0] + 0.08),
              fontsize=5, ha='left',
              arrowprops=dict(arrowstyle='->', lw=0.5, color=(0.5, 0.5, 0.5)))
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.text(-0.15, 1.05, 'B', transform=ax_b.transAxes, fontsize=8, fontweight='bold', va='top')

# ── Row 2: (C) Bootstrap CI forest plot (full width) ──
ax_c = fig.add_subplot(outer_gs[1])

y_positions = np.arange(n_targets)

for i, (_, row) in enumerate(boot_df_sorted.iterrows()):
    color = row['color']
    if isinstance(color, float):
        color = (0.5, 0.5, 0.5)
    r2 = row['r2_point']
    ci_lo = row['ci_low_clipped']
    ci_hi = row['ci_high_clipped']

    # CI line
    ax_c.plot([ci_lo, ci_hi], [i, i], color=color, linewidth=0.8, solid_capstyle='round')
    # Point estimate
    ax_c.plot(r2, i, 'o', color=color, markersize=3, markeredgecolor='white',
              markeredgewidth=0.3, zorder=5)
    # R2 annotation
    ax_c.text(max(ci_hi, r2) + 0.02, i, f'{r2:.3f}',
              ha='left', va='center', fontsize=4.5, color=(0.3, 0.3, 0.3))

# Reference lines
ax_c.axvline(x=0, color=(0.5, 0.5, 0.5), linewidth=0.25, linestyle='--', zorder=0)
ax_c.axvline(x=0.2, color=(0.7, 0.7, 0.7), linewidth=0.25, linestyle=':', zorder=0)

ax_c.set_yticks(y_positions)
ax_c.set_yticklabels(boot_df_sorted['label'].values, fontsize=5)
ax_c.set_xlabel('Test R2 (80/20 split)')
ax_c.set_xlim(-0.55, 0.85)
ax_c.set_ylim(-0.5, n_targets - 0.5)

# Category legend
categories_present = boot_df_sorted['category'].dropna().unique()
legend_handles = []
for cat in sorted(categories_present):
    color = ENV_CATEGORY_COLORS.get(cat, (0.5, 0.5, 0.5))
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=color, markersize=3,
                                      label=cat, markeredgewidth=0.3,
                                      markeredgecolor='white'))

ax_c.legend(handles=legend_handles, loc='upper center', framealpha=0.9,
            edgecolor='none', fontsize=5, ncol=min(len(legend_handles), 7), columnspacing=0.5,
            handletextpad=0.3, bbox_to_anchor=(0.5, 1.06))

ax_c.set_title('Reverse model R2 with 95% bootstrap CI (1,000 resamples)',
               fontsize=6, fontweight='bold', pad=18)
ax_c.text(0.5, 1.035, '95% CI from 1,000 bootstrap resamples of test-set predictions',
          transform=ax_c.transAxes, ha='center', va='top', fontsize=4.5,
          color=(0.4, 0.4, 0.4))

ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.text(-0.08, 1.06, 'C', transform=ax_c.transAxes, fontsize=8, fontweight='bold', va='top')

# ============================================================================
# SAVE OUTPUTS
# ============================================================================
print("\n" + "=" * 70)
print("SAVING MERGED FIGURE S4")
print("=" * 70)

for fmt in ['pdf', 'svg']:
    filename = SCRIPT_DIR / f"FigureS4_correlation_modeling_{TIMESTAMP}.{fmt}"
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
print("SUMMARY — Merged Figure S4: Correlation & Modeling Performance")
print("=" * 70)
print(f"  Panel A: Spearman correlation distribution ({total_tests:,} tests)")
print(f"  Panel B: XGBoost CV R2 ({len(dims_sorted)} dimensions, best={dims_sorted[0]})")
print(f"  Panel C: Bootstrap CI forest plot ({n_targets} targets)")
print(f"\n{'=' * 70}")
print(f"COMPLETE — {datetime.now().isoformat()}")
print(f"{'=' * 70}")
