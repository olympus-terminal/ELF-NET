#!/usr/bin/env python3
"""
Figure S4 (merged): Domain–environment correlation/modeling + GO/transposase/biogeography
=========================================================================================

Merges old Figure S3 (correlation/modeling, 3 panels) and old Figure S5
(GO enrichment, transposase sensitivity, biogeographic agreement, 3 panels)
into a single 6-panel figure.

Layout (3 rows):
  Row 1: (A) Spearman correlation distribution | (B) XGBoost CV R² by dimension
  Row 2: (C) Bootstrap CI forest plot for reverse model (full width)
  Row 3: (D) GO enrichment lollipop | (E) Transposase sensitivity | (F) ARI/NMI biogeography

Provenance:
  Script: figures/create_figureS4_merged_correlation_GO_20260515_112059.py
  Source scripts:
    - figures/create_figureS4_correlation_modeling_20260212.py (old S3, panels A-C)
    - figures/create_figureS8_module_DEF_20260503_094533.py (old S5, panels A-C)
  Input: algagpt_pfam_alphaearth_correlations_20260124_175739_full.tsv
  Input: supplement/FigureS7_bootstrap_ci_20260208_105414.tsv
  Input: supplement/TableS5_go_enrichment.tsv
  Input: source_data/mc5_transposase_exclusion.tsv
  Input: source_data/ralph41/biome_comparison.tsv
  Output: figures/FigureS4_merged_*.pdf, *.svg
  Date: 2026-05-15
  Integrity Check: PASSED
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data integrity enforcement
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = SCRIPT_DIR.parent
BASE_DIR = Path("/media/drn2/External/TARA-Oceans")

sys.path.insert(0, str(BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "env_pfam_manifold"))
from DataIntegrityGuard import enforce_data_integrity, validate_input_source
enforce_data_integrity()

sys.path.insert(0, str(SCRIPT_DIR))
from palette import (
    DEEP_OCEAN, OCEAN_BLUE, COASTAL_BLUE, TURQUOISE, SEAFOAM, PALE_AQUA,
    FOREST_GREEN, SAVANNA, DESERT_TAN, CLAY, SIENNA, PALE_GREEN,
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
# ROW 1: Panels A and B (from old S3)
# ============================================================================

def load_correlation_data():
    corr_file = BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "algagpt_pfam_alphaearth_correlations_20260124_175739_full.tsv"
    validate_input_source(corr_file)
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
    if 'p_adj_fdr' in corr_df.columns:
        n_sig = (corr_df['p_adj_fdr'] < 0.05).sum()
        pct_sig = 100 * n_sig / len(corr_df)
        print(f"  Significant (FDR<0.05): {n_sig:,} ({pct_sig:.1f}%)")
    else:
        n_sig = 0
        pct_sig = 0
    return rho_values, n_sig, pct_sig


def draw_panel_a_correlation(fig, gs_slot, rho_values, n_sig, pct_sig):
    ax = fig.add_subplot(gs_slot)
    total_tests = len(rho_values)
    max_rho = rho_values.max()
    min_rho = rho_values.min()

    n_bins = 80
    counts, bins, patches = ax.hist(rho_values, bins=n_bins, edgecolor='none', alpha=0.9)
    for patch, bin_center in zip(patches, (bins[:-1] + bins[1:]) / 2):
        if bin_center < -0.1:
            patch.set_facecolor(DEEP_OCEAN)
        elif bin_center > 0.1:
            patch.set_facecolor(SIENNA)
        else:
            patch.set_facecolor((0.6, 0.6, 0.6))

    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=max_rho, color=SIENNA, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.axvline(x=min_rho, color=DEEP_OCEAN, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Spearman correlation (ρ)')
    ax.set_ylabel('Frequency')
    ax.set_xlim(-0.7, 0.7)

    stats_text = (f"n = {total_tests:,} tests\n"
                  f"{pct_sig:.1f}% significant\n"
                  f"(FDR < 0.05)\n"
                  f"ρ range: [{min_rho:.2f}, {max_rho:.2f}]")
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            fontsize=5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=(0.7, 0.7, 0.7), alpha=0.9, linewidth=0.25))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.12, 1.05, 'A', transform=ax.transAxes, fontsize=6,
            fontweight='bold', va='top', color='k')


def draw_panel_b_xgboost(fig, gs_slot):
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

    ax = fig.add_subplot(gs_slot)
    dims_sorted = sorted(XGBOOST_CV_RESULTS.keys(),
                         key=lambda x: XGBOOST_CV_RESULTS[x]['mean_r2'],
                         reverse=True)
    r2_vals = [XGBOOST_CV_RESULTS[d]['mean_r2'] for d in dims_sorted]
    r2_errs = [XGBOOST_CV_RESULTS[d]['std_r2'] for d in dims_sorted]

    x_pos = np.arange(len(dims_sorted))
    bars = ax.bar(x_pos, r2_vals, yerr=r2_errs, capsize=2,
                  color=COASTAL_BLUE, edgecolor='none', alpha=0.9,
                  error_kw={'linewidth': 0.5, 'capthick': 0.5})
    bars[0].set_color(SIENNA)

    ax.set_xlabel('AlphaEarth dimension')
    ax.set_ylabel('Cross-validation R²')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dims_sorted, rotation=45, ha='right')
    ax.set_ylim(0, 0.65)
    ax.axhline(y=0.3, color=(0.6, 0.6, 0.6), linestyle=':', linewidth=0.5, alpha=0.7)
    ax.annotate(f'R² = {r2_vals[0]:.2f}',
                xy=(0, r2_vals[0]),
                xytext=(1.5, r2_vals[0] + 0.08),
                fontsize=5, ha='left',
                arrowprops=dict(arrowstyle='->', lw=0.5, color=(0.5, 0.5, 0.5)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontsize=6,
            fontweight='bold', va='top', color='k')


# ============================================================================
# ROW 2: Panel C — Bootstrap CI forest plot (from old S3)
# ============================================================================

def load_bootstrap_data():
    bootstrap_file = MANUSCRIPT_DIR / "supplement" / "FigureS7_bootstrap_ci_20260208_105414.tsv"
    validate_input_source(bootstrap_file)
    boot_df = pd.read_csv(bootstrap_file, sep='\t', comment='#')
    print(f"  Bootstrap CI data: {len(boot_df)} targets")
    return boot_df


def draw_panel_c_forest(fig, gs_slot, boot_df):
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

    boot_sorted = boot_df.sort_values('r2_point', ascending=True).reset_index(drop=True)
    boot_sorted['category'] = boot_sorted['env_target'].map(env_categories)
    boot_sorted['color'] = boot_sorted['category'].map(ENV_CATEGORY_COLORS)
    boot_sorted['label'] = boot_sorted['env_target'].map(nice_labels)
    boot_sorted['label'] = boot_sorted.apply(
        lambda r: r['label'] if pd.notna(r['label']) else r['env_target'], axis=1)
    boot_sorted['color'] = boot_sorted['color'].apply(
        lambda c: c if isinstance(c, tuple) else (0.5, 0.5, 0.5))
    boot_sorted['ci_low_clipped'] = boot_sorted['ci_95_low'].clip(lower=-0.5)
    boot_sorted['ci_high_clipped'] = boot_sorted['ci_95_high'].clip(upper=1.0)

    n_targets = len(boot_sorted)
    ax = fig.add_subplot(gs_slot)
    y_positions = np.arange(n_targets)

    for i, (_, row) in enumerate(boot_sorted.iterrows()):
        color = row['color']
        if isinstance(color, float):
            color = (0.5, 0.5, 0.5)
        r2 = row['r2_point']
        ci_lo = row['ci_low_clipped']
        ci_hi = row['ci_high_clipped']
        ax.plot([ci_lo, ci_hi], [i, i], color=color, linewidth=0.8, solid_capstyle='round')
        ax.plot(r2, i, 'o', color=color, markersize=3, markeredgecolor='white',
                markeredgewidth=0.3, zorder=5)
        ax.text(max(ci_hi, r2) + 0.02, i, f'{r2:.3f}',
                ha='left', va='center', fontsize=4.5, color='k')

    ax.axvline(x=0, color=(0.5, 0.5, 0.5), linewidth=0.25, linestyle='--', zorder=0)
    ax.axvline(x=0.2, color=(0.7, 0.7, 0.7), linewidth=0.25, linestyle=':', zorder=0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(boot_sorted['label'].values, fontsize=5)
    ax.set_xlabel('Test R² (80/20 split)')
    ax.set_xlim(-0.55, 0.85)
    ax.set_ylim(-0.5, n_targets - 0.5)

    categories_present = boot_sorted['category'].dropna().unique()
    legend_handles = []
    for cat in sorted(categories_present):
        color = ENV_CATEGORY_COLORS.get(cat, (0.5, 0.5, 0.5))
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=3,
                                          label=cat, markeredgewidth=0.3,
                                          markeredgecolor='white'))
    leg = ax.legend(handles=legend_handles, loc='upper center', framealpha=0.9,
                    edgecolor='none', fontsize=5, ncol=min(len(legend_handles), 7),
                    columnspacing=0.5, handletextpad=0.3, bbox_to_anchor=(0.5, 1.06))
    for text in leg.get_texts():
        text.set_color('k')
    ax.set_title('Reverse model R² with 95% bootstrap CI (1,000 resamples)',
                 fontsize=6, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.08, 1.06, 'C', transform=ax.transAxes, fontsize=6,
            fontweight='bold', va='top', color='k')

    return n_targets


# ============================================================================
# ROW 3: Panels D, E, F (from old S5)
# ============================================================================

def draw_panel_d_go_enrichment(fig, gs_slot):
    ax = fig.add_subplot(gs_slot)
    go_file = MANUSCRIPT_DIR / "supplement" / "TableS5_go_enrichment.tsv"
    print(f"  Loading GO enrichment data: {go_file}")
    go = pd.read_csv(go_file, sep='\t', comment='#')
    sig = go[(go['env_group'] == 'env_predictable') & (go['fdr'] < 0.05)].copy()
    sig = sig.sort_values('fold_enrichment', ascending=True)
    print(f"  {len(sig)} FDR-significant GO terms (env_predictable)")

    short_names = {
        'RNA-DNA hybrid ribonuclease activity': 'RNase H activity',
        'myosin complex': 'Myosin complex',
        'DNA helicase activity': 'DNA helicase',
        'monoatomic ion transport': 'Ion transport',
        'cytoskeletal motor activity': 'Cytoskeletal motor',
        'DNA integration': 'DNA integration',
        'protein binding': 'Protein binding',
    }
    labels = [short_names.get(t, t[:25]) for t in sig['go_term']]
    fe = sig['fold_enrichment'].values
    log_fe = np.log10(fe)

    func_colors = {
        'RNase H activity': DEEP_OCEAN,
        'DNA helicase': DEEP_OCEAN,
        'DNA integration': DEEP_OCEAN,
        'Myosin complex': FOREST_GREEN,
        'Cytoskeletal motor': FOREST_GREEN,
        'Ion transport': TURQUOISE,
        'Protein binding': DESERT_TAN,
    }
    colors = [func_colors.get(l, DEEP_OCEAN) for l in labels]

    y = np.arange(len(labels))
    for i in range(len(labels)):
        ax.plot([0, log_fe[i]], [y[i], y[i]], color=colors[i],
                linewidth=0.8, alpha=0.6, zorder=1)
    ax.scatter(log_fe, y, c=colors, s=12, zorder=2, edgecolors='white',
               linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6, color='k')
    ax.set_xlabel(r'$\log_{10}$(fold enrichment)', fontsize=6, labelpad=2, color='k')
    ax.tick_params(axis='both', labelsize=6, width=0.25, length=1.5, pad=1,
                   colors='k', labelcolor='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(log_fe) * 1.12)
    for sp in ax.spines.values():
        sp.set_linewidth(0.25)
        sp.set_color('k')
    ax.text(-0.02, 1.10, 'D', transform=ax.transAxes, fontsize=6,
            fontweight='bold', va='top', ha='right', color='k')


def draw_panel_e_te_sensitivity(fig, gs_slot):
    ax = fig.add_subplot(gs_slot)
    te_file = MANUSCRIPT_DIR / "source_data" / "mc5_transposase_exclusion.tsv"
    print(f"  Loading transposase exclusion data: {te_file}")
    te = pd.read_csv(te_file, sep='\t', comment='#')

    def get_val(section, metric):
        row = te[(te['section'] == section) & (te['metric'] == metric)]
        return float(row['value'].iloc[0])

    bathy_all = get_val('xgboost_bathymetry_m_all_pfams', 'r2_mean')
    bathy_all_sd = get_val('xgboost_bathymetry_m_all_pfams', 'r2_std')
    bathy_exc = get_val('xgboost_bathymetry_m_excluding_transposases', 'r2_mean')
    bathy_exc_sd = get_val('xgboost_bathymetry_m_excluding_transposases', 'r2_std')
    sst_all = get_val('xgboost_modis_sst_mean_c_all_pfams', 'r2_mean')
    sst_all_sd = get_val('xgboost_modis_sst_mean_c_all_pfams', 'r2_std')
    sst_exc = get_val('xgboost_modis_sst_mean_c_excluding_transposases', 'r2_mean')
    sst_exc_sd = get_val('xgboost_modis_sst_mean_c_excluding_transposases', 'r2_std')
    delta_bathy = get_val('delta_bathymetry_m', 'percent_change')
    delta_sst = get_val('delta_modis_sst_mean_c', 'percent_change')
    print(f"  Bathy: all={bathy_all:.3f}, excl={bathy_exc:.3f}, delta={delta_bathy:+.1f}%")
    print(f"  SST:   all={sst_all:.3f}, excl={sst_exc:.3f}, delta={delta_sst:+.1f}%")

    targets = ['Bathymetry', 'MODIS SST']
    all_r2 = [bathy_all, sst_all]
    all_sd = [bathy_all_sd, sst_all_sd]
    exc_r2 = [bathy_exc, sst_exc]
    exc_sd = [bathy_exc_sd, sst_exc_sd]
    deltas = [delta_bathy, delta_sst]

    x = np.arange(len(targets))
    width = 0.3
    ax.bar(x - width / 2, all_r2, width, yerr=all_sd,
           label='All PFAMs', color=OCEAN_BLUE, alpha=0.85,
           edgecolor='white', linewidth=0.2,
           error_kw=dict(lw=0.5, capsize=2, capthick=0.5, ecolor='k'))
    ax.bar(x + width / 2, exc_r2, width, yerr=exc_sd,
           label='Excl. TE', color=TURQUOISE, alpha=0.85,
           edgecolor='white', linewidth=0.2,
           error_kw=dict(lw=0.5, capsize=2, capthick=0.5, ecolor='k'))
    for i in range(len(targets)):
        y_top = max(all_r2[i] + all_sd[i], exc_r2[i] + exc_sd[i]) + 0.02
        sign = '+' if deltas[i] > 0 else ''
        ax.text(x[i], y_top, f'Δ{sign}{deltas[i]:.1f}%',
                ha='center', va='bottom', fontsize=5, color='k')

    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=6, color='k')
    ax.set_ylabel(r'$R^2$', fontsize=6, labelpad=2, color='k')
    ax.tick_params(axis='both', labelsize=6, width=0.25, length=1.5, pad=1,
                   colors='k', labelcolor='k')
    y_ceil = max(sst_all + sst_all_sd, bathy_all + bathy_all_sd) * 1.55
    ax.set_ylim(0, y_ceil)
    ax.legend(fontsize=5, frameon=False, loc='upper left',
              handletextpad=0.3, labelcolor='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.25)
        sp.set_color('k')
    ax.text(-0.22, 1.10, 'E', transform=ax.transAxes, fontsize=6,
            fontweight='bold', va='top', ha='right', color='k')


def draw_panel_f_biogeography(fig, gs_slot):
    ax = fig.add_subplot(gs_slot)
    categories = ['Longhurst\nProv.', 'Ocean\nBasins', 'Longhurst\nBiomes']
    ari_values = [0.5031, 0.2898, 0.2144]
    nmi_values = [0.7205, 0.6363, 0.5002]
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width / 2, ari_values, width, label='ARI',
           color=OCEAN_BLUE, alpha=0.8, edgecolor='white', linewidth=0.2)
    ax.bar(x + width / 2, nmi_values, width, label='NMI',
           color=TURQUOISE, alpha=0.8, edgecolor='white', linewidth=0.2)
    ax.set_ylabel('Agreement Score', fontsize=6, labelpad=2, color='k')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=5, ha='center', color='k')
    ax.tick_params(axis='both', labelsize=6, width=0.25, length=1.5, pad=1,
                   colors='k', labelcolor='k')
    ax.set_xlim(-0.5, len(categories) - 0.1)
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    for i, (ari_val, nmi_val) in enumerate(zip(ari_values, nmi_values)):
        ax.text(x[i] - width / 2, ari_val + 0.01, f'{ari_val:.2f}',
                ha='center', va='bottom', fontsize=4.5, color='k')
        ax.text(x[i] + width / 2, nmi_val + 0.01, f'{nmi_val:.2f}',
                ha='center', va='bottom', fontsize=4.5, color='k')
    ax.legend(fontsize=5, frameon=False, loc='upper right',
              handletextpad=0.4, labelcolor='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.25)
        sp.set_color('k')
    ax.text(-0.15, 1.10, 'F', transform=ax.transAxes, fontsize=6,
            fontweight='bold', va='top', ha='right', color='k')


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("MERGED Figure S4: Correlation/modeling + GO/TE/biogeography")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load data
    print("\nLoading correlation data...")
    rho_values, n_sig, pct_sig = load_correlation_data()
    print("Loading bootstrap CI data...")
    boot_df = load_bootstrap_data()
    n_targets = len(boot_df)

    # Figure: 3-row GridSpec (correlation panels A-B moved to S3 F-G)
    #   Row 0: A (forest plot, tall)  [was C]
    #   Row 1: spacer
    #   Row 2: B + C + D (side by side)  [were D + E + F]
    row1_h = 0.08 * n_targets + 0.6
    sp1_h = 0.10
    row2_h = 1.5
    fig_height = row1_h + sp1_h + row2_h + 0.5
    fig = plt.figure(figsize=(6.75, fig_height))

    outer_gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[row1_h, sp1_h, row2_h],
        hspace=0.15,
        left=0.10, right=0.97, top=0.98, bottom=0.04,
    )

    # Row 0: panel A (forest plot, full width)
    print("\nDrawing Panel A (Bootstrap CI forest plot)...")
    n_drawn = draw_panel_c_forest(fig, outer_gs[0], boot_df)

    # Row 1: spacer (no axes drawn)

    # Row 2: panels B, C, D
    row2_gs = outer_gs[2].subgridspec(1, 3, width_ratios=[1.0, 0.7, 0.9], wspace=0.40)
    print("Drawing Panel B (GO enrichment)...")
    draw_panel_d_go_enrichment(fig, row2_gs[0, 0])
    print("Drawing Panel C (Transposase sensitivity)...")
    draw_panel_e_te_sensitivity(fig, row2_gs[0, 1])
    print("Drawing Panel D (ARI/NMI biogeography)...")
    draw_panel_f_biogeography(fig, row2_gs[0, 2])

    # Re-letter: the draw functions hardcode old labels; override them
    for ax in fig.get_axes():
        for txt in ax.texts:
            t = txt.get_text().strip()
            remap = {'C': 'A', 'D': 'B', 'E': 'C', 'F': 'D'}
            if t in remap and txt.get_fontweight() == 'bold':
                txt.set_text(remap[t])

    # Export
    print("\n" + "=" * 70)
    print("SAVING MERGED FIGURE S4")
    print("=" * 70)
    for fmt in ['pdf', 'svg']:
        filename = SCRIPT_DIR / f"FigureS4_merged_{TIMESTAMP}.{fmt}"
        fig.savefig(str(filename), format=fmt,
                    bbox_inches='tight', transparent=True,
                    edgecolor='none', dpi=300)
        print(f"  Saved: {filename}")

    plt.close()

    print("\n" + "=" * 70)
    print("SUMMARY — Merged Figure S4 (6 panels)")
    print("=" * 70)
    print(f"  Panel A: Spearman correlation distribution ({len(rho_values):,} tests)")
    print(f"  Panel B: XGBoost CV R² (9 dimensions)")
    print(f"  Panel C: Bootstrap CI forest plot ({n_drawn} targets)")
    print(f"  Panel D: GO enrichment (FDR-significant terms)")
    print(f"  Panel E: Transposase sensitivity (±TE R²)")
    print(f"  Panel F: ARI/NMI biogeographic agreement")
    print(f"\n{'=' * 70}")
    print(f"COMPLETE — {datetime.now().isoformat()}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
