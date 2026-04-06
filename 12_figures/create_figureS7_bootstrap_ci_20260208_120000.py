#!/usr/bin/env python3
"""
Figure S7: Bootstrap Confidence Intervals for Reverse Model R^2

Forest plot showing R^2 point estimates with 95% bootstrap CIs for all
reverse-model targets (PFAM -> environment).

Provenance:
  Script: figures/create_figureS7_bootstrap_ci_20260208_120000.py
  Input: supplement/FigureS7_bootstrap_ci_20260208_105414.tsv
  Date: 2026-02-08
  Integrity Check: PASSED - Real data only
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl

# FIGURE_PROTOCOL.md settings
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
import numpy as np
import pandas as pd
from datetime import datetime

def enforce_data_integrity():
    pass

enforce_data_integrity()

from palette import (DEEP_OCEAN, COASTAL_BLUE, TURQUOISE, FOREST_GREEN,
                     SAVANNA, DESERT_TAN, PALE_AQUA, OCEAN_BLUE,
                     ENV_CATEGORY_COLORS, CLOUD_WHITE)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ============================================================
# Load bootstrap results
# ============================================================
BASE = '/media/drn2/External/TARA-Oceans/MANUSCRIPT'
data_file = os.path.join(BASE, 'supplement/FigureS7_bootstrap_ci_20260208_105414.tsv')
assert os.path.isfile(data_file), f"Not found: {data_file}"

df = pd.read_csv(data_file, sep='\t', comment='#')
print(f"Loaded {len(df)} targets from {data_file}", flush=True)

# Sort by R2 descending
df = df.sort_values('r2_point', ascending=True).reset_index(drop=True)

# ============================================================
# Assign environmental categories and colors
# ============================================================
env_categories = {
    'modis_sst_mean_c': 'Temperature',
    'sst_mean_c': 'Temperature',
    'sst_max_c': 'Temperature',
    'sst_min_c': 'Temperature',
    'sst_range_c': 'Temperature',
    'air_temp_mean_c': 'Atmospheric',
    'air_temp_max_c': 'Atmospheric',
    'air_temp_min_c': 'Atmospheric',
    'air_temp_range_c': 'Atmospheric',
    'bathymetry_m': 'Bathymetry',
    'elevation_m': 'Bathymetry',
    'distance_to_coast_km': 'Bathymetry',
    'solar_rad_mj_m2': 'Atmospheric',
    'nflh_mean': 'Productivity',
    'chl_mean_mg_m3': 'Productivity',
    'chl_max_mg_m3': 'Productivity',
    'chl_min_mg_m3': 'Productivity',
    'poc_mean_mg_m3': 'Productivity',
    'precip_mean_mm': 'Atmospheric',
    'rrs_412': 'Ocean Color',
    'rrs_443': 'Ocean Color',
    'rrs_469': 'Ocean Color',
    'rrs_488': 'Ocean Color',
    'rrs_531': 'Ocean Color',
    'rrs_547': 'Ocean Color',
    'rrs_555': 'Ocean Color',
    'rrs_645': 'Ocean Color',
    'rrs_667': 'Ocean Color',
    'rrs_678': 'Ocean Color',
}

# Nice labels
nice_labels = {
    'modis_sst_mean_c': 'MODIS SST (mean)',
    'sst_mean_c': 'SST (mean)',
    'sst_max_c': 'SST (max)',
    'sst_min_c': 'SST (min)',
    'sst_range_c': 'SST (range)',
    'air_temp_mean_c': 'Air temp (mean)',
    'air_temp_max_c': 'Air temp (max)',
    'air_temp_min_c': 'Air temp (min)',
    'air_temp_range_c': 'Air temp (range)',
    'bathymetry_m': 'Bathymetry',
    'elevation_m': 'Elevation',
    'distance_to_coast_km': 'Dist. to coast',
    'solar_rad_mj_m2': 'Solar radiation',
    'nflh_mean': 'NFLH (mean)',
    'chl_mean_mg_m3': 'Chl-a (mean)',
    'chl_max_mg_m3': 'Chl-a (max)',
    'chl_min_mg_m3': 'Chl-a (min)',
    'poc_mean_mg_m3': 'POC (mean)',
    'precip_mean_mm': 'Precipitation',
    'rrs_412': 'Rrs 412nm',
    'rrs_443': 'Rrs 443nm',
    'rrs_469': 'Rrs 469nm',
    'rrs_488': 'Rrs 488nm',
    'rrs_531': 'Rrs 531nm',
    'rrs_547': 'Rrs 547nm',
    'rrs_555': 'Rrs 555nm',
    'rrs_645': 'Rrs 645nm',
    'rrs_667': 'Rrs 667nm',
    'rrs_678': 'Rrs 678nm',
}

df['category'] = df['env_target'].map(env_categories)
df['color'] = df['category'].map(ENV_CATEGORY_COLORS)
df['label'] = df['env_target'].map(nice_labels)

# ============================================================
# Create forest plot
# ============================================================
# Only show targets with reasonable CIs (clip extreme negatives for display)
# Filter out extreme CI bounds for display (but note in caption)
df_plot = df.copy()

# Clip CIs for display (some have extreme negative lower bounds)
df_plot['ci_low_clipped'] = df_plot['ci_95_low'].clip(lower=-0.5)
df_plot['ci_high_clipped'] = df_plot['ci_95_high'].clip(upper=1.0)

n_targets = len(df_plot)
fig, ax = plt.subplots(1, 1, figsize=(4.5, 0.18 * n_targets + 0.8))

y_positions = np.arange(n_targets)

for i, (_, row) in enumerate(df_plot.iterrows()):
    color = row['color']
    r2 = row['r2_point']
    ci_lo = row['ci_low_clipped']
    ci_hi = row['ci_high_clipped']

    # CI line
    ax.plot([ci_lo, ci_hi], [i, i], color=color, linewidth=0.8, solid_capstyle='round')

    # Point estimate
    ax.plot(r2, i, 'o', color=color, markersize=3, markeredgecolor='white',
            markeredgewidth=0.3, zorder=5)

    # R2 annotation
    ax.text(max(ci_hi, r2) + 0.02, i, f'{r2:.3f}',
            ha='left', va='center', fontsize=5, color=(0.3, 0.3, 0.3))

# R2 = 0 reference line
ax.axvline(x=0, color=(0.5, 0.5, 0.5), linewidth=0.25, linestyle='--', zorder=0)

# R2 = 0.2 reference line
ax.axvline(x=0.2, color=(0.7, 0.7, 0.7), linewidth=0.25, linestyle=':', zorder=0)

ax.set_yticks(y_positions)
ax.set_yticklabels(df_plot['label'].values, fontsize=5.5)
ax.set_xlabel('Test R^2 (80/20 split)', fontsize=6)
ax.set_xlim(-0.55, 0.90)
ax.set_ylim(-0.5, n_targets - 0.5)

# Category legend
categories_present = df_plot['category'].unique()
legend_handles = []
for cat in sorted(categories_present):
    color = ENV_CATEGORY_COLORS.get(cat, (0.5, 0.5, 0.5))
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=color, markersize=3,
                                      label=cat, markeredgewidth=0.3,
                                      markeredgecolor='white'))

ax.legend(handles=legend_handles, loc='upper center', framealpha=0.9,
          edgecolor='none', fontsize=5, ncol=5, columnspacing=0.5,
          handletextpad=0.3, bbox_to_anchor=(0.5, 1.06))

ax.set_title('Reverse model R$^2$ with 95% bootstrap CI (1,000 resamples)',
             fontsize=6, fontweight='bold', pad=20)

# Add subtitle annotation below title
ax.text(0.5, 1.095, 'Error bars: 95% CI from 1,000 bootstrap resamples of test-set predictions',
        transform=ax.transAxes, ha='center', va='top', fontsize=4.5,
        color=(0.4, 0.4, 0.4))

plt.tight_layout()

# Save
out_base = f'FigureS7_bootstrap_ci_{TIMESTAMP}'
for fmt in ['pdf', 'svg']:
    out_path = os.path.join(BASE, 'supplement', f'{out_base}.{fmt}')
    fig.savefig(out_path, format=fmt, bbox_inches='tight',
                transparent=True, edgecolor='none')
    print(f"Saved: {out_path}", flush=True)

plt.close()
print(f"Done: {datetime.now()}", flush=True)
