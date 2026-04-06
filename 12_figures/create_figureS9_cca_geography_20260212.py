#!/usr/bin/env python3
"""
Ralph25 Task 3: Geographic mapping of CCA space — CC1 on world map

Creates a Robinson-projection world map with 1809 CCA samples colored by their
CC1 score (env-side canonical variate). Uses the Earth-from-Space diverging
colormap since CC1 scores are centered at zero.

Provenance:
  Input: /media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data/*.npy
  Input: /media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv
  Output: figures/FigureS9_cca_geography_*.pdf, *.svg
  Date: 2026-02-12
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
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

sys.path.insert(0, str(BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "env_pfam_manifold"))
from DataIntegrityGuard import enforce_data_integrity, validate_input_source, create_provenance_header
enforce_data_integrity()

# Import palette
sys.path.insert(0, str(SCRIPT_DIR))
from palette import get_diverging_cmap, CLOUD_WHITE, ABYSS, DEEP_OCEAN, PALE_AQUA

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# FIGURE_PROTOCOL.md: matplotlib settings
# ============================================================================
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

# ============================================================================
# Load data
# ============================================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

input_files = {
    'env_embedding': CCA_DATA_DIR / "cca_env_embedding_20260122_104244.npy",
    'sample_ids': CCA_DATA_DIR / "sample_ids_20260122_101559.npy",
    'merged_dataset': BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "algagpt_gee_pfam_merged_SMART_20260119_100639.tsv",
}

for name, path in input_files.items():
    validate_input_source(path)
    print(f"  Validated: {name} ({path})")

# Load CCA env-side embedding (1809 x 10)
env_embedding = np.load(input_files['env_embedding'])
sample_ids = np.load(input_files['sample_ids'], allow_pickle=True)
print(f"\n  Env embedding shape: {env_embedding.shape}")
print(f"  Samples: {len(sample_ids)}")

# CC1 scores (env-side canonical variate 1)
cc1_scores = env_embedding[:, 0]
print(f"  CC1 range: [{cc1_scores.min():.4f}, {cc1_scores.max():.4f}]")
print(f"  CC1 mean: {cc1_scores.mean():.4f}, std: {cc1_scores.std():.4f}")

# Load GPS coordinates from merged dataset
merged = pd.read_csv(input_files['merged_dataset'],
                     sep='\t', comment='#',
                     usecols=['assembly_id', 'latitude', 'longitude'])
print(f"  Merged dataset rows: {len(merged)}")

# Create a lookup for GPS by assembly_id
gps_lookup = merged.set_index('assembly_id')[['latitude', 'longitude']]

# Match GPS to CCA sample order
lats = np.array([gps_lookup.loc[sid, 'latitude'] for sid in sample_ids])
lons = np.array([gps_lookup.loc[sid, 'longitude'] for sid in sample_ids])
print(f"  GPS matched: {len(lats)} samples")
print(f"  Lat range: [{lats.min():.2f}, {lats.max():.2f}]")
print(f"  Lon range: [{lons.min():.2f}, {lons.max():.2f}]")

# ============================================================================
# Create Robinson projection world map
# ============================================================================
print("\n" + "=" * 70)
print("CREATING FIGURE")
print("=" * 70)

# Use diverging colormap — CC1 is centered at zero
cmap = get_diverging_cmap('ocean_land')

# Symmetric color limits around zero
vmax = max(abs(cc1_scores.min()), abs(cc1_scores.max()))
vmin = -vmax
print(f"  Color limits: [{vmin:.4f}, {vmax:.4f}]")

# Sort by absolute CC1 value so extreme values plot on top
sort_idx = np.argsort(np.abs(cc1_scores))
lats_sorted = lats[sort_idx]
lons_sorted = lons[sort_idx]
cc1_sorted = cc1_scores[sort_idx]

# Figure: single column width (3.5 in) with compact aspect ratio
fig = plt.figure(figsize=(7.0, 3.5))

# Main map axis
ax = fig.add_axes([0.02, 0.08, 0.88, 0.88],
                  projection=ccrs.Robinson())

# Land and ocean features — minimal styling
ax.add_feature(cfeature.LAND, facecolor=(0.92, 0.92, 0.92),
               edgecolor='none', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor=(0.97, 0.97, 0.98),
               edgecolor='none', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.15, edgecolor=(0.5, 0.5, 0.5),
               zorder=1)

# Plot samples
sc = ax.scatter(lons_sorted, lats_sorted,
                c=cc1_sorted, cmap=cmap,
                s=3, alpha=0.85,
                vmin=vmin, vmax=vmax,
                linewidth=0.1, edgecolor=(0.3, 0.3, 0.3),
                transform=ccrs.PlateCarree(),
                zorder=2)

# Gridlines — subtle
gl = ax.gridlines(draw_labels=False, linewidth=0.15,
                  color=(0.7, 0.7, 0.7), alpha=0.5,
                  linestyle='--')

# Title
ax.set_title('CCA CC1 env-side scores (n = 1,809 samples)',
             fontsize=6, fontweight='bold', pad=2)

# Colorbar — compact, on the right
cax = fig.add_axes([0.91, 0.15, 0.015, 0.65])
cbar = plt.colorbar(sc, cax=cax, orientation='vertical')
cbar.set_label('CC1 score', fontsize=6)
cbar.ax.tick_params(labelsize=6, width=0.25, length=2)
cbar.outline.set_linewidth(0.25)

# Add CC1 interpretation annotation
ax.text(0.01, 0.01,
        'CC1 (r = 0.82): bathymetry-temperature gradient',
        transform=ax.transAxes, fontsize=5,
        color=(0.3, 0.3, 0.3), ha='left', va='bottom')

# ============================================================================
# Save outputs
# ============================================================================
print("\n" + "=" * 70)
print("SAVING OUTPUTS")
print("=" * 70)

for fmt in ['pdf', 'svg']:
    filename = SCRIPT_DIR / f"FigureS9_cca_geography_{TIMESTAMP}.{fmt}"
    fig.savefig(filename, format=fmt,
                bbox_inches='tight',
                transparent=True,
                edgecolor='none',
                dpi=300)
    print(f"  Saved: {filename}")

plt.close()

# ============================================================================
# Summary statistics for provenance
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Samples plotted: {len(cc1_scores)}")
print(f"  CC1 score range: [{cc1_scores.min():.4f}, {cc1_scores.max():.4f}]")
print(f"  CC1 score mean: {cc1_scores.mean():.6f}")
print(f"  CC1 score std: {cc1_scores.std():.4f}")
print(f"  Latitude range: [{lats.min():.2f}, {lats.max():.2f}]")
print(f"  Longitude range: [{lons.min():.2f}, {lons.max():.2f}]")
print(f"  Colormap: Earth diverging (ocean-land)")
print(f"  Projection: Robinson")

# Count samples per quadrant
n_positive = (cc1_scores > 0).sum()
n_negative = (cc1_scores < 0).sum()
print(f"  Samples with CC1 > 0: {n_positive} ({n_positive/len(cc1_scores)*100:.1f}%)")
print(f"  Samples with CC1 < 0: {n_negative} ({n_negative/len(cc1_scores)*100:.1f}%)")

# Geographic distribution of extreme CC1 values
top_10pct = np.percentile(np.abs(cc1_scores), 90)
extreme_mask = np.abs(cc1_scores) >= top_10pct
print(f"\n  Top 10% extreme CC1 samples (|CC1| >= {top_10pct:.4f}): n={extreme_mask.sum()}")
print(f"    Lat range: [{lats[extreme_mask].min():.2f}, {lats[extreme_mask].max():.2f}]")
print(f"    Mean lat: {np.abs(lats[extreme_mask]).mean():.2f} (absolute)")

print(f"\n{'=' * 70}")
print(f"TASK 3 COMPLETE — {datetime.now().isoformat()}")
print(f"{'=' * 70}")
