#!/usr/bin/env python3
"""
LA4SR vs AlgaGPT Comparison with Taxonomic Horizon Plots

Creates a sophisticated comparison figure showing:
- Retention differences between LA4SR and AlgaGPT filtering methods
- Taxonomic distributions from RuBisCO HMM results
- Horizon plots for compact visualization of lineage-specific patterns

Provenance:
  Script: /media/drn2/External/TARA-Oceans/03_analyses/compare_la4sr_vs_algagpt_horizon_20260203_120000.py
  Input:  la4sr_vs_algagpt_comparison_20260114_172214.tsv, rubisco_all_samples_20260114.tsv
  Date:   2026-02-03
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FIGURE_PROTOCOL.md settings
# ============================================================================
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
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

# ============================================================================
# SATELLITE-INSPIRED COLOR PALETTE - NASA Blue Marble / MODIS inspired
# ============================================================================
SATELLITE_COLORS = {
    # Ocean colors (deep to shallow)
    'ocean_deep': '#0a1628',
    'ocean_mid': '#1a4a6e',
    'ocean_shallow': '#3d8eb9',
    'ocean_surface': '#7ec8e3',

    # Land colors
    'land_forest': '#1a4d2e',
    'land_grass': '#4a7c4e',
    'land_desert': '#c4a35a',
    'land_ice': '#e8f4f8',

    # SST gradient (cold to hot)
    'sst_cold': '#1e3a5f',
    'sst_cool': '#2d6a8e',
    'sst_warm': '#e07b39',
    'sst_hot': '#b22222',

    # Accent colors
    'sample_point': '#ffd700',
    'arrow': '#2f4f4f',
}

# Taxonomic group colors - ocean/earth themed
TAXA_COLORS = {
    'mamiellophyceae': '#7ec8e3',    # Ocean surface - light blue
    'prasinophyceae': '#3d8eb9',     # Ocean shallow - medium blue
    'pyramimonadales': '#1a4a6e',    # Ocean mid - deep blue
    'chlorellaceae': '#4a7c4e',      # Land grass - green
    'trebouxiophyceae': '#1a4d2e',   # Land forest - dark green
    'scenedesmaceae': '#2d6a8e',     # SST cool - teal
    'pelagophyceae': '#e07b39',      # SST warm - orange
    'bolidophyceae': '#c4a35a',      # Land desert - tan
    'haptophyta': '#b22222',         # SST hot - red
    'cryptophyta': '#ffd700',        # Sample point - gold
}

# Source database colors - satellite themed
SOURCE_COLORS = {
    'MGnify': SATELLITE_COLORS['ocean_mid'],
    'MMETSP': SATELLITE_COLORS['sst_warm'],
    'GenBank': SATELLITE_COLORS['land_grass'],
    'AAC': SATELLITE_COLORS['ocean_surface'],
    'PhycoCosm': SATELLITE_COLORS['land_desert'],
    'RefSeq': SATELLITE_COLORS['sst_hot'],
    'Other': SATELLITE_COLORS['arrow'],
}

# ============================================================================
# Load data
# ============================================================================
base_dir = "/media/drn2/External/TARA-Oceans/03_analyses"

# Load comparison data
comparison_df = pd.read_csv(f"{base_dir}/la4sr_vs_algagpt_comparison_20260114_172214.tsv",
                            sep='\t', comment='#')
print(f"Comparison data: {len(comparison_df)} samples")

# Load RuBisCO taxonomy data
rubisco_df = pd.read_csv(f"{base_dir}/rubisco_hmms/rubisco_all_samples_20260114.tsv",
                         sep='\t')
print(f"RuBisCO data: {len(rubisco_df)} samples")

# ============================================================================
# Normalize sample names for matching
# ============================================================================
def normalize_name(name):
    """Normalize sample name for matching between datasets."""
    name = str(name)
    # Remove various suffixes
    name = re.sub(r'\.aa_algagpt\.tsv$', '', name)
    name = re.sub(r'\.fa\.aa$', '', name)
    name = re.sub(r'\.aa\.fa$', '', name)
    name = re.sub(r'\.aa\.algal\.fa$', '', name)
    name = re.sub(r'\.fa$', '', name)
    name = re.sub(r'\.aa$', '', name)
    # Normalize MMETSP naming
    name = re.sub(r'\.Trinity\.fasta\.transdecoder', '.Trinitysta.transdecoder', name)
    return name

comparison_df['norm_name'] = comparison_df['base_name'].apply(normalize_name)
rubisco_df['norm_name'] = rubisco_df['sample_id'].apply(normalize_name)

# Merge datasets
merged = pd.merge(
    comparison_df,
    rubisco_df,
    on='norm_name',
    how='inner'
)
print(f"Merged samples: {len(merged)}")

# Taxa columns
taxa_cols = ['mamiellophyceae', 'prasinophyceae', 'pyramimonadales', 'chlorellaceae',
             'trebouxiophyceae', 'scenedesmaceae', 'pelagophyceae', 'bolidophyceae',
             'haptophyta', 'cryptophyta']

# Calculate total taxa per sample
merged['total_taxa'] = merged[taxa_cols].sum(axis=1)

# ============================================================================
# Classify source datasets
# ============================================================================
def classify_source(name):
    name = str(name)
    if 'MMETSP' in name:
        return 'MMETSP'
    elif 'MGYA' in name:
        return 'MGnify'
    elif name.startswith('GCA_'):
        return 'GenBank'
    elif name.startswith('GCF_'):
        return 'RefSeq'
    elif '.AAC' in name:
        return 'AAC'
    elif '.PRE' in name:
        return 'PhycoCosm'
    else:
        return 'Other'

merged['source'] = merged['base_name'].apply(classify_source)

# ============================================================================
# Create horizon plot data
# ============================================================================
def create_horizon_data(df, taxa_col, n_bins=100):
    """Create horizon plot data for a taxonomic group."""
    # Filter to samples with this taxon
    has_taxon = df[df[taxa_col] > 0].copy()
    if len(has_taxon) == 0:
        return None, None, 0

    # Sort by difference
    has_taxon = has_taxon.sort_values('difference')

    # Bin the data for horizon visualization
    differences = has_taxon['difference'].values

    return differences, has_taxon, len(has_taxon)

def draw_horizon_plot(ax, differences, title, n_samples, n_bands=3):
    """Draw a horizon plot showing distribution of differences."""
    if differences is None or len(differences) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=5, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return

    # Create histogram for horizon plot
    n_bins = min(50, len(differences))
    hist, bin_edges = np.histogram(differences, bins=n_bins, range=(-30, 70))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize histogram
    hist_norm = hist / hist.max() if hist.max() > 0 else hist

    # Create horizon bands
    band_height = 1.0 / n_bands

    # Color gradients for positive (AlgaGPT > LA4SR) and negative (LA4SR > AlgaGPT)
    pos_colors = [SATELLITE_COLORS['sst_warm'], SATELLITE_COLORS['sst_hot'], '#8B0000']
    neg_colors = [SATELLITE_COLORS['ocean_shallow'], SATELLITE_COLORS['ocean_mid'], SATELLITE_COLORS['ocean_deep']]

    for i in range(n_bands):
        lower = i * band_height
        upper = (i + 1) * band_height

        # Positive differences (AlgaGPT retains more)
        pos_mask = bin_centers >= 0
        pos_vals = np.clip(hist_norm[pos_mask] - lower, 0, band_height)
        if len(pos_vals) > 0 and np.any(pos_vals > 0):
            ax.bar(bin_centers[pos_mask], pos_vals, width=2.0,
                   bottom=0, color=pos_colors[min(i, len(pos_colors)-1)],
                   alpha=0.8 - i*0.15, linewidth=0)

        # Negative differences (LA4SR retains more)
        neg_mask = bin_centers < 0
        neg_vals = np.clip(hist_norm[neg_mask] - lower, 0, band_height)
        if len(neg_vals) > 0 and np.any(neg_vals > 0):
            ax.bar(bin_centers[neg_mask], neg_vals, width=2.0,
                   bottom=0, color=neg_colors[min(i, len(neg_colors)-1)],
                   alpha=0.8 - i*0.15, linewidth=0)

    # Zero line
    ax.axvline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.5)

    # Formatting
    ax.set_xlim(-30, 70)
    ax.set_ylim(0, band_height * 1.1)
    ax.set_yticks([])

    # Title with sample count
    ax.text(-28, band_height * 0.5, f"{title}\n(n={n_samples})",
            fontsize=5, va='center', ha='left', fontweight='bold')

    # Remove spines
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.25)

# ============================================================================
# Create figure
# ============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
fig = plt.figure(figsize=(7.5, 7))

# Grid: 5 rows, 3 columns
gs = gridspec.GridSpec(5, 3, figure=fig,
                       height_ratios=[1.5, 0.5, 0.5, 0.5, 0.8],
                       width_ratios=[1, 1, 1],
                       hspace=0.4, wspace=0.35)

# ============================================================================
# Panel A: Scatter plot - LA4SR vs AlgaGPT retention (top left, 2 cols)
# ============================================================================
ax_scatter = fig.add_subplot(gs[0, :2])

for src in ['MGnify', 'MMETSP', 'GenBank', 'AAC', 'PhycoCosm', 'RefSeq', 'Other']:
    subset = merged[merged['source'] == src]
    if len(subset) > 0:
        ax_scatter.scatter(subset['la4sr_pct'], subset['algagpt_pct'],
                          c=SOURCE_COLORS.get(src, 'gray'), s=8, alpha=0.6,
                          edgecolors='white', linewidth=0.2,
                          label=f"{src} (n={len(subset)})")

# 1:1 line
ax_scatter.plot([0, 100], [0, 100], color=SATELLITE_COLORS['arrow'],
                linestyle='--', linewidth=0.5, alpha=0.7, label='1:1')

# Regression line
slope, intercept, r, p, se = stats.linregress(merged['la4sr_pct'], merged['algagpt_pct'])
x_line = np.array([0, 100])
ax_scatter.plot(x_line, slope * x_line + intercept,
                color=SATELLITE_COLORS['sst_hot'], linewidth=1,
                label=f'Fit: r={r:.2f}')

ax_scatter.set_xlabel('Greedy decoding retention (%)')
ax_scatter.set_ylabel('Top-k sampling retention (%)')
ax_scatter.set_xlim(0, 105)
ax_scatter.set_ylim(0, 105)
ax_scatter.legend(fontsize=4, loc='lower right', frameon=False, ncol=2)
ax_scatter.text(-0.08, 1.02, 'A', transform=ax_scatter.transAxes,
                fontweight='bold', fontsize=8)
ax_scatter.set_title('Decoding Strategy Comparison: Greedy vs Top-k Sampling', fontsize=6, fontweight='bold')

# ============================================================================
# Panel B: Taxa legend and summary (top right)
# ============================================================================
ax_legend = fig.add_subplot(gs[0, 2])
ax_legend.axis('off')

# Draw taxa color legend
y_pos = 0.95
for taxa in taxa_cols:
    taxa_display = taxa.replace('phyceae', '.').replace('ales', '.').capitalize()
    n_samples = (merged[taxa] > 0).sum()

    # Color box
    rect = Rectangle((0.05, y_pos - 0.04), 0.08, 0.06,
                     facecolor=TAXA_COLORS[taxa], edgecolor='none')
    ax_legend.add_patch(rect)

    # Label
    ax_legend.text(0.16, y_pos - 0.01, f"{taxa_display} ({n_samples})",
                  fontsize=5, va='center')
    y_pos -= 0.09

ax_legend.set_xlim(0, 1)
ax_legend.set_ylim(0, 1)
ax_legend.text(0.05, 1.0, 'Taxonomic Groups', fontsize=6, fontweight='bold', va='top')
ax_legend.text(-0.05, 1.08, 'B', transform=ax_legend.transAxes,
               fontweight='bold', fontsize=8)

# ============================================================================
# Panels C-L: Horizon plots for each taxon (rows 1-3, all cols)
# ============================================================================
horizon_axes = []
taxa_display_names = {
    'mamiellophyceae': 'Mamiello.',
    'prasinophyceae': 'Prasino.',
    'pyramimonadales': 'Pyramimon.',
    'chlorellaceae': 'Chlorella.',
    'trebouxiophyceae': 'Treboux.',
    'scenedesmaceae': 'Scenedesm.',
    'pelagophyceae': 'Pelago.',
    'bolidophyceae': 'Bolido.',
    'haptophyta': 'Hapto.',
    'cryptophyta': 'Crypto.',
}

# Create subgrid for horizon plots
gs_horizon = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs[1:4, :],
                                               hspace=0.15, wspace=0.1)

panel_labels = ['C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
for idx, taxa in enumerate(taxa_cols):
    row = idx // 3
    col = idx % 3

    ax = fig.add_subplot(gs_horizon[row, col])

    # Get data for this taxon
    differences, subset_df, n_samples = create_horizon_data(merged, taxa)

    # Draw horizon plot
    draw_horizon_plot(ax, differences, taxa_display_names[taxa], n_samples)

    # Y-axis label for leftmost column
    if col == 0:
        ax.set_ylabel('Count', fontsize=5)

    # Panel label - kitty corner to top-left
    if idx < len(panel_labels):
        ax.text(0.02, 0.96, panel_labels[idx], transform=ax.transAxes,
                fontweight='bold', fontsize=7, ha='left', va='top')

    # X-axis label only on bottom row
    if row == 3:
        ax.set_xlabel('Difference (Top-k - Greedy) %', fontsize=5)
    else:
        ax.set_xticklabels([])

# Fill remaining cell with legend
ax_info = fig.add_subplot(gs_horizon[3, 2])
ax_info.axis('off')

# Horizon interpretation legend
ax_info.text(0.1, 0.9, 'Horizon Interpretation:', fontsize=5, fontweight='bold', va='top')
ax_info.add_patch(Rectangle((0.1, 0.6), 0.15, 0.15,
                             facecolor=SATELLITE_COLORS['sst_warm'], alpha=0.8))
ax_info.text(0.3, 0.68, 'Top-k > Greedy', fontsize=4, va='center')
ax_info.add_patch(Rectangle((0.1, 0.35), 0.15, 0.15,
                             facecolor=SATELLITE_COLORS['ocean_mid'], alpha=0.8))
ax_info.text(0.3, 0.43, 'Greedy > Top-k', fontsize=4, va='center')
ax_info.set_xlim(0, 1)
ax_info.set_ylim(0, 1)

# ============================================================================
# Panel M: Difference vs assembly size (bottom row)
# ============================================================================
ax_size = fig.add_subplot(gs[4, :])

for src in ['MGnify', 'MMETSP', 'GenBank', 'AAC', 'PhycoCosm', 'RefSeq', 'Other']:
    subset = merged[merged['source'] == src]
    if len(subset) > 0:
        ax_size.scatter(subset['sequences_before'] / 1000, subset['difference'],
                       c=SOURCE_COLORS.get(src, 'gray'), s=5, alpha=0.5,
                       edgecolors='none')

ax_size.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
ax_size.set_xlabel('Assembly size (thousands of sequences)')
ax_size.set_ylabel('Difference (Top-k - Greedy) %')
ax_size.set_xscale('log')
ax_size.set_xlim(0.5, 5000)
ax_size.set_ylim(-25, 75)
ax_size.text(0.01, 0.96, 'M', transform=ax_size.transAxes, fontweight='bold', fontsize=8, ha='left', va='top')
ax_size.set_title('Retention Difference vs Assembly Size', fontsize=6, fontweight='bold')

# ============================================================================
# Save figure
# ============================================================================
output_base = f"{base_dir}/la4sr_vs_algagpt_horizon_{timestamp}"

for fmt in ['pdf', 'svg']:
    fig.savefig(f"{output_base}.{fmt}", format=fmt,
                bbox_inches='tight', transparent=True, edgecolor='none', dpi=300)
    print(f"Saved: {output_base}.{fmt}")

plt.close()

# ============================================================================
# Save provenance
# ============================================================================
# Calculate summary statistics for provenance
mean_la4sr = merged['la4sr_pct'].mean()
mean_algagpt = merged['algagpt_pct'].mean()
mean_diff = merged['difference'].mean()
corr = merged['la4sr_pct'].corr(merged['algagpt_pct'])

provenance_file = f"{output_base}_provenance.txt"
with open(provenance_file, 'w') as f:
    f.write(f"# Provenance\n")
    f.write(f"Script: {__file__}\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Input 1: {base_dir}/la4sr_vs_algagpt_comparison_20260114_172214.tsv\n")
    f.write(f"Input 2: {base_dir}/rubisco_hmms/rubisco_all_samples_20260114.tsv\n")
    f.write(f"\n# Statistics\n")
    f.write(f"Total merged samples: {len(merged)}\n")
    f.write(f"LA4SR mean retention: {mean_la4sr:.2f}%\n")
    f.write(f"AlgaGPT mean retention: {mean_algagpt:.2f}%\n")
    f.write(f"Mean difference: {mean_diff:+.2f}%\n")
    f.write(f"Correlation (r): {corr:.4f}\n")
    f.write(f"\n# Taxa sample counts\n")
    for taxa in taxa_cols:
        n = (merged[taxa] > 0).sum()
        f.write(f"{taxa}: {n} samples\n")

print(f"Saved: {provenance_file}")
print("\n=== COMPLETE ===")
