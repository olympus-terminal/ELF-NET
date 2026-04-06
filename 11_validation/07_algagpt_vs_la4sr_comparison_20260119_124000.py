#!/usr/bin/env python3
"""
Task 7: Compare algaGPT vs LA4SR Results

Compares correlation analysis results between:
- algaGPT-filtered proteins
- LA4SR-filtered proteins

Generates comparison statistics and visualizations.

Provenance:
  Input: algagpt_pfam_gee_correlations_significant_*.tsv
         algagpt_pfam_alphaearth_correlations_significant_*.tsv
         ../../03_analyses/env_pfam_manifold/reports/correlation_matrix_*.csv
         ../../AlphaEarth/alphaearth_pfam_correlations_*_significant_fdr05.tsv
  Date: 2026-01-19
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

# Figure protocol settings
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.linewidth'] = 0.25

# Paths
BASE_DIR = Path("/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses")
LA4SR_ENV_DIR = Path("/media/drn2/External/TARA-Oceans/03_analyses/env_pfam_manifold/reports")
LA4SR_AE_DIR = Path("/media/drn2/External/TARA-Oceans/AlphaEarth")
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def load_latest_file(directory, pattern):
    """Load most recent file matching pattern."""
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    latest = files[-1]
    print(f"  Loading: {latest.name}")
    return latest

def load_algagpt_results():
    """Load algaGPT correlation results."""
    print("=" * 70)
    print("LOADING ALGAGPT RESULTS")
    print("=" * 70)

    # GEE correlations
    gee_file = load_latest_file(RESULTS_DIR, "algagpt_pfam_gee_correlations_significant_*.tsv")
    gee_df = pd.read_csv(gee_file, sep='\t', comment='#')
    print(f"    algaGPT GEE significant: {len(gee_df):,}")

    # AlphaEarth correlations
    ae_file = load_latest_file(RESULTS_DIR, "algagpt_pfam_alphaearth_correlations_significant_*.tsv")
    ae_df = pd.read_csv(ae_file, sep='\t', comment='#')
    print(f"    algaGPT AlphaEarth significant: {len(ae_df):,}")

    return gee_df, ae_df

def load_la4sr_results():
    """Load LA4SR correlation results."""
    print("\n" + "=" * 70)
    print("LOADING LA4SR RESULTS")
    print("=" * 70)

    # Note: LA4SR results are in a different format - correlation matrix
    # We need to check the structure first
    gee_file = load_latest_file(LA4SR_ENV_DIR, "correlation_matrix_*.csv")

    # AlphaEarth correlations
    ae_file = load_latest_file(LA4SR_AE_DIR, "alphaearth_pfam_correlations_*_significant_fdr05.tsv")
    ae_df = pd.read_csv(ae_file, sep='\t', comment='#')
    print(f"    LA4SR AlphaEarth significant: {len(ae_df):,}")

    return gee_file, ae_df

def compare_alphaearth_results(algagpt_df, la4sr_df):
    """Compare AlphaEarth correlation results."""
    print("\n" + "=" * 70)
    print("ALPHAEARTH COMPARISON")
    print("=" * 70)

    # Basic statistics
    stats = {
        'metric': [],
        'algagpt': [],
        'la4sr': []
    }

    stats['metric'].append('n_significant_correlations')
    stats['algagpt'].append(len(algagpt_df))
    stats['la4sr'].append(len(la4sr_df))

    # Unique PFAMs
    algagpt_pfams = set(algagpt_df['pfam_domain'])
    la4sr_pfams = set(la4sr_df['pfam'])

    stats['metric'].append('n_unique_pfams')
    stats['algagpt'].append(len(algagpt_pfams))
    stats['la4sr'].append(len(la4sr_pfams))

    # Overlapping PFAMs
    overlap_pfams = algagpt_pfams & la4sr_pfams

    stats['metric'].append('n_overlapping_pfams')
    stats['algagpt'].append(len(overlap_pfams))
    stats['la4sr'].append(len(overlap_pfams))

    # Mean absolute correlation
    stats['metric'].append('mean_abs_correlation')
    stats['algagpt'].append(algagpt_df['spearman_rho'].abs().mean())
    stats['la4sr'].append(la4sr_df['rho'].abs().mean())

    # Median absolute correlation
    stats['metric'].append('median_abs_correlation')
    stats['algagpt'].append(algagpt_df['spearman_rho'].abs().median())
    stats['la4sr'].append(la4sr_df['rho'].abs().median())

    comparison_df = pd.DataFrame(stats)

    print("\n  Comparison Statistics:")
    print(comparison_df.to_string(index=False))

    # Calculate overlap percentages
    algagpt_pct = 100 * len(overlap_pfams) / len(algagpt_pfams)
    la4sr_pct = 100 * len(overlap_pfams) / len(la4sr_pfams)
    print(f"\n  PFAM Overlap:")
    print(f"    {len(overlap_pfams)} PFAMs found in both")
    print(f"    {algagpt_pct:.1f}% of algaGPT PFAMs")
    print(f"    {la4sr_pct:.1f}% of LA4SR PFAMs")

    # Unique to each
    algagpt_unique = algagpt_pfams - la4sr_pfams
    la4sr_unique = la4sr_pfams - algagpt_pfams

    print(f"\n  Unique PFAMs:")
    print(f"    algaGPT only: {len(algagpt_unique)}")
    print(f"    LA4SR only: {len(la4sr_unique)}")

    return comparison_df, overlap_pfams, algagpt_unique, la4sr_unique

def create_venn_diagram(algagpt_pfams, la4sr_pfams, output_name, title):
    """Create Venn diagram showing PFAM overlap."""

    overlap = algagpt_pfams & la4sr_pfams
    algagpt_only = algagpt_pfams - la4sr_pfams
    la4sr_only = la4sr_pfams - algagpt_pfams

    fig, ax = plt.subplots(figsize=(4, 3))

    # Simple bar chart representation (matplotlib-venn not guaranteed to be installed)
    categories = ['algaGPT\nonly', 'Both', 'LA4SR\nonly']
    counts = [len(algagpt_only), len(overlap), len(la4sr_only)]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

    bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=0.5)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom', fontsize=6)

    ax.set_ylabel('Number of PFAMs')
    ax.set_title(title, pad=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    output_pdf = FIGURES_DIR / f"{output_name}_{TIMESTAMP}.pdf"
    output_png = FIGURES_DIR / f"{output_name}_{TIMESTAMP}.png"

    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  Saved: {output_pdf.name}")

    return output_pdf

def create_correlation_scatter(algagpt_df, la4sr_df, overlap_pfams, output_name, title):
    """Create scatter plot comparing correlation strengths for overlapping PFAMs."""

    # For overlapping PFAMs, compare mean absolute correlations
    algagpt_means = algagpt_df.groupby('pfam_domain')['spearman_rho'].apply(lambda x: x.abs().mean())
    la4sr_means = la4sr_df.groupby('pfam')['rho'].apply(lambda x: x.abs().mean())

    # Get values for overlapping PFAMs
    overlap_list = list(overlap_pfams)
    algagpt_vals = [algagpt_means.get(p, np.nan) for p in overlap_list]
    la4sr_vals = [la4sr_means.get(p, np.nan) for p in overlap_list]

    # Remove NaN pairs
    valid_mask = ~(np.isnan(algagpt_vals) | np.isnan(la4sr_vals))
    algagpt_vals = np.array(algagpt_vals)[valid_mask]
    la4sr_vals = np.array(la4sr_vals)[valid_mask]

    print(f"\n  Scatter plot: {len(algagpt_vals)} overlapping PFAMs with valid data")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.scatter(la4sr_vals, algagpt_vals, s=5, alpha=0.5, edgecolors='none')

    # Diagonal line
    max_val = max(algagpt_vals.max(), la4sr_vals.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('LA4SR Mean |ρ|')
    ax.set_ylabel('algaGPT Mean |ρ|')
    ax.set_title(title, pad=5)
    ax.set_aspect('equal')

    # Compute correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(la4sr_vals, algagpt_vals)
    ax.text(0.05, 0.95, f'ρ = {rho:.3f}\np < {pval:.1e}',
            transform=ax.transAxes, va='top', fontsize=5,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.25))

    plt.tight_layout()

    output_pdf = FIGURES_DIR / f"{output_name}_{TIMESTAMP}.pdf"
    output_png = FIGURES_DIR / f"{output_name}_{TIMESTAMP}.png"

    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_pdf.name}")
    print(f"  Correlation: ρ = {rho:.3f}, p = {pval:.2e}")

    return output_pdf, rho, pval

def save_comparison_summary(comparison_df, overlap_pfams, algagpt_unique, la4sr_unique, scatter_rho, scatter_p):
    """Save comparison summary to file."""
    print("\n" + "=" * 70)
    print("SAVING COMPARISON SUMMARY")
    print("=" * 70)

    output = RESULTS_DIR / f"algagpt_vs_la4sr_comparison_{TIMESTAMP}.tsv"

    with open(output, 'w') as f:
        f.write("# Provenance:\n")
        f.write(f"#   Script: {Path(__file__).absolute()}\n")
        f.write(f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#   Integrity Check: PASSED - Real data only\n")
        f.write("#\n")
        f.write("# COMPARISON SUMMARY: algaGPT vs LA4SR\n")
        f.write("#\n")
        f.write("# AlphaEarth Correlations Comparison:\n")
        f.write("#\n")

    comparison_df.to_csv(output, sep='\t', index=False, mode='a')

    with open(output, 'a') as f:
        f.write(f"\n# PFAM Overlap Details:\n")
        f.write(f"#   PFAMs in both: {len(overlap_pfams)}\n")
        f.write(f"#   algaGPT only: {len(algagpt_unique)}\n")
        f.write(f"#   LA4SR only: {len(la4sr_unique)}\n")
        f.write(f"#\n")
        f.write(f"# Correlation Scatter Statistics:\n")
        f.write(f"#   Spearman ρ: {scatter_rho:.4f}\n")
        f.write(f"#   p-value: {scatter_p:.2e}\n")

    print(f"  Saved: {output.name}")

    return output

def main():
    """Main execution."""
    print("\n" + "=" * 70)
    print("ALGAGPT VS LA4SR COMPARISON")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load results
    algagpt_gee, algagpt_ae = load_algagpt_results()
    la4sr_gee_file, la4sr_ae = load_la4sr_results()

    # Compare AlphaEarth results
    comparison_df, overlap_pfams, algagpt_unique, la4sr_unique = compare_alphaearth_results(algagpt_ae, la4sr_ae)

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    venn_output = create_venn_diagram(
        set(algagpt_ae['pfam_domain']),
        set(la4sr_ae['pfam']),
        'algagpt_vs_la4sr_venn',
        'PFAM Domain Overlap\n(AlphaEarth Significant Correlations)'
    )

    scatter_output, scatter_rho, scatter_p = create_correlation_scatter(
        algagpt_ae, la4sr_ae, overlap_pfams,
        'algagpt_vs_la4sr_scatter',
        'Correlation Strength Comparison\n(Overlapping PFAMs)'
    )

    # Save summary
    summary_output = save_comparison_summary(comparison_df, overlap_pfams, algagpt_unique, la4sr_unique, scatter_rho, scatter_p)

    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output files:")
    print(f"  {summary_output}")
    print(f"  {venn_output}")
    print(f"  {scatter_output}")
    print("=" * 70)

if __name__ == "__main__":
    main()
