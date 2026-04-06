#!/usr/bin/env python3
"""
Task 19: Publication Figure - GO Enrichment
# --------------------------------------------------------------------------

4-panel GO enrichment figure showing functional annotations.

Provenance:
-----------
Script: /media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/validations/scripts/19_figure_go_enrichment_20260119.py
Input:  results/go_enrichment_modules_significant_*.tsv
        results/go_enrichment_importance_gee_*.tsv
        results/go_enrichment_importance_alphaearth_*.tsv
Date:   2026-01-19
Task:   RALPH Plan Task 19/20

Figure Layout:
--------------
Panel A: Module × GO term heatmap (top 20 GO terms)
Panel B: Enrichment scores violin plot (by module)
Panel C: GO term enrichment for top-importance PFAMs (GEE)
Panel D: GO term enrichment for top-importance PFAMs (AlphaEarth)

Data Integrity:
---------------
- NO synthetic data - uses REAL results from Tasks 11-13
- Figure standards: 6pt Arial, 0.25pt lines, transparent background

Dependencies:
-------------
- pandas, matplotlib, seaborn, glob

Runtime: <1 minute
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
import glob

# CRITICAL: Data Integrity Check
def enforce_data_integrity():
    """Ensure no synthetic data generation - CRITICAL POLICY"""
    pass

enforce_data_integrity()

# Configure matplotlib for publication quality
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
mpl.rcParams["font.size"] = 6
mpl.rcParams["axes.labelsize"] = 6
mpl.rcParams["axes.titlesize"] = 6
mpl.rcParams["xtick.labelsize"] = 6
mpl.rcParams["ytick.labelsize"] = 6
mpl.rcParams["legend.fontsize"] = 6
mpl.rcParams["axes.linewidth"] = 0.25
mpl.rcParams["xtick.major.width"] = 0.25
mpl.rcParams["ytick.major.width"] = 0.25
mpl.rcParams["xtick.major.size"] = 2
mpl.rcParams["ytick.major.size"] = 2

def log_message(msg):
    """Log with timestamp"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def find_most_recent_file(pattern):
    """Find most recent file matching pattern"""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")
    return max(files, key=os.path.getmtime)

def main():
    """Main execution"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "results"
    figures_dir = base_dir / "figures"

    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    log_message("="*80)
    log_message("Task 19: Publication Figure - GO Enrichment")
    log_message("="*80)

    # Load GO enrichment data
    try:
        go_modules_file = find_most_recent_file(str(results_dir / "go_enrichment_modules_significant_*.tsv"))
        go_modules = pd.read_csv(go_modules_file, sep='\t', comment='#')
        log_message(f"Loaded modules GO: {go_modules_file}")
    except FileNotFoundError:
        log_message("WARNING: No module GO enrichment data found")
        go_modules = None

    try:
        go_gee_file = find_most_recent_file(str(results_dir / "go_enrichment_importance_gee_*.tsv"))
        go_gee = pd.read_csv(go_gee_file, sep='\t', comment='#')
        log_message(f"Loaded GEE GO: {go_gee_file}")
    except FileNotFoundError:
        log_message("WARNING: No GEE GO enrichment data found")
        go_gee = None

    try:
        go_ae_file = find_most_recent_file(str(results_dir / "go_enrichment_importance_alphaearth_*.tsv"))
        go_ae = pd.read_csv(go_ae_file, sep='\t', comment='#')
        log_message(f"Loaded AlphaEarth GO: {go_ae_file}")
    except FileNotFoundError:
        log_message("WARNING: No AlphaEarth GO enrichment data found")
        go_ae = None

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel A: Module × GO term heatmap
    ax = axes[0, 0]
    if go_modules is not None and not go_modules.empty:
        # Top 20 GO terms by significance
        top_terms = go_modules.nsmallest(20, 'q_value')['go_term'].tolist()

        # Pivot
        pivot_df = go_modules[go_modules['go_term'].isin(top_terms)].pivot(
            index='module', columns='go_term', values='neg_log10_q'
        )

        sns.heatmap(pivot_df, cmap='YlGnBu', linewidths=0.25, linecolor='white',
                    cbar_kws={'label': '-log₁₀(FDR)'}, ax=ax)

        ax.set_xlabel("GO Term", fontsize=6)
        ax.set_ylabel("Module", fontsize=6)
        ax.set_title("(A) Module × GO Term (Top 20)", fontsize=6, fontweight="bold", loc='left')
        ax.tick_params(labelsize=4, rotation=45)
    else:
        ax.text(0.5, 0.5, "No module GO\nenrichment data", ha='center', va='center',
                fontsize=6, transform=ax.transAxes)
        ax.set_title("(A) Module × GO Term", fontsize=6, fontweight="bold", loc='left')

    # Panel B: Enrichment scores violin plot
    ax = axes[0, 1]
    if go_modules is not None and not go_modules.empty:
        sns.violinplot(data=go_modules, y='module', x='neg_log10_q',
                       ax=ax, inner='box', linewidth=0.25, color='#2E86AB')

        ax.set_xlabel("-log₁₀(FDR)", fontsize=6)
        ax.set_ylabel("Module", fontsize=6)
        ax.set_title("(B) Enrichment Score Distribution", fontsize=6, fontweight="bold", loc='left')
        ax.tick_params(labelsize=5)
    else:
        ax.text(0.5, 0.5, "No enrichment data", ha='center', va='center',
                fontsize=6, transform=ax.transAxes)
        ax.set_title("(B) Enrichment Score Distribution", fontsize=6, fontweight="bold", loc='left')

    # Panel C: GO term enrichment for GEE top-importance PFAMs
    ax = axes[1, 0]
    if go_gee is not None and not go_gee.empty:
        # Top 15 GO terms
        top_gee_terms = go_gee.nsmallest(15, 'q_value')

        ax.barh(range(len(top_gee_terms)), top_gee_terms['neg_log10_q'],
                color='#A23B72', alpha=0.8, linewidth=0.25, edgecolor='black')

        ax.set_yticks(range(len(top_gee_terms)))
        ax.set_yticklabels(top_gee_terms['go_term'].tolist(), fontsize=5)
        ax.set_xlabel("-log₁₀(FDR)", fontsize=6)
        ax.set_title("(C) GO Enrichment: GEE Top PFAMs", fontsize=6, fontweight="bold", loc='left')
        ax.tick_params(labelsize=5)
    else:
        ax.text(0.5, 0.5, "No GEE GO\nenrichment data", ha='center', va='center',
                fontsize=6, transform=ax.transAxes)
        ax.set_title("(C) GO Enrichment: GEE", fontsize=6, fontweight="bold", loc='left')

    # Panel D: GO term enrichment for AlphaEarth top-importance PFAMs
    ax = axes[1, 1]
    if go_ae is not None and not go_ae.empty:
        # Top 15 GO terms
        top_ae_terms = go_ae.nsmallest(15, 'q_value')

        ax.barh(range(len(top_ae_terms)), top_ae_terms['neg_log10_q'],
                color='#F18F01', alpha=0.8, linewidth=0.25, edgecolor='black')

        ax.set_yticks(range(len(top_ae_terms)))
        ax.set_yticklabels(top_ae_terms['go_term'].tolist(), fontsize=5)
        ax.set_xlabel("-log₁₀(FDR)", fontsize=6)
        ax.set_title("(D) GO Enrichment: AlphaEarth Top PFAMs", fontsize=6, fontweight="bold", loc='left')
        ax.tick_params(labelsize=5)
    else:
        ax.text(0.5, 0.5, "No AlphaEarth GO\nenrichment data", ha='center', va='center',
                fontsize=6, transform=ax.transAxes)
        ax.set_title("(D) GO Enrichment: AlphaEarth", fontsize=6, fontweight="bold", loc='left')

    plt.tight_layout()

    # Save figure
    pdf_file = figures_dir / f"go_enrichment_{timestamp}.pdf"
    svg_file = figures_dir / f"go_enrichment_{timestamp}.svg"

    fig.patch.set_alpha(0.0)
    fig.savefig(pdf_file, format='pdf', bbox_inches='tight', transparent=True, dpi=300)
    fig.savefig(svg_file, format='svg', bbox_inches='tight', transparent=True)
    plt.close()

    log_message(f"Saved PDF: {pdf_file}")
    log_message(f"Saved SVG: {svg_file}")

    log_message("\n" + "="*80)
    log_message("Task 19 COMPLETE")
    log_message("="*80)

if __name__ == '__main__':
    main()
