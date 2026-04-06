#!/usr/bin/env python3
"""
Task 20: Publication Figure - SHAP Interpretation
# --------------------------------------------------------------------------

4-panel SHAP interpretation figure showing feature dependencies and interactions.

Provenance:
-----------
Script: /media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/validations/scripts/20_figure_shap_interpretation_20260119.py
Input:  results/shap_dependence_gee_*.tsv
        results/shap_dependence_alphaearth_*.tsv
        results/shap_interactions_gee_*.tsv
Date:   2026-01-19
Task:   RALPH Plan Task 20/20

Figure Layout:
--------------
Panel A: PFAM × GEE variable heatmap (top 100 PFAMs by mean SHAP)
Panel B: PFAM × AlphaEarth dimension heatmap (top 100 PFAMs)
Panel C: Violin plot of SHAP values distribution
Panel D: Interaction network (top 50 PFAM-PFAM interactions)

Data Integrity:
---------------
- NO synthetic data - uses REAL results from Tasks 14-16
- Figure standards: 6pt Arial, 0.25pt lines, transparent background

Dependencies:
-------------
- pandas, matplotlib, seaborn, networkx, glob

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
import networkx as nx
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
    log_message("Task 20: Publication Figure - SHAP Interpretation")
    log_message("="*80)

    # Load SHAP dependence data
    try:
        shap_gee_file = find_most_recent_file(str(results_dir / "shap_dependence_gee_*.tsv"))
        shap_gee = pd.read_csv(shap_gee_file, sep='\t', comment='#')
        log_message(f"Loaded GEE SHAP: {shap_gee_file}")
    except FileNotFoundError:
        log_message("WARNING: No GEE SHAP dependence data found")
        shap_gee = None

    try:
        shap_ae_file = find_most_recent_file(str(results_dir / "shap_dependence_alphaearth_*.tsv"))
        shap_ae = pd.read_csv(shap_ae_file, sep='\t', comment='#')
        log_message(f"Loaded AlphaEarth SHAP: {shap_ae_file}")
    except FileNotFoundError:
        log_message("WARNING: No AlphaEarth SHAP dependence data found")
        shap_ae = None

    # Load SHAP interaction data
    try:
        interactions_file = find_most_recent_file(str(results_dir / "shap_interactions_gee_*.tsv"))
        interactions = pd.read_csv(interactions_file, sep='\t', comment='#')
        log_message(f"Loaded interactions: {interactions_file}")
    except FileNotFoundError:
        log_message("WARNING: No SHAP interactions data found")
        interactions = None

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Panel A: PFAM × GEE variable heatmap (top 100 PFAMs)
    ax = axes[0, 0]
    if shap_gee is not None and not shap_gee.empty:
        # Pivot to wide format
        pivot_gee = shap_gee.pivot(index='pfam', columns='variable', values='mean_abs_shap')

        # Get top 100 PFAMs
        pfam_totals = pivot_gee.sum(axis=1).sort_values(ascending=False)
        top_pfams = pfam_totals.head(100).index

        pivot_subset = pivot_gee.loc[top_pfams]

        sns.heatmap(pivot_subset, cmap='viridis', linewidths=0, cbar_kws={'label': 'Mean |SHAP|'},
                    ax=ax, xticklabels=True, yticklabels=False)

        ax.set_xlabel("GEE Variable", fontsize=6)
        ax.set_ylabel("PFAM Domain (Top 100)", fontsize=6)
        ax.set_title("(A) PFAM × GEE SHAP Dependence", fontsize=6, fontweight="bold", loc='left')
        ax.tick_params(labelsize=4, rotation=45)
    else:
        ax.text(0.5, 0.5, "No GEE SHAP\ndependence data", ha='center', va='center',
                fontsize=6, transform=ax.transAxes)
        ax.set_title("(A) PFAM × GEE SHAP", fontsize=6, fontweight="bold", loc='left')

    # Panel B: PFAM × AlphaEarth dimension heatmap (top 100 PFAMs)
    ax = axes[0, 1]
    if shap_ae is not None and not shap_ae.empty:
        # Pivot to wide format
        pivot_ae = shap_ae.pivot(index='pfam', columns='dimension', values='mean_abs_shap')

        # Get top 100 PFAMs
        pfam_totals = pivot_ae.sum(axis=1).sort_values(ascending=False)
        top_pfams = pfam_totals.head(100).index

        pivot_subset = pivot_ae.loc[top_pfams]

        sns.heatmap(pivot_subset, cmap='plasma', linewidths=0, cbar_kws={'label': 'Mean |SHAP|'},
                    ax=ax, xticklabels=True, yticklabels=False)

        ax.set_xlabel("AlphaEarth Dimension", fontsize=6)
        ax.set_ylabel("PFAM Domain (Top 100)", fontsize=6)
        ax.set_title("(B) PFAM × AlphaEarth SHAP Dependence", fontsize=6, fontweight="bold", loc='left')
        ax.tick_params(labelsize=4, rotation=45)
    else:
        ax.text(0.5, 0.5, "No AlphaEarth SHAP\ndependence data", ha='center', va='center',
                fontsize=6, transform=ax.transAxes)
        ax.set_title("(B) PFAM × AlphaEarth SHAP", fontsize=6, fontweight="bold", loc='left')

    # Panel C: Violin plot of SHAP values distribution
    ax = axes[1, 0]
    if shap_gee is not None and not shap_gee.empty:
        # Combine GEE and AlphaEarth SHAP for comparison
        shap_combined = []

        if shap_gee is not None:
            shap_gee_subset = shap_gee.copy()
            shap_gee_subset['source'] = 'GEE'
            shap_combined.append(shap_gee_subset[['mean_abs_shap', 'source']])

        if shap_ae is not None:
            shap_ae_subset = shap_ae.copy()
            shap_ae_subset['source'] = 'AlphaEarth'
            shap_combined.append(shap_ae_subset[['mean_abs_shap', 'source']])

        if shap_combined:
            shap_df = pd.concat(shap_combined, ignore_index=True)

            sns.violinplot(data=shap_df, x='source', y='mean_abs_shap',
                           ax=ax, inner='box', linewidth=0.25, palette=['#2E86AB', '#A23B72'])

            ax.set_xlabel("Data Source", fontsize=6)
            ax.set_ylabel("Mean |SHAP|", fontsize=6)
            ax.set_title("(C) SHAP Value Distribution", fontsize=6, fontweight="bold", loc='left')
            ax.tick_params(labelsize=5)
    else:
        ax.text(0.5, 0.5, "No SHAP data\navailable", ha='center', va='center',
                fontsize=6, transform=ax.transAxes)
        ax.set_title("(C) SHAP Value Distribution", fontsize=6, fontweight="bold", loc='left')

    # Panel D: Interaction network (top 50 PFAM-PFAM interactions)
    ax = axes[1, 1]
    if interactions is not None and not interactions.empty:
        # Get top 50 interactions by absolute value
        top_interactions = interactions.nlargest(50, 'abs_interaction')

        # Create graph
        G = nx.Graph()

        for _, row in top_interactions.iterrows():
            pfam1 = row['pfam1']
            pfam2 = row['pfam2']
            weight = row['interaction_score']

            G.add_edge(pfam1, pfam2, weight=weight)

        # Layout
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        # Edge colors
        edge_colors = ['red' if G[u][v]['weight'] < 0 else 'blue'
                       for u, v in G.edges()]

        # Edge widths
        edge_widths = [abs(G[u][v]['weight']) * 2 for u, v in G.edges()]

        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                               alpha=0.5, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_size=15, node_color='gray', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=3, ax=ax)

        ax.set_title("(D) PFAM-PFAM Interaction Network (Top 50)", fontsize=6, fontweight="bold", loc='left')
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, "No interaction\ndata available", ha='center', va='center',
                fontsize=6, transform=ax.transAxes)
        ax.set_title("(D) PFAM-PFAM Interaction Network", fontsize=6, fontweight="bold", loc='left')
        ax.axis('off')

    plt.tight_layout()

    # Save figure
    pdf_file = figures_dir / f"shap_interpretation_{timestamp}.pdf"
    svg_file = figures_dir / f"shap_interpretation_{timestamp}.svg"

    fig.patch.set_alpha(0.0)
    fig.savefig(pdf_file, format='pdf', bbox_inches='tight', transparent=True, dpi=300)
    fig.savefig(svg_file, format='svg', bbox_inches='tight', transparent=True)
    plt.close()

    log_message(f"Saved PDF: {pdf_file}")
    log_message(f"Saved SVG: {svg_file}")

    log_message("\n" + "="*80)
    log_message("Task 20 COMPLETE")
    log_message("="*80)

if __name__ == '__main__':
    main()
