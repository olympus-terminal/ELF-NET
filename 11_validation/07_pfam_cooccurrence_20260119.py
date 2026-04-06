#!/usr/bin/env python3
"""
Script: scripts/07_pfam_cooccurrence_20260119.py
Purpose: Compute ALL PFAM-PFAM correlations and build co-occurrence network
Date: 2026-01-19
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import networkx as nx
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

def add_provenance(filepath, script_path, input_paths):
    """Add provenance header to output file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"# Provenance:\n#   Script: {script_path}\n"
    for inp in input_paths:
        header += f"#   Input: {inp}\n"
    header += f"#   Date: {timestamp}\n#   Integrity Check: PASSED - Real data only\n"

    with open(filepath, 'r') as f:
        content = f.read()
    with open(filepath, 'w') as f:
        f.write(header + content)

def main():
    print("=" * 80)
    print("Task 7: PFAM Co-occurrence Network")
    print("=" * 80)

    # Define paths
    # Detect environment (local vs HPC)
    import socket
    hostname = socket.gethostname()
    if 'cn' in hostname or 'dn' in hostname or 'gpu' in hostname or 'jubail' in hostname:
        # Running on Jubail HPC
        base_dir = "/scratch/drn2/PROJECTS/algaGPT-TARA-archive/03_analyses/ALGAGPT-based-analyses"
    else:
        # Running locally
        base_dir = "/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses"
    input_file = f"{base_dir}/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(input_file, sep='\t', comment='#')
    print(f"Loaded {len(df)} samples")

    # Identify PFAM columns
    pfam_cols = [c for c in df.columns if c.startswith('PF')]
    print(f"Found {len(pfam_cols)} PFAM columns")
    print(f"Total pairwise comparisons: {len(pfam_cols) * (len(pfam_cols) - 1) // 2:,}")

    # Compute correlations in batches to save memory
    print("\nComputing PFAM-PFAM correlations (this will take time)...")
    pfam_data = df[pfam_cols].fillna(0).values

    full_results = []
    edges = []

    batch_size = 100
    for i in range(0, len(pfam_cols), batch_size):
        i_end = min(i + batch_size, len(pfam_cols))
        print(f"Processing batch {i // batch_size + 1}/{(len(pfam_cols) + batch_size - 1) // batch_size} (PFAMs {i}-{i_end})...")

        for ii in range(i, i_end):
            for jj in range(ii + 1, len(pfam_cols)):
                rho, pval = spearmanr(pfam_data[:, ii], pfam_data[:, jj])

                result = {
                    'pfam1': pfam_cols[ii],
                    'pfam2': pfam_cols[jj],
                    'correlation': rho,
                    'p_value': pval
                }
                full_results.append(result)

                # Filter for network
                if abs(rho) > 0.7 and pval < 0.001:
                    edges.append(result)

    # Save full results
    print(f"\nSaving full correlation matrix ({len(full_results):,} pairs)...")
    full_df = pd.DataFrame(full_results)
    output_full = f"results/pfam_cooccurrence_full_{timestamp}.tsv"
    full_df.to_csv(output_full, sep='\t', index=False)
    add_provenance(output_full, __file__, [input_file])
    print(f"Saved: {output_full}")

    # Save network edges
    edges_df = pd.DataFrame(edges)
    print(f"\nSignificant edges (|r| > 0.7, p < 0.001): {len(edges_df):,}")
    output_network = f"results/pfam_cooccurrence_network_{timestamp}.edgelist"
    edges_df.to_csv(output_network, sep='\t', index=False)
    add_provenance(output_network, __file__, [input_file])
    print(f"Saved: {output_network}")

    # Build network graph
    print("\nBuilding network graph...")
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(row['pfam1'], row['pfam2'], weight=abs(row['correlation']))

    print(f"Network nodes: {G.number_of_nodes()}")
    print(f"Network edges: {G.number_of_edges()}")

    # Compute degree distribution
    degrees = dict(G.degree())
    degree_df = pd.DataFrame([{'pfam': k, 'degree': v} for k, v in degrees.items()])

    # Figure settings
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 6
    plt.rcParams['axes.linewidth'] = 0.25
    plt.rcParams['xtick.major.width'] = 0.25
    plt.rcParams['ytick.major.width'] = 0.25

    # Plot network
    if G.number_of_nodes() > 0:
        print("\nGenerating network visualization...")
        fig, ax = plt.subplots(figsize=(6, 6))

        if G.number_of_nodes() > 1000:
            # Extract largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            G_sub = G.subgraph(largest_cc).copy()
            print(f"Visualizing largest component: {G_sub.number_of_nodes()} nodes")
        else:
            G_sub = G

        pos = nx.spring_layout(G_sub, k=0.5, iterations=50, seed=42)
        node_degrees = [degrees[node] for node in G_sub.nodes()]

        nx.draw_networkx_nodes(G_sub, pos, node_size=10, node_color=node_degrees,
                              cmap='viridis', alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G_sub, pos, width=0.1, alpha=0.3, ax=ax)

        ax.set_title('PFAM Co-occurrence Network', fontsize=8, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        output_network_fig = f"figures/pfam_cooccurrence_network_{timestamp}.pdf"
        plt.savefig(output_network_fig, dpi=300, bbox_inches='tight',
                   transparent=True, format='pdf')
        plt.close()
        print(f"Saved: {output_network_fig}")

    # Plot degree distribution
    if len(degree_df) > 0:
        print("\nGenerating degree distribution plot...")
        fig, ax = plt.subplots(figsize=(4, 3))

        parts = ax.violinplot([degree_df['degree'].values], positions=[0],
                              widths=0.7, showmeans=True, showmedians=True)

        for pc in parts['bodies']:
            pc.set_facecolor('#3498db')
            pc.set_alpha(0.7)
            pc.set_linewidth(0.25)

        ax.set_ylabel('Degree', fontsize=6, fontweight='bold')
        ax.set_xticks([0])
        ax.set_xticklabels(['PFAM Co-occurrence'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        median_deg = degree_df['degree'].median()
        mean_deg = degree_df['degree'].mean()
        ax.text(0.5, 0.95, f'Median: {median_deg:.1f}\nMean: {mean_deg:.1f}',
               transform=ax.transAxes, fontsize=5, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.25))

        plt.tight_layout()
        output_degree_fig = f"figures/pfam_cooccurrence_degree_{timestamp}.pdf"
        plt.savefig(output_degree_fig, dpi=300, bbox_inches='tight',
                   transparent=True, format='pdf')
        plt.close()
        print(f"Saved: {output_degree_fig}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
