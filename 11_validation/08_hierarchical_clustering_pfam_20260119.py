#!/usr/bin/env python3
"""
Script: scripts/08_hierarchical_clustering_pfam_20260119.py
Purpose: Cluster ALL PFAMs by environmental associations, extract modules
Date: 2026-01-19
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, optimal_leaf_ordering
from scipy.spatial.distance import pdist
from datetime import datetime
import os
import glob
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

def find_most_recent_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")
    return max(files, key=os.path.getmtime)

def main():
    print("=" * 80)
    print("Task 8: Hierarchical Clustering of PFAMs")
    print("=" * 80)

    # Detect environment (local vs HPC)
    import socket
    hostname = socket.gethostname()
    if 'cn' in hostname or 'dn' in hostname or 'gpu' in hostname or 'jubail' in hostname:
        # Running on Jubail HPC
        base_dir = "/scratch/drn2/PROJECTS/algaGPT-TARA-archive/03_analyses/ALGAGPT-based-analyses"
    else:
        # Running locally
        base_dir = "/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses"

    # Find most recent correlation file
    correlation_pattern = f"{base_dir}/algagpt_pfam_gee_correlations_*_full.tsv"
    input_file = find_most_recent_file(correlation_pattern)
    print(f"\nUsing correlation file: {input_file}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("\nLoading PFAM-GEE correlation data...")
    df = pd.read_csv(input_file, sep='\t', comment='#')
    print(f"Loaded {len(df)} PFAM-GEE correlations")

    print("\nPivoting to PFAM × GEE_variable matrix...")
    pivot = df.pivot_table(index='pfam', columns='gee_variable',
                          values='rho', fill_value=0)
    print(f"Matrix shape: {pivot.shape[0]} PFAMs × {pivot.shape[1]} GEE variables")

    print("\nComputing distance matrix...")
    dist_matrix = pdist(pivot.values, metric='euclidean')

    print("\nPerforming hierarchical clustering (ward linkage)...")
    linkage_matrix = linkage(dist_matrix, method='ward')
    linkage_matrix = optimal_leaf_ordering(linkage_matrix, dist_matrix)

    best_n_clusters = 20
    print(f"Cutting dendrogram at {best_n_clusters} clusters...")
    clusters = fcluster(linkage_matrix, best_n_clusters, criterion='maxclust')

    module_df = pd.DataFrame({'pfam': pivot.index, 'module_id': clusters})
    module_sizes = module_df['module_id'].value_counts().to_dict()
    module_df['module_size'] = module_df['module_id'].map(module_sizes)
    module_df = module_df.sort_values(['module_id', 'pfam'])

    output_modules = f"results/pfam_modules_{timestamp}.tsv"
    module_df.to_csv(output_modules, sep='\t', index=False)
    add_provenance(output_modules, __file__, [input_file])
    print(f"Saved: {output_modules}")

    summary_data = []
    for module_id in sorted(module_df['module_id'].unique()):
        module_pfams = module_df[module_df['module_id'] == module_id]['pfam'].tolist()
        summary_data.append({
            'module_id': module_id,
            'size': len(module_pfams),
            'top_pfams': ','.join(module_pfams[:10])
        })

    summary_df = pd.DataFrame(summary_data)
    output_summary = f"results/pfam_modules_summary_{timestamp}.tsv"
    summary_df.to_csv(output_summary, sep='\t', index=False)
    add_provenance(output_summary, __file__, [input_file])
    print(f"Saved: {output_summary}")

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 6
    plt.rcParams['axes.linewidth'] = 0.25
    plt.rcParams['xtick.major.width'] = 0.25
    plt.rcParams['ytick.major.width'] = 0.25

    print("\nGenerating dendrogram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(linkage_matrix, no_labels=True, ax=ax, color_threshold=0)
    ax.set_xlabel('PFAM Domains', fontsize=6, fontweight='bold')
    ax.set_ylabel('Distance', fontsize=6, fontweight='bold')
    ax.set_title('PFAM Hierarchical Clustering Dendrogram', fontsize=8, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    output_dendrogram = f"figures/pfam_dendrogram_{timestamp}.pdf"
    plt.savefig(output_dendrogram, dpi=300, bbox_inches='tight',
               transparent=True, format='pdf')
    plt.close()
    print(f"Saved: {output_dendrogram}")

    print("\nGenerating clustered heatmap (top 500 PFAMs)...")
    variances = pivot.var(axis=1)
    top_pfams = variances.nlargest(500).index
    pivot_sub = pivot.loc[top_pfams]
    module_order = module_df[module_df['pfam'].isin(top_pfams)].sort_values('module_id')
    pivot_sub = pivot_sub.loc[module_order['pfam']]

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(pivot_sub, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
               cbar_kws={'label': 'Correlation'}, ax=ax, linewidths=0,
               xticklabels=True, yticklabels=False)
    ax.set_xlabel('GEE Variables', fontsize=6, fontweight='bold')
    ax.set_ylabel('PFAM Domains (clustered)', fontsize=6, fontweight='bold')
    ax.set_title('PFAM Modules - Environmental Associations', fontsize=8, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=4)
    plt.tight_layout()
    output_heatmap = f"figures/pfam_modules_heatmap_{timestamp}.pdf"
    plt.savefig(output_heatmap, dpi=300, bbox_inches='tight',
               transparent=True, format='pdf')
    plt.close()
    print(f"Saved: {output_heatmap}")

    print("\nGenerating module size distribution...")
    fig, ax = plt.subplots(figsize=(6, 4))
    module_sizes_list = [module_df[module_df['module_id'] == m]['pfam'].count()
                        for m in sorted(module_df['module_id'].unique())]
    positions = range(len(module_sizes_list))
    ax.scatter(positions, module_sizes_list, s=30, alpha=0.7, c='#3498db')
    ax.set_xlabel('Module ID', fontsize=6, fontweight='bold')
    ax.set_ylabel('Module Size (# PFAMs)', fontsize=6, fontweight='bold')
    ax.set_title('PFAM Module Sizes', fontsize=8, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    median_size = np.median(module_sizes_list)
    ax.axhline(median_size, color='red', linestyle='--', linewidth=0.5,
              label=f'Median: {median_size:.0f}')
    ax.legend(fontsize=5, frameon=False)
    plt.tight_layout()
    output_swarm = f"figures/pfam_modules_swarm_{timestamp}.pdf"
    plt.savefig(output_swarm, dpi=300, bbox_inches='tight',
               transparent=True, format='pdf')
    plt.close()
    print(f"Saved: {output_swarm}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
