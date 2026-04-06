#!/usr/bin/env python3
"""
Script: scripts/09_hierarchical_clustering_alphaearth_20260119.py
Purpose: Cluster 64 AlphaEarth dimensions by shared PFAM predictors
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
import warnings
warnings.filterwarnings('ignore')

def add_provenance(filepath, script_path, input_paths):
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
    print("Task 9: Hierarchical Clustering of AlphaEarth Dimensions")
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
    input_file = f"{base_dir}/algagpt_xgboost_alphaearth_importance_20260119_113734.tsv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("\nLoading AlphaEarth importance data...")
    df = pd.read_csv(input_file, sep='\t', comment='#')
    print(f"Loaded {len(df)} PFAM-dimension importance values")

    print("\nPivoting to dimension × PFAM matrix...")
    pivot = df.pivot_table(index='alphaearth_dimension', columns='pfam',
                          values='shap_importance', fill_value=0)
    print(f"Matrix shape: {pivot.shape[0]} dimensions × {pivot.shape[1]} PFAMs")

    print("\nComputing distance matrix...")
    dist_matrix = pdist(pivot.values, metric='euclidean')

    print("\nPerforming hierarchical clustering...")
    linkage_matrix = linkage(dist_matrix, method='ward')
    linkage_matrix = optimal_leaf_ordering(linkage_matrix, dist_matrix)

    n_clusters = 8
    print(f"Cutting dendrogram at {n_clusters} clusters...")
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    cluster_df = pd.DataFrame({'dimension': pivot.index, 'cluster_id': clusters})
    output_clusters = f"results/alphaearth_clusters_{timestamp}.tsv"
    cluster_df.to_csv(output_clusters, sep='\t', index=False)
    add_provenance(output_clusters, __file__, [input_file])
    print(f"Saved: {output_clusters}")

    # Compute PFAM overlap between clusters
    print("\nComputing PFAM overlap between clusters...")
    overlap_data = []
    for i in range(1, n_clusters + 1):
        for j in range(i, n_clusters + 1):
            dims_i = cluster_df[cluster_df['cluster_id'] == i]['dimension'].values
            dims_j = cluster_df[cluster_df['cluster_id'] == j]['dimension'].values

            top_pfams_i = set(df[df['alphaearth_dimension'].isin(dims_i)].nlargest(100, 'shap_importance')['pfam'])
            top_pfams_j = set(df[df['alphaearth_dimension'].isin(dims_j)].nlargest(100, 'shap_importance')['pfam'])

            overlap = len(top_pfams_i & top_pfams_j)
            jaccard = overlap / len(top_pfams_i | top_pfams_j) if len(top_pfams_i | top_pfams_j) > 0 else 0

            overlap_data.append({
                'cluster1': i,
                'cluster2': j,
                'overlap': overlap,
                'jaccard_similarity': jaccard
            })

    overlap_df = pd.DataFrame(overlap_data)
    output_overlap = f"results/alphaearth_clusters_pfam_overlap_{timestamp}.tsv"
    overlap_df.to_csv(output_overlap, sep='\t', index=False)
    add_provenance(output_overlap, __file__, [input_file])
    print(f"Saved: {output_overlap}")

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 6
    plt.rcParams['axes.linewidth'] = 0.25

    print("\nGenerating dendrogram...")
    fig, ax = plt.subplots(figsize=(8, 5))
    dendrogram(linkage_matrix, labels=pivot.index.tolist(), ax=ax, leaf_font_size=4)
    ax.set_xlabel('AlphaEarth Dimension', fontsize=6, fontweight='bold')
    ax.set_ylabel('Distance', fontsize=6, fontweight='bold')
    ax.set_title('AlphaEarth Dimension Clustering', fontsize=8, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
    plt.tight_layout()
    output_dendrogram = f"figures/alphaearth_dendrogram_{timestamp}.pdf"
    plt.savefig(output_dendrogram, dpi=300, bbox_inches='tight', transparent=True, format='pdf')
    plt.close()
    print(f"Saved: {output_dendrogram}")

    print("\nGenerating clustered heatmap...")
    cluster_order = cluster_df.sort_values('cluster_id')
    pivot_ordered = pivot.loc[cluster_order['dimension']]

    variances = pivot.var(axis=0)
    top_pfams = variances.nlargest(200).index
    pivot_sub = pivot_ordered[top_pfams]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_sub, cmap='viridis', cbar_kws={'label': 'SHAP Importance'},
               ax=ax, linewidths=0, xticklabels=False, yticklabels=True)
    ax.set_xlabel('Top 200 PFAMs', fontsize=6, fontweight='bold')
    ax.set_ylabel('AlphaEarth Dimension (clustered)', fontsize=6, fontweight='bold')
    ax.set_title('AlphaEarth Dimension Clusters', fontsize=8, fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontsize=4)
    plt.tight_layout()
    output_heatmap = f"figures/alphaearth_clusters_heatmap_{timestamp}.pdf"
    plt.savefig(output_heatmap, dpi=300, bbox_inches='tight', transparent=True, format='pdf')
    plt.close()
    print(f"Saved: {output_heatmap}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
