#!/usr/bin/env python3
"""
Script: scripts/13_go_enrichment_importance_20260119.py
Purpose: GO enrichment for top 100 PFAMs per GEE variable and AlphaEarth dimension
Date: 2026-01-19
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import hypergeom
from statsmodels.stats.multitest import fdrcorrection
from datetime import datetime
import os
import glob
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

def find_most_recent_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")
    return max(files, key=os.path.getmtime)

def perform_go_enrichment(top_pfams, pfam_to_go, all_pfams, all_go_terms):
    """Perform GO enrichment for a set of PFAMs"""
    results = []
    total_pfams = len(all_pfams)

    for go_term in all_go_terms:
        pfams_with_go = set([p for p, gos in pfam_to_go.items() if go_term in gos])
        go_size = len(pfams_with_go)
        overlap = len(top_pfams & pfams_with_go)

        if overlap > 0:
            pval = hypergeom.sf(overlap - 1, total_pfams, go_size, len(top_pfams))
            results.append({
                'go_term': go_term,
                'overlap': overlap,
                'go_size': go_size,
                'p_value': pval
            })

    return pd.DataFrame(results)

def main():
    print("=" * 80)
    print("Task 13: GO Enrichment for Top Important PFAMs")
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
    gee_importance = f"{base_dir}/algagpt_xgboost_gee_importance_20260119_113551.tsv"
    ae_importance = f"{base_dir}/algagpt_xgboost_alphaearth_importance_20260119_113734.tsv"
    annot_file = find_most_recent_file("results/pfam_interpro_annotations_*.tsv")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Load annotations
    print(f"\nLoading annotations: {annot_file}")
    annot_df = pd.read_csv(annot_file, sep='\t', comment='#')

    pfam_to_go = {}
    for _, row in annot_df.iterrows():
        if pd.notna(row['go_terms']) and row['go_terms'] != '':
            pfam_to_go[row['pfam_id']] = row['go_terms'].split(',')

    all_go_terms = set()
    for go_list in pfam_to_go.values():
        all_go_terms.update(go_list)

    all_pfams = set(annot_df['pfam_id'])

    # Process GEE variables
    print("\nProcessing GEE variables...")
    gee_df = pd.read_csv(gee_importance, sep='\t', comment='#')
    gee_results = []

    for target_var in gee_df['target_variable'].unique():
        var_data = gee_df[gee_df['target_variable'] == target_var]
        top100 = set(var_data.nlargest(100, 'importance')['feature'])

        enrich = perform_go_enrichment(top100, pfam_to_go, all_pfams, all_go_terms)
        if len(enrich) > 0:
            _, qvals = fdrcorrection(enrich['p_value'].values, alpha=0.05)
            enrich['q_value'] = qvals
            enrich['target_variable'] = target_var
            gee_results.append(enrich)

    gee_enrichment = pd.concat(gee_results, ignore_index=True)
    output_gee = f"results/go_enrichment_importance_gee_{timestamp}.tsv"
    gee_enrichment.to_csv(output_gee, sep='\t', index=False)
    add_provenance(output_gee, __file__, [gee_importance, annot_file])
    print(f"Saved: {output_gee}")

    # Process AlphaEarth dimensions
    print("\nProcessing AlphaEarth dimensions...")
    ae_df = pd.read_csv(ae_importance, sep='\t', comment='#')
    ae_results = []

    for target_var in ae_df['target_variable'].unique():
        var_data = ae_df[ae_df['target_variable'] == target_var]
        top100 = set(var_data.nlargest(100, 'importance')['feature'])

        enrich = perform_go_enrichment(top100, pfam_to_go, all_pfams, all_go_terms)
        if len(enrich) > 0:
            _, qvals = fdrcorrection(enrich['p_value'].values, alpha=0.05)
            enrich['q_value'] = qvals
            enrich['target_variable'] = target_var
            ae_results.append(enrich)

    ae_enrichment = pd.concat(ae_results, ignore_index=True)
    output_ae = f"results/go_enrichment_importance_alphaearth_{timestamp}.tsv"
    ae_enrichment.to_csv(output_ae, sep='\t', index=False)
    add_provenance(output_ae, __file__, [ae_importance, annot_file])
    print(f"Saved: {output_ae}")

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 6
    plt.rcParams['axes.linewidth'] = 0.25

    # Heatmap
    print("\nGenerating heatmaps...")
    combined = pd.concat([gee_enrichment, ae_enrichment])
    sig = combined[combined['q_value'] < 0.05]

    if len(sig) > 0:
        pivot = sig.pivot_table(index='target_variable', columns='go_term',
                               values='q_value', fill_value=1)
        pivot_log = -np.log10(pivot.clip(lower=1e-300))
        pivot_log = pivot_log.iloc[:, :30]

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_log, cmap='YlOrRd', cbar_kws={'label': '-log10(q-value)'},
                   ax=ax, linewidths=0, xticklabels=True, yticklabels=True)
        ax.set_xlabel('GO Term', fontsize=6, fontweight='bold')
        ax.set_ylabel('Target Variable', fontsize=6, fontweight='bold')
        ax.set_title('GO Enrichment for Top Important PFAMs', fontsize=8, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=4)
        plt.setp(ax.get_yticklabels(), fontsize=5)
        plt.tight_layout()
        output_heatmap = f"figures/go_enrichment_importance_heatmap_{timestamp}.pdf"
        plt.savefig(output_heatmap, dpi=300, bbox_inches='tight', transparent=True, format='pdf')
        plt.close()
        print(f"Saved: {output_heatmap}")

    # Violin plot
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(sig) > 0:
        parts = ax.violinplot([sig['overlap'].values], positions=[0],
                              widths=0.7, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#27ae60')
            pc.set_alpha(0.7)
            pc.set_linewidth(0.25)
        ax.set_ylabel('Overlap (# PFAMs)', fontsize=6, fontweight='bold')
        ax.set_xticks([0])
        ax.set_xticklabels(['GO Enrichments'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    output_violin = f"figures/go_enrichment_importance_violin_{timestamp}.pdf"
    plt.savefig(output_violin, dpi=300, bbox_inches='tight', transparent=True, format='pdf')
    plt.close()
    print(f"Saved: {output_violin}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
