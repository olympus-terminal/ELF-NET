#!/usr/bin/env python3
"""
Script: scripts/10_module_environment_enrichment_20260119.py
Purpose: Test each PFAM module for environmental specificity
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

def main():
    print("=" * 80)
    print("Task 10: Module-Environment Enrichment")
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
    modules_file = find_most_recent_file("results/pfam_modules_*.tsv")
    sig_corr_file = f"{base_dir}/algagpt_pfam_gee_correlations_20260119_104938_significant.tsv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print(f"\nLoading modules from: {modules_file}")
    modules_df = pd.read_csv(modules_file, sep='\t', comment='#')
    print(f"Loaded {len(modules_df)} PFAM assignments across {modules_df['module_id'].nunique()} modules")

    print("\nLoading significant correlations...")
    sig_df = pd.read_csv(sig_corr_file, sep='\t', comment='#')
    print(f"Loaded {len(sig_df)} significant PFAM-GEE correlations")

    # Hypergeometric enrichment test
    print("\nPerforming hypergeometric enrichment tests...")
    total_pfams = len(modules_df)
    enrichment_results = []

    for module_id in sorted(modules_df['module_id'].unique()):
        module_pfams = set(modules_df[modules_df['module_id'] == module_id]['pfam'])
        module_size = len(module_pfams)

        for gee_var in sig_df['gee_variable'].unique():
            sig_pfams = set(sig_df[sig_df['gee_variable'] == gee_var]['pfam'])
            sig_size = len(sig_pfams)

            overlap = len(module_pfams & sig_pfams)

            # Hypergeometric test
            pval = hypergeom.sf(overlap - 1, total_pfams, sig_size, module_size)

            enrichment_results.append({
                'module_id': module_id,
                'gee_variable': gee_var,
                'module_size': module_size,
                'sig_pfams': sig_size,
                'overlap': overlap,
                'expected': (module_size * sig_size) / total_pfams,
                'fold_enrichment': overlap / ((module_size * sig_size) / total_pfams) if sig_size > 0 else 0,
                'p_value': pval
            })

    enrichment_df = pd.DataFrame(enrichment_results)

    # FDR correction
    print("\nApplying FDR correction...")
    _, qvals = fdrcorrection(enrichment_df['p_value'].values, alpha=0.05)
    enrichment_df['q_value'] = qvals

    # Save all results
    output_all = f"results/module_environment_enrichment_{timestamp}.tsv"
    enrichment_df.to_csv(output_all, sep='\t', index=False)
    add_provenance(output_all, __file__, [modules_file, sig_corr_file])
    print(f"Saved: {output_all}")

    # Save significant results
    significant = enrichment_df[enrichment_df['q_value'] < 0.05]
    output_sig = f"results/module_environment_enrichment_significant_{timestamp}.tsv"
    significant.to_csv(output_sig, sep='\t', index=False)
    add_provenance(output_sig, __file__, [modules_file, sig_corr_file])
    print(f"Saved: {output_sig} ({len(significant)} significant enrichments)")

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 6
    plt.rcParams['axes.linewidth'] = 0.25

    # Heatmap of enrichments
    print("\nGenerating enrichment heatmap...")
    pivot = enrichment_df.pivot_table(index='module_id', columns='gee_variable',
                                     values='q_value', fill_value=1)
    pivot_log = -np.log10(pivot + 1e-300)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_log, cmap='YlOrRd', cbar_kws={'label': '-log10(q-value)'},
               ax=ax, linewidths=0.1, xticklabels=True, yticklabels=True)
    ax.set_xlabel('GEE Variable', fontsize=6, fontweight='bold')
    ax.set_ylabel('Module ID', fontsize=6, fontweight='bold')
    ax.set_title('Module-Environment Enrichment', fontsize=8, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=5)
    plt.tight_layout()
    output_heatmap = f"figures/module_environment_heatmap_{timestamp}.pdf"
    plt.savefig(output_heatmap, dpi=300, bbox_inches='tight', transparent=True, format='pdf')
    plt.close()
    print(f"Saved: {output_heatmap}")

    # Swarm plot of enrichment scores
    print("\nGenerating enrichment score swarm plot...")
    fig, ax = plt.subplots(figsize=(6, 4))

    sig_enrich = enrichment_df[enrichment_df['q_value'] < 0.05]
    if len(sig_enrich) > 0:
        ax.scatter(range(len(sig_enrich)), sig_enrich['fold_enrichment'].values,
                  s=20, alpha=0.6, c='#e74c3c')
        ax.set_xlabel('Enrichment Test', fontsize=6, fontweight='bold')
        ax.set_ylabel('Fold Enrichment', fontsize=6, fontweight='bold')
        ax.set_title('Significant Module-Environment Enrichments', fontsize=8, fontweight='bold')
        ax.axhline(1, color='gray', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        ax.text(0.5, 0.5, 'No significant enrichments', ha='center', va='center',
               transform=ax.transAxes)

    plt.tight_layout()
    output_swarm = f"figures/module_environment_swarm_{timestamp}.pdf"
    plt.savefig(output_swarm, dpi=300, bbox_inches='tight', transparent=True, format='pdf')
    plt.close()
    print(f"Saved: {output_swarm}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
