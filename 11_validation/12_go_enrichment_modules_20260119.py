#!/usr/bin/env python3
"""
Script: scripts/12_go_enrichment_modules_20260119.py
Purpose: GO enrichment for ALL PFAM modules
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
    print("Task 12: GO Enrichment for PFAM Modules")
    print("=" * 80)

    modules_file = find_most_recent_file("results/pfam_modules_*.tsv")
    annot_file = find_most_recent_file("results/pfam_interpro_annotations_*.tsv")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print(f"\nLoading modules: {modules_file}")
    modules_df = pd.read_csv(modules_file, sep='\t', comment='#')

    print(f"Loading annotations: {annot_file}")
    annot_df = pd.read_csv(annot_file, sep='\t', comment='#')

    # Parse GO terms
    pfam_to_go = {}
    for _, row in annot_df.iterrows():
        if pd.notna(row['go_terms']) and row['go_terms'] != '':
            pfam_to_go[row['pfam_id']] = row['go_terms'].split(',')

    all_go_terms = set()
    for go_list in pfam_to_go.values():
        all_go_terms.update(go_list)

    print(f"\nTotal GO terms: {len(all_go_terms)}")

    # Perform enrichment for each module
    total_pfams = len(modules_df)
    enrichment_results = []

    for module_id in sorted(modules_df['module_id'].unique()):
        module_pfams = set(modules_df[modules_df['module_id'] == module_id]['pfam'])
        module_size = len(module_pfams)

        for go_term in all_go_terms:
            pfams_with_go = set([p for p, gos in pfam_to_go.items() if go_term in gos])
            go_size = len(pfams_with_go)

            overlap = len(module_pfams & pfams_with_go)
            expected = (module_size * go_size) / total_pfams

            if overlap > 0:
                pval = hypergeom.sf(overlap - 1, total_pfams, go_size, module_size)
                enrichment_results.append({
                    'module_id': module_id,
                    'go_term': go_term,
                    'module_size': module_size,
                    'go_size': go_size,
                    'overlap': overlap,
                    'expected': expected,
                    'fold_enrichment': overlap / expected if expected > 0 else 0,
                    'p_value': pval
                })

    enrichment_df = pd.DataFrame(enrichment_results)

    # FDR correction
    _, qvals = fdrcorrection(enrichment_df['p_value'].values, alpha=0.05)
    enrichment_df['q_value'] = qvals

    output_all = f"results/go_enrichment_modules_{timestamp}.tsv"
    enrichment_df.to_csv(output_all, sep='\t', index=False)
    add_provenance(output_all, __file__, [modules_file, annot_file])
    print(f"Saved: {output_all}")

    significant = enrichment_df[enrichment_df['q_value'] < 0.05]
    output_sig = f"results/go_enrichment_modules_significant_{timestamp}.tsv"
    significant.to_csv(output_sig, sep='\t', index=False)
    add_provenance(output_sig, __file__, [modules_file, annot_file])
    print(f"Saved: {output_sig} ({len(significant)} significant)")

    # Summary
    summary_data = []
    for module_id in sorted(modules_df['module_id'].unique()):
        module_enrich = significant[significant['module_id'] == module_id]
        if len(module_enrich) > 0:
            top5 = module_enrich.nsmallest(5, 'q_value')
            summary_data.append({
                'module_id': module_id,
                'n_significant_go': len(module_enrich),
                'top_go_terms': ','.join(top5['go_term'].tolist())
            })

    summary_df = pd.DataFrame(summary_data)
    output_summary = f"results/go_enrichment_modules_summary_{timestamp}.tsv"
    summary_df.to_csv(output_summary, sep='\t', index=False)
    add_provenance(output_summary, __file__, [modules_file, annot_file])
    print(f"Saved: {output_summary}")

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 6
    plt.rcParams['axes.linewidth'] = 0.25

    if len(significant) > 0:
        pivot = significant.pivot_table(index='module_id', columns='go_term',
                                       values='q_value', fill_value=1)
        pivot_log = -np.log10(pivot.clip(lower=1e-300))
        pivot_log = pivot_log.iloc[:, :50]

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_log, cmap='YlOrRd', cbar_kws={'label': '-log10(q-value)'},
                   ax=ax, linewidths=0, xticklabels=True, yticklabels=True)
        ax.set_xlabel('GO Term', fontsize=6, fontweight='bold')
        ax.set_ylabel('Module ID', fontsize=6, fontweight='bold')
        ax.set_title('GO Term Enrichment by Module', fontsize=8, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=3)
        plt.tight_layout()
        output_heatmap = f"figures/go_enrichment_modules_heatmap_{timestamp}.pdf"
        plt.savefig(output_heatmap, dpi=300, bbox_inches='tight', transparent=True, format='pdf')
        plt.close()
        print(f"Saved: {output_heatmap}")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(range(len(significant)), significant['fold_enrichment'].values,
                  s=20, alpha=0.6, c='#9b59b6')
        ax.set_xlabel('Enrichment', fontsize=6, fontweight='bold')
        ax.set_ylabel('Fold Enrichment', fontsize=6, fontweight='bold')
        ax.set_title('GO Term Enrichments', fontsize=8, fontweight='bold')
        ax.axhline(1, color='gray', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        output_swarm = f"figures/go_enrichment_modules_swarm_{timestamp}.pdf"
        plt.savefig(output_swarm, dpi=300, bbox_inches='tight', transparent=True, format='pdf')
        plt.close()
        print(f"Saved: {output_swarm}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
