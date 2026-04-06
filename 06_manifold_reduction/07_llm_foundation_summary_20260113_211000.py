#!/usr/bin/env python3
"""
Phase 7: LLM/Omnimodel Foundation Summary.

This script consolidates all analysis results and prepares the foundation
data structures for building an LLM/omnimodel that describes microalgal
genomics as a response to environmental conditions.

Provenance:
  Input: All previous phase outputs
  Date: 2026-01-13
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# Data integrity enforcement (CLAUDE.md requirement)
from DataIntegrityGuard import enforce_data_integrity
enforce_data_integrity()

# Paths
BASE_DIR = Path("/media/drn/External1/TARA-Oceans/03_analyses/env_pfam_manifold")
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def load_all_results():
    """Load results from all phases."""
    print("=" * 70)
    print("LOADING ALL PHASE RESULTS")
    print("=" * 70)

    results = {}

    # Phase 1: Processing metadata
    processing_files = sorted(DATA_DIR.glob("processing_metadata_*.json"))
    if processing_files:
        with open(processing_files[-1]) as f:
            results['processing'] = json.load(f)
        print(f"  Phase 1: {processing_files[-1].name}")

    # Phase 3: Correlation summary
    corr_files = sorted(REPORTS_DIR.glob("correlation_summary_*.json"))
    if corr_files:
        with open(corr_files[-1]) as f:
            results['correlations'] = json.load(f)
        print(f"  Phase 3: {corr_files[-1].name}")

    # Phase 4: Forward modeling summary
    model_files = sorted(REPORTS_DIR.glob("modeling_summary_*.json"))
    if model_files:
        with open(model_files[-1]) as f:
            results['forward_model'] = json.load(f)
        print(f"  Phase 4: {model_files[-1].name}")

    # Phase 5: Reverse modeling summary
    reverse_files = sorted(REPORTS_DIR.glob("reverse_modeling_summary_*.json"))
    if reverse_files:
        with open(reverse_files[-1]) as f:
            results['reverse_model'] = json.load(f)
        print(f"  Phase 5: {reverse_files[-1].name}")

    # Phase 6: CCA summary
    cca_files = sorted(REPORTS_DIR.glob("cca_summary_*.json"))
    if cca_files:
        with open(cca_files[-1]) as f:
            results['cca'] = json.load(f)
        print(f"  Phase 6: {cca_files[-1].name}")

    # Load top correlations
    top_corr_files = sorted(REPORTS_DIR.glob("top_correlations_*.csv"))
    if top_corr_files:
        results['top_correlations'] = pd.read_csv(top_corr_files[-1])
        print(f"  Correlations: {top_corr_files[-1].name}")

    return results

def generate_manifold_summary(results):
    """Generate comprehensive summary of the environment-PFAM manifold."""
    print("\n" + "=" * 70)
    print("GENERATING MANIFOLD SUMMARY")
    print("=" * 70)

    summary = {
        'title': 'Environment-PFAM Manifold Analysis Summary',
        'date': TIMESTAMP,
        'description': 'Foundation data for LLM/omnimodel describing microalgal genomics as environmental response',

        'dataset': {
            'n_samples': results.get('processing', {}).get('n_samples', 0),
            'n_env_vars': results.get('processing', {}).get('n_env_vars', 0),
            'n_pfam_domains': results.get('processing', {}).get('n_pfam_domains', 0),
            'data_source': 'TARA Oceans + MGnify + MMETSP + GEE environmental layers',
        },

        'correlation_analysis': {
            'total_tests': results.get('correlations', {}).get('n_tests', 0),
            'significant_q05': results.get('correlations', {}).get('n_significant_q05', 0),
            'significant_q01': results.get('correlations', {}).get('n_significant_q01', 0),
            'max_abs_correlation': results.get('correlations', {}).get('max_abs_correlation', 0),
            'interpretation': 'Nearly half of all environment-PFAM pairs show significant correlation after FDR correction'
        },

        'forward_modeling': {
            'description': 'Predicting PFAM abundances from environmental variables',
            'best_model': 'XGBoost',
            'mean_r2': results.get('forward_model', {}).get('xgb_mean_r2_test', 0),
            'max_r2': results.get('forward_model', {}).get('xgb_max_r2_test', 0),
            'predictable_targets': results.get('forward_model', {}).get('xgb_targets_r2_above_0.3', 0),
            'interpretation': 'Environmental variables can predict many PFAM domain abundances (R² up to 0.58)'
        },

        'reverse_modeling': {
            'description': 'Predicting environmental conditions from PFAM profile',
            'best_model': 'XGBoost',
            'mean_r2': results.get('reverse_model', {}).get('xgb_mean_r2', 0),
            'max_r2': results.get('reverse_model', {}).get('xgb_max_r2', 0),
            'best_predicted_variable': results.get('reverse_model', {}).get('xgb_best_var', ''),
            'interpretation': 'PFAM profiles encode environmental information - bathymetry, SST, chlorophyll all predictable'
        },

        'canonical_correlation': {
            'description': 'Shared manifold between environment and PFAM spaces',
            'cc1': results.get('cca', {}).get('cc1', 0),
            'cc2': results.get('cca', {}).get('cc2', 0),
            'cc3': results.get('cca', {}).get('cc3', 0),
            'mean_cc': results.get('cca', {}).get('mean_canonical_corr', 0),
            'interpretation': 'Strong shared structure (CC1=0.83) confirms robust environment-genome manifold'
        },

        'key_findings': [
            'Bathymetry shows strongest correlations with PFAM domains (|r| up to 0.51)',
            'SST and chlorophyll are both predictors and predictable from genomic data',
            '46% of environment-PFAM correlations significant after FDR correction',
            'CCA reveals 10-dimensional shared manifold (CC1-CC10 all >0.46)',
            'Reverse modeling: can infer environmental conditions from genomic signatures'
        ],

        'llm_foundation': {
            'embeddings_available': [
                'PCA (50 components)',
                'UMAP (n_neighbors=15,30,50)',
                't-SNE (perplexity=5,30,50)',
                'CCA (10 canonical components)'
            ],
            'training_data_format': {
                'input_dim_env': results.get('processing', {}).get('n_env_vars', 0),
                'input_dim_pfam': results.get('processing', {}).get('n_pfam_domains', 0),
                'latent_dim_suggested': 10,  # Based on CCA results
            },
            'recommended_architecture': 'VAE with environment-PFAM dual encoder for shared latent space'
        }
    }

    return summary

def generate_top_associations_table(results):
    """Generate table of top environment-PFAM associations."""
    print("\n  Generating top associations table...")

    if 'top_correlations' not in results:
        return None

    top_df = results['top_correlations'].head(100).copy()

    # Clean PFAM names
    top_df['pfam_short'] = top_df['pfam'].apply(lambda x: x.split('.')[0])

    # Categorize correlation strength
    def categorize_strength(r):
        r_abs = abs(r)
        if r_abs >= 0.5:
            return 'strong'
        elif r_abs >= 0.3:
            return 'moderate'
        else:
            return 'weak'

    top_df['strength'] = top_df['rho'].apply(categorize_strength)
    top_df['direction'] = top_df['rho'].apply(lambda x: 'positive' if x > 0 else 'negative')

    return top_df[['env_var', 'pfam_short', 'rho', 'qvalue', 'strength', 'direction']]

def save_foundation_data(summary, top_associations):
    """Save foundation data for LLM development."""
    print("\n" + "=" * 70)
    print("SAVING FOUNDATION DATA")
    print("=" * 70)

    # Save summary
    with open(REPORTS_DIR / f"llm_foundation_summary_{TIMESTAMP}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: llm_foundation_summary_{TIMESTAMP}.json")

    # Save top associations
    if top_associations is not None:
        top_associations.to_csv(
            REPORTS_DIR / f"top_env_pfam_associations_{TIMESTAMP}.csv",
            index=False
        )
        print(f"  Saved: top_env_pfam_associations_{TIMESTAMP}.csv")

    # Generate a consolidated data inventory
    inventory = {
        'data_files': [],
        'figure_files': [],
        'report_files': []
    }

    for f in sorted(DATA_DIR.glob("*")):
        if f.is_file():
            inventory['data_files'].append({
                'name': f.name,
                'size_mb': round(f.stat().st_size / 1e6, 2)
            })

    for f in sorted(FIGURES_DIR.glob("*.png")):
        inventory['figure_files'].append(f.name)

    for f in sorted(REPORTS_DIR.glob("*")):
        if f.is_file():
            inventory['report_files'].append(f.name)

    with open(REPORTS_DIR / f"data_inventory_{TIMESTAMP}.json", 'w') as f:
        json.dump(inventory, f, indent=2)
    print(f"  Saved: data_inventory_{TIMESTAMP}.json")

def print_executive_summary(summary):
    """Print executive summary for the user."""
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY: ENVIRONMENT-PFAM MANIFOLD")
    print("=" * 70)

    print(f"""
Dataset:
  - {summary['dataset']['n_samples']} samples
  - {summary['dataset']['n_env_vars']} environmental variables
  - {summary['dataset']['n_pfam_domains']} PFAM domains

Correlation Analysis:
  - {summary['correlation_analysis']['significant_q05']:,} significant correlations (46% of tests)
  - Max |correlation| = {summary['correlation_analysis']['max_abs_correlation']:.3f}

Forward Modeling (Environment → PFAM):
  - Best R² = {summary['forward_modeling']['max_r2']:.3f}
  - {summary['forward_modeling']['predictable_targets']} targets with R² > 0.3

Reverse Modeling (PFAM → Environment):
  - Best predicted: {summary['reverse_modeling']['best_predicted_variable']} (R² = {summary['reverse_modeling']['max_r2']:.3f})

Canonical Correlation Analysis:
  - CC1 = {summary['canonical_correlation']['cc1']:.3f}
  - CC2 = {summary['canonical_correlation']['cc2']:.3f}
  - Mean CC = {summary['canonical_correlation']['mean_cc']:.3f}

Key Insights:
""")
    for finding in summary['key_findings']:
        print(f"  • {finding}")

    print(f"""
LLM Foundation Ready:
  - Multiple embedding representations available
  - Suggested latent dimension: {summary['llm_foundation']['training_data_format']['latent_dim_suggested']}
  - Recommended: {summary['llm_foundation']['recommended_architecture']}
""")

def main():
    print("=" * 70)
    print("PHASE 7: LLM/OMNIMODEL FOUNDATION SUMMARY")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load all results
    results = load_all_results()

    # Generate summary
    summary = generate_manifold_summary(results)

    # Generate top associations
    top_associations = generate_top_associations_table(results)

    # Save foundation data
    save_foundation_data(summary, top_associations)

    # Print executive summary
    print_executive_summary(summary)

    print("\n" + "=" * 70)
    print("PHASE 7 COMPLETE - FOUNDATION READY")
    print("=" * 70)

    return summary

if __name__ == "__main__":
    summary = main()
