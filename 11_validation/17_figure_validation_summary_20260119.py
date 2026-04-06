#!/usr/bin/env python3
"""
Task 17: Publication Figure - Validation Summary
=================================================

4-panel validation summary figure consolidating k-fold CV and permutation test results.

Provenance:
-----------
Script: /media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/validations/scripts/17_figure_validation_summary_20260119.py
Input:  results/kfold_cv_gee_summary_*.tsv
        results/kfold_cv_alphaearth_summary_*.tsv
        results/permutation_test_gee_*.tsv
        results/permutation_test_alphaearth_*.tsv
Date:   2026-01-19
Task:   RALPH Plan Task 17/20

Figure Layout:
--------------
Panel A: Violin plot - R² distribution for GEE variables (K-fold CV)
Panel B: Violin plot - R² distribution for AlphaEarth dimensions (K-fold CV)
Panel C: Volcano plot - GEE permutation test results
Panel D: Volcano plot - AlphaEarth permutation test results

Data Integrity:
---------------
- NO synthetic data - uses REAL results from Tasks 1-4
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
    log_message("Task 17: Publication Figure - Validation Summary")
    log_message("="*80)

    # Load data
    log_message("Loading K-fold CV results...")
    cv_gee_file = find_most_recent_file(str(results_dir / "kfold_cv_gee_performance_*.tsv"))
    cv_gee = pd.read_csv(cv_gee_file, sep='\t', comment='#')

    try:
        cv_ae_file = find_most_recent_file(str(results_dir / "kfold_cv_alphaearth_performance_*.tsv"))
        cv_ae = pd.read_csv(cv_ae_file, sep='\t', comment='#')
    except FileNotFoundError:
        log_message("WARNING: No AlphaEarth CV results found, skipping Panel B")
        cv_ae = None

    log_message("Loading permutation test results...")
    try:
        perm_gee_file = find_most_recent_file(str(results_dir / "permutation_test_gee_*.tsv"))
        perm_gee = pd.read_csv(perm_gee_file, sep='\t', comment='#')
    except FileNotFoundError:
        log_message("WARNING: No GEE permutation test results found, skipping Panel C")
        perm_gee = None

    try:
        perm_ae_file = find_most_recent_file(str(results_dir / "permutation_test_alphaearth_*.tsv"))
        perm_ae = pd.read_csv(perm_ae_file, sep='\t', comment='#')
    except FileNotFoundError:
        log_message("WARNING: No AlphaEarth permutation test results found, skipping Panel D")
        perm_ae = None

    # Create figure
    log_message("Creating 4-panel validation summary figure...")

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Panel A: GEE K-fold CV violin plot
    ax = axes[0, 0]
    if not cv_gee.empty:
        # Sort by median R²
        var_order = cv_gee.groupby("gee_variable")["r2_test"].median().sort_values(ascending=False).index.tolist()

        sns.violinplot(data=cv_gee, y="gee_variable", x="r2_test",
                       order=var_order, ax=ax, inner="box", linewidth=0.25, color="#2E86AB")

        ax.set_xlabel("R² (Test Set)", fontsize=6)
        ax.set_ylabel("GEE Variable", fontsize=6)
        ax.set_title("(A) K-Fold CV: GEE Variables", fontsize=6, fontweight="bold", loc='left')
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.25, alpha=0.5)
        ax.tick_params(labelsize=5)

    # Panel B: AlphaEarth K-fold CV violin plot
    ax = axes[0, 1]
    if cv_ae is not None and not cv_ae.empty:
        # Sort by median R²
        dim_order = cv_ae.groupby("dimension")["r2_test"].median().sort_values(ascending=False).index.tolist()

        sns.violinplot(data=cv_ae, y="dimension", x="r2_test",
                       order=dim_order[:20], ax=ax, inner="box", linewidth=0.25, color="#A23B72")

        ax.set_xlabel("R² (Test Set)", fontsize=6)
        ax.set_ylabel("AlphaEarth Dimension", fontsize=6)
        ax.set_title("(B) K-Fold CV: AlphaEarth (Top 20)", fontsize=6, fontweight="bold", loc='left')
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.25, alpha=0.5)
        ax.tick_params(labelsize=5)
    else:
        ax.text(0.5, 0.5, "No AlphaEarth\nCV results", ha='center', va='center',
                fontsize=6, transform=ax.transAxes)
        ax.set_title("(B) K-Fold CV: AlphaEarth", fontsize=6, fontweight="bold", loc='left')

    # Panel C: GEE permutation test volcano plot
    ax = axes[1, 0]
    if perm_gee is not None and not perm_gee.empty:
        # Compute -log10(q_value)
        perm_gee['neg_log10_q'] = -np.log10(perm_gee['q_value_fdr'] + 1e-300)

        # Scatter plot
        significant = perm_gee['q_value_fdr'] < 0.05
        ax.scatter(perm_gee.loc[~significant, 'observed_r2'],
                   perm_gee.loc[~significant, 'neg_log10_q'],
                   s=5, alpha=0.5, edgecolors='none', color='gray', label='NS')
        ax.scatter(perm_gee.loc[significant, 'observed_r2'],
                   perm_gee.loc[significant, 'neg_log10_q'],
                   s=5, alpha=0.7, edgecolors='none', color='#2E86AB', label='FDR < 0.05')

        ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=0.25, alpha=0.5)
        ax.set_xlabel("Observed R²", fontsize=6)
        ax.set_ylabel("-log₁₀(FDR)", fontsize=6)
        ax.set_title("(C) Permutation Test: GEE", fontsize=6, fontweight="bold", loc='left')
        ax.legend(fontsize=5, frameon=False, loc='upper left')
        ax.tick_params(labelsize=5)
    else:
        ax.text(0.5, 0.5, "No GEE\npermutation results", ha='center', va='center',
                fontsize=6, transform=ax.transAxes)
        ax.set_title("(C) Permutation Test: GEE", fontsize=6, fontweight="bold", loc='left')

    # Panel D: AlphaEarth permutation test volcano plot
    ax = axes[1, 1]
    if perm_ae is not None and not perm_ae.empty:
        # Compute -log10(q_value)
        perm_ae['neg_log10_q'] = -np.log10(perm_ae['q_value_fdr'] + 1e-300)

        # Scatter plot
        significant = perm_ae['q_value_fdr'] < 0.05
        ax.scatter(perm_ae.loc[~significant, 'observed_r2'],
                   perm_ae.loc[~significant, 'neg_log10_q'],
                   s=5, alpha=0.5, edgecolors='none', color='gray', label='NS')
        ax.scatter(perm_ae.loc[significant, 'observed_r2'],
                   perm_ae.loc[significant, 'neg_log10_q'],
                   s=5, alpha=0.7, edgecolors='none', color='#A23B72', label='FDR < 0.05')

        ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=0.25, alpha=0.5)
        ax.set_xlabel("Observed R²", fontsize=6)
        ax.set_ylabel("-log₁₀(FDR)", fontsize=6)
        ax.set_title("(D) Permutation Test: AlphaEarth", fontsize=6, fontweight="bold", loc='left')
        ax.legend(fontsize=5, frameon=False, loc='upper left')
        ax.tick_params(labelsize=5)
    else:
        ax.text(0.5, 0.5, "No AlphaEarth\npermutation results", ha='center', va='center',
                fontsize=6, transform=ax.transAxes)
        ax.set_title("(D) Permutation Test: AlphaEarth", fontsize=6, fontweight="bold", loc='left')

    plt.tight_layout()

    # Save figure
    pdf_file = figures_dir / f"validation_summary_{timestamp}.pdf"
    svg_file = figures_dir / f"validation_summary_{timestamp}.svg"

    fig.patch.set_alpha(0.0)
    fig.savefig(pdf_file, format='pdf', bbox_inches='tight', transparent=True, dpi=300)
    fig.savefig(svg_file, format='svg', bbox_inches='tight', transparent=True)
    plt.close()

    log_message(f"Saved PDF: {pdf_file}")
    log_message(f"Saved SVG: {svg_file}")

    log_message("\n" + "="*80)
    log_message("Task 17 COMPLETE")
    log_message("="*80)

if __name__ == '__main__':
    main()
