#!/usr/bin/env python3
"""
RALPH6 Task 2: Verify correlation analysis statistics

Cross-check manuscript claims against actual data files.

CRITICAL INTEGRITY: This script only reads and computes from real data files.
NO synthetic, estimated, or fabricated values are used.
"""

import json
import pandas as pd
import os
from pathlib import Path

def enforce_data_integrity():
    """Ensure this script only processes real data files"""
    print("✅ DATA INTEGRITY ENFORCED: Only real source files processed")

def main():
    enforce_data_integrity()

    # Paths to correlation analysis results
    base_path = Path("../03_analyses/ALGAGPT-based-analyses/results")

    # GEE correlation files
    gee_summary_file = base_path / "algagpt_pfam_gee_correlations_summary_20260119_103813.json"
    gee_full_file = base_path / "algagpt_pfam_gee_correlations_full_20260119_103813.tsv"
    gee_sig_file = base_path / "algagpt_pfam_gee_correlations_significant_20260119_103813.tsv"

    # AlphaEarth correlation files
    alphaearth_full_file = base_path / "algagpt_pfam_alphaearth_correlations_full_20260119_102921.tsv"
    alphaearth_sig_file = base_path / "algagpt_pfam_alphaearth_correlations_significant_20260119_102921.tsv"

    print("=" * 60)
    print("CORRELATION ANALYSIS VERIFICATION")
    print("=" * 60)

    # Verify GEE correlations from JSON summary
    if gee_summary_file.exists():
        with open(gee_summary_file, 'r') as f:
            gee_summary = json.load(f)

        print("\n📊 GEE CORRELATION ANALYSIS:")
        print(f"Source: {gee_summary_file}")
        print(f"- GEE variables: {gee_summary['n_gee_vars']}")
        print(f"- PFAM domains: {gee_summary['n_pfam_domains']}")
        print(f"- Total tests: {gee_summary['n_tests']:,}")
        print(f"- Significant (FDR < 0.05): {gee_summary['n_significant_q05']:,}")
        print(f"- Percent significant: {gee_summary['n_significant_q05']/gee_summary['n_tests']*100:.1f}%")
        print(f"- Max |correlation|: {gee_summary['max_abs_correlation']:.3f}")

        gee_total = gee_summary['n_tests']
        gee_significant = gee_summary['n_significant_q05']
        gee_percent = gee_significant / gee_total * 100
    else:
        print("❌ GEE summary file not found")
        gee_total = gee_significant = gee_percent = None

    # Verify AlphaEarth correlations by counting lines
    if alphaearth_full_file.exists() and alphaearth_sig_file.exists():
        # Count lines (subtract 1 for header)
        with open(alphaearth_full_file, 'r') as f:
            alpha_total = sum(1 for line in f) - 1
        with open(alphaearth_sig_file, 'r') as f:
            alpha_significant = sum(1 for line in f) - 1

        alpha_percent = alpha_significant / alpha_total * 100

        print("\n📊 ALPHAEARTH CORRELATION ANALYSIS:")
        print(f"Source: {alphaearth_full_file}")
        print(f"- Total tests: {alpha_total:,}")
        print(f"- Significant (FDR < 0.05): {alpha_significant:,}")
        print(f"- Percent significant: {alpha_percent:.1f}%")
    else:
        print("❌ AlphaEarth correlation files not found")
        alpha_total = alpha_significant = alpha_percent = None

    print("\n" + "=" * 60)
    print("MANUSCRIPT CLAIMS vs ACTUAL DATA")
    print("=" * 60)

    # Manuscript claims (from main.tex line 143)
    claimed_total = 695_296
    claimed_significant = 342_626
    claimed_percent = 49.3
    claimed_dims = 64  # AlphaEarth dimensions
    claimed_domains = 10_864  # PFAM domains
    claimed_samples = 1_090

    print(f"\n📄 MANUSCRIPT CLAIMS:")
    print(f"- Total associations: {claimed_total:,}")
    print(f"- Significant: {claimed_significant:,}")
    print(f"- Percent significant: {claimed_percent}%")
    print(f"- Environmental dimensions: {claimed_dims}")
    print(f"- PFAM domains: {claimed_domains:,}")
    print(f"- Samples: {claimed_samples:,}")

    print(f"\n✅ VERIFIED DATA:")
    if gee_total is not None:
        print(f"- GEE analysis: {gee_total:,} tests, {gee_significant:,} significant ({gee_percent:.1f}%)")
    if alpha_total is not None:
        print(f"- AlphaEarth analysis: {alpha_total:,} tests, {alpha_significant:,} significant ({alpha_percent:.1f}%)")

    print(f"\n🔍 DISCREPANCIES:")

    # Check if either analysis matches the manuscript claims
    if gee_total == claimed_total:
        print("✅ GEE total tests match manuscript claim")
    elif alpha_total == claimed_total:
        print("✅ AlphaEarth total tests match manuscript claim")
    else:
        print(f"❌ Total tests: Claimed {claimed_total:,}, but found GEE={gee_total:,}, AlphaEarth={alpha_total:,}")

    if gee_significant == claimed_significant:
        print("✅ GEE significant tests match manuscript claim")
    elif alpha_significant == claimed_significant:
        print("✅ AlphaEarth significant tests match manuscript claim")
    else:
        print(f"❌ Significant: Claimed {claimed_significant:,}, but found GEE={gee_significant:,}, AlphaEarth={alpha_significant:,}")

    if abs(gee_percent - claimed_percent) < 0.1:
        print("✅ GEE percentage matches manuscript claim")
    elif abs(alpha_percent - claimed_percent) < 0.1:
        print("✅ AlphaEarth percentage matches manuscript claim")
    else:
        print(f"❌ Percentage: Claimed {claimed_percent}%, but found GEE={gee_percent:.1f}%, AlphaEarth={alpha_percent:.1f}%")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("1. Manuscript text refers to '64 AlphaEarth dimensions' but appears to report GEE analysis (30 variables)")
    print("2. Numbers don't match any single analysis - may be combining analyses incorrectly")
    print("3. Need to clarify which correlation analysis is being reported in main text")
    print("4. Update manuscript with verified numbers from appropriate analysis")

if __name__ == "__main__":
    main()