#!/usr/bin/env python3
"""
Bootstrap CI on the count of domains exceeding R2 > 0.3.

Reviewer concern R3#5: The "65 domains (0.7%)" count is a plug-in estimate
that ignores selection bias and CV noise. Bootstrap CIs on the COUNT of
domains exceeding threshold give a realistic range.

Strategy:
  Each domain has 5-fold CV R2 values. In each bootstrap iteration:
    1. Resample the 5 folds with replacement (block bootstrap over folds)
    2. Compute the mean R2 for each domain from the resampled folds
    3. Count how many domains exceed R2 > 0.3
  This captures the variability due to fold assignment.

  Additionally, we do a domain-level bootstrap:
    1. Resample domains with replacement from the 9,989
    2. For each resampled domain, use its observed mean R2
    3. Count how many exceed R2 > 0.3
  This captures sampling variability over domains.

  We report both, with the fold-level bootstrap as primary (captures CV noise).
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd

SCRIPT_PATH = os.path.abspath(__file__)
INPUT_PATH = "/media/drn2/External/TARA-Oceans/MANUSCRIPT/ralph4_statistical_reanalysis/forward_r2_all_pfams.tsv"
OUTPUT_DIR = "/media/drn2/External/TARA-Oceans/MANUSCRIPT/.wt43/task11/source_data"
THRESHOLD = 0.3
N_BOOTSTRAP = 10000
SEED = 42

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH, sep='\t', comment='#')
    print(f"Loaded {len(df)} domains from {INPUT_PATH}")
    print(f"Columns: {list(df.columns)}")

    fold_cols = [c for c in df.columns if c.startswith('r2_fold')]
    print(f"Fold columns: {fold_cols}")
    n_folds = len(fold_cols)

    fold_matrix = df[fold_cols].values  # shape: (n_domains, n_folds)
    r2_means = df['r2_mean'].values

    observed_count = np.sum(r2_means > THRESHOLD)
    observed_pct = 100 * observed_count / len(df)
    print(f"\nObserved: {observed_count} domains exceed R2 > {THRESHOLD} ({observed_pct:.1f}%)")

    rng = np.random.default_rng(SEED)

    # --- Fold-level bootstrap (primary) ---
    fold_boot_counts = np.empty(N_BOOTSTRAP, dtype=int)
    for i in range(N_BOOTSTRAP):
        fold_idx = rng.integers(0, n_folds, size=n_folds)
        boot_means = fold_matrix[:, fold_idx].mean(axis=1)
        fold_boot_counts[i] = np.sum(boot_means > THRESHOLD)

    fold_ci_lo = np.percentile(fold_boot_counts, 2.5)
    fold_ci_hi = np.percentile(fold_boot_counts, 97.5)
    fold_median = np.median(fold_boot_counts)
    print(f"\nFold-level bootstrap (n={N_BOOTSTRAP}):")
    print(f"  Median count: {fold_median:.0f}")
    print(f"  95% CI: [{fold_ci_lo:.0f}, {fold_ci_hi:.0f}]")

    # --- Domain-level bootstrap (secondary) ---
    domain_boot_counts = np.empty(N_BOOTSTRAP, dtype=int)
    n_domains = len(r2_means)
    for i in range(N_BOOTSTRAP):
        boot_idx = rng.integers(0, n_domains, size=n_domains)
        domain_boot_counts[i] = np.sum(r2_means[boot_idx] > THRESHOLD)

    domain_ci_lo = np.percentile(domain_boot_counts, 2.5)
    domain_ci_hi = np.percentile(domain_boot_counts, 97.5)
    domain_median = np.median(domain_boot_counts)
    print(f"\nDomain-level bootstrap (n={N_BOOTSTRAP}):")
    print(f"  Median count: {domain_median:.0f}")
    print(f"  95% CI: [{domain_ci_lo:.0f}, {domain_ci_hi:.0f}]")

    # --- Write summary ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_path = os.path.join(OUTPUT_DIR, "bootstrap_count_ci_20260413.md")
    with open(summary_path, 'w') as f:
        f.write("# Bootstrap CI on Domain Count Exceeding R² > 0.3\n\n")
        f.write("## Provenance\n")
        f.write(f"- **Script:** `{SCRIPT_PATH}`\n")
        f.write(f"- **Input:** `{INPUT_PATH}`\n")
        f.write(f"- **Date:** {timestamp}\n")
        f.write(f"- **Threshold:** R² > {THRESHOLD}\n")
        f.write(f"- **Bootstrap iterations:** {N_BOOTSTRAP:,}\n")
        f.write(f"- **Random seed:** {SEED}\n")
        f.write(f"- **Total domains:** {len(df):,}\n")
        f.write(f"- **CV folds:** {n_folds}\n\n")
        f.write("## Observed Count\n\n")
        f.write(f"- **{observed_count} domains** ({observed_pct:.1f}% of {len(df):,}) exceed R² > {THRESHOLD}\n\n")
        f.write("## Fold-Level Bootstrap (Primary)\n\n")
        f.write("Resamples the 5 CV folds with replacement in each iteration,\n")
        f.write("recomputes per-domain mean R², and counts domains exceeding threshold.\n")
        f.write("Captures variability due to fold assignment in the cross-validation.\n\n")
        f.write(f"- **Median count:** {fold_median:.0f}\n")
        f.write(f"- **95% CI:** [{fold_ci_lo:.0f}, {fold_ci_hi:.0f}]\n")
        f.write(f"- **Mean count:** {np.mean(fold_boot_counts):.1f}\n")
        f.write(f"- **SD:** {np.std(fold_boot_counts):.1f}\n\n")
        f.write("## Domain-Level Bootstrap (Secondary)\n\n")
        f.write("Resamples the 9,989 domains with replacement, using observed mean R².\n")
        f.write("Captures sampling variability over the domain population.\n\n")
        f.write(f"- **Median count:** {domain_median:.0f}\n")
        f.write(f"- **95% CI:** [{domain_ci_lo:.0f}, {domain_ci_hi:.0f}]\n")
        f.write(f"- **Mean count:** {np.mean(domain_boot_counts):.1f}\n")
        f.write(f"- **SD:** {np.std(domain_boot_counts):.1f}\n\n")
        f.write("## Summary for Manuscript\n\n")
        f.write(f"65 domains [fold-bootstrap 95% CI: {fold_ci_lo:.0f}--{fold_ci_hi:.0f}]\n")

    print(f"\nSummary written to: {summary_path}")


if __name__ == '__main__':
    main()
