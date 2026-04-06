#!/usr/bin/env python3
"""
Step 9: E-value Sensitivity Analysis for Novel Domain Pipeline

Provenance:
    Script: scripts/novel_domains/09_evalue_sensitivity.py
    Generated: 2026-03-20

Purpose:
    Address reviewer concern about asymmetric E-value thresholds:
    Pfam search uses E < 10^-9 while novel HMM search uses E < 10^-5.
    Post-filter existing .tbl files at E < 1e-7 and E < 1e-9 and
    recompute environmental correlations to test whether the 2.3-fold
    enrichment of novel domain coupling persists at matched stringency.

    Key insight: The .tbl files from Step 6 already contain E-values
    for every hit (hmmsearch was run with -E 1e-5). We post-filter
    without re-running hmmsearch.

Input:
    - 03_analyses/novel_domains/novel_hits/{sample}.novel.tbl (from Step 6)
    - 03_analyses/novel_domains/sample_list.txt
    - 03_analyses/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv

Output:
    - 03_analyses/novel_domains/results/sensitivity/sensitivity_comparison.tsv
    - 03_analyses/novel_domains/results/sensitivity/per_threshold_details/

Usage:
    python3 scripts/novel_domains/09_evalue_sensitivity.py
"""

import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas required. Install with: pip install pandas")
    sys.exit(1)

try:
    from scipy.stats import spearmanr
except ImportError:
    print("ERROR: scipy required. Install with: pip install scipy")
    sys.exit(1)

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None
    print("WARNING: statsmodels not available, FDR correction will use Bonferroni")

warnings.filterwarnings("ignore")

BASE = Path("/scratch/drn2/PROJECTS/TARA-LA4SR")
HITS_DIR = BASE / "03_analyses/novel_domains/novel_hits"
SAMPLE_LIST = BASE / "03_analyses/novel_domains/sample_list.txt"
GEE_MERGED = BASE / "03_analyses/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
RESULTS_DIR = BASE / "03_analyses/novel_domains/results"
OUT_DIR = RESULTS_DIR / "sensitivity"
DETAIL_DIR = OUT_DIR / "per_threshold_details"

MIN_PREVALENCE = 10

THRESHOLDS = [1e-5, 1e-7, 1e-9]

ENV_COLS = [
    "salinity_psu_est", "air_temp_mean_c", "air_temp_max_c", "air_temp_min_c",
    "air_temp_range_c", "precip_mean_mm", "solar_rad_mj_m2", "bathymetry_m",
    "distance_to_coast_km", "sst_mean_c", "sst_max_c", "sst_min_c",
    "sst_range_c", "chl_mean_mg_m3", "chl_max_mg_m3", "chl_min_mg_m3",
    "nflh_mean", "poc_mean_mg_m3", "modis_sst_mean_c",
]

def clr_transform(count_matrix):
    """Centered log-ratio transform for compositional data."""
    X = count_matrix + 1.0
    log_X = np.log(X)
    geo_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geo_mean

def parse_tblout_multi_threshold(tbl_path, thresholds):
    """Parse hmmsearch tblout, return dict of {threshold: {domain: count}}.

    Reads the file once and bins hits into multiple threshold buckets.
    tblout format: target_name(0) target_acc(1) query_name(2) query_acc(3)
                   E-value(4) score(5) bias(6) ...
    """
    counts = {t: defaultdict(int) for t in thresholds}
    with open(tbl_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                evalue = float(fields[4])
            except ValueError:
                continue
            domain = fields[2]
            for t in thresholds:
                if evalue <= t:
                    counts[t][domain] += 1
    return {t: dict(counts[t]) for t in thresholds}

def build_count_matrices(samples, thresholds):
    """Build count matrices for all thresholds in a single pass."""
    # {threshold: {sample: {domain: count}}}
    all_counts = {t: {} for t in thresholds}
    all_domains = {t: set() for t in thresholds}
    n_found = 0
    n_missing = 0

    for i, sample in enumerate(samples):
        tbl_path = HITS_DIR / f"{sample}.novel.tbl"
        if tbl_path.exists():
            tc = parse_tblout_multi_threshold(tbl_path, thresholds)
            for t in thresholds:
                all_counts[t][sample] = tc[t]
                all_domains[t].update(tc[t].keys())
            n_found += 1
        else:
            for t in thresholds:
                all_counts[t][sample] = {}
            n_missing += 1

        if (i + 1) % 200 == 0:
            print(f"    Parsed {i + 1}/{len(samples)} samples...")

    print(f"  Samples with .tbl files: {n_found}")
    print(f"  Samples missing .tbl:    {n_missing}")

    # Build DataFrames
    matrices = {}
    for t in thresholds:
        domain_list = sorted(all_domains[t])
        rows = []
        for sample in samples:
            sc = all_counts[t][sample]
            rows.append([sc.get(d, 0) for d in domain_list])
        df = pd.DataFrame(rows, index=samples, columns=domain_list)
        # Strip .aa suffix
        df.index = df.index.str.replace(r"\.aa$", "", regex=True)
        matrices[t] = df
        print(f"  Threshold {t:.0e}: {df.shape[0]} samples x {df.shape[1]} domains, "
              f"total hits = {df.values.sum():,}")

    return matrices

def filter_by_prevalence(df, min_prev=MIN_PREVALENCE):
    """Filter domains by sample prevalence."""
    prevalence = (df > 0).sum(axis=0)
    prevalent = prevalence[prevalence >= min_prev].index
    return df[prevalent]

def run_correlations(domain_df, env_df):
    """Run Spearman correlations between domains and environmental variables.

    Returns results DataFrame with columns:
        domain, env_variable, rho, p_value, n_samples, q_value, significant
    """
    results = []

    shared = domain_df.index.intersection(env_df.index)
    if len(shared) < 20:
        print(f"  WARNING: Only {len(shared)} shared samples, need >=20")
        return pd.DataFrame()

    clr_matrix = clr_transform(domain_df.loc[shared].values)
    clr_df = pd.DataFrame(clr_matrix, index=shared, columns=domain_df.columns)

    env_vars = list(env_df.columns)
    domains = list(clr_df.columns)

    print(f"  Shared samples: {len(shared)}")
    print(f"  Testing {len(domains)} domains x {len(env_vars)} env variables = "
          f"{len(domains) * len(env_vars):,} tests")

    for env_var in env_vars:
        env_vals = env_df.loc[shared, env_var].values
        valid = ~np.isnan(env_vals)
        if valid.sum() < 20:
            continue
        ev = env_vals[valid]
        if np.std(ev) < 1e-10:
            continue

        for domain in domains:
            dv = clr_df.loc[shared, domain].values[valid]
            rho, pval = spearmanr(dv, ev)
            if not np.isnan(rho):
                results.append({
                    "domain": domain,
                    "env_variable": env_var,
                    "rho": rho,
                    "p_value": pval,
                    "n_samples": int(valid.sum()),
                })

    if not results:
        print("  WARNING: No valid correlations computed")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    if multipletests is not None:
        reject, qvals, _, _ = multipletests(
            results_df["p_value"].values, method="fdr_bh"
        )
        results_df["q_value"] = qvals
        results_df["significant"] = reject
    else:
        n_tests = len(results_df)
        results_df["q_value"] = np.minimum(results_df["p_value"] * n_tests, 1.0)
        results_df["significant"] = results_df["q_value"] < 0.05

    results_df = results_df.sort_values("p_value")

    n_sig = results_df["significant"].sum()
    pct_sig = 100.0 * n_sig / len(results_df) if len(results_df) > 0 else 0
    print(f"  Significant (FDR<0.05): {n_sig:,} / {len(results_df):,} ({pct_sig:.1f}%)")

    return results_df

def load_gee_env():
    """Load GEE environmental variables."""
    if not GEE_MERGED.exists():
        print(f"ERROR: {GEE_MERGED} not found")
        sys.exit(1)

    df = pd.read_csv(GEE_MERGED, sep="\t", comment="#")
    df = df.set_index("assembly_id")
    env_available = [c for c in ENV_COLS if c in df.columns]
    env_df = df[env_available].apply(pd.to_numeric, errors="coerce")
    print(f"  GEE env variables: {len(env_available)} across {env_df.shape[0]} samples")
    return env_df

def load_pfam_and_correlate(env_df):
    """Load Pfam data and compute baseline correlations."""
    df = pd.read_csv(GEE_MERGED, sep="\t", comment="#")
    df = df.set_index("assembly_id")

    pfam_cols = [c for c in df.columns if c.startswith("PF")]
    pfam_df = df[pfam_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Filter by prevalence
    prevalence = (pfam_df > 0).sum(axis=0)
    prevalent = prevalence[prevalence >= MIN_PREVALENCE].index
    pfam_df = pfam_df[prevalent]
    print(f"  Pfam domains (prevalence>={MIN_PREVALENCE}): {pfam_df.shape[1]}")

    print("  Running Pfam baseline correlations...")
    pfam_corr = run_correlations(pfam_df, env_df)
    return pfam_corr

def main():
    print("=" * 70)
    print("  Step 9: E-value Sensitivity Analysis")
    print("  Thresholds: " + ", ".join(f"{t:.0e}" for t in THRESHOLDS))
    print("=" * 70)
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)

    # Load sample list
    with open(SAMPLE_LIST) as f:
        samples = [line.strip() for line in f if line.strip()]
    print(f"  Samples in list: {len(samples)}")
    print()

    # Load GEE env
    print("Loading GEE environmental data...")
    env_df = load_gee_env()
    print()

    # Build count matrices for all thresholds in a single pass
    print("Parsing .tbl files (single pass, three thresholds)...")
    matrices = build_count_matrices(samples, THRESHOLDS)
    print()

    # Pfam baseline
    print("=" * 50)
    print("  Pfam Baseline Correlations")
    print("=" * 50)
    pfam_corr = load_pfam_and_correlate(env_df)
    pfam_abs_rho = pfam_corr["rho"].abs() if len(pfam_corr) > 0 else pd.Series(dtype=float)
    pfam_n_sig = pfam_corr["significant"].sum() if len(pfam_corr) > 0 else 0
    pfam_pct_sig = 100.0 * pfam_n_sig / len(pfam_corr) if len(pfam_corr) > 0 else 0
    print()

    # Run correlations for each threshold
    summary_rows = []

    # Pfam baseline row
    summary_rows.append({
        "threshold": "Pfam_1e-9",
        "n_domains_total": pfam_corr["domain"].nunique() if len(pfam_corr) > 0 else 0,
        "n_domains_prevalent": pfam_corr["domain"].nunique() if len(pfam_corr) > 0 else 0,
        "n_correlations": len(pfam_corr),
        "n_significant": int(pfam_n_sig),
        "pct_significant": round(pfam_pct_sig, 2),
        "median_abs_rho": round(pfam_abs_rho.median(), 4) if len(pfam_abs_rho) > 0 else np.nan,
        "mean_abs_rho": round(pfam_abs_rho.mean(), 4) if len(pfam_abs_rho) > 0 else np.nan,
        "fold_vs_pfam_median": 1.0,
        "fold_vs_pfam_pct_sig": 1.0,
    })

    pfam_median = pfam_abs_rho.median() if len(pfam_abs_rho) > 0 else 1.0

    for t in THRESHOLDS:
        t_label = f"Novel_{t:.0e}"
        print("=" * 50)
        print(f"  Novel Domains at E < {t:.0e}")
        print("=" * 50)

        raw_df = matrices[t]
        n_total = raw_df.shape[1]

        # Filter by prevalence
        filt_df = filter_by_prevalence(raw_df)
        n_prevalent = filt_df.shape[1]
        print(f"  Total domains with hits: {n_total}")
        print(f"  Domains passing prevalence filter (>={MIN_PREVALENCE}): {n_prevalent}")

        if n_prevalent == 0:
            print("  No domains pass prevalence filter, skipping.")
            summary_rows.append({
                "threshold": t_label,
                "n_domains_total": n_total,
                "n_domains_prevalent": 0,
                "n_correlations": 0,
                "n_significant": 0,
                "pct_significant": 0.0,
                "median_abs_rho": np.nan,
                "mean_abs_rho": np.nan,
                "fold_vs_pfam_median": np.nan,
                "fold_vs_pfam_pct_sig": np.nan,
            })
            continue

        # Run correlations
        corr_df = run_correlations(filt_df, env_df)

        if len(corr_df) > 0:
            abs_rho = corr_df["rho"].abs()
            n_sig = int(corr_df["significant"].sum())
            pct_sig = 100.0 * n_sig / len(corr_df)

            summary_rows.append({
                "threshold": t_label,
                "n_domains_total": n_total,
                "n_domains_prevalent": n_prevalent,
                "n_correlations": len(corr_df),
                "n_significant": n_sig,
                "pct_significant": round(pct_sig, 2),
                "median_abs_rho": round(abs_rho.median(), 4),
                "mean_abs_rho": round(abs_rho.mean(), 4),
                "fold_vs_pfam_median": round(abs_rho.median() / pfam_median, 2) if pfam_median > 0 else np.nan,
                "fold_vs_pfam_pct_sig": round(pct_sig / pfam_pct_sig, 2) if pfam_pct_sig > 0 else np.nan,
            })

            # Save per-threshold details
            detail_path = DETAIL_DIR / f"correlations_{t:.0e}.tsv"
            corr_df.to_csv(detail_path, sep="\t", index=False)
            print(f"  Written: {detail_path}")
        else:
            summary_rows.append({
                "threshold": t_label,
                "n_domains_total": n_total,
                "n_domains_prevalent": n_prevalent,
                "n_correlations": 0,
                "n_significant": 0,
                "pct_significant": 0.0,
                "median_abs_rho": np.nan,
                "mean_abs_rho": np.nan,
                "fold_vs_pfam_median": np.nan,
                "fold_vs_pfam_pct_sig": np.nan,
            })
        print()

    # Write summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "sensitivity_comparison.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print("=" * 70)
    print("  Summary written to:", summary_path)
    print("=" * 70)
    print()
    print(summary_df.to_string(index=False))
    print()

    # Also save Pfam correlations for figure script
    if len(pfam_corr) > 0:
        pfam_detail_path = DETAIL_DIR / "correlations_pfam_baseline.tsv"
        pfam_corr.to_csv(pfam_detail_path, sep="\t", index=False)
        print(f"  Pfam baseline correlations: {pfam_detail_path}")

    print()
    print("  Done.")

if __name__ == "__main__":
    main()
