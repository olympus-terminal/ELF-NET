#!/usr/bin/env python3
"""
Dark-vs-white piN/piS analysis for Seminavis robusta.

Input: Source Data from Osuna-Cruz et al. 2020 (Nat Commun 11:3320)
       File: "Suppl. Fig. 20, 22.xlsx" — per-gene piN, piS, piN/piS
       for all 36,254 S. robusta reference genes, with InterPro annotations.

Dark = no InterPro domain annotation ("-" in InterPro_description)
White = at least one InterPro domain annotation

Provenance:
  Script: analyze_seminavis_pnps_dark_white_20260531_080200.py
  Input:  Source_Data/Suppl. Fig. 20, 22.xlsx
  Source: https://doi.org/10.1038/s41467-020-17191-8 (MOESM4)
"""

import os
import sys
from datetime import datetime

import numpy as np
import openpyxl
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "Source_Data", "Suppl. Fig. 20, 22.xlsx")

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: Input file not found: {INPUT_FILE}", file=sys.stderr)
    sys.exit(1)


def load_data(path):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["Data 4"]

    genes = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i <= 1:
            continue
        gene_id, interpro, pan, piN, piS, pnps = row

        is_dark = (interpro == "-" or interpro is None
                   or (isinstance(interpro, str) and interpro.strip() == ""))

        if isinstance(piN, str) or isinstance(piS, str):
            continue
        if piN is None or piS is None:
            continue

        genes.append({
            "gene_id": gene_id,
            "dark": is_dark,
            "pan": pan,
            "piN": float(piN),
            "piS": float(piS),
        })
    wb.close()
    return genes


def analyze(genes):
    dark_piN = np.array([g["piN"] for g in genes if g["dark"]])
    dark_piS = np.array([g["piS"] for g in genes if g["dark"]])
    white_piN = np.array([g["piN"] for g in genes if not g["dark"]])
    white_piS = np.array([g["piS"] for g in genes if not g["dark"]])

    results = {}
    results["n_total_with_data"] = len(genes)
    results["n_dark_total"] = int(np.sum([1 for g in genes if g["dark"]]))
    results["n_white_total"] = int(np.sum([1 for g in genes if not g["dark"]]))

    # Filter: piS > 0 for per-gene piN/piS ratio
    dark_ratio = dark_piN[dark_piS > 0] / dark_piS[dark_piS > 0]
    white_ratio = white_piN[white_piS > 0] / white_piS[white_piS > 0]

    results["n_dark_piS_gt0"] = len(dark_ratio)
    results["n_white_piS_gt0"] = len(white_ratio)

    results["dark_median_pnps"] = float(np.median(dark_ratio))
    results["dark_q25_pnps"] = float(np.percentile(dark_ratio, 25))
    results["dark_q75_pnps"] = float(np.percentile(dark_ratio, 75))
    results["dark_mean_pnps"] = float(np.mean(dark_ratio))

    results["white_median_pnps"] = float(np.median(white_ratio))
    results["white_q25_pnps"] = float(np.percentile(white_ratio, 25))
    results["white_q75_pnps"] = float(np.percentile(white_ratio, 75))
    results["white_mean_pnps"] = float(np.mean(white_ratio))

    results["fold_difference_median"] = results["dark_median_pnps"] / results["white_median_pnps"]

    # Mann-Whitney U test
    mw_stat, mw_p = stats.mannwhitneyu(dark_ratio, white_ratio, alternative="two-sided")
    results["mannwhitney_U"] = float(mw_stat)
    results["mannwhitney_p"] = float(mw_p)

    # Aggregate pN/pS (sum piN / sum piS per group)
    dark_agg_pnps = float(np.sum(dark_piN) / np.sum(dark_piS)) if np.sum(dark_piS) > 0 else float("nan")
    white_agg_pnps = float(np.sum(white_piN) / np.sum(white_piS)) if np.sum(white_piS) > 0 else float("nan")
    results["dark_aggregate_pnps"] = dark_agg_pnps
    results["white_aggregate_pnps"] = white_agg_pnps
    results["fold_difference_aggregate"] = dark_agg_pnps / white_agg_pnps if white_agg_pnps > 0 else float("nan")

    # Bootstrap 95% CI on median difference
    rng = np.random.default_rng(seed=42)
    n_boot = 10000
    boot_diffs = np.empty(n_boot)
    for b in range(n_boot):
        d_idx = rng.integers(0, len(dark_ratio), len(dark_ratio))
        w_idx = rng.integers(0, len(white_ratio), len(white_ratio))
        boot_diffs[b] = np.median(dark_ratio[d_idx]) - np.median(white_ratio[w_idx])
    results["median_diff_observed"] = results["dark_median_pnps"] - results["white_median_pnps"]
    results["median_diff_ci95_lo"] = float(np.percentile(boot_diffs, 2.5))
    results["median_diff_ci95_hi"] = float(np.percentile(boot_diffs, 97.5))

    # Bootstrap 95% CI on fold-difference
    boot_folds = np.empty(n_boot)
    for b in range(n_boot):
        d_idx = rng.integers(0, len(dark_ratio), len(dark_ratio))
        w_idx = rng.integers(0, len(white_ratio), len(white_ratio))
        w_med = np.median(white_ratio[w_idx])
        if w_med > 0:
            boot_folds[b] = np.median(dark_ratio[d_idx]) / w_med
        else:
            boot_folds[b] = float("nan")
    valid_folds = boot_folds[~np.isnan(boot_folds)]
    results["fold_diff_ci95_lo"] = float(np.percentile(valid_folds, 2.5))
    results["fold_diff_ci95_hi"] = float(np.percentile(valid_folds, 97.5))

    return results


def format_results(results):
    lines = []
    lines.append("---")
    lines.append("name: seminavis-pnps-dark-vs-white")
    lines.append("description: Seminavis robusta per-gene piN/piS dark-vs-white comparison")
    lines.append("metadata:")
    lines.append("  type: analysis-result")
    lines.append(f"  date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("  source_paper: Osuna-Cruz et al. 2020, Nat Commun 11:3320")
    lines.append("  source_doi: 10.1038/s41467-020-17191-8")
    lines.append("  source_file: Source_Data/Suppl. Fig. 20, 22.xlsx")
    lines.append(f"  script: {os.path.basename(__file__)}")
    lines.append("  dark_definition: no InterPro domain annotation")
    lines.append("  white_definition: at least one InterPro domain annotation")
    lines.append("  species: Seminavis robusta")
    lines.append("  lineage: Bacillariophyta (pennate diatom)")
    lines.append("  metric: piN/piS (within-species polymorphism)")
    lines.append("  n_strains: 48")
    lines.append("---")
    lines.append("")
    lines.append("# Seminavis robusta: Dark-vs-White piN/piS Comparison")
    lines.append("")
    lines.append("## Data Source")
    lines.append("")
    lines.append("Osuna-Cruz et al. 2020. \"The Seminavis robusta genome provides insights")
    lines.append("into the evolutionary adaptations of benthic diatoms.\" Nature Communications")
    lines.append("11:3320. DOI: 10.1038/s41467-020-17191-8")
    lines.append("")
    lines.append("Per-gene piN and piS computed from resequencing of 48 S. robusta strains.")
    lines.append("Source Data file: Suppl. Fig. 20, 22.xlsx (MOESM4)")
    lines.append("")
    lines.append("## Partitioning")
    lines.append("")
    lines.append("- Dark (unannotated): InterPro_description == \"-\" (no domain hit)")
    lines.append("- White (annotated): any InterPro domain annotation present")
    lines.append("- Note: InterPro integrates Pfam, PANTHER, CDD, SUPERFAMILY, etc.")
    lines.append("  Genes with \"-\" have no hit in ANY member database, which is")
    lines.append("  more stringent than Pfam-only absence.")
    lines.append("")
    lines.append("## Sample Sizes")
    lines.append("")
    lines.append(f"- Total genes with piN/piS data: {results['n_total_with_data']}")
    lines.append(f"- Dark genes (no InterPro): {results['n_dark_total']}")
    lines.append(f"- White genes (has InterPro): {results['n_white_total']}")
    lines.append(f"- Dark genes with piS > 0: {results['n_dark_piS_gt0']}")
    lines.append(f"- White genes with piS > 0: {results['n_white_piS_gt0']}")
    lines.append("")
    lines.append("## Per-Gene piN/piS Distributions")
    lines.append("")
    lines.append(f"- Dark median piN/piS: {results['dark_median_pnps']:.4f}")
    lines.append(f"  (IQR: {results['dark_q25_pnps']:.4f} – {results['dark_q75_pnps']:.4f})")
    lines.append(f"- White median piN/piS: {results['white_median_pnps']:.4f}")
    lines.append(f"  (IQR: {results['white_q25_pnps']:.4f} – {results['white_q75_pnps']:.4f})")
    lines.append(f"- Fold-difference (dark/white median): {results['fold_difference_median']:.2f}x")
    lines.append(f"  (95% bootstrap CI: {results['fold_diff_ci95_lo']:.2f}x – {results['fold_diff_ci95_hi']:.2f}x)")
    lines.append(f"- Dark mean piN/piS: {results['dark_mean_pnps']:.4f}")
    lines.append(f"- White mean piN/piS: {results['white_mean_pnps']:.4f}")
    lines.append("")
    lines.append("## Statistical Test")
    lines.append("")
    lines.append(f"- Mann-Whitney U = {results['mannwhitney_U']:.1f}")
    if results['mannwhitney_p'] == 0.0:
        lines.append(f"- p < 2.2e-308 (below float64 precision)")
    elif results['mannwhitney_p'] < 1e-300:
        lines.append(f"- p < 1e-300")
    else:
        lines.append(f"- p = {results['mannwhitney_p']:.2e}")
    lines.append(f"- Alternative: two-sided")
    lines.append("")
    lines.append("## Aggregate pN/pS (sum piN / sum piS per group)")
    lines.append("")
    lines.append(f"- Dark aggregate pN/pS: {results['dark_aggregate_pnps']:.4f}")
    lines.append(f"- White aggregate pN/pS: {results['white_aggregate_pnps']:.4f}")
    lines.append(f"- Fold-difference (aggregate): {results['fold_difference_aggregate']:.2f}x")
    lines.append("")
    lines.append("## Bootstrap 95% CI on Median Difference")
    lines.append("")
    lines.append(f"- Observed median difference (dark - white): {results['median_diff_observed']:.4f}")
    lines.append(f"- 95% CI: [{results['median_diff_ci95_lo']:.4f}, {results['median_diff_ci95_hi']:.4f}]")
    lines.append(f"- 10,000 bootstrap resamples, seed=42")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Dark (unannotated) genes in S. robusta show elevated piN/piS")
    lines.append("relative to white (annotated) genes, consistent with relaxed")
    lines.append("purifying selection on the dark proteome. This pattern mirrors")
    lines.append("the finding in C. reinhardtii (Flowers et al. 2015; our analysis).")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Loading data from: {INPUT_FILE}")
    genes = load_data(INPUT_FILE)
    print(f"Loaded {len(genes)} genes with numeric piN/piS data")

    print("Running analysis...")
    results = analyze(genes)

    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    output_md = format_results(results)

    out_dir = os.path.join(SCRIPT_DIR, "..", "..", "..", "source_data", "ralph59")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "seminavis_pnps_dark_vs_white.md")
    with open(out_path, "w") as f:
        f.write(output_md)
    print(f"\nResults written to: {os.path.realpath(out_path)}")
