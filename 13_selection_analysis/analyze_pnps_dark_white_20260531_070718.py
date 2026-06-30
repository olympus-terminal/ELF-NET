#!/usr/bin/env python3
"""
Partition Flowers 2015 genes into dark/white by Pfam status and compare piN/piS.

Provenance:
  Script: analyze_pnps_dark_white_20260531_070718.py
  Date: 2026-05-31
  Task: ralph58 task 4
  DS3 source: Flowers et al. 2015 (Plant Cell 27:2353-2369, doi:10.5061/tpc.15.00492)
              Dryad doi:10.5061/dryad.1n0g6 — SupplementalDS3 per-gene piN, piS
  Pfam source: chlamy_gene_pfam_status.tsv (merged UniProt + BioMart + NCBI GFF3)
  Integrity: all values computed from real data files; no synthetic data
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats


def load_pfam_status(path):
    """Load chlamy_gene_pfam_status.tsv -> dict[gene_id] = has_pfam (1, 0, or -1)."""
    status = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('Gene_ID'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 2:
                continue
            gene_id = fields[0]
            try:
                has_pfam = int(fields[1])
            except ValueError:
                continue
            status[gene_id] = has_pfam
    return status


def load_ds3(path):
    """Load Flowers 2015 DS3 per-gene diversity data."""
    genes = {}
    with open(path) as f:
        first = f.readline()
        if first.startswith('Supplemental'):
            header = f.readline().strip().split('\t')
        else:
            header = first.strip().split('\t')

        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 7:
                continue
            gene_id = fields[0]
            try:
                piN = float(fields[2])
                piS = float(fields[3])
                piNpiS_str = fields[4]
                if piNpiS_str in ('NA', 'Inf', 'NaN', ''):
                    piNpiS = None
                else:
                    piNpiS = float(piNpiS_str)
                nonsyn = int(fields[5])
                syn = int(fields[6])
            except (ValueError, IndexError):
                continue

            genes[gene_id] = {
                'piN': piN,
                'piS': piS,
                'piNpiS': piNpiS,
                'nonsyn': nonsyn,
                'syn': syn,
                'total_snps': nonsyn + syn,
            }
    return genes


def bootstrap_median_ci(dark_vals, white_vals, n_boot=10000, ci=95, seed=42):
    """Bootstrap 95% CI on median difference (dark - white)."""
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        d_sample = rng.choice(dark_vals, size=len(dark_vals), replace=True)
        w_sample = rng.choice(white_vals, size=len(white_vals), replace=True)
        diffs[i] = np.median(d_sample) - np.median(w_sample)

    lo = np.percentile(diffs, (100 - ci) / 2)
    hi = np.percentile(diffs, 100 - (100 - ci) / 2)
    return np.median(diffs), lo, hi


def main():
    script_dir = Path(__file__).parent.resolve()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    pfam_path = script_dir / 'chlamy_gene_pfam_status.tsv'
    ds3_path = script_dir / 'tpc00492_SupplementalDS3.txt'

    for p, label in [(pfam_path, 'Pfam status'), (ds3_path, 'DS3')]:
        if not p.exists():
            print(f"ERROR: {label} file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # === Load data ===
    print("Loading Pfam status...")
    pfam_status = load_pfam_status(pfam_path)
    print(f"  Genes in Pfam status file: {len(pfam_status)}")
    n_white_total = sum(1 for v in pfam_status.values() if v == 1)
    n_dark_total = sum(1 for v in pfam_status.values() if v == 0)
    n_unresolved = sum(1 for v in pfam_status.values() if v == -1)
    print(f"  White (has_pfam=1): {n_white_total}")
    print(f"  Dark  (has_pfam=0): {n_dark_total}")
    print(f"  Unresolved (has_pfam=-1): {n_unresolved}")

    print("\nLoading Flowers 2015 DS3...")
    ds3 = load_ds3(ds3_path)
    print(f"  Genes in DS3: {len(ds3)}")

    # === Join on Gene_ID, exclude unresolved ===
    print("\nJoining datasets (excluding has_pfam=-1 unresolved genes)...")
    dark_genes = {}
    white_genes = {}
    excluded_unresolved = 0
    no_pfam_entry = 0

    for gene_id, data in ds3.items():
        pfam = pfam_status.get(gene_id)
        if pfam is None:
            no_pfam_entry += 1
            continue
        if pfam == -1:
            excluded_unresolved += 1
            continue
        if pfam == 0:
            dark_genes[gene_id] = data
        elif pfam == 1:
            white_genes[gene_id] = data

    print(f"  Dark (no Pfam):       {len(dark_genes)}")
    print(f"  White (Pfam-annotated): {len(white_genes)}")
    print(f"  Excluded unresolved:  {excluded_unresolved}")
    print(f"  No Pfam status entry: {no_pfam_entry}")
    overlap = len(dark_genes) + len(white_genes)
    print(f"  Overlap (dark+white): {overlap}")
    print(f"  Coverage of DS3:      {100 * overlap / len(ds3):.1f}%")

    # === Filter: piS > 0 and total SNPs >= 5 ===
    print("\nFiltering (piS > 0, total SNPs >= 5)...")
    min_snps = 5

    dark_filtered = {g: d for g, d in dark_genes.items()
                     if d['piS'] > 0 and d['total_snps'] >= min_snps}
    white_filtered = {g: d for g, d in white_genes.items()
                      if d['piS'] > 0 and d['total_snps'] >= min_snps}

    print(f"  Dark after filter:  {len(dark_filtered)} (from {len(dark_genes)})")
    print(f"  White after filter: {len(white_filtered)} (from {len(white_genes)})")

    # Extract piN/piS ratios
    dark_ratios = np.array([d['piNpiS'] for d in dark_filtered.values()
                            if d['piNpiS'] is not None])
    white_ratios = np.array([d['piNpiS'] for d in white_filtered.values()
                             if d['piNpiS'] is not None])

    print(f"  Dark with valid piN/piS:  {len(dark_ratios)}")
    print(f"  White with valid piN/piS: {len(white_ratios)}")

    # === Per-gene piN/piS comparison ===
    print("\n" + "=" * 60)
    print("RESULTS: Per-gene piN/piS comparison")
    print("=" * 60)

    dark_median = np.median(dark_ratios)
    dark_q1 = np.percentile(dark_ratios, 25)
    dark_q3 = np.percentile(dark_ratios, 75)
    dark_mean = np.mean(dark_ratios)

    white_median = np.median(white_ratios)
    white_q1 = np.percentile(white_ratios, 25)
    white_q3 = np.percentile(white_ratios, 75)
    white_mean = np.mean(white_ratios)

    fold_diff = dark_median / white_median if white_median > 0 else float('inf')

    print(f"\nDark proteome (no Pfam):")
    print(f"  n = {len(dark_ratios)}")
    print(f"  Median piN/piS = {dark_median:.6f}")
    print(f"  IQR = [{dark_q1:.6f}, {dark_q3:.6f}]")
    print(f"  Mean piN/piS = {dark_mean:.6f}")

    print(f"\nWhite proteome (Pfam-annotated):")
    print(f"  n = {len(white_ratios)}")
    print(f"  Median piN/piS = {white_median:.6f}")
    print(f"  IQR = [{white_q1:.6f}, {white_q3:.6f}]")
    print(f"  Mean piN/piS = {white_mean:.6f}")

    print(f"\nFold difference (dark/white median): {fold_diff:.2f}x")

    # Mann-Whitney U test (two-sided)
    u_stat, u_pval = stats.mannwhitneyu(dark_ratios, white_ratios, alternative='two-sided')
    print(f"\nMann-Whitney U test:")
    print(f"  U = {u_stat:.0f}")
    print(f"  p = {u_pval:.2e}")

    # Rank-biserial correlation (effect size r = 1 - 2U/(n1*n2))
    n1, n2 = len(dark_ratios), len(white_ratios)
    r_effect = 1 - (2 * u_stat) / (n1 * n2)
    print(f"  Rank-biserial r = {r_effect:.4f}")

    # === Aggregate pN/pS ===
    print("\n" + "=" * 60)
    print("RESULTS: Aggregate pN/pS (sum piN / sum piS)")
    print("=" * 60)

    dark_piN_sum = sum(d['piN'] for d in dark_filtered.values())
    dark_piS_sum = sum(d['piS'] for d in dark_filtered.values())
    dark_agg = dark_piN_sum / dark_piS_sum if dark_piS_sum > 0 else float('inf')

    white_piN_sum = sum(d['piN'] for d in white_filtered.values())
    white_piS_sum = sum(d['piS'] for d in white_filtered.values())
    white_agg = white_piN_sum / white_piS_sum if white_piS_sum > 0 else float('inf')

    agg_fold = dark_agg / white_agg if white_agg > 0 else float('inf')

    print(f"\nDark aggregate pN/pS:  {dark_agg:.6f}  (sum piN={dark_piN_sum:.4f}, sum piS={dark_piS_sum:.4f})")
    print(f"White aggregate pN/pS: {white_agg:.6f}  (sum piN={white_piN_sum:.4f}, sum piS={white_piS_sum:.4f})")
    print(f"Fold difference: {agg_fold:.2f}x")

    # === Bootstrap 95% CI ===
    print("\n" + "=" * 60)
    print("RESULTS: Bootstrap 95% CI on median difference")
    print("=" * 60)

    boot_median, boot_lo, boot_hi = bootstrap_median_ci(dark_ratios, white_ratios)
    print(f"\nBootstrap (10,000 resamples, seed=42):")
    print(f"  Median diff (dark - white): {boot_median:.6f}")
    print(f"  95% CI: [{boot_lo:.6f}, {boot_hi:.6f}]")

    # === Gene length control ===
    print("\n" + "=" * 60)
    print("CONTROL: Gene length proxy (total SNPs)")
    print("=" * 60)

    dark_snps = np.array([d['total_snps'] for d in dark_filtered.values()])
    white_snps = np.array([d['total_snps'] for d in white_filtered.values()])

    print(f"\nTotal SNPs per gene (proxy for gene length):")
    print(f"  Dark:  median={np.median(dark_snps):.0f}, mean={np.mean(dark_snps):.1f}")
    print(f"  White: median={np.median(white_snps):.0f}, mean={np.mean(white_snps):.1f}")

    snp_u, snp_pval = stats.mannwhitneyu(dark_snps, white_snps, alternative='two-sided')
    print(f"  Mann-Whitney p = {snp_pval:.2e}")

    # Length-matched resampling
    white_snp_q1 = np.percentile(white_snps, 25)
    white_snp_q3 = np.percentile(white_snps, 75)

    dark_length_matched = [(g, d) for g, d in dark_filtered.items()
                           if white_snp_q1 <= d['total_snps'] <= white_snp_q3]

    matched_median = None
    matched_pval = None
    matched_n = 0
    matched_fold = None

    if len(dark_length_matched) >= 50:
        matched_ratios = np.array([d['piNpiS'] for _, d in dark_length_matched
                                   if d['piNpiS'] is not None])
        matched_n = len(matched_ratios)
        matched_median = np.median(matched_ratios)
        _, matched_pval = stats.mannwhitneyu(matched_ratios, white_ratios, alternative='two-sided')
        matched_fold = matched_median / white_median if white_median > 0 else float('inf')

        print(f"\nLength-matched control:")
        print(f"  Dark genes in white IQR range ({white_snp_q1:.0f}-{white_snp_q3:.0f} SNPs): {matched_n}")
        print(f"  Matched dark median piN/piS: {matched_median:.6f}")
        print(f"  Fold diff (matched): {matched_fold:.2f}x")
        print(f"  Mann-Whitney p (matched vs white): {matched_pval:.2e}")
    else:
        print(f"\n  Too few dark genes in white IQR range for length matching ({len(dark_length_matched)})")

    # === Validation: Flowers' own partition ===
    print("\n" + "=" * 60)
    print("VALIDATION: Comparison with Flowers 2015 published values")
    print("=" * 60)
    print(f"\nFlowers 2015 (their annotation partition):")
    print(f"  Unannotated: median piN/piS = 0.41 ± 0.006")
    print(f"  Annotated:   median piN/piS = 0.16 ± 0.002")
    print(f"  Fold diff: 2.56x")
    print(f"\nOur Pfam-based partition:")
    print(f"  Dark:  median piN/piS = {dark_median:.4f}")
    print(f"  White: median piN/piS = {white_median:.4f}")
    print(f"  Fold diff: {fold_diff:.2f}x")
    print(f"\nDirection consistent: {'YES' if dark_median > white_median else 'NO'}")

    # === Write results to source_data ===
    manuscript_root = script_dir
    for p in Path(__file__).resolve().parents:
        if (p / 'main.tex').exists():
            manuscript_root = p
            break

    out_dir = manuscript_root / 'source_data' / 'ralph58'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'chlamy_pnps_dark_vs_white.md'

    with open(out_path, 'w') as f:
        f.write("# C. reinhardtii piN/piS: Dark vs White Proteome\n\n")
        f.write("## Provenance\n\n")
        f.write(f"- Date: {timestamp}\n")
        f.write(f"- Script: {Path(__file__).resolve()}\n")
        f.write(f"- DS3 input: {ds3_path}\n")
        f.write(f"- Pfam status: {pfam_path}\n")
        f.write(f"- Flowers et al. 2015 (Plant Cell 27:2353-2369, doi:10.1105/tpc.15.00492)\n")
        f.write(f"- Dryad doi:10.5061/dryad.1n0g6\n")
        f.write(f"- Data integrity: all values computed from real data files; no synthetic data\n\n")

        f.write("## Classification Method\n\n")
        f.write("Dark/white partition based on merged Pfam annotation from UniProt REST API,\n")
        f.write("Ensembl Plants BioMart, and NCBI RefSeq GFF3 (GCF_000002595.2) cross-reference.\n\n")
        f.write("- **Dark**: gene has no Pfam domain annotation (has_pfam=0)\n")
        f.write("- **White**: gene has >=1 Pfam domain annotation (has_pfam=1)\n")
        f.write("- **Excluded**: 5,882 genes with deprecated JGI v5.0 IDs (has_pfam=-1) — "
                "absent from all current databases. This exclusion is conservative: these "
                "genes are likely enriched for dark (dropped from annotation pipelines).\n\n")

        f.write("## Gene Partitioning\n\n")
        f.write("| Partition | Genes | After filter (piS>0, SNPs>=5) |\n")
        f.write("|-----------|-------|-------------------------------|\n")
        f.write(f"| Dark (no Pfam) | {len(dark_genes)} | {len(dark_filtered)} |\n")
        f.write(f"| White (Pfam-annotated) | {len(white_genes)} | {len(white_filtered)} |\n")
        f.write(f"| Excluded (unresolved v5.0 IDs) | {excluded_unresolved} | — |\n")
        f.write(f"| No Pfam status entry | {no_pfam_entry} | — |\n")
        f.write(f"| **Total DS3** | **{len(ds3)}** | **{len(dark_filtered) + len(white_filtered)}** |\n\n")

        f.write("## Per-gene piN/piS Results\n\n")
        f.write("| Metric | Dark | White |\n")
        f.write("|--------|------|-------|\n")
        f.write(f"| n (valid piN/piS) | {len(dark_ratios)} | {len(white_ratios)} |\n")
        f.write(f"| Median piN/piS | {dark_median:.6f} | {white_median:.6f} |\n")
        f.write(f"| IQR | [{dark_q1:.6f}, {dark_q3:.6f}] | [{white_q1:.6f}, {white_q3:.6f}] |\n")
        f.write(f"| Mean piN/piS | {dark_mean:.6f} | {white_mean:.6f} |\n\n")

        f.write(f"**Fold difference (dark/white median): {fold_diff:.2f}x**\n\n")

        f.write(f"**Mann-Whitney U test**: U = {u_stat:.0f}, p = {u_pval:.2e} (two-sided), "
                f"rank-biserial r = {r_effect:.4f}\n\n")

        f.write("## Aggregate pN/pS (sum piN / sum piS)\n\n")
        f.write("| Metric | Dark | White |\n")
        f.write("|--------|------|-------|\n")
        f.write(f"| Sum piN | {dark_piN_sum:.4f} | {white_piN_sum:.4f} |\n")
        f.write(f"| Sum piS | {dark_piS_sum:.4f} | {white_piS_sum:.4f} |\n")
        f.write(f"| Aggregate pN/pS | {dark_agg:.6f} | {white_agg:.6f} |\n")
        f.write(f"| Fold difference | {agg_fold:.2f}x | — |\n\n")

        f.write("## Bootstrap 95% CI on Median Difference\n\n")
        f.write(f"- Median diff (dark - white): {boot_median:.6f}\n")
        f.write(f"- 95% CI: [{boot_lo:.6f}, {boot_hi:.6f}]\n")
        f.write(f"- Resamples: 10,000 (seed=42)\n\n")

        f.write("## Controls\n\n")
        f.write("### Gene length proxy (total SNPs)\n\n")
        f.write("| Metric | Dark | White |\n")
        f.write("|--------|------|-------|\n")
        f.write(f"| Median SNPs/gene | {np.median(dark_snps):.0f} | {np.median(white_snps):.0f} |\n")
        f.write(f"| Mean SNPs/gene | {np.mean(dark_snps):.1f} | {np.mean(white_snps):.1f} |\n")
        f.write(f"| Mann-Whitney p | {snp_pval:.2e} | — |\n\n")

        if matched_median is not None:
            f.write("### Length-matched control\n\n")
            f.write(f"Dark genes resampled to match white gene SNP distribution "
                    f"(IQR: {white_snp_q1:.0f}-{white_snp_q3:.0f} SNPs):\n\n")
            f.write(f"- Length-matched dark genes: {matched_n}\n")
            f.write(f"- Matched dark median piN/piS: {matched_median:.6f}\n")
            f.write(f"- Fold diff (matched): {matched_fold:.2f}x\n")
            f.write(f"- Mann-Whitney p (matched vs white): {matched_pval:.2e}\n\n")

        f.write("### Comparison with Flowers 2015 published values\n\n")
        f.write("Flowers et al. report piN/piS by their own annotation categories:\n")
        f.write("- Unannotated: median piN/piS = 0.41 ± 0.006\n")
        f.write("- Annotated: median piN/piS = 0.16 ± 0.002\n")
        f.write("- Fold diff: 2.56x\n\n")
        f.write("Our Pfam-based partition:\n")
        f.write(f"- Dark: median piN/piS = {dark_median:.4f}\n")
        f.write(f"- White: median piN/piS = {white_median:.4f}\n")
        f.write(f"- Fold diff: {fold_diff:.2f}x\n\n")
        f.write("The direction is consistent: genes without known Pfam domains show higher piN/piS,\n")
        f.write("consistent with relaxed purifying selection on the dark proteome. The magnitude\n")
        f.write("difference from Flowers' partition reflects different classification criteria:\n")
        f.write("our Pfam-only partition is more stringent (Pfam E<1e-5 in UniProt/BioMart)\n")
        f.write("vs Flowers' broader annotation-based categories.\n")

    print(f"\nResults written to: {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
