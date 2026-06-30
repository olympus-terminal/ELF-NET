#!/usr/bin/env python3
"""
Verify and extend T. pseudonana dark-vs-white positive selection analysis.

Re-computes from raw data files (Koester Table S2 + UniProt Pfam mapping).

Provenance:
  Script: verify_tpseudo_analysis_20260531_082500.py
  Input 1: koester2013_table_s2_with_pfam.tsv (merged Koester S2 + UniProt Pfam)
  Input 2: supp_mss242_Table_S2.xlsx (original supplementary table)
  Input 3: tpseudo_uniprot_pfam.tsv (UniProt proteome UP000001449 Pfam annotations)
  Date: 2026-05-31
"""

import sys
import os
import numpy as np
from scipy import stats
from collections import Counter

# ── Load merged data ──
data_file = "koester2013_table_s2_with_pfam.tsv"
if not os.path.exists(data_file):
    print(f"ERROR: {data_file} not found", file=sys.stderr)
    sys.exit(1)

genes = []
with open(data_file) as f:
    header = f.readline().strip().split("\t")
    for line in f:
        fields = line.strip().split("\t")
        if len(fields) < 12:
            continue
        row = dict(zip(header, fields))
        try:
            row["lrt_float"] = float(row["lrt"]) if row["lrt"] else None
            row["pvalue_float"] = float(row["pvalue"]) if row["pvalue"] else None
            row["bonf_int"] = int(row["bonf_sig"]) if row["bonf_sig"] else 0
            row["fdr_int"] = int(row["fdr_sig"]) if row["fdr_sig"] else 0
            row["is_orphan"] = row.get("orphan", "N") == "Y"
            row["is_dark"] = row.get("pfam_status", "") == "dark"
        except (ValueError, KeyError):
            continue
        genes.append(row)

print(f"Loaded {len(genes)} genes from {data_file}")
print(f"Columns: {header}")

# ── Basic counts ──
n_total = len(genes)
n_dark = sum(1 for g in genes if g["is_dark"])
n_white = n_total - n_dark
n_orphan = sum(1 for g in genes if g["is_orphan"])
n_bonf = sum(1 for g in genes if g["bonf_int"] == 1)
n_fdr = sum(1 for g in genes if g["fdr_int"] == 1)

print(f"\n{'='*70}")
print("GENE COUNTS")
print(f"{'='*70}")
print(f"Total genes in Table S2: {n_total}")
print(f"Dark (no Pfam): {n_dark} ({100*n_dark/n_total:.1f}%)")
print(f"White (≥1 Pfam): {n_white} ({100*n_white/n_total:.1f}%)")
print(f"Orphan (Koester def): {n_orphan} ({100*n_orphan/n_total:.1f}%)")
print(f"Bonferroni significant: {n_bonf}")
print(f"FDR significant: {n_fdr}")

# ── Enrichment: Bonferroni ──
dark_bonf = sum(1 for g in genes if g["is_dark"] and g["bonf_int"] == 1)
white_bonf = sum(1 for g in genes if not g["is_dark"] and g["bonf_int"] == 1)
dark_not_bonf = n_dark - dark_bonf
white_not_bonf = n_white - white_bonf

print(f"\n{'='*70}")
print("ENRICHMENT: BONFERRONI (p ≤ 1.5e-5)")
print(f"{'='*70}")
print(f"                 Pos.Selected  Not Selected  Total")
print(f"  Dark           {dark_bonf:>12}  {dark_not_bonf:>12}  {n_dark:>5}")
print(f"  White          {white_bonf:>12}  {white_not_bonf:>12}  {n_white:>5}")
print(f"  Total          {n_bonf:>12}  {n_total-n_bonf:>12}  {n_total:>5}")

table_bonf = np.array([[dark_bonf, dark_not_bonf],
                        [white_bonf, white_not_bonf]])
or_bonf, p_bonf = stats.fisher_exact(table_bonf, alternative='greater')
chi2_bonf, p_chi2_bonf, _, _ = stats.chi2_contingency(table_bonf)

frac_dark_selected = dark_bonf / n_bonf if n_bonf > 0 else 0
frac_dark_genome = n_dark / n_total
fold_dark = frac_dark_selected / frac_dark_genome if frac_dark_genome > 0 else 0

print(f"\nDark fraction among selected: {frac_dark_selected:.3f}")
print(f"Dark fraction genome-wide: {frac_dark_genome:.3f}")
print(f"Fold enrichment: {fold_dark:.2f}×")
print(f"Fisher's exact OR: {or_bonf:.3f} (p = {p_bonf:.2e})")
print(f"Chi-squared: {chi2_bonf:.2f} (p = {p_chi2_bonf:.2e})")

# Rate of positive selection per group
rate_dark = dark_bonf / n_dark if n_dark > 0 else 0
rate_white = white_bonf / n_white if n_white > 0 else 0
print(f"\nPositive selection rate:")
print(f"  Dark: {dark_bonf}/{n_dark} = {100*rate_dark:.1f}%")
print(f"  White: {white_bonf}/{n_white} = {100*rate_white:.1f}%")
print(f"  Fold diff in rate: {rate_dark/rate_white:.2f}×" if rate_white > 0 else "")

# ── Enrichment: FDR ──
dark_fdr = sum(1 for g in genes if g["is_dark"] and g["fdr_int"] == 1)
white_fdr = sum(1 for g in genes if not g["is_dark"] and g["fdr_int"] == 1)
dark_not_fdr = n_dark - dark_fdr
white_not_fdr = n_white - white_fdr

print(f"\n{'='*70}")
print("ENRICHMENT: FDR (q ≤ 0.01)")
print(f"{'='*70}")
print(f"                 Pos.Selected  Not Selected  Total")
print(f"  Dark           {dark_fdr:>12}  {dark_not_fdr:>12}  {n_dark:>5}")
print(f"  White          {white_fdr:>12}  {white_not_fdr:>12}  {n_white:>5}")

table_fdr = np.array([[dark_fdr, dark_not_fdr],
                       [white_fdr, white_not_fdr]])
or_fdr, p_fdr = stats.fisher_exact(table_fdr, alternative='greater')
print(f"\nFisher's exact OR: {or_fdr:.3f} (p = {p_fdr:.2e})")

# ── LRT distribution comparison ──
dark_lrt = [g["lrt_float"] for g in genes if g["is_dark"] and g["lrt_float"] is not None]
white_lrt = [g["lrt_float"] for g in genes if not g["is_dark"] and g["lrt_float"] is not None]

print(f"\n{'='*70}")
print("LRT DISTRIBUTION (all genes in Table S2)")
print(f"{'='*70}")
print(f"                n     Median     IQR")
print(f"  Dark          {len(dark_lrt):<5} {np.median(dark_lrt):.3f}     [{np.percentile(dark_lrt,25):.3f}, {np.percentile(dark_lrt,75):.3f}]")
print(f"  White         {len(white_lrt):<5} {np.median(white_lrt):.3f}     [{np.percentile(white_lrt,25):.3f}, {np.percentile(white_lrt,75):.3f}]")

u_stat, p_mw = stats.mannwhitneyu(dark_lrt, white_lrt, alternative='greater')
fold_lrt = np.median(dark_lrt) / np.median(white_lrt) if np.median(white_lrt) > 0 else float('inf')
print(f"\nMedian fold difference: {fold_lrt:.2f}×")
print(f"Mann-Whitney U = {u_stat:.0f} (p = {p_mw:.2e}, one-sided dark > white)")

# ── Orphan vs Dark cross-tabulation ──
orphan_dark = sum(1 for g in genes if g["is_orphan"] and g["is_dark"])
orphan_white = sum(1 for g in genes if g["is_orphan"] and not g["is_dark"])
non_orphan_dark = sum(1 for g in genes if not g["is_orphan"] and g["is_dark"])
non_orphan_white = sum(1 for g in genes if not g["is_orphan"] and not g["is_dark"])

print(f"\n{'='*70}")
print("ORPHAN vs DARK CROSS-TABULATION")
print(f"{'='*70}")
print(f"                 Dark    White   Total")
print(f"  Orphan         {orphan_dark:>5}   {orphan_white:>5}   {orphan_dark+orphan_white:>5}")
print(f"  Non-orphan     {non_orphan_dark:>5}   {non_orphan_white:>5}   {non_orphan_dark+non_orphan_white:>5}")
if orphan_dark + orphan_white > 0:
    print(f"\n  {100*orphan_dark/(orphan_dark+orphan_white):.1f}% of orphans are dark")
if n_dark > 0:
    print(f"  {100*orphan_dark/n_dark:.1f}% of dark genes are orphan")

# ── Genome-wide context (from UniProt) ──
uniprot_file = "tpseudo_uniprot_pfam.tsv"
if os.path.exists(uniprot_file):
    n_uniprot_total = 0
    n_uniprot_pfam = 0
    with open(uniprot_file) as f:
        f.readline()  # skip header
        for line in f:
            fields = line.strip().split("\t")
            n_uniprot_total += 1
            if len(fields) >= 4 and fields[3].strip():
                n_uniprot_pfam += 1
    n_uniprot_dark = n_uniprot_total - n_uniprot_pfam
    print(f"\n{'='*70}")
    print("GENOME-WIDE CONTEXT (UniProt UP000001449)")
    print(f"{'='*70}")
    print(f"Total UniProt proteins: {n_uniprot_total}")
    print(f"With Pfam: {n_uniprot_pfam} ({100*n_uniprot_pfam/n_uniprot_total:.1f}%)")
    print(f"Without Pfam (dark): {n_uniprot_dark} ({100*n_uniprot_dark/n_uniprot_total:.1f}%)")

    # Now compute enrichment using genome-wide dark fraction
    dark_frac_genome = n_uniprot_dark / n_uniprot_total
    dark_frac_selected = dark_bonf / n_bonf if n_bonf > 0 else 0
    
    # Build full genome contingency table
    # Scale: Table S2 has 3,280 genes with p<0.05 from 11,390 total
    # UniProt has 11,612 genes
    # Bonferroni selected: 809 from 11,390
    # We know 414 dark in selected, n_dark(genome) from UniProt
    total_genome = n_uniprot_total
    total_dark_genome = n_uniprot_dark
    total_white_genome = n_uniprot_pfam
    
    # Enrichment using genome-wide denominators
    # But we can only use Table S2 for selected/not-selected counts
    # since genome-wide coverage matches

    # Fisher test with genome-wide denominators
    # selected: dark_bonf, white_bonf
    # not-selected: total_dark_genome - dark_bonf, total_white_genome - white_bonf
    dark_not_sel_genome = total_dark_genome - dark_bonf
    white_not_sel_genome = total_white_genome - white_bonf
    
    if dark_not_sel_genome > 0 and white_not_sel_genome > 0:
        table_genome = np.array([[dark_bonf, dark_not_sel_genome],
                                  [white_bonf, white_not_sel_genome]])
        or_genome, p_genome = stats.fisher_exact(table_genome, alternative='greater')
        
        fold_genome = (dark_bonf / total_dark_genome) / (white_bonf / total_white_genome) if white_bonf > 0 and total_white_genome > 0 else float('inf')
        
        print(f"\nGenome-wide enrichment test (dark selected vs genome-wide dark fraction):")
        print(f"  Dark fraction genome-wide: {100*dark_frac_genome:.1f}%")
        print(f"  Dark fraction among selected: {100*dark_frac_selected:.1f}%")
        print(f"  Dark pos.sel. rate: {100*dark_bonf/total_dark_genome:.1f}%")
        print(f"  White pos.sel. rate: {100*white_bonf/total_white_genome:.1f}%")
        print(f"  Rate fold-diff: {fold_genome:.2f}×")
        print(f"  Fisher's exact OR: {or_genome:.3f} (p = {p_genome:.2e})")

# ── Bootstrap CI on enrichment ──
print(f"\n{'='*70}")
print("BOOTSTRAP 95% CI ON DARK ENRICHMENT FOLD")
print(f"{'='*70}")
rng = np.random.default_rng(42)
n_boot = 10000
is_dark_arr = np.array([1 if g["is_dark"] else 0 for g in genes])
is_bonf_arr = np.array([1 if g["bonf_int"] == 1 else 0 for g in genes])

boot_folds = []
for _ in range(n_boot):
    idx = rng.integers(0, n_total, size=n_total)
    dark_b = is_dark_arr[idx]
    bonf_b = is_bonf_arr[idx]
    n_sel_b = bonf_b.sum()
    if n_sel_b == 0:
        continue
    dark_sel_b = (dark_b * bonf_b).sum()
    dark_frac_sel = dark_sel_b / n_sel_b
    dark_frac_all = dark_b.sum() / len(dark_b)
    if dark_frac_all == 0:
        continue
    boot_folds.append(dark_frac_sel / dark_frac_all)

boot_folds = np.array(boot_folds)
ci_lo = np.percentile(boot_folds, 2.5)
ci_hi = np.percentile(boot_folds, 97.5)
print(f"Observed fold: {fold_dark:.3f}")
print(f"Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"Resamples: {len(boot_folds)} (seed=42)")

# ── Final summary ──
print(f"\n{'='*70}")
print("SUMMARY FOR MANUSCRIPT")
print(f"{'='*70}")
print(f"""
T. pseudonana (Koester et al. 2013, Mol Biol Evol 30:422–434):
  Species: Thalassiosira pseudonana CCMP 1335 (centric diatom)
  Metric: per-gene dN/dS site test (PAML M8a vs M8), 7 strains
  
  Dark (no Pfam domain): {n_dark} genes ({100*n_dark/n_total:.1f}% of tested)
  White (≥1 Pfam domain): {n_white} genes ({100*n_white/n_total:.1f}% of tested)
  
  Positively selected (Bonferroni): {n_bonf} genes
    Dark selected: {dark_bonf} ({100*dark_bonf/n_bonf:.1f}% of selected)
    White selected: {white_bonf} ({100*white_bonf/n_bonf:.1f}% of selected)
  
  Dark enrichment fold: {fold_dark:.2f}× (95% CI: [{ci_lo:.2f}, {ci_hi:.2f}])
  Fisher's exact OR: {or_bonf:.3f} (p = {p_bonf:.2e})
  
  LRT: dark median {np.median(dark_lrt):.3f} vs white {np.median(white_lrt):.3f}
       ({fold_lrt:.2f}×, Mann-Whitney p = {p_mw:.2e})
  
  Note: dN/dS (divergence across 7 strains), not piN/piS (within-population)
""")

