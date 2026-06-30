#!/usr/bin/env python3
"""
T. pseudonana dark-vs-white dN/dS enrichment analysis.

Uses:
- NCBI GFF3 InterPro annotations (GCF_000149405.2) for domain status
- Published statistics from Koester et al. 2013 (Mol Biol Evol 30:422-434)
  for positive selection gene counts

Provenance:
  Script: analyze_tpseudo_dnds_dark_white_20260531_081500.py
  NCBI proteome: GCF_000149405.2_ASM14940v2
  GFF3 annotation: tpseudo_genomic.gff (downloaded 2026-05-31)
  Published data: Koester et al. 2013, DOI: 10.1093/molbev/mss242
"""

import sys
import re
from collections import defaultdict
from scipy import stats
import numpy as np

# ── Parse InterPro annotation from GFF3 ──
gff_file = "tpseudo_genomic.gff"
protein_interpro = defaultdict(set)
all_proteins = set()
protein_to_locus = {}

with open(gff_file) as f:
    for line in f:
        if line.startswith("#"):
            continue
        fields = line.strip().split("\t")
        if len(fields) < 9 or fields[2] != "CDS":
            continue
        attrs = fields[8]
        prot_match = re.search(r'protein_id=([^;]+)', attrs)
        if not prot_match:
            continue
        protein_id = prot_match.group(1)
        all_proteins.add(protein_id)
        locus_match = re.search(r'locus_tag=([^;]+)', attrs)
        if locus_match:
            protein_to_locus[protein_id] = locus_match.group(1)
        for ipr_match in re.finditer(r'InterPro:(IPR\d+)', attrs):
            protein_interpro[protein_id].add(ipr_match.group(1))

n_total = len(all_proteins)
n_annotated = len(protein_interpro)  # has at least one InterPro domain
n_unannotated = n_total - n_annotated  # "dark" — no recognized domain

print("=" * 70)
print("T. pseudonana dark-vs-white dN/dS enrichment analysis")
print("=" * 70)
print(f"\nNCBI Reference: GCF_000149405.2 (T. pseudonana CCMP 1335)")
print(f"Total protein-coding genes: {n_total}")
print(f"Genes WITH InterPro domain annotation (white): {n_annotated} ({100*n_annotated/n_total:.1f}%)")
print(f"Genes WITHOUT InterPro domain annotation (dark): {n_unannotated} ({100*n_unannotated/n_total:.1f}%)")

# ── Published data from Koester et al. 2013 ──
# Paper reports: 11,390 gene models, 11,355 testable (35 removed for stop codons)
# 809 positively selected genes (Bonferroni), 1,784 (FDR 0.01)
# "Orphan" definition: no BLAST match at E<=1e-5
# 1,718 orphan genes (15% of all), 191 orphans among 809 selected (24%)

koester_total = 11390
koester_testable = 11355
koester_pos_selected = 809
koester_pos_selected_fdr = 1784
koester_orphan_total = 1718  # 15% of all
koester_orphan_in_selected = 191  # 24% of 809
koester_non_orphan_in_selected = koester_pos_selected - koester_orphan_in_selected  # 618
koester_orphan_not_selected = koester_orphan_total - koester_orphan_in_selected  # 1527
koester_non_orphan_total = koester_total - koester_orphan_total  # 9672
koester_non_orphan_not_selected = koester_non_orphan_total - koester_non_orphan_in_selected  # 9054

print("\n" + "=" * 70)
print("Published data from Koester et al. 2013")
print("=" * 70)
print(f"Total gene models: {koester_total}")
print(f"Positively selected (Bonferroni): {koester_pos_selected}")
print(f"Positively selected (FDR 0.01): {koester_pos_selected_fdr}")
print(f"Orphan definition: no BLAST match at E<=1e-5")
print(f"Total orphan genes: {koester_orphan_total} ({100*koester_orphan_total/koester_total:.1f}%)")
print(f"Orphans in positively selected: {koester_orphan_in_selected} ({100*koester_orphan_in_selected/koester_pos_selected:.1f}%)")

# ── Fisher's exact test on published contingency table ──
# Contingency table (orphan definition from paper):
#                    | Positively selected | Not selected |
# Orphan (no BLAST)  |        191          |     1527     |
# Non-orphan (BLAST) |        618          |     9054     |

table_orphan = np.array([
    [koester_orphan_in_selected, koester_orphan_not_selected],
    [koester_non_orphan_in_selected, koester_non_orphan_not_selected]
])

print("\n" + "-" * 70)
print("Contingency table (Koester orphan definition):")
print(f"                      Pos. Selected  Not Selected  Total")
print(f"  Orphan (no BLAST)   {table_orphan[0,0]:>13}  {table_orphan[0,1]:>12}  {table_orphan[0].sum():>5}")
print(f"  Non-orphan (BLAST)  {table_orphan[1,0]:>13}  {table_orphan[1,1]:>12}  {table_orphan[1].sum():>5}")
print(f"  Total               {table_orphan[:,0].sum():>13}  {table_orphan[:,1].sum():>12}  {table_orphan.sum():>5}")

odds_ratio_orphan, p_fisher_orphan = stats.fisher_exact(table_orphan, alternative='greater')
print(f"\nFisher's exact test (one-sided, orphan enrichment in positive selection):")
print(f"  Odds ratio: {odds_ratio_orphan:.3f}")
print(f"  p-value: {p_fisher_orphan:.2e}")

# Enrichment fold
orphan_frac_selected = koester_orphan_in_selected / koester_pos_selected
orphan_frac_genome = koester_orphan_total / koester_total
fold_orphan = orphan_frac_selected / orphan_frac_genome
print(f"  Fold enrichment: {fold_orphan:.2f}× ({orphan_frac_selected*100:.1f}% vs {orphan_frac_genome*100:.1f}%)")

# ── Our dark proteome fraction (InterPro-based) ──
# Compare our InterPro "dark" fraction with Koester's BLAST-based "orphan" fraction
print("\n" + "=" * 70)
print("Dark proteome comparison: our definition vs. Koester's")
print("=" * 70)
print(f"Koester orphan (no BLAST E<=1e-5):    {koester_orphan_total}/{koester_total} = {100*koester_orphan_total/koester_total:.1f}%")
print(f"Our dark (no InterPro domain, NCBI):   {n_unannotated}/{n_total} = {100*n_unannotated/n_total:.1f}%")

# Note: Our dark fraction is based on InterPro domain presence in NCBI annotation.
# This is NOT identical to our standard Pfam E<1e-9 threshold, but InterPro includes
# Pfam entries and provides a comparable (if slightly more inclusive) domain coverage.

# ── Chi-square test on published data ──
chi2, p_chi2, dof, expected = stats.chi2_contingency(table_orphan)
print(f"\nChi-square test on published contingency table:")
print(f"  chi2 = {chi2:.2f}, p = {p_chi2:.2e}, dof = {dof}")

# ── Summary for manuscript ──
print("\n" + "=" * 70)
print("SUMMARY FOR MANUSCRIPT")
print("=" * 70)
print(f"""
T. pseudonana (Koester et al. 2013):
  - {koester_pos_selected} of {koester_total} genes under positive selection (dN/dS > 1, Bonferroni)
  - Orphan genes (no BLAST at E<=1e-5): {orphan_frac_selected*100:.0f}% of positively selected vs {orphan_frac_genome*100:.0f}% genome-wide
  - {fold_orphan:.1f}-fold enrichment (Fisher's exact p = {p_fisher_orphan:.2e})
  - Our NCBI InterPro dark proteome fraction: {100*n_unannotated/n_total:.1f}% (N={n_unannotated})
  - Koester orphan fraction: {100*koester_orphan_total/koester_total:.1f}% (N={koester_orphan_total})
  - Note: metric is between-strain dN/dS (7 strains), not within-population piN/piS
""")

