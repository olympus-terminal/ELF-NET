#!/usr/bin/env python3
"""
Cross-validate T. pseudonana dark/white partition using hmmsearch (Pfam-A, E<1e-9)
against the UniProt-based partition used in the primary analysis.

Provenance:
  Script: crossval_hmmsearch_pfam_20260531_083500.py
  hmmsearch input: tpseudo_ncbi_pfam.domtbl (Pfam-A vs NCBI GCF_000149405.2 proteome)
  UniProt partition: tpseudo_uniprot_pfam.tsv
  Koester data: koester2013_table_s2_with_pfam.tsv
  Date: 2026-05-31
"""

import sys
import os
import re
from collections import defaultdict

DOMTBL = "tpseudo_ncbi_pfam.domtbl"
PROTEIN_FAA = "tpseudo_protein.faa"
GFF = "tpseudo_genomic.gff"
KOESTER = "koester2013_table_s2_with_pfam.tsv"
E_THRESHOLD = 1e-9

if not os.path.exists(DOMTBL):
    print(f"ERROR: {DOMTBL} not found — hmmsearch not yet complete", file=sys.stderr)
    sys.exit(1)

# ── Parse hmmsearch domtbl ──
pfam_hits = defaultdict(set)
with open(DOMTBL) as f:
    for line in f:
        if line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 22:
            continue
        protein_id = fields[0]
        pfam_acc = fields[3]
        try:
            i_evalue = float(fields[12])
        except ValueError:
            continue
        if i_evalue < E_THRESHOLD:
            pfam_hits[protein_id].add(pfam_acc)

# ── Count all proteins from FASTA ──
all_prots = set()
with open(PROTEIN_FAA) as f:
    for line in f:
        if line.startswith(">"):
            prot_id = line.split()[0][1:]
            all_prots.add(prot_id)

n_total = len(all_prots)
n_with_pfam = len(pfam_hits)
n_dark = n_total - n_with_pfam

print(f"hmmsearch Pfam partition (E < {E_THRESHOLD}):")
print(f"  Total proteins: {n_total}")
print(f"  White (≥1 Pfam hit): {n_with_pfam} ({100*n_with_pfam/n_total:.1f}%)")
print(f"  Dark (no Pfam hit): {n_dark} ({100*n_dark/n_total:.1f}%)")
print(f"  Unique Pfam domains: {len(set().union(*pfam_hits.values()) if pfam_hits else set())}")

# ── Map NCBI protein IDs to THAPSDRAFT locus tags via GFF3 ──
prot_to_locus = {}
with open(GFF) as f:
    for line in f:
        if line.startswith("#"):
            continue
        fields = line.strip().split("\t")
        if len(fields) < 9 or fields[2] != "CDS":
            continue
        attrs = fields[8]
        prot_match = re.search(r'protein_id=([^;]+)', attrs)
        locus_match = re.search(r'locus_tag=([^;]+)', attrs)
        if prot_match and locus_match:
            prot_to_locus[prot_match.group(1)] = locus_match.group(1)

# ── Build hmmsearch-based dark/white by locus tag ──
locus_dark = set()
locus_white = set()
for prot_id in all_prots:
    locus = prot_to_locus.get(prot_id, prot_id)
    if prot_id in pfam_hits:
        locus_white.add(locus)
    else:
        locus_dark.add(locus)

# ── Compare with UniProt partition ──
print(f"\n--- Comparison with UniProt Pfam partition ---")

# Load Koester data with UniProt-based Pfam status
koester_genes = []
with open(KOESTER) as f:
    header = f.readline().strip().split("\t")
    for line in f:
        fields = line.strip().split("\t")
        if len(fields) >= 12:
            row = dict(zip(header, fields))
            koester_genes.append(row)

# Check agreement
agree = 0
disagree = 0
uniprot_dark_hmm_white = 0
uniprot_white_hmm_dark = 0
unmapped = 0

for g in koester_genes:
    gene_id = g.get("gene_id", "")
    uniprot_status = g.get("pfam_status", "")

    # Try to find locus tag pattern
    locus = None
    for lt in prot_to_locus.values():
        # Extract the number from THAPSDRAFT_XXXX
        num = lt.replace("THAPSDRAFT_", "").replace("THAPS_", "")
        prot_id_str = str(g.get("protein_id", ""))
        if num == prot_id_str:
            locus = lt
            break

    if locus is None:
        unmapped += 1
        continue

    hmm_dark = locus in locus_dark
    up_dark = uniprot_status == "dark"

    if hmm_dark == up_dark:
        agree += 1
    else:
        disagree += 1
        if up_dark and not hmm_dark:
            uniprot_dark_hmm_white += 1
        else:
            uniprot_white_hmm_dark += 1

total_compared = agree + disagree
if total_compared > 0:
    print(f"Mapped Koester genes to NCBI locus tags: {total_compared}")
    print(f"Unmapped: {unmapped}")
    print(f"Agreement: {agree}/{total_compared} ({100*agree/total_compared:.1f}%)")
    print(f"UniProt dark → hmmsearch white: {uniprot_dark_hmm_white}")
    print(f"UniProt white → hmmsearch dark: {uniprot_white_hmm_dark}")
else:
    print(f"Could not map Koester gene IDs to NCBI locus tags")
    print(f"Mapped: {total_compared}, Unmapped: {unmapped}")
    print("Falling back to genome-wide comparison only")

# ── Genome-wide comparison ──
print(f"\n--- Genome-wide dark fraction comparison ---")
print(f"hmmsearch (E<1e-9): {n_dark}/{n_total} = {100*n_dark/n_total:.1f}% dark")

# UniProt stats from file
with open("tpseudo_uniprot_pfam.tsv") as f:
    f.readline()
    up_total = 0
    up_pfam = 0
    for line in f:
        up_total += 1
        fields = line.strip().split("\t")
        if len(fields) >= 4 and fields[3].strip():
            up_pfam += 1
up_dark = up_total - up_pfam
print(f"UniProt Pfam:       {up_dark}/{up_total} = {100*up_dark/up_total:.1f}% dark")

# InterPro from GFF3
with open("tpseudo_protein_interpro_status.tsv") as f:
    f.readline()
    ipr_total = 0
    ipr_annotated = 0
    for line in f:
        ipr_total += 1
        fields = line.strip().split("\t")
        if len(fields) >= 3 and fields[2] == "yes":
            ipr_annotated += 1
ipr_dark = ipr_total - ipr_annotated
print(f"NCBI InterPro:      {ipr_dark}/{ipr_total} = {100*ipr_dark/ipr_total:.1f}% dark")

print(f"\nNote: UniProt Pfam annotations may differ from direct hmmsearch")
print(f"because UniProt uses InterPro's Pfam processing pipeline with")
print(f"model-specific gathering thresholds, while we use a uniform E<1e-9 cutoff.")
