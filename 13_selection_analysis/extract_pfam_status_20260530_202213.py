#!/usr/bin/env python3
"""
Extract per-gene Pfam domain status for C. reinhardtii DS3 genes.

Approach: Use UniProt Chlorophyta Pfam annotations (downloaded bulk TSV)
which contain CHLRE_##g######v5 gene identifiers that map 1:1 to
JGI v5.6 Cre##.g###### locus IDs used in Flowers et al. (2015).

Provenance:
  Script: extract_pfam_status_20260530_202213.py
  Input 1: chlorophyta_pfam_annotations.tsv.gz (UniProt bulk download)
  Input 2: tpc00492_SupplementalDS3.txt (Flowers 2015 per-gene diversity)
  Output: chlamy_gene_pfam_status.tsv
"""

import gzip
import re
import sys
import os
from collections import defaultdict
from datetime import datetime

UNIPROT_FILE = "/media/drn2/External/TARA-Oceans/01_raw_data/uniprot_proteomes/chlorophyta_pfam_annotations.tsv.gz"
DS3_FILE = os.path.join(os.path.dirname(__file__), "tpc00492_SupplementalDS3.txt")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "chlamy_gene_pfam_status.tsv")

CHLRE_PATTERN = re.compile(r'CHLRE_(\d+)g(\d+)v5')
CRE_PATTERN = re.compile(r'Cre(\d+)\.g(\d+)')

def chlre_to_cre(chlre_id):
    m = CHLRE_PATTERN.match(chlre_id)
    if m:
        return f"Cre{m.group(1)}.g{m.group(2)}"
    return None

def main():
    gene_pfams = defaultdict(set)
    gene_found_in_uniprot = set()

    print(f"Reading UniProt Chlorophyta Pfam annotations: {UNIPROT_FILE}")
    with gzip.open(UNIPROT_FILE, 'rt') as f:
        header = f.readline().strip().split('\t')
        gene_col = header.index('Gene Names')
        pfam_col = header.index('Pfam')
        org_col = header.index('Organism')

        for line in f:
            fields = line.strip().split('\t')
            if len(fields) <= max(gene_col, pfam_col, org_col):
                continue

            organism = fields[org_col]
            if 'reinhardtii' not in organism:
                continue

            gene_names = fields[gene_col]
            pfam_field = fields[pfam_col] if pfam_col < len(fields) else ""

            chlre_ids = CHLRE_PATTERN.findall(gene_names)
            cre_ids_direct = CRE_PATTERN.findall(gene_names)

            cre_ids = set()
            for chrom, gene_num in chlre_ids:
                cre_ids.add(f"Cre{chrom}.g{gene_num}")
            for chrom, gene_num in cre_ids_direct:
                cre_ids.add(f"Cre{chrom}.g{gene_num}")

            for cre_id in cre_ids:
                gene_found_in_uniprot.add(cre_id)
                if pfam_field:
                    pfam_accessions = [p.strip() for p in pfam_field.split(';') if p.strip().startswith('PF')]
                    gene_pfams[cre_id].update(pfam_accessions)

    print(f"  C. reinhardtii genes found in UniProt: {len(gene_found_in_uniprot)}")
    print(f"  Genes with >=1 Pfam domain: {sum(1 for g in gene_found_in_uniprot if gene_pfams[g])}")
    print(f"  Genes with 0 Pfam domains: {sum(1 for g in gene_found_in_uniprot if not gene_pfams[g])}")

    print(f"\nReading Flowers 2015 DS3: {DS3_FILE}")
    ds3_genes = []
    with open(DS3_FILE, 'r') as f:
        f.readline()  # skip citation line
        header = f.readline().strip().split('\t')
        for line in f:
            fields = line.strip().split('\t')
            if fields:
                ds3_genes.append(fields[0])

    print(f"  DS3 genes: {len(ds3_genes)}")

    overlap = set(ds3_genes) & gene_found_in_uniprot
    ds3_only = set(ds3_genes) - gene_found_in_uniprot
    uniprot_only = gene_found_in_uniprot - set(ds3_genes)
    print(f"  Overlap (DS3 ∩ UniProt): {len(overlap)}")
    print(f"  DS3-only (not in UniProt): {len(ds3_only)}")
    print(f"  UniProt-only (not in DS3): {len(uniprot_only)}")

    print(f"\nWriting output: {OUTPUT_FILE}")
    n_dark = 0
    n_white = 0
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"# Provenance:\n")
        f.write(f"#   Script: {os.path.abspath(__file__)}\n")
        f.write(f"#   UniProt input: {UNIPROT_FILE}\n")
        f.write(f"#   DS3 input: {DS3_FILE}\n")
        f.write(f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"#   Method: UniProt Chlorophyta Pfam bulk annotations\n")
        f.write(f"#   ID mapping: CHLRE_##g######v5 -> Cre##.g######\n")
        f.write(f"#   Note: genes not found in UniProt are classified as dark (has_pfam=0)\n")
        f.write(f"#         because absence from UniProt implies no reviewed annotation\n")
        f.write("Gene_ID\thas_pfam\tpfam_domains\n")

        for gene in ds3_genes:
            pfams = gene_pfams.get(gene, set())
            has_pfam = 1 if pfams else 0
            pfam_str = ",".join(sorted(pfams)) if pfams else ""
            f.write(f"{gene}\t{has_pfam}\t{pfam_str}\n")
            if has_pfam:
                n_white += 1
            else:
                n_dark += 1

    print(f"\nResult:")
    print(f"  White (has_pfam=1): {n_white} genes ({100*n_white/len(ds3_genes):.1f}%)")
    print(f"  Dark  (has_pfam=0): {n_dark} genes ({100*n_dark/len(ds3_genes):.1f}%)")
    print(f"  Total: {n_white + n_dark} genes")

    if ds3_only:
        print(f"\n  WARNING: {len(ds3_only)} DS3 genes not found in UniProt — classified as dark")
        print(f"  Examples: {sorted(ds3_only)[:5]}")

if __name__ == '__main__':
    main()
