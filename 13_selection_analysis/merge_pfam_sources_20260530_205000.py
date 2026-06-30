#!/usr/bin/env python3
"""
Merge Pfam domain status from multiple sources for C. reinhardtii DS3 genes.

Sources:
  1. UniProt REST API (all 19,802 C. reinhardtii proteins) — primary
  2. Ensembl Plants BioMart (CHLRE_ gene IDs with Pfam) — secondary
  3. NCBI RefSeq GFF3 (Cre -> XP_ protein mapping) — for gene existence

A gene has has_pfam=1 if ANY source reports a Pfam domain.
A gene has has_pfam=0 if found in at least one source but NO Pfam reported.
A gene has has_pfam=-1 if not found in any source (needs hmmsearch, task 3).

Provenance:
  Script: merge_pfam_sources_20260530_205000.py
  UniProt: chlamy_uniprot_all_proteins.tsv
  BioMart: chlamy_biomart_pfam.tsv
  GFF3: GCF_000002595.2 genomic.gff
  DS3: tpc00492_SupplementalDS3.txt
"""

import os
import re
from collections import defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UNIPROT_TSV = os.path.join(SCRIPT_DIR, "chlamy_uniprot_all_proteins.tsv")
BIOMART_TSV = os.path.join(SCRIPT_DIR, "chlamy_biomart_pfam.tsv")
GFF3_FILE = "/media/drn2/External/TARA-Oceans/01_raw_data/reference_genomes/ncbi_refseq/euglenozoa/ncbi_dataset/data/GCF_000002595.2/genomic.gff"
DS3_FILE = os.path.join(SCRIPT_DIR, "tpc00492_SupplementalDS3.txt")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "chlamy_gene_pfam_status.tsv")

CHLRE_PATTERN = re.compile(r'CHLRE_(\d+)g(\d+)v5')
CRE_PATTERN = re.compile(r'Cre(\d+)\.g(\d+)')

def main():
    gene_pfams = defaultdict(set)
    gene_sources = defaultdict(set)

    # Source 1: UniProt
    print(f"Source 1: UniProt ({UNIPROT_TSV})")
    with open(UNIPROT_TSV, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('Entry\t'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 2:
                continue
            gene_names = fields[1]
            pfam_field = fields[2] if len(fields) > 2 else ""

            cre_ids = set()
            for chrom, gene_num in CHLRE_PATTERN.findall(gene_names):
                cre_ids.add(f"Cre{chrom}.g{gene_num}")
            for chrom, gene_num in CRE_PATTERN.findall(gene_names):
                cre_ids.add(f"Cre{chrom}.g{gene_num}")

            for cre_id in cre_ids:
                gene_sources[cre_id].add('UniProt')
                if pfam_field:
                    pfam_accessions = {p.strip() for p in pfam_field.split(';') if p.strip().startswith('PF')}
                    gene_pfams[cre_id].update(pfam_accessions)

    uniprot_genes = {g for g in gene_sources if 'UniProt' in gene_sources[g]}
    print(f"  Unique Cre genes: {len(uniprot_genes)}")
    print(f"  With Pfam: {sum(1 for g in uniprot_genes if gene_pfams[g])}")

    # Source 2: BioMart
    print(f"\nSource 2: BioMart ({BIOMART_TSV})")
    with open(BIOMART_TSV, 'r') as f:
        header = f.readline()
        for line in f:
            fields = line.strip().split('\t')
            if not fields:
                continue
            gene_id = fields[0].strip()
            pfam_id = fields[1].strip() if len(fields) > 1 else ''

            m = CHLRE_PATTERN.match(gene_id)
            if m:
                cre_id = f"Cre{m.group(1)}.g{m.group(2)}"
            else:
                continue

            gene_sources[cre_id].add('BioMart')
            if pfam_id and pfam_id.startswith('PF'):
                gene_pfams[cre_id].add(pfam_id)

    biomart_genes = {g for g in gene_sources if 'BioMart' in gene_sources[g]}
    print(f"  Unique Cre genes: {len(biomart_genes)}")
    print(f"  With Pfam: {sum(1 for g in biomart_genes if gene_pfams[g])}")

    # Source 3: GFF3 (for gene existence only)
    print(f"\nSource 3: GFF3 ({GFF3_FILE})")
    gff_genes = set()
    with open(GFF3_FILE, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            m = re.search(r'Phytozome:(Cre\d+\.g\d+)', line)
            if m:
                cre_id = m.group(1)
                gff_genes.add(cre_id)
                gene_sources[cre_id].add('GFF3')

    print(f"  Unique Cre genes: {len(gff_genes)}")

    # All resolved genes
    all_resolved = set(gene_sources.keys())
    print(f"\nTotal resolved genes (any source): {len(all_resolved)}")
    print(f"  With >=1 Pfam: {sum(1 for g in all_resolved if gene_pfams[g])}")
    print(f"  Without Pfam: {sum(1 for g in all_resolved if not gene_pfams[g])}")

    # Read DS3
    print(f"\nDS3: {DS3_FILE}")
    ds3_genes = []
    with open(DS3_FILE, 'r') as f:
        f.readline()
        f.readline()
        for line in f:
            fields = line.strip().split('\t')
            if fields:
                ds3_genes.append(fields[0])

    ds3_set = set(ds3_genes)
    print(f"  Total genes: {len(ds3_genes)}")

    resolved = ds3_set & all_resolved
    unresolved = ds3_set - all_resolved
    print(f"  Resolved: {len(resolved)}")
    print(f"  Unresolved: {len(unresolved)}")

    n_cre_unresolved = sum(1 for g in unresolved if g.startswith('Cre'))
    n_scaffold_unresolved = sum(1 for g in unresolved if not g.startswith('Cre'))
    print(f"    Cre-prefix unresolved: {n_cre_unresolved}")
    print(f"    Scaffold unresolved: {n_scaffold_unresolved}")

    # Write output
    print(f"\nWriting: {OUTPUT_FILE}")
    n_white = 0
    n_dark = 0
    n_unresolved = 0

    with open(OUTPUT_FILE, 'w') as f:
        f.write("# Provenance:\n")
        f.write(f"#   Script: {os.path.abspath(__file__)}\n")
        f.write(f"#   Source 1: UniProt REST API (all C. reinhardtii proteins, organism_id:3055)\n")
        f.write(f"#   Source 2: Ensembl Plants BioMart (creinhardtii_eg_gene dataset)\n")
        f.write(f"#   Source 3: NCBI RefSeq GFF3 (GCF_000002595.2, gene existence only)\n")
        f.write(f"#   DS3: Flowers et al. 2015 (doi:10.5061/dryad.1n0g6)\n")
        f.write(f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"#   Method: Merge UniProt Pfam + Ensembl BioMart Pfam + GFF3 gene mapping\n")
        f.write(f"#   has_pfam=1: >=1 Pfam domain in UniProt and/or BioMart\n")
        f.write(f"#   has_pfam=0: gene found in databases but no Pfam domain\n")
        f.write(f"#   has_pfam=-1: gene not found in any database (JGI scaffold gene)\n")
        f.write(f"#   NOTE: {n_scaffold_unresolved} scaffold + {n_cre_unresolved} Cre genes unresolved.\n")
        f.write(f"#         These are JGI v5.6 gene models not in NCBI RefSeq or Ensembl.\n")
        f.write(f"#         They require hmmsearch against the JGI proteome (task 3).\n")
        f.write("Gene_ID\thas_pfam\tpfam_domains\n")

        for gene in ds3_genes:
            pfams = gene_pfams.get(gene, set())
            if gene in all_resolved:
                has_pfam = 1 if pfams else 0
                pfam_str = ",".join(sorted(pfams)) if pfams else ""
                if has_pfam:
                    n_white += 1
                else:
                    n_dark += 1
            else:
                has_pfam = -1
                pfam_str = ""
                n_unresolved += 1
            f.write(f"{gene}\t{has_pfam}\t{pfam_str}\n")

    print(f"\nFinal DS3 classification:")
    print(f"  White (has_pfam=1): {n_white} ({100*n_white/len(ds3_genes):.1f}%)")
    print(f"  Dark  (has_pfam=0): {n_dark} ({100*n_dark/len(ds3_genes):.1f}%)")
    print(f"  Unresolved (has_pfam=-1): {n_unresolved} ({100*n_unresolved/len(ds3_genes):.1f}%)")

    # Sanity check: compare with BioMart vs UniProt agreement
    both = uniprot_genes & biomart_genes
    agree_pfam = sum(1 for g in both if (bool(gene_pfams[g])))
    print(f"\nSanity: {len(both)} genes in both UniProt and BioMart")

    # Check agreement on Pfam status
    pfam_in_uniprot = set()
    pfam_in_biomart = set()
    for g in both:
        # Check if gene had Pfam from either source independently
        # (gene_pfams merges both, so we check the raw data)
        pass

    print("  (Pfam domains from both sources are merged)")

if __name__ == '__main__':
    main()
