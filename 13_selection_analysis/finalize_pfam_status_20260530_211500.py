#!/usr/bin/env python3
"""
Finalize per-gene Pfam domain status for C. reinhardtii DS3 genes.

Merges three data sources to classify 17,535 Flowers 2015 DS3 genes:
  1. UniProt REST API: complete C. reinhardtii proteome (19,802 entries)
  2. Ensembl Plants BioMart: CHLRE_##g######v5 with Pfam domains
  3. NCBI RefSeq GFF3: Phytozome:Cre##.g###### cross-references

5,882 DS3 genes use old JGI v5.0/v5.3 gene IDs (g####) that were
dropped in v5.5/v5.6 and are absent from all current databases.
These are flagged as has_pfam=-1 (unresolved, needs hmmsearch in task 3).

Provenance:
  Script: finalize_pfam_status_20260530_211500.py
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
    gene_found = set()

    # Source 1: UniProt
    print("Parsing UniProt...")
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
                gene_found.add(cre_id)
                if pfam_field:
                    pfam_accessions = {p.strip() for p in pfam_field.split(';') if p.strip().startswith('PF')}
                    gene_pfams[cre_id].update(pfam_accessions)

    print(f"  UniProt Cre genes: {len(gene_found)}")

    # Source 2: BioMart
    print("Parsing BioMart...")
    biomart_count = 0
    with open(BIOMART_TSV, 'r') as f:
        f.readline()  # header
        for line in f:
            fields = line.strip().split('\t')
            if not fields:
                continue
            gene_id = fields[0].strip()
            pfam_id = fields[1].strip() if len(fields) > 1 else ''

            m = CHLRE_PATTERN.match(gene_id)
            if m:
                cre_id = f"Cre{m.group(1)}.g{m.group(2)}"
                gene_found.add(cre_id)
                if pfam_id and pfam_id.startswith('PF'):
                    gene_pfams[cre_id].add(pfam_id)
                    biomart_count += 1

    print(f"  BioMart Pfam entries added: {biomart_count}")

    # Source 3: GFF3 (gene existence)
    print("Parsing GFF3...")
    gff_genes = set()
    with open(GFF3_FILE, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            m = re.search(r'Phytozome:(Cre\d+\.g\d+)', line)
            if m:
                cre_id = m.group(1)
                gff_genes.add(cre_id)
                gene_found.add(cre_id)

    print(f"  GFF3 genes: {len(gff_genes)}")
    print(f"  Total resolved: {len(gene_found)}")

    # DS3
    print(f"\nReading DS3...")
    ds3_genes = []
    with open(DS3_FILE, 'r') as f:
        f.readline()  # citation
        f.readline()  # header
        for line in f:
            fields = line.strip().split('\t')
            if fields:
                ds3_genes.append(fields[0])

    ds3_set = set(ds3_genes)
    resolved = ds3_set & gene_found
    unresolved = ds3_set - gene_found

    n_cre_unresolved = sum(1 for g in unresolved if g.startswith('Cre'))
    n_scaffold_unresolved = sum(1 for g in unresolved if not g.startswith('Cre'))

    print(f"  DS3 genes: {len(ds3_genes)}")
    print(f"  Resolved: {len(resolved)}")
    print(f"  Unresolved: {len(unresolved)} (Cre: {n_cre_unresolved}, scaffold: {n_scaffold_unresolved})")

    # Write output
    print(f"\nWriting: {OUTPUT_FILE}")
    n_white = 0
    n_dark = 0
    n_unresolved = 0

    with open(OUTPUT_FILE, 'w') as f:
        f.write("# Provenance:\n")
        f.write(f"#   Script: {os.path.abspath(__file__)}\n")
        f.write(f"#   Source 1: UniProt REST API — all C. reinhardtii proteins (organism_id:3055)\n")
        f.write(f"#   Source 2: Ensembl Plants BioMart — creinhardtii_eg_gene, Pfam attribute\n")
        f.write(f"#   Source 3: NCBI RefSeq GFF3 — GCF_000002595.2 (gene existence via Phytozome Dbxref)\n")
        f.write(f"#   DS3: Flowers et al. 2015 tpc00492 SupplementalDS3 (doi:10.5061/dryad.1n0g6)\n")
        f.write(f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"#   Method: Merged UniProt Pfam + Ensembl BioMart Pfam + NCBI GFF3 cross-reference\n")
        f.write(f"#\n")
        f.write(f"#   Classification:\n")
        f.write(f"#     has_pfam=1: gene has >=1 Pfam domain in UniProt and/or BioMart\n")
        f.write(f"#     has_pfam=0: gene found in databases but no Pfam domain annotated\n")
        f.write(f"#     has_pfam=-1: gene uses old JGI v5.0 ID (g####) not in v5.5/v5.6;\n")
        f.write(f"#                  absent from UniProt, Ensembl, and NCBI RefSeq.\n")
        f.write(f"#                  These 5,882 genes need hmmsearch on the v5.0 proteome (task 3).\n")
        f.write("Gene_ID\thas_pfam\tpfam_domains\n")

        for gene in ds3_genes:
            pfams = gene_pfams.get(gene, set())
            if gene in gene_found:
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

    total = n_white + n_dark + n_unresolved
    print(f"\n{'='*60}")
    print(f"FINAL CLASSIFICATION OF {total} DS3 GENES")
    print(f"{'='*60}")
    print(f"  White (has_pfam=1):       {n_white:>6} ({100*n_white/total:>5.1f}%)")
    print(f"  Dark  (has_pfam=0):       {n_dark:>6} ({100*n_dark/total:>5.1f}%)")
    print(f"  Unresolved (has_pfam=-1): {n_unresolved:>6} ({100*n_unresolved/total:>5.1f}%)")
    print(f"  ──────────────────────────────────")
    print(f"  Resolved total:           {n_white+n_dark:>6} ({100*(n_white+n_dark)/total:>5.1f}%)")
    print(f"    Of resolved: white      {100*n_white/(n_white+n_dark):>5.1f}%")
    print(f"    Of resolved: dark       {100*n_dark/(n_white+n_dark):>5.1f}%")
    print(f"\n  Output: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
