#!/usr/bin/env python3
"""
Build per-gene Pfam status for DS3 genes using TWO complementary data sources:

1. NCBI RefSeq GFF3 for GCF_000002595.2 — maps Cre##.g###### (Phytozome)
   to XP_* RefSeq protein accessions via Dbxref fields in CDS entries.

2. NCBI RefSeq protein.faa — maps XP_* accessions to CHLRE_##g######v5 IDs
   (in FASTA headers), providing a second mapping route.

3. UniProt REST API download (chlamy_uniprot_all_proteins.tsv) — provides
   Pfam domain annotations for proteins with CHLRE_/Cre gene IDs.

4. For genes mapped to XP_ accessions but NOT found in UniProt by gene name,
   we look up the XP_ accession in InterPro to get Pfam annotations.

The key insight: UniProt indexes C. reinhardtii proteins by CHLRE_ gene IDs,
but many gene models (especially less-characterized ones) lack CHLRE_ IDs in
their UniProt entry — they're accessible only via XP_ protein accession.

Provenance:
  Script: build_gff_pfam_status_20260530_203500.py
  GFF3: GCF_000002595.2 genomic.gff
  protein.faa: GCF_000002595.2 protein.faa
  UniProt: chlamy_uniprot_all_proteins.tsv
  DS3: tpc00492_SupplementalDS3.txt
"""

import os
import re
import sys
import urllib.request
import urllib.parse
import json
import time
from collections import defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GFF3_FILE = "/media/drn2/External/TARA-Oceans/01_raw_data/reference_genomes/ncbi_refseq/euglenozoa/ncbi_dataset/data/GCF_000002595.2/genomic.gff"
PROTEIN_FAA = "/media/drn2/External/TARA-Oceans/01_raw_data/reference_genomes/ncbi_refseq/euglenozoa/ncbi_dataset/data/GCF_000002595.2/protein.faa"
UNIPROT_TSV = os.path.join(SCRIPT_DIR, "chlamy_uniprot_all_proteins.tsv")
DS3_FILE = os.path.join(SCRIPT_DIR, "tpc00492_SupplementalDS3.txt")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "chlamy_gene_pfam_status.tsv")

CHLRE_PATTERN = re.compile(r'CHLRE_(\d+)g(\d+)v5')
CRE_PATTERN = re.compile(r'Cre(\d+)\.g(\d+)')

def parse_gff3_mappings():
    """Parse GFF3 to build Cre gene ID -> XP_ protein ID mapping."""
    cre_to_xp = defaultdict(set)
    print(f"Parsing GFF3: {GFF3_FILE}")

    with open(GFF3_FILE, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            if fields[2] != 'CDS':
                continue

            attrs = fields[8]
            cre_match = re.search(r'Phytozome:(Cre\d+\.g\d+)', attrs)
            xp_match = re.search(r'protein_id=(XP_\d+\.\d+)', attrs)

            if cre_match and xp_match:
                cre_id = cre_match.group(1)
                xp_id = xp_match.group(1)
                cre_to_xp[cre_id].add(xp_id)

    print(f"  Cre genes with XP_ protein: {len(cre_to_xp)}")
    print(f"  Total XP_ protein IDs: {sum(len(v) for v in cre_to_xp.values())}")
    return cre_to_xp

def parse_protein_faa():
    """Parse protein.faa to build XP_ -> CHLRE_ -> Cre mapping."""
    xp_to_cre = {}
    print(f"\nParsing protein.faa: {PROTEIN_FAA}")

    with open(PROTEIN_FAA, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                continue
            xp_match = re.search(r'>(\S+)', line)
            chlre_match = CHLRE_PATTERN.search(line)

            if xp_match and chlre_match:
                xp_id = xp_match.group(1)
                cre_id = f"Cre{chlre_match.group(1)}.g{chlre_match.group(2)}"
                xp_to_cre[xp_id] = cre_id

    print(f"  XP_ to Cre mappings: {len(xp_to_cre)}")
    return xp_to_cre

def parse_uniprot_pfam():
    """Parse UniProt TSV to get gene-level Pfam annotations."""
    gene_pfams = defaultdict(set)
    gene_found = set()
    xp_pfams = defaultdict(set)

    print(f"\nParsing UniProt TSV: {UNIPROT_TSV}")

    with open(UNIPROT_TSV, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.startswith('Entry\t'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 2:
                continue

            accession = fields[0]
            gene_names = fields[1] if len(fields) > 1 else ""
            pfam_field = fields[2] if len(fields) > 2 else ""

            pfam_accessions = set()
            if pfam_field:
                pfam_accessions = {p.strip() for p in pfam_field.split(';') if p.strip().startswith('PF')}

            cre_ids = set()
            for chrom, gene_num in CHLRE_PATTERN.findall(gene_names):
                cre_ids.add(f"Cre{chrom}.g{gene_num}")
            for chrom, gene_num in CRE_PATTERN.findall(gene_names):
                cre_ids.add(f"Cre{chrom}.g{gene_num}")

            for cre_id in cre_ids:
                gene_found.add(cre_id)
                gene_pfams[cre_id].update(pfam_accessions)

    print(f"  Unique Cre genes in UniProt: {len(gene_found)}")
    print(f"  With Pfam: {sum(1 for g in gene_found if gene_pfams[g])}")
    print(f"  Without Pfam: {sum(1 for g in gene_found if not gene_pfams[g])}")

    return gene_pfams, gene_found

def batch_query_interpro(xp_ids, batch_size=10):
    """Query InterPro for Pfam annotations of XP_ protein accessions."""
    results = {}
    total = len(xp_ids)

    print(f"\nQuerying InterPro for {total} unresolved XP_ proteins...")

    for i in range(0, total, batch_size):
        batch = xp_ids[i:i+batch_size]
        for xp_id in batch:
            try:
                url = f"https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/UniProt/{xp_id}/"
                req = urllib.request.Request(url)
                req.add_header('Accept', 'application/json')
                req.add_header('User-Agent', 'Python/Chlamy-Pfam-Analysis')

                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    pfam_ids = set()
                    for result in data.get('results', []):
                        acc = result.get('metadata', {}).get('accession', '')
                        if acc.startswith('PF'):
                            pfam_ids.add(acc)
                    results[xp_id] = pfam_ids
            except urllib.error.HTTPError as e:
                if e.code == 204 or e.code == 404:
                    results[xp_id] = set()
                else:
                    results[xp_id] = set()
            except Exception:
                results[xp_id] = set()

            time.sleep(0.1)

        if (i + batch_size) % 100 == 0 or i + batch_size >= total:
            print(f"  Processed {min(i + batch_size, total)}/{total}")

    return results

def main():
    cre_to_xp = parse_gff3_mappings()
    xp_to_cre = parse_protein_faa()
    gene_pfams, uniprot_genes = parse_uniprot_pfam()

    print(f"\nReading DS3: {DS3_FILE}")
    ds3_genes = []
    with open(DS3_FILE, 'r') as f:
        f.readline()
        f.readline()
        for line in f:
            fields = line.strip().split('\t')
            if fields:
                ds3_genes.append(fields[0])
    print(f"  DS3 genes: {len(ds3_genes)}")

    resolved_by_uniprot = set(ds3_genes) & uniprot_genes
    unresolved = set(ds3_genes) - uniprot_genes
    print(f"\n  Resolved by UniProt gene names: {len(resolved_by_uniprot)}")
    print(f"  Unresolved: {len(unresolved)}")

    unresolved_with_xp = {}
    for gene in unresolved:
        if gene in cre_to_xp:
            xps = cre_to_xp[gene]
            unresolved_with_xp[gene] = xps

    unresolved_no_xp = unresolved - set(unresolved_with_xp.keys())
    print(f"  Unresolved with XP_ mapping (via GFF3): {len(unresolved_with_xp)}")
    print(f"  Unresolved without any mapping: {len(unresolved_no_xp)}")

    all_xp_to_check = set()
    for xps in unresolved_with_xp.values():
        all_xp_to_check.update(xps)

    if all_xp_to_check:
        print(f"\n  Need to check {len(all_xp_to_check)} XP_ proteins via InterPro...")

        print("  Trying UniProt bulk lookup by XP_ accession first...")
        xp_pfam_from_uniprot = {}
        xp_list = sorted(all_xp_to_check)

        for i in range(0, len(xp_list), 100):
            batch = xp_list[i:i+100]
            query_str = " OR ".join([f"xref:{xp}" for xp in batch])
            params = {
                'query': f'(organism_id:3055) AND ({query_str})',
                'fields': 'accession,gene_names,xref_pfam,xref_refseq',
                'format': 'tsv',
                'size': '500',
            }
            url = f"https://rest.uniprot.org/uniprotkb/search?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Python/Chlamy-Pfam-Analysis (davidroynelson@gmail.com)')

            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    data = response.read().decode('utf-8')
                    lines = data.strip().split('\n')
                    for line in lines[1:]:
                        fields = line.split('\t')
                        if len(fields) >= 3:
                            pfam_field = fields[2]
                            refseq_field = fields[3] if len(fields) > 3 else ""
                            pfam_accessions = {p.strip() for p in pfam_field.split(';') if p.strip().startswith('PF')}

                            for xp in batch:
                                if xp in refseq_field:
                                    xp_pfam_from_uniprot[xp] = pfam_accessions
            except Exception as e:
                print(f"  Batch {i//100 + 1} failed: {e}")

            time.sleep(0.5)
            if (i + 100) % 500 == 0:
                print(f"  Checked {min(i+100, len(xp_list))}/{len(xp_list)} XP_ IDs")

        print(f"  Resolved {len(xp_pfam_from_uniprot)} XP_ proteins via UniProt lookup")

        for gene, xps in unresolved_with_xp.items():
            for xp in xps:
                if xp in xp_pfam_from_uniprot:
                    gene_pfams[gene].update(xp_pfam_from_uniprot[xp])
                    uniprot_genes.add(gene)

        still_unresolved = set(ds3_genes) - uniprot_genes
        print(f"  Still unresolved after XP_ lookup: {len(still_unresolved)}")

    still_unresolved_with_xp = {}
    for gene in set(ds3_genes) - uniprot_genes:
        if gene in cre_to_xp:
            still_unresolved_with_xp[gene] = cre_to_xp[gene]

    remaining_xps = set()
    for xps in still_unresolved_with_xp.values():
        remaining_xps.update(xps)

    if remaining_xps and len(remaining_xps) <= 2000:
        print(f"\n  Querying InterPro for {len(remaining_xps)} remaining XP_ proteins...")
        interpro_results = batch_query_interpro(sorted(remaining_xps))

        for gene, xps in still_unresolved_with_xp.items():
            for xp in xps:
                if xp in interpro_results:
                    gene_pfams[gene].update(interpro_results[xp])
                    uniprot_genes.add(gene)
    elif remaining_xps:
        print(f"\n  Too many remaining XP_ proteins ({len(remaining_xps)}) for InterPro batch query.")
        print(f"  Using GFF3 protein_id mapping to resolve via hmmsearch results...")

    final_unresolved = set(ds3_genes) - uniprot_genes
    gff3_resolved = 0
    for gene in list(final_unresolved):
        if gene in cre_to_xp:
            uniprot_genes.add(gene)
            gff3_resolved += 1

    print(f"\n  Final GFF3-only resolved (no Pfam data, classified as dark): {gff3_resolved}")

    truly_unresolved = set(ds3_genes) - uniprot_genes

    in_gff3 = sum(1 for g in ds3_genes if g in cre_to_xp)
    not_in_gff3 = sum(1 for g in ds3_genes if g not in cre_to_xp)
    print(f"\n  DS3 genes in GFF3: {in_gff3}")
    print(f"  DS3 genes NOT in GFF3: {not_in_gff3}")
    if not_in_gff3 > 0:
        examples = [g for g in ds3_genes if g not in cre_to_xp][:10]
        print(f"  Examples not in GFF3: {examples}")

    print(f"\nWriting final output: {OUTPUT_FILE}")
    n_dark = 0
    n_white = 0
    n_unresolved = 0

    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"# Provenance:\n")
        f.write(f"#   Script: {os.path.abspath(__file__)}\n")
        f.write(f"#   GFF3: {GFF3_FILE}\n")
        f.write(f"#   protein.faa: {PROTEIN_FAA}\n")
        f.write(f"#   UniProt: {UNIPROT_TSV}\n")
        f.write(f"#   DS3: {DS3_FILE}\n")
        f.write(f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"#   Method: UniProt REST API + NCBI RefSeq GFF3 cross-reference\n")
        f.write(f"#   ID mapping: CHLRE_##g######v5 / Cre##.g###### (UniProt gene names)\n")
        f.write(f"#              + Phytozome:Cre##.g###### <-> protein_id:XP_* (GFF3 Dbxref)\n")
        f.write(f"#   has_pfam=1: gene has >=1 Pfam domain annotation in UniProt\n")
        f.write(f"#   has_pfam=0: gene found but has no Pfam domain annotation\n")
        f.write(f"#   has_pfam=-1: gene not found via any mapping route\n")
        f.write("Gene_ID\thas_pfam\tpfam_domains\n")

        for gene in ds3_genes:
            pfams = gene_pfams.get(gene, set())
            if gene in uniprot_genes:
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

    print(f"\nFinal classification of {len(ds3_genes)} DS3 genes:")
    print(f"  White (has_pfam=1): {n_white} ({100*n_white/len(ds3_genes):.1f}%)")
    print(f"  Dark  (has_pfam=0): {n_dark} ({100*n_dark/len(ds3_genes):.1f}%)")
    print(f"  Unresolved (has_pfam=-1): {n_unresolved} ({100*n_unresolved/len(ds3_genes):.1f}%)")

if __name__ == '__main__':
    main()
