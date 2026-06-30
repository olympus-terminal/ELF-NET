#!/usr/bin/env python3
"""
Download complete C. reinhardtii proteome from UniProt with Pfam annotations.

Unlike the bulk chlorophyta_pfam_annotations.tsv.gz (which only includes entries
WITH Pfam), this downloads ALL C. reinhardtii proteins (with and without Pfam)
so we can properly classify genes as dark (no Pfam) vs white (has Pfam).

Provenance:
  Script: download_uniprot_pfam_20260530_202800.py
  Source: UniProt REST API (organism_id:3055)
  Date: 2026-05-30
"""

import urllib.request
import urllib.parse
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_TSV = os.path.join(OUTPUT_DIR, "chlamy_uniprot_all_proteins.tsv")
STATUS_TSV = os.path.join(OUTPUT_DIR, "chlamy_gene_pfam_status.tsv")
DS3_FILE = os.path.join(OUTPUT_DIR, "tpc00492_SupplementalDS3.txt")

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"
BATCH_SIZE = 500
CHLRE_PATTERN = re.compile(r'CHLRE_(\d+)g(\d+)v5')
CRE_PATTERN = re.compile(r'Cre(\d+)\.g(\d+)')

def fetch_batch(cursor=None):
    params = {
        'query': '(organism_id:3055)',
        'fields': 'accession,gene_names,xref_pfam',
        'format': 'tsv',
        'size': str(BATCH_SIZE),
    }
    if cursor:
        params['cursor'] = cursor

    url = f"{UNIPROT_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Python/Chlamy-Pfam-Analysis (davidroynelson@gmail.com)')

    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read().decode('utf-8')
        link_header = response.headers.get('Link', '')

    next_cursor = None
    if 'rel="next"' in link_header:
        match = re.search(r'cursor=([^&>]+)', link_header)
        if match:
            next_cursor = match.group(1)

    return data, next_cursor

def main():
    print(f"Downloading all C. reinhardtii proteins from UniProt...")
    all_lines = []
    header = None
    cursor = None
    batch_num = 0

    while True:
        batch_num += 1
        data, cursor = fetch_batch(cursor)
        lines = data.strip().split('\n')

        if batch_num == 1:
            header = lines[0]
            all_lines.extend(lines[1:])
        else:
            all_lines.extend(lines[1:] if lines[0] == header else lines)

        print(f"  Batch {batch_num}: {len(lines)-1} entries (total: {len(all_lines)})")

        if not cursor:
            break
        time.sleep(0.5)

    print(f"\nTotal proteins downloaded: {len(all_lines)}")

    with open(RAW_TSV, 'w') as f:
        f.write(f"# Provenance:\n")
        f.write(f"#   Script: {os.path.abspath(__file__)}\n")
        f.write(f"#   Source: UniProt REST API (organism_id:3055, all proteins)\n")
        f.write(f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"#   Total entries: {len(all_lines)}\n")
        f.write(header + '\n')
        for line in all_lines:
            f.write(line + '\n')

    print(f"Raw data saved to: {RAW_TSV}")

    gene_pfams = defaultdict(set)
    gene_found = set()
    n_with_pfam = 0
    n_without_pfam = 0

    for line in all_lines:
        fields = line.split('\t')
        if len(fields) < 3:
            continue

        gene_names = fields[1]
        pfam_field = fields[2] if len(fields) > 2 else ""

        cre_ids = set()
        for chrom, gene_num in CHLRE_PATTERN.findall(gene_names):
            cre_ids.add(f"Cre{chrom}.g{gene_num}")
        for chrom, gene_num in CRE_PATTERN.findall(gene_names):
            cre_ids.add(f"Cre{chrom}.g{gene_num}")

        has_pfam = bool(pfam_field.strip() and 'PF' in pfam_field)

        for cre_id in cre_ids:
            gene_found.add(cre_id)
            if has_pfam:
                pfam_accessions = [p.strip() for p in pfam_field.split(';') if p.strip().startswith('PF')]
                gene_pfams[cre_id].update(pfam_accessions)

        if has_pfam:
            n_with_pfam += 1
        else:
            n_without_pfam += 1

    print(f"\nUniProt protein-level stats:")
    print(f"  With Pfam: {n_with_pfam}")
    print(f"  Without Pfam: {n_without_pfam}")
    print(f"  Unique Cre gene IDs found: {len(gene_found)}")
    print(f"  Cre genes with >=1 Pfam: {sum(1 for g in gene_found if gene_pfams[g])}")
    print(f"  Cre genes with 0 Pfam: {sum(1 for g in gene_found if not gene_pfams[g])}")

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

    overlap = set(ds3_genes) & gene_found
    ds3_only = set(ds3_genes) - gene_found
    print(f"  Overlap: {len(overlap)}")
    print(f"  DS3-only (not in UniProt with Cre ID): {len(ds3_only)}")

    print(f"\nWriting Pfam status: {STATUS_TSV}")
    n_dark = 0
    n_white = 0
    n_unresolved = 0

    with open(STATUS_TSV, 'w') as f:
        f.write(f"# Provenance:\n")
        f.write(f"#   Script: {os.path.abspath(__file__)}\n")
        f.write(f"#   UniProt API: organism_id:3055, all 19,802 proteins\n")
        f.write(f"#   DS3 input: {DS3_FILE}\n")
        f.write(f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"#   Method: UniProt REST API complete proteome download\n")
        f.write(f"#   ID mapping: CHLRE_##g######v5 -> Cre##.g######\n")
        f.write(f"#   Classification: has_pfam=1 if any UniProt entry for gene has Pfam annotation\n")
        f.write(f"#   has_pfam=-1 means gene not found in UniProt (unresolved)\n")
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

    print(f"\nFinal classification:")
    print(f"  White (has_pfam=1): {n_white}")
    print(f"  Dark  (has_pfam=0): {n_dark}")
    print(f"  Unresolved (has_pfam=-1, not in UniProt): {n_unresolved}")
    print(f"  Total: {n_white + n_dark + n_unresolved}")

    if n_unresolved > 0:
        print(f"\n  WARNING: {n_unresolved} DS3 genes have no UniProt entry with Cre/CHLRE ID.")
        print(f"  These genes cannot be classified by UniProt alone.")
        examples = [g for g in ds3_genes if g not in gene_found][:10]
        print(f"  Examples: {examples}")

if __name__ == '__main__':
    main()
