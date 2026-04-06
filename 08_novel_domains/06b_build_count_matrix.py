#!/usr/bin/env python3
"""
Step 6b: Build Novel Domain Count Matrix from hmmsearch Results

Provenance:
    Script: scripts/novel_domains/06b_build_count_matrix.py
    Generated: 2026-02-21

Purpose:
    Parse per-sample hmmsearch .tbl files from Step 6 and build a count matrix
    analogous to the Pfam count matrix: rows = samples, columns = novel domains.

Input:
    - 03_analyses/novel_domains/novel_hits/{sample}.novel.tbl (from Step 6)
    - 03_analyses/novel_domains/sample_list.txt

Output:
    - 03_analyses/novel_domains/results/novel_domain_count_matrix.tsv

Usage:
    python3 scripts/novel_domains/06b_build_count_matrix.py
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path("/scratch/drn2/PROJECTS/TARA-LA4SR")
HITS_DIR = BASE / "03_analyses/novel_domains/novel_hits"
SAMPLE_LIST = BASE / "03_analyses/novel_domains/sample_list.txt"
OUT_DIR = BASE / "03_analyses/novel_domains/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "novel_domain_count_matrix.tsv"

def parse_tblout(tbl_path):
    """Parse hmmsearch tblout, return dict of {domain_name: count}."""
    counts = defaultdict(int)
    with open(tbl_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                continue
            # tblout format: target_name, target_acc, query_name, ...
            # query_name (field 3, 0-indexed 2) is the HMM name (novel domain)
            domain = fields[2]
            counts[domain] += 1
    return dict(counts)

def main():
    print("=" * 50)
    print("Step 6b: Build Novel Domain Count Matrix")
    print("=" * 50)
    print()

    # Read sample list
    with open(SAMPLE_LIST) as f:
        samples = [line.strip() for line in f if line.strip()]
    print(f"  Samples in list: {len(samples)}")

    # Parse all .tbl files
    all_domains = set()
    sample_counts = {}
    n_found = 0
    n_missing = 0

    for sample in samples:
        tbl_path = HITS_DIR / f"{sample}.novel.tbl"
        if tbl_path.exists():
            counts = parse_tblout(tbl_path)
            sample_counts[sample] = counts
            all_domains.update(counts.keys())
            n_found += 1
        else:
            sample_counts[sample] = {}
            n_missing += 1

    print(f"  Samples with .tbl files: {n_found}")
    print(f"  Samples missing .tbl:    {n_missing}")
    print(f"  Unique novel domains:    {len(all_domains)}")
    print()

    if len(all_domains) == 0:
        print("  WARNING: No novel domain hits found.")
        print("  Check that Step 6 completed successfully.")
        sys.exit(1)

    # Sort domains for consistent column order
    domain_list = sorted(all_domains)

    # Write count matrix
    print(f"  Writing count matrix: {OUTPUT}")
    with open(OUTPUT, "w") as f:
        # Header
        f.write("assembly_id\t" + "\t".join(domain_list) + "\n")
        # Data rows
        for sample in samples:
            counts = sample_counts[sample]
            vals = [str(counts.get(d, 0)) for d in domain_list]
            f.write(f"{sample}\t" + "\t".join(vals) + "\n")

    print(f"  Matrix dimensions: {len(samples)} samples x {len(domain_list)} domains")

    # Summary statistics
    total_hits = sum(sum(c.values()) for c in sample_counts.values())
    print(f"  Total hits across all samples: {total_hits:,}")

    # Domain prevalence
    domain_prevalence = defaultdict(int)
    for counts in sample_counts.values():
        for d in counts:
            domain_prevalence[d] += 1

    n_prevalent = sum(1 for v in domain_prevalence.values() if v >= 10)
    print(f"  Domains in >=10 samples:       {n_prevalent}")

    # Write domain summary
    summary_path = OUT_DIR / "novel_domain_prevalence.tsv"
    with open(summary_path, "w") as f:
        f.write("domain\tn_samples\ttotal_hits\n")
        for d in sorted(domain_prevalence, key=domain_prevalence.get, reverse=True):
            total = sum(sample_counts[s].get(d, 0) for s in samples)
            f.write(f"{d}\t{domain_prevalence[d]}\t{total}\n")
    print(f"  Domain prevalence: {summary_path}")

    print()
    print("  Done.")

if __name__ == "__main__":
    main()
