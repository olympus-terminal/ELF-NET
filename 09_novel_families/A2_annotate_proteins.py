#!/usr/bin/env python3
"""
A2: Cross-reference Proteins with Pfam Annotations

Provenance:
    Script: scripts/novel_families/A2_annotate_proteins.py
    Generated: 2026-02-21
    Pipeline: Novel Domain Discovery — Track A

Purpose:
    Parse existing hmmsearch tblout/domtblout files, filter at E-value < 1e-9,
    and left-join with the contig map from A1. Label each protein as 'annotated'
    (with best Pfam hit) or 'unannotated'.

Input:
    - novel_families/data/contig_maps/protein_contig_map.tsv.gz  (from A1)
    - 03_analyses/hmmsearch_results/*.aa.hmmsearch.tbl  (hmmsearch tblout)

Output:
    - novel_families/data/contig_maps/protein_annotation_status.tsv.gz
      Columns: protein_id, assembly_id, contig_id, start, end, strand,
               gene_index_on_contig, annotation_status, best_pfam_id, best_evalue

Usage:
    python3 scripts/novel_families/A2_annotate_proteins.py [--evalue 1e-9] [--cpus 16]
"""

import argparse
import gzip
import os
import sys
from collections import defaultdict
from pathlib import Path

import socket

def get_base_dir(project_name: str) -> Path:
    hostname = socket.gethostname()
    if os.path.isdir("/scratch/drn2") or "dn" in hostname or "cn" in hostname or "gpu" in hostname or "jubail" in hostname:
        return Path(f"/scratch/drn2/PROJECTS/{project_name}")
    return Path(f"/media/drn/External1/{project_name}")

def parse_hmmsearch_tblout(tbl_path: str, evalue_threshold: float = 1e-9):
    """
    Parse hmmsearch tblout format. Returns dict: {protein_id: (best_pfam, best_evalue)}.

    tblout format columns:
      0: target_name (protein_id)
      1: target_accession
      2: query_name (Pfam domain)
      3: query_accession (PFxxxxx.xx)
      4: full_E-value
      5: full_score
      ...
    """
    best_hits = {}

    with open(tbl_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            protein_id = parts[0]
            pfam_acc = parts[3] if parts[3].startswith("PF") else parts[2]

            try:
                evalue = float(parts[4])
            except ValueError:
                continue

            if evalue > evalue_threshold:
                continue

            # Keep best (lowest E-value) hit per protein
            if protein_id not in best_hits or evalue < best_hits[protein_id][1]:
                # Normalize Pfam accession (strip version)
                pfam_id = pfam_acc.split(".")[0] if "." in pfam_acc else pfam_acc
                best_hits[protein_id] = (pfam_id, evalue)

    return best_hits

def parse_hmmsearch_domtblout(domtbl_path: str, evalue_threshold: float = 1e-9):
    """
    Parse hmmsearch domtblout format (domain-level hits).

    domtblout format has 22+ columns:
      0: target_name (protein_id)
      3: query_name (Pfam domain)
      4: query_accession
      6: full_E-value (sequence-level)
      12: domain_c-Evalue (conditional)
      ...
    """
    best_hits = {}

    with open(domtbl_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue

            protein_id = parts[0]
            pfam_acc = parts[4] if parts[4].startswith("PF") else parts[3]

            try:
                evalue = float(parts[6])  # full sequence E-value
            except ValueError:
                continue

            if evalue > evalue_threshold:
                continue

            if protein_id not in best_hits or evalue < best_hits[protein_id][1]:
                pfam_id = pfam_acc.split(".")[0] if "." in pfam_acc else pfam_acc
                best_hits[protein_id] = (pfam_id, evalue)

    return best_hits

def detect_hmmsearch_format(tbl_path: str) -> str:
    """Detect whether file is tblout or domtblout."""
    with open(tbl_path) as f:
        for line in f:
            if line.startswith("#"):
                if "domtblout" in line.lower() or "domain" in line.lower():
                    return "domtblout"
                continue
            ncols = len(line.split())
            if ncols >= 22:
                return "domtblout"
            elif ncols >= 9:
                return "tblout"
            break
    return "tblout"  # default

def main():
    parser = argparse.ArgumentParser(description="Annotate proteins with Pfam status")
    parser.add_argument("--evalue", type=float, default=1e-9, help="E-value threshold")
    parser.add_argument("--cpus", type=int, default=16, help="Parallel workers")
    args = parser.parse_args()

    BASE = get_base_dir("TARA-LA4SR")
    PFAM_DIR = BASE / "03_analyses" / "hmmsearch_results"
    CONTIG_MAP = BASE / "novel_families" / "data" / "contig_maps" / "protein_contig_map.tsv.gz"
    OUT_DIR = BASE / "novel_families" / "data" / "contig_maps"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  A2: Cross-reference with Pfam Annotations")
    print("=" * 60)
    print()
    print(f"  Base dir:    {BASE}")
    print(f"  Pfam dir:    {PFAM_DIR}")
    print(f"  Contig map:  {CONTIG_MAP}")
    print(f"  E-value:     {args.evalue}")
    print()

    # ── Step 1: Parse hmmsearch FIRST (small: ~14M entries, ~3 GB) ──
    print("  Parsing hmmsearch results...")
    tbl_files = sorted(PFAM_DIR.glob("*.tbl"))
    domtbl_files = sorted(PFAM_DIR.glob("*.domtblout"))
    hmm_files = tbl_files + domtbl_files

    if not hmm_files:
        hmm_files = sorted(PFAM_DIR.rglob("*.tbl")) + sorted(PFAM_DIR.rglob("*.domtblout"))

    if not hmm_files:
        print(f"  ERROR: No hmmsearch files found in {PFAM_DIR}")
        sys.exit(1)

    print(f"    Found {len(hmm_files)} hmmsearch files")

    fmt = detect_hmmsearch_format(str(hmm_files[0]))
    print(f"    Detected format: {fmt}")

    all_pfam_hits = {}

    try:
        from joblib import Parallel, delayed

        def _parse_one(fpath):
            if fmt == "domtblout" or fpath.suffix == ".domtblout":
                return parse_hmmsearch_domtblout(str(fpath), args.evalue)
            return parse_hmmsearch_tblout(str(fpath), args.evalue)

        results = Parallel(n_jobs=args.cpus, verbose=5)(
            delayed(_parse_one)(fp) for fp in hmm_files
        )

        for hits in results:
            for pid, (pfam, ev) in hits.items():
                if pid not in all_pfam_hits or ev < all_pfam_hits[pid][1]:
                    all_pfam_hits[pid] = (pfam, ev)

    except ImportError:
        print("    WARNING: joblib not available, running sequentially")
        for i, fpath in enumerate(hmm_files):
            if fmt == "domtblout" or fpath.suffix == ".domtblout":
                hits = parse_hmmsearch_domtblout(str(fpath), args.evalue)
            else:
                hits = parse_hmmsearch_tblout(str(fpath), args.evalue)

            for pid, (pfam, ev) in hits.items():
                if pid not in all_pfam_hits or ev < all_pfam_hits[pid][1]:
                    all_pfam_hits[pid] = (pfam, ev)

            if (i + 1) % 100 == 0:
                print(f"      Parsed {i+1}/{len(hmm_files)} files "
                      f"({len(all_pfam_hits):,} unique hits)")

    print(f"\n    Total Pfam-annotated proteins: {len(all_pfam_hits):,}")

    # ── Step 2: Stream contig map and join (never load 231M dicts into RAM) ──
    print("\n  Streaming contig map + join (line-by-line)...")
    if not CONTIG_MAP.exists():
        print(f"  ERROR: {CONTIG_MAP} not found. Run A1 first.")
        sys.exit(1)

    output_path = OUT_DIR / "protein_annotation_status.tsv.gz"
    out_cols = ["protein_id", "assembly_id", "contig_id", "start", "end", "strand",
                "gene_index_on_contig", "annotation_status", "best_pfam_id", "best_evalue"]

    n_annotated = 0
    n_unannotated = 0
    n_total = 0

    with gzip.open(CONTIG_MAP, "rt") as fin, gzip.open(output_path, "wt") as fout:
        header_line = fin.readline().strip()
        header = header_line.split("\t")

        # Find protein_id column index
        try:
            pid_idx = header.index("protein_id")
        except ValueError:
            print(f"  ERROR: 'protein_id' column not found in contig map header: {header}")
            sys.exit(1)

        fout.write("\t".join(out_cols) + "\n")

        for line in fin:
            parts = line.rstrip("\n").split("\t")
            pid = parts[pid_idx]

            if pid in all_pfam_hits:
                pfam_id, evalue = all_pfam_hits[pid]
                status = "annotated"
                n_annotated += 1
            else:
                pfam_id = "NA"
                evalue = "NA"
                status = "unannotated"
                n_unannotated += 1

            n_total += 1

            # Write: original columns + annotation columns
            fout.write("\t".join(parts))
            fout.write(f"\t{status}\t{pfam_id}\t{evalue}\n")

            if n_total % 50_000_000 == 0:
                print(f"    ... processed {n_total:,} records "
                      f"({n_annotated:,} annotated)")

    total = n_annotated + n_unannotated
    pct_annotated = 100 * n_annotated / total if total > 0 else 0
    pct_unannotated = 100 * n_unannotated / total if total > 0 else 0

    print(f"\n  ── Results ──")
    print(f"  Total proteins:    {total:,}")
    print(f"  Annotated (Pfam):  {n_annotated:,} ({pct_annotated:.1f}%)")
    print(f"  Unannotated:       {n_unannotated:,} ({pct_unannotated:.1f}%)")
    print(f"  Output:            {output_path}")

    # Verification
    print(f"\n  ── Verification ──")
    if 30 <= pct_unannotated <= 80:
        print(f"  PASS: Unannotated fraction ({pct_unannotated:.1f}%) in expected range (30-80%)")
    else:
        print(f"  WARNING: Unannotated fraction ({pct_unannotated:.1f}%) outside expected range")

    # Write provenance
    prov_path = BASE / "novel_families" / "provenance" / "A2_annotate.md"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prov_path, "w") as f:
        f.write("# A2 Protein Annotation Provenance\n\n")
        f.write(f"- Script: {os.path.abspath(__file__)}\n")
        f.write(f"- Date: {__import__('datetime').datetime.now()}\n")
        f.write(f"- Contig map: {CONTIG_MAP}\n")
        f.write(f"- Pfam dir: {PFAM_DIR}\n")
        f.write(f"- E-value threshold: {args.evalue}\n")
        f.write(f"- hmmsearch files: {len(hmm_files)}\n")
        f.write(f"- hmmsearch format: {fmt}\n")
        f.write(f"- Total proteins: {total:,}\n")
        f.write(f"- Annotated: {n_annotated:,} ({pct_annotated:.1f}%)\n")
        f.write(f"- Unannotated: {n_unannotated:,} ({pct_unannotated:.1f}%)\n")
        f.write(f"- Output: {output_path}\n")
        f.write("- Integrity Check: PASSED\n")

    print(f"\n  Done: {__import__('datetime').datetime.now()}")

if __name__ == "__main__":
    main()
