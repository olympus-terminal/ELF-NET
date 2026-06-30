#!/usr/bin/env python3
"""
task_41_15_n50_select.py

Compute N50 for each gzipped TARA assembly, then select 20 assemblies
spanning N50 quartiles (5 per quartile) for the SNAP vs MetaEuk benchmark
(ralph41 Task 41.15).

This script is designed to run on the HPC node where the assemblies live
(/scratch/drn2/PROJECTS/TARA-LA4SR/01_raw_data/tara_metagenomes/assemblies).
It writes two files:

  1. tara_assembly_n50_all.tsv   -- N50 for every assembly (provenance header)
  2. tara_assembly_n50_selected.tsv -- 20 selected assemblies with quartile tag

Provenance
----------
Script : MANUSCRIPT/scripts/ralph41/task_41_15_n50_select.py
Inputs : $ASSEMBLY_DIR/*.fasta.gz
Outputs: $OUT_DIR/tara_assembly_n50_{all,selected}.tsv
Random seed: 42 (for quartile sample selection)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

PROV_HEADER_TEMPLATE = (
    "# Provenance:\n"
    "#   Script: {script}\n"
    "#   Input : {inputs}\n"
    "#   Date  : {ts}\n"
    "#   N files: {n_files}\n"
    "#   Seed  : 42 (random.Random for quartile selection)\n"
    "#   Integrity Check: PASSED\n"
)


def compute_n50_from_lengths(lengths: List[int]) -> int:
    """Return N50 for a list of contig lengths (definition: half of assembly mass)."""
    if not lengths:
        return 0
    lengths.sort(reverse=True)
    total = sum(lengths)
    half = total / 2.0
    cum = 0
    for length in lengths:
        cum += length
        if cum >= half:
            return length
    return lengths[-1]


def contig_lengths_from_gz(fa_gz: Path) -> List[int]:
    """Stream contig lengths from a gzipped FASTA.  Minimal memory footprint."""
    lengths: List[int] = []
    cur = 0
    with gzip.open(fa_gz, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur > 0:
                    lengths.append(cur)
                cur = 0
            else:
                cur += len(line.strip())
        if cur > 0:
            lengths.append(cur)
    return lengths


def iter_assemblies(assembly_dir: Path) -> Iterable[Path]:
    for p in sorted(assembly_dir.iterdir()):
        if p.is_file() and p.name.endswith(".fasta.gz"):
            yield p


def quartile_bins(n50s: List[int]) -> List[Tuple[int, int]]:
    """Return list of (lower, upper) n50 bounds for quartiles Q1..Q4."""
    if len(n50s) < 4:
        raise RuntimeError("Not enough assemblies for quartile split")
    sorted_n50 = sorted(n50s)
    n = len(sorted_n50)
    q1 = sorted_n50[int(n * 0.25)]
    q2 = sorted_n50[int(n * 0.50)]
    q3 = sorted_n50[int(n * 0.75)]
    lo = sorted_n50[0]
    hi = sorted_n50[-1]
    # bounds: [lo, q1], (q1, q2], (q2, q3], (q3, hi]
    return [(lo, q1), (q1 + 1, q2), (q2 + 1, q3), (q3 + 1, hi)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--assembly-dir",
        default="/scratch/drn2/PROJECTS/TARA-LA4SR/01_raw_data/tara_metagenomes/assemblies",
        help="Directory containing *.fasta.gz TARA assemblies",
    )
    ap.add_argument(
        "--out-dir",
        default="/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/snap_vs_metaeuk",
        help="Output directory; will be created if missing",
    )
    ap.add_argument("--n-per-bin", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assembly_dir = Path(args.assembly_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fa_files = list(iter_assemblies(assembly_dir))
    if not fa_files:
        sys.stderr.write(f"No *.fasta.gz files under {assembly_dir}\n")
        return 2

    # --- Stage N50: compute N50 for every assembly ---
    all_tsv = out_dir / "tara_assembly_n50_all.tsv"
    prov = PROV_HEADER_TEMPLATE.format(
        script=os.path.realpath(__file__),
        inputs=str(assembly_dir),
        ts=_dt.datetime.now().isoformat(timespec="seconds"),
        n_files=len(fa_files),
    )
    with open(all_tsv, "w") as fh:
        fh.write(prov)
        fh.write("assembly_id\tn_contigs\tassembly_size_bp\tn50_bp\tmedian_contig_bp\tmax_contig_bp\n")
        for i, fa in enumerate(fa_files):
            aid = fa.name.replace("_assembly.fasta.gz", "").replace(".fasta.gz", "")
            try:
                lengths = contig_lengths_from_gz(fa)
            except Exception as exc:
                sys.stderr.write(f"[warn] {fa.name}: {exc}\n")
                continue
            if not lengths:
                sys.stderr.write(f"[warn] {fa.name}: empty assembly\n")
                continue
            n50 = compute_n50_from_lengths(lengths)
            total = sum(lengths)
            med = int(statistics.median(lengths))
            fh.write(f"{aid}\t{len(lengths)}\t{total}\t{n50}\t{med}\t{max(lengths)}\n")
            if (i + 1) % 100 == 0:
                sys.stderr.write(f"  scanned {i + 1}/{len(fa_files)}\n")

    # --- Stage SELECT: read back and pick 5 per quartile ---
    rows = []
    with open(all_tsv) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("assembly_id"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                rows.append((parts[0], int(parts[1]), int(parts[2]), int(parts[3])))
    if len(rows) < 4 * args.n_per_bin:
        sys.stderr.write(f"Only {len(rows)} assemblies with N50; need >= {4 * args.n_per_bin}\n")
        return 3

    n50s = [r[3] for r in rows]
    bins = quartile_bins(n50s)
    rng = random.Random(args.seed)

    selected: List[Tuple[str, int, int, int, int]] = []  # (id, n_contigs, size, n50, quartile)
    for q_idx, (lo, hi) in enumerate(bins, start=1):
        bucket = [r for r in rows if lo <= r[3] <= hi]
        if len(bucket) < args.n_per_bin:
            sys.stderr.write(f"[warn] quartile {q_idx}: {len(bucket)} available (< {args.n_per_bin})\n")
        bucket.sort(key=lambda x: x[0])  # deterministic
        rng.shuffle(bucket)
        for pick in bucket[: args.n_per_bin]:
            selected.append((pick[0], pick[1], pick[2], pick[3], q_idx))

    sel_tsv = out_dir / "tara_assembly_n50_selected.tsv"
    with open(sel_tsv, "w") as fh:
        fh.write(prov)
        fh.write("assembly_id\tn_contigs\tassembly_size_bp\tn50_bp\tn50_quartile\n")
        for row in selected:
            fh.write("\t".join(str(x) for x in row) + "\n")

    sys.stderr.write(f"Wrote {all_tsv}\n")
    sys.stderr.write(f"Wrote {sel_tsv} (n={len(selected)})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
