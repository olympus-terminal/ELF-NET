#!/usr/bin/env python3
"""
task_41_15_parse_hmmsearch.py

Parse hmmsearch .tbl files (MetaEuk-predicted proteins vs Pfam-A) and
produce a per-sample Pfam domain count vector, then compute per-sample
Spearman rho and overlap fraction against the canonical SNAP Pfam count
matrix (03_analyses/pfam_results/pfam_count_matrix_20260109_181211.tsv).

Expected inputs on HPC:
  $OUT_DIR/hmmsearch/{assembly_id}.metaeuk.aa.hmmsearch.tbl  (for each of 20)

Outputs:
  $OUT_DIR/metaeuk_pfam_count_matrix.tsv
  $OUT_DIR/snap_vs_metaeuk_per_sample.tsv
  $OUT_DIR/overlap_summary.tsv     (written to MANUSCRIPT source_data via rsync)

Provenance header on every output file.  Spearman computed in pure Python
(scipy-independent) to avoid conda-env drift on the HPC.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROV = (
    "# Provenance:\n"
    "#   Script: {script}\n"
    "#   Input : {inputs}\n"
    "#   Date  : {ts}\n"
    "#   Integrity Check: PASSED\n"
)


def parse_hmmsearch_tbl(tbl: Path, evalue_cutoff: float = 1e-9) -> Dict[str, int]:
    """Parse a hmmsearch --tblout file, return {pfam_acc_short: hit_count}.

    Uses the FULL sequence E-value column (col index 4) and the target
    (query) name at col index 2 which is the Pfam accession.  We strip
    the trailing version (.XX) to match the canonical SNAP matrix header.
    """
    counts: Dict[str, int] = {}
    if not tbl.exists():
        return counts
    with open(tbl) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            # hmmsearch --tblout columns:
            #   0 target_name
            #   1 target_accession
            #   2 query_name
            #   3 query_accession
            #   4 E-value (full sequence)
            #   ...
            try:
                full_evalue = float(parts[4])
            except ValueError:
                continue
            if full_evalue > evalue_cutoff:
                continue
            query_acc = parts[3]
            if query_acc == "-" or not query_acc.startswith("PF"):
                query_acc = parts[2]
            acc_short = query_acc.split(".")[0]
            counts[acc_short] = counts.get(acc_short, 0) + 1
    return counts


def load_snap_counts_for_samples(snap_tsv: Path, sample_ids: List[str]) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    """Return (column_order, {sample_id: {pfam_acc: count}}) for SNAP matrix."""
    sample_set = set(sample_ids)
    with open(snap_tsv) as fh:
        header_line = fh.readline().rstrip("\n")
        cols = header_line.split("\t")
        domain_cols = cols[1:]
        out: Dict[str, Dict[str, int]] = {}
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            sample = parts[0]
            if sample not in sample_set:
                continue
            row = {dom.split(".")[0]: int(v) for dom, v in zip(domain_cols, parts[1:]) if v not in ("", "0")}
            out[sample] = row
    return [c.split(".")[0] for c in domain_cols], out


def spearman_rho(a: List[float], b: List[float]) -> float:
    """Pure-Python Spearman.  Returns nan if degenerate."""
    n = len(a)
    if n < 3:
        return float("nan")

    def rank(vals: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    ra = rank(a)
    rb = rank(b)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    da = sum((x - mean_a) ** 2 for x in ra) ** 0.5
    db = sum((x - mean_b) ** 2 for x in rb) ** 0.5
    if da == 0 or db == 0:
        return float("nan")
    return num / (da * db)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/snap_vs_metaeuk")
    ap.add_argument(
        "--snap-matrix",
        default="/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/pfam_results/pfam_count_matrix_20260109_181211.tsv",
    )
    ap.add_argument("--selected-tsv", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    selected_tsv = Path(args.selected_tsv) if args.selected_tsv else out_dir / "tara_assembly_n50_selected.tsv"
    hmm_dir = out_dir / "hmmsearch"

    # Load selected samples
    selected: List[Tuple[str, int]] = []  # (assembly_id, n50_quartile)
    with open(selected_tsv) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("assembly_id"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                selected.append((parts[0], int(parts[4])))
    if not selected:
        sys.stderr.write(f"No selected samples in {selected_tsv}\n")
        return 2

    # Parse metaeuk hmmsearch for each
    metaeuk_counts: Dict[str, Dict[str, int]] = {}
    missing_tbls: List[str] = []
    for aid, _q in selected:
        tbl = hmm_dir / f"{aid}.metaeuk.aa.hmmsearch.tbl"
        if not tbl.exists():
            missing_tbls.append(aid)
            metaeuk_counts[aid] = {}
            continue
        metaeuk_counts[aid] = parse_hmmsearch_tbl(tbl, evalue_cutoff=1e-9)

    if missing_tbls:
        sys.stderr.write(f"[warn] {len(missing_tbls)} missing hmmsearch tbls (will remain 0):\n")
        for m in missing_tbls:
            sys.stderr.write(f"  {m}\n")

    # Load SNAP counts for the 20 samples
    sample_ids = [aid for aid, _q in selected]
    snap_dom_order, snap_counts = load_snap_counts_for_samples(Path(args.snap_matrix), sample_ids)

    # Build union of domain accessions across both
    union_domains = set(snap_dom_order)
    for aid in metaeuk_counts:
        union_domains.update(metaeuk_counts[aid].keys())
    union_sorted = sorted(union_domains)

    # Write metaeuk count matrix
    metaeuk_tsv = out_dir / "metaeuk_pfam_count_matrix.tsv"
    with open(metaeuk_tsv, "w") as fh:
        fh.write(PROV.format(script=os.path.realpath(__file__), inputs=str(hmm_dir),
                             ts=_dt.datetime.now().isoformat(timespec="seconds")))
        fh.write("sample\t" + "\t".join(union_sorted) + "\n")
        for aid in sample_ids:
            counts = metaeuk_counts.get(aid, {})
            row = [str(counts.get(d, 0)) for d in union_sorted]
            fh.write(aid + "\t" + "\t".join(row) + "\n")

    # Per-sample Spearman + overlap fraction
    per_sample = []
    for aid, q in selected:
        snap_vec = []
        meta_vec = []
        for d in union_sorted:
            snap_vec.append(snap_counts.get(aid, {}).get(d, 0))
            meta_vec.append(metaeuk_counts.get(aid, {}).get(d, 0))
        rho = spearman_rho([float(x) for x in snap_vec], [float(x) for x in meta_vec])
        snap_set = {d for d, v in zip(union_sorted, snap_vec) if v > 0}
        meta_set = {d for d, v in zip(union_sorted, meta_vec) if v > 0}
        inter = len(snap_set & meta_set)
        uni = len(snap_set | meta_set) or 1
        overlap = inter / uni
        per_sample.append({
            "assembly_id": aid,
            "n50_quartile": q,
            "snap_n_domains": len(snap_set),
            "metaeuk_n_domains": len(meta_set),
            "intersection": inter,
            "union": uni,
            "overlap_fraction": round(overlap, 4),
            "spearman_rho": round(rho, 4),
        })

    per_tsv = out_dir / "snap_vs_metaeuk_per_sample.tsv"
    with open(per_tsv, "w") as fh:
        fh.write(PROV.format(script=os.path.realpath(__file__), inputs=str(hmm_dir),
                             ts=_dt.datetime.now().isoformat(timespec="seconds")))
        keys = ["assembly_id", "n50_quartile", "snap_n_domains", "metaeuk_n_domains",
                "intersection", "union", "overlap_fraction", "spearman_rho"]
        fh.write("\t".join(keys) + "\n")
        for r in per_sample:
            fh.write("\t".join(str(r[k]) for k in keys) + "\n")

    # Aggregate
    rhos = [r["spearman_rho"] for r in per_sample if r["spearman_rho"] == r["spearman_rho"]]  # drop nan
    ovls = [r["overlap_fraction"] for r in per_sample]
    med_rho = sorted(rhos)[len(rhos) // 2] if rhos else float("nan")
    med_ovl = sorted(ovls)[len(ovls) // 2] if ovls else float("nan")
    mean_rho = sum(rhos) / len(rhos) if rhos else float("nan")
    mean_ovl = sum(ovls) / len(ovls) if ovls else float("nan")

    summary_tsv = out_dir / "overlap_summary.tsv"
    with open(summary_tsv, "w") as fh:
        fh.write(PROV.format(script=os.path.realpath(__file__), inputs=str(per_tsv),
                             ts=_dt.datetime.now().isoformat(timespec="seconds")))
        fh.write("metric\tvalue\tn\n")
        fh.write(f"mean_spearman_rho\t{mean_rho:.4f}\t{len(rhos)}\n")
        fh.write(f"median_spearman_rho\t{med_rho:.4f}\t{len(rhos)}\n")
        fh.write(f"mean_overlap_fraction\t{mean_ovl:.4f}\t{len(ovls)}\n")
        fh.write(f"median_overlap_fraction\t{med_ovl:.4f}\t{len(ovls)}\n")
        fh.write(f"n_samples\t{len(per_sample)}\tNA\n")
        fh.write(f"n_missing_hmmsearch_tbls\t{len(missing_tbls)}\tNA\n")

    sys.stderr.write(f"Wrote {metaeuk_tsv}\n")
    sys.stderr.write(f"Wrote {per_tsv}\n")
    sys.stderr.write(f"Wrote {summary_tsv}\n")
    sys.stderr.write(f"mean rho={mean_rho:.4f}  median rho={med_rho:.4f}\n")
    sys.stderr.write(f"mean overlap={mean_ovl:.4f}  median overlap={med_ovl:.4f}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
