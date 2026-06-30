#!/usr/bin/env python3
"""
task_41_18_parse_fp_rates.py - Ralph41 Task 41.18

Parse LA4SR inference TSV outputs produced by task_41_18_la4sr_benchmark.sbatch
for 40 non-algal eukaryotic reference proteomes and compute false-positive
rates per proteome, per lineage, and overall.

CLASSIFICATION RULE (from tools/la4sr/llm-metrics-two-files.py):
    After LA4SR generates up to 14 tokens per input sequence, the generated
    model_output string is scanned:
        '@' present -> predicted ALGAL (positive)
        '!' present -> predicted CONTAMINANT (negative)
        neither     -> UNKNOWN
    For a non-algal input proteome, the FP rate is:
        FP_rate = fraction of proteins whose model_output contains '@'
    (regardless of whether '!' also appears; '@' at all triggers a positive call)

PRE-REGISTERED GATE (ralph41_plan.md task 41.18):
    GREEN   if mean FP rate <= 10%
    YELLOW  if 10% < mean FP rate <= 20%
    RED     if mean FP rate > 20%

INPUT:
    - la4sr_benchmark/results/*.tsv     (one per proteome; from inference sbatch)
    - la4sr_benchmark_curation.tsv      (40-row manifest with lineage column)

OUTPUT:
    - source_data/ralph41/la4sr_benchmark_results.tsv
    - source_data/ralph41/la4sr_benchmark_results.md
    - source_data/ralph41/gate_41.18.md

Provenance header is written to every output file.

Data integrity: no synthetic data is ever generated; every FP rate is
computed by parsing real LA4SR inference TSV output files on disk.
"""

import argparse
import datetime
import os
import sys
from collections import defaultdict


def parse_curation_manifest(path):
    """Read the task 41.14 curation manifest, skip provenance header, return
    a dict keyed by the FASTA basename (no extension)."""
    manifest = {}
    with open(path) as fh:
        header = None
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            row = dict(zip(header, parts))
            # FASTA basename is proteome_fasta without directory or .fa
            fasta_name = os.path.basename(row.get("proteome_fasta", ""))
            if fasta_name.endswith(".fa"):
                fasta_basename = fasta_name[:-3]
            elif fasta_name.endswith(".fasta"):
                fasta_basename = fasta_name[:-6]
            else:
                fasta_basename = os.path.splitext(fasta_name)[0]
            row["_basename"] = fasta_basename
            manifest[fasta_basename] = row
    return manifest


def parse_la4sr_tsv(path):
    """Parse an LA4SR inference TSV with columns
    record_id, sequence, model_output and return (n_total, n_algal_calls,
    n_contam_calls, n_unknown)."""
    n_total = 0
    n_algal = 0
    n_contam = 0
    n_unknown = 0
    with open(path) as fh:
        header_line = fh.readline().rstrip("\n").split("\t")
        try:
            out_idx = header_line.index("model_output")
        except ValueError:
            sys.stderr.write(
                f"[WARN] {path}: no 'model_output' column; using last column\n"
            )
            out_idx = len(header_line) - 1
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) <= out_idx:
                # incomplete row, count as unknown
                n_total += 1
                n_unknown += 1
                continue
            model_output = parts[out_idx]
            n_total += 1
            has_algal = "@" in model_output
            has_contam = "!" in model_output
            if has_algal:
                # '@' dominates; matches llm-metrics-two-files.py behavior where
                # the first matching symbol determines the call
                n_algal += 1
            elif has_contam:
                n_contam += 1
            else:
                n_unknown += 1
    return n_total, n_algal, n_contam, n_unknown


def main():
    ap = argparse.ArgumentParser(
        description="Parse LA4SR non-algal FP benchmark (Ralph41 Task 41.18)"
    )
    ap.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing one *.tsv per proteome from LA4SR inference",
    )
    ap.add_argument(
        "--manifest",
        required=True,
        help="Path to la4sr_benchmark_curation.tsv from Task 41.14",
    )
    ap.add_argument(
        "--out-tsv",
        required=True,
        help="Output TSV (source_data/ralph41/la4sr_benchmark_results.tsv)",
    )
    ap.add_argument(
        "--out-md",
        required=True,
        help="Output Markdown summary (la4sr_benchmark_results.md)",
    )
    ap.add_argument(
        "--gate-file",
        required=True,
        help="Output gate file (gate_41.18.md)",
    )
    args = ap.parse_args()

    start_ts = datetime.datetime.now().isoformat(timespec="seconds")
    script_path = os.path.abspath(__file__)

    manifest = parse_curation_manifest(args.manifest)
    if not manifest:
        sys.stderr.write(
            f"[ERR] manifest {args.manifest} has no data rows; refusing to run\n"
        )
        return 2

    rows = []
    n_tsv_parsed = 0
    n_tsv_missing = 0
    for basename, meta in sorted(manifest.items()):
        tsv_path = os.path.join(args.results_dir, basename + ".tsv")
        if not os.path.exists(tsv_path):
            # Try alternative naming convention (without .fa stripped)
            alt_path = os.path.join(
                args.results_dir, os.path.basename(meta.get("proteome_fasta", "")) + ".tsv"
            )
            if os.path.exists(alt_path):
                tsv_path = alt_path
            else:
                n_tsv_missing += 1
                sys.stderr.write(f"[WARN] no TSV for {basename}\n")
                continue
        n_total, n_algal, n_contam, n_unknown = parse_la4sr_tsv(tsv_path)
        if n_total == 0:
            sys.stderr.write(f"[WARN] {tsv_path} has 0 sequences\n")
            continue
        fp_rate = n_algal / n_total
        rows.append(
            {
                "organism": meta.get("organism", basename),
                "lineage": meta.get("lineage", ""),
                "lineage_bucket": meta.get("lineage_bucket", ""),
                "uniprot_upid": meta.get("uniprot_upid", ""),
                "n_proteins": n_total,
                "n_algal_calls": n_algal,
                "n_contam_calls": n_contam,
                "n_unknown_calls": n_unknown,
                "fp_rate": fp_rate,
                "tsv_path": tsv_path,
            }
        )
        n_tsv_parsed += 1

    if not rows:
        sys.stderr.write(
            f"[ERR] no inference TSVs found in {args.results_dir}; refusing to write outputs\n"
        )
        return 3

    # Per-lineage aggregates (bucket: fungi / animals / protists)
    bucket_sum = defaultdict(list)
    for r in rows:
        bucket_sum[r["lineage_bucket"]].append(r["fp_rate"])

    per_bucket = {}
    for bucket, fps in sorted(bucket_sum.items()):
        mean_fp = sum(fps) / len(fps)
        per_bucket[bucket] = {
            "n_proteomes": len(fps),
            "mean_fp": mean_fp,
            "min_fp": min(fps),
            "max_fp": max(fps),
        }

    overall_mean_fp = sum(r["fp_rate"] for r in rows) / len(rows)
    total_proteins = sum(r["n_proteins"] for r in rows)
    total_algal_calls = sum(r["n_algal_calls"] for r in rows)
    pooled_fp = total_algal_calls / total_proteins if total_proteins else 0.0

    # Gate decision
    if overall_mean_fp <= 0.10:
        gate = "GREEN"
    elif overall_mean_fp <= 0.20:
        gate = "YELLOW"
    else:
        gate = "RED"

    # Write TSV with provenance header
    os.makedirs(os.path.dirname(args.out_tsv), exist_ok=True)
    with open(args.out_tsv, "w") as fh:
        fh.write("# Provenance\n")
        fh.write(f"# Script: {script_path}\n")
        fh.write(f"# Results directory: {os.path.abspath(args.results_dir)}\n")
        fh.write(f"# Manifest: {os.path.abspath(args.manifest)}\n")
        fh.write(f"# Run timestamp: {start_ts}\n")
        fh.write(
            "# Classification rule: model_output contains '@' -> algal (positive);\n"
            "# '!' -> contaminant; neither -> unknown. FP rate = algal_calls / n_proteins.\n"
        )
        fh.write(
            f"# n_proteomes parsed: {n_tsv_parsed}; n_proteomes missing: {n_tsv_missing}\n"
        )
        fh.write(f"# Overall mean FP rate (macro): {overall_mean_fp:.4f}\n")
        fh.write(f"# Pooled FP rate (micro): {pooled_fp:.4f}\n")
        fh.write(f"# Gate decision: {gate}\n")
        fh.write(
            "organism\tlineage\tlineage_bucket\tuniprot_upid\t"
            "n_proteins\tn_algal_calls\tn_contam_calls\tn_unknown_calls\t"
            "fp_rate\ttsv_path\n"
        )
        for r in rows:
            fh.write(
                f"{r['organism']}\t{r['lineage']}\t{r['lineage_bucket']}\t"
                f"{r['uniprot_upid']}\t{r['n_proteins']}\t{r['n_algal_calls']}\t"
                f"{r['n_contam_calls']}\t{r['n_unknown_calls']}\t"
                f"{r['fp_rate']:.6f}\t{r['tsv_path']}\n"
            )

    # Write Markdown summary
    with open(args.out_md, "w") as fh:
        fh.write("# Task 41.18: LA4SR non-algal FP benchmark results\n\n")
        fh.write("## Provenance\n\n")
        fh.write(f"- Script: `{script_path}`\n")
        fh.write(f"- Results directory: `{os.path.abspath(args.results_dir)}`\n")
        fh.write(f"- Manifest: `{os.path.abspath(args.manifest)}`\n")
        fh.write(f"- Run timestamp: `{start_ts}`\n\n")
        fh.write("## Classification rule\n\n")
        fh.write(
            "LA4SR generates up to 14 tokens per input sequence. "
            "The generated `model_output` string is scanned:\n\n"
        )
        fh.write("- `@` present -> predicted **algal** (positive)\n")
        fh.write("- `!` present (and no `@`) -> predicted **contaminant** (negative)\n")
        fh.write("- neither symbol -> **unknown**\n\n")
        fh.write(
            "For a non-algal input proteome, the FP rate is `n_algal_calls / n_proteins`.\n\n"
        )
        fh.write("## Aggregate statistics\n\n")
        fh.write(f"- Proteomes parsed: {n_tsv_parsed}\n")
        fh.write(f"- Proteomes missing: {n_tsv_missing}\n")
        fh.write(f"- Total proteins: {total_proteins}\n")
        fh.write(f"- Total algal calls: {total_algal_calls}\n")
        fh.write(f"- Overall mean FP rate (macro, equal-weight per proteome): {overall_mean_fp:.4f}\n")
        fh.write(f"- Pooled FP rate (micro, equal-weight per protein): {pooled_fp:.4f}\n\n")
        fh.write("## Per-lineage FP rates (macro mean across proteomes)\n\n")
        fh.write("| Lineage bucket | N proteomes | Mean FP | Min FP | Max FP |\n")
        fh.write("|---|---|---|---|---|\n")
        for bucket, stats in per_bucket.items():
            fh.write(
                f"| {bucket} | {stats['n_proteomes']} | {stats['mean_fp']:.4f} | "
                f"{stats['min_fp']:.4f} | {stats['max_fp']:.4f} |\n"
            )
        fh.write("\n## Per-proteome detail\n\n")
        fh.write("| Organism | Lineage | n_proteins | n_algal | FP rate |\n")
        fh.write("|---|---|---|---|---|\n")
        for r in sorted(rows, key=lambda x: (x["lineage_bucket"], x["organism"])):
            fh.write(
                f"| {r['organism']} | {r['lineage_bucket']} | "
                f"{r['n_proteins']} | {r['n_algal_calls']} | {r['fp_rate']:.4f} |\n"
            )
        fh.write("\n## Gate decision\n\n")
        fh.write(
            f"**Rule**: GREEN if mean FP <= 10%; YELLOW if 10% < mean FP <= 20%; "
            f"RED if mean FP > 20%.\n\n"
        )
        fh.write(f"**Observed overall mean FP**: {overall_mean_fp:.4f}\n\n")
        fh.write(f"**Decision**: {gate}\n")

    # Write gate file
    with open(args.gate_file, "w") as fh:
        fh.write("# Gate decision: 41.18\n")
        fh.write(f"- Decision: {gate}\n")
        fh.write("- Metric: LA4SR overall mean FP rate on 40 non-algal eukaryotic proteomes\n")
        fh.write(f"- Value: {overall_mean_fp:.4f}\n")
        fh.write("- Threshold: GREEN <=0.10; YELLOW (0.10,0.20]; RED >0.20\n")
        fh.write(
            "- Rule applied: per-proteome FP rate = (#proteins with '@' in model_output) / n_proteins; "
            "aggregate as macro mean across proteomes\n"
        )
        fh.write(f"- Timestamp: {start_ts}\n")
        fh.write(f"- Provenance: {os.path.abspath(args.out_tsv)}\n")
        fh.write(
            "- Per-lineage macro mean: "
            + "; ".join(
                f"{b}={s['mean_fp']:.4f} (n={s['n_proteomes']})"
                for b, s in per_bucket.items()
            )
            + "\n"
        )
        fh.write(f"- Pooled micro mean: {pooled_fp:.4f}\n")
        fh.write(f"- Total proteins scored: {total_proteins}\n")
        fh.write(f"- Proteomes parsed: {n_tsv_parsed} / expected 40\n")
        if n_tsv_missing > 0:
            fh.write(f"- WARNING: {n_tsv_missing} proteomes had no inference TSV\n")

    print(
        f"OK: parsed {n_tsv_parsed}/{len(manifest)} proteomes; overall mean FP={overall_mean_fp:.4f}; gate={gate}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
