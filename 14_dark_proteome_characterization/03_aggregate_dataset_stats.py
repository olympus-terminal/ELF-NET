#!/usr/bin/env python3
"""
Aggregate per-sample Pfam density statistics by dataset.

Reads per-sample TSVs from 02_pfam_density.sbatch output, joins with
metadata to get dataset labels, and produces:
  1. Per-sample aggregate TSV (all samples, one row each)
  2. Per-dataset summary table (grouped statistics)
  3. Source-type summary table (MGYA, MMETSP, GCA, etc.)

Provenance:
  Script: scripts/dark_proteome/03_aggregate_dataset_stats.py
  Generated: 2026-03-01

Usage (on HPC after 02 completes):
  cd /scratch/drn2/PROJECTS/TARA-LA4SR
  python3 MANUSCRIPT/scripts/dark_proteome/03_aggregate_dataset_stats.py

Usage (locally with rsync'd data):
  python3 scripts/dark_proteome/03_aggregate_dataset_stats.py \
      --density-dir /path/to/pfam_density/ \
      --master-key omen-work/ASSEMBLY_GPS_MASTER_KEY_20260119_101500.tsv \
      --per-sample-summary source_data/per_sample_summary.tsv \
      --outdir source_data/
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--density-dir",
                    default="/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/dark_proteome/pfam_density",
                    help="Directory containing per-sample .pfam_density.tsv files")
    p.add_argument("--aggregate-tsv",
                    default=None,
                    help="Pre-aggregated TSV (pfam_density_all_samples.tsv). "
                         "If provided, --density-dir is ignored.")
    p.add_argument("--master-key",
                    default=None,
                    help="ASSEMBLY_GPS_MASTER_KEY TSV (for dataset labels). "
                         "Auto-detected from MANUSCRIPT/omen-work/ or current dir.")
    p.add_argument("--per-sample-summary",
                    default=None,
                    help="per_sample_summary.tsv (for source_type labels). "
                         "Auto-detected from MANUSCRIPT/source_data/ or current dir.")
    p.add_argument("--outdir",
                    default=None,
                    help="Output directory (default: same as density-dir parent)")
    return p.parse_args()

def find_file(candidates, label):
    """Try multiple candidate paths, return first that exists."""
    for c in candidates:
        if os.path.isfile(c):
            return c
    print(f"WARNING: Could not find {label}. Tried:")
    for c in candidates:
        print(f"  {c}")
    return None

def load_master_key(path):
    """Load ASSEMBLY_GPS_MASTER_KEY, return dict: assembly_id -> dataset."""
    if not path:
        return {}
    mapping = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 6 or parts[0] == "assembly_id":
                continue
            assembly_id = parts[0]
            dataset = parts[5]
            mapping[assembly_id] = dataset
    print(f"  Loaded {len(mapping)} assembly->dataset mappings from master key")
    return mapping

def load_source_types(path):
    """Load per_sample_summary.tsv, return dict: stem -> source_type.

    Filenames like 'Alexandrium_andersonii.AAC.aa.algal.fa' ->
    stem 'Alexandrium_andersonii.AAC' after stripping .aa.algal.fa / .aa.algae.fa
    """
    if not path:
        return {}
    mapping = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            fname = row["filename"]
            # Strip known suffixes to get stem
            stem = fname
            for suffix in [".aa.algal.fa", ".aa.algae.fa", ".fa", ".faa", ".fasta"]:
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            mapping[stem] = row["source_type"]
    print(f"  Loaded {len(mapping)} stem->source_type mappings")
    return mapping

def infer_source_type(sample_name):
    """Infer source_type from sample name prefix when metadata is unavailable."""
    if sample_name.startswith("MGYA"):
        return "MGYA_metagenome"
    elif sample_name.startswith("MMETSP"):
        return "MMETSP_transcriptome"
    elif sample_name.startswith("GCA_"):
        return "GCA_genbank"
    elif sample_name.startswith("GCF_"):
        return "GCF_refseq"
    elif sample_name.startswith("euglenozoa"):
        return "Euglenozoa_reference"
    else:
        return "Other_reference"

def load_aggregate_tsv(path):
    """Load a pre-aggregated TSV with all samples (one row per sample)."""
    records = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            for key in ["n_proteins", "total_aa", "n_annotated", "n_dark",
                        "n_domain_hits", "n_unique_pfam"]:
                row[key] = int(row[key])
            for key in ["pct_dark", "pfam_per_orf", "pfam_per_1000aa"]:
                row[key] = float(row[key])
            records.append(row)
    print(f"  Loaded {len(records)} sample records from {path}")
    return records

def load_density_files(density_dir):
    """Load all per-sample pfam_density TSVs into a list of dicts."""
    records = []
    density_path = Path(density_dir)
    if not density_path.is_dir():
        print(f"ERROR: Density directory not found: {density_dir}")
        sys.exit(1)

    files = sorted(density_path.glob("*.pfam_density.tsv"))
    print(f"  Found {len(files)} per-sample density files")

    for f in files:
        with open(f) as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                # Convert numeric fields
                for key in ["n_proteins", "total_aa", "n_annotated", "n_dark",
                            "n_domain_hits", "n_unique_pfam"]:
                    row[key] = int(row[key])
                for key in ["pct_dark", "pfam_per_orf", "pfam_per_1000aa"]:
                    row[key] = float(row[key])
                records.append(row)

    print(f"  Loaded {len(records)} sample records")
    return records

def classify_dataset_group(dataset):
    """Map fine-grained dataset labels into broader groups for Figure 1a comparison.

    Groups:
      - TARA: TARA_Oceans, TARA_protist, OSD (environmental metagenomic/amplicon)
      - MMETSP: MMETSP (cultured transcriptomes)
      - Reference: RefGenome_GenBank, Reference_Genome, RefGenome_PRE_REF, AAC
    """
    if dataset in ("TARA_Oceans", "TARA_protist", "OSD"):
        return "Environmental (TARA/OSD)"
    elif dataset == "MMETSP":
        return "Transcriptome (MMETSP)"
    elif dataset in ("RefGenome_GenBank", "Reference_Genome", "RefGenome_PRE_REF", "AAC"):
        return "Reference genome"
    else:
        return dataset

def classify_source_group(source_type):
    """Map source_type to broader groups."""
    if source_type == "MGYA_metagenome":
        return "Environmental (TARA metagenomes)"
    elif source_type == "MMETSP_transcriptome":
        return "Transcriptome (MMETSP)"
    elif source_type in ("GCA_genbank", "GCF_refseq"):
        return "Reference genome (NCBI)"
    elif source_type in ("Cultured_reference", "Euglenozoa_reference"):
        return "Cultured/Reference"
    else:
        return "Other"

def aggregate_group(records):
    """Compute aggregate statistics for a group of sample records."""
    n_samples = len(records)
    total_proteins = sum(r["n_proteins"] for r in records)
    total_aa = sum(r["total_aa"] for r in records)
    total_annotated = sum(r["n_annotated"] for r in records)
    total_dark = sum(r["n_dark"] for r in records)
    total_domain_hits = sum(r["n_domain_hits"] for r in records)

    # Collect per-sample unique Pfam counts (can't just sum — families overlap)
    mean_unique_pfam = (sum(r["n_unique_pfam"] for r in records) / n_samples
                        if n_samples > 0 else 0)

    pct_dark = 100.0 * total_dark / total_proteins if total_proteins > 0 else 0
    pfam_per_orf = total_domain_hits / total_proteins if total_proteins > 0 else 0
    pfam_per_1000aa = total_domain_hits * 1000.0 / total_aa if total_aa > 0 else 0

    # Per-sample pct_dark distribution
    pct_darks = sorted(r["pct_dark"] for r in records)
    median_pct_dark = pct_darks[n_samples // 2] if n_samples > 0 else 0

    return {
        "n_samples": n_samples,
        "total_proteins": total_proteins,
        "total_aa": total_aa,
        "total_annotated": total_annotated,
        "total_dark": total_dark,
        "total_domain_hits": total_domain_hits,
        "pct_dark": pct_dark,
        "median_pct_dark": median_pct_dark,
        "pfam_per_orf": pfam_per_orf,
        "pfam_per_1000aa": pfam_per_1000aa,
        "mean_unique_pfam_per_sample": mean_unique_pfam,
    }

def write_per_sample_aggregate(records, outpath):
    """Write single TSV with all samples."""
    fieldnames = ["sample", "dataset", "dataset_group", "source_type", "source_group",
                  "n_proteins", "total_aa", "n_annotated", "n_dark",
                  "n_domain_hits", "n_unique_pfam",
                  "pct_dark", "pfam_per_orf", "pfam_per_1000aa"]
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        for r in sorted(records, key=lambda x: x["sample"]):
            writer.writerow(r)
    print(f"  Wrote per-sample aggregate: {outpath} ({len(records)} rows)")

def write_group_summary(groups, outpath, group_key_name):
    """Write per-group summary TSV."""
    fieldnames = [group_key_name, "n_samples", "total_proteins", "total_aa",
                  "total_annotated", "total_dark", "total_domain_hits",
                  "pct_dark", "median_pct_dark",
                  "pfam_per_orf", "pfam_per_1000aa",
                  "mean_unique_pfam_per_sample"]

    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        # Sort groups by total_proteins descending for readability
        for group_name, stats in sorted(groups.items(),
                                        key=lambda x: -x[1]["total_proteins"]):
            row = {group_key_name: group_name}
            row.update(stats)
            # Format floats
            for k in ["pct_dark", "median_pct_dark", "pfam_per_orf",
                       "pfam_per_1000aa", "mean_unique_pfam_per_sample"]:
                row[k] = f"{row[k]:.4f}"
            writer.writerow(row)

        # Add global totals row
        all_records_stats = aggregate_group(
            [r for records_list in [[] for _ in groups] for r in records_list])
        # Actually recompute from all records passed in
    print(f"  Wrote group summary: {outpath} ({len(groups)} groups)")

def main():
    args = parse_args()

    # ---- Auto-detect metadata files ----
    # Try HPC paths first, then local MANUSCRIPT paths
    manuscript_base = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))  # scripts/dark_proteome/ -> MANUSCRIPT/

    master_key_path = args.master_key or find_file([
        os.path.join(manuscript_base, "omen-work",
                     "ASSEMBLY_GPS_MASTER_KEY_20260119_101500.tsv"),
        "omen-work/ASSEMBLY_GPS_MASTER_KEY_20260119_101500.tsv",
        "ASSEMBLY_GPS_MASTER_KEY_20260119_101500.tsv",
    ], "ASSEMBLY_GPS_MASTER_KEY")

    per_sample_path = args.per_sample_summary or find_file([
        os.path.join(manuscript_base, "source_data", "per_sample_summary.tsv"),
        "source_data/per_sample_summary.tsv",
        "per_sample_summary.tsv",
    ], "per_sample_summary.tsv")

    outdir = args.outdir or os.path.dirname(args.density_dir)

    print("=" * 60)
    print("  Pfam Density Aggregation")
    print("=" * 60)
    print(f"  Density dir:   {args.density_dir}")
    print(f"  Master key:    {master_key_path or 'NOT FOUND'}")
    print(f"  Sample summary:{per_sample_path or 'NOT FOUND'}")
    print(f"  Output dir:    {outdir}")
    print("")

    # ---- Load data ----
    print("Loading data...")
    if args.aggregate_tsv and os.path.isfile(args.aggregate_tsv):
        records = load_aggregate_tsv(args.aggregate_tsv)
    else:
        records = load_density_files(args.density_dir)
    dataset_map = load_master_key(master_key_path) if master_key_path else {}
    source_map = load_source_types(per_sample_path) if per_sample_path else {}

    # ---- Annotate records with dataset and source_type ----
    print("\nAnnotating samples with metadata...")
    n_dataset_matched = 0
    n_source_matched = 0

    for r in records:
        sample = r["sample"]

        # Dataset from master key
        dataset = dataset_map.get(sample, "Unknown")
        if dataset != "Unknown":
            n_dataset_matched += 1
        r["dataset"] = dataset
        r["dataset_group"] = classify_dataset_group(dataset)

        # Source type from per_sample_summary (or infer from name)
        source = source_map.get(sample)
        if source:
            n_source_matched += 1
        else:
            source = infer_source_type(sample)
        r["source_type"] = source
        r["source_group"] = classify_source_group(source)

    print(f"  Dataset matched:     {n_dataset_matched}/{len(records)}")
    print(f"  Source type matched:  {n_source_matched}/{len(records)}")
    print("")

    # ---- Write per-sample aggregate ----
    per_sample_out = os.path.join(outdir, "pfam_density_per_sample.tsv")
    write_per_sample_aggregate(records, per_sample_out)

    # ---- Group by dataset (fine-grained, from master key) ----
    by_dataset = defaultdict(list)
    for r in records:
        by_dataset[r["dataset"]].append(r)

    dataset_stats = {k: aggregate_group(v) for k, v in by_dataset.items()}
    dataset_out = os.path.join(outdir, "pfam_density_by_dataset.tsv")
    write_group_summary(dataset_stats, dataset_out, "dataset")

    # ---- Group by dataset_group (broad: Environmental vs MMETSP vs Reference) ----
    by_group = defaultdict(list)
    for r in records:
        by_group[r["dataset_group"]].append(r)

    group_stats = {k: aggregate_group(v) for k, v in by_group.items()}
    group_out = os.path.join(outdir, "pfam_density_by_dataset_group.tsv")
    write_group_summary(group_stats, group_out, "dataset_group")

    # ---- Group by source_type (MGYA, MMETSP, GCA, etc.) ----
    by_source = defaultdict(list)
    for r in records:
        by_source[r["source_type"]].append(r)

    source_stats = {k: aggregate_group(v) for k, v in by_source.items()}
    source_out = os.path.join(outdir, "pfam_density_by_source_type.tsv")
    write_group_summary(source_stats, source_out, "source_type")

    # ---- Print summary to stdout ----
    print("\n" + "=" * 80)
    print("  SUMMARY: Pfam Annotation Density by Dataset Group")
    print("=" * 80)
    print(f"{'Group':<35} {'Samples':>8} {'Proteins':>14} {'%Dark':>8} "
          f"{'Pfam/ORF':>10} {'Pfam/1kaa':>10}")
    print("-" * 80)

    # Global row
    global_stats = aggregate_group(records)
    print(f"{'GLOBAL':.<35} {global_stats['n_samples']:>8,} "
          f"{global_stats['total_proteins']:>14,} "
          f"{global_stats['pct_dark']:>7.2f}% "
          f"{global_stats['pfam_per_orf']:>10.4f} "
          f"{global_stats['pfam_per_1000aa']:>10.4f}")
    print("-" * 80)

    for group_name in sorted(group_stats.keys()):
        s = group_stats[group_name]
        print(f"{group_name:<35} {s['n_samples']:>8,} "
              f"{s['total_proteins']:>14,} "
              f"{s['pct_dark']:>7.2f}% "
              f"{s['pfam_per_orf']:>10.4f} "
              f"{s['pfam_per_1000aa']:>10.4f}")

    print("")
    print("=" * 80)
    print("  SUMMARY: Pfam Annotation Density by Source Type")
    print("=" * 80)
    print(f"{'Source Type':<30} {'Samples':>8} {'Proteins':>14} {'%Dark':>8} "
          f"{'Pfam/ORF':>10} {'Pfam/1kaa':>10}")
    print("-" * 80)

    for src in sorted(source_stats.keys()):
        s = source_stats[src]
        print(f"{src:<30} {s['n_samples']:>8,} "
              f"{s['total_proteins']:>14,} "
              f"{s['pct_dark']:>7.2f}% "
              f"{s['pfam_per_orf']:>10.4f} "
              f"{s['pfam_per_1000aa']:>10.4f}")

    print("")
    print("=" * 80)
    print("  SUMMARY: Pfam Annotation Density by Dataset (fine-grained)")
    print("=" * 80)
    print(f"{'Dataset':<25} {'Samples':>8} {'Proteins':>14} {'%Dark':>8} "
          f"{'Pfam/ORF':>10} {'Pfam/1kaa':>10}")
    print("-" * 80)

    for ds in sorted(dataset_stats.keys()):
        s = dataset_stats[ds]
        print(f"{ds:<25} {s['n_samples']:>8,} "
              f"{s['total_proteins']:>14,} "
              f"{s['pct_dark']:>7.2f}% "
              f"{s['pfam_per_orf']:>10.4f} "
              f"{s['pfam_per_1000aa']:>10.4f}")

    print("")
    print("Output files:")
    print(f"  {per_sample_out}")
    print(f"  {dataset_out}")
    print(f"  {group_out}")
    print(f"  {source_out}")
    print("")

if __name__ == "__main__":
    main()
