#!/usr/bin/env python3
"""
Step 7: Characterize Novel Domain Families

Provenance:
    Script: scripts/novel_domains/07_characterize.py
    Generated: 2026-02-21

Purpose:
    Comprehensive characterization of novel protein domain families discovered
    by MMseqs2 clustering of the Pfam-dark proteome:
    1. Size distribution (expect power-law)
    2. Length distribution per cluster
    3. Geographic breadth (TARA stations per family)
    4. Sample prevalence (filter >=10 samples)
    5. Taxonomic context (LA4SR classification labels)
    6. DIAMOND cross-reference (truly novel vs Pfam-missing)
    7. Comparison to known Pfam domain prevalence

Input:
    - 03_analyses/novel_domains/clusters_30/clusters_30_cluster.tsv
    - 03_analyses/novel_domains/clusters_30/cluster_sizes.tsv
    - 03_analyses/novel_domains/results/novel_domain_count_matrix.tsv
    - 03_analyses/novel_domains/dark_proteome_filtered.fa
    - 03_analyses/novel_domains/dark_ids/*.dark_stats.tsv
    - 01_raw_data/metadata/ALL_assemblies_GPS_mapping.tsv
    - 03_analyses/hmmsearch_results/pfam_count_matrix_*.tsv
    - 03_analyses/dark_proteome_v2/ (if exists)

Output:
    - 03_analyses/novel_domains/results/novel_domain_summary.tsv
    - 03_analyses/novel_domains/results/novel_vs_pfam_comparison.tsv
    - 03_analyses/novel_domains/results/dark_proteome_overview.tsv
    - 03_analyses/novel_domains/results/cluster_size_distribution.tsv

Usage:
    python3 scripts/novel_domains/07_characterize.py

Can also be submitted via SLURM wrapper (07_characterize.sbatch).
"""

import os
import sys
import glob
from collections import defaultdict, Counter
from pathlib import Path
import csv

# Try numpy/pandas but fall back to pure Python if unavailable
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

BASE = Path("/scratch/drn2/PROJECTS/TARA-LA4SR")
NOVEL_DIR = BASE / "03_analyses/novel_domains"
RESULTS_DIR = NOVEL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_cluster_membership(cluster_tsv):
    """Load cluster TSV: {representative: [member1, member2, ...]}."""
    clusters = defaultdict(list)
    with open(cluster_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                clusters[rep].append(member)
    return dict(clusters)

def load_cluster_sizes(sizes_tsv):
    """Load cluster_sizes.tsv: {representative: size}."""
    sizes = {}
    with open(sizes_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sizes[parts[0]] = int(parts[1])
    return sizes

def load_dark_stats():
    """Load per-sample dark proteome statistics."""
    stats = {}
    stats_dir = NOVEL_DIR / "dark_ids"
    for f in stats_dir.glob("*.dark_stats.tsv"):
        with open(f) as fh:
            header = fh.readline()
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 5 and parts[0] != "sample":
                    stats[parts[0]] = {
                        "total": int(parts[1]),
                        "pfam_hits": int(parts[2]),
                        "dark": int(parts[3]),
                        "pct_dark": float(parts[4]),
                    }
    return stats

def load_gps_metadata():
    """Load GPS/station metadata for geographic analysis."""
    meta = {}
    meta_patterns = [
        BASE / "01_raw_data/metadata/ALL_assemblies_GPS_mapping.tsv",
        BASE / "MANUSCRIPT/omen-work/ASSEMBLY_GPS_MASTER_KEY_*.tsv",
    ]
    for pattern in meta_patterns:
        for f in glob.glob(str(pattern)):
            with open(f) as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    sample_id = row.get("assembly_id", row.get("sample", ""))
                    if sample_id:
                        meta[sample_id] = {
                            "lat": row.get("latitude", ""),
                            "lon": row.get("longitude", ""),
                            "station": row.get("station", row.get("TARA_station", "")),
                            "dataset": row.get("dataset", ""),
                        }
            break  # Use first match
    return meta

def load_novel_count_matrix():
    """Load the novel domain count matrix."""
    matrix_path = RESULTS_DIR / "novel_domain_count_matrix.tsv"
    if not matrix_path.exists():
        return None, None, None

    with open(matrix_path) as f:
        header = f.readline().strip().split("\t")
        domains = header[1:]  # skip assembly_id
        samples = []
        counts = []
        for line in f:
            parts = line.strip().split("\t")
            samples.append(parts[0])
            counts.append([int(x) for x in parts[1:]])

    return samples, domains, counts

def load_pfam_count_matrix():
    """Load Pfam count matrix for comparison."""
    pfam_pattern = BASE / "03_analyses/hmmsearch_results/pfam_count_matrix_*.tsv"
    pfam_files = sorted(glob.glob(str(pfam_pattern)))
    if not pfam_files:
        return None

    # Load just the prevalence info (avoid loading full matrix into memory)
    pfam_prevalence = {}
    with open(pfam_files[-1]) as f:
        header = f.readline().strip().split("\t")
        # Find Pfam columns (PF#####)
        pfam_cols = [i for i, h in enumerate(header) if h.startswith("PF")]
        pfam_names = [header[i] for i in pfam_cols]

        # Initialize presence counters
        presence = defaultdict(int)
        n_samples = 0

        for line in f:
            parts = line.strip().split("\t")
            n_samples += 1
            for idx in pfam_cols:
                if idx < len(parts) and parts[idx] not in ("0", "", "0.0"):
                    presence[header[idx]] += 1

    return {name: presence[name] for name in pfam_names}, n_samples

def check_diamond_dark(sample_clusters):
    """Cross-reference with DIAMOND dark_proteome_v2 results."""
    dark_v2_dir = BASE / "03_analyses/dark_proteome_v2"
    if not dark_v2_dir.exists():
        return None

    # Load DIAMOND no-hit IDs
    diamond_dark_ids = set()
    for f in dark_v2_dir.glob("*no_hit*"):
        with open(f) as fh:
            for line in fh:
                diamond_dark_ids.add(line.strip())

    if not diamond_dark_ids:
        # Try alternative pattern
        for f in dark_v2_dir.glob("*.dark_ids.txt"):
            with open(f) as fh:
                for line in fh:
                    diamond_dark_ids.add(line.strip())

    return diamond_dark_ids if diamond_dark_ids else None

def main():
    print("=" * 60)
    print("  Step 7: Characterize Novel Domain Families")
    print("=" * 60)
    print()

    # =====================================================================
    # 1. Load data
    # =====================================================================

    print("Loading data...")

    # Cluster membership and sizes
    cluster_tsv = NOVEL_DIR / "clusters_30/clusters_30_cluster.tsv"
    sizes_tsv = NOVEL_DIR / "clusters_30/cluster_sizes.tsv"

    if not cluster_tsv.exists():
        print(f"  ERROR: Cluster TSV not found: {cluster_tsv}")
        sys.exit(1)

    clusters = load_cluster_membership(cluster_tsv)
    sizes = load_cluster_sizes(sizes_tsv) if sizes_tsv.exists() else {
        k: len(v) for k, v in clusters.items()
    }

    print(f"  Clusters loaded:   {len(clusters):,}")
    print(f"  Total members:     {sum(sizes.values()):,}")
    print()

    # Dark proteome stats
    dark_stats = load_dark_stats()
    print(f"  Dark stats loaded: {len(dark_stats)} samples")

    # GPS metadata
    gps_meta = load_gps_metadata()
    print(f"  GPS metadata:      {len(gps_meta)} samples")

    # Novel domain count matrix
    mat_samples, mat_domains, mat_counts = load_novel_count_matrix()
    if mat_samples:
        print(f"  Novel count matrix: {len(mat_samples)} samples x {len(mat_domains)} domains")
    else:
        print("  Novel count matrix: not available (run 06b first)")

    # Pfam comparison
    pfam_data = load_pfam_count_matrix()
    if pfam_data:
        pfam_prevalence, pfam_n_samples = pfam_data
        print(f"  Pfam domains:      {len(pfam_prevalence)} (across {pfam_n_samples} samples)")
    else:
        pfam_prevalence, pfam_n_samples = None, None
        print("  Pfam count matrix: not found")

    print()

    # =====================================================================
    # 2. Cluster size distribution
    # =====================================================================

    print("--- Cluster Size Distribution ---")

    size_counts = Counter(sizes.values())
    n_singletons = size_counts.get(1, 0)
    n_large = sum(1 for s in sizes.values() if s >= 10)
    n_very_large = sum(1 for s in sizes.values() if s >= 100)
    n_massive = sum(1 for s in sizes.values() if s >= 1000)

    print(f"  Total clusters:        {len(sizes):>10,}")
    print(f"  Singletons:            {n_singletons:>10,} ({100*n_singletons/len(sizes):.1f}%)")
    print(f"  2-9 members:           {sum(1 for s in sizes.values() if 2 <= s < 10):>10,}")
    print(f"  10-99 members:         {sum(1 for s in sizes.values() if 10 <= s < 100):>10,}")
    print(f"  100-999 members:       {sum(1 for s in sizes.values() if 100 <= s < 1000):>10,}")
    print(f"  1000+ members:         {n_massive:>10,}")
    print(f"  Largest cluster:       {max(sizes.values()):>10,}")
    print()

    # Write distribution
    dist_path = RESULTS_DIR / "cluster_size_distribution.tsv"
    with open(dist_path, "w") as f:
        f.write("cluster_size\tn_clusters\n")
        for size in sorted(size_counts):
            f.write(f"{size}\t{size_counts[size]}\n")
    print(f"  Written: {dist_path}")
    print()

    # =====================================================================
    # 3. Dark proteome overview
    # =====================================================================

    print("--- Dark Proteome Overview ---")

    if dark_stats:
        total_seqs = sum(s["total"] for s in dark_stats.values())
        total_dark = sum(s["dark"] for s in dark_stats.values())
        total_pfam = sum(s["pfam_hits"] for s in dark_stats.values())
        global_pct = 100.0 * total_dark / total_seqs if total_seqs > 0 else 0

        print(f"  Total proteins:        {total_seqs:>12,}")
        print(f"  Pfam-annotated:        {total_pfam:>12,} ({100*total_pfam/total_seqs:.1f}%)")
        print(f"  Pfam-dark:             {total_dark:>12,} ({global_pct:.1f}%)")
        print()

        # Per-sample stats
        pct_darks = [s["pct_dark"] for s in dark_stats.values()]
        pct_darks.sort()
        median_pct = pct_darks[len(pct_darks)//2]
        print(f"  Per-sample %dark: min={min(pct_darks):.1f}%, "
              f"median={median_pct:.1f}%, max={max(pct_darks):.1f}%")

        # Write overview
        overview_path = RESULTS_DIR / "dark_proteome_overview.tsv"
        with open(overview_path, "w") as f:
            f.write("metric\tvalue\n")
            f.write(f"total_proteins\t{total_seqs}\n")
            f.write(f"pfam_annotated\t{total_pfam}\n")
            f.write(f"pfam_dark\t{total_dark}\n")
            f.write(f"pct_dark\t{global_pct:.2f}\n")
            f.write(f"n_samples\t{len(dark_stats)}\n")
            f.write(f"median_pct_dark_per_sample\t{median_pct:.2f}\n")
            f.write(f"total_clusters_30pct\t{len(sizes)}\n")
            f.write(f"large_clusters_ge10\t{n_large}\n")
        print(f"  Written: {overview_path}")
    print()

    # =====================================================================
    # 4. Geographic breadth (if count matrix available)
    # =====================================================================

    if mat_samples and gps_meta:
        print("--- Geographic Breadth ---")

        # Map samples to stations
        sample_stations = {}
        for s in mat_samples:
            if s in gps_meta and gps_meta[s].get("station"):
                sample_stations[s] = gps_meta[s]["station"]

        n_with_station = len(sample_stations)
        n_stations = len(set(sample_stations.values()))
        print(f"  Samples with station info: {n_with_station}")
        print(f"  Unique stations: {n_stations}")

    # =====================================================================
    # 5. Novel domain summary table
    # =====================================================================

    print("--- Novel Domain Summary ---")

    if mat_samples and mat_domains:
        summary_rows = []
        for j, domain in enumerate(mat_domains):
            # Sample prevalence
            n_samples_present = sum(1 for i in range(len(mat_samples))
                                    if mat_counts[i][j] > 0)
            # Total abundance
            total_abundance = sum(mat_counts[i][j] for i in range(len(mat_samples)))
            # Mean abundance (in samples where present)
            present_counts = [mat_counts[i][j] for i in range(len(mat_samples))
                              if mat_counts[i][j] > 0]
            mean_abundance = (sum(present_counts) / len(present_counts)
                              if present_counts else 0)

            # Cluster size (from direct clustering)
            cluster_size = sizes.get(domain.replace("NOVEL_", ""), 0)

            summary_rows.append({
                "domain": domain,
                "cluster_size": cluster_size,
                "n_samples": n_samples_present,
                "total_abundance": total_abundance,
                "mean_abundance_when_present": f"{mean_abundance:.1f}",
            })

        # Sort by sample prevalence
        summary_rows.sort(key=lambda x: x["n_samples"], reverse=True)

        summary_path = RESULTS_DIR / "novel_domain_summary.tsv"
        with open(summary_path, "w") as f:
            cols = ["domain", "cluster_size", "n_samples",
                    "total_abundance", "mean_abundance_when_present"]
            f.write("\t".join(cols) + "\n")
            for row in summary_rows:
                f.write("\t".join(str(row[c]) for c in cols) + "\n")

        n_prevalent = sum(1 for r in summary_rows if r["n_samples"] >= 10)
        print(f"  Total novel domains in matrix:  {len(mat_domains)}")
        print(f"  Domains in >=10 samples:        {n_prevalent}")
        print(f"  Top 10 most prevalent:")
        for r in summary_rows[:10]:
            print(f"    {r['domain']}: {r['n_samples']} samples, "
                  f"{r['total_abundance']} hits")
        print(f"  Written: {summary_path}")
    print()

    # =====================================================================
    # 6. Comparison to Pfam
    # =====================================================================

    if pfam_prevalence and mat_domains:
        print("--- Novel vs Pfam Comparison ---")

        # Novel domain prevalence distribution
        novel_prev = []
        for j, domain in enumerate(mat_domains):
            n = sum(1 for i in range(len(mat_samples)) if mat_counts[i][j] > 0)
            novel_prev.append(n)

        # Pfam prevalence distribution
        pfam_prev_vals = sorted(pfam_prevalence.values(), reverse=True)

        # How many novel domains are as prevalent as the median Pfam domain?
        pfam_median = pfam_prev_vals[len(pfam_prev_vals) // 2]
        novel_above_pfam_median = sum(1 for n in novel_prev if n >= pfam_median)

        print(f"  Pfam median prevalence:          {pfam_median} samples")
        print(f"  Novel domains >= Pfam median:     {novel_above_pfam_median}")
        print(f"  Novel domains in top 10% of Pfam: "
              f"{sum(1 for n in novel_prev if n >= pfam_prev_vals[len(pfam_prev_vals)//10])}")

        # Write comparison
        comp_path = RESULTS_DIR / "novel_vs_pfam_comparison.tsv"
        with open(comp_path, "w") as f:
            f.write("metric\tpfam\tnovel\n")
            f.write(f"total_domains\t{len(pfam_prevalence)}\t{len(mat_domains)}\n")
            f.write(f"domains_ge10_samples\t"
                    f"{sum(1 for v in pfam_prevalence.values() if v >= 10)}\t"
                    f"{sum(1 for n in novel_prev if n >= 10)}\n")
            f.write(f"domains_ge100_samples\t"
                    f"{sum(1 for v in pfam_prevalence.values() if v >= 100)}\t"
                    f"{sum(1 for n in novel_prev if n >= 100)}\n")
            f.write(f"median_prevalence\t{pfam_median}\t"
                    f"{sorted(novel_prev)[len(novel_prev)//2]}\n")
            f.write(f"max_prevalence\t{max(pfam_prev_vals)}\t{max(novel_prev)}\n")
        print(f"  Written: {comp_path}")
    print()

    # =====================================================================
    # 7. DIAMOND cross-reference
    # =====================================================================

    diamond_dark = check_diamond_dark(clusters)
    if diamond_dark:
        print("--- DIAMOND Cross-Reference ---")
        print(f"  DIAMOND nr-dark protein IDs: {len(diamond_dark):,}")

        # For each cluster, check what fraction of members are also nr-dark
        n_checked = 0
        n_fully_dark = 0  # All members are nr-dark
        n_partially_dark = 0

        for rep, members in clusters.items():
            if sizes.get(rep, 0) < 10:
                continue
            n_checked += 1
            n_dark_members = sum(1 for m in members if m in diamond_dark)
            frac = n_dark_members / len(members) if members else 0
            if frac > 0.9:
                n_fully_dark += 1
            elif frac > 0:
                n_partially_dark += 1

        print(f"  Large clusters checked: {n_checked}")
        print(f"  >90% nr-dark (truly novel): {n_fully_dark}")
        print(f"  Partially nr-dark:          {n_partially_dark}")
        print(f"  Pfam-only dark (have nr hits): {n_checked - n_fully_dark - n_partially_dark}")
    else:
        print("--- DIAMOND cross-reference: not available ---")
    print()

    # =====================================================================
    # Summary
    # =====================================================================

    print("=" * 60)
    print("  Characterization Complete")
    print("=" * 60)
    print()
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"  Files generated:")
    for f in sorted(RESULTS_DIR.glob("*.tsv")):
        size = f.stat().st_size
        print(f"    {f.name} ({size:,} bytes)")
    print()

if __name__ == "__main__":
    main()
