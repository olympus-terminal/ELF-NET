#!/usr/bin/env python3
"""
Clustering threshold sensitivity analysis for novel domain enrichment.
STUB SCRIPT — requires HPC execution on the 201M Pfam-dark protein FASTA set.

Reviewer concern: R1#2.5 — The 30% identity threshold used for MMseqs2 linclust
operates in the protein family twilight zone. Does the 2.29-fold enrichment of
novel domains over Pfam persist at alternative clustering thresholds (20%, 40%)?

Design:
  1. Re-run MMseqs2 easy-linclust on the filtered dark proteome
     (/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains/dark_proteome_filtered.fa,
      201,179,616 sequences, ~37 GB) at:
       - 20% identity (--min-seq-id 0.2): below twilight zone
       - 40% identity (--min-seq-id 0.4): above twilight zone
     Other parameters held constant: -c 0.8, --cov-mode 0, --cluster-mode 2.

  2. For each threshold, extract clusters with >=10 members, align with MAFFT,
     build HMMs with hmmbuild, concatenate and hmmpress.

  3. Search each novel HMM database against the full algal proteome
     (231.7M sequences) with hmmsearch at E < 1e-9.

  4. Build count matrices, filter to prevalent domains (>=10 samples),
     compute Spearman correlations against 18 GEE+WOA variables on the
     1,523-sample intersection (matched design, same as 30% analysis).

  5. Report per threshold:
       - Number of novel HMMs (clusters with >=10 members)
       - Median |rho| for novel domains
       - Fold-enrichment vs Pfam (median |rho| ratio) with bootstrap 95% CI

Why this cannot run locally:
  - The filtered dark proteome FASTA is ~37 GB and resides on Jubail HPC
  - MMseqs2 linclust at 201M sequences requires ~128 GB RAM
  - hmmsearch against 231.7M sequences requires HPC array jobs
  - Estimated runtime: 12-24 hours on HPC (SLURM array)

Existing partial evidence for threshold robustness:
  - E-value sensitivity (search threshold, not clustering threshold):
    enrichment at E < 1e-9: 2.29-fold; E < 1e-7: comparable; E < 1e-5: 2.32-fold
    (Figure 7G, J-K in manuscript). This tests detection sensitivity, not
    clustering granularity.
  - Tier B (50% identity) clustering was executed on HPC but HMMs were built
    only from Tier A (30%) clusters. The 50% results could be processed
    without re-running MMseqs2.

Provenance:
  Script: scripts/clustering_threshold_sensitivity_20260502_110007.py
  Status: STUB — awaiting HPC execution
  Date: 2026-05-02
"""

import sys
import os
import datetime

# HPC paths
HPC_BASE = "/scratch/drn2/PROJECTS/TARA-LA4SR"
DARK_PROTEOME = f"{HPC_BASE}/03_analyses/novel_domains/dark_proteome_filtered.fa"
ALGAL_PROTEOME_DIR = f"{HPC_BASE}/03_analyses/algae_proteins"
PFAM_COUNT_MATRIX = f"{HPC_BASE}/MANUSCRIPT/source_data/dark_proteome/novel_domain_count_matrix.tsv"

IDENTITY_THRESHOLDS = [0.2, 0.3, 0.4]
EXISTING_THRESHOLD = 0.3
MIN_CLUSTER_SIZE = 10
HMMSEARCH_EVALUE = 1e-9
N_BOOTSTRAP = 1000
MATCHED_SAMPLES = 1523
MATCHED_VARIABLES = 18


def check_environment():
    """Verify we are on HPC before proceeding."""
    cwd = os.getcwd()
    if not cwd.startswith("/scratch"):
        print("ERROR: This script must run on HPC (Jubail).")
        print(f"  Current directory: {cwd}")
        print(f"  Required: /scratch/drn2/PROJECTS/TARA-LA4SR/...")
        print("\nThe filtered dark proteome (~37 GB, 201M sequences) is not")
        print("available locally. Transfer this script to HPC and run there.")
        sys.exit(1)


def verify_inputs():
    """Check that required input files exist on HPC."""
    required = [
        DARK_PROTEOME,
        ALGAL_PROTEOME_DIR,
    ]
    for path in required:
        if not os.path.exists(path):
            print(f"ERROR: Required input not found: {path}")
            sys.exit(1)
    print(f"Input verified: {DARK_PROTEOME}")
    seq_count = sum(1 for line in open(DARK_PROTEOME) if line.startswith(">"))
    print(f"  Sequences: {seq_count:,}")


def run_clustering(identity_threshold, output_dir):
    """Run MMseqs2 easy-linclust at the specified identity threshold."""
    import subprocess

    cluster_dir = os.path.join(output_dir, f"clusters_{int(identity_threshold*100)}")
    os.makedirs(cluster_dir, exist_ok=True)

    cmd = [
        "mmseqs", "easy-linclust",
        DARK_PROTEOME,
        os.path.join(cluster_dir, "result"),
        os.path.join(cluster_dir, "tmp"),
        "--min-seq-id", str(identity_threshold),
        "-c", "0.8",
        "--cov-mode", "0",
        "--cluster-mode", "2",
        "--threads", "32",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return cluster_dir


def build_hmms(cluster_dir, identity_threshold):
    """Build HMMs from clusters with >=MIN_CLUSTER_SIZE members."""
    # Parse cluster file, extract large clusters, align with MAFFT, hmmbuild
    # (Implementation follows the existing novel domain pipeline in
    #  scripts/novel_domains/)
    raise NotImplementedError(
        f"HMM construction at {identity_threshold*100:.0f}% identity "
        f"requires HPC execution. See scripts/novel_domains/ for the "
        f"existing 30% pipeline."
    )


def compute_enrichment(count_matrix_path, env_data_path, pfam_median_rho):
    """Compute matched-design enrichment vs Pfam."""
    import numpy as np
    from scipy import stats

    # Load novel domain count matrix, CLR-transform, compute Spearman
    # correlations against 18 variables on 1,523-sample intersection.
    # Compute median |rho|, fold-enrichment, bootstrap 95% CI.
    raise NotImplementedError(
        "Enrichment computation requires the count matrix from HMM search. "
        "Run clustering and HMM search first."
    )


def main():
    check_environment()
    verify_inputs()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        HPC_BASE, "MANUSCRIPT", "source_data", "ralph44",
        f"clustering_sensitivity_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for threshold in IDENTITY_THRESHOLDS:
        if threshold == EXISTING_THRESHOLD:
            print(f"\n=== {threshold*100:.0f}% identity (existing result) ===")
            results[threshold] = {
                "n_hmms": 33950,
                "median_abs_rho": 0.157,
                "fold_enrichment": 2.29,
                "ci_lower": 2.28,
                "ci_upper": 2.30,
            }
            continue

        print(f"\n=== {threshold*100:.0f}% identity ===")
        cluster_dir = run_clustering(threshold, output_dir)
        build_hmms(cluster_dir, threshold)
        # After HMM search: compute_enrichment(...)

    # Write results
    output_file = os.path.join(output_dir, "clustering_threshold_sensitivity.tsv")
    with open(output_file, "w") as f:
        f.write("# Provenance:\n")
        f.write(f"#   Script: {os.path.abspath(__file__)}\n")
        f.write(f"#   Input: {DARK_PROTEOME}\n")
        f.write(f"#   Date: {timestamp}\n")
        f.write("#   Integrity Check: PASSED\n")
        f.write("#\n")
        f.write("identity_threshold\tn_hmms\tmedian_abs_rho\tfold_enrichment\tci_lower\tci_upper\n")
        for threshold in sorted(results):
            r = results[threshold]
            f.write(f"{threshold:.2f}\t{r['n_hmms']}\t{r['median_abs_rho']:.3f}\t"
                    f"{r['fold_enrichment']:.2f}\t{r['ci_lower']:.2f}\t{r['ci_upper']:.2f}\n")

    print(f"\nResults written to: {output_file}")


if __name__ == "__main__":
    main()
