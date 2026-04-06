#!/bin/bash
#SBATCH --job-name=mmseqs_rerun
#SBATCH --output=logs/novel_domains/04b_resubmit_%j.out
#SBATCH --error=logs/novel_domains/04b_resubmit_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=250G
#SBATCH --partition=compute

# =============================================================================
# Step 4b: Resubmit Clustering — Complete Tier A stats + run Tier B
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_domains/04b_resubmit_clustering.sbatch
#   Generated: 2026-02-21
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Purpose:
#   Original Step 4 (job 14173396) completed Tier A (30% clustering) but crashed
#   during statistics computation due to SIGPIPE from head on massive pipe with
#   set -euo pipefail. Exit code 13 (SIGPIPE signal).
#
#   This script:
#     1. Generates Tier A statistics files (cluster_sizes.tsv, large_clusters.txt)
#     2. Runs Tier B (50% identity clustering)
#     3. Generates Tier B statistics files
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CRITICAL ENVIRONMENT SETUP
# =============================================================================

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp/mmseqs_$$
mkdir -p $TMPDIR
export PYTHONUSERBASE=/scratch/drn2/newhome/.local

module purge
module load gcc/13.2.0

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

# =============================================================================
# Configuration
# =============================================================================

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
INPUT_FA="${BASE}/03_analyses/novel_domains/dark_proteome_filtered.fa"
OUT_BASE="${BASE}/03_analyses/novel_domains"
THREADS=28

echo "========================================"
echo "Step 4b: Clustering Resubmission"
echo "========================================"
echo ""
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "Date:       $(date)"
echo "Threads:    ${THREADS}"
echo "TMPDIR:     ${TMPDIR}"
echo ""

# =============================================================================
# Part 1: Generate Tier A statistics (30% clusters already exist)
# =============================================================================

echo "========================================"
echo "Part 1: Tier A statistics"
echo "========================================"
echo ""

CLUSTER_30_DIR="${OUT_BASE}/clusters_30"
TSV_30="${CLUSTER_30_DIR}/clusters_30_cluster.tsv"

if [[ -f "${TSV_30}" ]]; then
    echo "  Tier A cluster TSV found: $(wc -l < "${TSV_30}") lines"
    echo "  Generating cluster_sizes.tsv..."

    # Use awk to count cluster sizes in a single pass (memory-efficient)
    awk -F'\t' '{count[$1]++} END {for (c in count) print c "\t" count[c]}' \
        "${TSV_30}" | sort -t$'\t' -k2,2rn \
        > "${CLUSTER_30_DIR}/cluster_sizes.tsv"

    N_CLUSTERS_30=$(wc -l < "${CLUSTER_30_DIR}/cluster_sizes.tsv")
    echo "  Total clusters (30%): ${N_CLUSTERS_30}"

    # Large clusters (>=10 members)
    awk -F'\t' '$2 >= 10 {print $1}' "${CLUSTER_30_DIR}/cluster_sizes.tsv" \
        > "${CLUSTER_30_DIR}/large_clusters.txt"

    N_LARGE_30=$(wc -l < "${CLUSTER_30_DIR}/large_clusters.txt")
    echo "  Large clusters (>=10 members): ${N_LARGE_30}"

    # Singletons
    N_SINGLETONS_30=$(awk -F'\t' '$2 == 1' "${CLUSTER_30_DIR}/cluster_sizes.tsv" | wc -l)
    echo "  Singletons: ${N_SINGLETONS_30}"
    echo ""
else
    echo "ERROR: Tier A cluster TSV not found: ${TSV_30}"
    exit 1
fi

# =============================================================================
# Part 2: Tier B clustering (50% identity)
# =============================================================================

echo "========================================"
echo "Tier B: Tight subfamilies (50% identity)"
echo "========================================"
echo ""

CLUSTER_50_DIR="${OUT_BASE}/clusters_50"
mkdir -p "${CLUSTER_50_DIR}"

INPUT_SEQS=$(grep -c "^>" "${INPUT_FA}")
echo "  Input sequences: ${INPUT_SEQS}"
echo ""

START=$(date +%s)

mmseqs easy-linclust \
    "${INPUT_FA}" \
    "${CLUSTER_50_DIR}/clusters_50" \
    "${TMPDIR}/tmp_50" \
    --min-seq-id 0.5 \
    -c 0.8 \
    --cov-mode 0 \
    --threads ${THREADS}

END=$(date +%s)
echo ""
echo "  Tier B elapsed: $((END - START)) seconds ($((( END - START ) / 60)) min)"
echo ""

# =============================================================================
# Part 3: Tier B statistics
# =============================================================================

TSV_50="${CLUSTER_50_DIR}/clusters_50_cluster.tsv"

if [[ -f "${TSV_50}" ]]; then
    echo "  Generating Tier B cluster_sizes.tsv..."

    awk -F'\t' '{count[$1]++} END {for (c in count) print c "\t" count[c]}' \
        "${TSV_50}" | sort -t$'\t' -k2,2rn \
        > "${CLUSTER_50_DIR}/cluster_sizes.tsv"

    N_CLUSTERS_50=$(wc -l < "${CLUSTER_50_DIR}/cluster_sizes.tsv")
    echo "  Total clusters (50%): ${N_CLUSTERS_50}"

    awk -F'\t' '$2 >= 10 {print $1}' "${CLUSTER_50_DIR}/cluster_sizes.tsv" \
        > "${CLUSTER_50_DIR}/large_clusters.txt"

    N_LARGE_50=$(wc -l < "${CLUSTER_50_DIR}/large_clusters.txt")
    echo "  Large clusters (>=10 members): ${N_LARGE_50}"

    N_SINGLETONS_50=$(awk -F'\t' '$2 == 1' "${CLUSTER_50_DIR}/cluster_sizes.tsv" | wc -l)
    echo "  Singletons: ${N_SINGLETONS_50}"
    echo ""
else
    echo "ERROR: Tier B cluster TSV not found after MMseqs2 run"
    exit 1
fi

# Clean tmp
rm -rf "${TMPDIR}"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "========================================"
echo "  Clustering Summary"
echo "========================================"
echo ""
echo "  Input sequences:         ${INPUT_SEQS}"
echo ""
echo "  Tier A (30% identity):"
echo "    Total clusters:        ${N_CLUSTERS_30}"
echo "    Singletons:            ${N_SINGLETONS_30}"
echo "    Large (>=10 members):  ${N_LARGE_30}"
echo ""
echo "  Tier B (50% identity):"
echo "    Total clusters:        ${N_CLUSTERS_50}"
echo "    Singletons:            ${N_SINGLETONS_50}"
echo "    Large (>=10 members):  ${N_LARGE_50}"
echo ""
echo "  Done: $(date)"
