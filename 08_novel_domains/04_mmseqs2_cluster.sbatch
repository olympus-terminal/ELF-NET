#!/bin/bash
#SBATCH --job-name=mmseqs_cluster
#SBATCH --output=logs/novel_domains/04_mmseqs2_cluster_%j.out
#SBATCH --error=logs/novel_domains/04_mmseqs2_cluster_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=90G
#SBATCH --partition=compute

# =============================================================================
# Step 4: MMseqs2 Clustering of Dark Proteome
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_domains/04_mmseqs2_cluster.sbatch
#   Generated: 2026-02-21
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Purpose:
#   Cluster the filtered dark proteome into novel protein families using
#   MMseqs2 linclust (linear-time clustering for 100M+ sequences).
#   Two-tier strategy:
#     Tier A: 30% sequence identity (broad families)
#     Tier B: 50% sequence identity (tight subfamilies)
#
# Input:
#   - 03_analyses/novel_domains/dark_proteome_filtered.fa (from Step 3)
#
# Output:
#   - 03_analyses/novel_domains/clusters_30/ (broad family clusters)
#   - 03_analyses/novel_domains/clusters_50/ (tight subfamily clusters)
#   - Cluster membership TSVs, representative FASTAs, size distributions
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   sbatch MANUSCRIPT/scripts/novel_domains/04_mmseqs2_cluster.sbatch
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
echo "Step 4: MMseqs2 Clustering"
echo "========================================"
echo ""
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "Date:       $(date)"
echo "Threads:    ${THREADS}"
echo "TMPDIR:     ${TMPDIR}"
echo ""

# Validate input
if [[ ! -f "${INPUT_FA}" ]]; then
    echo "ERROR: Filtered FASTA not found: ${INPUT_FA}"
    exit 1
fi

INPUT_SEQS=$(grep -c "^>" "${INPUT_FA}")
INPUT_SIZE=$(du -sh "${INPUT_FA}" | awk '{print $1}')
echo "Input:"
echo "  File:      ${INPUT_FA}"
echo "  Sequences: ${INPUT_SEQS}"
echo "  Size:      ${INPUT_SIZE}"
echo ""

# Check MMseqs2
if ! command -v mmseqs &>/dev/null; then
    echo "ERROR: mmseqs not found in PATH"
    echo "  Install: conda install -c bioconda mmseqs2"
    exit 1
fi
echo "MMseqs2: $(mmseqs version 2>&1 || echo 'version unknown')"
echo ""

# =============================================================================
# Tier A: Broad families (30% identity)
# =============================================================================

echo "========================================"
echo "Tier A: Broad families (30% identity)"
echo "========================================"
echo ""

CLUSTER_30_DIR="${OUT_BASE}/clusters_30"
mkdir -p "${CLUSTER_30_DIR}"

START=$(date +%s)

mmseqs easy-linclust \
    "${INPUT_FA}" \
    "${CLUSTER_30_DIR}/clusters_30" \
    "${TMPDIR}/tmp_30" \
    --min-seq-id 0.3 \
    -c 0.8 \
    --cov-mode 0 \
    --threads ${THREADS}

END=$(date +%s)
echo ""
echo "  Tier A elapsed: $((END - START)) seconds ($((( END - START ) / 60)) min)"
echo ""

# Parse cluster results
echo "  Tier A cluster statistics:"
# The output includes:
#   clusters_30_cluster.tsv    — representative → member mapping
#   clusters_30_rep_seq.fasta  — representative sequences
#   clusters_30_all_seqs.fasta — all sequences (rewritten)

if [[ -f "${CLUSTER_30_DIR}/clusters_30_cluster.tsv" ]]; then
    N_CLUSTERS_30=$(cut -f1 "${CLUSTER_30_DIR}/clusters_30_cluster.tsv" | sort -u | wc -l)
    N_MEMBERS_30=$(wc -l < "${CLUSTER_30_DIR}/clusters_30_cluster.tsv")
    echo "    Total clusters:  ${N_CLUSTERS_30}"
    echo "    Total members:   ${N_MEMBERS_30}"

    # Cluster size distribution
    echo ""
    echo "    Size distribution (top 20):"
    cut -f1 "${CLUSTER_30_DIR}/clusters_30_cluster.tsv" \
        | sort | uniq -c | sort -rn | head -20 \
        | awk '{printf "      Cluster %s: %d members\n", $2, $1}'

    # Size histogram
    echo ""
    echo "    Size histogram:"
    cut -f1 "${CLUSTER_30_DIR}/clusters_30_cluster.tsv" \
        | sort | uniq -c | awk '{print $1}' | sort -n | uniq -c | sort -rn | head -20 \
        | awk '{printf "      %8d clusters with %d members\n", $1, $2}'

    # Singletons
    N_SINGLETONS_30=$(cut -f1 "${CLUSTER_30_DIR}/clusters_30_cluster.tsv" \
        | sort | uniq -c | awk '$1==1' | wc -l)
    echo ""
    echo "    Singletons: ${N_SINGLETONS_30} ($(awk "BEGIN{printf \"%.1f\", 100.0*${N_SINGLETONS_30}/${N_CLUSTERS_30}")%)"

    # Clusters with >=10 members (candidates for HMM building)
    N_LARGE_30=$(cut -f1 "${CLUSTER_30_DIR}/clusters_30_cluster.tsv" \
        | sort | uniq -c | awk '$1>=10' | wc -l)
    echo "    Clusters >=10 members: ${N_LARGE_30}"
fi
echo ""

# Clean Tier A tmp
rm -rf "${TMPDIR}/tmp_30"

# =============================================================================
# Tier B: Tight subfamilies (50% identity)
# =============================================================================

echo "========================================"
echo "Tier B: Tight subfamilies (50% identity)"
echo "========================================"
echo ""

CLUSTER_50_DIR="${OUT_BASE}/clusters_50"
mkdir -p "${CLUSTER_50_DIR}"

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

if [[ -f "${CLUSTER_50_DIR}/clusters_50_cluster.tsv" ]]; then
    N_CLUSTERS_50=$(cut -f1 "${CLUSTER_50_DIR}/clusters_50_cluster.tsv" | sort -u | wc -l)
    N_MEMBERS_50=$(wc -l < "${CLUSTER_50_DIR}/clusters_50_cluster.tsv")
    echo "    Total clusters:  ${N_CLUSTERS_50}"
    echo "    Total members:   ${N_MEMBERS_50}"

    N_SINGLETONS_50=$(cut -f1 "${CLUSTER_50_DIR}/clusters_50_cluster.tsv" \
        | sort | uniq -c | awk '$1==1' | wc -l)
    echo "    Singletons: ${N_SINGLETONS_50} ($(awk "BEGIN{printf \"%.1f\", 100.0*${N_SINGLETONS_50}/${N_CLUSTERS_50}")%)"

    N_LARGE_50=$(cut -f1 "${CLUSTER_50_DIR}/clusters_50_cluster.tsv" \
        | sort | uniq -c | awk '$1>=10' | wc -l)
    echo "    Clusters >=10 members: ${N_LARGE_50}"
fi
echo ""

# Clean Tier B tmp
rm -rf "${TMPDIR}/tmp_50"

# =============================================================================
# Generate cluster size lists for downstream steps
# =============================================================================

echo "Generating cluster member count files..."

for TIER in 30 50; do
    CDIR="${OUT_BASE}/clusters_${TIER}"
    TSV="${CDIR}/clusters_${TIER}_cluster.tsv"
    if [[ -f "${TSV}" ]]; then
        # cluster_sizes.tsv: representative_id\tsize
        cut -f1 "${TSV}" | sort | uniq -c | sort -rn \
            | awk '{printf "%s\t%d\n", $2, $1}' \
            > "${CDIR}/cluster_sizes.tsv"

        # large_clusters.txt: representative IDs with >=10 members
        awk '$2 >= 10 {print $1}' "${CDIR}/cluster_sizes.tsv" \
            > "${CDIR}/large_clusters.txt"

        N_LARGE=$(wc -l < "${CDIR}/large_clusters.txt")
        echo "  Tier ${TIER}%: ${N_LARGE} large clusters (>=10 members)"
    fi
done

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
echo "    Total clusters:        ${N_CLUSTERS_30:-N/A}"
echo "    Singletons:            ${N_SINGLETONS_30:-N/A}"
echo "    Large (>=10 members):  ${N_LARGE_30:-N/A}"
echo ""
echo "  Tier B (50% identity):"
echo "    Total clusters:        ${N_CLUSTERS_50:-N/A}"
echo "    Singletons:            ${N_SINGLETONS_50:-N/A}"
echo "    Large (>=10 members):  ${N_LARGE_50:-N/A}"
echo ""
echo "  Output directory: ${OUT_BASE}"
echo ""
echo "  Done: $(date)"

# Cleanup
rm -rf "${TMPDIR}"
