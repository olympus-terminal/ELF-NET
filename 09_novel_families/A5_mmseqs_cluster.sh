#!/bin/bash
# =============================================================================
# A5: MMseqs2 Clustering at 30% Identity
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_families/A5_mmseqs_cluster.sh
#   Generated: 2026-02-21
#   Pipeline: Novel Domain Discovery — Track A
#
# Purpose:
#   Cluster unannotated neighbor proteins at 30% sequence identity using
#   MMseqs2. This groups divergent homologs into novel protein families.
#
# Parameters:
#   --min-seq-id 0.30   30% minimum sequence identity
#   -c 0.80             80% minimum coverage
#   --cov-mode 0        Coverage of query AND target
#   --cluster-mode 0    Greedy set cover
#
# Input:
#   novel_families/data/anchor_neighborhoods/unannotated_neighbor_proteins.fasta.gz
#
# Output:
#   novel_families/data/mmseqs_clusters/cluster_membership.tsv
#   novel_families/data/mmseqs_clusters/cluster_reps.fasta
#   novel_families/data/mmseqs_clusters/cluster_sizes.tsv
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   bash MANUSCRIPT/scripts/novel_families/A5_mmseqs_cluster.sh
#
# =============================================================================

set -euo pipefail

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
INPUT_FASTA="${BASE}/novel_families/data/anchor_neighborhoods/unannotated_neighbor_proteins.fasta.gz"
OUT_DIR="${BASE}/novel_families/data/mmseqs_clusters"
TMP_DIR="${TMPDIR:-/scratch/drn2/tmp}/mmseqs_nf_$$"

mkdir -p "${OUT_DIR}" "${TMP_DIR}"

echo "=============================================="
echo "  A5: MMseqs2 Clustering (30% identity)"
echo "=============================================="
echo ""
echo "Date:   $(date)"
echo "Host:   $(hostname)"
echo "Input:  ${INPUT_FASTA}"
echo "Output: ${OUT_DIR}"
echo "Temp:   ${TMP_DIR}"
echo ""

# Verify input
if [[ ! -f "${INPUT_FASTA}" ]]; then
    echo "ERROR: Input FASTA not found: ${INPUT_FASTA}"
    exit 1
fi

# Decompress if needed
FASTA_UNZIPPED="${TMP_DIR}/input.fasta"
if [[ "${INPUT_FASTA}" == *.gz ]]; then
    echo "  Decompressing input..."
    zcat "${INPUT_FASTA}" > "${FASTA_UNZIPPED}"
else
    cp "${INPUT_FASTA}" "${FASTA_UNZIPPED}"
fi

N_SEQS=$(grep -c "^>" "${FASTA_UNZIPPED}")
echo "  Input sequences: ${N_SEQS}"
echo ""

# Create MMseqs2 database
echo "--- Step 1: Create MMseqs2 database ---"
DB="${TMP_DIR}/seqDB"
mmseqs createdb "${FASTA_UNZIPPED}" "${DB}"
echo ""

# Run clustering
echo "--- Step 2: Cluster at 30% identity ---"
CLUSTER_DB="${TMP_DIR}/clusterDB"
mmseqs cluster "${DB}" "${CLUSTER_DB}" "${TMP_DIR}/tmp_cluster" \
    --min-seq-id 0.30 \
    -c 0.80 \
    --cov-mode 0 \
    --cluster-mode 0 \
    --threads "${SLURM_CPUS_PER_TASK:-28}"
echo ""

# Extract cluster membership TSV
echo "--- Step 3: Extract cluster membership ---"
MEMBERSHIP_TSV="${OUT_DIR}/cluster_membership.tsv"
mmseqs createtsv "${DB}" "${DB}" "${CLUSTER_DB}" "${MEMBERSHIP_TSV}"

N_CLUSTERS=$(awk '{print $1}' "${MEMBERSHIP_TSV}" | sort -u | wc -l)
N_MEMBERS=$(wc -l < "${MEMBERSHIP_TSV}")
echo "  Total clusters: ${N_CLUSTERS}"
echo "  Total membership records: ${N_MEMBERS}"
echo ""

# Extract representative sequences
echo "--- Step 4: Extract representative sequences ---"
REP_DB="${TMP_DIR}/repDB"
mmseqs result2repseq "${DB}" "${CLUSTER_DB}" "${REP_DB}"
mmseqs result2flat "${DB}" "${DB}" "${REP_DB}" "${OUT_DIR}/cluster_reps.fasta" --use-fasta-header
echo "  Representatives written: $(grep -c "^>" "${OUT_DIR}/cluster_reps.fasta")"
echo ""

# Compute cluster sizes
echo "--- Step 5: Compute cluster size distribution ---"
SIZES_TSV="${OUT_DIR}/cluster_sizes.tsv"
awk '{print $1}' "${MEMBERSHIP_TSV}" | sort | uniq -c | sort -rn | \
    awk '{print $2"\t"$1}' > "${SIZES_TSV}"

echo "  Cluster size distribution:"
echo "    Singletons (size=1):   $(awk '$2==1' "${SIZES_TSV}" | wc -l)"
echo "    Size 2-9:              $(awk '$2>=2 && $2<10' "${SIZES_TSV}" | wc -l)"
echo "    Size 10-49:            $(awk '$2>=10 && $2<50' "${SIZES_TSV}" | wc -l)"
echo "    Size 50-99:            $(awk '$2>=50 && $2<100' "${SIZES_TSV}" | wc -l)"
echo "    Size 100-499:          $(awk '$2>=100 && $2<500' "${SIZES_TSV}" | wc -l)"
echo "    Size 500-999:          $(awk '$2>=500 && $2<1000' "${SIZES_TSV}" | wc -l)"
echo "    Size >=1000:           $(awk '$2>=1000' "${SIZES_TSV}" | wc -l)"
echo ""

# Largest clusters
echo "  Top 20 clusters by size:"
head -20 "${SIZES_TSV}" | awk '{printf "    %-40s %s\n", $1, $2}'
echo ""

# Verification
echo "--- Verification ---"
echo "  Input sequences: ${N_SEQS}"
echo "  Clustered records: ${N_MEMBERS}"
if [[ ${N_MEMBERS} -eq ${N_SEQS} ]]; then
    echo "  PASS: All input sequences assigned to clusters"
else
    echo "  WARNING: ${N_MEMBERS} records vs ${N_SEQS} input sequences"
fi

# Cleanup temp files (keep output)
echo ""
echo "  Cleaning temp directory..."
rm -rf "${TMP_DIR}"

echo ""
echo "  Output files:"
echo "    ${MEMBERSHIP_TSV}"
echo "    ${OUT_DIR}/cluster_reps.fasta"
echo "    ${SIZES_TSV}"
echo ""
echo "  Done: $(date)"
