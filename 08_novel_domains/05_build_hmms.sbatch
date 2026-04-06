#!/bin/bash
#SBATCH --job-name=build_hmms
#SBATCH --output=logs/novel_domains/05_build_hmms_%A_%a.out
#SBATCH --error=logs/novel_domains/05_build_hmms_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=compute
#SBATCH --array=1-100%50

# =============================================================================
# Step 5: Build HMM Profiles from Large Clusters
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_domains/05_build_hmms.sbatch
#   Generated: 2026-02-21
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Purpose:
#   For each large cluster (>=10 members) from Step 4 (30% identity tier):
#   1. Extract member sequences
#   2. Align with MAFFT (--auto)
#   3. Build HMM profile with hmmbuild
#   Array job processes clusters in batches of ~1000.
#
# Input:
#   - 03_analyses/novel_domains/clusters_30/clusters_30_cluster.tsv
#   - 03_analyses/novel_domains/clusters_30/large_clusters.txt
#   - 03_analyses/novel_domains/dark_proteome_filtered.fa
#
# Output:
#   - 03_analyses/novel_domains/hmms/per_cluster/{clusterID}.hmm
#   - 03_analyses/novel_domains/cluster_alignments/{clusterID}.afa
#   - 03_analyses/novel_domains/hmms/novel_domains.hmm (combined, after all done)
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   # First, count large clusters and adjust array size:
#   #   N=$(wc -l < 03_analyses/novel_domains/clusters_30/large_clusters.txt)
#   #   NBATCHES=$(( (N + 999) / 1000 ))
#   #   sbatch --array=1-${NBATCHES}%50 MANUSCRIPT/scripts/novel_domains/05_build_hmms.sbatch
#   sbatch MANUSCRIPT/scripts/novel_domains/05_build_hmms.sbatch
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CRITICAL ENVIRONMENT SETUP
# =============================================================================

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp/hmm_build_$$
mkdir -p $TMPDIR
export PYTHONUSERBASE=/scratch/drn2/newhome/.local

module purge
module load gcc/13.2.0
module load hmmer/3.3

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

# =============================================================================
# Configuration
# =============================================================================

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
NOVEL_DIR="${BASE}/03_analyses/novel_domains"
CLUSTER_TSV="${NOVEL_DIR}/clusters_30/clusters_30_cluster.tsv"
LARGE_CLUSTERS="${NOVEL_DIR}/clusters_30/large_clusters.txt"
INPUT_FA="${NOVEL_DIR}/dark_proteome_filtered.fa"
HMM_DIR="${NOVEL_DIR}/hmms/per_cluster"
ALN_DIR="${NOVEL_DIR}/cluster_alignments"
BATCH_SIZE=1000

mkdir -p "${HMM_DIR}" "${ALN_DIR}"

echo "========================================"
echo "Step 5: Build HMM Profiles"
echo "========================================"
echo ""
echo "Job ID:     ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "Date:       $(date)"
echo ""

# =============================================================================
# Validate inputs
# =============================================================================

if [[ ! -f "${LARGE_CLUSTERS}" ]]; then
    echo "ERROR: large_clusters.txt not found: ${LARGE_CLUSTERS}"
    exit 1
fi

TOTAL_CLUSTERS=$(wc -l < "${LARGE_CLUSTERS}")
echo "  Total large clusters: ${TOTAL_CLUSTERS}"

# Calculate batch range for this array task
START_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) * BATCH_SIZE + 1 ))
END_IDX=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
if [[ ${END_IDX} -gt ${TOTAL_CLUSTERS} ]]; then
    END_IDX=${TOTAL_CLUSTERS}
fi

if [[ ${START_IDX} -gt ${TOTAL_CLUSTERS} ]]; then
    echo "  Task ${SLURM_ARRAY_TASK_ID}: no clusters to process (start=${START_IDX} > total=${TOTAL_CLUSTERS})"
    exit 0
fi

echo "  Processing clusters ${START_IDX}-${END_IDX} (batch ${SLURM_ARRAY_TASK_ID})"
echo ""

# =============================================================================
# Build a sequence index for fast extraction
# =============================================================================

echo "  Building sequence index..."

# Create a sequence index: extract all sequences into a lookup-friendly format
# We use mmseqs createdb + extract, or simpler: build a Python dict
# For efficiency at scale, use samtools faidx if available

if command -v samtools &>/dev/null && [[ ! -f "${INPUT_FA}.fai" ]]; then
    samtools faidx "${INPUT_FA}"
fi

# =============================================================================
# Process each cluster in this batch
# =============================================================================

N_SUCCESS=0
N_FAIL=0
N_SKIP=0

BATCH_CLUSTERS=$(sed -n "${START_IDX},${END_IDX}p" "${LARGE_CLUSTERS}")

for REP_ID in ${BATCH_CLUSTERS}; do
    # Skip if HMM already exists
    if [[ -f "${HMM_DIR}/${REP_ID}.hmm" ]]; then
        N_SKIP=$((N_SKIP + 1))
        continue
    fi

    # Extract member IDs for this cluster
    MEMBER_IDS=$(mktemp "${TMPDIR}/members_XXXX.txt")
    awk -v rep="${REP_ID}" '$1 == rep {print $2}' "${CLUSTER_TSV}" > "${MEMBER_IDS}"
    N_MEMBERS=$(wc -l < "${MEMBER_IDS}")

    if [[ ${N_MEMBERS} -lt 3 ]]; then
        # Need at least 3 sequences for a meaningful alignment
        rm -f "${MEMBER_IDS}"
        N_SKIP=$((N_SKIP + 1))
        continue
    fi

    # Extract sequences
    MEMBER_FA=$(mktemp "${TMPDIR}/members_XXXX.fa")

    if command -v samtools &>/dev/null && [[ -f "${INPUT_FA}.fai" ]]; then
        # Fast indexed extraction
        xargs -a "${MEMBER_IDS}" samtools faidx "${INPUT_FA}" > "${MEMBER_FA}" 2>/dev/null || true
    elif command -v seqtk &>/dev/null; then
        seqtk subseq "${INPUT_FA}" "${MEMBER_IDS}" > "${MEMBER_FA}"
    else
        # Python fallback
        python3 -c "
ids = set(open('${MEMBER_IDS}').read().split())
keep = False
with open('${INPUT_FA}') as fin, open('${MEMBER_FA}', 'w') as fout:
    for line in fin:
        if line.startswith('>'):
            keep = line[1:].split()[0] in ids
        if keep:
            fout.write(line)
"
    fi

    N_EXTRACTED=$(grep -c "^>" "${MEMBER_FA}" 2>/dev/null || echo 0)
    if [[ ${N_EXTRACTED} -lt 3 ]]; then
        rm -f "${MEMBER_IDS}" "${MEMBER_FA}"
        N_SKIP=$((N_SKIP + 1))
        continue
    fi

    # Align with MAFFT
    ALN_FILE="${ALN_DIR}/${REP_ID}.afa"
    mafft --auto --thread 2 --quiet "${MEMBER_FA}" > "${ALN_FILE}" 2>/dev/null

    if [[ ! -s "${ALN_FILE}" ]]; then
        echo "  WARNING: MAFFT failed for cluster ${REP_ID}"
        rm -f "${MEMBER_IDS}" "${MEMBER_FA}" "${ALN_FILE}"
        N_FAIL=$((N_FAIL + 1))
        continue
    fi

    # Build HMM
    HMM_FILE="${HMM_DIR}/${REP_ID}.hmm"
    hmmbuild --amino --cpu 2 -n "NOVEL_${REP_ID}" \
        "${HMM_FILE}" "${ALN_FILE}" > /dev/null 2>&1

    if [[ -f "${HMM_FILE}" ]]; then
        N_SUCCESS=$((N_SUCCESS + 1))
    else
        echo "  WARNING: hmmbuild failed for cluster ${REP_ID}"
        N_FAIL=$((N_FAIL + 1))
    fi

    # Cleanup temp files
    rm -f "${MEMBER_IDS}" "${MEMBER_FA}"
done

echo ""
echo "  Batch ${SLURM_ARRAY_TASK_ID} results:"
echo "    Successful HMMs: ${N_SUCCESS}"
echo "    Skipped:         ${N_SKIP}"
echo "    Failed:          ${N_FAIL}"
echo ""
echo "  Done: $(date)"

# Cleanup
rm -rf "${TMPDIR}"
