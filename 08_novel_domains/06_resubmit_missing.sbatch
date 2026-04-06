#!/bin/bash
#SBATCH --job-name=hmmsearch_novel
#SBATCH --output=logs/novel_domains/06_resubmit_%A_%a.out
#SBATCH --error=logs/novel_domains/06_resubmit_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=90G
#SBATCH --partition=compute
#SBATCH --array=1-514%50

# =============================================================================
# Step 6 RESUBMIT: Search Novel HMMs — 514 Missing Samples
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_domains/06_resubmit_missing.sbatch
#   Generated: 2026-03-14
#   Based on: 06_hmmsearch_novel.sbatch
#
# Changes from original:
#   - Walltime: 48h (was 24h — 21 samples timed out at 24h)
#   - Array: 1-514 (reads from missing_novel_hits_514.txt, not sample_list.txt)
#   - Removes skip-if-exists guard (we know these are missing)
#
# Input:
#   - 03_analyses/novel_domains/missing_novel_hits_514.txt (list of 514 samples)
#   - 03_analyses/novel_domains/hmms/novel_domains.hmm
#   - 03_analyses/algae_proteins/{sample}.{ext}
#
# Output:
#   - 03_analyses/novel_domains/novel_hits/{sample}.novel.tbl
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   mkdir -p logs/novel_domains
#   sbatch MANUSCRIPT/scripts/novel_domains/06_resubmit_missing.sbatch
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CRITICAL ENVIRONMENT SETUP
# =============================================================================

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
export PYTHONUSERBASE=/scratch/drn2/newhome/.local
mkdir -p $TMPDIR
unset NETWORKX_BACKEND_CONFIG
unset NX_BACKEND_CONFIG

module purge
module load gcc/13.2.0
module load hmmer/3.3

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

# =============================================================================
# Configuration
# =============================================================================

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
ALGAL_DIR="${BASE}/03_analyses/algae_proteins"
HMM_DB="${BASE}/03_analyses/novel_domains/hmms/novel_domains.hmm"
OUT_DIR="${BASE}/03_analyses/novel_domains/novel_hits"
SAMPLE_LIST="${BASE}/03_analyses/novel_domains/missing_novel_hits_514.txt"
THREADS=28
EVALUE="1e-5"

mkdir -p "${OUT_DIR}"

# =============================================================================
# Get sample for this array task
# =============================================================================

SAMPLE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SAMPLE_LIST}")

if [[ -z "${SAMPLE}" ]]; then
    echo "ERROR: No sample at line ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "========================================"
echo "Step 6 RESUBMIT: hmmsearch Novel Domains"
echo "========================================"
echo ""
echo "Job ID:     ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Sample:     ${SAMPLE}"
echo "Date:       $(date)"
echo ""

# =============================================================================
# Validate inputs
# =============================================================================

FASTA=""
for ext in ".algae.fa" ".fa" ".faa" ".fasta"; do
    if [[ -f "${ALGAL_DIR}/${SAMPLE}${ext}" ]]; then
        FASTA="${ALGAL_DIR}/${SAMPLE}${ext}"
        break
    fi
done

if [[ -z "${FASTA}" || ! -f "${FASTA}" ]]; then
    echo "ERROR: FASTA not found for sample ${SAMPLE} in ${ALGAL_DIR}"
    exit 1
fi

if [[ ! -f "${HMM_DB}" ]]; then
    echo "ERROR: HMM database not found: ${HMM_DB}"
    exit 1
fi

if [[ ! -f "${HMM_DB}.h3m" ]]; then
    echo "ERROR: HMM database not pressed (missing .h3m)"
    exit 1
fi

N_SEQS=$(grep -c '^>' "${FASTA}")
echo "  FASTA:    ${FASTA} (${N_SEQS} sequences)"
echo "  HMM DB:   ${HMM_DB}"
echo "  E-value:  ${EVALUE}"
echo "  Walltime: 48h"
echo ""

# =============================================================================
# Run hmmsearch
# =============================================================================

OUTPUT_TBL="${OUT_DIR}/${SAMPLE}.novel.tbl"

# Safety: skip if somehow already exists (e.g. parallel submission)
if [[ -f "${OUTPUT_TBL}" && -s "${OUTPUT_TBL}" ]]; then
    echo "  Output already exists, skipping: ${OUTPUT_TBL}"
    echo "  Done: $(date)"
    exit 0
fi

START=$(date +%s)

hmmsearch \
    --tblout "${OUTPUT_TBL}" \
    --noali \
    -E ${EVALUE} \
    --cpu ${THREADS} \
    "${HMM_DB}" \
    "${FASTA}" \
    > /dev/null

EXIT_CODE=$?
END=$(date +%s)
ELAPSED=$((END - START))

echo ""
echo "  hmmsearch exit code: ${EXIT_CODE}"
echo "  Elapsed: ${ELAPSED} seconds ($(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m)"
echo ""

if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "  ERROR: hmmsearch failed"
    exit ${EXIT_CODE}
fi

# =============================================================================
# Parse results
# =============================================================================

if [[ -f "${OUTPUT_TBL}" ]]; then
    N_HITS=$(grep -cv "^#" "${OUTPUT_TBL}" 2>/dev/null || echo 0)
    N_PROTEINS=$(grep -v "^#" "${OUTPUT_TBL}" | awk '{print $1}' | sort -u | wc -l)
    N_DOMAINS=$(grep -v "^#" "${OUTPUT_TBL}" | awk '{print $3}' | sort -u | wc -l)

    echo "  Results:"
    echo "    Total hits:       ${N_HITS}"
    echo "    Unique proteins:  ${N_PROTEINS}"
    echo "    Unique domains:   ${N_DOMAINS}"
    echo "    Output: ${OUTPUT_TBL}"
fi

echo ""
echo "  Done: $(date)"
