#!/bin/bash
#SBATCH --job-name=hmmsearch_novel
#SBATCH --output=logs/novel_domains/06_hmmsearch_novel_%A_%a.out
#SBATCH --error=logs/novel_domains/06_hmmsearch_novel_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=90G
#SBATCH --partition=compute
#SBATCH --array=1-2044%50

# =============================================================================
# Step 6: Search Novel HMMs Against Full Dataset (Per-Sample)
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_domains/06_hmmsearch_novel.sbatch
#   Generated: 2026-02-21
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Purpose:
#   Search the novel domain HMM database against ALL algal proteins (not just
#   dark proteome) per sample. This:
#   - Recovers members missed by linclust (which is approximate)
#   - Detects novel domains in multi-domain proteins that also have Pfam hits
#   - Produces a per-sample hit table for building the count matrix
#
# Input:
#   - 03_analyses/novel_domains/hmms/novel_domains.hmm (from Step 5b)
#   - 03_analyses/algae_proteins/{sample}.fa (all proteins)
#
# Output:
#   - 03_analyses/novel_domains/novel_hits/{sample}.novel.tbl (per-sample hits)
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   sbatch MANUSCRIPT/scripts/novel_domains/06_hmmsearch_novel.sbatch
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CRITICAL ENVIRONMENT SETUP
# =============================================================================

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
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
ALGAL_DIR="${BASE}/03_analyses/algae_proteins"
HMM_DB="${BASE}/03_analyses/novel_domains/hmms/novel_domains.hmm"
OUT_DIR="${BASE}/03_analyses/novel_domains/novel_hits"
SAMPLE_LIST="${BASE}/03_analyses/novel_domains/sample_list.txt"
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
echo "Step 6: hmmsearch Novel Domains"
echo "========================================"
echo ""
echo "Job ID:     ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Sample:     ${SAMPLE}"
echo "Date:       $(date)"
echo ""

# =============================================================================
# Validate inputs
# =============================================================================

# Find FASTA (actual naming: {stem}.algae.fa)
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
    echo "  Run Step 5b first."
    exit 1
fi

# Check that HMM is pressed
if [[ ! -f "${HMM_DB}.h3m" ]]; then
    echo "ERROR: HMM database not pressed (missing .h3m)"
    echo "  Run: hmmpress ${HMM_DB}"
    exit 1
fi

echo "  FASTA:  ${FASTA} ($(grep -c '^>' "${FASTA}") sequences)"
echo "  HMM DB: ${HMM_DB}"
echo "  E-value: ${EVALUE}"
echo ""

# =============================================================================
# Run hmmsearch
# =============================================================================

OUTPUT_TBL="${OUT_DIR}/${SAMPLE}.novel.tbl"

# Skip if already done
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

echo ""
echo "  hmmsearch exit code: ${EXIT_CODE}"
echo "  Elapsed: $((END - START)) seconds"
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
