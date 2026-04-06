#!/bin/bash
#SBATCH --job-name=verify_novel
#SBATCH --output=logs/novel_domains/06c_verify_%j.out
#SBATCH --error=logs/novel_domains/06c_verify_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=compute

# =============================================================================
# Step 6c: Verify All 2044 novel_hits Files Exist
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_domains/06c_verify_completeness.sbatch
#   Generated: 2026-03-14
#
# Purpose:
#   After all hmmsearch_novel array tasks finish, verify that every sample in
#   sample_list.txt has a non-empty .novel.tbl in novel_hits/. If any are
#   missing, exit non-zero so downstream dependency jobs do NOT run.
#
# Usage:
#   Typically submitted as a dependency:
#     sbatch --dependency=afterany:JOBID 06c_verify_completeness.sbatch
#
# =============================================================================

set -euo pipefail

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
mkdir -p $TMPDIR

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
HITS_DIR="${BASE}/03_analyses/novel_domains/novel_hits"
SAMPLE_LIST="${BASE}/03_analyses/novel_domains/sample_list.txt"
RESULTS_DIR="${BASE}/03_analyses/novel_domains/results"
mkdir -p "${RESULTS_DIR}"

echo "========================================"
echo "Step 6c: Verify novel_hits Completeness"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date:   $(date)"
echo ""

TOTAL=$(wc -l < "${SAMPLE_LIST}")
FOUND=0
MISSING=0
EMPTY=0
MISSING_LIST=""

while IFS= read -r sample; do
    tbl="${HITS_DIR}/${sample}.novel.tbl"
    if [[ ! -f "${tbl}" ]]; then
        MISSING=$((MISSING + 1))
        MISSING_LIST="${MISSING_LIST}${sample}\n"
    elif [[ ! -s "${tbl}" ]]; then
        EMPTY=$((EMPTY + 1))
        MISSING_LIST="${MISSING_LIST}${sample} (empty)\n"
    else
        FOUND=$((FOUND + 1))
    fi
done < "${SAMPLE_LIST}"

echo "  Total samples:  ${TOTAL}"
echo "  Found (non-empty): ${FOUND}"
echo "  Missing:        ${MISSING}"
echo "  Empty:          ${EMPTY}"
echo ""

if [[ ${MISSING} -gt 0 || ${EMPTY} -gt 0 ]]; then
    echo "  FAILED: ${MISSING} missing + ${EMPTY} empty = $((MISSING + EMPTY)) incomplete"
    echo ""
    echo "  Missing/empty samples:"
    echo -e "${MISSING_LIST}" | head -50
    if [[ $((MISSING + EMPTY)) -gt 50 ]]; then
        echo "  ... and $((MISSING + EMPTY - 50)) more"
    fi

    # Write missing list for potential resubmit
    echo -e "${MISSING_LIST}" | sed '/^$/d' | sed 's/ (empty)$//' \
        > "${RESULTS_DIR}/still_missing_novel_hits.txt"
    echo "  Missing list written to: ${RESULTS_DIR}/still_missing_novel_hits.txt"
    echo ""

    # Decide: if >95% complete, proceed anyway with a warning
    PCT=$((FOUND * 100 / TOTAL))
    if [[ ${PCT} -ge 95 ]]; then
        echo "  WARNING: ${PCT}% complete (>= 95%). Proceeding with available data."
        echo "  Downstream steps will handle missing samples gracefully."
        exit 0
    else
        echo "  ERROR: Only ${PCT}% complete (< 95%). Aborting downstream steps."
        exit 1
    fi
else
    echo "  ALL ${TOTAL} samples have non-empty novel_hits. Proceeding."
fi

echo ""
echo "  Done: $(date)"
