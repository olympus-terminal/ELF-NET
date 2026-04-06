#!/bin/bash
# =============================================================================
# Submit Dark Proteome Extraction — One-liner Orchestrator
# =============================================================================
#
# Provenance:
#   Script: scripts/dark_proteome/submit.sh
#   Generated: 2026-02-21
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Prerequisites:
#   - Run 00_audit.sh first to generate sample_list.txt
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   bash MANUSCRIPT/scripts/dark_proteome/submit.sh
#
# =============================================================================

set -euo pipefail

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
SCRIPT_DIR="${BASE}/MANUSCRIPT/scripts/dark_proteome"
OUT_DIR="${BASE}/03_analyses/dark_proteome"
LOG_DIR="${BASE}/logs/dark_proteome"
SAMPLE_LIST="${OUT_DIR}/sample_list.txt"

echo "========================================"
echo "  Dark Proteome Extraction — Submit"
echo "========================================"
echo ""

# ---- Pre-flight checks ----
if [[ ! -f "${SAMPLE_LIST}" ]]; then
    echo "ERROR: sample_list.txt not found at ${SAMPLE_LIST}"
    echo "       Run 00_audit.sh first."
    exit 1
fi

N_SAMPLES=$(wc -l < "${SAMPLE_LIST}")
echo "Sample list: ${SAMPLE_LIST} (${N_SAMPLES} samples)"
echo ""

if [[ ${N_SAMPLES} -eq 0 ]]; then
    echo "ERROR: sample_list.txt is empty."
    exit 1
fi

# ---- Create output and log directories ----
mkdir -p "${LOG_DIR}" "${OUT_DIR}/dark_ids" "${OUT_DIR}/dark_fasta"

# ---- Submit array job with correct array size ----
JOB_ID=$(sbatch --parsable \
    --array=1-${N_SAMPLES}%100 \
    "${SCRIPT_DIR}/01_extract_dark.sbatch")

echo "Submitted SLURM array job: ${JOB_ID}"
echo "  Array range: 1-${N_SAMPLES}%100"
echo "  Log dir:     ${LOG_DIR}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$(whoami) -n dark_extract"
echo "  sacct -j ${JOB_ID} --format=JobID,State,Elapsed,MaxRSS"
echo ""
echo "After completion, aggregate stats with:"
echo "  head -1 ${OUT_DIR}/dark_ids/\$(head -1 ${SAMPLE_LIST}).dark_stats.tsv > ${OUT_DIR}/aggregate_stats.tsv"
echo "  for f in ${OUT_DIR}/dark_ids/*.dark_stats.tsv; do tail -1 \"\$f\"; done >> ${OUT_DIR}/aggregate_stats.tsv"
echo "  column -t ${OUT_DIR}/aggregate_stats.tsv | head -20"
echo ""
echo "Submitted: $(date)"
