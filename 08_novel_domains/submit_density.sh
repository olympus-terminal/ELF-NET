#!/bin/bash
# =============================================================================
# Submit Pfam density array job
# =============================================================================
#
# Provenance:
#   Script: scripts/dark_proteome/submit_density.sh
#   Generated: 2026-03-01
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Prerequisites:
#   - Run 00_audit.sh first to generate sample_list.txt
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   bash MANUSCRIPT/scripts/dark_proteome/submit_density.sh
#
# =============================================================================

set -euo pipefail

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
SCRIPT_DIR="${BASE}/MANUSCRIPT/scripts/dark_proteome"
OUT_DIR="${BASE}/03_analyses/dark_proteome"
LOG_DIR="${BASE}/logs/dark_proteome"
SAMPLE_LIST="${OUT_DIR}/sample_list.txt"

echo "========================================"
echo "  Pfam Density — Submit"
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

# ---- Create directories ----
mkdir -p "${LOG_DIR}" "${OUT_DIR}/pfam_density"

# ---- Submit ----
JOB_ID=$(sbatch --parsable \
    --array=1-${N_SAMPLES}%200 \
    "${SCRIPT_DIR}/02_pfam_density.sbatch")

echo "Submitted SLURM array job: ${JOB_ID}"
echo "  Array range: 1-${N_SAMPLES}%200"
echo "  Log dir:     ${LOG_DIR}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$(whoami) -n pfam_dens"
echo "  sacct -j ${JOB_ID} --format=JobID,State,Elapsed,MaxRSS"
echo ""
echo "After completion, aggregate with:"
echo "  python3 MANUSCRIPT/scripts/dark_proteome/03_aggregate_dataset_stats.py"
echo ""
echo "Submitted: $(date)"
