#!/bin/bash
# Submit rank-binned ESMFold pipeline:
#   Step 1: Sample ~3,600 domains across 10 rank bins (quick, runs on login node)
#   Step 2: Submit ESMFold API prediction job via sbatch
#
# Run from: /scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses"
NOVEL_DIR="${BASE}/novel_domains"

echo "=============================================="
echo "  Rank-Binned ESMFold Pipeline"
echo "  Date: $(date)"
echo "=============================================="

# Ensure log directory exists
mkdir -p "${NOVEL_DIR}/logs"

# Step 1: Run sampling script
echo ""
echo "Step 1: Rank-bin sampling..."
cd "${BASE}"
python3 "${SCRIPT_DIR}/11_rankbin_esmfold_sample_20260530_085412.py"

# Verify output
FASTA="${NOVEL_DIR}/results/esmfold_rankbin/rankbin_sequences.fasta"
if [[ ! -f "${FASTA}" ]]; then
    echo "ERROR: Sampling script did not produce ${FASTA}"
    exit 1
fi

N_SEQ=$(grep -c "^>" "${FASTA}")
echo ""
echo "Sampling produced ${N_SEQ} sequences"

# Step 2: Submit ESMFold prediction
echo ""
echo "Step 2: Submitting ESMFold prediction job..."
cd "${BASE}"
JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/11b_esmfold_rankbin_predict_20260530_085412.sbatch")
echo "  Submitted job: ${JOB_ID}"
echo "  Monitor: squeue -j ${JOB_ID}"
echo "  Logs: ${NOVEL_DIR}/logs/11b_esmfold_rankbin_${JOB_ID}.out"

echo ""
echo "=============================================="
echo "  Pipeline submitted. Job ID: ${JOB_ID}"
echo "  Estimated time: 5-10 hours for ~${N_SEQ} sequences"
echo "=============================================="
