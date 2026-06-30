#!/bin/bash
# =============================================================================
# Submit Downstream Chain: Verify → 06b+07 → 08
# =============================================================================
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   bash MANUSCRIPT/scripts/novel_domains/submit_downstream_chain.sh HMMSEARCH_JOBID
#
# Example:
#   bash MANUSCRIPT/scripts/novel_domains/submit_downstream_chain.sh 14364986
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="MANUSCRIPT/scripts/novel_domains"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 HMMSEARCH_JOBID"
    echo "  HMMSEARCH_JOBID = the array job ID from 06_resubmit_missing.sbatch"
    exit 1
fi

HMMSEARCH_JOB="$1"

echo "========================================"
echo "Novel Domains: Downstream Chain"
echo "========================================"
echo ""
echo "  Waiting on hmmsearch job: ${HMMSEARCH_JOB}"
echo ""

mkdir -p logs/novel_domains

# Step 6c: Verify all 2044 novel_hits exist
# afterany = run even if some array tasks failed (we check inside)
VERIFY=$(sbatch --parsable \
    --dependency=afterany:${HMMSEARCH_JOB} \
    "${SCRIPT_DIR}/06c_verify_completeness.sbatch")
echo "  Step 6c (verify):       ${VERIFY}  [dep: afterany:${HMMSEARCH_JOB}]"

# Step 07 (includes 06b count matrix): depends on verify success
CHAR=$(sbatch --parsable \
    --dependency=afterok:${VERIFY} \
    "${SCRIPT_DIR}/07_characterize.sbatch")
echo "  Step 07 (06b+char):     ${CHAR}  [dep: afterok:${VERIFY}]"

# Step 08: depends on characterize success
ENV=$(sbatch --parsable \
    --dependency=afterok:${CHAR} \
    "${SCRIPT_DIR}/08_env_correlation.sbatch")
echo "  Step 08 (env corr):     ${ENV}  [dep: afterok:${CHAR}]"

echo ""
echo "  Chain submitted. Monitor with:"
echo "    squeue -u drn2"
echo "    sacct -j ${VERIFY},${CHAR},${ENV} --format=JobID,JobName%30,State,ExitCode,Elapsed"
echo ""

# Save state
STATE_FILE="/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains/pipeline_state_downstream.txt"
cat > "${STATE_FILE}" <<EOF
# Pipeline State — Downstream Chain
# Generated: $(date -Iseconds)
HMMSEARCH_RESUBMIT=${HMMSEARCH_JOB}
VERIFY=${VERIFY}
CHARACTERIZE=${CHAR}
ENV_CORRELATION=${ENV}
# Chain: hmmsearch → verify → 06b+07 → 08
EOF

echo "  State saved to: ${STATE_FILE}"
