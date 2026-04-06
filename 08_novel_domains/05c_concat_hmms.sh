#!/bin/bash
# =============================================================================
# Step 5c: Concatenate and Press HMM Database
# =============================================================================
#
# Run after ALL Step 5b array tasks complete.
# Concatenates per-cluster HMMs into a single database and presses it.
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   bash MANUSCRIPT/scripts/novel_domains/05c_concat_hmms.sh
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
HMM_DIR="${BASE}/03_analyses/novel_domains/hmms"
PER_CLUSTER="${HMM_DIR}/per_cluster"
COMBINED="${HMM_DIR}/novel_domains.hmm"

echo "========================================"
echo "Step 5c: Concatenate & Press HMM Database"
echo "========================================"
echo ""
echo "Date: $(date)"
echo ""

# Count individual HMMs
N_HMMS=$(find "${PER_CLUSTER}" -name '*.hmm' -type f | wc -l)
echo "  Individual HMMs found: ${N_HMMS}"

if [[ ${N_HMMS} -eq 0 ]]; then
    echo "  ERROR: No HMM files found in ${PER_CLUSTER}/"
    exit 1
fi

# Concatenate (use find+xargs to avoid ARG_MAX with many files)
echo "  Concatenating into ${COMBINED}..."
find "${PER_CLUSTER}" -name '*.hmm' -type f -print0 | sort -z | xargs -0 cat > "${COMBINED}"
echo "  Combined file size: $(du -sh "${COMBINED}" | awk '{print $1}')"

# Press for hmmsearch
echo "  Pressing HMM database..."
hmmpress -f "${COMBINED}"

echo ""
echo "  HMM database files:"
ls -lh "${HMM_DIR}"/novel_domains.hmm* | awk '{print "    "$5" "$NF}'
echo ""
echo "  Total novel domain HMMs: ${N_HMMS}"
echo "  Done: $(date)"
