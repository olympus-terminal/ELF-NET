#!/bin/bash
# =============================================================================
# Master Launch Script: Track B — Dark Proteome Environment Coupling
# =============================================================================
#
# Trigger: Run after dark proteome agent delivers FASTA output.
# Total wall time: ~60 hours.
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   bash MANUSCRIPT/scripts/novel_families/launch_track_b.sh
#
# =============================================================================

set -euo pipefail

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
SCRIPTS="${BASE}/MANUSCRIPT/scripts/novel_families"

mkdir -p novel_families/logs
mkdir -p novel_families/data/{mmseqs_clusters_dark,track_b}
mkdir -p novel_families/results/track_b/{pdb,foldseek}

echo "=============================================="
echo "  Track B: Dark Proteome Coupling Pipeline"
echo "=============================================="
echo ""

# ── B0: Validate input ──
echo "  Running B0: Validate dark proteome input..."
python3 -u "${SCRIPTS}/B0_validate_dark_input.py"
B0_EXIT=$?

if [[ ${B0_EXIT} -ne 0 ]]; then
    echo "  B0 FAILED — dark proteome input not ready."
    echo "  Track B is staged — waiting for dark proteome agent."
    exit 1
fi

# ── B1: MMseqs2 clustering ──
echo "  Submitting B1: MMseqs2 dark clustering..."
JOB_B1=$(sbatch --parsable "${SCRIPTS}/B1_mmseqs_dark.sbatch")
echo "    Job B1: ${JOB_B1}"

# ── B2: Filter families ──
echo "  Submitting B2: Filter dark families (depends on B1)..."
JOB_B2=$(sbatch --parsable --dependency=afterok:${JOB_B1} \
    --wrap="cd ${BASE} && \
    export HOME=/scratch/drn2/newhome && \
    export TMPDIR=/scratch/drn2/tmp && \
    source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh && \
    conda activate /scratch/drn2/software/conda-mamba_1 && \
    python3 -u MANUSCRIPT/scripts/novel_families/B2_filter_dark_families.py" \
    --job-name=nf_B2 --time=00:30:00 --mem=32G --partition=compute \
    --output=novel_families/logs/B2_%j.out --error=novel_families/logs/B2_%j.err)
echo "    Job B2: ${JOB_B2}"

# ── B3: Abundance matrix ──
echo "  Submitting B3: Dark abundance matrix (depends on B2)..."
JOB_B3=$(sbatch --parsable --dependency=afterok:${JOB_B2} \
    --wrap="cd ${BASE} && \
    export HOME=/scratch/drn2/newhome && \
    export TMPDIR=/scratch/drn2/tmp && \
    source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh && \
    conda activate /scratch/drn2/software/conda-mamba_1 && \
    python3 -u MANUSCRIPT/scripts/novel_families/B3_dark_abundance.py" \
    --job-name=nf_B3 --time=00:30:00 --mem=32G --partition=compute \
    --output=novel_families/logs/B3_%j.out --error=novel_families/logs/B3_%j.err)
echo "    Job B3: ${JOB_B3}"

# ── B4: Environment coupling (array) ──
echo "  Submitting B4: Dark env coupling (depends on B3, array)..."
JOB_B4=$(sbatch --parsable --dependency=afterok:${JOB_B3} "${SCRIPTS}/B4_coupling.sbatch")
echo "    Job B4: ${JOB_B4}"

# ── B5: Statistical comparison ──
echo "  Submitting B5: Dark vs known comparison (depends on B4)..."
JOB_B5=$(sbatch --parsable --dependency=afterok:${JOB_B4} \
    --wrap="cd ${BASE} && \
    export HOME=/scratch/drn2/newhome && \
    export TMPDIR=/scratch/drn2/tmp && \
    source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh && \
    conda activate /scratch/drn2/software/conda-mamba_1 && \
    python3 -u MANUSCRIPT/scripts/novel_families/B5_dark_vs_known.py" \
    --job-name=nf_B5 --time=01:00:00 --mem=32G --partition=compute \
    --output=novel_families/logs/B5_%j.out --error=novel_families/logs/B5_%j.err)
echo "    Job B5: ${JOB_B5}"

# ── B6: ESMFold (GPU, parallel with B5) ──
echo "  Submitting B6: ESMFold prediction (depends on B4)..."
JOB_B6=$(sbatch --parsable --dependency=afterok:${JOB_B4} "${SCRIPTS}/B6_esmfold_predict.sbatch")
echo "    Job B6: ${JOB_B6}"

# ── B7: Foldseek (depends on B6) ──
echo "  Submitting B7: Foldseek search (depends on B6)..."
JOB_B7=$(sbatch --parsable --dependency=afterok:${JOB_B6} "${SCRIPTS}/B7_foldseek_search.sbatch")
echo "    Job B7: ${JOB_B7}"

echo ""
echo "=============================================="
echo "  Track B Pipeline Submitted"
echo "=============================================="
echo ""
echo "  Job Chain:"
echo "    B1 (${JOB_B1}) → B2 (${JOB_B2}) → B3 (${JOB_B3}) → B4 (${JOB_B4})"
echo "    → B5 (${JOB_B5})"
echo "    → B6 (${JOB_B6}) → B7 (${JOB_B7})"
echo ""
echo "  Expected wall time: ~60 hours"
echo ""
