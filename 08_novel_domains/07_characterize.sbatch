#!/bin/bash
#SBATCH --job-name=novel_char
#SBATCH --output=logs/novel_domains/07_characterize_%j.out
#SBATCH --error=logs/novel_domains/07_characterize_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --partition=compute

# =============================================================================
# Step 7: Characterize Novel Domain Families (SLURM wrapper)
# =============================================================================

set -euo pipefail

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
mkdir -p $TMPDIR
export PYTHONUSERBASE=/scratch/drn2/newhome/.local

module purge
module load gcc/13.2.0

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

echo "========================================"
echo "Step 7: Characterize Novel Domain Families"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date:   $(date)"
echo ""

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"

# 06b count matrix already built (136 MB, 2044 x 33950) — skip re-running
echo "--- Skipping 06b: count matrix already exists ---"
ls -lh "${BASE}/03_analyses/novel_domains/results/novel_domain_count_matrix.tsv"
echo ""

# Characterize
echo "--- Running 07: Characterize ---"
python3 "${BASE}/MANUSCRIPT/scripts/novel_domains/07_characterize.py"

echo ""
echo "Done: $(date)"
