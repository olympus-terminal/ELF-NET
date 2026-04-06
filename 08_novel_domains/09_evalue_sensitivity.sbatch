#!/bin/bash
#SBATCH --job-name=eval_sens
#SBATCH --output=logs/novel_domains/09_evalue_sensitivity_%j.out
#SBATCH --error=logs/novel_domains/09_evalue_sensitivity_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --partition=compute

# =============================================================================
# Step 9: E-value Sensitivity Analysis (SLURM wrapper)
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_domains/09_evalue_sensitivity.sbatch
#   Generated: 2026-03-20
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Purpose:
#   Post-filter novel domain hmmsearch results at E < 1e-7 and E < 1e-9
#   (in addition to the original E < 1e-5) and recompute Spearman
#   correlations with environmental variables. Addresses reviewer concern
#   about asymmetric E-value thresholds between Pfam and novel HMM searches.
#
# Input:
#   - 03_analyses/novel_domains/novel_hits/{sample}.novel.tbl
#   - 03_analyses/novel_domains/sample_list.txt
#   - 03_analyses/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv
#
# Output:
#   - 03_analyses/novel_domains/results/sensitivity/sensitivity_comparison.tsv
#   - 03_analyses/novel_domains/results/sensitivity/per_threshold_details/
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   sbatch MANUSCRIPT/scripts/novel_domains/09_evalue_sensitivity.sbatch
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

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

# Pin threads to allocated CPUs — prevents OpenBLAS from spawning 56 threads
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Unbuffered Python output so SLURM logs show progress in real time
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Step 9: E-value Sensitivity Analysis"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "Date:   $(date)"
echo ""

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"

# Ensure log and output directories exist
mkdir -p "${BASE}/03_analyses/novel_domains/results/sensitivity/per_threshold_details"

python3 "${BASE}/MANUSCRIPT/scripts/novel_domains/09_evalue_sensitivity.py"

echo ""
echo "Done: $(date)"
