#!/bin/bash
#SBATCH --job-name=novel_env
#SBATCH --output=logs/novel_domains/08_env_correlation_%j.out
#SBATCH --error=logs/novel_domains/08_env_correlation_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=compute

# =============================================================================
# Step 8: Environmental Correlation Analysis (SLURM wrapper)
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_domains/08_env_correlation.sbatch
#   Generated: 2026-02-21
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Purpose:
#   SLURM wrapper for 08_env_correlation.py. Runs Spearman correlation + FDR
#   framework on novel domains vs environmental variables, and optionally
#   gradient boosting predictive power comparison (Pfam vs Pfam+novel).
#
# Input:
#   - 03_analyses/novel_domains/results/novel_domain_count_matrix.tsv
#   - 03_analyses/hmmsearch_results/pfam_count_matrix_*.tsv
#   - Environmental data (AlphaEarth, GEE)
#
# Output:
#   - 03_analyses/novel_domains/results/novel_domain_env_correlations.tsv
#   - 03_analyses/novel_domains/results/novel_domain_gbr_performance.tsv
#   - 03_analyses/novel_domains/results/novel_vs_pfam_effect_sizes.tsv
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   sbatch MANUSCRIPT/scripts/novel_domains/08_env_correlation.sbatch
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

echo "========================================"
echo "Step 8: Environmental Correlation Analysis"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "Date:   $(date)"
echo ""

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"

python3 "${BASE}/MANUSCRIPT/scripts/novel_domains/08_env_correlation.py"

echo ""
echo "Done: $(date)"
