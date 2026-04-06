#!/bin/bash
#SBATCH --job-name=agpt_vs_diamond
#SBATCH --output=logs/agpt_vs_diamond_%j.out
#SBATCH --error=logs/agpt_vs_diamond_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=90GB
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute

# =============================================================================
# AlgaGPT vs DIAMOND BLAST Comparative Analysis
# =============================================================================
# Compares LLM-based classification against DIAMOND BLASTp hits on the
# original unfiltered protein set (02_processed_data/proteins/).
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   mkdir -p logs
#   sbatch MANUSCRIPT/scripts/algagpt_vs_diamond_comparison.sbatch
#
# Date: 2026-03-15
# =============================================================================

set -euo pipefail

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
export PYTHONUSERBASE=/scratch/drn2/newhome/.local
mkdir -p $TMPDIR

unset NETWORKX_BACKEND_CONFIG
unset NX_BACKEND_CONFIG

module load gcc/13.2.0
source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

echo "========================================"
echo "AlgaGPT vs DIAMOND BLAST Comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================"

cd /scratch/drn2/PROJECTS/TARA-LA4SR

python MANUSCRIPT/scripts/algagpt_vs_diamond_comparison.py

echo ""
echo "Done: $(date)"
