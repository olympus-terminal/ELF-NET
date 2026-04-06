#!/bin/bash
#SBATCH --job-name=nutrient_reanalysis
#SBATCH --output=/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/woa23_nutrients/logs/reanalysis_%j.out
#SBATCH --error=/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/woa23_nutrients/logs/reanalysis_%j.err
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8

# ==============================================================================
# 04_rerun_analyses.sbatch — Re-run analyses with expanded environmental variables
#
# After merging nutrients, re-run:
#   1. KAN-CCA preprocessing with 37 env variables
#   2. Novel domain environmental correlations
#   3. Bidirectional XGBoost models
#
# Depends on: 03_run_pipeline.sbatch (nutrients must be merged first)
#
# Usage:
#   sbatch --dependency=afterok:PIPELINE_JOBID scripts/nutrients/04_rerun_analyses.sbatch
# ==============================================================================

set -euo pipefail

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate base

PROJ_DIR="/scratch/drn2/PROJECTS/TARA-LA4SR"
SCRIPT_DIR="/scratch/drn2/PROJECTS/TARA-LA4SR/MANUSCRIPT/scripts"
LOG_DIR="${PROJ_DIR}/03_analyses/woa23_nutrients/logs"
mkdir -p "${LOG_DIR}"

echo "=== Re-analysis with Nutrients ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"

# --- 1. KAN-CCA Preprocessing ---
echo ""
echo "--- Step 1: KAN-CCA preprocessing (37 env vars) ---"
python3 "${PROJ_DIR}/scripts/kan_cca/01_preprocess.py" 2>&1 | tee "${LOG_DIR}/kan_cca_preprocess.log"

# --- 2. Novel domain correlations ---
echo ""
echo "--- Step 2: Novel domain environmental correlations ---"
python3 "${SCRIPT_DIR}/novel_domains/08_env_correlation.py" 2>&1 | tee "${LOG_DIR}/novel_domain_corr.log"

# --- 3. Bidirectional XGBoost (env → pfam, pfam → env) ---
echo ""
echo "--- Step 3: Bidirectional XGBoost models ---"
python3 "${SCRIPT_DIR}/prepare_env_to_pfam_data_20260125_190213.py" 2>&1 | tee "${LOG_DIR}/env_to_pfam_prep.log"

echo ""
echo "=== Re-analysis complete ==="
echo "Date: $(date)"
