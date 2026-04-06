#!/bin/bash
#SBATCH --job-name=nutrient_pipeline
#SBATCH --output=/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/woa23_nutrients/logs/pipeline_%j.out
#SBATCH --error=/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/woa23_nutrients/logs/pipeline_%j.err
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# ==============================================================================
# 03_run_pipeline.sbatch — Run full nutrient download+extract+merge pipeline
#
# Runs:
#   1. Download WOA23 + MLD data (if not already present)
#   2. Extract nutrient values at sample coordinates
#   3. Merge into existing GEE+PFAM dataset
#
# Usage:
#   sbatch scripts/nutrients/03_run_pipeline.sbatch
# ==============================================================================

set -euo pipefail

module purge
module load python/3.10.13
module load gcc/12.3.0

PROJ_DIR="/scratch/drn2/PROJECTS/TARA-LA4SR"
SCRIPT_DIR="/scratch/drn2/PROJECTS/TARA-LA4SR/MANUSCRIPT/scripts/nutrients"
LOG_DIR="${PROJ_DIR}/03_analyses/woa23_nutrients/logs"
mkdir -p "${LOG_DIR}"

echo "=== Nutrient Pipeline ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"

# Step 1: Download
echo ""
echo "--- Step 1: Download WOA23 + MLD ---"
bash "${SCRIPT_DIR}/00_download_woa23.sh" --hpc

# Step 2: Extract
echo ""
echo "--- Step 2: Extract nutrients at coordinates ---"
python3 "${SCRIPT_DIR}/01_extract_nutrients.py"

# Step 3: Merge
echo ""
echo "--- Step 3: Merge with existing dataset ---"
python3 "${SCRIPT_DIR}/02_merge_nutrients.py"

echo ""
echo "=== Pipeline complete ==="
echo "Date: $(date)"
