#!/bin/bash
#SBATCH --job-name=algagpt_xgb_gee
#SBATCH --output=slurm-logs/algagpt_xgb_gee_%j.out
#SBATCH --error=slurm-logs/algagpt_xgb_gee_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --partition=standard

# algaGPT XGBoost: Predict GEE from PFAM
# Author: algaGPT Environmental Analysis Pipeline
# Date: 2026-01-19

# Environment detection
if [[ $(pwd) == /scratch/* ]]; then
    BASE_DIR="/scratch/drn2/PROJECTS/algaGPT-TARA-archive"
elif [[ $(pwd) == /media/* ]]; then
    BASE_DIR="/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses"
else
    echo "ERROR: Unknown environment"
    exit 1
fi

cd "${BASE_DIR}"

# Load modules
module purge
module load anaconda3/2021.11
module load gcc/11.2.0

# Activate conda environment with XGBoost
source activate /scratch/drn2/conda_envs/ml_env

# Run analysis
echo "Starting XGBoost GEE prediction analysis..."
echo "Working directory: $(pwd)"
echo "Input: algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
echo "---"

python3 algagpt_xgboost_gee_20260119_104500.py

echo "---"
echo "Analysis complete!"
