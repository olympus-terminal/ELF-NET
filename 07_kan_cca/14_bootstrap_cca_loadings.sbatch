#!/bin/bash
#SBATCH --job-name=cca_bootstrap
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/drn2/PROJECTS/TARA-LA4SR/logs/kan_cca/14_bootstrap_%j.out
#SBATCH --error=/scratch/drn2/PROJECTS/TARA-LA4SR/logs/kan_cca/14_bootstrap_%j.err

# === Mandatory HPC environment ===
export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
export PYTHONUSERBASE=/scratch/drn2/newhome/.local
mkdir -p $TMPDIR
unset NETWORKX_BACKEND_CONFIG
unset NX_BACKEND_CONFIG

# === Load modules and conda ===
module load gcc/13.2.0
source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

# === Create log directory ===
mkdir -p /scratch/drn2/PROJECTS/TARA-LA4SR/logs/kan_cca

# === Run ===
python /scratch/drn2/PROJECTS/TARA-LA4SR/MANUSCRIPT/scripts/kan_cca/14_bootstrap_cca_loadings.py

echo "Job completed at $(date)"
