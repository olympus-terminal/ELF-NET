#!/bin/bash
#SBATCH --job-name=nf_A8_couple
#SBATCH --output=novel_families/logs/A8_coupling_%j.out
#SBATCH --error=novel_families/logs/A8_coupling_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --partition=compute

# =============================================================================
# A8: Environment coupling (Spearman + XGBoost)
# =============================================================================

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
export PYTHONUSERBASE=/scratch/drn2/newhome/.local
mkdir -p $TMPDIR

unset NETWORKX_BACKEND_CONFIG
unset NX_BACKEND_CONFIG

module purge
module load gcc/13.2.0

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
cd "${BASE}"

python3 -u MANUSCRIPT/scripts/novel_families/A8_environment_coupling.py
