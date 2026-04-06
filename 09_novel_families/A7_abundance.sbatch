#!/bin/bash
#SBATCH --job-name=nf_A7_abund
#SBATCH --output=novel_families/logs/A7_abundance_%j.out
#SBATCH --error=novel_families/logs/A7_abundance_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=compute

# =============================================================================
# A7: Build per-sample abundance matrix + CLR transform
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

python3 -u MANUSCRIPT/scripts/novel_families/A7_build_abundance_matrix.py
