#!/bin/bash
#SBATCH --job-name=nf_A5_mmseqs
#SBATCH --output=novel_families/logs/A5_mmseqs_%j.out
#SBATCH --error=novel_families/logs/A5_mmseqs_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=90G
#SBATCH --partition=compute

# =============================================================================
# A5: MMseqs2 clustering at 30% identity
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

bash MANUSCRIPT/scripts/novel_families/A5_mmseqs_cluster.sh
