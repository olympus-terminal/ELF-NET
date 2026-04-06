#!/bin/bash
#SBATCH --job-name=nf_A6_filter
#SBATCH --output=novel_families/logs/A6_filter_%j.out
#SBATCH --error=novel_families/logs/A6_filter_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --partition=compute

# =============================================================================
# A6: Filter recurrent novel families
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

python3 -u MANUSCRIPT/scripts/novel_families/A6_filter_novel_families.py \
    --min-members 50 \
    --min-samples 10
