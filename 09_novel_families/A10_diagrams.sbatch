#!/bin/bash
#SBATCH --job-name=nf_A10_diag
#SBATCH --output=novel_families/logs/A10_diagrams_%j.out
#SBATCH --error=novel_families/logs/A10_diagrams_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=compute

# =============================================================================
# A10: Contig neighborhood diagrams
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

python3 -u MANUSCRIPT/scripts/novel_families/A10_neighborhood_diagrams.py --top-n 8
