#!/bin/bash
#SBATCH --job-name=nf_A1_map
#SBATCH --output=novel_families/logs/A1_contig_map_%j.out
#SBATCH --error=novel_families/logs/A1_contig_map_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=compute

# =============================================================================
# A1: Build protein → contig coordinate map
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
mkdir -p novel_families/logs novel_families/data/contig_maps novel_families/provenance

python3 -u MANUSCRIPT/scripts/novel_families/A1_build_contig_map.py \
    --format auto \
    --cpus 4
