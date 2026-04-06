#!/bin/bash
#SBATCH --job-name=nf_A2_annot
#SBATCH --output=novel_families/logs/A2_annotate_%j.out
#SBATCH --error=novel_families/logs/A2_annotate_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --partition=compute

# =============================================================================
# A2: Cross-reference proteins with Pfam annotations
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

python3 -u MANUSCRIPT/scripts/novel_families/A2_annotate_proteins.py \
    --evalue 1e-9 \
    --cpus 16
