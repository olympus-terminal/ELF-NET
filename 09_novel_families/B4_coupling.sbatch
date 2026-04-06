#!/bin/bash
#SBATCH --job-name=nf_B4_couple
#SBATCH --output=novel_families/logs/B4_coupling_%A_%a.out
#SBATCH --error=novel_families/logs/B4_coupling_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --partition=compute
#SBATCH --array=0-9

# =============================================================================
# B4: Dark proteome environment coupling (array job)
# Each task processes ~10 families for parallelism
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

# Each array task processes a chunk of 10 families
CHUNK_SIZE=10
CHUNK_START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))

python3 -u MANUSCRIPT/scripts/novel_families/B4_dark_env_coupling.py \
    --chunk-start ${CHUNK_START} \
    --chunk-size ${CHUNK_SIZE}
