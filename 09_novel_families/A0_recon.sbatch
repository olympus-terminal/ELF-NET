#!/bin/bash
#SBATCH --job-name=nf_A0_recon
#SBATCH --output=novel_families/logs/A0_recon_%j.out
#SBATCH --error=novel_families/logs/A0_recon_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=compute

# =============================================================================
# A0: Reconnaissance — SNAP output format discovery
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
mkdir -p novel_families/logs

bash MANUSCRIPT/scripts/novel_families/A0_recon_snap_format.sh
