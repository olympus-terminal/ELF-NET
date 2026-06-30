#!/bin/bash
#SBATCH --job-name=dipep_lcr
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=90G
#SBATCH --time=168:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# --- Mandatory Jubail environment block ---
export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
export PYTHONUSERBASE=/scratch/drn2/newhome/.local
mkdir -p $TMPDIR
unset NETWORKX_BACKEND_CONFIG
unset NX_BACKEND_CONFIG

module load gcc/13.2.0
source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

cd /scratch/drn2/PROJECTS/TARA-LA4SR/MANUSCRIPT/scripts/dark_proteome

echo "Host: $(hostname)  Job: $SLURM_JOB_ID  Start: $(date)"
python 04_compute_dipeptide_oe_lcrmasked_20260530.py
echo "End: $(date)"
