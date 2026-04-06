#!/bin/bash
#SBATCH --job-name=sample_15m
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j_sample_15m.out
#SBATCH --error=logs/slurm_%j_sample_15m.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=drn2@nyu.edu

# --- Mandatory HPC environment ---
export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
export PYTHONUSERBASE=/scratch/drn2/newhome/.local
mkdir -p $TMPDIR
unset NETWORKX_BACKEND_CONFIG
unset NX_BACKEND_CONFIG

module load gcc/13.2.0
source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

# --- Run ---
cd /scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains

echo "============================================"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "Working dir : $(pwd)"
echo "============================================"
echo ""

python sample_15m.py --target-seqs 15000000 --seed 42 --holdout-frac 0.10

echo ""
echo "============================================"
echo "Job finished: $(date)"
echo "============================================"
