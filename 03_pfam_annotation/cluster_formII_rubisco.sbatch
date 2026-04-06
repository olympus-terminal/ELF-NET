#!/bin/bash
#SBATCH --job-name=formII_cluster
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --mem=90G
#SBATCH --cpus-per-task=16
#SBATCH --output=/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/rubisco_hmms/formII_cluster_%j.out
#SBATCH --error=/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/rubisco_hmms/formII_cluster_%j.err

# Cluster Form II RuBisCO sequences at 97% identity
# 2026-03-21

module load hmmer/3.3
module load cd-hit/4.8.1
module load python/3.9.0

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "hmmsearch: $(which hmmsearch)"
echo "cd-hit: $(which cd-hit)"
echo ""

cd /scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/rubisco_hmms

# Clean up previous failed run
rm -rf formII_clustering/

python3 /scratch/drn2/PROJECTS/TARA-LA4SR/MANUSCRIPT/scripts/cluster_formII_rubisco.py

echo ""
echo "Job finished: $(date)"
