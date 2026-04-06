#!/bin/bash
#SBATCH --job-name=interpro_api
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains/results/interpro_api_%j.out
#SBATCH --error=/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains/results/interpro_api_%j.err

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
mkdir -p $TMPDIR
source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains

python 11_interpro_api_scan.py --resume
