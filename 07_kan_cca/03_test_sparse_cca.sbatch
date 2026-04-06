#!/bin/bash
#SBATCH --job-name=kan_cca_test_sparse
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/drn2/PROJECTS/TARA-LA4SR/logs/kan_cca/03_test_sparse_cca_%j.out
#SBATCH --error=/scratch/drn2/PROJECTS/TARA-LA4SR/logs/kan_cca/03_test_sparse_cca_%j.err

# --- Mandatory environment block ---
export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
export PYTHONUSERBASE=/scratch/drn2/newhome/.local
export OPENBLAS_NUM_THREADS=4
mkdir -p $TMPDIR
unset NETWORKX_BACKEND_CONFIG
unset NX_BACKEND_CONFIG
module load gcc/13.2.0
source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1
# --- End mandatory block ---

SCRIPT_DIR=/scratch/drn2/PROJECTS/TARA-LA4SR/MANUSCRIPT/scripts/kan_cca

echo "=== 03_test_sparse_cca.sbatch ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

python "$SCRIPT_DIR/03_test_sparse_cca.py"

echo ""
echo "Exit code: $?"
echo "Done: $(date)"
