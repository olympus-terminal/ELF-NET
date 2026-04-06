#!/bin/bash
#SBATCH --job-name=manifold_sweep
#SBATCH --output=logs/manifold_sweep_%A_%a.out
#SBATCH --error=logs/manifold_sweep_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --array=1-12

# =============================================================================
# Hyperparameter Sweep: Pfam -> AlphaEarth Projection Model
# =============================================================================
#
# Configurations (12 total):
#   Task 1:  model=light, loss=mse,      lr=1e-3
#   Task 2:  model=light, loss=mse,      lr=1e-4
#   Task 3:  model=light, loss=cosine,   lr=1e-3
#   Task 4:  model=light, loss=cosine,   lr=1e-4
#   Task 5:  model=light, loss=combined, lr=1e-3
#   Task 6:  model=light, loss=combined, lr=1e-4
#   Task 7:  model=full,  loss=mse,      lr=1e-3
#   Task 8:  model=full,  loss=mse,      lr=1e-4
#   Task 9:  model=full,  loss=cosine,   lr=1e-3
#   Task 10: model=full,  loss=cosine,   lr=1e-4
#   Task 11: model=full,  loss=combined, lr=1e-3
#   Task 12: model=full,  loss=combined, lr=1e-4
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR/MANUSCRIPT
#   mkdir -p logs checkpoints
#   sbatch scripts/sweep_manifold_jubail.sbatch
#
# =============================================================================

# =============================================================================
# CRITICAL ENVIRONMENT SETUP (per JUBAIL_BEST_PRACTICES.md)
# =============================================================================

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
export PYTHONUSERBASE=/scratch/drn2/newhome/.local
mkdir -p $TMPDIR

unset NETWORKX_BACKEND_CONFIG
unset NX_BACKEND_CONFIG

module purge
module load gcc/13.2.0 cuda/12.2.0

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

# =============================================================================
# Configuration Selection
# =============================================================================

TASK_ID=$SLURM_ARRAY_TASK_ID

# Define configurations
MODELS=("light" "light" "light" "light" "light" "light" "full" "full" "full" "full" "full" "full")
LOSSES=("mse" "mse" "cosine" "cosine" "combined" "combined" "mse" "mse" "cosine" "cosine" "combined" "combined")
LRS=("0.001" "0.0001" "0.001" "0.0001" "0.001" "0.0001" "0.001" "0.0001" "0.001" "0.0001" "0.001" "0.0001")

# Array index (0-based)
IDX=$((TASK_ID - 1))

MODEL=${MODELS[$IDX]}
LOSS=${LOSSES[$IDX]}
LR=${LRS[$IDX]}

# =============================================================================
# Job Info
# =============================================================================

echo "========================================"
echo "Manifold Sweep - Task $TASK_ID / 12"
echo "========================================"
echo ""
echo "Job ID:      $SLURM_ARRAY_JOB_ID"
echo "Task ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Date:        $(date)"
echo ""
echo "Configuration:"
echo "  Model:     $MODEL"
echo "  Loss:      $LOSS"
echo "  LR:        $LR"
echo ""

# =============================================================================
# Directory Setup
# =============================================================================

BASE_DIR="/scratch/drn2/PROJECTS/TARA-LA4SR"
MANUSCRIPT_DIR="${BASE_DIR}/MANUSCRIPT"
SCRIPTS_DIR="${MANUSCRIPT_DIR}/scripts"
DATA_DIR="${MANUSCRIPT_DIR}/data"

cd "${MANUSCRIPT_DIR}" || exit 1
mkdir -p logs checkpoints

TRAIN_SCRIPT="${SCRIPTS_DIR}/train_manifold_20260125_173853.py"
DATA_FILE=$(ls -t ${DATA_DIR}/manifold_train_val_test_*.npz 2>/dev/null | head -1)

if [[ -z "${DATA_FILE}" || ! -f "${DATA_FILE}" ]]; then
    echo "ERROR: No data file found in ${DATA_DIR}"
    exit 1
fi

echo "Data file: $(basename ${DATA_FILE})"
echo ""

# =============================================================================
# Run Training
# =============================================================================

echo "Starting training..."
echo ""

START_TIME=$(date +%s)

python -u "${TRAIN_SCRIPT}" \
    --model "${MODEL}" \
    --loss "${LOSS}" \
    --lr "${LR}" \
    --epochs 200 \
    --batch-size 32 \
    --seed 42 \
    --device cuda \
    --data-file "${DATA_FILE}"

EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "Task $TASK_ID Complete"
echo "========================================"
echo ""
echo "Exit code:    ${EXIT_CODE}"
echo "Elapsed time: ${ELAPSED} seconds ($((ELAPSED / 60)) min)"
echo "Config:       model=${MODEL}, loss=${LOSS}, lr=${LR}"
echo "Completed:    $(date)"

exit $EXIT_CODE
