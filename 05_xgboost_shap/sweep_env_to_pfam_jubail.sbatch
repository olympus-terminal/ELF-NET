#!/bin/bash
#SBATCH --job-name=env2pfam_sweep
#SBATCH --output=logs/env2pfam_sweep_%A_%a.out
#SBATCH --error=logs/env2pfam_sweep_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --array=1-8

# =============================================================================
# Hyperparameter Sweep: Environment → Pfam Prediction
# =============================================================================
#
# Comparing two manifold filtering strategies:
#   - Pythia-LA4SR filtering
#   - AlgaGPT SMART filtering
#
# Configurations (8 total = 2 datasets × 2 models × 2 learning rates):
#   Task 1: dataset=pythia,  model=light, lr=1e-3
#   Task 2: dataset=pythia,  model=light, lr=1e-4
#   Task 3: dataset=pythia,  model=full,  lr=1e-3
#   Task 4: dataset=pythia,  model=full,  lr=1e-4
#   Task 5: dataset=algagpt, model=light, lr=1e-3
#   Task 6: dataset=algagpt, model=light, lr=1e-4
#   Task 7: dataset=algagpt, model=full,  lr=1e-3
#   Task 8: dataset=algagpt, model=full,  lr=1e-4
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR/MANUSCRIPT
#   mkdir -p logs checkpoints data
#   sbatch scripts/sweep_env_to_pfam_jubail.sbatch
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
DATASETS=("pythia" "pythia" "pythia" "pythia" "algagpt" "algagpt" "algagpt" "algagpt")
MODELS=("light" "light" "full" "full" "light" "light" "full" "full")
LRS=("0.001" "0.0001" "0.001" "0.0001" "0.001" "0.0001" "0.001" "0.0001")

# Array index (0-based)
IDX=$((TASK_ID - 1))

DATASET=${DATASETS[$IDX]}
MODEL=${MODELS[$IDX]}
LR=${LRS[$IDX]}

# =============================================================================
# Job Info
# =============================================================================

echo "========================================"
echo "Env→Pfam Sweep - Task $TASK_ID / 8"
echo "========================================"
echo ""
echo "Job ID:      $SLURM_ARRAY_JOB_ID"
echo "Task ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Date:        $(date)"
echo ""
echo "Configuration:"
echo "  Dataset:   $DATASET"
echo "  Model:     $MODEL"
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
mkdir -p logs checkpoints data

# Find scripts
PREP_SCRIPT="${SCRIPTS_DIR}/prepare_env_to_pfam_data_20260125_190213.py"
TRAIN_SCRIPT="${SCRIPTS_DIR}/train_env_to_pfam_20260125_190213.py"

# Data preparation was done locally and synced - no need to run here

# Find data file
DATA_FILE=$(ls -t ${DATA_DIR}/env_to_pfam_${DATASET}_*.npz 2>/dev/null | head -1)

if [[ -z "${DATA_FILE}" || ! -f "${DATA_FILE}" ]]; then
    echo "ERROR: No data file found for ${DATASET} in ${DATA_DIR}"
    echo "Waiting 60s for data preparation to complete..."
    sleep 60
    DATA_FILE=$(ls -t ${DATA_DIR}/env_to_pfam_${DATASET}_*.npz 2>/dev/null | head -1)
    if [[ -z "${DATA_FILE}" || ! -f "${DATA_FILE}" ]]; then
        echo "ERROR: Still no data file. Exiting."
        exit 1
    fi
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
    --dataset "${DATASET}" \
    --model "${MODEL}" \
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
echo "Config:       dataset=${DATASET}, model=${MODEL}, lr=${LR}"
echo "Completed:    $(date)"

exit $EXIT_CODE
