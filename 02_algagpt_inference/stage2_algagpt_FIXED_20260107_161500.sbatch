#!/bin/bash
#SBATCH --job-name=algagpt_fix
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --array=1-238%8
#SBATCH --output=/scratch/drn2/PROJECTS/TARA-LA4SR/slurm-logs/algagpt_fix_%A_%a.out
#SBATCH --error=/scratch/drn2/PROJECTS/TARA-LA4SR/slurm-logs/algagpt_fix_%A_%a.err

# algaGPT FIXED Production Run - Position IDs computed from attention_mask
# 2372 files / 10 per task = 238 tasks

BASE_DIR="/scratch/drn2/PROJECTS/TARA-LA4SR"
MODEL_DIR="/scratch/drn2/PROJECTS/AI/KouroshModels/algaGPT2-S_set4_running_interactive_David-clone2-ht/out"
OUTPUT_DIR="${BASE_DIR}/03_analyses/algagpt_results_fixed"
FILELIST="${BASE_DIR}/algagpt_filelist.txt"
PYTHON=/scratch/drn2/software/conda-mamba_1/bin/python3

FILES_PER_TASK=10

mkdir -p "$OUTPUT_DIR"

# Calculate which files this task handles
START_LINE=$(( (SLURM_ARRAY_TASK_ID - 1) * FILES_PER_TASK + 1 ))
END_LINE=$(( SLURM_ARRAY_TASK_ID * FILES_PER_TASK ))

# Get files for this task
FILES=$(sed -n "${START_LINE},${END_LINE}p" "$FILELIST")

if [ -z "$FILES" ]; then
    echo "No files for task $SLURM_ARRAY_TASK_ID"
    exit 0
fi

echo "Task $SLURM_ARRAY_TASK_ID: Processing files $START_LINE-$END_LINE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Run inference
$PYTHON -u ${BASE_DIR}/infer_algagpt_masked_fixed_20260107_150000.py \
    "$MODEL_DIR" \
    "$OUTPUT_DIR" \
    $FILES
