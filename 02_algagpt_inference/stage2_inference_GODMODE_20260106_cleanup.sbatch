#!/bin/bash
#SBATCH --job-name=ELF_cleanup
#SBATCH --array=1-12
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/drn2/PROJECTS/TARA-LA4SR/slurm-logs/cleanup_%A_%a.out
#SBATCH --error=/scratch/drn2/PROJECTS/TARA-LA4SR/slurm-logs/cleanup_%A_%a.err

#
# GODMODE Inference - Cleanup run for 888 missing files
# Processes only files that failed or weren't completed in previous run
#

echo "========================================================================"
echo "GODMODE Cleanup - Array task ${SLURM_ARRAY_TASK_ID}/12"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "========================================================================"

# Paths
BASE_DIR="/scratch/drn2/PROJECTS/TARA-LA4SR"
MODEL_DIR="${BASE_DIR}/tools/Pythia70m-b-checkpoint-55000"
FILELIST="${BASE_DIR}/missing_proteins_20260106.txt"
INFER_SCRIPT="${BASE_DIR}/infer_godmode_v2_20260103_015000.py"
OUTPUT_DIR="${BASE_DIR}/03_analyses/alkhidr_results"
PYTHON=/scratch/drn2/software/conda-mamba_1/bin/python3

BATCH_SIZE=512

mkdir -p "$OUTPUT_DIR"

# Calculate file range for this array task
TOTAL_FILES=$(wc -l < "$FILELIST")
FILES_PER_JOB=$(( (TOTAL_FILES + 11) / 12 ))
START_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) * FILES_PER_JOB + 1 ))
END_IDX=$(( SLURM_ARRAY_TASK_ID * FILES_PER_JOB ))

if [[ $END_IDX -gt $TOTAL_FILES ]]; then
    END_IDX=$TOTAL_FILES
fi

N_FILES=$(( END_IDX - START_IDX + 1 ))

echo ""
echo "Configuration:"
echo "  Missing files list: ${FILELIST}"
echo "  Total missing: ${TOTAL_FILES}"
echo "  This task: ${START_IDX} to ${END_IDX} (${N_FILES} files)"
echo "  Batch size: ${BATCH_SIZE}"
echo "  CPUs: ${SLURM_CPUS_PER_TASK}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Walltime: 48 hours"
echo ""

FILES_TO_PROCESS=$(sed -n "${START_IDX},${END_IDX}p" "$FILELIST" | tr '\n' ' ')
N_ACTUAL=$(echo $FILES_TO_PROCESS | wc -w)

if [[ $N_ACTUAL -eq 0 ]]; then
    echo "No files to process for this task - already complete!"
    exit 0
fi

echo "Processing ${N_ACTUAL} files..."
echo ""

$PYTHON -u "$INFER_SCRIPT" "$MODEL_DIR" "$BATCH_SIZE" "$OUTPUT_DIR" $FILES_TO_PROCESS

EXITCODE=$?

echo ""
echo "========================================================================"
if [[ $EXITCODE -eq 0 ]]; then
    N_DONE=$(ls -1 "${OUTPUT_DIR}"/*_alkhidr.tsv 2>/dev/null | wc -l)
    echo "SUCCESS: Total result files now: ${N_DONE}"
else
    echo "ERROR: Python script exited with code $EXITCODE"
fi
echo "Completed: $(date)"
echo "========================================================================"

exit $EXITCODE
