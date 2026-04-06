#!/bin/bash
#SBATCH --job-name=algaGPT
#SBATCH --array=1-12
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/drn2/PROJECTS/TARA-LA4SR/slurm-logs/algagpt_%A_%a.out
#SBATCH --error=/scratch/drn2/PROJECTS/TARA-LA4SR/slurm-logs/algagpt_%A_%a.err

#
# algaGPT Inference - Classify proteins as algae/contaminant
#
# Uses the same input files as al-Khidr for cross-validation
# Model: nanoGPT-based algaGPT from Kourosh
#
# Array: 12 jobs, each processes ~200 files
#

echo "========================================================================"
echo "algaGPT Inference - Array task ${SLURM_ARRAY_TASK_ID}/12"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Started: $(date)"
echo "========================================================================"

# Paths
BASE_DIR="/scratch/drn2/PROJECTS/TARA-LA4SR"
MODEL_BASE="/scratch/drn2/PROJECTS/AI/KouroshModels/algaGPT2-S_set4_running_interactive_David-clone2-ht"
MODEL_DIR="${MODEL_BASE}/out"
INFER_SCRIPT="${BASE_DIR}/infer_algagpt_kvcache_20260107_040000.py"
INPUT_DIR="${BASE_DIR}/02_processed_data/proteins"
OUTPUT_DIR="${BASE_DIR}/03_analyses/algagpt_results"
PYTHON=/scratch/drn2/software/conda-mamba_1/bin/python3

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "${BASE_DIR}/slurm-logs"

# Generate file list on-the-fly
FILELIST=$(mktemp)
find "${INPUT_DIR}" -name "*.fa" -type f | sort > "${FILELIST}"

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
echo "  Total files: ${TOTAL_FILES}"
echo "  This job: files ${START_IDX} to ${END_IDX} (${N_FILES} files)"
echo "  Model: ${MODEL_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# Skip if no files for this task
if [[ $START_IDX -gt $TOTAL_FILES ]]; then
    echo "No files for task ${SLURM_ARRAY_TASK_ID}, exiting."
    rm "${FILELIST}"
    exit 0
fi

# Collect files for this job
FILES_TO_PROCESS=$(sed -n "${START_IDX},${END_IDX}p" "$FILELIST" | tr '\n' ' ')
rm "${FILELIST}"

N_ACTUAL=$(echo $FILES_TO_PROCESS | wc -w)
echo "Processing ${N_ACTUAL} files..."
echo ""

# Run inference
$PYTHON -u "$INFER_SCRIPT" "$MODEL_DIR" "$OUTPUT_DIR" $FILES_TO_PROCESS

EXITCODE=$?

echo ""
echo "========================================================================"
if [[ $EXITCODE -eq 0 ]]; then
    N_RESULTS=$(ls -1 "${OUTPUT_DIR}"/*_algagpt.tsv 2>/dev/null | wc -l)
    echo "SUCCESS: ${N_RESULTS} result files in ${OUTPUT_DIR}"
else
    echo "ERROR: Python script exited with code $EXITCODE"
fi
echo "Completed: $(date)"
echo "========================================================================"

exit $EXITCODE
