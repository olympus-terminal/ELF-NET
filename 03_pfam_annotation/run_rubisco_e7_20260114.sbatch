#!/bin/bash
#SBATCH --job-name=rub_e7
#SBATCH --output=/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/rubisco_hmms/slurm-logs/rubisco_e7_%A_%a.out
#SBATCH --error=/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/rubisco_hmms/slurm-logs/rubisco_e7_%A_%a.err
#SBATCH --array=1-50
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# Rubisco HMM search with E-value 1e-7
# Processes multiple samples per task to stay under job limits
# Total: 2370 samples / 50 tasks = ~48 samples per task

module load hmmer/3.3

BASEDIR="/scratch/drn2/PROJECTS/TARA-LA4SR"
SCRIPT="${BASEDIR}/03_analyses/rubisco_hmms/rubisco_per_sample_20260114_e7.py"
HMM_DIR="${BASEDIR}/03_analyses/rubisco_hmms/hmms"
OUTPUT_DIR="${BASEDIR}/03_analyses/rubisco_hmms/per_sample_results_e7"
FILELIST="${BASEDIR}/03_analyses/rubisco_hmms/all_samples_filelist.txt"

mkdir -p "${OUTPUT_DIR}"

TOTAL_SAMPLES=2370
TASKS=50
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Calculate range for this task
SAMPLES_PER_TASK=$(( (TOTAL_SAMPLES + TASKS - 1) / TASKS ))
START=$(( (TASK_ID - 1) * SAMPLES_PER_TASK + 1 ))
END=$(( TASK_ID * SAMPLES_PER_TASK ))

# Don't exceed total
if [ $END -gt $TOTAL_SAMPLES ]; then
    END=$TOTAL_SAMPLES
fi

echo "Task ${TASK_ID}: Processing samples ${START}-${END}"
echo "Date: $(date)"

PROCESSED=0
FAILED=0

for i in $(seq $START $END); do
    LINE=$(sed -n "${i}p" "${FILELIST}")
    SAMPLE_ID=$(echo "$LINE" | awk '{print $1}')
    CLASSIFICATION_FILE=$(echo "$LINE" | awk '{print $2}')
    PROTEIN_FILE=$(echo "$LINE" | awk '{print $3}')

    # Skip if output already exists
    if [[ -f "${OUTPUT_DIR}/${SAMPLE_ID}_rubisco.tsv" ]]; then
        echo "SKIP (exists): ${SAMPLE_ID}"
        continue
    fi

    if [[ ! -f "${CLASSIFICATION_FILE}" ]] || [[ ! -f "${PROTEIN_FILE}" ]]; then
        echo "ERROR: Missing files for ${SAMPLE_ID}"
        FAILED=$((FAILED + 1))
        continue
    fi

    python3 "${SCRIPT}" "${SAMPLE_ID}" "${CLASSIFICATION_FILE}" "${PROTEIN_FILE}" "${HMM_DIR}" "${OUTPUT_DIR}"
    PROCESSED=$((PROCESSED + 1))
done

echo ""
echo "Task ${TASK_ID} complete: ${PROCESSED} processed, ${FAILED} failed"
echo "Date: $(date)"
