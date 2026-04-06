#!/bin/bash
#SBATCH --job-name=rolling_submit
#SBATCH --partition=compute
#SBATCH --qos=small
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/scratch/drn2/PROJECTS/TARA-LA4SR/logs/novel_domains/rolling_submit_%j.out
#SBATCH --error=/scratch/drn2/PROJECTS/TARA-LA4SR/logs/novel_domains/rolling_submit_%j.err

# Rolling submitter for Step 6 hmmsearch tasks 45-1704
# Submits small batches (50 tasks each) as SLURM slots become available.
# Keeps total expanded queue <=1200 to leave room for DIAMOND + batch 3.

set -uo pipefail

BASE=/scratch/drn2/PROJECTS/TARA-LA4SR
SCRIPT_DIR=${BASE}/MANUSCRIPT/scripts/novel_domains
LOG_DIR=${BASE}/logs/novel_domains
STATE_DIR=${BASE}/03_analyses/novel_domains
MAX_SUBMIT=1500
TARGET_QUEUE=1200  # Keep queue below this to leave headroom
BATCH_SIZE=50      # Submit 50 tasks at a time

# Track submitted ranges in a state file
STATE_FILE=${STATE_DIR}/rolling_submit_state.txt
if [[ ! -f ${STATE_FILE} ]]; then
    echo 45 > ${STATE_FILE}  # Start from task 45
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*"; }

NEXT_TASK=$(cat ${STATE_FILE})
LAST_TASK=1704  # End of batch 2

log "Rolling submitter started. Next task: ${NEXT_TASK}, last task: ${LAST_TASK}"

while [[ ${NEXT_TASK} -le ${LAST_TASK} ]]; do
    CURRENT=$(squeue -u drn2 -r --noheader 2>/dev/null | wc -l)
    AVAIL=$((MAX_SUBMIT - CURRENT))
    IN_USE=${CURRENT}
    
    log "Queue status: ${IN_USE} expanded jobs, ${AVAIL} slots available, next_task=${NEXT_TASK}"
    
    if [[ ${IN_USE} -lt ${TARGET_QUEUE} ]] && [[ ${AVAIL} -ge $((BATCH_SIZE + 5)) ]]; then
        # Calculate end of this batch
        END_TASK=$((NEXT_TASK + BATCH_SIZE - 1))
        if [[ ${END_TASK} -gt ${LAST_TASK} ]]; then
            END_TASK=${LAST_TASK}
        fi
        
        log "Submitting tasks ${NEXT_TASK}-${END_TASK} ($((END_TASK - NEXT_TASK + 1)) tasks)..."
        
        JOB=$(sbatch --parsable \
            --array=${NEXT_TASK}-${END_TASK}%50 \
            --time=24:00:00 \
            --output="${LOG_DIR}/06_hmmsearch_novel_%A_%a.out" \
            --error="${LOG_DIR}/06_hmmsearch_novel_%A_%a.err" \
            "${SCRIPT_DIR}/06_hmmsearch_novel.sbatch" 2>&1)
        
        if [[ $? -eq 0 ]] && [[ "${JOB}" =~ ^[0-9]+$ ]]; then
            log "Submitted job ${JOB} (tasks ${NEXT_TASK}-${END_TASK})"
            NEXT_TASK=$((END_TASK + 1))
            echo ${NEXT_TASK} > ${STATE_FILE}
        else
            log "Submission failed: ${JOB}. Will retry in 5 min."
        fi
    else
        log "Queue too full (${IN_USE} >= ${TARGET_QUEUE} or ${AVAIL} < $((BATCH_SIZE + 5))). Waiting..."
    fi
    
    # Wait before next check
    sleep 300
done

log "All tasks submitted (45-${LAST_TASK}). Rolling submitter complete."

# Now create Step 7 coordinator that depends on ALL hmmsearch jobs
log "Setting up Step 7 coordinator..."

# Get all hmmsearch job IDs
ALL_JOBS=$(sacct -u drn2 --format=JobID,JobName --parsable2 --noheader 2>/dev/null | grep hmmsearch_novel | grep -v '\.' | awk -F'|' '{print $1}' | grep -v '_' | sort -u | tr '\n' ',' | sed 's/,$//')

if [[ -n "${ALL_JOBS}" ]]; then
    DEPS=$(echo ${ALL_JOBS} | sed 's/,/,afterany:/g' | sed 's/^/afterany:/')
    COORD=$(sbatch --parsable \
        --dependency="${DEPS}" \
        --job-name=coord_step7 \
        --partition=compute --qos=small \
        --time=01:00:00 --ntasks=1 --cpus-per-task=1 --mem=4G \
        --output="${LOG_DIR}/coord_step7_%j.out" \
        --error="${LOG_DIR}/coord_step7_%j.err" \
        --wrap="cd ${BASE} && bash ${SCRIPT_DIR}/submit_next_step.sh 7" 2>&1)
    log "Coordinator -> Step 7: Job ${COORD} (depends on ${ALL_JOBS})"
fi

log "Rolling submitter exiting."
