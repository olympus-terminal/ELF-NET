#!/bin/bash
#SBATCH --job-name=batch_monitor_v2
#SBATCH --partition=compute
#SBATCH --qos=small
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/scratch/drn2/PROJECTS/TARA-LA4SR/logs/novel_domains/batch_monitor_v2_%j.out
#SBATCH --error=/scratch/drn2/PROJECTS/TARA-LA4SR/logs/novel_domains/batch_monitor_v2_%j.err

# Auto-submit Step 6 batches 2 and 3 when SLURM slots become available.
# Uses expanded job count (-r) which is what SLURM actually enforces for MaxSubmitJobs.
# Strategy: submit batch 3 (340 elements) first since it needs fewer slots.

set -uo pipefail

BASE=/scratch/drn2/PROJECTS/TARA-LA4SR
SCRIPT_DIR=${BASE}/MANUSCRIPT/scripts/novel_domains
LOG_DIR=${BASE}/logs/novel_domains
MAX_SUBMIT=1500

BATCH3_SUBMITTED=false
BATCH2_SUBMITTED=false
BATCH3_JOB=""
BATCH2_JOB=""

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*"; }

for i in $(seq 1 576); do  # Check every 5 min for 48 hours
    CURRENT=$(squeue -u drn2 -r --noheader 2>/dev/null | wc -l)
    AVAIL=$((MAX_SUBMIT - CURRENT))

    log "Check ${i}: ${CURRENT} expanded jobs, ${AVAIL} slots available (batch3=${BATCH3_SUBMITTED}, batch2=${BATCH2_SUBMITTED})"

    # Submit batch 3 first (smaller: 340 elements, needs 345 free slots)
    if [[ "${BATCH3_SUBMITTED}" == "false" ]] && [[ ${AVAIL} -ge 345 ]]; then
        log "Submitting batch 3 (array 1705-2044, 340 tasks)..."
        BATCH3_JOB=$(sbatch --parsable \
            --array=1705-2044%50 \
            --time=24:00:00 \
            --output="${LOG_DIR}/06_hmmsearch_novel_%A_%a.out" \
            --error="${LOG_DIR}/06_hmmsearch_novel_%A_%a.err" \
            "${SCRIPT_DIR}/06_hmmsearch_novel.sbatch" 2>&1)
        if [[ "$?" -eq 0 ]] && [[ "${BATCH3_JOB}" =~ ^[0-9]+$ ]]; then
            log "Batch 3 submitted: Job ${BATCH3_JOB}"
            BATCH3_SUBMITTED=true
        else
            log "Batch 3 failed: ${BATCH3_JOB}"
        fi
    fi

    # Recalculate after possible batch 3 submission
    if [[ "${BATCH2_SUBMITTED}" == "false" ]]; then
        CURRENT=$(squeue -u drn2 -r --noheader 2>/dev/null | wc -l)
        AVAIL=$((MAX_SUBMIT - CURRENT))
    fi

    # Submit batch 2 (larger: 852 elements, needs 860 free slots)
    if [[ "${BATCH2_SUBMITTED}" == "false" ]] && [[ ${AVAIL} -ge 860 ]]; then
        log "Submitting batch 2 (array 853-1704, 852 tasks)..."
        BATCH2_JOB=$(sbatch --parsable \
            --array=853-1704%50 \
            --time=24:00:00 \
            --output="${LOG_DIR}/06_hmmsearch_novel_%A_%a.out" \
            --error="${LOG_DIR}/06_hmmsearch_novel_%A_%a.err" \
            "${SCRIPT_DIR}/06_hmmsearch_novel.sbatch" 2>&1)
        if [[ "$?" -eq 0 ]] && [[ "${BATCH2_JOB}" =~ ^[0-9]+$ ]]; then
            log "Batch 2 submitted: Job ${BATCH2_JOB}"
            BATCH2_SUBMITTED=true
        else
            log "Batch 2 failed: ${BATCH2_JOB}"
        fi
    fi

    # If both submitted, set up Step 7 coordinator and exit
    if [[ "${BATCH2_SUBMITTED}" == "true" ]] && [[ "${BATCH3_SUBMITTED}" == "true" ]]; then
        log "Both batches submitted. Creating Step 7 coordinator..."
        DEPS="afterany:14180452"
        [[ -n "${BATCH2_JOB}" ]] && DEPS="${DEPS},afterany:${BATCH2_JOB}"
        [[ -n "${BATCH3_JOB}" ]] && DEPS="${DEPS},afterany:${BATCH3_JOB}"
        COORD=$(sbatch --parsable \
            --dependency="${DEPS}" \
            --job-name=coord_step7 \
            --partition=compute \
            --qos=small \
            --time=01:00:00 \
            --ntasks=1 --cpus-per-task=1 --mem=4G \
            --output="${LOG_DIR}/coord_step7_%j.out" \
            --error="${LOG_DIR}/coord_step7_%j.err" \
            --wrap="cd ${BASE} && bash ${SCRIPT_DIR}/submit_next_step.sh 7" 2>&1)
        log "Coordinator -> Step 7: Job ${COORD}"
        log "All batches submitted. Monitor complete."
        exit 0
    fi

    sleep 300  # Check every 5 minutes
done

log "Monitor timed out after 48 hours."
