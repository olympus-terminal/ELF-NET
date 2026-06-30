#!/bin/bash
# =============================================================================
# submit_next_step.sh — Coordinator script for phased pipeline submission
# =============================================================================
#
# Called by coordinator SLURM jobs after each step completes.
# Submits the next step in the pipeline, dynamically splitting
# array jobs to stay under the MaxSubmitJobs=1500 limit.
#
# Usage:
#   bash submit_next_step.sh <step_number>
#
# =============================================================================

set -euo pipefail

STEP=${1:?Usage: submit_next_step.sh <step_number>}

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
SCRIPT_DIR="${BASE}/MANUSCRIPT/scripts/novel_domains"
NOVEL_DIR="${BASE}/03_analyses/novel_domains"
LOG_DIR="${BASE}/logs/novel_domains"
STATE_FILE="${NOVEL_DIR}/pipeline_state.txt"
SAMPLE_LIST="${NOVEL_DIR}/sample_list.txt"

N_SAMPLES=$(wc -l < "${SAMPLE_LIST}")
MAX_SUBMIT=1500

echo "=== submit_next_step.sh: Step ${STEP} ==="
echo "Date: $(date)"
echo "N_SAMPLES: ${N_SAMPLES}"

# -------------------------------------------------------
# Helper: submit an array step with automatic batch splitting
# Arguments: step_name sbatch_script throttle next_step
# -------------------------------------------------------
submit_array_step() {
    local STEP_NAME=$1
    local SBATCH_SCRIPT=$2
    local THROTTLE=$3
    local NEXT_STEP=$4

    local CURRENT_JOBS
    CURRENT_JOBS=$(squeue -u "$(whoami)" -r -h | wc -l)
    local AVAILABLE=$(( MAX_SUBMIT - CURRENT_JOBS - 5 ))

    echo "  Current jobs: ${CURRENT_JOBS}, available slots: ${AVAILABLE}"

    if [[ ${AVAILABLE} -ge ${N_SAMPLES} ]]; then
        # Submit all at once
        local JOBID
        JOBID=$(sbatch --parsable --array=1-${N_SAMPLES}%${THROTTLE} \
            "${SBATCH_SCRIPT}")
        echo "  ${STEP_NAME}: Job ${JOBID} (array 1-${N_SAMPLES})"
        echo "${STEP_NAME}_JOB=${JOBID}" >> "${STATE_FILE}"
        LAST_JOB=${JOBID}
    else
        # Split into sequential batches
        local BATCH_SIZE=$(( AVAILABLE - 2 ))
        if [[ ${BATCH_SIZE} -lt 100 ]]; then
            echo "  ERROR: Only ${AVAILABLE} slots, need at least 100. Retrying in 5 min..."
            # Resubmit ourselves with a delay
            local RETRY
            RETRY=$(sbatch --parsable \
                --begin=now+5minutes \
                --job-name=retry_${STEP_NAME} \
                --output="${LOG_DIR}/retry_${STEP_NAME}_%j.out" \
                --error="${LOG_DIR}/retry_${STEP_NAME}_%j.err" \
                --time=00:10:00 --cpus-per-task=1 --mem=1G --partition=compute \
                --wrap="bash ${SCRIPT_DIR}/submit_next_step.sh ${STEP}")
            echo "  Retry job: ${RETRY} (will try again in 5 minutes)"
            echo "${STEP_NAME}_RETRY=${RETRY}" >> "${STATE_FILE}"
            exit 0
        fi

        local N_BATCHES=$(( (N_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE ))
        echo "  Splitting into ${N_BATCHES} sequential batches of ~${BATCH_SIZE}"

        local PREV_JOB=""
        for (( i=1; i<=N_BATCHES; i++ )); do
            local BSTART=$(( (i - 1) * BATCH_SIZE + 1 ))
            local BEND=$(( i * BATCH_SIZE ))
            [[ ${BEND} -gt ${N_SAMPLES} ]] && BEND=${N_SAMPLES}

            if [[ -z "${PREV_JOB}" ]]; then
                JOBID=$(sbatch --parsable --array=${BSTART}-${BEND}%${THROTTLE} \
                    "${SBATCH_SCRIPT}")
                echo "  ${STEP_NAME} batch ${i}/${N_BATCHES}: Job ${JOBID} (array ${BSTART}-${BEND})"
            else
                JOBID=$(sbatch --parsable --dependency=afterany:${PREV_JOB} \
                    --job-name=${STEP_NAME}_b${i} \
                    --output="${LOG_DIR}/${STEP_NAME}_batch${i}_%j.out" \
                    --error="${LOG_DIR}/${STEP_NAME}_batch${i}_%j.err" \
                    --time=00:10:00 --cpus-per-task=1 --mem=1G --partition=compute \
                    --wrap="sbatch --array=${BSTART}-${BEND}%${THROTTLE} ${SBATCH_SCRIPT}")
                echo "  ${STEP_NAME} batch ${i}/${N_BATCHES}: deferred Job ${JOBID} (array ${BSTART}-${BEND})"
            fi
            echo "${STEP_NAME}_BATCH${i}=${JOBID}" >> "${STATE_FILE}"
            PREV_JOB=${JOBID}
        done
        LAST_JOB=${PREV_JOB}
    fi
}

# -------------------------------------------------------
# Helper: submit a single-node step
# Arguments: step_name sbatch_script next_step
# -------------------------------------------------------
submit_single_step() {
    local STEP_NAME=$1
    local SBATCH_SCRIPT=$2
    local NEXT_STEP=$3

    local JOBID
    JOBID=$(sbatch --parsable "${SBATCH_SCRIPT}")
    echo "  ${STEP_NAME}: Job ${JOBID}"
    echo "${STEP_NAME}_JOB=${JOBID}" >> "${STATE_FILE}"
    LAST_JOB=${JOBID}
}

# -------------------------------------------------------
# Main dispatch
# -------------------------------------------------------

LAST_JOB=""

case ${STEP} in
    2)
        echo "--- Step 2: Extract Dark FASTA ---"
        submit_array_step "step2" "${SCRIPT_DIR}/02_extract_dark_fasta.sbatch" 100 3
        # Coordinator → Step 3
        COORD=$(sbatch --parsable --dependency=afterany:${LAST_JOB} \
            --job-name=coord_step3 \
            --output="${LOG_DIR}/coord_step3_%j.out" \
            --error="${LOG_DIR}/coord_step3_%j.err" \
            --time=00:10:00 --cpus-per-task=1 --mem=1G --partition=compute \
            --wrap="bash ${SCRIPT_DIR}/submit_next_step.sh 3")
        echo "  Coordinator → Step 3: Job ${COORD}"
        echo "COORD_STEP3=${COORD}" >> "${STATE_FILE}"
        ;;

    3)
        echo "--- Step 3: Filter & Concatenate ---"
        submit_single_step "step3" "${SCRIPT_DIR}/03_filter_concat.sbatch" 4
        COORD=$(sbatch --parsable --dependency=afterok:${LAST_JOB} \
            --job-name=coord_step4 \
            --output="${LOG_DIR}/coord_step4_%j.out" \
            --error="${LOG_DIR}/coord_step4_%j.err" \
            --time=00:10:00 --cpus-per-task=1 --mem=1G --partition=compute \
            --wrap="bash ${SCRIPT_DIR}/submit_next_step.sh 4")
        echo "  Coordinator → Step 4: Job ${COORD}"
        echo "COORD_STEP4=${COORD}" >> "${STATE_FILE}"
        ;;

    4)
        echo "--- Step 4: MMseqs2 Clustering ---"
        submit_single_step "step4" "${SCRIPT_DIR}/04_mmseqs2_cluster.sbatch" 5
        COORD=$(sbatch --parsable --dependency=afterok:${LAST_JOB} \
            --job-name=coord_step5 \
            --output="${LOG_DIR}/coord_step5_%j.out" \
            --error="${LOG_DIR}/coord_step5_%j.err" \
            --time=00:10:00 --cpus-per-task=1 --mem=1G --partition=compute \
            --wrap="bash ${SCRIPT_DIR}/submit_next_step.sh 5")
        echo "  Coordinator → Step 5: Job ${COORD}"
        echo "COORD_STEP5=${COORD}" >> "${STATE_FILE}"
        ;;

    5)
        echo "--- Step 5: Prep HMM Input (Phase 1) ---"
        # Phase 1: Build faidx, filter clusters, pre-split membership
        submit_single_step "step5a" "${SCRIPT_DIR}/05a_prep_hmm_input.sbatch" 5b
        COORD=$(sbatch --parsable --dependency=afterok:${LAST_JOB} \
            --job-name=coord_step5b \
            --output="${LOG_DIR}/coord_step5b_%j.out" \
            --error="${LOG_DIR}/coord_step5b_%j.err" \
            --time=00:10:00 --cpus-per-task=1 --mem=1G --partition=compute \
            --wrap="bash ${SCRIPT_DIR}/submit_next_step.sh 5b")
        echo "  Coordinator → Step 5b: Job ${COORD}"
        echo "COORD_STEP5B=${COORD}" >> "${STATE_FILE}"
        ;;

    5b)
        echo "--- Step 5b: Build HMMs (Phase 2) ---"
        # Phase 2: Array job to build HMMs from pre-indexed clusters
        # Dynamically compute array size from hmm_target_clusters.txt
        TARGET_LIST_FILE="${NOVEL_DIR}/hmm_target_clusters.txt"
        if [[ ! -f "${TARGET_LIST_FILE}" ]]; then
            echo "  ERROR: hmm_target_clusters.txt not found. Step 5a may not have completed."
            exit 1
        fi
        N_TARGETS=$(wc -l < "${TARGET_LIST_FILE}")
        BATCH_SZ=100
        N_ARRAY=$(( (N_TARGETS + BATCH_SZ - 1) / BATCH_SZ ))
        echo "  Target clusters: ${N_TARGETS}, batch size: ${BATCH_SZ}, array tasks: ${N_ARRAY}"

        CURRENT_JOBS=$(squeue -u "$(whoami)" -r -h | wc -l)
        AVAILABLE=$(( MAX_SUBMIT - CURRENT_JOBS - 5 ))
        echo "  SLURM slots available: ${AVAILABLE}"

        if [[ ${AVAILABLE} -lt 50 ]]; then
            echo "  ERROR: Only ${AVAILABLE} slots. Retrying in 5 min..."
            RETRY=$(sbatch --parsable --begin=now+5minutes \
                --job-name=retry_step5b \
                --output="${LOG_DIR}/retry_step5b_%j.out" \
                --error="${LOG_DIR}/retry_step5b_%j.err" \
                --time=00:10:00 --cpus-per-task=1 --mem=1G --partition=compute \
                --wrap="bash ${SCRIPT_DIR}/submit_next_step.sh 5b")
            echo "  Retry job: ${RETRY}"
            echo "STEP5B_RETRY=${RETRY}" >> "${STATE_FILE}"
            exit 0
        fi

        JOBID=$(sbatch --parsable --array=1-${N_ARRAY}%50 \
            "${SCRIPT_DIR}/05b_build_hmms_v2.sbatch")
        echo "  step5b: Job ${JOBID} (array 1-${N_ARRAY})"
        echo "STEP5B_JOB=${JOBID}" >> "${STATE_FILE}"
        LAST_JOB=${JOBID}

        COORD=$(sbatch --parsable --dependency=afterany:${LAST_JOB} \
            --job-name=coord_step5c \
            --output="${LOG_DIR}/coord_step5c_%j.out" \
            --error="${LOG_DIR}/coord_step5c_%j.err" \
            --time=00:10:00 --cpus-per-task=1 --mem=1G --partition=compute \
            --wrap="bash ${SCRIPT_DIR}/submit_next_step.sh 5c")
        echo "  Coordinator → Step 5c: Job ${COORD}"
        echo "COORD_STEP5C=${COORD}" >> "${STATE_FILE}"
        ;;

    5c)
        echo "--- Step 5c: Concatenate & Press HMMs (Phase 3) ---"
        JOB5C=$(sbatch --parsable \
            --job-name=concat_hmms \
            --output="${LOG_DIR}/05c_concat_hmms_%j.out" \
            --error="${LOG_DIR}/05c_concat_hmms_%j.err" \
            --time=01:00:00 --cpus-per-task=4 --mem=16G --partition=compute \
            --wrap="bash ${SCRIPT_DIR}/05c_concat_hmms.sh")
        echo "  Step 5c: Job ${JOB5C}"
        echo "STEP5C_JOB=${JOB5C}" >> "${STATE_FILE}"
        LAST_JOB=${JOB5C}
        COORD=$(sbatch --parsable --dependency=afterok:${LAST_JOB} \
            --job-name=coord_step6 \
            --output="${LOG_DIR}/coord_step6_%j.out" \
            --error="${LOG_DIR}/coord_step6_%j.err" \
            --time=00:10:00 --cpus-per-task=1 --mem=1G --partition=compute \
            --wrap="bash ${SCRIPT_DIR}/submit_next_step.sh 6")
        echo "  Coordinator → Step 6: Job ${COORD}"
        echo "COORD_STEP6=${COORD}" >> "${STATE_FILE}"
        ;;

    6)
        echo "--- Step 6: hmmsearch Novel HMMs ---"
        submit_array_step "step6" "${SCRIPT_DIR}/06_hmmsearch_novel.sbatch" 50 7
        COORD=$(sbatch --parsable --dependency=afterany:${LAST_JOB} \
            --job-name=coord_step7 \
            --output="${LOG_DIR}/coord_step7_%j.out" \
            --error="${LOG_DIR}/coord_step7_%j.err" \
            --time=00:10:00 --cpus-per-task=1 --mem=1G --partition=compute \
            --wrap="bash ${SCRIPT_DIR}/submit_next_step.sh 7")
        echo "  Coordinator → Step 7: Job ${COORD}"
        echo "COORD_STEP7=${COORD}" >> "${STATE_FILE}"
        ;;

    7)
        echo "--- Step 7: Characterize Novel Families ---"
        submit_single_step "step7" "${SCRIPT_DIR}/07_characterize.sbatch" 8
        COORD=$(sbatch --parsable --dependency=afterok:${LAST_JOB} \
            --job-name=coord_step8 \
            --output="${LOG_DIR}/coord_step8_%j.out" \
            --error="${LOG_DIR}/coord_step8_%j.err" \
            --time=00:10:00 --cpus-per-task=1 --mem=1G --partition=compute \
            --wrap="bash ${SCRIPT_DIR}/submit_next_step.sh 8")
        echo "  Coordinator → Step 8: Job ${COORD}"
        echo "COORD_STEP8=${COORD}" >> "${STATE_FILE}"
        ;;

    8)
        echo "--- Step 8: Environmental Correlation (FINAL) ---"
        submit_single_step "step8" "${SCRIPT_DIR}/08_env_correlation.sbatch" done
        echo "PIPELINE_FINAL_JOB=${LAST_JOB}" >> "${STATE_FILE}"
        echo "PIPELINE_COMPLETE_PENDING=$(date -Iseconds)" >> "${STATE_FILE}"
        echo ""
        echo "=== FINAL STEP SUBMITTED ==="
        echo "Pipeline will be complete when Job ${LAST_JOB} finishes."
        ;;

    *)
        echo "ERROR: Unknown step: ${STEP}"
        exit 1
        ;;
esac

echo ""
echo "State file updated: ${STATE_FILE}"
echo "Completed: $(date)"
