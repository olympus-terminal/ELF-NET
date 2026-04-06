#!/bin/bash
# =============================================================================
# Novel Protein Domain Discovery Pipeline — Master Orchestrator
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_domains/submit_pipeline.sh
#   Generated: 2026-02-21
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Purpose:
#   Submit the novel domain discovery pipeline using phased submission.
#   Jubail enforces MaxSubmitJobs=1500 counting each array element,
#   so we submit one step at a time. After each step completes, a
#   lightweight coordinator job submits the next step automatically.
#
#   Step 0 (audit) should be run interactively BEFORE this script.
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   bash MANUSCRIPT/scripts/novel_domains/submit_pipeline.sh
#
# =============================================================================

set -euo pipefail

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
SCRIPT_DIR="${BASE}/MANUSCRIPT/scripts/novel_domains"
NOVEL_DIR="${BASE}/03_analyses/novel_domains"
LOG_DIR="${BASE}/logs/novel_domains"
STATE_FILE="${NOVEL_DIR}/pipeline_state.txt"

echo "========================================================"
echo "  Novel Domain Discovery Pipeline — Phased Submission"
echo "========================================================"
echo ""
echo "Date:       $(date)"
echo "Host:       $(hostname)"
echo "Base dir:   ${BASE}"
echo "Script dir: ${SCRIPT_DIR}"
echo ""

# =============================================================================
# Pre-flight checks
# =============================================================================

echo "--- Pre-flight Checks ---"
echo ""

ERRORS=0

# Check sample list
SAMPLE_LIST="${NOVEL_DIR}/sample_list.txt"
if [[ ! -f "${SAMPLE_LIST}" ]]; then
    echo "  ERROR: sample_list.txt not found at ${SAMPLE_LIST}"
    echo "         Run 00_audit_hpc.sh first."
    ERRORS=$((ERRORS + 1))
else
    N_SAMPLES=$(wc -l < "${SAMPLE_LIST}")
    echo "  Sample list: ${SAMPLE_LIST} (${N_SAMPLES} samples)"
fi

# Check required scripts exist
MISSING_SCRIPTS=0
for script in \
    01_extract_dark_ids.sbatch \
    02_extract_dark_fasta.sbatch \
    03_filter_concat.sbatch \
    04_mmseqs2_cluster.sbatch \
    05a_prep_hmm_input.sbatch \
    05b_build_hmms_v2.sbatch \
    05c_concat_hmms.sh \
    06_hmmsearch_novel.sbatch \
    06b_build_count_matrix.py \
    07_characterize.sbatch \
    08_env_correlation.sbatch \
    submit_next_step.sh; do
    if [[ ! -f "${SCRIPT_DIR}/${script}" ]]; then
        echo "  ERROR: Missing script: ${script}"
        MISSING_SCRIPTS=$((MISSING_SCRIPTS + 1))
    fi
done
ERRORS=$((ERRORS + MISSING_SCRIPTS))
if [[ ${MISSING_SCRIPTS} -eq 0 ]]; then
    echo "  All pipeline scripts: present"
fi

# Check log directory
mkdir -p "${LOG_DIR}"
echo "  Log directory: ${LOG_DIR}"

# Check algal sequences directory
ALGAL_DIR="${BASE}/03_analyses/algae_proteins"
if [[ ! -d "${ALGAL_DIR}" ]]; then
    echo "  ERROR: Algal sequences directory not found: ${ALGAL_DIR}"
    ERRORS=$((ERRORS + 1))
else
    echo "  Algal sequences: ${ALGAL_DIR}"
fi

# Check novel_domains output directory
mkdir -p "${NOVEL_DIR}"
echo "  Output directory: ${NOVEL_DIR}"

echo ""

if [[ ${ERRORS} -gt 0 ]]; then
    echo "  ABORTING: ${ERRORS} pre-flight error(s) found."
    echo "  Fix the issues above and re-run."
    exit 1
fi

echo "  All checks passed."
echo ""

# =============================================================================
# Check available SLURM capacity
# =============================================================================

CURRENT_JOBS=$(squeue -u "$(whoami)" -r -h | wc -l)
MAX_SUBMIT=1500
AVAILABLE=$(( MAX_SUBMIT - CURRENT_JOBS - 5 ))

echo "--- SLURM Capacity ---"
echo "  Current jobs (expanded):  ${CURRENT_JOBS}"
echo "  MaxSubmitJobs limit:      ${MAX_SUBMIT}"
echo "  Available slots:          ${AVAILABLE}"
echo ""

if [[ ${AVAILABLE} -lt 200 ]]; then
    echo "  WARNING: Only ${AVAILABLE} slots available."
    echo "  Need at least 200 to start Step 1 efficiently."
    echo "  Consider waiting for existing jobs to complete."
    echo ""
fi

# =============================================================================
# Submit Step 1 — fit within available capacity
# =============================================================================

echo "--- Submitting Step 1 (Extract Dark IDs) ---"
echo ""

# Initialize state file
cat > "${STATE_FILE}" <<EOF
# Pipeline State File
# Generated: $(date -Iseconds)
N_SAMPLES=${N_SAMPLES}
EOF

# Determine how to batch Step 1
if [[ ${AVAILABLE} -ge ${N_SAMPLES} ]]; then
    # Can submit all at once (unlikely with current queue load)
    JOB1=$(sbatch --parsable --array=1-${N_SAMPLES}%100 \
        "${SCRIPT_DIR}/01_extract_dark_ids.sbatch")
    echo "  Step 1: Job ${JOB1} (array 1-${N_SAMPLES})"
    echo "STEP1_JOB=${JOB1}" >> "${STATE_FILE}"
    LAST_STEP1_JOB=${JOB1}
else
    # Split into batches that fit
    BATCH_SIZE=$(( AVAILABLE - 2 ))  # Reserve 2 for coordinator jobs
    if [[ ${BATCH_SIZE} -lt 100 ]]; then
        echo "  ERROR: Not enough slots (${AVAILABLE}) to submit even 100 tasks."
        echo "  Wait for existing jobs to complete."
        exit 1
    fi
    N_BATCHES=$(( (N_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE ))
    echo "  Splitting ${N_SAMPLES} samples into ${N_BATCHES} sequential batches of ~${BATCH_SIZE}"
    echo "  (Each batch waits for the previous to complete)"
    echo ""

    PREV_JOB=""
    for (( i=1; i<=N_BATCHES; i++ )); do
        BSTART=$(( (i - 1) * BATCH_SIZE + 1 ))
        BEND=$(( i * BATCH_SIZE ))
        [[ ${BEND} -gt ${N_SAMPLES} ]] && BEND=${N_SAMPLES}
        BCOUNT=$(( BEND - BSTART + 1 ))

        if [[ -z "${PREV_JOB}" ]]; then
            # First batch: submit directly
            JOBID=$(sbatch --parsable --array=${BSTART}-${BEND}%100 \
                "${SCRIPT_DIR}/01_extract_dark_ids.sbatch")
            echo "  Step 1 batch ${i}/${N_BATCHES}: Job ${JOBID} (array ${BSTART}-${BEND}, ${BCOUNT} tasks)"
        else
            # Subsequent batches: submit via coordinator after previous finishes
            # This keeps total queued jobs under the limit
            JOBID=$(sbatch --parsable --dependency=afterany:${PREV_JOB} \
                --job-name=dark_ids_b${i} \
                --output="${LOG_DIR}/01_batch${i}_%j.out" \
                --error="${LOG_DIR}/01_batch${i}_%j.err" \
                --time=00:10:00 --cpus-per-task=1 --mem=1G --partition=compute \
                --wrap="sbatch --array=${BSTART}-${BEND}%100 ${SCRIPT_DIR}/01_extract_dark_ids.sbatch")
            echo "  Step 1 batch ${i}/${N_BATCHES}: deferred Job ${JOBID} (will submit array ${BSTART}-${BEND}, ${BCOUNT} tasks)"
        fi
        echo "STEP1_BATCH${i}=${JOBID} # array ${BSTART}-${BEND}" >> "${STATE_FILE}"
        PREV_JOB=${JOBID}
    done
    LAST_STEP1_JOB=${PREV_JOB}
fi

echo ""

# Coordinator: after Step 1 completes, submit Step 2
COORD=$(sbatch --parsable --dependency=afterany:${LAST_STEP1_JOB} \
    --job-name=coord_step2 \
    --output="${LOG_DIR}/coord_step2_%j.out" \
    --error="${LOG_DIR}/coord_step2_%j.err" \
    --time=00:10:00 --cpus-per-task=1 --mem=1G --partition=compute \
    --wrap="bash ${SCRIPT_DIR}/submit_next_step.sh 2")
echo "  Coordinator → Step 2: Job ${COORD}"
echo "COORD_STEP2=${COORD}" >> "${STATE_FILE}"

echo ""
echo "========================================================"
echo "  Pipeline Submission Started"
echo "========================================================"
echo ""
echo "  Step 1 submitted (${N_SAMPLES} samples)"
echo "  Steps 2–8 will be submitted automatically by coordinator jobs"
echo ""
echo "  Monitor progress:"
echo "    squeue -u \$(whoami)"
echo "    cat ${STATE_FILE}"
echo ""
echo "  Log directory: ${LOG_DIR}"
echo "  State file:    ${STATE_FILE}"
echo ""
echo "  Started: $(date)"
