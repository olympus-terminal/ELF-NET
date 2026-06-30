#!/bin/bash
# =============================================================================
# Novel Domains Pipeline — REMEDIATION RE-RUN
# =============================================================================
#
# Reason: Previous run (Feb 21) used deprecated pfam_results/ (AlKhidr subset).
#         Now using hmmsearch_results/ (full Pfam-A, 2044 assemblies).
#
# SLURM QOS 'normal' allows max ~1000 submitted jobs at once.
# Strategy: submit Step 01 (2 × ~500 arrays), then a coordinator job
# that submits Steps 02-04 after Step 01 completes.
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   bash MANUSCRIPT/scripts/novel_domains/submit_remediation.sh
# =============================================================================

set -euo pipefail

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
SCRIPTS="${BASE}/MANUSCRIPT/scripts/novel_domains"
LOGDIR="${BASE}/logs/novel_domains"
SAMPLE_LIST="${BASE}/03_analyses/novel_domains/sample_list.txt"

mkdir -p "${LOGDIR}"

N=$(wc -l < "${SAMPLE_LIST}")
MID=$(( N / 2 ))           # 1022
MID1=$(( MID + 1 ))        # 1023

echo "=============================================="
echo "  Novel Domains — Remediation Pipeline"
echo "=============================================="
echo "Date:       $(date)"
echo "Samples:    ${N}"
echo "hmmsearch:  ${BASE}/03_analyses/hmmsearch_results/"
echo ""

# Verify data
if [[ -d "${BASE}/03_analyses/pfam_results" ]]; then
    echo "ERROR: pfam_results/ still exists!"; exit 1
fi
if [[ ! -d "${BASE}/03_analyses/hmmsearch_results" ]]; then
    echo "ERROR: hmmsearch_results/ not found!"; exit 1
fi
HMM_COUNT=$(ls "${BASE}/03_analyses/hmmsearch_results/"*.hmmsearch.tbl 2>/dev/null | wc -l)
echo "hmmsearch .tbl files: ${HMM_COUNT}"
[[ ${HMM_COUNT} -lt 2000 ]] && echo "ERROR: too few .tbl files" && exit 1

# ---- Step 01: Extract dark IDs (2 batches to stay under 1000 job limit) ----
echo ""
echo "=== Step 01: Extract dark IDs ==="
JOB1a=$(sbatch --parsable \
    --array=1-${MID}%100 \
    --output="${LOGDIR}/01_dark_ids_%A_%a.out" \
    --error="${LOGDIR}/01_dark_ids_%A_%a.err" \
    "${SCRIPTS}/01_extract_dark_ids.sbatch")
echo "  Batch A (1-${MID}): ${JOB1a}"

JOB1b=$(sbatch --parsable \
    --array=${MID1}-${N}%100 \
    --output="${LOGDIR}/01_dark_ids_%A_%a.out" \
    --error="${LOGDIR}/01_dark_ids_%A_%a.err" \
    "${SCRIPTS}/01_extract_dark_ids.sbatch")
echo "  Batch B (${MID1}-${N}): ${JOB1b}"

# ---- Coordinator: submits Steps 02-04 after Step 01 completes ----
echo ""
echo "=== Coordinator job (chains Steps 02-04) ==="

COORD_SCRIPT="${LOGDIR}/coord_steps_02_04.sh"
cat > "${COORD_SCRIPT}" <<COORDEOF
#!/bin/bash
#SBATCH --job-name=coord_02_04
#SBATCH --output=${LOGDIR}/coord_02_04_%j.out
#SBATCH --error=${LOGDIR}/coord_02_04_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --partition=compute

set -euo pipefail
export HOME=/scratch/drn2/newhome

echo "Coordinator started: \$(date)"
echo "Submitting Steps 02-04..."

N=${N}
MID=${MID}
MID1=${MID1}
SCRIPTS="${SCRIPTS}"
LOGDIR="${LOGDIR}"

# Step 02: Extract dark FASTA
JOB2a=\$(sbatch --parsable \\
    --array=1-\${MID}%100 \\
    --output="\${LOGDIR}/02_dark_fasta_%A_%a.out" \\
    --error="\${LOGDIR}/02_dark_fasta_%A_%a.err" \\
    "\${SCRIPTS}/02_extract_dark_fasta.sbatch")
echo "  Step 02 Batch A: \${JOB2a}"

JOB2b=\$(sbatch --parsable \\
    --array=\${MID1}-\${N}%100 \\
    --output="\${LOGDIR}/02_dark_fasta_%A_%a.out" \\
    --error="\${LOGDIR}/02_dark_fasta_%A_%a.err" \\
    "\${SCRIPTS}/02_extract_dark_fasta.sbatch")
echo "  Step 02 Batch B: \${JOB2b}"

# Step 03: Filter & concatenate (after Step 02)
JOB3=\$(sbatch --parsable \\
    --dependency=afterok:\${JOB2a}:\${JOB2b} \\
    --output="\${LOGDIR}/03_filter_concat_%j.out" \\
    --error="\${LOGDIR}/03_filter_concat_%j.err" \\
    "\${SCRIPTS}/03_filter_concat.sbatch")
echo "  Step 03: \${JOB3}"

# Step 04: MMseqs2 clustering (after Step 03)
JOB4=\$(sbatch --parsable \\
    --dependency=afterok:\${JOB3} \\
    --output="\${LOGDIR}/04_mmseqs2_%j.out" \\
    --error="\${LOGDIR}/04_mmseqs2_%j.err" \\
    "\${SCRIPTS}/04_mmseqs2_cluster.sbatch")
echo "  Step 04: \${JOB4}"

# Update pipeline state
cat >> ${BASE}/03_analyses/novel_domains/pipeline_state.txt <<EOF2
STEP02_A=\${JOB2a}
STEP02_B=\${JOB2b}
STEP03=\${JOB3}
STEP04=\${JOB4}
EOF2

echo "Steps 02-04 submitted. Done: \$(date)"
COORDEOF

JOB_COORD=$(sbatch --parsable \
    --dependency=afterok:${JOB1a}:${JOB1b} \
    "${COORD_SCRIPT}")
echo "  Coordinator: ${JOB_COORD} (runs after Step 01 finishes)"

# ---- Pipeline state ----
STATE_FILE="${BASE}/03_analyses/novel_domains/pipeline_state.txt"
cat > "${STATE_FILE}" <<EOF
# Pipeline State — REMEDIATION RE-RUN
# Generated: $(date -Iseconds)
# Reason: Previous run used deprecated pfam_results/ (AlKhidr)
# Now using: hmmsearch_results/ (full Pfam-A, Jan 14 2026)
N_SAMPLES=${N}
STEP01_A=${JOB1a}
STEP01_B=${JOB1b}
COORDINATOR=${JOB_COORD}
# Steps 02-04 job IDs appended by coordinator after Step 01 completes
# Steps 05-08: submit manually after Step 04 completes
EOF

echo ""
echo "=== Submitted ==="
echo "  Step 01A: ${JOB1a}  (array 1-${MID})"
echo "  Step 01B: ${JOB1b}  (array ${MID1}-${N})"
echo "  Coord:    ${JOB_COORD}  (submits Steps 02-04 after 01 finishes)"
echo ""
echo "Monitor: squeue -u drn2"
echo "Done: $(date)"
