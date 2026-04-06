#!/bin/bash
#SBATCH --job-name=build_hmms_v2
#SBATCH --output=logs/novel_domains/05b_build_hmms_v2_%A_%a.out
#SBATCH --error=logs/novel_domains/05b_build_hmms_v2_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=compute
#SBATCH --array=1-100%50

# =============================================================================
# Step 5b (v2): Build HMM Profiles — Scalable Version
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_domains/05b_build_hmms_v2.sbatch
#   Generated: 2026-02-21
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Purpose:
#   Build HMM profiles from pre-indexed cluster members. Requires Step 5a
#   to have completed (cluster members pre-split into per-cluster files).
#
#   Each array task processes BATCH_SIZE clusters:
#   1. Load pre-extracted sequences from hmm_target_sequences.fa into memory
#   2. For each cluster: extract members, align with MAFFT, build HMM
#
#   Step 5a pre-extracts all target member sequences into a small FASTA
#   (~700MB vs 50GB), so loading into memory is fast.
#
# Prerequisites:
#   - Step 5a completed: hmm_target_clusters.txt, cluster_members/, hmm_target_sequences.fa
#
# Input:
#   - 03_analyses/novel_domains/hmm_target_clusters.txt
#   - 03_analyses/novel_domains/cluster_members/{rep_id}.members.txt
#   - 03_analyses/novel_domains/hmm_target_sequences.fa (pre-extracted by 5a)
#
# Output:
#   - 03_analyses/novel_domains/hmms/per_cluster/{rep_id}.hmm
#   - 03_analyses/novel_domains/cluster_alignments/{rep_id}.afa
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   N=$(wc -l < 03_analyses/novel_domains/hmm_target_clusters.txt)
#   NBATCHES=$(( (N + 99) / 100 ))
#   sbatch --array=1-${NBATCHES}%50 MANUSCRIPT/scripts/novel_domains/05b_build_hmms_v2.sbatch
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CRITICAL ENVIRONMENT SETUP
# =============================================================================

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp/hmm_v2_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $TMPDIR
export PYTHONUSERBASE=/scratch/drn2/newhome/.local

module purge
module load gcc/13.2.0
module load hmmer/3.3

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

# =============================================================================
# Configuration
# =============================================================================

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
NOVEL_DIR="${BASE}/03_analyses/novel_domains"
INPUT_FA="${NOVEL_DIR}/hmm_target_sequences.fa"
TARGET_LIST="${NOVEL_DIR}/hmm_target_clusters.txt"
MEMBER_DIR="${NOVEL_DIR}/cluster_members"
HMM_DIR="${NOVEL_DIR}/hmms/per_cluster"
ALN_DIR="${NOVEL_DIR}/cluster_alignments"
BATCH_SIZE=100

mkdir -p "${HMM_DIR}" "${ALN_DIR}"

echo "========================================"
echo "Step 5b (v2): Build HMM Profiles"
echo "========================================"
echo ""
echo "Job ID:     ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "Date:       $(date)"
echo ""

# =============================================================================
# Validate inputs
# =============================================================================

if [[ ! -f "${TARGET_LIST}" ]]; then
    echo "ERROR: hmm_target_clusters.txt not found. Run Step 5a first."
    exit 1
fi

TOTAL_CLUSTERS=$(wc -l < "${TARGET_LIST}")
echo "  Total target clusters: ${TOTAL_CLUSTERS}"

# Calculate batch range for this array task
START_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) * BATCH_SIZE + 1 ))
END_IDX=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
if [[ ${END_IDX} -gt ${TOTAL_CLUSTERS} ]]; then
    END_IDX=${TOTAL_CLUSTERS}
fi

if [[ ${START_IDX} -gt ${TOTAL_CLUSTERS} ]]; then
    echo "  Task ${SLURM_ARRAY_TASK_ID}: no clusters to process (start=${START_IDX} > total=${TOTAL_CLUSTERS})"
    exit 0
fi

BATCH_COUNT=$(( END_IDX - START_IDX + 1 ))
echo "  Processing clusters ${START_IDX}-${END_IDX} (batch ${SLURM_ARRAY_TASK_ID}, ${BATCH_COUNT} clusters)"
echo ""

# Get list of cluster IDs for this batch
BATCH_CLUSTERS_FILE="${TMPDIR}/batch_clusters.txt"
sed -n "${START_IDX},${END_IDX}p" "${TARGET_LIST}" > "${BATCH_CLUSTERS_FILE}"

# =============================================================================
# Phase 1: Check which clusters need processing
# =============================================================================

echo "  Phase 1: Checking which clusters need processing..."

NEED_PROCESSING="${TMPDIR}/need_processing.txt"
> "${NEED_PROCESSING}"

N_SKIP=0
while IFS= read -r REP_ID; do
    if [[ -f "${HMM_DIR}/${REP_ID}.hmm" ]]; then
        N_SKIP=$((N_SKIP + 1))
        continue
    fi
    echo "${REP_ID}" >> "${NEED_PROCESSING}"
done < "${BATCH_CLUSTERS_FILE}"

N_NEED=$(wc -l < "${NEED_PROCESSING}")
echo "    Clusters to process: ${N_NEED} (skipped ${N_SKIP} existing)"

if [[ ${N_NEED} -eq 0 ]]; then
    echo "  All clusters already processed, nothing to do."
    rm -rf "${TMPDIR}"
    exit 0
fi

# Validate pre-extracted FASTA exists
if [[ ! -f "${INPUT_FA}" ]]; then
    echo "  ERROR: Pre-extracted FASTA not found: ${INPUT_FA}"
    echo "  Run Step 5a first."
    rm -rf "${TMPDIR}"
    exit 1
fi

# =============================================================================
# Phase 2: Load sequences and build HMMs
# =============================================================================

echo "  Phase 2: Loading pre-extracted sequences and building HMMs..."
START_TIME=$(date +%s)

python3 << 'PYEOF'
import os
import sys
import subprocess

tmpdir = os.environ["TMPDIR"]
member_dir = os.environ.get("MEMBER_DIR",
    "/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains/cluster_members")
hmm_dir = os.environ.get("HMM_DIR",
    "/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains/hmms/per_cluster")
aln_dir = os.environ.get("ALN_DIR",
    "/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains/cluster_alignments")
extracted_fa = os.environ.get("INPUT_FA",
    "/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains/hmm_target_sequences.fa")
need_processing = os.path.join(tmpdir, "need_processing.txt")

# Load all extracted sequences into a dict
print("    Loading extracted sequences into memory...")
seqs = {}
current_id = None
current_seq = []
with open(extracted_fa) as f:
    for line in f:
        if line.startswith('>'):
            if current_id:
                seqs[current_id] = ''.join(current_seq)
            current_id = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line)
    if current_id:
        seqs[current_id] = ''.join(current_seq)
print(f"    Loaded {len(seqs)} sequences into memory")
sys.stdout.flush()

# Process each cluster
clusters = [line.strip() for line in open(need_processing)]
n_success = 0
n_fail = 0
n_too_few = 0

for i, rep_id in enumerate(clusters):
    if (i + 1) % 10 == 0:
        print(f"    Processing cluster {i+1}/{len(clusters)} ({n_success} HMMs built)...")
        sys.stdout.flush()

    member_file = os.path.join(member_dir, f"{rep_id}.members.txt")
    if not os.path.exists(member_file):
        n_fail += 1
        continue

    # Read member IDs
    with open(member_file) as f:
        member_ids = [line.strip() for line in f if line.strip()]

    # Get sequences for this cluster
    cluster_seqs = [(mid, seqs[mid]) for mid in member_ids if mid in seqs]

    if len(cluster_seqs) < 3:
        n_too_few += 1
        continue

    # Write per-cluster FASTA to temp file
    cluster_fa = os.path.join(tmpdir, f"{rep_id}.fa")
    with open(cluster_fa, 'w') as f:
        for sid, seq in cluster_seqs:
            f.write(f">{sid}\n{seq}")

    # Align with MAFFT
    aln_file = os.path.join(aln_dir, f"{rep_id}.afa")
    try:
        result = subprocess.run(
            ["mafft", "--auto", "--thread", "2", "--quiet", cluster_fa],
            capture_output=True, timeout=600
        )
        if result.returncode != 0 or not result.stdout:
            print(f"    WARNING: MAFFT failed for {rep_id} ({len(cluster_seqs)} seqs)")
            n_fail += 1
            os.remove(cluster_fa)
            continue
        with open(aln_file, 'wb') as f:
            f.write(result.stdout)
    except subprocess.TimeoutExpired:
        print(f"    WARNING: MAFFT timed out for {rep_id} ({len(cluster_seqs)} seqs)")
        n_fail += 1
        os.remove(cluster_fa)
        continue

    if not os.path.exists(aln_file) or os.path.getsize(aln_file) == 0:
        print(f"    WARNING: Empty alignment for {rep_id}")
        n_fail += 1
        os.remove(cluster_fa)
        continue

    # Build HMM
    hmm_file = os.path.join(hmm_dir, f"{rep_id}.hmm")
    try:
        result = subprocess.run(
            ["hmmbuild", "--amino", "--cpu", "2", "-n", f"NOVEL_{rep_id}",
             hmm_file, aln_file],
            capture_output=True, timeout=120
        )
        if result.returncode == 0 and os.path.exists(hmm_file):
            n_success += 1
        else:
            print(f"    WARNING: hmmbuild failed for {rep_id}")
            n_fail += 1
    except subprocess.TimeoutExpired:
        print(f"    WARNING: hmmbuild timed out for {rep_id}")
        n_fail += 1

    # Cleanup temp FASTA
    os.remove(cluster_fa)

print(f"")
print(f"    Results:")
print(f"      Successful HMMs: {n_success}")
print(f"      Too few seqs:    {n_too_few}")
print(f"      Failed:          {n_fail}")
print(f"      Total processed: {len(clusters)}")
PYEOF

END_TIME=$(date +%s)
echo ""
echo "  Total elapsed: $((END_TIME - START_TIME)) seconds"
echo "  Skipped (existing): ${N_SKIP}"
echo ""
echo "  Done: $(date)"

# Cleanup
rm -rf "${TMPDIR}"
