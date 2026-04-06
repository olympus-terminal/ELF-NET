#!/bin/bash
#SBATCH --job-name=prep_hmm
#SBATCH --output=logs/novel_domains/05a_prep_hmm_input_%j.out
#SBATCH --error=logs/novel_domains/05a_prep_hmm_input_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=compute

# =============================================================================
# Step 5a: Prepare HMM Input — Pre-index and Pre-split Cluster Membership
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_domains/05a_prep_hmm_input.sbatch
#   Generated: 2026-02-21
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Purpose:
#   One-time preprocessing to enable fast, parallel HMM building:
#   1. Build samtools faidx on dark_proteome_filtered.fa (enables O(1) seq extraction)
#   2. Filter clusters to those with >=50 members (biologically meaningful threshold)
#   3. Pre-split cluster membership: read 29GB cluster TSV once, write per-cluster
#      member ID files to cluster_members/ directory
#
#   This avoids the O(N^2) problem in the original Step 5 where each array task
#   scanned the entire 29GB TSV for every cluster.
#
# Input:
#   - 03_analyses/novel_domains/dark_proteome_filtered.fa (50 GB)
#   - 03_analyses/novel_domains/clusters_30/clusters_30_cluster.tsv (29 GB)
#   - 03_analyses/novel_domains/clusters_30/cluster_sizes.tsv (sorted descending)
#
# Output:
#   - 03_analyses/novel_domains/dark_proteome_filtered.fa.fai
#   - 03_analyses/novel_domains/hmm_target_clusters.txt (clusters with >=50 members)
#   - 03_analyses/novel_domains/cluster_members/{rep_id}.members.txt
#   - 03_analyses/novel_domains/prep_hmm_stats.txt
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CRITICAL ENVIRONMENT SETUP
# =============================================================================

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp/prep_hmm_$$
mkdir -p $TMPDIR
export PYTHONUSERBASE=/scratch/drn2/newhome/.local

module purge
module load gcc/13.2.0
module load samtools/1.10

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

# =============================================================================
# Configuration
# =============================================================================

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
NOVEL_DIR="${BASE}/03_analyses/novel_domains"
INPUT_FA="${NOVEL_DIR}/dark_proteome_filtered.fa"
CLUSTER_TSV="${NOVEL_DIR}/clusters_30/clusters_30_cluster.tsv"
CLUSTER_SIZES="${NOVEL_DIR}/clusters_30/cluster_sizes.tsv"
MEMBER_DIR="${NOVEL_DIR}/cluster_members"
TARGET_LIST="${NOVEL_DIR}/hmm_target_clusters.txt"
STATS_FILE="${NOVEL_DIR}/prep_hmm_stats.txt"

MIN_CLUSTER_SIZE=50

echo "========================================"
echo "Step 5a: Prepare HMM Input"
echo "========================================"
echo ""
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "Date:       $(date)"
echo "Min cluster size: ${MIN_CLUSTER_SIZE}"
echo ""

# =============================================================================
# Phase 1: Build samtools faidx
# =============================================================================

echo "--- Phase 1: Build FASTA index ---"
echo ""

if [[ -f "${INPUT_FA}.fai" ]]; then
    echo "  FASTA index already exists: ${INPUT_FA}.fai"
    echo "  Sequences indexed: $(wc -l < "${INPUT_FA}.fai")"
else
    echo "  Building samtools faidx on $(du -sh "${INPUT_FA}" | cut -f1) FASTA..."
    START=$(date +%s)
    samtools faidx "${INPUT_FA}"
    END=$(date +%s)
    echo "  Done in $((END - START)) seconds"
    echo "  Sequences indexed: $(wc -l < "${INPUT_FA}.fai")"
fi
echo ""

# =============================================================================
# Phase 2: Filter clusters to target list (>= MIN_CLUSTER_SIZE members)
# =============================================================================

echo "--- Phase 2: Filter target clusters (>=${MIN_CLUSTER_SIZE} members) ---"
echo ""

# cluster_sizes.tsv is sorted descending by size: rep_id<TAB>count
# Extract rep IDs where count >= MIN_CLUSTER_SIZE
echo "  Reading cluster_sizes.tsv..."
START=$(date +%s)
awk -F'\t' -v min="${MIN_CLUSTER_SIZE}" '$2 >= min {print $1}' "${CLUSTER_SIZES}" > "${TARGET_LIST}"
END=$(date +%s)

N_TARGETS=$(wc -l < "${TARGET_LIST}")
echo "  Target clusters (>=${MIN_CLUSTER_SIZE} members): ${N_TARGETS}"
echo "  Filtering took: $((END - START)) seconds"
echo ""

# =============================================================================
# Phase 3: Pre-split cluster membership into per-cluster files
# =============================================================================

echo "--- Phase 3: Pre-split cluster membership ---"
echo ""
echo "  Reading ${CLUSTER_TSV} ($(du -sh "${CLUSTER_TSV}" | cut -f1))..."
echo "  Writing per-cluster member files to ${MEMBER_DIR}/"

mkdir -p "${MEMBER_DIR}"

# Build a hash set of target cluster rep IDs, then split membership in one pass
# This reads the 29GB TSV exactly once, writing only target cluster members
START=$(date +%s)

python3 << 'PYEOF'
import os
import sys
from collections import defaultdict

novel_dir = os.environ.get("NOVEL_DIR", "/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains")
target_list = os.path.join(novel_dir, "hmm_target_clusters.txt")
cluster_tsv = os.path.join(novel_dir, "clusters_30", "clusters_30_cluster.tsv")
member_dir = os.path.join(novel_dir, "cluster_members")

# Load target cluster IDs into a set
print(f"  Loading target cluster IDs from {target_list}...")
targets = set()
with open(target_list) as f:
    for line in f:
        targets.add(line.strip())
print(f"  Loaded {len(targets)} target clusters")

# Single pass through the 29GB cluster TSV
# Accumulate members in memory (dict of lists) to avoid too-many-open-files
# With ~42K clusters and ~2M total members, memory usage is manageable (~200MB)
print(f"  Scanning {cluster_tsv}...")
cluster_members = defaultdict(list)
lines_read = 0
members_found = 0

with open(cluster_tsv) as f:
    for line in f:
        lines_read += 1
        if lines_read % 50_000_000 == 0:
            print(f"    Processed {lines_read:,} lines, found {members_found:,} target members...")
            sys.stdout.flush()

        parts = line.rstrip('\n').split('\t')
        if len(parts) != 2:
            continue
        rep_id, member_id = parts

        if rep_id not in targets:
            continue

        cluster_members[rep_id].append(member_id)
        members_found += 1

print(f"  Total lines read: {lines_read:,}")
print(f"  Total target members: {members_found:,}")
print(f"  Clusters with members: {len(cluster_members)}")
sys.stdout.flush()

# Write per-cluster member files
print(f"  Writing {len(cluster_members)} member files...")
for rep_id, members in cluster_members.items():
    fpath = os.path.join(member_dir, f"{rep_id}.members.txt")
    with open(fpath, 'w') as f:
        f.write('\n'.join(members) + '\n')

print(f"  Done writing member files")
PYEOF

END=$(date +%s)
echo "  Pre-splitting took: $((END - START)) seconds"
echo ""

# Verify
N_MEMBER_FILES=$(find "${MEMBER_DIR}" -name '*.members.txt' -type f | wc -l)
echo "  Member files created: ${N_MEMBER_FILES}"

# =============================================================================
# Phase 4: Extract ALL member sequences in one seqtk subseq pass
# =============================================================================

echo ""
echo "--- Phase 4: Extract all member sequences (single FASTA scan) ---"
echo ""

EXTRACTED_FA="${NOVEL_DIR}/hmm_target_sequences.fa"
ALL_MEMBER_IDS="${TMPDIR}/all_member_ids.txt"

# Collect all unique member IDs across all clusters
echo "  Collecting all unique member IDs..."
START=$(date +%s)
cat "${MEMBER_DIR}"/*.members.txt | sort -u > "${ALL_MEMBER_IDS}"
N_UNIQUE=$(wc -l < "${ALL_MEMBER_IDS}")
END=$(date +%s)
echo "  Unique member IDs: ${N_UNIQUE} (collected in $((END - START))s)"

# Single seqtk subseq pass over the 50GB FASTA
echo "  Extracting sequences via seqtk subseq (single 50GB scan)..."
START=$(date +%s)
seqtk subseq "${INPUT_FA}" "${ALL_MEMBER_IDS}" > "${EXTRACTED_FA}"
END=$(date +%s)

N_EXTRACTED=$(grep -c '^>' "${EXTRACTED_FA}" 2>/dev/null || echo 0)
EXTRACTED_SIZE=$(du -sh "${EXTRACTED_FA}" | cut -f1)
echo "  Extracted: ${N_EXTRACTED} sequences (${EXTRACTED_SIZE})"
echo "  Extraction took: $((END - START)) seconds"
echo ""

# =============================================================================
# Write stats file
# =============================================================================

echo ""
echo "--- Writing stats ---"

N_INDEXED=$(wc -l < "${INPUT_FA}.fai" 2>/dev/null || echo "N/A")

cat > "${STATS_FILE}" << EOF
# Step 5a: Prep HMM Input Statistics
# Generated: $(date -Iseconds)
# Job: ${SLURM_JOB_ID}
FASTA_INDEXED=${N_INDEXED}
MIN_CLUSTER_SIZE=${MIN_CLUSTER_SIZE}
TARGET_CLUSTERS=${N_TARGETS}
MEMBER_FILES=${N_MEMBER_FILES}
UNIQUE_MEMBERS=${N_UNIQUE}
EXTRACTED_SEQUENCES=${N_EXTRACTED}
EXTRACTED_FASTA=${EXTRACTED_FA}
EOF

echo "  Stats written to: ${STATS_FILE}"
echo ""
echo "  Summary:"
echo "    FASTA sequences indexed: ${N_INDEXED}"
echo "    Target clusters (>=${MIN_CLUSTER_SIZE}): ${N_TARGETS}"
echo "    Per-cluster member files: ${N_MEMBER_FILES}"
echo "    Unique members: ${N_UNIQUE}"
echo "    Extracted sequences: ${N_EXTRACTED} (${EXTRACTED_SIZE})"
echo ""
echo "  Next: submit 05b_build_hmms_v2.sbatch with --array=1-$((( N_TARGETS + 99 ) / 100))%50"
echo ""
echo "  Done: $(date)"

rm -rf "${TMPDIR}"
