#!/bin/bash
#SBATCH --job-name=dark_fasta
#SBATCH --output=logs/novel_domains/02_dark_fasta_%A_%a.out
#SBATCH --error=logs/novel_domains/02_dark_fasta_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=compute
#SBATCH --array=1-21

# =============================================================================
# Step 2: Extract Dark Proteome FASTA Sequences (chunked)
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_domains/02_extract_dark_fasta.sbatch
#   Generated: 2026-02-28
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Purpose:
#   Extract protein sequences for Pfam-dark IDs identified in Step 1.
#   Uses seqtk subseq for efficient random-access FASTA extraction.
#   Each array task processes ~100 samples to stay within SLURM job limits.
#
# Input:
#   - 03_analyses/novel_domains/dark_ids/{sample}.dark_ids.txt (from Step 1)
#   - 03_analyses/algae_proteins/{sample}*.fa (protein FASTA)
#
# Output:
#   - 03_analyses/novel_domains/dark_fasta/{sample}.dark.fa
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   mkdir -p logs/novel_domains
#   sbatch MANUSCRIPT/scripts/novel_domains/02_extract_dark_fasta.sbatch
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CRITICAL ENVIRONMENT SETUP
# =============================================================================

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
mkdir -p $TMPDIR
export PYTHONUSERBASE=/scratch/drn2/newhome/.local
unset NETWORKX_BACKEND_CONFIG 2>/dev/null || true
unset NX_BACKEND_CONFIG 2>/dev/null || true

module purge
module load gcc/13.2.0

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

# =============================================================================
# Configuration
# =============================================================================

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
ALGAL_DIR="${BASE}/03_analyses/algae_proteins"
DARK_IDS_DIR="${BASE}/03_analyses/novel_domains/dark_ids"
OUT_DIR="${BASE}/03_analyses/novel_domains/dark_fasta"
SAMPLE_LIST="${BASE}/03_analyses/novel_domains/sample_list.txt"

CHUNK_SIZE=100  # samples per array task

mkdir -p "${OUT_DIR}"

# =============================================================================
# Compute sample range for this chunk
# =============================================================================

N_TOTAL=$(wc -l < "${SAMPLE_LIST}")
START_LINE=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK_SIZE + 1 ))
END_LINE=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
if [[ ${END_LINE} -gt ${N_TOTAL} ]]; then
    END_LINE=${N_TOTAL}
fi

if [[ ${START_LINE} -gt ${N_TOTAL} ]]; then
    echo "Chunk ${SLURM_ARRAY_TASK_ID}: no samples (start=${START_LINE} > total=${N_TOTAL})"
    exit 0
fi

echo "========================================"
echo "Step 2: Extract Dark Proteome FASTA"
echo "========================================"
echo ""
echo "Job ID:       ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:         ${SLURMD_NODENAME}"
echo "Date:         $(date)"
echo "Total samples: ${N_TOTAL}"
echo "This chunk:    lines ${START_LINE}-${END_LINE} ($(( END_LINE - START_LINE + 1 )) samples)"
echo ""

# =============================================================================
# Process each sample in this chunk
# =============================================================================

PROCESSED=0
SKIPPED=0
ERRORS=0

for LINE_NUM in $(seq ${START_LINE} ${END_LINE}); do
    SAMPLE=$(sed -n "${LINE_NUM}p" "${SAMPLE_LIST}")

    if [[ -z "${SAMPLE}" ]]; then
        continue
    fi

    DARK_IDS="${DARK_IDS_DIR}/${SAMPLE}.dark_ids.txt"
    OUTPUT="${OUT_DIR}/${SAMPLE}.dark.fa"

    # Skip if already done
    if [[ -f "${OUTPUT}" && -s "${OUTPUT}" ]]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Find FASTA
    FASTA=""
    for ext in ".algae.fa" ".fa" ".faa" ".fasta"; do
        if [[ -f "${ALGAL_DIR}/${SAMPLE}${ext}" ]]; then
            FASTA="${ALGAL_DIR}/${SAMPLE}${ext}"
            break
        fi
    done

    if [[ ! -f "${DARK_IDS}" ]]; then
        echo "  SKIP ${SAMPLE}: no dark_ids file"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if [[ -z "${FASTA}" ]]; then
        echo "  ERROR ${SAMPLE}: FASTA not found"
        ERRORS=$((ERRORS + 1))
        continue
    fi

    N_DARK=$(wc -l < "${DARK_IDS}")
    if [[ ${N_DARK} -eq 0 ]]; then
        touch "${OUTPUT}"
        PROCESSED=$((PROCESSED + 1))
        continue
    fi

    # Extract with seqtk or Python fallback
    if command -v seqtk &>/dev/null; then
        seqtk subseq "${FASTA}" "${DARK_IDS}" > "${OUTPUT}" 2>/dev/null
    else
        python3 -c "
ids = set(open('${DARK_IDS}').read().split())
keep = False
with open('${FASTA}') as fin, open('${OUTPUT}', 'w') as fout:
    for line in fin:
        if line.startswith('>'):
            keep = line[1:].split()[0] in ids
        if keep:
            fout.write(line)
"
    fi

    N_EXTRACTED=$(grep -c "^>" "${OUTPUT}" 2>/dev/null || echo 0)
    PROCESSED=$((PROCESSED + 1))

    # Progress every 20 samples
    if [[ $((PROCESSED % 20)) -eq 0 ]]; then
        echo "  Progress: ${PROCESSED} processed, ${SKIPPED} skipped (at line ${LINE_NUM})"
    fi
done

echo ""
echo "========================================"
echo "  Chunk ${SLURM_ARRAY_TASK_ID} Summary"
echo "========================================"
echo "  Processed: ${PROCESSED}"
echo "  Skipped:   ${SKIPPED}"
echo "  Errors:    ${ERRORS}"
echo "  Done:      $(date)"
