#!/bin/bash
#SBATCH --job-name=algagpt_hmm
#SBATCH --output=/scratch/drn2/PROJECTS/algaGPT-TARA-archive/logs/extract_hmm_%A_%a.out
#SBATCH --error=/scratch/drn2/PROJECTS/algaGPT-TARA-archive/logs/extract_hmm_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=1-49%10

# =============================================================================
# Extract algaGPT-classified algae sequences and run hmmsearch
# Each array task processes 42 files
#
# Provenance:
#   Script: extract_and_hmmsearch_algagpt_20260114_174000.sbatch
#   Date: 2026-01-14
# =============================================================================

set -e

# Load required modules
module load hmmer/3.3

# Directories
ARCHIVE_DIR="/scratch/drn2/PROJECTS/algaGPT-TARA-archive"
PROTEINS_DIR="/scratch/drn2/PROJECTS/TARA-LA4SR/02_processed_data/proteins"
ALGAGPT_DIR="${ARCHIVE_DIR}/algagpt_results_fixed"
ALGAE_DIR="${ARCHIVE_DIR}/algae_proteins"
HMMSEARCH_DIR="${ARCHIVE_DIR}/hmmsearch_results"
PFAM_DB="/scratch/drn2/fastq/data/raw/db/Pfam-A.hmm"

# Create output directories
mkdir -p "${ALGAE_DIR}"
mkdir -p "${HMMSEARCH_DIR}"
mkdir -p "${ARCHIVE_DIR}/logs"

# Get file list
FILELIST="${ARCHIVE_DIR}/algagpt_filelist.txt"
if [[ ! -f "${FILELIST}" ]]; then
    ls "${ALGAGPT_DIR}"/*.tsv | sort > "${FILELIST}"
fi

# Calculate file range for this task (42 files per task)
CHUNK_SIZE=42
START_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK_SIZE + 1 ))
END_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))

# Get files for this chunk
mapfile -t FILES_TO_PROCESS < <(sed -n "${START_IDX},${END_IDX}p" "${FILELIST}")

if [[ ${#FILES_TO_PROCESS[@]} -eq 0 ]]; then
    echo "No files found for task ${SLURM_ARRAY_TASK_ID} (lines ${START_IDX}-${END_IDX})"
    exit 0
fi

echo "=============================================="
echo "Task ${SLURM_ARRAY_TASK_ID}: Processing ${#FILES_TO_PROCESS[@]} files (lines ${START_IDX}-${END_IDX})"
echo "=============================================="

# Process each file in this chunk
FILE_COUNT=0
for ALGAGPT_FILE in "${FILES_TO_PROCESS[@]}"; do
    FILE_COUNT=$((FILE_COUNT + 1))

    # Extract base name
    BASENAME=$(basename "${ALGAGPT_FILE}" _algagpt.tsv)
    SEARCH_BASE="${BASENAME%.aa}"

    echo ""
    echo "[${FILE_COUNT}/${#FILES_TO_PROCESS[@]}] Processing: ${BASENAME}"

    # Find corresponding protein file
    PROTEIN_FILE=""
    for pattern in \
        "${PROTEINS_DIR}/${SEARCH_BASE}.fa.aa.fa" \
        "${PROTEINS_DIR}/${SEARCH_BASE}.aa.fa" \
        "${PROTEINS_DIR}/${BASENAME}.fa"; do
        if [[ -f "${pattern}" ]]; then
            PROTEIN_FILE="${pattern}"
            break
        fi
    done

    # If not found, try glob match
    if [[ -z "${PROTEIN_FILE}" ]]; then
        PROTEIN_FILE=$(ls "${PROTEINS_DIR}/${SEARCH_BASE}"*.fa 2>/dev/null | head -1)
    fi

    if [[ -z "${PROTEIN_FILE}" || ! -f "${PROTEIN_FILE}" ]]; then
        echo "  WARNING: Cannot find protein file for ${BASENAME}, skipping"
        continue
    fi

    # Output files
    ALGAE_FASTA="${ALGAE_DIR}/${BASENAME}.algae.fa"
    HMMSEARCH_OUT="${HMMSEARCH_DIR}/${BASENAME}.hmmsearch.tbl"
    HMMSEARCH_DOM="${HMMSEARCH_DIR}/${BASENAME}.hmmsearch.domtbl"

    # Skip if already done
    if [[ -f "${HMMSEARCH_OUT}" && -s "${HMMSEARCH_OUT}" ]]; then
        echo "  Already completed, skipping."
        continue
    fi

    # Step 1: Extract algae sequences
    echo "  Extracting algae sequences..."
    ALGAE_IDS="${ALGAE_DIR}/${BASENAME}.algae_ids.txt"
    grep "|label|>algae" "${ALGAGPT_FILE}" | cut -f1 > "${ALGAE_IDS}"

    N_ALGAE=$(wc -l < "${ALGAE_IDS}")
    N_TOTAL=$(wc -l < "${ALGAGPT_FILE}")
    PCT=$(echo "scale=1; ${N_ALGAE} * 100 / ${N_TOTAL}" | bc)
    echo "  Sequences: ${N_ALGAE}/${N_TOTAL} (${PCT}% algae)"

    # Extract sequences using seqtk if available, otherwise use awk
    if command -v seqtk &> /dev/null; then
        seqtk subseq "${PROTEIN_FILE}" "${ALGAE_IDS}" > "${ALGAE_FASTA}"
    else
        awk 'NR==FNR {ids[$1]; next}
             /^>/ {p=0; id=substr($1,2); if(id in ids) p=1}
             p' "${ALGAE_IDS}" "${PROTEIN_FILE}" > "${ALGAE_FASTA}"
    fi

    rm -f "${ALGAE_IDS}"

    if [[ ! -s "${ALGAE_FASTA}" ]]; then
        echo "  WARNING: Extraction failed - empty output, skipping"
        continue
    fi

    N_SEQS=$(grep -c "^>" "${ALGAE_FASTA}" || echo 0)
    echo "  Extracted ${N_SEQS} sequences"

    # Step 2: Run hmmsearch
    echo "  Running hmmsearch..."
    if [[ ! -f "${PFAM_DB}" ]]; then
        echo "  ERROR: Pfam database not found at ${PFAM_DB}"
        exit 1
    fi

    hmmsearch \
        --cpu ${SLURM_CPUS_PER_TASK} \
        --tblout "${HMMSEARCH_OUT}" \
        --domtblout "${HMMSEARCH_DOM}" \
        --noali \
        -E 1e-5 \
        "${PFAM_DB}" \
        "${ALGAE_FASTA}" \
        > /dev/null

    N_HITS=$(grep -cv "^#" "${HMMSEARCH_OUT}" || echo 0)
    echo "  Done: ${N_HITS} Pfam hits"

done

echo ""
echo "=============================================="
echo "Task ${SLURM_ARRAY_TASK_ID} complete: processed ${FILE_COUNT} files"
echo "=============================================="
