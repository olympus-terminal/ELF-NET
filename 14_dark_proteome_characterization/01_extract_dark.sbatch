#!/bin/bash
#SBATCH --job-name=dark_extract
#SBATCH --output=logs/dark_proteome/extract_%A_%a.out
#SBATCH --error=logs/dark_proteome/extract_%A_%a.err
#SBATCH --array=1-PLACEHOLDER%100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=compute

# =============================================================================
# Extract Dark Proteome — SLURM Array Job (combined ID + FASTA extraction)
# =============================================================================
#
# Provenance:
#   Script: scripts/dark_proteome/01_extract_dark.sbatch
#   Generated: 2026-02-21
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Purpose:
#   For each sample (one per array task):
#     1. Extract all protein IDs from FASTA
#     2. Extract Pfam-hit IDs from hmmsearch .tbl
#     3. Set difference → dark protein IDs
#     4. Extract dark sequences with seqtk (Python fallback)
#     5. Write per-sample outputs:
#        - dark_ids/{sample}.dark_ids.txt
#        - dark_fasta/{sample}.dark.fa
#        - dark_ids/{sample}.dark_stats.tsv
#
# Prerequisites:
#   - Run 00_audit.sh first to generate sample_list.txt
#   - Replace PLACEHOLDER in --array with actual sample count
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   # After audit: sbatch MANUSCRIPT/scripts/dark_proteome/01_extract_dark.sbatch
#   # Or use submit.sh
#
# =============================================================================

set -euo pipefail

# ---- HPC environment ----
export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
export PYTHONUSERBASE=/scratch/drn2/newhome/.local
mkdir -p $TMPDIR

unset NETWORKX_BACKEND_CONFIG
unset NX_BACKEND_CONFIG

module purge
module load gcc/13.2.0

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

# ---- Paths ----
BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
FASTA_DIR="${BASE}/03_analyses/algae_proteins"
HMMSEARCH_DIR="${BASE}/03_analyses/hmmsearch_results"
OUT_DIR="${BASE}/03_analyses/dark_proteome"
SAMPLE_LIST="${OUT_DIR}/sample_list.txt"

DARK_IDS_DIR="${OUT_DIR}/dark_ids"
DARK_FASTA_DIR="${OUT_DIR}/dark_fasta"

mkdir -p "${DARK_IDS_DIR}" "${DARK_FASTA_DIR}"

# ---- Resolve sample from array index ----
SAMPLE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SAMPLE_LIST}")

if [[ -z "${SAMPLE}" ]]; then
    echo "ERROR: No sample at line ${SLURM_ARRAY_TASK_ID} in ${SAMPLE_LIST}"
    exit 1
fi

echo "========================================"
echo "Dark Proteome Extraction"
echo "========================================"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Array ID:  ${SLURM_ARRAY_TASK_ID:-0}"
echo "Sample:    ${SAMPLE}"
echo "Date:      $(date)"
echo "Host:      $(hostname)"
echo ""

# ---- Locate input files ----
# Naming conventions (confirmed from HPC):
#   FASTA: {stem}.aa.algae.fa
#   TBL:   {stem}.aa.hmmsearch.tbl
FASTA="${FASTA_DIR}/${SAMPLE}.aa.algae.fa"
TBL="${HMMSEARCH_DIR}/${SAMPLE}.aa.hmmsearch.tbl"

if [[ ! -f "${FASTA}" ]]; then
    echo "ERROR: FASTA not found: ${FASTA}"
    exit 1
fi

if [[ ! -f "${TBL}" ]]; then
    echo "WARNING: .tbl not found: ${TBL}"
    echo "         All sequences for this sample will be classified as dark."
    TBL=""
fi

echo "FASTA: ${FASTA}"
echo "TBL:   ${TBL:-NONE (all sequences will be dark)}"
echo ""

# ---- Step 1: Extract all protein IDs from FASTA ----
echo "Step 1: Extracting all protein IDs from FASTA..."
ALL_IDS_FILE="${TMPDIR}/${SAMPLE}.all_ids.txt"
grep '^>' "${FASTA}" | awk '{print substr($1, 2)}' | sort -u > "${ALL_IDS_FILE}"
N_ALL=$(wc -l < "${ALL_IDS_FILE}")
echo "  Total proteins: ${N_ALL}"

# ---- Step 2: Extract Pfam-hit IDs from .tbl ----
echo "Step 2: Extracting Pfam-hit IDs from hmmsearch output..."
PFAM_IDS_FILE="${TMPDIR}/${SAMPLE}.pfam_ids.txt"

if [[ -n "${TBL}" && -f "${TBL}" ]]; then
    grep -v '^#' "${TBL}" | awk '{print $1}' | sort -u > "${PFAM_IDS_FILE}"
    N_PFAM=$(wc -l < "${PFAM_IDS_FILE}")
else
    # No hmmsearch results → everything is dark
    : > "${PFAM_IDS_FILE}"
    N_PFAM=0
fi
echo "  Pfam-hit proteins: ${N_PFAM}"

# ---- Step 3: Set difference → dark protein IDs ----
echo "Step 3: Computing set difference (all - pfam_hits = dark)..."
DARK_IDS_FILE="${DARK_IDS_DIR}/${SAMPLE}.dark_ids.txt"

if [[ ${N_PFAM} -gt 0 ]]; then
    comm -23 "${ALL_IDS_FILE}" "${PFAM_IDS_FILE}" > "${DARK_IDS_FILE}"
else
    cp "${ALL_IDS_FILE}" "${DARK_IDS_FILE}"
fi

N_DARK=$(wc -l < "${DARK_IDS_FILE}")
echo "  Dark proteins: ${N_DARK}"

# ---- Step 4: Extract dark sequences ----
echo "Step 4: Extracting dark sequences..."
DARK_FA="${DARK_FASTA_DIR}/${SAMPLE}.dark.fa"

if [[ ${N_DARK} -eq 0 ]]; then
    : > "${DARK_FA}"
    echo "  No dark sequences to extract"
elif command -v seqtk &>/dev/null; then
    seqtk subseq "${FASTA}" "${DARK_IDS_FILE}" > "${DARK_FA}"
    echo "  Extracted with seqtk"
else
    # Python fallback
    echo "  seqtk not found, using Python fallback..."
    python3 -c "
import sys
ids = set()
with open(sys.argv[2]) as f:
    for line in f:
        ids.add(line.strip())
keep = False
with open(sys.argv[1]) as fin, open(sys.argv[3], 'w') as fout:
    for line in fin:
        if line.startswith('>'):
            name = line[1:].split()[0]
            keep = name in ids
        if keep:
            fout.write(line)
" "${FASTA}" "${DARK_IDS_FILE}" "${DARK_FA}"
    echo "  Extracted with Python fallback"
fi

# Verify extraction
N_EXTRACTED=$(grep -c '^>' "${DARK_FA}" || echo 0)
echo "  Sequences in dark FASTA: ${N_EXTRACTED}"

# ---- Step 5: Write per-sample stats ----
echo "Step 5: Writing stats..."
STATS_FILE="${DARK_IDS_DIR}/${SAMPLE}.dark_stats.tsv"

if [[ ${N_ALL} -gt 0 ]]; then
    PCT_DARK=$(awk "BEGIN {printf \"%.2f\", 100 * ${N_DARK} / ${N_ALL}}")
else
    PCT_DARK="0.00"
fi

# Header + data in one file (makes concatenation easy with tail -n+2)
cat > "${STATS_FILE}" <<EOF
sample	total_proteins	pfam_hit_proteins	dark_proteins	pct_dark	dark_extracted
${SAMPLE}	${N_ALL}	${N_PFAM}	${N_DARK}	${PCT_DARK}	${N_EXTRACTED}
EOF

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "  Sample:          ${SAMPLE}"
echo "  Total proteins:  ${N_ALL}"
echo "  Pfam hits:       ${N_PFAM}"
echo "  Dark proteins:   ${N_DARK} (${PCT_DARK}%)"
echo "  Dark extracted:  ${N_EXTRACTED}"
echo ""

# Sanity check: hits + dark should equal total
SANITY=$((N_PFAM + N_DARK))
if [[ ${SANITY} -ne ${N_ALL} ]]; then
    echo "  WARNING: pfam_hits (${N_PFAM}) + dark (${N_DARK}) = ${SANITY} != total (${N_ALL})"
    echo "  (This can happen if some IDs appear in .tbl but not in FASTA, or vice versa)"
else
    echo "  SANITY CHECK PASSED: ${N_PFAM} + ${N_DARK} = ${N_ALL}"
fi

echo ""
echo "Output files:"
echo "  ${DARK_IDS_FILE}"
echo "  ${DARK_FA}"
echo "  ${STATS_FILE}"
echo ""

# Cleanup temp files
rm -f "${ALL_IDS_FILE}" "${PFAM_IDS_FILE}"

echo "Done: $(date)"
