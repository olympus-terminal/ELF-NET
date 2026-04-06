#!/bin/bash
#SBATCH --job-name=dark_ids
#SBATCH --output=logs/novel_domains/01_dark_ids_%A_%a.out
#SBATCH --error=logs/novel_domains/01_dark_ids_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=compute
#SBATCH --array=1-2044%100

# =============================================================================
# Step 1: Extract Pfam-dark Protein IDs
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_domains/01_extract_dark_ids.sbatch
#   Generated: 2026-02-21
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Purpose:
#   For each sample, identify proteins that have NO Pfam-A hit (E<1e-9).
#   These constitute the "Pfam-dark proteome" — proteins with no recognizable
#   domain architecture in Pfam.
#
# Logic:
#   1. Parse hmmsearch .tbl (tblout format) → set of protein IDs with Pfam hits
#   2. Parse corresponding FASTA → set of all protein IDs
#   3. Set difference → Pfam-dark protein IDs
#   4. Write dark IDs and per-sample statistics
#
# Input:
#   - 03_analyses/hmmsearch_results/{sample}*.tbl (hmmsearch tblout)
#   - 03_analyses/algae_proteins/{sample}.fa (protein FASTA)
#
# Output:
#   - 03_analyses/novel_domains/dark_ids/{sample}.dark_ids.txt
#   - 03_analyses/novel_domains/dark_ids/{sample}.dark_stats.tsv
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   mkdir -p logs/novel_domains
#   sbatch MANUSCRIPT/scripts/novel_domains/01_extract_dark_ids.sbatch
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CRITICAL ENVIRONMENT SETUP (per JUBAIL_BEST_PRACTICES.md)
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
PFAM_DIR="${BASE}/03_analyses/hmmsearch_results"
ALGAL_DIR="${BASE}/03_analyses/algae_proteins"
OUT_DIR="${BASE}/03_analyses/novel_domains/dark_ids"
SAMPLE_LIST="${BASE}/03_analyses/novel_domains/sample_list.txt"

mkdir -p "${OUT_DIR}"

# =============================================================================
# Get sample for this array task
# =============================================================================

if [[ ! -f "${SAMPLE_LIST}" ]]; then
    echo "ERROR: Sample list not found: ${SAMPLE_LIST}"
    echo "  Run 00_audit_hpc.sh first to generate sample_list.txt"
    exit 1
fi

SAMPLE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SAMPLE_LIST}")

if [[ -z "${SAMPLE}" ]]; then
    echo "ERROR: No sample at line ${SLURM_ARRAY_TASK_ID} in ${SAMPLE_LIST}"
    exit 1
fi

echo "========================================"
echo "Step 1: Extract Pfam-dark Protein IDs"
echo "========================================"
echo ""
echo "Job ID:     ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "Date:       $(date)"
echo "Sample:     ${SAMPLE}"
echo ""

# =============================================================================
# Locate input files
# =============================================================================

# Find the FASTA file
# Actual naming on Jubail: {stem}.algae.fa (e.g., Alexandrium_andersonii.AAC.aa.algae.fa)
FASTA=""
for ext in ".algae.fa" ".fa" ".faa" ".fasta" ".protein.fa"; do
    if [[ -f "${ALGAL_DIR}/${SAMPLE}${ext}" ]]; then
        FASTA="${ALGAL_DIR}/${SAMPLE}${ext}"
        break
    fi
done

if [[ -z "${FASTA}" || ! -f "${FASTA}" ]]; then
    echo "ERROR: FASTA not found for sample ${SAMPLE} in ${ALGAL_DIR}"
    exit 1
fi

# Find the hmmsearch result file
# Naming in hmmsearch_results/: {stem}.hmmsearch.tbl  (stem already includes .aa)
# e.g. Alexandrium_andersonii.AAC.aa.hmmsearch.tbl
TBL=""
for pattern in \
    "${PFAM_DIR}/${SAMPLE}.hmmsearch.tbl" \
    "${PFAM_DIR}/${SAMPLE}.tbl" \
    "${PFAM_DIR}/${SAMPLE}*.hmmsearch.tbl" \
    "${PFAM_DIR}/${SAMPLE}*.tbl"; do
    MATCH=$(ls ${pattern} 2>/dev/null | head -1 || true)
    if [[ -n "${MATCH}" && -f "${MATCH}" ]]; then
        TBL="${MATCH}"
        break
    fi
done

echo "  FASTA: ${FASTA}"
echo "  TBL:   ${TBL:-NOT FOUND}"
echo ""

# =============================================================================
# Extract protein IDs
# =============================================================================

# All protein IDs from FASTA
echo "  Extracting all protein IDs from FASTA..."
ALL_IDS=$(mktemp "${TMPDIR}/all_ids_XXXX")
grep "^>" "${FASTA}" | sed 's/^>//' | awk '{print $1}' | sort -u > "${ALL_IDS}"
TOTAL=$(wc -l < "${ALL_IDS}")
echo "    Total proteins: ${TOTAL}"

if [[ -z "${TBL}" || ! -f "${TBL}" ]]; then
    # No .tbl file → ALL proteins are dark (no Pfam annotation available)
    echo "  WARNING: No hmmsearch .tbl found for ${SAMPLE}"
    echo "    Treating all ${TOTAL} proteins as Pfam-dark (no annotation data)"
    cp "${ALL_IDS}" "${OUT_DIR}/${SAMPLE}.dark_ids.txt"

    # Write stats
    printf "sample\ttotal_seqs\tpfam_hit_seqs\tdark_seqs\tpct_dark\tnote\n" \
        > "${OUT_DIR}/${SAMPLE}.dark_stats.tsv"
    printf "%s\t%d\t%d\t%d\t%.2f\tno_tbl_file\n" \
        "${SAMPLE}" "${TOTAL}" 0 "${TOTAL}" 100.00 \
        >> "${OUT_DIR}/${SAMPLE}.dark_stats.tsv"

    rm -f "${ALL_IDS}"
    echo "  Done (no .tbl file — all proteins marked dark)."
    exit 0
fi

# Pfam-hit protein IDs from .tbl
# tblout format: lines starting with # are comments
# Fields: target_name, target_accession, query_name, ...
# The target_name (column 1) is the protein ID
echo "  Extracting Pfam-hit protein IDs from .tbl..."
HIT_IDS=$(mktemp "${TMPDIR}/hit_ids_XXXX")
{ grep -v "^#" "${TBL}" || true; } | awk '{print $1}' | sort -u > "${HIT_IDS}"
NHITS=$(wc -l < "${HIT_IDS}")
echo "    Pfam-hit proteins: ${NHITS}"

# Set difference: all IDs minus hit IDs = dark IDs
echo "  Computing set difference..."
DARK_IDS="${OUT_DIR}/${SAMPLE}.dark_ids.txt"
comm -23 "${ALL_IDS}" "${HIT_IDS}" > "${DARK_IDS}"
NDARK=$(wc -l < "${DARK_IDS}")

# Sanity check: hits + dark should equal total
SUM=$((NHITS + NDARK))
if [[ ${SUM} -ne ${TOTAL} ]]; then
    # Some hit IDs may not be in FASTA (shouldn't happen but handle gracefully)
    echo "  WARNING: hits(${NHITS}) + dark(${NDARK}) = ${SUM} != total(${TOTAL})"
    echo "    This may indicate IDs in .tbl not present in FASTA"
    # Count hits that are in FASTA
    VALID_HITS=$(comm -12 "${ALL_IDS}" "${HIT_IDS}" | wc -l)
    echo "    Valid hits (in FASTA): ${VALID_HITS}"
    NDARK=$((TOTAL - VALID_HITS))
fi

PCT_DARK=$(awk "BEGIN {printf \"%.2f\", 100.0 * ${NDARK} / ${TOTAL}}")

echo ""
echo "  Results:"
echo "    Total proteins: ${TOTAL}"
echo "    Pfam-annotated: ${NHITS}"
echo "    Pfam-dark:      ${NDARK} (${PCT_DARK}%)"
echo ""

# Write per-sample stats
printf "sample\ttotal_seqs\tpfam_hit_seqs\tdark_seqs\tpct_dark\tnote\n" \
    > "${OUT_DIR}/${SAMPLE}.dark_stats.tsv"
printf "%s\t%d\t%d\t%d\t%s\tok\n" \
    "${SAMPLE}" "${TOTAL}" "${NHITS}" "${NDARK}" "${PCT_DARK}" \
    >> "${OUT_DIR}/${SAMPLE}.dark_stats.tsv"

# Cleanup
rm -f "${ALL_IDS}" "${HIT_IDS}"

echo "  Output: ${DARK_IDS}"
echo "  Stats:  ${OUT_DIR}/${SAMPLE}.dark_stats.tsv"
echo ""
echo "  Done: $(date)"
