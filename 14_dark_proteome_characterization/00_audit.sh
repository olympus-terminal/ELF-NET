#!/bin/bash
# =============================================================================
# Step 0: Audit — Discover file naming patterns and cross-reference samples
# =============================================================================
#
# Provenance:
#   Script: scripts/dark_proteome/00_audit.sh
#   Generated: 2026-02-21
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Purpose:
#   Run interactively on login node to:
#   - Discover file naming patterns in algae_proteins/ and hmmsearch_results/
#   - Cross-reference which samples have both FASTA and .tbl
#   - Show format samples (first few lines of .tbl, FASTA headers)
#   - Generate sample_list.txt from the intersection
#   - Report disk space
#
# Usage:
#   ssh drn2@jubail.abudhabi.nyu.edu
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   bash MANUSCRIPT/scripts/dark_proteome/00_audit.sh 2>&1 | tee audit_dark_proteome_$(date +%Y%m%d_%H%M%S).log
#
# =============================================================================

set -euo pipefail

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
FASTA_DIR="${BASE}/03_analyses/algae_proteins"
HMMSEARCH_DIR="${BASE}/03_analyses/hmmsearch_results"
OUT_DIR="${BASE}/03_analyses/dark_proteome"

echo "=============================================="
echo "  Dark Proteome Extraction — HPC Audit"
echo "=============================================="
echo ""
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "User:     $(whoami)"
echo "Base dir: ${BASE}"
echo ""

# =============================================================================
# 1. Algal protein FASTA files
# =============================================================================
echo "--- 1. Algal protein FASTA files ---"
if [[ -d "${FASTA_DIR}" ]]; then
    echo "  Directory: ${FASTA_DIR}"
    echo ""

    # Count by extension
    for ext in fa fasta faa fa.gz; do
        COUNT=$(find "${FASTA_DIR}" -maxdepth 1 -name "*.${ext}" 2>/dev/null | wc -l)
        if [[ ${COUNT} -gt 0 ]]; then
            echo "  *.${ext} files: ${COUNT}"
        fi
    done

    # Show first 5 files
    echo ""
    echo "  First 5 files:"
    ls "${FASTA_DIR}"/ 2>/dev/null | head -5 | sed 's/^/    /'

    # Show naming pattern
    echo ""
    echo "  Naming pattern (unique extensions):"
    ls "${FASTA_DIR}"/ 2>/dev/null | sed 's/^[^.]*\.//' | sort | uniq -c | sort -rn | head -5 | sed 's/^/    /'

    # Show FASTA header format from first file
    echo ""
    echo "  FASTA header format (first file, first 3 headers):"
    FIRST_FA=$(ls "${FASTA_DIR}"/*.aa.algae.fa 2>/dev/null | head -1)
    if [[ -n "${FIRST_FA}" ]]; then
        grep '^>' "${FIRST_FA}" | head -3 | sed 's/^/    /'
    fi

    # Total size
    echo ""
    echo "  Total size:"
    du -sh "${FASTA_DIR}" | awk '{print "    "$1}'
else
    echo "  WARNING: Directory not found: ${FASTA_DIR}"
fi
echo ""

# =============================================================================
# 2. hmmsearch result files
# =============================================================================
echo "--- 2. hmmsearch result files ---"
if [[ -d "${HMMSEARCH_DIR}" ]]; then
    echo "  Directory: ${HMMSEARCH_DIR}"
    echo ""

    # Count by extension pattern
    for ext in tbl hmmsearch.tbl domtblout tblout; do
        COUNT=$(find "${HMMSEARCH_DIR}" -maxdepth 1 -name "*.${ext}" 2>/dev/null | wc -l)
        if [[ ${COUNT} -gt 0 ]]; then
            echo "  *.${ext} files: ${COUNT}"
        fi
    done

    # Also check recursively
    DEEP_COUNT=$(find "${HMMSEARCH_DIR}" -name "*.tbl" -o -name "*.tblout" 2>/dev/null | wc -l)
    echo "  Total .tbl/.tblout (recursive): ${DEEP_COUNT}"

    # Show first 5 files
    echo ""
    echo "  First 5 files:"
    ls "${HMMSEARCH_DIR}"/ 2>/dev/null | head -5 | sed 's/^/    /'

    # Show naming pattern
    echo ""
    echo "  Naming pattern (unique extensions):"
    ls "${HMMSEARCH_DIR}"/ 2>/dev/null | sed 's/^[^.]*\.//' | sort | uniq -c | sort -rn | head -5 | sed 's/^/    /'

    # Show .tbl format from first file
    echo ""
    echo "  .tbl format (first file, first 5 lines):"
    FIRST_TBL=$(find "${HMMSEARCH_DIR}" -maxdepth 1 \( -name "*.tbl" -o -name "*.tblout" \) 2>/dev/null | head -1)
    if [[ -n "${FIRST_TBL}" ]]; then
        head -5 "${FIRST_TBL}" | sed 's/^/    /'
    fi

    # Total size
    echo ""
    echo "  Total size:"
    du -sh "${HMMSEARCH_DIR}" | awk '{print "    "$1}'
else
    echo "  WARNING: Directory not found: ${HMMSEARCH_DIR}"
    echo ""
    echo "  Checking alternative locations..."
    for subdir in pfam_results pfam_hmmsearch hmmsearch pfam; do
        ALT_DIR="${BASE}/03_analyses/${subdir}"
        if [[ -d "${ALT_DIR}" ]]; then
            ALT_COUNT=$(find "${ALT_DIR}" -maxdepth 1 -name "*.tbl" -o -name "*.tblout" 2>/dev/null | wc -l)
            echo "    ${ALT_DIR}: ${ALT_COUNT} .tbl files"
        fi
    done
fi
echo ""

# =============================================================================
# 3. Cross-reference: which samples have both FASTA and .tbl
# =============================================================================
echo "--- 3. Cross-reference: FASTA ↔ hmmsearch ---"

# Extract sample stems from FASTA files
# Naming: {stem}.aa.algae.fa  →  stem = e.g. Alexandrium_andersonii.AAC
FASTA_SAMPLES_FILE=$(mktemp)
if [[ -d "${FASTA_DIR}" ]]; then
    ls "${FASTA_DIR}"/*.aa.algae.fa 2>/dev/null | xargs -n1 basename | sed 's/\.aa\.algae\.fa$//' | sort -u > "${FASTA_SAMPLES_FILE}"
    N_FASTA=$(wc -l < "${FASTA_SAMPLES_FILE}")
    echo "  FASTA samples (*.aa.algae.fa): ${N_FASTA}"
else
    touch "${FASTA_SAMPLES_FILE}"
    N_FASTA=0
    echo "  FASTA samples: 0 (directory missing)"
fi

# Extract sample stems from hmmsearch files
# Naming: {stem}.aa.hmmsearch.tbl  →  same stem
TBL_SAMPLES_FILE=$(mktemp)
if [[ -d "${HMMSEARCH_DIR}" ]]; then
    ls "${HMMSEARCH_DIR}"/*.aa.hmmsearch.tbl 2>/dev/null | xargs -n1 basename | sed 's/\.aa\.hmmsearch\.tbl$//' | sort -u > "${TBL_SAMPLES_FILE}"
    N_TBL=$(wc -l < "${TBL_SAMPLES_FILE}")
    echo "  hmmsearch samples (*.aa.hmmsearch.tbl): ${N_TBL}"
else
    touch "${TBL_SAMPLES_FILE}"
    N_TBL=0
    echo "  hmmsearch samples: 0 (directory missing)"
fi

# Compute intersection and differences
if [[ ${N_FASTA} -gt 0 && ${N_TBL} -gt 0 ]]; then
    N_BOTH=$(comm -12 "${FASTA_SAMPLES_FILE}" "${TBL_SAMPLES_FILE}" | wc -l)
    N_FASTA_ONLY=$(comm -23 "${FASTA_SAMPLES_FILE}" "${TBL_SAMPLES_FILE}" | wc -l)
    N_TBL_ONLY=$(comm -13 "${FASTA_SAMPLES_FILE}" "${TBL_SAMPLES_FILE}" | wc -l)
    echo ""
    echo "  Samples with BOTH FASTA + hmmsearch: ${N_BOTH}"
    echo "  Samples with FASTA only (no hmmsearch): ${N_FASTA_ONLY}"
    echo "  Samples with hmmsearch only (no FASTA): ${N_TBL_ONLY}"

    if [[ ${N_FASTA_ONLY} -gt 0 ]]; then
        echo ""
        echo "  First 10 FASTA-only samples (will be skipped):"
        comm -23 "${FASTA_SAMPLES_FILE}" "${TBL_SAMPLES_FILE}" | head -10 | sed 's/^/    /'
    fi
else
    N_BOTH=0
    echo ""
    echo "  Cannot cross-reference (missing FASTA or hmmsearch files)"
fi
echo ""

# =============================================================================
# 4. Generate sample_list.txt (intersection only)
# =============================================================================
echo "--- 4. Generate sample_list.txt ---"

mkdir -p "${OUT_DIR}"
SAMPLE_LIST="${OUT_DIR}/sample_list.txt"

if [[ ${N_BOTH:-0} -gt 0 ]]; then
    comm -12 "${FASTA_SAMPLES_FILE}" "${TBL_SAMPLES_FILE}" > "${SAMPLE_LIST}"
    echo "  Written: ${SAMPLE_LIST}"
    echo "  Total samples: $(wc -l < "${SAMPLE_LIST}")"
    echo ""
    echo "  First 5 samples:"
    head -5 "${SAMPLE_LIST}" | sed 's/^/    /'
    echo ""
    echo "  Last 5 samples:"
    tail -5 "${SAMPLE_LIST}" | sed 's/^/    /'
elif [[ ${N_FASTA} -gt 0 && ${N_TBL} -eq 0 ]]; then
    echo "  WARNING: No hmmsearch results found. Using all FASTA samples."
    echo "  (Samples without hmmsearch hits → entire proteome is 'dark')"
    cp "${FASTA_SAMPLES_FILE}" "${SAMPLE_LIST}"
    echo "  Written: ${SAMPLE_LIST}"
    echo "  Total samples: $(wc -l < "${SAMPLE_LIST}")"
else
    echo "  ERROR: Cannot generate sample list (no matching files found)"
fi
echo ""

# =============================================================================
# 5. Detect .tbl naming convention for use in extraction script
# =============================================================================
echo "--- 5. hmmsearch .tbl naming convention ---"
if [[ -d "${HMMSEARCH_DIR}" ]]; then
    echo "  Discovering .tbl suffix pattern..."
    FIRST_TBL_NAME=$(find "${HMMSEARCH_DIR}" -maxdepth 1 \( -name "*.tbl" -o -name "*.tblout" \) 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "")
    if [[ -n "${FIRST_TBL_NAME}" ]]; then
        # Extract the suffix after the sample name
        FIRST_SAMPLE=$(head -1 "${SAMPLE_LIST}" 2>/dev/null || echo "")
        if [[ -n "${FIRST_SAMPLE}" ]]; then
            SUFFIX="${FIRST_TBL_NAME#${FIRST_SAMPLE}}"
            echo "  Example file: ${FIRST_TBL_NAME}"
            echo "  Sample name:  ${FIRST_SAMPLE}"
            echo "  Suffix:       ${SUFFIX}"
            echo ""
            echo "  → Use in extraction script: \${HMMSEARCH_DIR}/\${SAMPLE}${SUFFIX}"
        fi
    fi
fi
echo ""

# =============================================================================
# 6. Software availability
# =============================================================================
echo "--- 6. Software availability ---"

export HOME=/scratch/drn2/newhome
source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate /scratch/drn2/software/conda-mamba_1 2>/dev/null || true

for tool in seqtk hmmsearch comm sort awk grep; do
    LOC=$(which ${tool} 2>/dev/null || echo "NOT FOUND")
    if [[ "${LOC}" != "NOT FOUND" ]]; then
        echo "  ${tool}: ${LOC}"
    else
        echo "  ${tool}: NOT FOUND"
    fi
done
echo ""

# =============================================================================
# 7. Disk space
# =============================================================================
echo "--- 7. Disk space ---"
echo "  /scratch/drn2/ usage:"
df -h /scratch/drn2/ 2>/dev/null | tail -1 | awk '{print "    Total: "$2"  Used: "$3"  Avail: "$4"  Use%: "$5}'
echo ""
echo "  TARA-LA4SR project size:"
du -sh "${BASE}" 2>/dev/null | awk '{print "    "$1" "$2}'
echo ""
echo "  03_analyses breakdown (top 10):"
du -sh "${BASE}"/03_analyses/*/ 2>/dev/null | sort -rh | head -10 | sed 's/^/    /'
echo ""

# =============================================================================
# 8. Estimated output size
# =============================================================================
echo "--- 8. Estimated output size ---"
if [[ ${N_FASTA} -gt 0 ]]; then
    TOTAL_FA_BYTES=$(du -sb "${FASTA_DIR}" 2>/dev/null | awk '{print $1}')
    TOTAL_FA_GB=$(awk "BEGIN {printf \"%.1f\", ${TOTAL_FA_BYTES} / 1073741824}")
    echo "  Total algae protein FASTA: ${TOTAL_FA_GB} GB"
    echo "  Estimated dark proteome (~88% of total): $(awk "BEGIN {printf \"%.1f\", ${TOTAL_FA_GB}*0.88}") GB"
    echo "  Plus ID files + stats: ~50 MB"
fi
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "  AUDIT SUMMARY"
echo "=============================================="
echo ""
echo "  Algal protein FASTA files:  ${N_FASTA}"
echo "  hmmsearch result files:     ${N_TBL}"
echo "  Samples in intersection:    ${N_BOTH:-0}"
echo "  sample_list.txt generated:  ${SAMPLE_LIST}"
echo ""

N_SAMPLES=$(wc -l < "${SAMPLE_LIST}" 2>/dev/null || echo 0)
if [[ ${N_SAMPLES} -gt 0 ]]; then
    echo "  NEXT STEPS:"
    echo "    1. Review this audit log"
    echo "    2. Update --array=1-${N_SAMPLES}%100 in 01_extract_dark.sbatch"
    echo "    3. Submit: bash MANUSCRIPT/scripts/dark_proteome/submit.sh"
else
    echo "  WARNING: No samples found. Check paths above."
fi
echo ""

# Cleanup temp files
rm -f "${FASTA_SAMPLES_FILE}" "${TBL_SAMPLES_FILE}"

echo "Audit complete: $(date)"
