#!/bin/bash
# =============================================================================
# A0: Reconnaissance — Determine SNAP Output Format
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_families/A0_recon_snap_format.sh
#   Generated: 2026-02-21
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Purpose:
#   Inspect SNAP gene prediction outputs to determine:
#   - File format (GFF3 vs ZFF vs FASTA headers with coordinates)
#   - Locate hmmsearch domtblout files (for Pfam annotations)
#   - Confirm protein FASTA paths
#   - Report SNAP coordinate format for downstream parsing
#
# Usage:
#   ssh jubail
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   bash MANUSCRIPT/scripts/novel_families/A0_recon_snap_format.sh 2>&1 | \
#     tee novel_families/logs/A0_recon_$(date +%Y%m%d_%H%M%S).log
#
# =============================================================================

set -euo pipefail

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
NOVEL_DIR="${BASE}/novel_families"
LOG_DIR="${NOVEL_DIR}/logs"
PROV_DIR="${NOVEL_DIR}/provenance"
mkdir -p "${LOG_DIR}" "${PROV_DIR}"

echo "=============================================="
echo "  A0: Reconnaissance — SNAP Output Format"
echo "=============================================="
echo ""
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "User:     $(whoami)"
echo "Base dir: ${BASE}"
echo ""

# ── 1. Locate SNAP outputs ──
echo "--- 1. Locating SNAP prediction outputs ---"

# Known possible locations for gene predictions
SNAP_DIRS=(
    "${BASE}/02_assemblies"
    "${BASE}/03_analyses/gene_predictions"
    "${BASE}/03_analyses/snap_results"
    "${BASE}/03_analyses/snap"
    "${BASE}/01_raw_data/proteins"
)

for dir in "${SNAP_DIRS[@]}"; do
    if [[ -d "${dir}" ]]; then
        echo "  FOUND: ${dir}"
        echo "    Contents (first 20):"
        ls "${dir}" 2>/dev/null | head -20 | sed 's/^/      /'
        echo "    File count: $(ls "${dir}" 2>/dev/null | wc -l)"
        echo "    Subdirectory count: $(find "${dir}" -maxdepth 1 -type d 2>/dev/null | wc -l)"
        echo ""
    fi
done

# Search for SNAP-related files
echo "  Searching for SNAP output files..."
for ext in ".zff" ".gff" ".gff3" ".snap.gff" ".snap_predictions"; do
    COUNT=$(find "${BASE}" -maxdepth 4 -name "*${ext}" 2>/dev/null | wc -l || true)
    if [[ ${COUNT} -gt 0 ]]; then
        echo "    *${ext}: ${COUNT} files"
        find "${BASE}" -maxdepth 4 -name "*${ext}" 2>/dev/null | head -3 | sed 's/^/      /' || true
    fi
done
echo ""

# ── 2. Inspect protein FASTA files ──
echo "--- 2. Protein FASTA files ---"

ALGAL_DIR="${BASE}/03_analyses/algae_proteins"
PROTEIN_DIRS=(
    "${ALGAL_DIR}"
    "${BASE}/03_analyses/predicted_proteins"
    "${BASE}/02_assemblies/proteins"
)

for dir in "${PROTEIN_DIRS[@]}"; do
    if [[ -d "${dir}" ]]; then
        echo "  FOUND: ${dir}"
        FA_COUNT=$(find "${dir}" -name "*.fa" -o -name "*.faa" -o -name "*.fasta" 2>/dev/null | wc -l || true)
        echo "    Protein FASTA count: ${FA_COUNT}"
        echo "    Total size: $(du -sh "${dir}" 2>/dev/null | awk '{print $1}')"

        # Inspect header format of first FASTA
        FIRST_FA=$(find "${dir}" -name "*.fa" -o -name "*.faa" 2>/dev/null | head -1 || true)
        if [[ -n "${FIRST_FA}" ]]; then
            echo "    First file: $(basename "${FIRST_FA}")"
            echo "    Header format (first 5 sequences):"
            grep "^>" "${FIRST_FA}" | head -5 | sed 's/^/      /' || true
            echo ""
            echo "    Header analysis:"
            FIRST_HEADER=$(grep "^>" "${FIRST_FA}" | head -1 || true)
            echo "      Raw: ${FIRST_HEADER}"
            # Check for coordinate info in header
            if echo "${FIRST_HEADER}" | grep -qE '[0-9]+\.\.[0-9]+|[0-9]+-[0-9]+|loc:|location:'; then
                echo "      COORDINATES DETECTED in FASTA headers"
            else
                echo "      No coordinates in FASTA headers — need external GFF/ZFF"
            fi
            # Check for contig info in header
            if echo "${FIRST_HEADER}" | grep -qiE 'contig|scaffold|NODE|k[0-9]+_'; then
                echo "      CONTIG INFO DETECTED in headers"
            fi
        fi
        echo ""
    fi
done

# ── 3. Inspect SNAP coordinate files (GFF/ZFF) ──
echo "--- 3. SNAP coordinate files ---"

# Look for GFF3 or ZFF files alongside assemblies
for pattern in "*.gff3" "*.gff" "*.zff" "*snap*" "*gene*predictions*"; do
    FILES=$(find "${BASE}" -maxdepth 5 -name "${pattern}" 2>/dev/null | head -5 || true)
    if [[ -n "${FILES}" ]]; then
        echo "  Pattern: ${pattern}"
        echo "${FILES}" | while read f; do
            echo "    File: ${f}"
            echo "    Size: $(ls -lh "${f}" | awk '{print $5}')"
            echo "    First 5 lines:"
            head -5 "${f}" | sed 's/^/      /' || true
            echo ""
        done
    fi
done

# ── 4. Locate hmmsearch domtblout files ──
echo "--- 4. hmmsearch domtblout files ---"

PFAM_DIR="${BASE}/03_analyses/hmmsearch_results"
HMMSEARCH_DIRS=(
    "${PFAM_DIR}"
    "${BASE}/03_analyses/hmmsearch"
    "${BASE}/03_analyses/pfam_hmmsearch"
)

for dir in "${HMMSEARCH_DIRS[@]}"; do
    if [[ -d "${dir}" ]]; then
        echo "  FOUND: ${dir}"
        # Look for domtblout files
        DOMTBL_COUNT=$(find "${dir}" -name "*.domtblout" -o -name "*domtbl*" 2>/dev/null | wc -l || true)
        TBL_COUNT=$(find "${dir}" -name "*.tbl" 2>/dev/null | wc -l || true)
        echo "    .domtblout files: ${DOMTBL_COUNT}"
        echo "    .tbl files: ${TBL_COUNT}"

        # Inspect format
        FIRST_TBL=$(find "${dir}" -name "*.tbl" -o -name "*.domtblout" 2>/dev/null | head -1 || true)
        if [[ -n "${FIRST_TBL}" ]]; then
            echo "    First file: $(basename "${FIRST_TBL}")"
            echo "    Format (first 5 non-comment lines):"
            grep -v "^#" "${FIRST_TBL}" | head -5 | sed 's/^/      /' || true
            echo ""

            # Determine if this is tblout or domtblout format
            NCOLS=$(grep -v "^#" "${FIRST_TBL}" | head -1 | awk '{print NF}' || true)
            echo "    Columns in first data line: ${NCOLS}"
            if [[ ${NCOLS} -ge 22 ]]; then
                echo "    FORMAT: domtblout (domain-level, >=22 columns)"
            elif [[ ${NCOLS} -ge 18 ]]; then
                echo "    FORMAT: tblout (sequence-level, ~18 columns)"
            else
                echo "    FORMAT: UNKNOWN (${NCOLS} columns)"
            fi
        fi
        echo ""
    fi
done

# ── 5. Assembly contig structure ──
echo "--- 5. Assembly contig structure ---"

ASSEMBLY_DIRS=(
    "${BASE}/02_assemblies"
    "${BASE}/01_raw_data/assemblies"
)

for dir in "${ASSEMBLY_DIRS[@]}"; do
    if [[ -d "${dir}" ]]; then
        echo "  FOUND: ${dir}"
        FIRST_ASM=$(find "${dir}" -name "*.fa" -o -name "*.fasta" -o -name "*.fna" 2>/dev/null | head -1 || true)
        if [[ -n "${FIRST_ASM}" ]]; then
            echo "    First assembly: $(basename "${FIRST_ASM}")"
            echo "    Contig headers (first 5):"
            grep "^>" "${FIRST_ASM}" | head -5 | sed 's/^/      /' || true
            echo "    Total contigs: $(grep -c "^>" "${FIRST_ASM}" || true)"
        fi
        echo ""
    fi
done

# ── 6. Check for prodigal as alternative to SNAP ──
echo "--- 6. Alternative gene callers (prodigal, etc.) ---"
for pattern in "*prodigal*" "*augustus*" "*glimmer*" "*genemark*"; do
    FILES=$(find "${BASE}" -maxdepth 4 -name "${pattern}" -type f 2>/dev/null | head -3 || true)
    if [[ -n "${FILES}" ]]; then
        echo "  ${pattern}:"
        echo "${FILES}" | sed 's/^/    /'
    fi
done
echo ""

# ── 7. Summary of discovered paths ──
echo "=============================================="
echo "  RECONNAISSANCE SUMMARY"
echo "=============================================="
echo ""

# Write provenance record
PROV_FILE="${PROV_DIR}/A0_recon_$(date +%Y%m%d_%H%M%S).md"
cat > "${PROV_FILE}" << 'PROVEOF'
# A0 Reconnaissance Provenance

## Purpose
Determine SNAP output format and data availability for novel families pipeline.

## Paths Discovered
(Populated by manual review of log output)

## Format Decisions
- Protein FASTA path: TBD
- Coordinate source: TBD (GFF3 / ZFF / FASTA headers)
- hmmsearch format: TBD (tblout / domtblout)

## Next Steps
- Update A1_build_contig_map.py with correct parsing logic
- Update A2_annotate_proteins.py with correct hmmsearch format
PROVEOF

echo "  Provenance written: ${PROV_FILE}"
echo ""
echo "  NEXT: Review this log and update A1/A2 parsers accordingly."
echo "  Done: $(date)"
