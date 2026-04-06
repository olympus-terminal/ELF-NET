#!/bin/bash
# =============================================================================
# Step 0: HPC Audit — Verify data availability before pipeline launch
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_domains/00_audit_hpc.sh
#   Generated: 2026-02-21
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Purpose:
#   Run interactively on login node to verify:
#   - hmmsearch .tbl files exist and are parseable
#   - Algal sequence FASTA files exist
#   - DIAMOND dark_proteome_v2 results exist
#   - Required software is available (MMseqs2, MAFFT, HMMER, seqtk)
#   - Sufficient disk space
#
# Usage:
#   ssh jubail
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   bash MANUSCRIPT/scripts/novel_domains/00_audit_hpc.sh 2>&1 | tee audit_novel_domains_$(date +%Y%m%d_%H%M%S).log
#
# =============================================================================

set -uo pipefail
# NOTE: not using -e; audit is diagnostic and should not abort on missing files

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
HMMSEARCH_DIR="${BASE}/03_analyses/hmmsearch_results"
ALGAL_DIR="${BASE}/03_analyses/algae_proteins"
DARK_V2_DIR="${BASE}/03_analyses/dark_proteome_v2"

echo "=============================================="
echo "  Novel Domain Discovery Pipeline — HPC Audit"
echo "=============================================="
echo ""
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "User:     $(whoami)"
echo "Base dir: ${BASE}"
echo ""

# ---- 1. Pfam hmmsearch result files ----
echo "--- 1. Pfam hmmsearch result files ---"
if [[ -d "${HMMSEARCH_DIR}" ]]; then
    # Check for .hmmsearch.tbl files (canonical format in hmmsearch_results/)
    TBL_COUNT=$(ls "${HMMSEARCH_DIR}"/*.aa.hmmsearch.tbl 2>/dev/null | wc -l)
    TBL_DOMTBL=$(ls "${HMMSEARCH_DIR}"/*.aa.hmmsearch.domtbl 2>/dev/null | wc -l)
    echo "  Directory exists: ${HMMSEARCH_DIR}"
    echo "  .aa.hmmsearch.tbl file count: ${TBL_COUNT}"
    echo "  .aa.hmmsearch.domtbl file count: ${TBL_DOMTBL}"
    if [[ ${TBL_COUNT} -gt 0 ]]; then
        echo "  First 5 .tbl files:"
        ls "${HMMSEARCH_DIR}"/*.aa.hmmsearch.tbl 2>/dev/null | head -5 | sed 's/^/    /'
        echo ""
        echo "  Format check (first .tbl file header):"
        FIRST_TBL=$(ls "${HMMSEARCH_DIR}"/*.aa.hmmsearch.tbl 2>/dev/null | head -1)
        head -5 "${FIRST_TBL}" | sed 's/^/    /'
        echo ""
        echo "  File size range:"
        ls -lhS "${HMMSEARCH_DIR}"/*.aa.hmmsearch.tbl 2>/dev/null | head -1 | awk '{print "    Largest:  "$5" "$NF}'
        ls -lhS "${HMMSEARCH_DIR}"/*.aa.hmmsearch.tbl 2>/dev/null | tail -1 | awk '{print "    Smallest: "$5" "$NF}'
    fi
else
    TBL_COUNT=0
    echo "  WARNING: Directory not found: ${HMMSEARCH_DIR}"
fi
echo ""

# ---- 3. Algal sequence FASTA files ----
echo "--- 3. Algal sequence FASTA files ---"
if [[ -d "${ALGAL_DIR}" ]]; then
    FA_COUNT=$(ls "${ALGAL_DIR}"/*.fa 2>/dev/null | wc -l)
    echo "  Directory exists: ${ALGAL_DIR}"
    echo "  .fa file count: ${FA_COUNT}"
    if [[ ${FA_COUNT} -gt 0 ]]; then
        echo "  First 5 files:"
        ls "${ALGAL_DIR}"/*.fa 2>/dev/null | head -5 | sed 's/^/    /'
        echo ""
        echo "  Total size:"
        du -sh "${ALGAL_DIR}" | awk '{print "    "$1}'
    fi
else
    echo "  WARNING: Directory not found: ${ALGAL_DIR}"
    echo "  Checking alternative paths..."
    for alt in "${BASE}/02_assemblies" "${BASE}/01_raw_data/proteins"; do
        if [[ -d "${alt}" ]]; then
            echo "    Found: ${alt} ($(ls "${alt}"/*.fa 2>/dev/null | wc -l) .fa files)"
        fi
    done
fi
echo ""

# ---- 4. DIAMOND dark_proteome_v2 results ----
echo "--- 4. DIAMOND dark_proteome_v2 results ---"
if [[ -d "${DARK_V2_DIR}" ]]; then
    echo "  Directory exists: ${DARK_V2_DIR}"
    du -sh "${DARK_V2_DIR}" | awk '{print "  Total size: "$1}'
    # Check for hit ID files
    HIT_COUNT=$(find "${DARK_V2_DIR}" -name "*.hit_ids.txt" 2>/dev/null | wc -l)
    echo "  hit_ids.txt files: ${HIT_COUNT}"
    # Check for no-hit (dark) files
    NOHIT_COUNT=$(find "${DARK_V2_DIR}" -name "*no_hit*" -o -name "*dark*" 2>/dev/null | wc -l)
    echo "  no-hit/dark files: ${NOHIT_COUNT}"
    echo "  Contents:"
    ls "${DARK_V2_DIR}"/ 2>/dev/null | head -20 | sed 's/^/    /'
else
    echo "  Not found: ${DARK_V2_DIR}"
fi
echo ""

# ---- 5. Existing dark_proteome analysis script ----
echo "--- 5. Existing dark_proteome analysis scripts ---"
for pattern in "dark_proteome" "dark_prot"; do
    FOUND=$(find "${BASE}/scripts" "${BASE}/MANUSCRIPT/scripts" -iname "*${pattern}*" 2>/dev/null)
    if [[ -n "${FOUND}" ]]; then
        echo "  Found scripts:"
        echo "${FOUND}" | sed 's/^/    /'
    fi
done
echo ""

# ---- 6. Software availability ----
echo "--- 6. Software availability ---"

# Load conda environment and HMMER module (same as pipeline scripts)
export HOME=/scratch/drn2/newhome
module load hmmer/3.3 2>/dev/null || true
source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate /scratch/drn2/software/conda-mamba_1 2>/dev/null || true

for tool in mmseqs hmmsearch hmmbuild hmmpress mafft seqtk; do
    LOC=$(which ${tool} 2>/dev/null || echo "NOT FOUND")
    if [[ "${LOC}" != "NOT FOUND" ]]; then
        VER=$(${tool} --version 2>&1 | head -1 || echo "version unknown")
        echo "  ${tool}: ${LOC} (${VER})"
    else
        echo "  ${tool}: NOT FOUND — needs installation"
    fi
done
echo ""

# ---- 7. Disk space ----
echo "--- 7. Disk space ---"
echo "  /scratch/drn2/ usage:"
df -h /scratch/drn2/ 2>/dev/null | tail -1 | awk '{print "    Total: "$2"  Used: "$3"  Avail: "$4"  Use%: "$5}'
echo ""
echo "  TARA-LA4SR project size:"
du -sh "${BASE}" 2>/dev/null | awk '{print "    "$1" "$2}'
echo ""
echo "  03_analyses breakdown:"
du -sh "${BASE}"/03_analyses/*/ 2>/dev/null | sort -rh | head -10 | sed 's/^/    /'
echo ""

# ---- 8. Sample list generation ----
echo "--- 8. Building sample list ---"
if [[ -d "${ALGAL_DIR}" ]]; then
    echo "  Generating sample list from FASTA files..."
    SAMPLE_LIST="${BASE}/03_analyses/novel_domains/sample_list.txt"
    mkdir -p "$(dirname "${SAMPLE_LIST}")"
    # Strip .algae.fa suffix to get sample stems (e.g., Alexandrium_andersonii.AAC.aa)
    ls "${ALGAL_DIR}"/*.fa 2>/dev/null | xargs -n1 basename | sed 's/\.algae\.fa$//; s/\.fa$//' | sort > "${SAMPLE_LIST}"
    NSAMP=$(wc -l < "${SAMPLE_LIST}")
    echo "  Sample list written: ${SAMPLE_LIST}"
    echo "  Total samples: ${NSAMP}"
    echo "  First 5:"
    head -5 "${SAMPLE_LIST}" | sed 's/^/    /'
fi
echo ""

# ---- 9. Cross-reference: hmmsearch vs FASTA coverage ----
echo "--- 9. Cross-reference: hmmsearch vs FASTA coverage ---"
if [[ ${TBL_COUNT:-0} -gt 0 && ${FA_COUNT:-0} -gt 0 ]]; then
    # Extract stems: {species}.AAC.aa from .hmmsearch.tbl and .algae.fa
    TBL_STEMS=$(ls "${HMMSEARCH_DIR}"/*.hmmsearch.tbl 2>/dev/null | xargs -n1 basename | sed 's/\.hmmsearch\.tbl$//' | sort -u)
    FA_STEMS=$(ls "${ALGAL_DIR}"/*.fa 2>/dev/null | xargs -n1 basename | sed 's/\.algae\.fa$//; s/\.fa$//' | sort -u)
    TBL_ONLY=$(comm -23 <(echo "${TBL_STEMS}") <(echo "${FA_STEMS}") | wc -l)
    FA_ONLY=$(comm -13 <(echo "${TBL_STEMS}") <(echo "${FA_STEMS}") | wc -l)
    BOTH=$(comm -12 <(echo "${TBL_STEMS}") <(echo "${FA_STEMS}") | wc -l)
    echo "  Samples with both hmmsearch and FASTA: ${BOTH}"
    echo "  Samples with hmmsearch only: ${TBL_ONLY}"
    echo "  Samples with FASTA only (no Pfam annotation): ${FA_ONLY}"
else
    echo "  Cannot cross-reference (missing hmmsearch or FASTA files)"
fi
echo ""

# ---- Summary ----
echo "=============================================="
echo "  AUDIT SUMMARY"
echo "=============================================="
echo ""
echo "  hmmsearch .domtblout files: ${TBL_COUNT:-0}"
echo "  Algal FASTA files:         ${FA_COUNT:-0}"
echo "  DIAMOND dark_v2 results:   ${HIT_COUNT:-0}"
echo "  MMseqs2 available:         $(which mmseqs 2>/dev/null && echo YES || echo NO)"
echo ""

if [[ ${TBL_COUNT:-0} -gt 0 ]]; then
    echo "  DECISION: hmmsearch results exist → proceed with Step 1 (extract dark IDs)"
else
    echo "  DECISION: No hmmsearch results → need to run hmmsearch first (major compute)"
fi
echo ""
echo "  Audit complete: $(date)"
