#!/bin/bash
# =============================================================================
# Master Launch Script: Track A — Contig Co-localization Pipeline
# =============================================================================
#
# Submits all Track A stages with SLURM dependency chains.
# Total wall time: ~22 hours (sequential pipeline).
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   bash MANUSCRIPT/scripts/novel_families/launch_track_a.sh
#
# =============================================================================

set -euo pipefail

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
SCRIPTS="${BASE}/MANUSCRIPT/scripts/novel_families"

# Create required directories
mkdir -p novel_families/logs
mkdir -p novel_families/data/{contig_maps,anchor_neighborhoods,mmseqs_clusters,abundance_matrices}
mkdir -p novel_families/results/{track_a,track_b}
mkdir -p novel_families/figures
mkdir -p novel_families/provenance

echo "=============================================="
echo "  Track A: Contig Co-localization Pipeline"
echo "=============================================="
echo ""
echo "  Date:    $(date)"
echo "  Base:    ${BASE}"
echo "  Scripts: ${SCRIPTS}"
echo ""

# ── A0: Reconnaissance ──
echo "  Submitting A0: Reconnaissance..."
JOB_A0=$(sbatch --parsable "${SCRIPTS}/A0_recon.sbatch")
echo "    Job A0: ${JOB_A0}"

# ── A1: Build contig map ──
echo "  Submitting A1: Build contig map (depends on A0)..."
JOB_A1=$(sbatch --parsable --dependency=afterok:${JOB_A0} "${SCRIPTS}/A1_contig_map.sbatch")
echo "    Job A1: ${JOB_A1}"

# ── A2: Annotate proteins ──
echo "  Submitting A2: Annotate proteins (depends on A1)..."
JOB_A2=$(sbatch --parsable --dependency=afterok:${JOB_A1} "${SCRIPTS}/A2_annotate.sbatch")
echo "    Job A2: ${JOB_A2}"

# ── A3: Extract anchor neighborhoods ──
echo "  Submitting A3: Anchor neighborhoods (depends on A2)..."
JOB_A3=$(sbatch --parsable --dependency=afterok:${JOB_A2} "${SCRIPTS}/A3_anchor.sbatch")
echo "    Job A3: ${JOB_A3}"

# ── A4: Extract unannotated sequences ──
echo "  Submitting A4: Extract sequences (depends on A3)..."
JOB_A4=$(sbatch --parsable --dependency=afterok:${JOB_A3} "${SCRIPTS}/A4_extract_seqs.sbatch")
echo "    Job A4: ${JOB_A4}"

# ── A5: MMseqs2 clustering ──
echo "  Submitting A5: MMseqs2 clustering (depends on A4)..."
JOB_A5=$(sbatch --parsable --dependency=afterok:${JOB_A4} "${SCRIPTS}/A5_mmseqs.sbatch")
echo "    Job A5: ${JOB_A5}"

# ── A6: Filter novel families ──
echo "  Submitting A6: Filter families (depends on A5)..."
JOB_A6=$(sbatch --parsable --dependency=afterok:${JOB_A5} "${SCRIPTS}/A6_filter.sbatch")
echo "    Job A6: ${JOB_A6}"

# ── A7: Build abundance matrix ──
echo "  Submitting A7: Abundance matrix (depends on A6)..."
JOB_A7=$(sbatch --parsable --dependency=afterok:${JOB_A6} "${SCRIPTS}/A7_abundance.sbatch")
echo "    Job A7: ${JOB_A7}"

# ── A8: Environment coupling ──
echo "  Submitting A8: Environment coupling (depends on A7)..."
JOB_A8=$(sbatch --parsable --dependency=afterok:${JOB_A7} "${SCRIPTS}/A8_coupling.sbatch")
echo "    Job A8: ${JOB_A8}"

# ── A9: Compare distributions ──
echo "  Submitting A9: Compare distributions (depends on A8)..."
JOB_A9=$(sbatch --parsable --dependency=afterok:${JOB_A8} "${SCRIPTS}/A9_compare.sbatch")
echo "    Job A9: ${JOB_A9}"

# ── A10: Neighborhood diagrams ──
echo "  Submitting A10: Diagrams (depends on A8)..."
JOB_A10=$(sbatch --parsable --dependency=afterok:${JOB_A8} "${SCRIPTS}/A10_diagrams.sbatch")
echo "    Job A10: ${JOB_A10}"

echo ""
echo "=============================================="
echo "  Pipeline Submitted Successfully"
echo "=============================================="
echo ""
echo "  Job Chain:"
echo "    A0 (${JOB_A0}) → A1 (${JOB_A1}) → A2 (${JOB_A2}) → A3 (${JOB_A3})"
echo "    → A4 (${JOB_A4}) → A5 (${JOB_A5}) → A6 (${JOB_A6}) → A7 (${JOB_A7})"
echo "    → A8 (${JOB_A8}) → A9 (${JOB_A9})"
echo "                     → A10 (${JOB_A10})  [parallel with A9]"
echo ""
echo "  Monitor with:"
echo "    squeue -u drn2"
echo "    tail -f novel_families/logs/A*_*.out"
echo ""
echo "  Expected wall time: ~22 hours"
echo ""
