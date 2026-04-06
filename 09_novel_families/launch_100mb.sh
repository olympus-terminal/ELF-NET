#!/bin/bash
# =============================================================================
# Master Launch Script: 100MB Subset Clustering Pipeline
# =============================================================================
#
# Runs the B0→B5 100mb pipeline on both dark and annotated Kourosh subsets.
# B0 runs inline (login node, ~5 min). B1→B4 are SLURM dependency chains.
# B5 depends only on B1 (needs cluster membership, not filtered families).
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   bash MANUSCRIPT/scripts/novel_families/launch_100mb.sh
#
# Total wall time per subset: ~4-5 hours. Both subsets run in parallel.
# =============================================================================

set -euo pipefail

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
SCRIPTS="${BASE}/MANUSCRIPT/scripts/novel_families"
KOUROSH_DIR="${BASE}/03_analyses/novel_domains/kourosh_subsets"

# --- Jubail env setup (for inline Python steps) ---
export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
export PYTHONUSERBASE=/scratch/drn2/newhome/.local
mkdir -p "$TMPDIR"

unset NETWORKX_BACKEND_CONFIG 2>/dev/null || true
unset NX_BACKEND_CONFIG 2>/dev/null || true

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

# --- Create output dirs ---
for SUBSET in dark annotated; do
    mkdir -p "${KOUROSH_DIR}/${SUBSET}"/{clusters_30,clusters_50,logs}
done
mkdir -p "${BASE}/novel_families/logs"

echo "============================================================"
echo "  100MB Subset Pipeline: Dark + Annotated"
echo "============================================================"
echo ""
echo "  Kourosh dir: ${KOUROSH_DIR}"
echo "  Date: $(date)"
echo ""

# Common sbatch wrapper for inline Python steps
CONDA_WRAP="cd ${BASE} && \
export HOME=/scratch/drn2/newhome && \
export TMPDIR=/scratch/drn2/tmp && \
export PYTHONUSERBASE=/scratch/drn2/newhome/.local && \
mkdir -p \$TMPDIR && \
unset NETWORKX_BACKEND_CONFIG 2>/dev/null; \
unset NX_BACKEND_CONFIG 2>/dev/null; \
module purge && module load gcc/13.2.0 && \
source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh && \
conda activate /scratch/drn2/software/conda-mamba_1"

for SUBSET in dark annotated; do
    echo "────────────────────────────────────────"
    echo "  Processing: ${SUBSET}"
    echo "────────────────────────────────────────"

    SUBSET_DIR="${KOUROSH_DIR}/${SUBSET}"
    LOG_DIR="${SUBSET_DIR}/logs"

    # ── B0: Build assembly map (inline, ~5 min) ──
    echo "  Running B0: Build assembly map (inline)..."
    python3 -u "${SCRIPTS}/B0_build_assembly_map_100mb.py" --subset "${SUBSET}"
    B0_EXIT=$?

    if [[ ${B0_EXIT} -ne 0 ]]; then
        echo "  B0 FAILED for ${SUBSET} — skipping this subset."
        echo ""
        continue
    fi

    # ── B1: MMseqs2 clustering (SLURM) ──
    echo "  Submitting B1: MMseqs2 clustering..."
    JOB_B1=$(SUBSET="${SUBSET}" sbatch --parsable \
        --export=ALL,SUBSET="${SUBSET}" \
        "${SCRIPTS}/B1_mmseqs_100mb.sbatch")
    echo "    Job B1: ${JOB_B1}"

    # ── B2: Filter families (depends on B1) ──
    echo "  Submitting B2: Filter families (depends on B1)..."
    JOB_B2=$(sbatch --parsable --dependency=afterok:${JOB_B1} \
        --wrap="${CONDA_WRAP} && \
        python3 -u ${SCRIPTS}/B2_filter_families_100mb.py --subset ${SUBSET}" \
        --job-name="nf_B2_100mb_${SUBSET}" --time=00:30:00 --mem=8G \
        --partition=compute \
        --output="${LOG_DIR}/B2_%j.out" --error="${LOG_DIR}/B2_%j.err")
    echo "    Job B2: ${JOB_B2}"

    # ── B3: Abundance matrix (depends on B2) ──
    echo "  Submitting B3: Abundance matrix (depends on B2)..."
    JOB_B3=$(sbatch --parsable --dependency=afterok:${JOB_B2} \
        --wrap="${CONDA_WRAP} && \
        python3 -u ${SCRIPTS}/B3_abundance_matrix_100mb.py --subset ${SUBSET}" \
        --job-name="nf_B3_100mb_${SUBSET}" --time=00:30:00 --mem=8G \
        --partition=compute \
        --output="${LOG_DIR}/B3_%j.out" --error="${LOG_DIR}/B3_%j.err")
    echo "    Job B3: ${JOB_B3}"

    # ── B4: Environment coupling — tier 30% (depends on B3) ──
    echo "  Submitting B4: Env coupling 30% (depends on B3)..."
    JOB_B4_30=$(sbatch --parsable --dependency=afterok:${JOB_B3} \
        --wrap="${CONDA_WRAP} && \
        python3 -u ${SCRIPTS}/B4_env_coupling_100mb.py --subset ${SUBSET} --tier 30" \
        --job-name="nf_B4_100mb_${SUBSET}_30" --time=02:00:00 --mem=32G \
        --cpus-per-task=8 --partition=compute \
        --output="${LOG_DIR}/B4_30_%j.out" --error="${LOG_DIR}/B4_30_%j.err")
    echo "    Job B4 (30%): ${JOB_B4_30}"

    # ── B4: Environment coupling — tier 50% (depends on B3) ──
    echo "  Submitting B4: Env coupling 50% (depends on B3)..."
    JOB_B4_50=$(sbatch --parsable --dependency=afterok:${JOB_B3} \
        --wrap="${CONDA_WRAP} && \
        python3 -u ${SCRIPTS}/B4_env_coupling_100mb.py --subset ${SUBSET} --tier 50" \
        --job-name="nf_B4_100mb_${SUBSET}_50" --time=02:00:00 --mem=32G \
        --cpus-per-task=8 --partition=compute \
        --output="${LOG_DIR}/B4_50_%j.out" --error="${LOG_DIR}/B4_50_%j.err")
    echo "    Job B4 (50%): ${JOB_B4_50}"

    # ── B5: Export membership (depends on B1 only) ──
    echo "  Submitting B5: Export membership (depends on B1)..."
    JOB_B5=$(sbatch --parsable --dependency=afterok:${JOB_B1} \
        --wrap="${CONDA_WRAP} && \
        python3 -u ${SCRIPTS}/B5_export_membership_100mb.py --subset ${SUBSET}" \
        --job-name="nf_B5_100mb_${SUBSET}" --time=00:15:00 --mem=4G \
        --partition=compute \
        --output="${LOG_DIR}/B5_%j.out" --error="${LOG_DIR}/B5_%j.err")
    echo "    Job B5: ${JOB_B5}"

    echo ""
    echo "  ${SUBSET} job chain:"
    echo "    B0 (inline) → B1 (${JOB_B1}) → B2 (${JOB_B2}) → B3 (${JOB_B3})"
    echo "      → B4/30% (${JOB_B4_30})"
    echo "      → B4/50% (${JOB_B4_50})"
    echo "    B1 (${JOB_B1}) → B5 (${JOB_B5})"
    echo ""
done

echo "============================================================"
echo "  All jobs submitted. Monitor with: squeue -u \$USER"
echo "============================================================"
echo ""
echo "  Verification checklist:"
echo "    1. After B0: check protein_to_assembly.tsv coverage (>85%)"
echo "    2. After B1: check cluster counts (wc -l clusters_*/membership.tsv)"
echo "    3. After B2: check filtered families count"
echo "    4. After B5: verify all proteins in membership TSV"
echo "       wc -l {dark,annotated}/membership_30pct.tsv"
echo ""
echo "  Key deliverables for Kourosh comparison:"
echo "    ${KOUROSH_DIR}/dark/membership_30pct.tsv"
echo "    ${KOUROSH_DIR}/dark/membership_50pct.tsv"
echo "    ${KOUROSH_DIR}/annotated/membership_30pct.tsv"
echo "    ${KOUROSH_DIR}/annotated/membership_50pct.tsv"
echo ""
