#!/bin/bash
#SBATCH --job-name=nf_B1_100mb
#SBATCH --output=novel_families/logs/B1_100mb_%j.out
#SBATCH --error=novel_families/logs/B1_100mb_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=compute

# =============================================================================
# B1_100mb: MMseqs2 two-tier clustering for Kourosh 100MB subsets
# =============================================================================
#
# Two-tier cascade: 30% identity → 50% identity (both 80% coverage).
# Uses `mmseqs cluster` (cascade mode) instead of easy-linclust for better
# sensitivity at this scale (~440K sequences).
#
# Env var: SUBSET=dark|annotated (set by launcher)
#
# Usage:
#   SUBSET=dark sbatch B1_mmseqs_100mb.sbatch
# =============================================================================

# --- Jubail mandatory env block ---
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

set -euo pipefail

# --- Validate SUBSET env var ---
SUBSET="${SUBSET:-}"
if [[ -z "${SUBSET}" ]]; then
    echo "ERROR: SUBSET env var not set. Use SUBSET=dark or SUBSET=annotated"
    exit 1
fi

if [[ "${SUBSET}" != "dark" && "${SUBSET}" != "annotated" ]]; then
    echo "ERROR: SUBSET must be 'dark' or 'annotated', got '${SUBSET}'"
    exit 1
fi

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
KOUROSH_DIR="${BASE}/03_analyses/novel_domains/kourosh_subsets"

if [[ "${SUBSET}" == "dark" ]]; then
    INPUT_FASTA="${KOUROSH_DIR}/dark_proteome_100mb.fa"
else
    INPUT_FASTA="${KOUROSH_DIR}/annotated_proteome_100mb.fa"
fi

OUT_DIR="${KOUROSH_DIR}/${SUBSET}"
TMP_DIR="${TMPDIR}/mmseqs_100mb_${SUBSET}_$$"
THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${OUT_DIR}/clusters_30" "${OUT_DIR}/clusters_50" "${OUT_DIR}/logs" "${TMP_DIR}"

echo "=============================================="
echo "  B1_100mb: MMseqs2 Clustering — ${SUBSET}"
echo "=============================================="
echo ""
echo "Date:    $(date)"
echo "Input:   ${INPUT_FASTA}"
echo "Output:  ${OUT_DIR}"
echo "Threads: ${THREADS}"

if [[ ! -f "${INPUT_FASTA}" ]]; then
    echo "ERROR: Input FASTA not found: ${INPUT_FASTA}"
    exit 1
fi

N_SEQS=$(grep -c "^>" "${INPUT_FASTA}")
echo "Input sequences: ${N_SEQS}"
echo ""

# ── Create MMseqs2 database ──
echo "--- Creating MMseqs2 database ---"
DB="${TMP_DIR}/seqDB"
mmseqs createdb "${INPUT_FASTA}" "${DB}"

# ══════════════════════════════════════════════
# Tier 1: Cluster at 30% identity, 80% coverage
# ══════════════════════════════════════════════
echo ""
echo "--- Tier 1: Clustering at 30% identity, 80% coverage ---"
CLUSTER30_DB="${TMP_DIR}/cluster30DB"
mmseqs cluster "${DB}" "${CLUSTER30_DB}" "${TMP_DIR}/tmp30" \
    --min-seq-id 0.30 \
    -c 0.80 \
    --cov-mode 0 \
    --cluster-mode 0 \
    --threads "${THREADS}"

echo "  Extracting 30% cluster results..."
mmseqs createtsv "${DB}" "${DB}" "${CLUSTER30_DB}" \
    "${OUT_DIR}/clusters_30/membership.tsv"

# Representative sequences
REP30_DB="${TMP_DIR}/rep30DB"
mmseqs result2repseq "${DB}" "${CLUSTER30_DB}" "${REP30_DB}"
mmseqs result2flat "${DB}" "${DB}" "${REP30_DB}" \
    "${OUT_DIR}/clusters_30/reps.fasta" --use-fasta-header

# Cluster sizes
awk '{print $1}' "${OUT_DIR}/clusters_30/membership.tsv" | sort | uniq -c | sort -rn | \
    awk '{print $2"\t"$1}' > "${OUT_DIR}/clusters_30/sizes.tsv"

N_CLUSTERS_30=$(wc -l < "${OUT_DIR}/clusters_30/sizes.tsv")
echo ""
echo "  30% clusters: ${N_CLUSTERS_30}"
echo "    Singletons:  $(awk '$2==1' "${OUT_DIR}/clusters_30/sizes.tsv" | wc -l)"
echo "    Size >=5:    $(awk '$2>=5' "${OUT_DIR}/clusters_30/sizes.tsv" | wc -l)"
echo "    Size >=50:   $(awk '$2>=50' "${OUT_DIR}/clusters_30/sizes.tsv" | wc -l)"

# ══════════════════════════════════════════════
# Tier 2: Cluster at 50% identity, 80% coverage
# ══════════════════════════════════════════════
echo ""
echo "--- Tier 2: Clustering at 50% identity, 80% coverage ---"
CLUSTER50_DB="${TMP_DIR}/cluster50DB"
mmseqs cluster "${DB}" "${CLUSTER50_DB}" "${TMP_DIR}/tmp50" \
    --min-seq-id 0.50 \
    -c 0.80 \
    --cov-mode 0 \
    --cluster-mode 0 \
    --threads "${THREADS}"

echo "  Extracting 50% cluster results..."
mmseqs createtsv "${DB}" "${DB}" "${CLUSTER50_DB}" \
    "${OUT_DIR}/clusters_50/membership.tsv"

# Representative sequences
REP50_DB="${TMP_DIR}/rep50DB"
mmseqs result2repseq "${DB}" "${CLUSTER50_DB}" "${REP50_DB}"
mmseqs result2flat "${DB}" "${DB}" "${REP50_DB}" \
    "${OUT_DIR}/clusters_50/reps.fasta" --use-fasta-header

# Cluster sizes
awk '{print $1}' "${OUT_DIR}/clusters_50/membership.tsv" | sort | uniq -c | sort -rn | \
    awk '{print $2"\t"$1}' > "${OUT_DIR}/clusters_50/sizes.tsv"

N_CLUSTERS_50=$(wc -l < "${OUT_DIR}/clusters_50/sizes.tsv")
echo ""
echo "  50% clusters: ${N_CLUSTERS_50}"
echo "    Singletons:  $(awk '$2==1' "${OUT_DIR}/clusters_50/sizes.tsv" | wc -l)"
echo "    Size >=5:    $(awk '$2>=5' "${OUT_DIR}/clusters_50/sizes.tsv" | wc -l)"
echo "    Size >=50:   $(awk '$2>=50' "${OUT_DIR}/clusters_50/sizes.tsv" | wc -l)"

# ── Cleanup ──
rm -rf "${TMP_DIR}"

echo ""
echo "=============================================="
echo "  B1_100mb Complete — ${SUBSET}"
echo "=============================================="
echo "  30% clusters: ${N_CLUSTERS_30}"
echo "  50% clusters: ${N_CLUSTERS_50}"
echo "  Done: $(date)"
