#!/bin/bash
#SBATCH --job-name=nf_B1_dark_mmseqs
#SBATCH --output=novel_families/logs/B1_mmseqs_dark_%j.out
#SBATCH --error=novel_families/logs/B1_mmseqs_dark_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=90G
#SBATCH --partition=compute

# =============================================================================
# B1: MMseqs2 clustering of full dark proteome (larger scale)
# =============================================================================
#
# Same parameters as A5 but on full dark proteome (potentially 60-130M seqs).
# Uses --split-memory-limit for large datasets.
#
# =============================================================================

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

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
INPUT_FASTA="${BASE}/03_analyses/novel_domains/dark_proteome_filtered.fa"
OUT_DIR="${BASE}/novel_families/data/mmseqs_clusters_dark"
TMP_DIR="${TMPDIR}/mmseqs_dark_$$"

mkdir -p "${OUT_DIR}" "${TMP_DIR}"

echo "=============================================="
echo "  B1: MMseqs2 Dark Proteome Clustering"
echo "=============================================="
echo ""
echo "Date:   $(date)"
echo "Input:  ${INPUT_FASTA}"

if [[ ! -f "${INPUT_FASTA}" ]]; then
    echo "ERROR: Dark proteome FASTA not found: ${INPUT_FASTA}"
    echo "Track B is staged — waiting for dark proteome agent output."
    exit 1
fi

N_SEQS=$(grep -c "^>" "${INPUT_FASTA}")
echo "Input sequences: ${N_SEQS}"

# Create database
echo "--- Creating MMseqs2 database ---"
DB="${TMP_DIR}/darkDB"
mmseqs createdb "${INPUT_FASTA}" "${DB}"

# Cluster
echo "--- Clustering at 30% identity ---"
CLUSTER_DB="${TMP_DIR}/clusterDB"
mmseqs cluster "${DB}" "${CLUSTER_DB}" "${TMP_DIR}/tmp" \
    --min-seq-id 0.30 \
    -c 0.80 \
    --cov-mode 0 \
    --cluster-mode 0 \
    --split-memory-limit 80G \
    --threads "${SLURM_CPUS_PER_TASK:-28}"

# Extract results
echo "--- Extracting results ---"
mmseqs createtsv "${DB}" "${DB}" "${CLUSTER_DB}" "${OUT_DIR}/cluster_membership.tsv"

# Representative sequences
REP_DB="${TMP_DIR}/repDB"
mmseqs result2repseq "${DB}" "${CLUSTER_DB}" "${REP_DB}"
mmseqs result2flat "${DB}" "${DB}" "${REP_DB}" "${OUT_DIR}/cluster_reps.fasta" --use-fasta-header

# Cluster sizes
awk '{print $1}' "${OUT_DIR}/cluster_membership.tsv" | sort | uniq -c | sort -rn | \
    awk '{print $2"\t"$1}' > "${OUT_DIR}/cluster_sizes.tsv"

N_CLUSTERS=$(wc -l < "${OUT_DIR}/cluster_sizes.tsv")
echo ""
echo "  Total clusters: ${N_CLUSTERS}"
echo "  Size distribution:"
echo "    Singletons:  $(awk '$2==1' "${OUT_DIR}/cluster_sizes.tsv" | wc -l)"
echo "    Size >=50:   $(awk '$2>=50' "${OUT_DIR}/cluster_sizes.tsv" | wc -l)"
echo "    Size >=100:  $(awk '$2>=100' "${OUT_DIR}/cluster_sizes.tsv" | wc -l)"

rm -rf "${TMP_DIR}"
echo ""
echo "Done: $(date)"
