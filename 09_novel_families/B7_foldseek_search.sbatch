#!/bin/bash
#SBATCH --job-name=nf_B7_foldseek
#SBATCH --output=novel_families/logs/B7_foldseek_%j.out
#SBATCH --error=novel_families/logs/B7_foldseek_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=90G
#SBATCH --partition=compute

# =============================================================================
# B7: Foldseek Structural Search
# =============================================================================
#
# Search predicted structures against AlphaFold DB (Swiss-Prot) + PDB
# for structural homologs and functional inference.
#
# Prerequisites:
#   conda install -c bioconda foldseek
#   foldseek databases AlphaFoldDB-SwissProt /scratch/drn2/databases/afdb_sp
#   foldseek databases PDB /scratch/drn2/databases/pdb
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
PDB_DIR="${BASE}/novel_families/results/track_b/pdb"
RESULTS_DIR="${BASE}/novel_families/results/track_b/foldseek"
DB_DIR="/scratch/drn2/databases"
TMP_DIR="${TMPDIR}/foldseek_$$"

mkdir -p "${RESULTS_DIR}" "${TMP_DIR}"

echo "=============================================="
echo "  B7: Foldseek Structural Search"
echo "=============================================="

# Check for PDB files
N_PDB=$(ls "${PDB_DIR}"/*.pdb 2>/dev/null | wc -l)
echo "  PDB files: ${N_PDB}"

if [[ ${N_PDB} -eq 0 ]]; then
    echo "ERROR: No PDB files found in ${PDB_DIR}"
    echo "Run B6 first."
    exit 1
fi

# Create query database
echo "--- Creating query database ---"
QUERY_DB="${TMP_DIR}/queryDB"
foldseek createdb "${PDB_DIR}" "${QUERY_DB}"

# Search against AlphaFold DB (if available)
AFDB="${DB_DIR}/afdb_sp"
if [[ -f "${AFDB}" || -f "${AFDB}.dbtype" ]]; then
    echo "--- Searching AlphaFold DB (Swiss-Prot) ---"
    RESULT_DB="${TMP_DIR}/resultDB_afdb"
    foldseek search "${QUERY_DB}" "${AFDB}" "${RESULT_DB}" "${TMP_DIR}/tmp_afdb" \
        --threads "${SLURM_CPUS_PER_TASK:-28}" \
        -e 1e-3 \
        --max-seqs 100

    foldseek convertalis "${QUERY_DB}" "${AFDB}" "${RESULT_DB}" \
        "${RESULTS_DIR}/foldseek_vs_afdb.tsv" \
        --format-output "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,alntmscore"
    echo "  Written: foldseek_vs_afdb.tsv"
else
    echo "  AlphaFold DB not found at ${AFDB} — skipping"
    echo "  Download with: foldseek databases AlphaFoldDB-SwissProt ${AFDB} ${TMP_DIR}"
fi

# Search against PDB
PDB_DB="${DB_DIR}/pdb"
if [[ -f "${PDB_DB}" || -f "${PDB_DB}.dbtype" ]]; then
    echo "--- Searching PDB ---"
    RESULT_DB="${TMP_DIR}/resultDB_pdb"
    foldseek search "${QUERY_DB}" "${PDB_DB}" "${RESULT_DB}" "${TMP_DIR}/tmp_pdb" \
        --threads "${SLURM_CPUS_PER_TASK:-28}" \
        -e 1e-3

    foldseek convertalis "${QUERY_DB}" "${PDB_DB}" "${RESULT_DB}" \
        "${RESULTS_DIR}/foldseek_vs_pdb.tsv" \
        --format-output "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,alntmscore"
    echo "  Written: foldseek_vs_pdb.tsv"
else
    echo "  PDB database not found at ${PDB_DB} — skipping"
fi

# Cleanup
rm -rf "${TMP_DIR}"

echo ""
echo "Done: $(date)"
