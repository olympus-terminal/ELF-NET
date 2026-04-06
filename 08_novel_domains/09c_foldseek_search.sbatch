#!/bin/bash
#SBATCH --job-name=nd_09c_foldseek
#SBATCH --output=novel_domains/logs/09c_foldseek_%j.out
#SBATCH --error=novel_domains/logs/09c_foldseek_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=90G
#SBATCH --partition=compute

# =============================================================================
# 09c: Foldseek Structural Homology Search for Top Novel Domains
# =============================================================================
#
# Searches ESMFold-predicted structures against AlphaFold DB (Swiss-Prot)
# and PDB for structural homologs. "No significant hit" supports genuine
# structural novelty; hits provide functional annotation transfer.
#
# Input:  novel_domains/results/structures/*.pdb
# Output: novel_domains/results/structures/foldseek_vs_afdb.tsv
#         novel_domains/results/structures/foldseek_vs_pdb.tsv
#         novel_domains/results/structures/foldseek_best_hits.tsv
#
# Prerequisites:
#   conda install -c bioconda foldseek
#   foldseek databases Alphafold/Swiss-Prot /scratch/drn2/databases/afdb_sp tmp
#   foldseek databases PDB /scratch/drn2/databases/pdb tmp
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

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains"
PDB_DIR="${BASE}/results/structures"
RESULTS_DIR="${BASE}/results/structures"
DB_DIR="/scratch/drn2/databases"
TMP_DIR="${TMPDIR}/foldseek_$$"

mkdir -p "${RESULTS_DIR}" "${TMP_DIR}"

echo "=============================================="
echo "  09c: Foldseek Structural Homology Search"
echo "  Top 10 Novel Domains"
echo "=============================================="
echo "  PDB dir: ${PDB_DIR}"
echo "  Date: $(date)"
echo ""

# Check for PDB files
N_PDB=$(ls "${PDB_DIR}"/*.pdb 2>/dev/null | wc -l)
echo "  PDB files found: ${N_PDB}"

if [[ ${N_PDB} -eq 0 ]]; then
    echo "ERROR: No PDB files in ${PDB_DIR}"
    echo "Run 09b_esmfold_predict.sbatch first."
    exit 1
fi

ls "${PDB_DIR}"/*.pdb

# Check foldseek
if ! command -v foldseek &>/dev/null; then
    echo "ERROR: foldseek not found. Install with: conda install -c bioconda foldseek"
    exit 1
fi
echo ""
foldseek version 2>/dev/null || true

# Create query database from predicted structures
echo ""
echo "--- Creating query database ---"
QUERY_DB="${TMP_DIR}/queryDB"
foldseek createdb "${PDB_DIR}" "${QUERY_DB}"
echo "  Query DB created"

# --- Download databases if missing ---
download_db_if_needed() {
    local db_name="$1"
    local db_path="$2"
    local foldseek_name="$3"

    if [[ -f "${db_path}" || -f "${db_path}.dbtype" ]]; then
        echo "  ${db_name} found at ${db_path}"
        return 0
    else
        echo "  ${db_name} not found — downloading..."
        mkdir -p "$(dirname "${db_path}")"
        foldseek databases "${foldseek_name}" "${db_path}" "${TMP_DIR}/dl_${db_name}" \
            --threads "${SLURM_CPUS_PER_TASK:-28}"
        echo "  ${db_name} downloaded"
        return 0
    fi
}

# --- Search AlphaFold DB (Swiss-Prot) ---
AFDB="${DB_DIR}/afdb_sp"
echo ""
echo "--- AlphaFold DB (Swiss-Prot) ---"
download_db_if_needed "AFDB-SwissProt" "${AFDB}" "Alphafold/Swiss-Prot"

if [[ -f "${AFDB}" || -f "${AFDB}.dbtype" ]]; then
    echo "  Searching against AlphaFold DB..."
    RESULT_DB="${TMP_DIR}/resultDB_afdb"
    foldseek search "${QUERY_DB}" "${AFDB}" "${RESULT_DB}" "${TMP_DIR}/tmp_afdb" \
        --threads "${SLURM_CPUS_PER_TASK:-28}" \
        -e 1e-3 \
        --max-seqs 100 \
        --alignment-type 2 \
        -a

    foldseek convertalis "${QUERY_DB}" "${AFDB}" "${RESULT_DB}" \
        "${RESULTS_DIR}/foldseek_vs_afdb.tsv" \
        --format-output "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,alntmscore"

    N_HITS=$(wc -l < "${RESULTS_DIR}/foldseek_vs_afdb.tsv")
    echo "  Written: foldseek_vs_afdb.tsv (${N_HITS} hits)"
else
    echo "  WARNING: AlphaFold DB not available — skipping"
fi

# --- Search PDB ---
PDB_DB="${DB_DIR}/pdb"
echo ""
echo "--- PDB ---"
download_db_if_needed "PDB" "${PDB_DB}" "PDB"

if [[ -f "${PDB_DB}" || -f "${PDB_DB}.dbtype" ]]; then
    echo "  Searching against PDB..."
    RESULT_DB="${TMP_DIR}/resultDB_pdb"
    foldseek search "${QUERY_DB}" "${PDB_DB}" "${RESULT_DB}" "${TMP_DIR}/tmp_pdb" \
        --threads "${SLURM_CPUS_PER_TASK:-28}" \
        -e 1e-3 \
        --max-seqs 100 \
        -a

    foldseek convertalis "${QUERY_DB}" "${PDB_DB}" "${RESULT_DB}" \
        "${RESULTS_DIR}/foldseek_vs_pdb.tsv" \
        --format-output "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,alntmscore"

    N_HITS=$(wc -l < "${RESULTS_DIR}/foldseek_vs_pdb.tsv")
    echo "  Written: foldseek_vs_pdb.tsv (${N_HITS} hits)"
else
    echo "  WARNING: PDB database not available — skipping"
fi

# --- Extract best hit per query ---
echo ""
echo "--- Extracting best hits per domain ---"
python3 -u - << 'PYEOF'
import os
import csv
from collections import defaultdict

results_dir = os.environ.get("RESULTS_DIR",
    "/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains/results/structures")

cols = ["query", "target", "fident", "alnlen", "mismatch", "gapopen",
        "qstart", "qend", "tstart", "tend", "evalue", "bits", "alntmscore"]

best_hits = {}  # query -> {db: best_row}

for db_name, fname in [("AFDB", "foldseek_vs_afdb.tsv"),
                         ("PDB", "foldseek_vs_pdb.tsv")]:
    fpath = os.path.join(results_dir, fname)
    if not os.path.exists(fpath):
        print(f"  {fname} not found — skipping")
        continue

    with open(fpath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 13:
                continue
            row = dict(zip(cols, parts))
            query = row["query"]
            evalue = float(row["evalue"])
            tmscore = float(row["alntmscore"])

            key = (query, db_name)
            if key not in best_hits or tmscore > float(best_hits[key]["alntmscore"]):
                best_hits[key] = row
                best_hits[key]["database"] = db_name

# Write summary
out_path = os.path.join(results_dir, "foldseek_best_hits.tsv")
# Get all unique queries
all_queries = sorted(set(q for q, _ in best_hits.keys()))

with open(out_path, "w") as f:
    f.write("domain\tdatabase\tbest_target\ttm_score\tevalue\tfident\talign_len\n")
    for query in all_queries:
        for db in ["AFDB", "PDB"]:
            key = (query, db)
            if key in best_hits:
                row = best_hits[key]
                f.write(f"{query}\t{db}\t{row['target']}\t{row['alntmscore']}\t"
                        f"{row['evalue']}\t{row['fident']}\t{row['alnlen']}\n")

# Also report domains with NO hits
plddt_file = os.path.join(results_dir, "plddt_summary.tsv")
if os.path.exists(plddt_file):
    with open(plddt_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        all_domains = [row["domain"] for row in reader if row["status"].startswith("success")]

    queries_with_hits = set(all_queries)
    no_hits = [d for d in all_domains if d not in queries_with_hits]
    if no_hits:
        print(f"\n  Domains with NO structural homologs:")
        for d in no_hits:
            print(f"    {d}")
            with open(out_path, "a") as f:
                f.write(f"{d}\tNONE\tno_significant_hit\t0.0\tNA\t0.0\t0\n")

print(f"\n  Best hits summary: {out_path}")
print(f"  {len(all_queries)} domains with hits, {len(best_hits)} total best hits")
PYEOF

# Cleanup
rm -rf "${TMP_DIR}"

echo ""
echo "=============================================="
echo "  Done: $(date)"
echo "=============================================="
echo ""
echo "Results:"
ls -la "${RESULTS_DIR}"/foldseek_*.tsv 2>/dev/null || echo "  (no Foldseek results)"
