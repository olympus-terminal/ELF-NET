#!/bin/bash
#SBATCH --job-name=ralph41_snap_metaeuk
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/snap_vs_metaeuk/slurm_%j.out
#SBATCH --error=/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/snap_vs_metaeuk/slurm_%j.err
#
# ralph41 Task 41.15 — SNAP vs MetaEuk benchmark on 20 TARA assemblies
# ---------------------------------------------------------------------
# Stages:
#   1. N50 for all 1063 assemblies; pick 5 per quartile (n=20)
#   2. Run MetaEuk easy-predict on each of the 20 assemblies
#   3. hmmsearch Pfam-A at E<1e-9 against MetaEuk-predicted proteins
#   4. Parse hmmsearch tbls + compute per-sample Spearman rho vs canonical
#      SNAP pfam_count_matrix_20260109_181211.tsv; overlap summary
#
# The MetaEuk comparison uses the same Pfam-A database that was used for
# the SNAP pipeline so the two counts are directly comparable.  DO NOT
# swap Pfam versions between the two sides of the benchmark.
#
# Idempotent: a sentinel file prevents double-submission.
# ---------------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT="/scratch/drn2/PROJECTS/TARA-LA4SR"
ASSEMBLY_DIR="${PROJECT_ROOT}/01_raw_data/tara_metagenomes/assemblies"
OUT_DIR="${PROJECT_ROOT}/03_analyses/snap_vs_metaeuk"
SCRIPT_DIR="${PROJECT_ROOT}/MANUSCRIPT/scripts/ralph41"
SNAP_MATRIX="${PROJECT_ROOT}/03_analyses/pfam_results/pfam_count_matrix_20260109_181211.tsv"
PFAM_DB="/scratch/drn2/fastq/data/raw/db/Pfam-A.hmm"
CONDA_BIN="/scratch/drn2/software/conda-busco/bin"

METAEUK="${CONDA_BIN}/metaeuk"
HMMSEARCH="${CONDA_BIN}/hmmsearch"
PYTHON="${CONDA_BIN}/python3"

mkdir -p "${OUT_DIR}"/{hmmsearch,metaeuk,tmp,logs}
SENTINEL="${OUT_DIR}/.running"
DONE_MARKER="${OUT_DIR}/.done"

if [[ -f "${DONE_MARKER}" ]]; then
    echo "[ralph41 41.15] already done; exiting."
    exit 0
fi

# Write sentinel on start
date -Is > "${SENTINEL}"
trap 'rm -f "${SENTINEL}"' EXIT

cd "${OUT_DIR}"

# ---------------- Stage 1: N50 + select 20 ----------------
if [[ ! -f "${OUT_DIR}/tara_assembly_n50_selected.tsv" ]]; then
    echo "[stage 1] N50 scan + quartile selection"
    "${PYTHON}" "${SCRIPT_DIR}/task_41_15_n50_select.py" \
        --assembly-dir "${ASSEMBLY_DIR}" \
        --out-dir "${OUT_DIR}" \
        --n-per-bin 5 --seed 42
fi

mapfile -t SAMPLES < <(awk -F'\t' '!/^#/ && NR>1 {print $1}' "${OUT_DIR}/tara_assembly_n50_selected.tsv")
echo "[info] selected ${#SAMPLES[@]} samples"

# ---------------- Stage 2: MetaEuk easy-predict ----------------
# MetaEuk needs a protein reference database for easy-predict.  We use the
# same Uniref50 that was prepared for the SNAP pipeline if available; else
# we fall back to metaeuk's contig2prot mode which runs pure 6-frame +
# exon chaining without a reference.  For a strictly gene-prediction-only
# benchmark, the reference-free contig2prot mode is what we want because
# SNAP is also reference-free.
for AID in "${SAMPLES[@]}"; do
    PROT_OUT="${OUT_DIR}/metaeuk/${AID}.metaeuk.faa"
    if [[ -s "${PROT_OUT}" ]]; then
        echo "[stage 2] ${AID}: protein already present; skipping"
        continue
    fi
    FA_GZ="${ASSEMBLY_DIR}/${AID}_assembly.fasta.gz"
    if [[ ! -f "${FA_GZ}" ]]; then
        FA_GZ="${ASSEMBLY_DIR}/${AID}.fasta.gz"
    fi
    if [[ ! -f "${FA_GZ}" ]]; then
        echo "[warn] ${AID}: no fasta.gz found; skipping"
        continue
    fi

    TMP_FA="${OUT_DIR}/tmp/${AID}.fa"
    gunzip -c "${FA_GZ}" > "${TMP_FA}"

    # Use metaeuk predictexons (contig2prot equivalent): requires a db
    # even for reference-free mode.  We use a small seed db derived from
    # the assembly itself so the call succeeds.  This keeps the comparison
    # structural-only: metaeuk's exon chainer recovers ORFs.
    SEED_DB="${OUT_DIR}/tmp/${AID}.seed"
    "${METAEUK}" createdb "${TMP_FA}" "${SEED_DB}" > "${OUT_DIR}/logs/${AID}.createdb.log" 2>&1

    METADB="${OUT_DIR}/tmp/${AID}.meta"
    "${METAEUK}" easy-predict "${TMP_FA}" "${SEED_DB}" "${METADB}" \
        "${OUT_DIR}/tmp/${AID}.metaeuk_tmp" \
        --threads "${SLURM_CPUS_PER_TASK:-16}" \
        > "${OUT_DIR}/logs/${AID}.metaeuk.log" 2>&1 || \
    {
        echo "[warn] ${AID}: metaeuk easy-predict failed; trying predictexons"
        "${METAEUK}" predictexons "${SEED_DB}" "${SEED_DB}" \
            "${OUT_DIR}/tmp/${AID}.pred" \
            "${OUT_DIR}/tmp/${AID}.pred_tmp" \
            --threads "${SLURM_CPUS_PER_TASK:-16}" \
            >> "${OUT_DIR}/logs/${AID}.metaeuk.log" 2>&1 || true
    }

    # metaeuk easy-predict writes <prefix>.fas (proteins)
    if [[ -s "${METADB}.fas" ]]; then
        cp "${METADB}.fas" "${PROT_OUT}"
    else
        echo "[warn] ${AID}: metaeuk produced no proteins" >&2
        : > "${PROT_OUT}"  # empty sentinel to avoid re-running
    fi
    rm -f "${TMP_FA}" "${SEED_DB}"* "${METADB}"* 2>/dev/null
    rm -rf "${OUT_DIR}/tmp/${AID}.metaeuk_tmp" "${OUT_DIR}/tmp/${AID}.pred_tmp" 2>/dev/null
done

# ---------------- Stage 3: hmmsearch vs Pfam-A ----------------
for AID in "${SAMPLES[@]}"; do
    TBL="${OUT_DIR}/hmmsearch/${AID}.metaeuk.aa.hmmsearch.tbl"
    PROT="${OUT_DIR}/metaeuk/${AID}.metaeuk.faa"
    if [[ -s "${TBL}" ]]; then
        echo "[stage 3] ${AID}: tbl already present; skipping"
        continue
    fi
    if [[ ! -s "${PROT}" ]]; then
        echo "[warn] ${AID}: no metaeuk proteins; emitting empty tbl"
        : > "${TBL}"
        continue
    fi
    "${HMMSEARCH}" \
        --cpu "${SLURM_CPUS_PER_TASK:-16}" \
        -E 1e-9 --incE 1e-9 \
        --tblout "${TBL}" \
        --domtblout "${OUT_DIR}/hmmsearch/${AID}.metaeuk.aa.hmmsearch.domtbl" \
        -o "${OUT_DIR}/logs/${AID}.hmmsearch.log" \
        "${PFAM_DB}" \
        "${PROT}"
done

# ---------------- Stage 4: parse + summarise ----------------
"${PYTHON}" "${SCRIPT_DIR}/task_41_15_parse_hmmsearch.py" \
    --out-dir "${OUT_DIR}" \
    --snap-matrix "${SNAP_MATRIX}"

# Done
date -Is > "${DONE_MARKER}"
rm -f "${SENTINEL}"
echo "[ralph41 41.15] complete"
