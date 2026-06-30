#!/bin/bash
#SBATCH --job-name=ralph41_la4sr_fp
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/la4sr_benchmark/slurm_%j.out
#SBATCH --error=/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/la4sr_benchmark/slurm_%j.err

###############################################################################
# Ralph41 Task 41.18 - LA4SR inference on 40 non-algal eukaryotic proteomes
#
# Invocation (from project root on Jubail):
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR && \
#       sbatch MANUSCRIPT/scripts/ralph41/task_41_18_la4sr_benchmark.sbatch
#
# Inputs  (already staged by Task 41.14 + rsync step in this iteration):
#   - 40 proteome FASTAs at 03_analyses/la4sr_benchmark/proteomes/*.fa
#   - tools/la4sr/la4sr_sp2.sif
#   - tools/la4sr/ckpt.pt + meta.pkl
#   - tools/la4sr/infer_TI-inc-algaGPT.py
#
# Outputs:
#   - 03_analyses/la4sr_benchmark/results/*.tsv  (one per proteome)
#   - 03_analyses/la4sr_benchmark/.running        (sentinel, removed on exit)
#   - 03_analyses/la4sr_benchmark/.done           (written on clean success)
#
# Idempotence: each per-proteome TSV is skipped if it already exists.
# Safe to re-run; the Python parser (Task 41.18 local step) consumes the TSVs.
###############################################################################

set -euo pipefail

BASE_DIR="/scratch/drn2/PROJECTS/TARA-LA4SR"
BENCH_DIR="${BASE_DIR}/03_analyses/la4sr_benchmark"
PROTEOME_DIR="${BENCH_DIR}/proteomes"
RESULTS_DIR="${BENCH_DIR}/results"
LOGS_DIR="${BENCH_DIR}/logs"
LA4SR_DIR="${BASE_DIR}/tools/la4sr"
SIF="${LA4SR_DIR}/la4sr_sp2.sif"
INFER="${LA4SR_DIR}/infer_TI-inc-algaGPT.py"
SENTINEL="${BENCH_DIR}/.running"
DONE_FLAG="${BENCH_DIR}/.done"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

# ---------------------- sentinel management --------------------------------
# Remove sentinel on ANY exit (success, failure, cancel). This ensures
# crashed jobs never leave a stale sentinel that blocks future submissions.
cleanup() {
  rc=$?
  rm -f "${SENTINEL}"
  if [[ ${rc} -eq 0 ]]; then
    echo "[$(date -Iseconds)] exit 0 -> writing ${DONE_FLAG}"
    echo "completed_at: $(date -Iseconds)" > "${DONE_FLAG}"
    echo "slurm_job_id: ${SLURM_JOB_ID:-unknown}" >> "${DONE_FLAG}"
  else
    echo "[$(date -Iseconds)] exit ${rc}; sentinel removed; no .done written" >&2
  fi
  exit ${rc}
}
trap cleanup EXIT

echo "[$(date -Iseconds)] started SLURM job ${SLURM_JOB_ID:-unknown} on $(hostname)"
echo "slurm_job_id: ${SLURM_JOB_ID:-unknown}" > "${SENTINEL}"
echo "started_at: $(date -Iseconds)" >> "${SENTINEL}"
echo "hostname: $(hostname)" >> "${SENTINEL}"

# ---------------------- sanity checks --------------------------------------
for f in "${SIF}" "${INFER}" "${LA4SR_DIR}/ckpt.pt" "${LA4SR_DIR}/meta.pkl"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERR] missing required file: ${f}" >&2
    exit 10
  fi
done

n_fastas=$(ls "${PROTEOME_DIR}"/*.fa 2>/dev/null | wc -l)
if [[ ${n_fastas} -lt 40 ]]; then
  echo "[ERR] expected >=40 proteomes in ${PROTEOME_DIR}; got ${n_fastas}" >&2
  exit 11
fi
echo "[$(date -Iseconds)] found ${n_fastas} proteome FASTAs in ${PROTEOME_DIR}"

# ---------------------- GPU report -----------------------------------------
nvidia-smi 2>&1 | head -20 || echo "[WARN] nvidia-smi unavailable"

# ---------------------- HuggingFace cache ----------------------------------
HF_CACHE="${LA4SR_DIR}/cache"
mkdir -p "${HF_CACHE}"

# ---------------------- per-proteome inference loop ------------------------
n_done=0
n_skipped=0
n_failed=0
for fasta in "${PROTEOME_DIR}"/*.fa; do
  basename=$(basename "${fasta}" .fa)
  out_tsv="${RESULTS_DIR}/${basename}.tsv"
  log_file="${LOGS_DIR}/${basename}.log"

  if [[ -f "${out_tsv}" && -s "${out_tsv}" ]]; then
    n_proteins=$(grep -c '^>' "${fasta}" || echo 0)
    n_rows=$(($(wc -l < "${out_tsv}") - 1))
    if [[ ${n_rows} -ge 1 ]]; then
      echo "[$(date -Iseconds)] SKIP ${basename}: ${n_rows} rows already present"
      n_skipped=$((n_skipped + 1))
      continue
    fi
  fi

  echo "[$(date -Iseconds)] RUN ${basename} (proteins=$(grep -c '^>' "${fasta}"))"

  # Bind fasta read-only into the container at /input.fasta, results dir at
  # /results, la4sr dir as /workdir. Use --nv for GPU passthrough.
  singularity exec --nv \
    -B "${fasta}:/input.fasta:ro" \
    -B "${LA4SR_DIR}:/workdir" \
    -B "${RESULTS_DIR}:/results" \
    -B "${HF_CACHE}:${HF_CACHE}" \
    --env TRANSFORMERS_CACHE="${HF_CACHE}" \
    "${SIF}" \
    bash -c "cd /workdir && python3 infer_TI-inc-algaGPT.py \
             --init_from resume \
             --out_dir /workdir \
             --device cuda \
             --dtype float16 \
             --seed 1337 \
             --max_new_tokens 14 \
             --temperature 0.1 \
             --top_k 10 \
             /input.fasta \
             -o /results/${basename}.tsv" \
    > "${log_file}" 2>&1 \
    && n_done=$((n_done + 1)) \
    || { echo "[ERR] inference failed for ${basename}; see ${log_file}" >&2; n_failed=$((n_failed + 1)); }
done

echo "[$(date -Iseconds)] loop complete: done=${n_done} skipped=${n_skipped} failed=${n_failed} total=${n_fastas}"
if [[ ${n_failed} -gt 0 ]]; then
  echo "[ERR] ${n_failed} proteomes failed inference" >&2
  exit 20
fi

# Count rows across all result TSVs
total_rows=0
for tsv in "${RESULTS_DIR}"/*.tsv; do
  rows=$(($(wc -l < "${tsv}") - 1))
  total_rows=$((total_rows + rows))
done
echo "[$(date -Iseconds)] total inferred sequences across 40 proteomes: ${total_rows}"
echo "[$(date -Iseconds)] sbatch completed successfully"
