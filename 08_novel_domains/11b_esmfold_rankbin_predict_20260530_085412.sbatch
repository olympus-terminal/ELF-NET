#!/bin/bash
#SBATCH --job-name=nd_11b_esmfold_rankbin
#SBATCH --output=novel_domains/logs/11b_esmfold_rankbin_%j.out
#SBATCH --error=novel_domains/logs/11b_esmfold_rankbin_%j.err
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=compute

# =============================================================================
# 11b: ESMFold Structure Prediction for Rank-Binned Novel Domains (~3,600 seqs)
# =============================================================================
#
# Uses the ESMFold API (api.esmatlas.com) to predict structures for ~3,600
# novel domains sampled across 10 rank bins (by environmental coupling strength).
#
# Sequences >400 aa are truncated to the first 400 residues (API limit).
# Stop codon characters (*) are removed before submission.
#
# Checkpoint: writes per-sequence results as they complete. On restart,
# skips sequences with existing PDB files.
#
# Input:  novel_domains/results/esmfold_rankbin/rankbin_sequences.fasta
# Output: novel_domains/results/esmfold_rankbin/structures/{domain_id}.pdb
#         novel_domains/results/esmfold_rankbin/structures/plddt_summary.tsv
#         novel_domains/results/esmfold_rankbin/structures/plddt_per_residue/{domain_id}_plddt.tsv
# =============================================================================

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
export OPENBLAS_NUM_THREADS=1
unset NETWORKX_AUTOMATIC_BACKENDS
unset NETWORKX_BACKEND_PRIORITY
unset NETWORKX_FALLBACK_TO_NX
mkdir -p $TMPDIR

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

set -euo pipefail

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains"
INPUT_FASTA="${BASE}/results/esmfold_rankbin/rankbin_sequences.fasta"
PDB_DIR="${BASE}/results/esmfold_rankbin/structures"
mkdir -p "${PDB_DIR}" "${PDB_DIR}/plddt_per_residue"

echo "=============================================="
echo "  11b: ESMFold Rank-Bin Structure Prediction"
echo "=============================================="
echo "  Input: ${INPUT_FASTA}"
echo "  Output: ${PDB_DIR}/"
echo "  Date: $(date)"
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo ""

if [[ ! -f "${INPUT_FASTA}" ]]; then
    echo "ERROR: Input FASTA not found: ${INPUT_FASTA}"
    echo "Run 11_rankbin_esmfold_sample_20260530_085412.py first."
    exit 1
fi

# Log sequence count
N_SEQ=$(grep -c "^>" "${INPUT_FASTA}" || true)
echo "  Total sequences: ${N_SEQ}"

# Test API connectivity
echo "  Testing API connectivity..."
if ! curl -sf --max-time 30 -X POST --data 'MKTIIALSYIFCLVFA' https://api.esmatlas.com/foldSequence/v1/pdb/ > /dev/null 2>&1; then
    echo "ERROR: Cannot reach ESMFold API at api.esmatlas.com"
    exit 1
fi
echo "  API reachable"
echo ""

python3 -u - "${INPUT_FASTA}" "${PDB_DIR}" << 'PYEOF'
import sys
import os
import time
import urllib.request
import urllib.error
import numpy as np
import re
from datetime import datetime

fasta_path = sys.argv[1]
pdb_dir = sys.argv[2]

ESMFOLD_API = "https://api.esmatlas.com/foldSequence/v1/pdb/"
MAX_SEQ_LEN = 400
RATE_LIMIT_SEC = 5
MAX_RETRIES = 5
BACKOFF_BASE = 15

summary_path = os.path.join(pdb_dir, "plddt_summary.tsv")

# Load existing results for checkpoint/resume
existing = set()
if os.path.exists(summary_path):
    with open(summary_path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("domain"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 1:
                existing.add(parts[0])
    print(f"  Checkpoint: {len(existing)} sequences already processed")

# Parse FASTA
sequences = []
current_id = None
current_seq = []
with open(fasta_path) as f:
    for line in f:
        if line.startswith(">"):
            if current_id:
                sequences.append((current_id, "".join(current_seq)))
            parts = line.strip().lstrip(">").split()
            current_id = parts[0]
            current_seq = []
        else:
            current_seq.append(line.strip())
    if current_id:
        sequences.append((current_id, "".join(current_seq)))

print(f"  Loaded {len(sequences)} sequences from FASTA")

# Filter out already-processed
pending = [(sid, seq) for sid, seq in sequences if sid not in existing]
print(f"  Pending: {len(pending)} sequences to predict")

if not pending:
    print("  All sequences already processed — nothing to do")
    sys.exit(0)

def clean_sequence(seq):
    seq = seq.replace("*", "")
    seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWYBXZJ]', '', seq.upper())
    return seq

def predict_structure(seq, timeout=300, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                ESMFOLD_API,
                data=seq.encode("utf-8"),
                headers={"Content-Type": "text/plain"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"      Rate limited (429), waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
            elif e.code >= 500:
                wait = BACKOFF_BASE * (attempt + 1)
                print(f"      Server error ({e.code}), waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
            else:
                raise e
        except Exception as e:
            if attempt < max_retries - 1:
                wait = BACKOFF_BASE * (attempt + 1)
                print(f"      Retry {attempt+1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise e
    raise RuntimeError(f"Failed after {max_retries} retries")

def extract_plddt(pdb_string):
    plddts = []
    for line in pdb_string.split("\n"):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                bfactor = float(line[60:66].strip())
                plddts.append(bfactor)
            except (ValueError, IndexError):
                pass
    return plddts

# Open summary file for appending
write_header = not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0
summary_fh = open(summary_path, "a")
if write_header:
    summary_fh.write("domain\tseq_length\tmean_plddt\tmedian_plddt\tpct_above_70\tstatus\n")
    summary_fh.flush()

n_success = 0
n_fail = 0
start_time = time.time()

for idx, (sid, raw_seq) in enumerate(pending):
    seq = clean_sequence(raw_seq)
    truncated = False
    original_len = len(seq)

    if len(seq) > MAX_SEQ_LEN:
        truncated = True
        seq = seq[:MAX_SEQ_LEN]

    elapsed = time.time() - start_time
    rate = (idx / elapsed * 3600) if elapsed > 0 and idx > 0 else 0
    remaining = (len(pending) - idx) / (rate / 3600) if rate > 0 else 0

    trunc_note = f", truncated to {MAX_SEQ_LEN}" if truncated else ""
    print(f"\n  [{idx+1}/{len(pending)}] {sid[:60]} ({original_len} aa{trunc_note})"
          f" [{rate:.0f}/hr, ~{remaining/3600:.1f}h remaining]", flush=True)

    if len(seq) < 20:
        print(f"    SKIP: sequence too short ({len(seq)} aa < 20 min)")
        summary_fh.write(f"{sid}\t{original_len}\t-1.0\t-1.0\t-1.0\ttoo_short\n")
        summary_fh.flush()
        continue

    try:
        pdb_string = predict_structure(seq)

        safe_name = sid.replace("/", "_").replace(" ", "_")
        pdb_path = os.path.join(pdb_dir, f"{safe_name}.pdb")
        with open(pdb_path, "w") as f:
            f.write(pdb_string)

        plddts = extract_plddt(pdb_string)

        if plddts:
            arr = np.array(plddts)
            if arr.max() <= 1.0:
                arr = arr * 100.0
            mean_plddt = float(np.mean(arr))
            median_plddt = float(np.median(arr))
            pct_above_70 = float(np.sum(arr > 70)) / len(arr) * 100

            status = "success" if not truncated else f"success_truncated_{MAX_SEQ_LEN}"
            print(f"    pLDDT: mean={mean_plddt:.1f}, median={median_plddt:.1f}, >70: {pct_above_70:.1f}%")

            per_res_path = os.path.join(pdb_dir, "plddt_per_residue", f"{safe_name}_plddt.tsv")
            with open(per_res_path, "w") as f:
                f.write("residue\tplddt\n")
                for i, v in enumerate(arr, 1):
                    f.write(f"{i}\t{v:.2f}\n")

            summary_fh.write(f"{sid}\t{original_len}\t{mean_plddt:.1f}\t{median_plddt:.1f}\t{pct_above_70:.1f}\t{status}\n")
            n_success += 1
        else:
            print(f"    WARNING: no pLDDT values extracted")
            summary_fh.write(f"{sid}\t{original_len}\t-1.0\t-1.0\t-1.0\tno_plddt\n")
            n_fail += 1

    except urllib.error.HTTPError as e:
        msg = ""
        try:
            msg = e.read().decode("utf-8")[:200]
        except:
            msg = str(e)
        print(f"    HTTP ERROR {e.code}: {msg}")
        summary_fh.write(f"{sid}\t{original_len}\t-1.0\t-1.0\t-1.0\thttp_error_{e.code}\n")
        n_fail += 1

    except Exception as e:
        print(f"    ERROR: {e}")
        summary_fh.write(f"{sid}\t{original_len}\t-1.0\t-1.0\t-1.0\terror\n")
        n_fail += 1

    summary_fh.flush()
    time.sleep(RATE_LIMIT_SEC)

summary_fh.close()

total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"  COMPLETE")
print(f"  Success: {n_success}, Failed: {n_fail}")
print(f"  Wall time: {total_time/3600:.1f} hours")
print(f"  Summary: {summary_path}")
print(f"{'='*60}")
PYEOF

echo ""
echo "Done: $(date)"
