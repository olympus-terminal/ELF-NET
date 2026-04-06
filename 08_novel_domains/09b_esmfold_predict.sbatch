#!/bin/bash
#SBATCH --job-name=nd_09b_esmfold
#SBATCH --output=novel_domains/logs/09b_esmfold_%j.out
#SBATCH --error=novel_domains/logs/09b_esmfold_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=compute

# =============================================================================
# 09b: ESMFold Structure Prediction for Top 10 Novel Domains
# =============================================================================
#
# Uses the ESMFold API (api.esmatlas.com) to predict structures.
# No GPU or local model needed — structures are predicted server-side.
#
# Sequences >400 aa are truncated to the first 400 residues (API limit).
# Stop codon characters (*) are removed before submission.
#
# Input:  novel_domains/results/top10_for_structure.fasta
# Output: novel_domains/results/structures/{domain_id}.pdb
#         novel_domains/results/structures/plddt_summary.tsv
#         novel_domains/results/structures/plddt_per_residue/{domain_id}_plddt.tsv
# =============================================================================

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
export OPENBLAS_NUM_THREADS=1
mkdir -p $TMPDIR

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

set -euo pipefail

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains"
INPUT_FASTA="${BASE}/results/top10_for_structure.fasta"
PDB_DIR="${BASE}/results/structures"
mkdir -p "${PDB_DIR}" "${PDB_DIR}/plddt_per_residue"

echo "=============================================="
echo "  09b: ESMFold Structure Prediction (API)"
echo "  Top 10 Novel Domains"
echo "=============================================="
echo "  Input: ${INPUT_FASTA}"
echo "  Output: ${PDB_DIR}/"
echo "  Date: $(date)"
echo ""

if [[ ! -f "${INPUT_FASTA}" ]]; then
    echo "ERROR: Input FASTA not found: ${INPUT_FASTA}"
    echo "Run 09_extract_top_reps.py first."
    exit 1
fi

# Test API connectivity
echo "  Testing API connectivity..."
if ! curl -sf --max-time 10 -X POST --data 'MKTIIALSYIFCLVFA' https://api.esmatlas.com/foldSequence/v1/pdb/ > /dev/null 2>&1; then
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

fasta_path = sys.argv[1]
pdb_dir = sys.argv[2]

ESMFOLD_API = "https://api.esmatlas.com/foldSequence/v1/pdb/"
MAX_SEQ_LEN = 400  # API limit

# Parse FASTA
sequences = {}
current_id = None
current_seq = []
with open(fasta_path) as f:
    for line in f:
        if line.startswith(">"):
            if current_id:
                sequences[current_id] = "".join(current_seq)
            parts = line.strip().lstrip(">").split()
            current_id = parts[0]
            current_seq = []
        else:
            current_seq.append(line.strip())
    if current_id:
        sequences[current_id] = "".join(current_seq)

print(f"  Loaded {len(sequences)} sequences")
for sid, seq in sequences.items():
    print(f"    {sid[:70]}: {len(seq)} aa")

def clean_sequence(seq):
    """Remove stop codons (*) and non-standard characters."""
    seq = seq.replace("*", "")
    # Keep only standard amino acids + common ambiguity codes
    seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWYBXZJ]', '', seq.upper())
    return seq

def predict_structure(seq, timeout=300, max_retries=3):
    """Submit sequence to ESMFold API with retry logic."""
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
            raise e
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"      Retry {attempt+1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise e

def extract_plddt(pdb_string):
    """Extract per-residue pLDDT from B-factor column of CA atoms."""
    plddts = []
    for line in pdb_string.split("\n"):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                bfactor = float(line[60:66].strip())
                plddts.append(bfactor)
            except (ValueError, IndexError):
                pass
    return plddts

# Predict structures via API
plddt_summary = []

for sid, raw_seq in sequences.items():
    seq = clean_sequence(raw_seq)
    truncated = False
    original_len = len(seq)

    if len(seq) > MAX_SEQ_LEN:
        truncated = True
        seq = seq[:MAX_SEQ_LEN]
        print(f"\n  Predicting: {sid[:60]} ({original_len} aa, truncated to {MAX_SEQ_LEN})...",
              flush=True)
    else:
        print(f"\n  Predicting: {sid[:60]} ({len(seq)} aa)...", flush=True)

    if len(seq) < 20:
        print(f"    SKIP: sequence too short ({len(seq)} aa < 20 min)")
        plddt_summary.append((sid, original_len, -1, -1, -1, "too_short"))
        continue

    try:
        pdb_string = predict_structure(seq)

        # Write PDB
        safe_name = sid.replace("/", "_").replace(" ", "_")
        pdb_path = os.path.join(pdb_dir, f"{safe_name}.pdb")
        with open(pdb_path, "w") as f:
            f.write(pdb_string)

        # Extract pLDDT
        plddts = extract_plddt(pdb_string)

        if plddts:
            arr = np.array(plddts)
            # API returns pLDDT in 0-1 scale; convert to 0-100 for consistency
            if arr.max() <= 1.0:
                arr = arr * 100.0
            mean_plddt = float(np.mean(arr))
            median_plddt = float(np.median(arr))
            min_plddt = float(np.min(arr))
            pct_above_70 = float(np.sum(arr > 70)) / len(arr) * 100

            status = "success" if not truncated else f"success_truncated_{MAX_SEQ_LEN}"
            print(f"    -> {pdb_path}")
            print(f"    pLDDT: mean={mean_plddt:.1f}, median={median_plddt:.1f}, "
                  f"min={min_plddt:.1f}, >70: {pct_above_70:.1f}%")
            if truncated:
                print(f"    NOTE: Only first {MAX_SEQ_LEN} of {original_len} residues predicted")
            plddt_summary.append((sid, original_len, mean_plddt, median_plddt,
                                  pct_above_70, status))

            # Write per-residue pLDDT
            per_res_dir = os.path.join(pdb_dir, "plddt_per_residue")
            per_res_path = os.path.join(per_res_dir, f"{safe_name}_plddt.tsv")
            with open(per_res_path, "w") as f:
                f.write("residue\tplddt\n")
                for i, v in enumerate(arr, 1):
                    f.write(f"{i}\t{v:.2f}\n")
        else:
            print(f"    WARNING: no pLDDT values extracted from PDB")
            plddt_summary.append((sid, original_len, -1, -1, -1, "no_plddt"))

    except urllib.error.HTTPError as e:
        msg = ""
        try:
            msg = e.read().decode("utf-8")[:200]
        except:
            msg = str(e)
        print(f"    HTTP ERROR {e.code}: {msg}")
        plddt_summary.append((sid, original_len, -1, -1, -1, f"http_error_{e.code}"))

    except Exception as e:
        print(f"    ERROR: {e}")
        plddt_summary.append((sid, original_len, -1, -1, -1, f"error: {e}"))

    # Rate limit: wait between requests
    time.sleep(5)

# Write pLDDT summary
summary_path = os.path.join(pdb_dir, "plddt_summary.tsv")
with open(summary_path, "w") as f:
    f.write("domain\tseq_length\tmean_plddt\tmedian_plddt\tpct_above_70\tstatus\n")
    for sid, slen, mean_p, med_p, pct70, status in plddt_summary:
        f.write(f"{sid}\t{slen}\t{mean_p:.1f}\t{med_p:.1f}\t{pct70:.1f}\t{status}\n")
print(f"\n  pLDDT summary: {summary_path}")

# Overall summary
successful = [x for x in plddt_summary if x[5].startswith("success")]
if successful:
    mean_all = np.mean([x[2] for x in successful])
    print(f"\n  Overall: {len(successful)}/{len(plddt_summary)} structures predicted")
    print(f"  Grand mean pLDDT: {mean_all:.1f}")
    min_mean = min(x[2] for x in successful)
    max_mean = max(x[2] for x in successful)
    print(f"  pLDDT range: {min_mean:.1f} -- {max_mean:.1f}")
    n_confident = sum(1 for x in successful if x[2] > 70)
    print(f"  Domains with mean pLDDT > 70: {n_confident}/{len(successful)}")
    n_truncated = sum(1 for x in successful if "truncated" in x[5])
    if n_truncated:
        print(f"  Truncated to {MAX_SEQ_LEN} aa: {n_truncated}/{len(successful)}")
else:
    print("\n  WARNING: No structures successfully predicted")
PYEOF

echo ""
echo "Done: $(date)"
echo "PDB files in: ${PDB_DIR}/"
ls -la "${PDB_DIR}"/*.pdb 2>/dev/null || echo "  (no PDB files found)"
