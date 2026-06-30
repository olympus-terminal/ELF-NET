#!/bin/bash
#SBATCH --job-name=dark_extract
#SBATCH --output=logs/novel_domains/dark_extract_%j.out
#SBATCH --error=logs/novel_domains/dark_extract_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=compute

# =============================================================================
# Extract Dark Proteome — Per-Sample Filtered FASTA
# =============================================================================
#
# For each sample: parse hmmsearch .tbl, identify proteins with zero Pfam-A
# hits, extract sequences, filter (length 30-5000 aa, low-complexity removal),
# write per-sample dark FASTA.
#
# Output:
#   03_analyses/novel_domains/dark_fasta/{sample}.dark.fa  (per-sample)
#   03_analyses/novel_domains/dark_extraction_stats.tsv     (summary)
#   03_analyses/novel_domains/filter_stats.tsv              (filter details)
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   mkdir -p logs/novel_domains
#   sbatch MANUSCRIPT/scripts/novel_domains/extract_dark_proteome.sh
#
# =============================================================================

set -euo pipefail

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
mkdir -p $TMPDIR
export PYTHONUSERBASE=/scratch/drn2/newhome/.local

module purge
module load gcc/13.2.0

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

echo "========================================"
echo "Extract Dark Proteome (per-sample)"
echo "========================================"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Date:    $(date)"
echo "Host:    $(hostname)"
echo ""

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
FASTA_DIR="${BASE}/03_analyses/algae_proteins"
HMMSEARCH_DIR="${BASE}/03_analyses/hmmsearch_results"
OUT_DIR="${BASE}/03_analyses/novel_domains"
DARK_IDS_DIR="${OUT_DIR}/dark_ids"
DARK_FASTA_DIR="${OUT_DIR}/dark_fasta"

mkdir -p "${DARK_IDS_DIR}" "${DARK_FASTA_DIR}"

# ---- Build sample list ----
SAMPLE_LIST="${OUT_DIR}/sample_list.txt"
realpath "${FASTA_DIR}"/*.fa | xargs -n1 basename | sed 's/\.fa$//' | sort > "${SAMPLE_LIST}"
TOTAL_SAMPLES=$(wc -l < "${SAMPLE_LIST}")
echo "Total samples: ${TOTAL_SAMPLES}"
echo ""

# ---- Extract and filter per sample ----
STATS_FILE="${OUT_DIR}/dark_extraction_stats.tsv"
FILTER_FILE="${OUT_DIR}/filter_stats.tsv"
echo -e "sample\ttotal_seqs\tpfam_hit_seqs\tdark_seqs\tpct_dark" > "${STATS_FILE}"
echo -e "sample\tdark_raw\tpassed\trejected_short\trejected_long\trejected_lowcx" > "${FILTER_FILE}"

TOTAL_ALL=0
TOTAL_DARK=0
TOTAL_PASSED=0
PROCESSED=0

while IFS= read -r SAMPLE; do
    FASTA="${FASTA_DIR}/${SAMPLE}.algae.fa"
    TBL="${HMMSEARCH_DIR}/${SAMPLE}.hmmsearch.tbl"

    if [ ! -f "$FASTA" ]; then
        echo "WARN: Missing FASTA for ${SAMPLE}, skipping"
        continue
    fi

    # All protein IDs from FASTA
    ALL_IDS=$(grep '^>' "$FASTA" | sed 's/^>//' | awk '{print $1}' | sort -u)
    N_ALL=$(echo "$ALL_IDS" | wc -l)

    # Pfam-hit IDs from tblout
    if [ -f "$TBL" ]; then
        PFAM_IDS=$(grep -v '^#' "$TBL" | awk '{print $1}' | sort -u)
        N_PFAM=$(echo "$PFAM_IDS" | grep -c . || true)
    else
        PFAM_IDS=""
        N_PFAM=0
    fi

    # Set difference: dark = all - pfam
    DARK_IDS_FILE="${DARK_IDS_DIR}/${SAMPLE}.dark_ids.txt"
    if [ "$N_PFAM" -gt 0 ]; then
        comm -23 <(echo "$ALL_IDS") <(echo "$PFAM_IDS") > "${DARK_IDS_FILE}"
    else
        echo "$ALL_IDS" > "${DARK_IDS_FILE}"
    fi

    N_DARK=$(wc -l < "${DARK_IDS_FILE}")
    if [ "$N_ALL" -gt 0 ]; then
        PCT_DARK=$(awk "BEGIN {printf \"%.1f\", 100 * ${N_DARK} / ${N_ALL}}")
    else
        PCT_DARK="0.0"
    fi

    echo -e "${SAMPLE}\t${N_ALL}\t${N_PFAM}\t${N_DARK}\t${PCT_DARK}" >> "${STATS_FILE}"
    TOTAL_ALL=$((TOTAL_ALL + N_ALL))
    TOTAL_DARK=$((TOTAL_DARK + N_DARK))

    # Extract dark sequences with seqtk, then filter in-place
    DARK_FA="${DARK_FASTA_DIR}/${SAMPLE}.dark.fa"
    if [ "$N_DARK" -gt 0 ]; then
        DARK_RAW="${DARK_FASTA_DIR}/${SAMPLE}.dark.raw.fa"
        seqtk subseq "$FASTA" "${DARK_IDS_FILE}" > "${DARK_RAW}"

        # Filter: length 30-5000, remove low-complexity/poly-X
        python3 -c "
import sys, math
MIN_LEN, MAX_LEN, ENT_THRESH = 30, 5000, 2.0
def entropy(s):
    n = len(s)
    if n == 0: return 0.0
    counts = {}
    for c in s: counts[c] = counts.get(c, 0) + 1
    return -sum((v/n)*math.log2(v/n) for v in counts.values())
def poly_x(s, t=0.9):
    if not s: return True
    return max(s.count(c) for c in set(s)) / len(s) > t
hdr, seq, ns, np, nsh, nl, nc = None, [], 0, 0, 0, 0, 0
out = open(sys.argv[2], 'w')
def flush():
    global ns, np, nsh, nl, nc
    if hdr is None: return
    s = ''.join(seq).upper()
    ns += 1
    if len(s) < MIN_LEN: nsh += 1
    elif len(s) > MAX_LEN: nl += 1
    elif entropy(s) < ENT_THRESH or poly_x(s): nc += 1
    else: out.write(f'{hdr}\n{s}\n'); np += 1
with open(sys.argv[1]) as f:
    for line in f:
        line = line.rstrip()
        if line.startswith('>'):
            flush(); hdr = line; seq = []
        else: seq.append(line)
    flush()
out.close()
print(f'{sys.argv[3]}\t{ns}\t{np}\t{nsh}\t{nl}\t{nc}')
" "${DARK_RAW}" "${DARK_FA}" "${SAMPLE}"  >> "${FILTER_FILE}"

        rm -f "${DARK_RAW}"

        N_PASSED=$(grep -c '^>' "${DARK_FA}" || true)
        TOTAL_PASSED=$((TOTAL_PASSED + N_PASSED))
    else
        : > "${DARK_FA}"
        echo -e "${SAMPLE}\t0\t0\t0\t0\t0" >> "${FILTER_FILE}"
    fi

    PROCESSED=$((PROCESSED + 1))
    if (( PROCESSED % 100 == 0 )); then
        echo "  Processed ${PROCESSED}/${TOTAL_SAMPLES} samples..."
    fi
done < "${SAMPLE_LIST}"

echo ""
echo "========================================"
echo "Extraction complete"
echo "========================================"
echo "Samples processed: ${PROCESSED}"
echo "Total proteins:    ${TOTAL_ALL}"
echo "Total dark (raw):  ${TOTAL_DARK}"
echo "Total dark (filtered): ${TOTAL_PASSED}"
if [ "$TOTAL_ALL" -gt 0 ]; then
    echo "Dark fraction (raw): $(awk "BEGIN {printf \"%.1f\", 100 * ${TOTAL_DARK} / ${TOTAL_ALL}}")%"
fi
echo ""
echo "Per-sample FASTA: ${DARK_FASTA_DIR}/"
echo "Extraction stats: ${STATS_FILE}"
echo "Filter stats:     ${FILTER_FILE}"
echo ""
echo "Done: $(date)"
