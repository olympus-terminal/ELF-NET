#!/bin/bash
#SBATCH --job-name=pfam_dens
#SBATCH --output=logs/dark_proteome/density_%j.out
#SBATCH --error=logs/dark_proteome/density_%j.err
#SBATCH --cpus-per-task=28
#SBATCH --mem=90G
#SBATCH --time=24:00:00
#SBATCH --partition=compute

# =============================================================================
# Per-sample Pfam density — single-node parallel version
# =============================================================================
#
# Runs all 2,044 samples on one node using xargs -P (28 cores).
# Each sample takes ~10-30s (awk on FASTA + .tbl), total ~30-60 min.
#
# Provenance:
#   Script: scripts/dark_proteome/02b_pfam_density_single.sbatch
#   Generated: 2026-03-02
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   mkdir -p logs/dark_proteome
#   sbatch MANUSCRIPT/scripts/dark_proteome/02b_pfam_density_single.sbatch
#
# =============================================================================

set -euo pipefail

# ---- HPC environment ----
export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
export PYTHONUSERBASE=/scratch/drn2/newhome/.local
mkdir -p "$TMPDIR"

unset NETWORKX_BACKEND_CONFIG 2>/dev/null || true
unset NX_BACKEND_CONFIG 2>/dev/null || true

module purge
module load gcc/13.2.0

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

# ---- Paths ----
BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
FASTA_DIR="${BASE}/03_analyses/algae_proteins"
HMMSEARCH_DIR="${BASE}/03_analyses/hmmsearch_results"
OUT_DIR="${BASE}/03_analyses/dark_proteome"
DENSITY_DIR="${OUT_DIR}/pfam_density"
SAMPLE_LIST="${OUT_DIR}/sample_list.txt"

mkdir -p "${DENSITY_DIR}"

N_SAMPLES=$(wc -l < "${SAMPLE_LIST}")
echo "========================================"
echo "Pfam Density — Single Node (${N_SAMPLES} samples)"
echo "========================================"
echo "Date:  $(date)"
echo "Host:  $(hostname)"
echo "Cores: ${SLURM_CPUS_PER_TASK:-28}"
echo ""

# ---- Write the per-sample worker function ----
process_sample() {
    local SAMPLE="$1"
    local FASTA_DIR="$2"
    local HMMSEARCH_DIR="$3"
    local DENSITY_DIR="$4"

    # Skip if already done
    local OUTFILE="${DENSITY_DIR}/${SAMPLE}.pfam_density.tsv"
    if [[ -f "${OUTFILE}" ]]; then
        return 0
    fi

    # Locate FASTA
    local FASTA=""
    for ext in ".aa.algae.fa" ".aa.algal.fa" ".fa" ".faa"; do
        if [[ -f "${FASTA_DIR}/${SAMPLE}${ext}" ]]; then
            FASTA="${FASTA_DIR}/${SAMPLE}${ext}"
            break
        fi
    done

    if [[ -z "${FASTA}" ]]; then
        echo "SKIP ${SAMPLE}: FASTA not found" >&2
        return 1
    fi

    # Locate hmmsearch .tbl
    local TBL=""
    for ext in ".aa.hmmsearch.tbl" ".hmmsearch.tbl" ".tbl"; do
        if [[ -f "${HMMSEARCH_DIR}/${SAMPLE}${ext}" ]]; then
            TBL="${HMMSEARCH_DIR}/${SAMPLE}${ext}"
            break
        fi
    done

    # Compute FASTA stats (total proteins + total amino acids)
    read N_PROTEINS TOTAL_AA < <(
        awk '
        /^>/ { n_proteins++ ; next }
        { total_aa += length($0) }
        END { print n_proteins+0, total_aa+0 }
        ' "${FASTA}"
    )

    # Compute hmmsearch stats
    local N_DOMAIN_HITS=0
    local N_ANNOTATED=0
    local N_UNIQUE_PFAM=0

    if [[ -n "${TBL}" && -f "${TBL}" ]]; then
        read N_DOMAIN_HITS N_ANNOTATED N_UNIQUE_PFAM < <(
            awk '
            /^#/ { next }
            {
                n_hits++
                proteins[$1] = 1
                pfams[$3] = 1
            }
            END {
                n_prot = 0; for (p in proteins) n_prot++
                n_pfam = 0; for (f in pfams) n_pfam++
                print n_hits+0, n_prot+0, n_pfam+0
            }
            ' "${TBL}"
        )
    fi

    local N_DARK=$((N_PROTEINS - N_ANNOTATED))
    if [[ ${N_DARK} -lt 0 ]]; then N_DARK=0; fi

    # Derived metrics
    local PCT_DARK="0.0000"
    local PFAM_PER_ORF="0.0000"
    local PFAM_PER_1000AA="0.0000"

    if [[ ${N_PROTEINS} -gt 0 ]]; then
        PCT_DARK=$(awk "BEGIN {printf \"%.4f\", 100.0 * ${N_DARK} / ${N_PROTEINS}}")
        PFAM_PER_ORF=$(awk "BEGIN {printf \"%.4f\", ${N_DOMAIN_HITS} * 1.0 / ${N_PROTEINS}}")
    fi
    if [[ ${TOTAL_AA} -gt 0 ]]; then
        PFAM_PER_1000AA=$(awk "BEGIN {printf \"%.4f\", ${N_DOMAIN_HITS} * 1000.0 / ${TOTAL_AA}}")
    fi

    # Write output
    cat > "${OUTFILE}" <<EOF
sample	n_proteins	total_aa	n_annotated	n_dark	n_domain_hits	n_unique_pfam	pct_dark	pfam_per_orf	pfam_per_1000aa
${SAMPLE}	${N_PROTEINS}	${TOTAL_AA}	${N_ANNOTATED}	${N_DARK}	${N_DOMAIN_HITS}	${N_UNIQUE_PFAM}	${PCT_DARK}	${PFAM_PER_ORF}	${PFAM_PER_1000AA}
EOF

    echo "OK ${SAMPLE}: ${N_PROTEINS} proteins, ${PCT_DARK}% dark, ${PFAM_PER_ORF} pfam/orf"
}

export -f process_sample

# ---- Run in parallel with xargs ----
NCORES=${SLURM_CPUS_PER_TASK:-28}
echo "Starting parallel processing (${NCORES} workers via xargs -P)..."
echo ""

cat "${SAMPLE_LIST}" | xargs -P "${NCORES}" -I {} \
    bash -c 'process_sample "$@"' _ {} "${FASTA_DIR}" "${HMMSEARCH_DIR}" "${DENSITY_DIR}"

echo ""
echo "========================================"
echo "  Parallel processing complete"
echo "========================================"

# ---- Count results ----
N_DONE=$(ls "${DENSITY_DIR}"/*.pfam_density.tsv 2>/dev/null | wc -l)
echo "  Completed: ${N_DONE} / ${N_SAMPLES}"
echo ""

# ---- Create aggregate TSV ----
AGG_FILE="${OUT_DIR}/pfam_density_all_samples.tsv"
echo "Creating aggregate file: ${AGG_FILE}"

# Header from first file
head -1 "$(ls "${DENSITY_DIR}"/*.pfam_density.tsv | head -1)" > "${AGG_FILE}"

# Data rows from all files
for f in "${DENSITY_DIR}"/*.pfam_density.tsv; do
    tail -1 "$f"
done >> "${AGG_FILE}"

N_AGG=$(tail -n +2 "${AGG_FILE}" | wc -l)
echo "  Aggregate rows: ${N_AGG}"
echo ""

# ---- Quick summary stats ----
echo "========================================"
echo "  Quick Summary"
echo "========================================"
python3 -c "
import csv, sys
from collections import defaultdict

records = []
with open('${AGG_FILE}') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        records.append(row)

total_proteins = sum(int(r['n_proteins']) for r in records)
total_aa = sum(int(r['total_aa']) for r in records)
total_annotated = sum(int(r['n_annotated']) for r in records)
total_dark = sum(int(r['n_dark']) for r in records)
total_hits = sum(int(r['n_domain_hits']) for r in records)

print(f'  Samples:          {len(records):,}')
print(f'  Total proteins:   {total_proteins:,}')
print(f'  Total amino acids:{total_aa:,}')
print(f'  Annotated:        {total_annotated:,} ({100*total_annotated/total_proteins:.2f}%)')
print(f'  Dark:             {total_dark:,} ({100*total_dark/total_proteins:.2f}%)')
print(f'  Domain hits:      {total_hits:,}')
print(f'  Pfam/ORF:         {total_hits/total_proteins:.4f}')
print(f'  Pfam/1000aa:      {total_hits*1000/total_aa:.4f}')
print()

# Group by prefix
groups = defaultdict(lambda: {'n':0, 'prot':0, 'ann':0, 'dark':0, 'hits':0, 'aa':0})
for r in records:
    s = r['sample']
    if s.startswith('MGYA'):
        g = 'MGYA (metagenome)'
    elif s.startswith('MMETSP'):
        g = 'MMETSP (transcriptome)'
    elif s.startswith('GCA_') or s.startswith('GCF_'):
        g = 'GCA/GCF (NCBI genome)'
    else:
        g = 'Other (cultured/ref)'
    groups[g]['n'] += 1
    groups[g]['prot'] += int(r['n_proteins'])
    groups[g]['ann'] += int(r['n_annotated'])
    groups[g]['dark'] += int(r['n_dark'])
    groups[g]['hits'] += int(r['n_domain_hits'])
    groups[g]['aa'] += int(r['total_aa'])

print(f\"{'Group':<30} {'N':>6} {'Proteins':>14} {'%Dark':>8} {'Pfam/ORF':>10} {'Pfam/1kaa':>10}\")
print('-' * 80)
for g in sorted(groups.keys()):
    d = groups[g]
    pct = 100*d['dark']/d['prot'] if d['prot'] else 0
    por = d['hits']/d['prot'] if d['prot'] else 0
    pka = d['hits']*1000/d['aa'] if d['aa'] else 0
    print(f\"{g:<30} {d['n']:>6,} {d['prot']:>14,} {pct:>7.2f}% {por:>10.4f} {pka:>10.4f}\")
"

echo ""
echo "Done: $(date)"
