#!/bin/bash
#SBATCH --job-name=dark_filter
#SBATCH --output=logs/novel_domains/03_filter_concat_%j.out
#SBATCH --error=logs/novel_domains/03_filter_concat_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=90G
#SBATCH --partition=compute

# =============================================================================
# Step 3: Filter & Concatenate Dark Proteome
# =============================================================================
#
# Provenance:
#   Script: scripts/novel_domains/03_filter_concat.sbatch
#   Generated: 2026-02-21
#   Target HPC: Jubail (NYU Abu Dhabi)
#
# Purpose:
#   Filter dark proteome sequences by length and complexity, then concatenate
#   into a single FASTA for clustering. Removes:
#   - Fragments (<30 aa)
#   - Misassemblies (>5000 aa)
#   - Low-complexity sequences (>90% single amino acid)
#   - Poly-X artifacts
#
# Input:
#   - 03_analyses/novel_domains/dark_fasta/{sample}.dark.fa (from Step 2)
#
# Output:
#   - 03_analyses/novel_domains/dark_proteome_filtered.fa (concatenated)
#   - 03_analyses/novel_domains/filter_stats.tsv (per-sample statistics)
#
# Usage:
#   cd /scratch/drn2/PROJECTS/TARA-LA4SR
#   sbatch MANUSCRIPT/scripts/novel_domains/03_filter_concat.sbatch
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CRITICAL ENVIRONMENT SETUP
# =============================================================================

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
mkdir -p $TMPDIR
export PYTHONUSERBASE=/scratch/drn2/newhome/.local

module purge
module load gcc/13.2.0

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

# =============================================================================
# Configuration
# =============================================================================

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
DARK_FASTA_DIR="${BASE}/03_analyses/novel_domains/dark_fasta"
OUT_DIR="${BASE}/03_analyses/novel_domains"
SAMPLE_LIST="${BASE}/03_analyses/novel_domains/sample_list.txt"

OUTPUT_FA="${OUT_DIR}/dark_proteome_filtered.fa"
STATS_FILE="${OUT_DIR}/filter_stats.tsv"

# Filter parameters
MIN_LEN=30
MAX_LEN=5000
MAX_SINGLE_AA_FRAC=0.90  # Remove if >90% is a single amino acid

echo "========================================"
echo "Step 3: Filter & Concatenate Dark Proteome"
echo "========================================"
echo ""
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "Date:       $(date)"
echo ""
echo "Parameters:"
echo "  Min length:      ${MIN_LEN} aa"
echo "  Max length:      ${MAX_LEN} aa"
echo "  Max single-AA:   ${MAX_SINGLE_AA_FRAC}"
echo ""

# =============================================================================
# Filter and concatenate using Python
# =============================================================================

python3 << 'PYEOF'
import os
import sys
from collections import Counter
from pathlib import Path

BASE = "/scratch/drn2/PROJECTS/TARA-LA4SR"
DARK_FASTA_DIR = os.path.join(BASE, "03_analyses/novel_domains/dark_fasta")
SAMPLE_LIST = os.path.join(BASE, "03_analyses/novel_domains/sample_list.txt")
OUTPUT_FA = os.path.join(BASE, "03_analyses/novel_domains/dark_proteome_filtered.fa")
STATS_FILE = os.path.join(BASE, "03_analyses/novel_domains/filter_stats.tsv")

MIN_LEN = 30
MAX_LEN = 5000
MAX_SINGLE_AA_FRAC = 0.90

def is_low_complexity(seq):
    """Check if sequence is dominated by a single amino acid."""
    if len(seq) == 0:
        return True
    counts = Counter(seq.upper())
    most_common_frac = counts.most_common(1)[0][1] / len(seq)
    return most_common_frac > MAX_SINGLE_AA_FRAC

def has_poly_x(seq, min_run=20):
    """Check for long runs of a single amino acid or X."""
    seq_upper = seq.upper()
    current_char = None
    run_len = 0
    for c in seq_upper:
        if c == current_char:
            run_len += 1
            if run_len >= min_run:
                return True
        else:
            current_char = c
            run_len = 1
    return False

# Read sample list
with open(SAMPLE_LIST) as f:
    samples = [line.strip() for line in f if line.strip()]

print(f"Processing {len(samples)} samples...")
print()

# Stats tracking
stats_rows = []
total_input = 0
total_pass = 0
total_short = 0
total_long = 0
total_lowcomp = 0
total_polyx = 0

with open(OUTPUT_FA, 'w') as fout:
    for i, sample in enumerate(samples):
        fasta_path = os.path.join(DARK_FASTA_DIR, f"{sample}.dark.fa")

        if not os.path.exists(fasta_path):
            stats_rows.append((sample, 0, 0, 0, 0, 0, 0))
            continue

        counts = {'input': 0, 'pass': 0, 'short': 0, 'long': 0, 'lowcomp': 0, 'polyx': 0}

        # Parse FASTA
        header = None
        seq_parts = []

        with open(fasta_path) as fin:
            for line in fin:
                line = line.rstrip()
                if line.startswith('>'):
                    if header is not None:
                        seq = ''.join(seq_parts)
                        counts['input'] += 1
                        seq_len = len(seq)
                        if seq_len < MIN_LEN:
                            counts['short'] += 1
                        elif seq_len > MAX_LEN:
                            counts['long'] += 1
                        elif is_low_complexity(seq):
                            counts['lowcomp'] += 1
                        elif has_poly_x(seq):
                            counts['polyx'] += 1
                        else:
                            counts['pass'] += 1
                            fout.write(f"{header}\n{seq}\n")
                    header = line
                    seq_parts = []
                else:
                    seq_parts.append(line)
            # Last sequence
            if header is not None:
                seq = ''.join(seq_parts)
                counts['input'] += 1
                seq_len = len(seq)
                if seq_len < MIN_LEN:
                    counts['short'] += 1
                elif seq_len > MAX_LEN:
                    counts['long'] += 1
                elif is_low_complexity(seq):
                    counts['lowcomp'] += 1
                elif has_poly_x(seq):
                    counts['polyx'] += 1
                else:
                    counts['pass'] += 1
                    fout.write(f"{header}\n{seq}\n")

        n_input = counts['input']
        n_pass = counts['pass']
        n_short = counts['short']
        n_long = counts['long']
        n_lowcomp = counts['lowcomp']
        n_polyx = counts['polyx']

        stats_rows.append((sample, n_input, n_pass, n_short, n_long, n_lowcomp, n_polyx))
        total_input += n_input
        total_pass += n_pass
        total_short += n_short
        total_long += n_long
        total_lowcomp += n_lowcomp
        total_polyx += n_polyx

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples "
                  f"({total_pass:,} passing so far)...")

# Write stats
with open(STATS_FILE, 'w') as f:
    f.write("sample\tinput\tpassing\ttoo_short\ttoo_long\tlow_complexity\tpoly_x\n")
    for row in stats_rows:
        f.write("\t".join(str(x) for x in row) + "\n")
    # Summary row
    f.write(f"TOTAL\t{total_input}\t{total_pass}\t{total_short}\t"
            f"{total_long}\t{total_lowcomp}\t{total_polyx}\n")

print()
print("============================================")
print("  Filter Summary")
print("============================================")
print(f"  Total input sequences:     {total_input:>12,}")
print(f"  Passing (filtered):        {total_pass:>12,}")
print(f"  Removed — too short:       {total_short:>12,}")
print(f"  Removed — too long:        {total_long:>12,}")
print(f"  Removed — low complexity:  {total_lowcomp:>12,}")
print(f"  Removed — poly-X:          {total_polyx:>12,}")
if total_input > 0:
    pct = 100.0 * total_pass / total_input
    print(f"  Pass rate:                 {pct:>11.1f}%")
print()
print(f"  Output: {OUTPUT_FA}")
print(f"  Stats:  {STATS_FILE}")

PYEOF

# =============================================================================
# Post-filter summary
# =============================================================================

echo ""
echo "  Output file size:"
ls -lh "${OUTPUT_FA}" | awk '{print "    "$5" "$NF}'
echo ""
echo "  Sequence count in filtered FASTA:"
{ grep -c "^>" "${OUTPUT_FA}" || echo 0; } | awk '{print "    "$1" sequences"}'
echo ""
echo "  Done: $(date)"
