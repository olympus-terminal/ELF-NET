#!/bin/bash
#SBATCH --job-name=boltz_msa
#SBATCH --output=logs/novel_domains/12_boltz_msa_%j.out
#SBATCH --error=logs/novel_domains/12_boltz_msa_%j.err
#SBATCH --partition=compute
#SBATCH --account=cgsb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#
# Build MSAs for 3,600 rank-binned novel domain reps using existing
# MMseqs2 cluster membership. Two phases:
#   Phase 1: Extract cluster member sequences (streaming, memory-efficient)
#   Phase 2: mafft align + convert to Boltz CSV format (parallel)
#
# Provenance:
#   Input:  clusters_30/clusters_30_cluster.tsv (rep→member mapping)
#           clusters_30/clusters_30_all_seqs.fasta (323M seqs)
#           results/esmfold_rankbin/rankbin_sequences.fasta (3,600 reps)
#   Output: results/esmfold_rankbin/boltz_msa/*.csv (Boltz-format MSAs)
#   Script: MANUSCRIPT/scripts/novel_domains/12_build_boltz_msa_20260530_122500.sbatch
#   Date:   2026-05-30

set -euo pipefail

module purge
module load gcc/13.2.0

MMSEQS=/scratch/drn2/software/conda-mamba_1/bin/mmseqs
MAFFT=/scratch/drn2/software/conda-mamba_1/bin/mafft

BASE=/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains
CLUSTER_TSV=$BASE/clusters_30/clusters_30_cluster.tsv
ALL_SEQS=$BASE/clusters_30/clusters_30_all_seqs.fasta
RANKBIN_FASTA=$BASE/results/esmfold_rankbin/rankbin_sequences.fasta

OUTDIR=$BASE/results/esmfold_rankbin/boltz_msa
CLUSTER_SEQS_DIR=$BASE/results/esmfold_rankbin/cluster_seqs
ALIGNED_DIR=$BASE/results/esmfold_rankbin/cluster_aligned

mkdir -p $OUTDIR $CLUSTER_SEQS_DIR $ALIGNED_DIR
mkdir -p /scratch/drn2/PROJECTS/TARA-LA4SR/logs/novel_domains

echo "[$(date)] Phase 1: Extracting cluster member sequences"
echo "[$(date)] Cluster TSV: $CLUSTER_TSV ($(wc -l < $CLUSTER_TSV) lines)"
echo "[$(date)] All seqs FASTA: $ALL_SEQS"
echo "[$(date)] Rank-bin reps: $RANKBIN_FASTA ($(grep -c '^>' $RANKBIN_FASTA) seqs)"

# Phase 1: Extract rep IDs, scan cluster TSV for their members,
#           then pull member sequences from the all_seqs FASTA.

python3 - << 'PYEOF'
import os
import sys
from pathlib import Path
from collections import defaultdict

BASE = Path("/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains")
RANKBIN_FASTA = BASE / "results" / "esmfold_rankbin" / "rankbin_sequences.fasta"
CLUSTER_TSV = BASE / "clusters_30" / "clusters_30_cluster.tsv"
ALL_SEQS = BASE / "clusters_30" / "clusters_30_all_seqs.fasta"
CLUSTER_SEQS_DIR = BASE / "results" / "esmfold_rankbin" / "cluster_seqs"

# Step 1a: Get the 3,600 rep IDs (strip NOVEL_ prefix to match cluster TSV)
print(f"[Step 1a] Reading rank-bin rep IDs from {RANKBIN_FASTA}")
rep_ids = set()
rep_id_to_seqname = {}  # cluster_tsv_id -> seq_NNNNN name for output
seq_idx = 0
with open(RANKBIN_FASTA) as f:
    for line in f:
        if line.startswith(">"):
            full_id = line[1:].split()[0]
            # Strip NOVEL_ prefix if present
            cluster_id = full_id.replace("NOVEL_", "")
            rep_ids.add(cluster_id)
            safe_name = f"seq_{seq_idx:05d}"
            rep_id_to_seqname[cluster_id] = safe_name
            seq_idx += 1

print(f"  Found {len(rep_ids)} rep IDs")

# Step 1b: Stream cluster TSV, collect member IDs for our reps
print(f"[Step 1b] Scanning cluster TSV for member IDs...")
rep_to_members = defaultdict(list)
n_lines = 0
with open(CLUSTER_TSV) as f:
    for line in f:
        n_lines += 1
        if n_lines % 50_000_000 == 0:
            print(f"  ...processed {n_lines/1e6:.0f}M lines, found members for {sum(len(v) for v in rep_to_members.values())} entries")
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[0] in rep_ids:
            rep_to_members[parts[0]].append(parts[1])

print(f"  Scanned {n_lines} lines total")
print(f"  Found members for {len(rep_to_members)} of {len(rep_ids)} reps")

# Stats
sizes = [len(v) for v in rep_to_members.values()]
sizes.sort()
print(f"  Cluster sizes: min={sizes[0]}, median={sizes[len(sizes)//2]}, max={sizes[-1]}, mean={sum(sizes)/len(sizes):.1f}")

# Save member lists for debugging
member_list_file = CLUSTER_SEQS_DIR.parent / "member_counts.tsv"
with open(member_list_file, "w") as f:
    f.write("rep_id\tseq_name\tn_members\n")
    for rep_id in sorted(rep_to_members.keys(), key=lambda x: rep_id_to_seqname[x]):
        f.write(f"{rep_id}\t{rep_id_to_seqname[rep_id]}\t{len(rep_to_members[rep_id])}\n")

# Step 1c: Collect ALL needed member IDs into a set for fast lookup
all_member_ids = set()
for members in rep_to_members.values():
    all_member_ids.update(members)
print(f"[Step 1c] Need to extract {len(all_member_ids)} unique member sequences")

# Step 1d: Stream all_seqs FASTA, extract needed sequences
print(f"[Step 1d] Scanning all_seqs FASTA for member sequences...")
member_seqs = {}
current_id = None
current_seq_parts = []
n_seqs_scanned = 0

def flush():
    global current_id, current_seq_parts
    if current_id and current_id in all_member_ids:
        member_seqs[current_id] = "".join(current_seq_parts).replace("*", "")
    current_id = None
    current_seq_parts = []

with open(ALL_SEQS) as f:
    for line in f:
        if line.startswith(">"):
            flush()
            n_seqs_scanned += 1
            if n_seqs_scanned % 50_000_000 == 0:
                print(f"  ...scanned {n_seqs_scanned/1e6:.0f}M seqs, extracted {len(member_seqs)}")
            current_id = line[1:].split()[0]
        else:
            if current_id in all_member_ids:
                current_seq_parts.append(line.strip())
    flush()

print(f"  Scanned {n_seqs_scanned} sequences total")
print(f"  Extracted {len(member_seqs)} member sequences")

missing = all_member_ids - set(member_seqs.keys())
if missing:
    print(f"  WARNING: {len(missing)} member IDs not found in FASTA")

# Step 1e: Write per-cluster multi-FASTA files
print(f"[Step 1e] Writing per-cluster FASTA files...")
n_written = 0
for rep_id, members in rep_to_members.items():
    seq_name = rep_id_to_seqname[rep_id]
    out_file = CLUSTER_SEQS_DIR / f"{seq_name}.fasta"
    with open(out_file, "w") as f:
        for member_id in members:
            if member_id in member_seqs:
                f.write(f">{member_id}\n{member_seqs[member_id]}\n")
    n_written += 1

print(f"  Wrote {n_written} cluster FASTA files to {CLUSTER_SEQS_DIR}")
print("[Phase 1 complete]")
PYEOF

echo "[$(date)] Phase 1 complete"

# Phase 2: Align each cluster with mafft and convert to Boltz CSV
echo "[$(date)] Phase 2: Aligning clusters with mafft (parallel)"

align_one() {
    local fasta=$1
    local seq_name=$(basename $fasta .fasta)
    local aligned=$ALIGNED_DIR/${seq_name}.a3m
    local boltz_csv=$OUTDIR/${seq_name}_0.csv
    local n_seqs=$(grep -c "^>" $fasta)

    # Skip if already done
    if [ -f "$boltz_csv" ]; then
        return 0
    fi

    # For singletons (1 seq), just make a self-only MSA
    if [ "$n_seqs" -le 1 ]; then
        local seq=$(grep -v "^>" $fasta | tr -d '\n')
        echo "key,sequence" > $boltz_csv
        echo "-1,$seq" >> $boltz_csv
        echo "-1,$seq" >> $boltz_csv
        return 0
    fi

    # Align with mafft (auto strategy picks FFT-NS-2 for small, L-INS-i for <200)
    $MAFFT --auto --quiet --thread 1 $fasta > $aligned 2>/dev/null

    # Convert aligned FASTA to Boltz CSV format
    # Boltz expects MSA sequences to match query length exactly.
    # Project alignment onto query: keep only columns where the query
    # (first sequence) has a residue, discard insertion columns.
    python3 -c "
import sys
aligned = '$aligned'
boltz_csv = '$boltz_csv'
seqs = []
current = []
with open(aligned) as f:
    for line in f:
        if line.startswith('>'):
            if current:
                seqs.append(''.join(current))
            current = []
        else:
            current.append(line.strip())
    if current:
        seqs.append(''.join(current))
if not seqs:
    sys.exit(1)
query = seqs[0]
keep = [i for i, c in enumerate(query) if c != '-']
with open(boltz_csv, 'w') as f:
    f.write('key,sequence\n')
    for s in seqs:
        projected = ''.join(s[i] for i in keep)
        f.write(f'-1,{projected}\n')
"
    rm -f $aligned
}
export -f align_one
export MAFFT ALIGNED_DIR OUTDIR

# Run in parallel using GNU parallel or xargs
N_CPUS=${SLURM_CPUS_PER_TASK:-28}
find $CLUSTER_SEQS_DIR -name "*.fasta" -size +0c | \
    xargs -P $N_CPUS -I {} bash -c 'align_one "$@"' _ {}

echo "[$(date)] Phase 2 complete"

# Summary
N_MSA=$(ls $OUTDIR/*.csv 2>/dev/null | wc -l)
echo "[$(date)] Generated $N_MSA Boltz MSA files in $OUTDIR"
echo "[$(date)] Done"
