#!/bin/bash
#SBATCH --job-name=fix_msa
#SBATCH --output=logs/novel_domains/12b_fix_boltz_msa_%j.out
#SBATCH --error=logs/novel_domains/12b_fix_boltz_msa_%j.err
#SBATCH --partition=compute
#SBATCH --account=cgsb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#
# Re-run Phase 2 only: re-align and fix MSA projection for Boltz.
# The aligned sequences must be projected onto the query (first sequence)
# so that MSA length matches query length exactly.
#
# Provenance:
#   Input:  results/esmfold_rankbin/cluster_seqs/*.fasta (from job 16145211)
#   Output: results/esmfold_rankbin/boltz_msa/*.csv (fixed Boltz-format MSAs)
#   Script: MANUSCRIPT/scripts/novel_domains/12b_fix_boltz_msa_20260530_160000.sbatch
#   Date:   2026-05-30

set -euo pipefail

module purge
module load gcc/13.2.0

MAFFT=/scratch/drn2/software/conda-mamba_1/bin/mafft

BASE=/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains
CLUSTER_SEQS_DIR=$BASE/results/esmfold_rankbin/cluster_seqs
ALIGNED_DIR=$BASE/results/esmfold_rankbin/cluster_aligned
OUTDIR=$BASE/results/esmfold_rankbin/boltz_msa

mkdir -p $ALIGNED_DIR

# Clear old MSAs
rm -f $OUTDIR/*.csv

echo "[$(date)] Re-running Phase 2: mafft align + query-projected Boltz CSV"
echo "[$(date)] Input: $CLUSTER_SEQS_DIR ($(ls $CLUSTER_SEQS_DIR/*.fasta | wc -l) files)"

align_one() {
    local fasta=$1
    local seq_name=$(basename $fasta .fasta)
    local aligned=$ALIGNED_DIR/${seq_name}.afa
    local boltz_csv=$OUTDIR/${seq_name}_0.csv
    local n_seqs=$(grep -c "^>" $fasta)

    # For singletons, make a self-only MSA
    if [ "$n_seqs" -le 1 ]; then
        local seq=$(grep -v "^>" $fasta | tr -d '\n')
        echo "key,sequence" > $boltz_csv
        echo "-1,$seq" >> $boltz_csv
        echo "-1,$seq" >> $boltz_csv
        return 0
    fi

    # Align with mafft
    $MAFFT --auto --quiet --thread 1 $fasta > $aligned 2>/dev/null

    # Convert to Boltz CSV: project onto query (first sequence)
    # Keep only columns where query has a residue
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

N_CPUS=${SLURM_CPUS_PER_TASK:-28}
find $CLUSTER_SEQS_DIR -name "*.fasta" -size +0c | \
    xargs -P $N_CPUS -I {} bash -c 'align_one "$@"' _ {}

echo "[$(date)] Phase 2 complete"
N_MSA=$(ls $OUTDIR/*.csv 2>/dev/null | wc -l)
echo "[$(date)] Generated $N_MSA Boltz MSA files in $OUTDIR"

# Verify: check that MSA lengths match query lengths
echo "[$(date)] Verifying MSA lengths..."
python3 -c "
from pathlib import Path
import sys
msa_dir = Path('$OUTDIR')
input_dir = Path('$CLUSTER_SEQS_DIR')
mismatches = 0
for csv in sorted(msa_dir.glob('*.csv')):
    seq_name = csv.stem.replace('_0', '')
    fasta = input_dir / (seq_name + '.fasta')
    if not fasta.exists():
        continue
    with open(fasta) as f:
        lines = f.readlines()
    query_len = len(''.join(l.strip() for l in lines[1:] if not l.startswith('>')).split('>')[0])
    with open(csv) as f:
        next(f)
        first_line = next(f).strip()
    msa_len = len(first_line.split(',')[1])
    if query_len != msa_len:
        mismatches += 1
        if mismatches <= 5:
            print(f'  MISMATCH: {seq_name} query={query_len} msa={msa_len}')
print(f'Total mismatches: {mismatches} / {len(list(msa_dir.glob(\"*.csv\")))}')
"

echo "[$(date)] Done"
