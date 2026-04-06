#!/bin/bash
#SBATCH --job-name=nf_B6_esmfold
#SBATCH --output=novel_families/logs/B6_esmfold_%j.out
#SBATCH --error=novel_families/logs/B6_esmfold_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1

# =============================================================================
# B6: ESMFold Structure Prediction for Top Novel Families
# =============================================================================
#
# Predict structures for representative sequences of top 20-50
# environment-responsive dark proteome families.
#
# =============================================================================

export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
export PYTHONUSERBASE=/scratch/drn2/newhome/.local
mkdir -p $TMPDIR

unset NETWORKX_BACKEND_CONFIG
unset NX_BACKEND_CONFIG

module purge
module load gcc/13.2.0 cuda/12.2.0

source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1

set -euo pipefail

BASE="/scratch/drn2/PROJECTS/TARA-LA4SR"
RESULTS_DIR="${BASE}/novel_families/results/track_b"
PDB_DIR="${BASE}/novel_families/results/track_b/pdb"
mkdir -p "${PDB_DIR}"

echo "=============================================="
echo "  B6: ESMFold Structure Prediction"
echo "=============================================="

# Select top families by R²
python3 -u - << 'PYEOF'
import pandas as pd
import sys
from pathlib import Path

BASE = Path("/scratch/drn2/PROJECTS/TARA-LA4SR")
r2_files = sorted((BASE / "novel_families/results/track_b").glob("dark_family_xgboost_forward_r2*.tsv"))
if not r2_files:
    print("ERROR: No R² results found")
    sys.exit(1)

dfs = [pd.read_csv(f, sep="\t") for f in r2_files]
r2_df = pd.concat(dfs).dropna(subset=["r2_overall"]).sort_values("r2_overall", ascending=False)
top = r2_df.head(50)

# Get representative sequences
fam_df = pd.read_csv(BASE / "novel_families/data/dark_families_filtered.tsv", sep="\t")
top_reps = fam_df[fam_df["family_id"].isin(top["family_id"])]["representative_protein_id"].tolist()

# Extract from cluster reps FASTA
reps_fasta = BASE / "novel_families/data/mmseqs_clusters_dark/cluster_reps.fasta"
if not reps_fasta.exists():
    print("ERROR: Representative FASTA not found")
    sys.exit(1)

# Write selected sequences
out_fasta = BASE / "novel_families/results/track_b/top_reps_for_esmfold.fasta"
target_set = set(top_reps)
writing = False

with open(reps_fasta) as fin, open(out_fasta, "w") as fout:
    for line in fin:
        if line.startswith(">"):
            pid = line.strip().lstrip(">").split()[0]
            writing = pid in target_set
        if writing:
            fout.write(line)

print(f"Written {len(target_set)} sequences to {out_fasta}")
PYEOF

# Run ESMFold (if available)
INPUT_FASTA="${RESULTS_DIR}/top_reps_for_esmfold.fasta"

if python3 -c "import esm" 2>/dev/null; then
    echo "  Running ESMFold predictions..."
    python3 -u - "${INPUT_FASTA}" "${PDB_DIR}" << 'ESMEOF'
import sys
import torch
import esm

fasta_path = sys.argv[1]
pdb_dir = sys.argv[2]

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

# Parse FASTA
sequences = {}
current_id = None
current_seq = []
with open(fasta_path) as f:
    for line in f:
        if line.startswith(">"):
            if current_id:
                sequences[current_id] = "".join(current_seq)
            current_id = line.strip().lstrip(">").split()[0]
            current_seq = []
        else:
            current_seq.append(line.strip())
    if current_id:
        sequences[current_id] = "".join(current_seq)

print(f"  Predicting {len(sequences)} structures...")

for pid, seq in sequences.items():
    if len(seq) > 800:
        print(f"  SKIP {pid}: length {len(seq)} > 800 aa")
        continue
    try:
        with torch.no_grad():
            output = model.infer_pdb(seq)
        pdb_path = f"{pdb_dir}/{pid}.pdb"
        with open(pdb_path, "w") as f:
            f.write(output)
        print(f"  {pid}: {len(seq)} aa -> {pdb_path}")
    except Exception as e:
        print(f"  ERROR {pid}: {e}")
ESMEOF
else
    echo "  ESMFold not installed — skipping structure prediction"
    echo "  Install with: pip install fair-esm"
fi

echo ""
echo "Done: $(date)"
