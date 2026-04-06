#!/bin/bash
#
# Run LA4SR TI-inc (algaGPT) on all 2,372 PROTEOMES using Singularity container
#
# Usage: ./run_la4sr_all_proteins.sh

set -euo pipefail

# Paths
BASE_DIR="/media/drn/External1/TARA-Oceans"
PROTEINS_DIR="${BASE_DIR}/02_processed_data/proteins"
LA4SR_DIR="${BASE_DIR}/tools/la4sr"
RESULTS_DIR="${BASE_DIR}/03_analyses/la4sr_results"
SIF="${LA4SR_DIR}/la4sr_sp2.sif"
INFER_SCRIPT="${LA4SR_DIR}/infer_TI-inc-algaGPT.py"
CKPT="${LA4SR_DIR}/ckpt.pt"

# Cache for HF
export TRANSFORMERS_CACHE="${LA4SR_DIR}/cache"
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$RESULTS_DIR"

# Check requirements
if [[ ! -f "$SIF" ]]; then
    echo "ERROR: Singularity container not found: $SIF"
    exit 1
fi

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Model checkpoint not found: $CKPT"
    exit 1
fi

if [[ ! -f "$INFER_SCRIPT" ]]; then
    echo "ERROR: Inference script not found: $INFER_SCRIPT"
    exit 1
fi

# Count proteomes
cd "$PROTEINS_DIR"
total=$(ls *.aa.fa | wc -l)

echo "========================================"
echo "LA4SR TI-inc Batch Processing (algaGPT)"
echo "========================================"
echo ""
echo "Total proteomes: $total"
echo "Model: ckpt.pt (TI-inc algaGPT - fungi-aware)"
echo "Container: la4sr_sp2.sif"
echo "Results: $RESULTS_DIR"
echo ""
echo "========================================"
echo ""

# Process each file
processed=0
failed=0
skipped=0

for protein_file in *.aa.fa; do
    ((processed++))

    basename="${protein_file%.aa.fa}"
    output_file="${RESULTS_DIR}/${basename}_la4sr.tsv"

    # Skip if already processed
    if [[ -f "$output_file" ]]; then
        ((skipped++))
        echo "[$processed/$total] ⊕ Skipped: $protein_file (already processed)"
        continue
    fi

    echo "[$processed/$total] Processing: $protein_file"

    # Run LA4SR via Singularity
    if singularity exec --nv \
        -B "${PROTEINS_DIR}/${protein_file}:/input.fasta" \
        -B "${LA4SR_DIR}:/workdir" \
        -B "$TRANSFORMERS_CACHE:$TRANSFORMERS_CACHE" \
        --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
        "$SIF" \
        bash -c 'cd /workdir && python3 infer_TI-inc-algaGPT.py --init_from resume --out_dir /workdir /input.fasta -o /tmp/output.tsv && cat /tmp/output.tsv' \
        > "$output_file" 2>&1; then

        echo "[$processed/$total] ✓ Success: $protein_file"
    else
        ((failed++))
        echo "[$processed/$total] ✗ Failed: $protein_file"
        rm -f "$output_file"
    fi

done

echo ""
echo "========================================"
echo "Processing Complete"
echo "========================================"
echo "Total proteomes: $total"
echo "Processed:       $((processed - skipped - failed))"
echo "Skipped:         $skipped"
echo "Failed:          $failed"
echo ""
echo "Results: $RESULTS_DIR"
echo "========================================"
