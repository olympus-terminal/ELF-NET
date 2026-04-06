#!/bin/bash
#
# LA4SR Protein Classification using Apptainer/Singularity with algaGPT
#
# Provenance:
#   Script: run_la4sr_apptainer_20251231_000000.sh
#   Date: 2025-12-31 00:00:00
#   Purpose: Classify algal proteins using LA4SR algaGPT in Apptainer container
#   Input: Protein files from 02_processed_data/proteins/
#   Output: Classifications in 03_analyses/la4sr_results/
#   Model: algaGPT (local ckpt.pt checkpoint)
#
# Usage:
#   ./run_la4sr_apptainer_20251231_000000.sh test      # Test on 5 files
#   ./run_la4sr_apptainer_20251231_000000.sh single    # Run on 1 CPU
#   ./run_la4sr_apptainer_20251231_000000.sh all       # Process all files
#

set -e

# Configuration
BASE_DIR="/media/drn/External1/TARA-Oceans"
CONTAINER="${BASE_DIR}/tools/la4sr/la4sr_sp2.sif"
INFER_SCRIPT="${BASE_DIR}/tools/la4sr/infer_TI-inc-algaGPT.py"
FILELIST="${BASE_DIR}/03_analyses/la4sr_inputs/la4sr_protein_filelist_LOCAL.txt"
OUTPUT_DIR="${BASE_DIR}/03_analyses/la4sr_results"
LA4SR_DIR="${BASE_DIR}/tools/la4sr"

# Check prerequisites
if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: Singularity container not found: $CONTAINER"
    exit 1
fi

if [[ ! -f "$INFER_SCRIPT" ]]; then
    echo "ERROR: Inference script not found: $INFER_SCRIPT"
    exit 1
fi

if [[ ! -f "${LA4SR_DIR}/ckpt.pt" ]]; then
    echo "ERROR: Model checkpoint not found: ${LA4SR_DIR}/ckpt.pt"
    exit 1
fi

if [[ ! -f "${LA4SR_DIR}/meta.pkl" ]]; then
    echo "ERROR: Metadata file not found: ${LA4SR_DIR}/meta.pkl"
    exit 1
fi

if [[ ! -f "$FILELIST" ]]; then
    echo "ERROR: Filelist not found: $FILELIST"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# CRITICAL: Verify GPU is available
echo "Checking GPU availability..."
GPU_CHECK=$(apptainer exec --nv "$CONTAINER" python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [[ "$GPU_CHECK" != "True" ]]; then
    echo "ERROR: GPU not available! LA4SR requires GPU for acceptable performance."
    echo "Please ensure:"
    echo "  1. NVIDIA drivers are installed"
    echo "  2. nvidia-container-toolkit is configured"
    echo "  3. GPU is not in use by another process"
    exit 1
fi
echo "✓ GPU detected and accessible"

# Function to process a single protein file
process_protein_file() {
    local PROTEIN_FILE="$1"
    local FILE_NUM="$2"
    local TOTAL_FILES="$3"

    # Extract filename
    BASENAME=$(basename "$PROTEIN_FILE")
    STEM="${BASENAME%.fa}"
    STEM="${STEM%.faa}"
    STEM="${STEM%.fasta}"

    # Output file
    OUTPUT_FILE="${OUTPUT_DIR}/${STEM}_la4sr_algaGPT.tsv"

    # Skip if already processed
    if [[ -f "$OUTPUT_FILE" ]]; then
        echo "[$FILE_NUM/$TOTAL_FILES] ✓ Already processed: $BASENAME"
        return 0
    fi

    echo "[$FILE_NUM/$TOTAL_FILES] Processing: $BASENAME"

    # Run LA4SR algaGPT via Apptainer
    # Bind mount model directory (read-only) and create separate output directory

    TEMP_OUTPUT=$(mktemp -d -p "$OUTPUT_DIR" la4sr_out_XXXXXX)

    apptainer exec --nv \
        -B "$PROTEIN_FILE:/input.fasta:ro" \
        -B "${LA4SR_DIR}:/model:ro" \
        -B "$TEMP_OUTPUT:/output" \
        --pwd /output \
        --env HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
        "$CONTAINER" \
        bash -c "python3 /model/infer_TI-inc-algaGPT.py \
          --init_from resume \
          --out_dir /model \
          --device cuda \
          --dtype float16 \
          /input.fasta \
          -o /output/output.tsv" \
        2> "${OUTPUT_DIR}/${STEM}_la4sr.log"

    # Move output from temp dir
    if [[ -f "$TEMP_OUTPUT/output.tsv" ]]; then
        mv "$TEMP_OUTPUT/output.tsv" "${OUTPUT_DIR}/${STEM}_temp.tsv"
    fi

    # Clean up temp directory
    rm -rf "$TEMP_OUTPUT"

    # Check if output was created
    if [[ -f "${OUTPUT_DIR}/${STEM}_temp.tsv" ]]; then
        # Add provenance header and reformat
        {
            echo "# Provenance:"
            echo "#   Script: run_la4sr_apptainer_20251231_000000.sh"
            echo "#   Input: $PROTEIN_FILE"
            echo "#   Model: algaGPT (ckpt.pt)"
            echo "#   Container: $CONTAINER"
            echo "#   Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
            echo "#"
            # Extract just the record_id and model_output columns, skip header
            tail -n +2 "${OUTPUT_DIR}/${STEM}_temp.tsv" | \
                awk -F'\t' '{print $1"\t"$3}'
        } > "$OUTPUT_FILE"

        # Clean up temp TSV
        rm "${OUTPUT_DIR}/${STEM}_temp.tsv"
    else
        echo "[$FILE_NUM/$TOTAL_FILES] ✗ ERROR: Output not created for $BASENAME"
        return 1
    fi

    # Count results (looking for 'algae' or 'conta' in the output)
    N_ALGAL=$(grep -o 'algae' "$OUTPUT_FILE" | wc -l || echo 0)
    N_CONTA=$(grep -o 'conta' "$OUTPUT_FILE" | wc -l || echo 0)
    TOTAL=$((N_ALGAL + N_CONTA))

    if [[ $TOTAL -gt 0 ]]; then
        PCT_ALGAL=$(awk "BEGIN {printf \"%.1f\", 100*$N_ALGAL/$TOTAL}")
        echo "[$FILE_NUM/$TOTAL_FILES] ✓ $BASENAME: $N_ALGAL algal ($PCT_ALGAL%), $N_CONTA non-algal"
    else
        echo "[$FILE_NUM/$TOTAL_FILES] ⚠ $BASENAME: Classifications generated, check log for details"
    fi
}

# Main processing logic
case "${1:-help}" in
    test)
        echo "=== LA4SR algaGPT Test Mode (5 files) ==="
        echo "Container: $CONTAINER"
        echo "Model: algaGPT (ckpt.pt)"
        echo "Output: $OUTPUT_DIR"
        echo ""

        # Process first 5 files
        FILES=($(head -5 "$FILELIST"))
        TOTAL=${#FILES[@]}

        for i in "${!FILES[@]}"; do
            process_protein_file "${FILES[$i]}" $((i+1)) $TOTAL
        done

        echo ""
        echo "Test complete! Check results in: $OUTPUT_DIR"
        ;;

    single)
        echo "=== LA4SR algaGPT Sequential Processing ==="
        echo "Container: $CONTAINER"
        echo "Model: algaGPT (ckpt.pt)"
        echo "Output: $OUTPUT_DIR"
        echo ""

        # Count total files
        TOTAL=$(wc -l < "$FILELIST")
        echo "Processing $TOTAL protein files..."
        echo ""

        COUNTER=0
        while IFS= read -r PROTEIN_FILE; do
            COUNTER=$((COUNTER + 1))
            process_protein_file "$PROTEIN_FILE" $COUNTER $TOTAL
        done < "$FILELIST"

        echo ""
        echo "Processing complete!"
        echo "Results saved to: $OUTPUT_DIR"
        ;;

    all)
        echo "=== LA4SR algaGPT Full Processing ==="
        echo "Container: $CONTAINER"
        echo "Model: algaGPT (ckpt.pt)"
        echo "Output: $OUTPUT_DIR"
        echo ""
        echo "This will process ALL protein files sequentially."
        echo "Estimated time: ~2-3 hours for 2,372 files"
        echo ""
        read -p "Continue? (y/n) " -n 1 -r
        echo

        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi

        # Same as 'single' mode
        TOTAL=$(wc -l < "$FILELIST")
        echo "Processing $TOTAL protein files..."
        echo ""

        COUNTER=0
        while IFS= read -r PROTEIN_FILE; do
            COUNTER=$((COUNTER + 1))
            process_protein_file "$PROTEIN_FILE" $COUNTER $TOTAL
        done < "$FILELIST"

        echo ""
        echo "Processing complete!"
        echo "Results saved to: $OUTPUT_DIR"

        # Generate summary
        echo ""
        echo "=== Summary ==="
        TOTAL_PROCESSED=$(find "$OUTPUT_DIR" -name "*_la4sr_algaGPT.tsv" | wc -l)
        echo "Files processed: $TOTAL_PROCESSED"
        ;;

    *)
        echo "LA4SR Protein Classification with Apptainer (algaGPT)"
        echo ""
        echo "Usage: $0 {test|single|all}"
        echo ""
        echo "Commands:"
        echo "  test    - Test on first 5 files"
        echo "  single  - Process all files sequentially (recommended)"
        echo "  all     - Same as 'single' with confirmation"
        echo ""
        echo "Configuration:"
        echo "  Container: $CONTAINER"
        echo "  Model:     algaGPT (ckpt.pt)"
        echo "  Filelist:  $FILELIST ($(wc -l < "$FILELIST") files)"
        echo "  Output:    $OUTPUT_DIR"
        echo ""
        exit 1
        ;;
esac
