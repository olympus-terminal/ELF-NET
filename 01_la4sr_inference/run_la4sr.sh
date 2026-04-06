#!/bin/bash
#
# LA4SR Batch Processing Wrapper
#
# This script provides easy-to-use commands for running LA4SR on protein files.
#
# Usage:
#   ./run_la4sr.sh sequential  # Run on 1 GPU
#   ./run_la4sr.sh parallel    # Run on 4 GPUs
#

set -e

# Configuration
BASE_DIR="/media/drn/External1/TARA-Oceans"
FILELIST="${BASE_DIR}/03_analyses/la4sr_inputs/la4sr_protein_filelist.txt"
MODEL="${BASE_DIR}/tools/la4sr/ckpt.pt"
OUTPUT="${BASE_DIR}/03_analyses/la4sr_results"
SCRIPT="${BASE_DIR}/run_la4sr_batch_20251230_180000.py"

# Check if Python script exists
if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: Batch processing script not found: $SCRIPT"
    exit 1
fi

# Check if filelist exists
if [[ ! -f "$FILELIST" ]]; then
    echo "ERROR: Filelist not found: $FILELIST"
    exit 1
fi

# Check if model exists
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: LA4SR model not found: $MODEL"
    exit 1
fi

# Parse command
case "$1" in
    sequential|seq|1)
        echo "Running LA4SR sequentially on 1 GPU..."
        python3 "$SCRIPT" \
            --filelist "$FILELIST" \
            --model "$MODEL" \
            --output "$OUTPUT" \
            --gpus 1
        ;;

    parallel|par|4)
        echo "Running LA4SR in parallel on 4 GPUs..."
        python3 "$SCRIPT" \
            --filelist "$FILELIST" \
            --model "$MODEL" \
            --output "$OUTPUT" \
            --gpus 4
        ;;

    test)
        echo "Running LA4SR on first 10 files (test mode)..."
        # Create temporary filelist with first 10 files
        head -10 "$FILELIST" > /tmp/la4sr_test_filelist.txt
        python3 "$SCRIPT" \
            --filelist /tmp/la4sr_test_filelist.txt \
            --model "$MODEL" \
            --output "${OUTPUT}_test" \
            --gpus 1
        rm /tmp/la4sr_test_filelist.txt
        ;;

    *)
        echo "Usage: $0 {sequential|parallel|test}"
        echo ""
        echo "Commands:"
        echo "  sequential  - Run on 1 GPU (~5.6 hours for 1,007 files)"
        echo "  parallel    - Run on 4 GPUs (~1.4 hours for 1,007 files)"
        echo "  test        - Run on first 10 files only (for testing)"
        echo ""
        echo "Configuration:"
        echo "  Filelist: $FILELIST"
        echo "  Model:    $MODEL"
        echo "  Output:   $OUTPUT"
        exit 1
        ;;
esac

echo ""
echo "LA4SR processing complete!"
echo "Results saved to: $OUTPUT"
