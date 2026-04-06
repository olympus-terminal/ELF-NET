#!/usr/bin/env python3
"""
Run LA4SR (Language model-Augmented Algae Sequence Retrieval) in batch mode.

This script processes multiple protein FASTA files through LA4SR for algal
sequence classification. Supports sequential and parallel processing.

Provenance:
  Script: run_la4sr_batch_20251230_180000.py
  Date: 2025-12-30 18:00:00
  Purpose: Batch process protein files through LA4SR for algal classification

Usage:
  # Sequential processing (1 GPU)
  python run_la4sr_batch_20251230_180000.py --filelist la4sr_protein_filelist.txt --model tools/la4sr/ckpt.pt

  # Parallel processing (4 GPUs)
  python run_la4sr_batch_20251230_180000.py --filelist la4sr_protein_filelist.txt --model tools/la4sr/ckpt.pt --gpus 4
"""

import os
import sys
import argparse
import subprocess
import gzip
import tempfile
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from typing import List, Tuple

def check_requirements():
    """
    Check if LA4SR model and inference script exist.
    """
    errors = []

    # Check for model checkpoint
    model_path = Path("tools/la4sr/ckpt.pt")
    if not model_path.exists():
        errors.append(f"LA4SR model not found: {model_path}")

    # Check for inference script
    infer_script = Path("tools/la4sr/infer-SL1.py")
    if not infer_script.exists():
        errors.append(f"LA4SR inference script not found: {infer_script}")

    # Check for CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available - LA4SR will run on CPU (very slow)")
        else:
            print(f"CUDA available: {torch.cuda.device_count()} GPU(s) detected")
    except ImportError:
        errors.append("PyTorch not installed - required for LA4SR")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

def load_filelist(filelist_path: Path) -> List[Path]:
    """
    Load protein file paths from filelist.

    Args:
        filelist_path: Path to filelist.txt

    Returns:
        List of protein file paths
    """
    protein_files = []

    with open(filelist_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                file_path = Path(line)
                if file_path.exists():
                    protein_files.append(file_path)
                else:
                    print(f"WARNING: File not found: {file_path}", file=sys.stderr)

    return protein_files

def run_la4sr_single(
    protein_file: Path,
    model_path: Path,
    output_dir: Path,
    gpu_id: int = 0
) -> Tuple[bool, str]:
    """
    Run LA4SR on a single protein file.

    Args:
        protein_file: Input protein FASTA file (can be .gz compressed)
        model_path: LA4SR model checkpoint
        output_dir: Output directory for results
        gpu_id: GPU device ID (for CUDA_VISIBLE_DEVICES)

    Returns:
        Tuple of (success: bool, message: str)
    """
    # Create output directory structure matching input
    rel_path = protein_file.relative_to(Path.cwd()) if protein_file.is_relative_to(Path.cwd()) else protein_file
    output_subdir = output_dir / rel_path.parent.name
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Output file path (remove .gz if present)
    base_stem = protein_file.stem
    if protein_file.suffix == '.gz':
        base_stem = Path(base_stem).stem  # Remove one more extension
    output_file = output_subdir / f"{base_stem}_la4sr_results.tsv"

    # Skip if already processed
    if output_file.exists():
        return True, f"Already processed: {protein_file.name}"

    # Handle compressed files
    temp_file = None
    input_file = protein_file

    try:
        if protein_file.suffix == '.gz':
            # Decompress to temporary file
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.faa',
                delete=False
            )

            with gzip.open(protein_file, 'rt') as gz_in:
                temp_file.write(gz_in.read())

            temp_file.close()
            input_file = Path(temp_file.name)

        # Prepare command
        cmd = [
            "python3",
            str(Path("tools/la4sr/infer-SL1.py")),
            str(model_path),
            str(input_file)
        ]

        # Set GPU device
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # Run LA4SR
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per file
        )

        if result.returncode == 0:
            # Save output
            with open(output_file, 'w') as f:
                f.write(f"# LA4SR Results\n")
                f.write(f"# Input: {protein_file}\n")
                f.write(f"# Model: {model_path}\n")
                f.write(f"# Date: {datetime.now().isoformat()}\n")
                f.write(f"# GPU: {gpu_id}\n")
                f.write(f"# Compressed: {protein_file.suffix == '.gz'}\n")
                f.write("#\n")
                f.write(result.stdout)

            return True, f"Success: {protein_file.name}"
        else:
            error_msg = f"Failed: {protein_file.name}\n{result.stderr}"
            return False, error_msg

    except subprocess.TimeoutExpired:
        return False, f"Timeout: {protein_file.name}"
    except Exception as e:
        return False, f"Error processing {protein_file.name}: {str(e)}"
    finally:
        # Clean up temporary file
        if temp_file and Path(temp_file.name).exists():
            os.unlink(temp_file.name)

def worker_process(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    model_path: Path,
    output_dir: Path,
    gpu_id: int
):
    """
    Worker process for parallel LA4SR processing.

    Args:
        task_queue: Queue of protein files to process
        result_queue: Queue for results
        model_path: LA4SR model checkpoint
        output_dir: Output directory
        gpu_id: GPU device ID
    """
    while True:
        protein_file = task_queue.get()
        if protein_file is None:  # Sentinel value to stop
            break

        success, message = run_la4sr_single(protein_file, model_path, output_dir, gpu_id)
        result_queue.put((success, message))

def run_batch_sequential(
    protein_files: List[Path],
    model_path: Path,
    output_dir: Path
):
    """
    Run LA4SR sequentially on a single GPU.

    Args:
        protein_files: List of protein files to process
        model_path: LA4SR model checkpoint
        output_dir: Output directory
    """
    print(f"\nProcessing {len(protein_files)} files sequentially on GPU 0...")
    print(f"Output directory: {output_dir}")
    print("="*70)

    successes = 0
    failures = 0

    for i, protein_file in enumerate(protein_files, 1):
        print(f"[{i}/{len(protein_files)}] Processing {protein_file.name}...")

        success, message = run_la4sr_single(protein_file, model_path, output_dir, gpu_id=0)

        if success:
            successes += 1
            print(f"  ✓ {message}")
        else:
            failures += 1
            print(f"  ✗ {message}", file=sys.stderr)

    print("="*70)
    print(f"Complete: {successes} succeeded, {failures} failed")

def run_batch_parallel(
    protein_files: List[Path],
    model_path: Path,
    output_dir: Path,
    num_gpus: int
):
    """
    Run LA4SR in parallel across multiple GPUs.

    Args:
        protein_files: List of protein files to process
        model_path: LA4SR model checkpoint
        output_dir: Output directory
        num_gpus: Number of GPUs to use
    """
    print(f"\nProcessing {len(protein_files)} files in parallel on {num_gpus} GPUs...")
    print(f"Output directory: {output_dir}")
    print("="*70)

    # Create task and result queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Add all files to task queue
    for protein_file in protein_files:
        task_queue.put(protein_file)

    # Add sentinel values to stop workers
    for _ in range(num_gpus):
        task_queue.put(None)

    # Start worker processes
    workers = []
    for gpu_id in range(num_gpus):
        worker = mp.Process(
            target=worker_process,
            args=(task_queue, result_queue, model_path, output_dir, gpu_id)
        )
        worker.start()
        workers.append(worker)

    # Collect results
    successes = 0
    failures = 0

    for i in range(len(protein_files)):
        success, message = result_queue.get()

        if success:
            successes += 1
            print(f"[{i+1}/{len(protein_files)}] ✓ {message}")
        else:
            failures += 1
            print(f"[{i+1}/{len(protein_files)}] ✗ {message}", file=sys.stderr)

    # Wait for all workers to finish
    for worker in workers:
        worker.join()

    print("="*70)
    print(f"Complete: {successes} succeeded, {failures} failed")

def main():
    parser = argparse.ArgumentParser(
        description="Batch process protein files through LA4SR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequential processing on 1 GPU
  python run_la4sr_batch_20251230_180000.py \\
      --filelist 03_analyses/la4sr_inputs/la4sr_protein_filelist.txt \\
      --model tools/la4sr/ckpt.pt \\
      --output 03_analyses/la4sr_results

  # Parallel processing on 4 GPUs
  python run_la4sr_batch_20251230_180000.py \\
      --filelist 03_analyses/la4sr_inputs/la4sr_protein_filelist.txt \\
      --model tools/la4sr/ckpt.pt \\
      --output 03_analyses/la4sr_results \\
      --gpus 4
        """
    )

    parser.add_argument(
        '--filelist',
        type=Path,
        required=True,
        help='Path to filelist.txt containing protein file paths'
    )

    parser.add_argument(
        '--model',
        type=Path,
        required=True,
        help='Path to LA4SR model checkpoint (ckpt.pt)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('03_analyses/la4sr_results'),
        help='Output directory for LA4SR results (default: 03_analyses/la4sr_results)'
    )

    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='Number of GPUs to use for parallel processing (default: 1)'
    )

    args = parser.parse_args()

    # Check requirements
    check_requirements()

    # Verify inputs exist
    if not args.filelist.exists():
        print(f"ERROR: Filelist not found: {args.filelist}", file=sys.stderr)
        sys.exit(1)

    if not args.model.exists():
        print(f"ERROR: Model checkpoint not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load protein files
    print(f"Loading filelist: {args.filelist}")
    protein_files = load_filelist(args.filelist)
    print(f"Found {len(protein_files)} protein files to process")

    if len(protein_files) == 0:
        print("ERROR: No valid protein files found in filelist", file=sys.stderr)
        sys.exit(1)

    # Run batch processing
    start_time = datetime.now()

    if args.gpus == 1:
        run_batch_sequential(protein_files, args.model, args.output)
    else:
        run_batch_parallel(protein_files, args.model, args.output, args.gpus)

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\nTotal processing time: {duration}")
    print(f"Average time per file: {duration / len(protein_files)}")
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
