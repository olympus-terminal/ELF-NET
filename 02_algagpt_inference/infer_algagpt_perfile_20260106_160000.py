#!/usr/bin/env python3
"""
algaGPT Per-File Inference - Process one file at a time with skip logic

Processes protein FASTA files through algaGPT to classify as algae/conta.
Each file is processed independently with output written immediately.
Skips files that already have output (resumable).

Usage:
    python infer_algagpt_perfile.py <model_dir> <output_dir> <file1.fa> [file2.fa ...]

Output format (TSV):
    sequence_id<TAB>classification
"""
import sys
import os
import time
import pickle
import torch
from contextlib import nullcontext

# Add model directory to path for model.py import
MODEL_BASE = "/scratch/drn2/PROJECTS/AI/KouroshModels/algaGPT2-S_set4_running_interactive_David-clone2-ht"
sys.path.insert(0, MODEL_BASE)

from model import GPTConfig, GPT

def load_fasta_file(file_path):
    """Read single FASTA file and convert to prompt format (SEQUENCE<)"""
    sequences = []
    ids = []

    current_header = None
    seq_buffer = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('>'):
                if current_header and seq_buffer:
                    seq = "".join(seq_buffer)
                    sequences.append(seq + "<")
                    ids.append(current_header)
                current_header = line[1:].split()[0]
                seq_buffer = []
            else:
                seq_buffer.append(line)

        # Last sequence
        if current_header and seq_buffer:
            seq = "".join(seq_buffer)
            sequences.append(seq + "<")
            ids.append(current_header)

    return sequences, ids

def get_output_filename(input_basename, output_dir):
    """Convert input filename to output filename"""
    out_basename = input_basename.replace('.fa', '').replace('.algal', '') + '_algagpt.tsv'
    return os.path.join(output_dir, out_basename)

@torch.inference_mode()
def run_inference():
    if len(sys.argv) < 4:
        print("Usage: python infer_algagpt_perfile.py <model_dir> <output_dir> <file1.fa> [file2.fa ...]")
        sys.exit(1)

    model_dir = sys.argv[1]
    output_dir = sys.argv[2]
    input_files = sys.argv[3:]

    # Configuration
    max_new_tokens = 14  # Enough for |label|>algae or |label|>conta
    temperature = 0.1
    top_k = 10
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

    print(f"=" * 70, file=sys.stderr)
    print(f"algaGPT Per-File Inference", file=sys.stderr)
    print(f"Model: {model_dir}", file=sys.stderr)
    print(f"Input files: {len(input_files)}", file=sys.stderr)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", file=sys.stderr)
    print(f"=" * 70, file=sys.stderr)

    # Setup torch
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)

    # Load model checkpoint
    print("Loading model checkpoint...", file=sys.stderr)
    ckpt_path = os.path.join(model_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)

    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters", file=sys.stderr)

    # Load tokenizer (meta.pkl with stoi/itos)
    print("Loading tokenizer...", file=sys.stderr)
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join(MODEL_BASE, 'data', checkpoint['config']['dataset'], 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi.get(c, 0) for c in s]  # 0 for unknown chars
            decode = lambda l: ''.join([itos[i] for i in l])
            print(f"Custom tokenizer loaded, vocab size: {len(stoi)}", file=sys.stderr)
        else:
            print(f"ERROR: meta.pkl not found at {meta_path}", file=sys.stderr)
            sys.exit(1)
    else:
        print("ERROR: No dataset config in checkpoint", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each file
    print(f"", file=sys.stderr)
    print(f"Processing files...", file=sys.stderr)

    start_time = time.time()
    total_seqs = 0
    files_processed = 0
    files_skipped = 0

    for file_idx, input_file in enumerate(input_files):
        basename = os.path.basename(input_file)
        output_path = get_output_filename(basename, output_dir)

        # Skip if output already exists
        if os.path.exists(output_path):
            print(f"[{file_idx+1}/{len(input_files)}] SKIP (exists): {basename}", file=sys.stderr)
            files_skipped += 1
            continue

        # Load sequences from this file
        sequences, ids = load_fasta_file(input_file)

        if not sequences:
            print(f"[{file_idx+1}/{len(input_files)}] EMPTY: {basename}", file=sys.stderr)
            continue

        print(f"[{file_idx+1}/{len(input_files)}] {basename}: {len(sequences):,} seqs...", file=sys.stderr, end='', flush=True)

        file_start = time.time()
        results = []

        with ctx:
            for seq_idx, (seq, seq_id) in enumerate(zip(sequences, ids)):
                # Encode sequence
                encoded = encode(seq)
                x = torch.tensor([encoded], dtype=torch.long, device=device)

                # Generate
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                output_text = decode(y[0].tolist())

                # Extract just the generated part (after input)
                input_len = len(seq)
                generated = output_text[input_len:] if len(output_text) > input_len else output_text

                results.append((seq_id, generated))

        # Write results for this file
        with open(output_path, 'w') as f:
            for seq_id, classification in results:
                f.write(f"{seq_id}\t{classification}\n")

        file_time = time.time() - file_start
        rate = len(sequences) / file_time if file_time > 0 else 0
        print(f" done ({rate:.0f} seq/s)", file=sys.stderr)

        total_seqs += len(sequences)
        files_processed += 1

    total_time = time.time() - start_time
    avg_rate = total_seqs / total_time if total_time > 0 else 0

    print(f"", file=sys.stderr)
    print(f"=" * 70, file=sys.stderr)
    print(f"COMPLETE:", file=sys.stderr)
    print(f"  Files processed: {files_processed}", file=sys.stderr)
    print(f"  Files skipped: {files_skipped}", file=sys.stderr)
    print(f"  Total sequences: {total_seqs:,}", file=sys.stderr)
    print(f"  Total time: {total_time:.1f}s", file=sys.stderr)
    print(f"  Average rate: {avg_rate:.0f} seqs/s", file=sys.stderr)
    print(f"=" * 70, file=sys.stderr)

if __name__ == "__main__":
    run_inference()
