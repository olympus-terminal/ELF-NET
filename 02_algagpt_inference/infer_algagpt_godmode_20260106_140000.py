#!/usr/bin/env python3
"""
algaGPT GODMODE Inference - Batched GPU inference for nanoGPT model

Processes protein FASTA files through algaGPT to classify as algae/conta.
Uses batched inference for GPU efficiency.

Usage:
    python infer_algagpt_godmode_20260106_140000.py <model_dir> <output_dir> <file1.fa> [file2.fa ...]

Output format (TSV):
    sequence_id<TAB>classification

Where classification contains <|label|>algae or <|label|>conta
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

def load_fasta_as_prompts(file_paths):
    """Read FASTA files and convert to prompt format (SEQUENCE<)"""
    sequences = []
    ids = []
    source_files = []

    for file_path in file_paths:
        current_header = None
        seq_buffer = []
        basename = os.path.basename(file_path)

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('>'):
                    if current_header and seq_buffer:
                        seq = "".join(seq_buffer)
                        # Prompt format: SEQUENCE<
                        sequences.append(seq + "<")
                        ids.append(current_header)
                        source_files.append(basename)
                    current_header = line[1:].split()[0]
                    seq_buffer = []
                else:
                    seq_buffer.append(line)

            # Last sequence
            if current_header and seq_buffer:
                seq = "".join(seq_buffer)
                sequences.append(seq + "<")
                ids.append(current_header)
                source_files.append(basename)

    return sequences, ids, source_files

@torch.inference_mode()
def run_inference():
    if len(sys.argv) < 4:
        print("Usage: python infer_algagpt_godmode.py <model_dir> <output_dir> <file1.fa> [file2.fa ...]")
        sys.exit(1)

    model_dir = sys.argv[1]
    output_dir = sys.argv[2]
    input_files = sys.argv[3:]

    # Configuration
    batch_size = 64  # Smaller batches for character-level model
    max_new_tokens = 14  # Enough for |label|>algae or |label|>conta
    temperature = 0.1
    top_k = 10
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

    print(f"=" * 70, file=sys.stderr)
    print(f"algaGPT GODMODE Inference", file=sys.stderr)
    print(f"Model: {model_dir}", file=sys.stderr)
    print(f"Batch size: {batch_size}", file=sys.stderr)
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

    # Load sequences
    print("Loading sequences...", file=sys.stderr)
    sequences, ids, source_files = load_fasta_as_prompts(input_files)
    print(f"Loaded {len(sequences):,} sequences from {len(input_files)} files", file=sys.stderr)

    if len(sequences) == 0:
        print("ERROR: No sequences loaded!", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Group sequences by source file for output
    file_results = {}
    for i, src in enumerate(source_files):
        if src not in file_results:
            file_results[src] = []
        file_results[src].append(i)

    print(f"Starting inference...", file=sys.stderr)
    start_time = time.time()
    total_processed = 0
    results = [''] * len(sequences)  # Pre-allocate results

    # Process in batches
    with ctx:
        for batch_start in range(0, len(sequences), batch_size):
            batch_end = min(batch_start + batch_size, len(sequences))
            batch_seqs = sequences[batch_start:batch_end]
            batch_ids = ids[batch_start:batch_end]

            # Encode batch - variable length sequences
            encoded_batch = [encode(s) for s in batch_seqs]

            # Pad to max length in batch
            max_len = max(len(e) for e in encoded_batch)
            padded = torch.zeros(len(encoded_batch), max_len, dtype=torch.long, device=device)
            for i, enc in enumerate(encoded_batch):
                padded[i, :len(enc)] = torch.tensor(enc, dtype=torch.long)

            # Generate (one sequence at a time for variable-length handling)
            for i in range(len(batch_seqs)):
                x = padded[i:i+1, :len(encoded_batch[i])]
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                output_text = decode(y[0].tolist())

                # Extract just the generated part (after input)
                input_len = len(batch_seqs[i])
                generated = output_text[input_len:] if len(output_text) > input_len else output_text

                results[batch_start + i] = generated

            total_processed += len(batch_seqs)

            # Progress
            if total_processed % 1000 == 0 or batch_end == len(sequences):
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                print(f"\rProcessed {total_processed:,} / {len(sequences):,} ({rate:.0f} seqs/s)",
                      end='', file=sys.stderr)

    print(f"", file=sys.stderr)

    # Write results grouped by source file
    print(f"Writing results...", file=sys.stderr)
    for src_file, indices in file_results.items():
        # Output filename: original.fa -> original.algagpt.tsv
        out_basename = src_file.replace('.fa', '').replace('.algal', '') + '_algagpt.tsv'
        out_path = os.path.join(output_dir, out_basename)

        with open(out_path, 'w') as f:
            for idx in indices:
                f.write(f"{ids[idx]}\t{results[idx]}\n")

    total_time = time.time() - start_time
    final_rate = total_processed / total_time if total_time > 0 else 0

    print(f"=" * 70, file=sys.stderr)
    print(f"COMPLETE: {total_processed:,} seqs in {total_time:.1f}s = {final_rate:.0f} seqs/s", file=sys.stderr)
    print(f"Output files: {len(file_results)}", file=sys.stderr)
    print(f"=" * 70, file=sys.stderr)

if __name__ == "__main__":
    run_inference()
