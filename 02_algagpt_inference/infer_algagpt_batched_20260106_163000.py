#!/usr/bin/env python3
"""
algaGPT Batched Inference - True batched GPU inference with DataLoader

Processes protein FASTA files through algaGPT using batched generation.
Uses left-padding for variable-length sequences so batch generation works correctly.

Key optimizations:
- DataLoader with num_workers for parallel data loading
- Left-padding enables true batched generation
- Per-file processing for memory efficiency and resumability
- Skip completed files

Usage:
    python infer_algagpt_batched.py <model_dir> <output_dir> <file1.fa> [file2.fa ...]
"""
import sys
import os
import time
import pickle
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext

# Add model directory to path for model.py import
MODEL_BASE = "/scratch/drn2/PROJECTS/AI/KouroshModels/algaGPT2-S_set4_running_interactive_David-clone2-ht"
sys.path.insert(0, MODEL_BASE)

from model import GPTConfig, GPT

class ProteinDataset(Dataset):
    """Dataset for protein sequences with encoding in __getitem__"""

    def __init__(self, sequences, ids, stoi):
        self.sequences = sequences
        self.ids = ids
        self.stoi = stoi

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Encode: convert characters to token indices
        encoded = [self.stoi.get(c, 0) for c in seq]
        return {
            'tokens': torch.tensor(encoded, dtype=torch.long),
            'id': self.ids[idx],
            'length': len(encoded)
        }

def collate_left_pad(batch, pad_token=0):
    """
    Collate function with LEFT padding.

    Left-padding ensures all sequences end at the same position,
    which is critical for batched autoregressive generation.
    """
    tokens_list = [item['tokens'] for item in batch]
    ids = [item['id'] for item in batch]
    lengths = [item['length'] for item in batch]

    max_len = max(lengths)

    # Left-pad: pad at the beginning
    padded = torch.full((len(batch), max_len), pad_token, dtype=torch.long)
    for i, tokens in enumerate(tokens_list):
        # Place tokens at the END (right-aligned)
        padded[i, max_len - len(tokens):] = tokens

    return {
        'input_ids': padded,
        'ids': ids,
        'lengths': lengths
    }

def load_fasta_file(file_path):
    """Read FASTA file and convert to prompt format (SEQUENCE<)"""
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
                    sequences.append(seq + "<")  # Add prompt suffix
                    ids.append(current_header)
                current_header = line[1:].split()[0]
                seq_buffer = []
            else:
                seq_buffer.append(line)

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
def batched_generate(model, input_ids, max_new_tokens, temperature=0.1, top_k=10):
    """
    Batched generation for left-padded sequences.

    Since sequences are left-padded, all sequences end at position -1,
    so standard batched generation works correctly.
    """
    idx = input_ids

    for _ in range(max_new_tokens):
        # Crop to block_size if needed
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

        # Forward pass
        logits, _ = model(idx_cond)

        # Get logits at final position and scale by temperature
        logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Sample
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

@torch.inference_mode()
def run_inference():
    mp.set_start_method('spawn', force=True)

    if len(sys.argv) < 4:
        print("Usage: python infer_algagpt_batched.py <model_dir> <output_dir> <file1.fa> [file2.fa ...]")
        sys.exit(1)

    model_dir = sys.argv[1]
    output_dir = sys.argv[2]
    input_files = sys.argv[3:]

    # Configuration
    batch_size = 256  # Larger batches for efficiency
    max_new_tokens = 14  # Enough for |label|>algae or |label|>conta
    temperature = 0.1
    top_k = 10
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    num_workers = min(8, os.cpu_count() or 4)

    print(f"=" * 70, file=sys.stderr)
    print(f"algaGPT Batched Inference (GODMODE)", file=sys.stderr)
    print(f"Model: {model_dir}", file=sys.stderr)
    print(f"Batch size: {batch_size}", file=sys.stderr)
    print(f"Input files: {len(input_files)}", file=sys.stderr)
    print(f"DataLoader workers: {num_workers}", file=sys.stderr)
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

    # Load tokenizer
    print("Loading tokenizer...", file=sys.stderr)
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join(MODEL_BASE, 'data', checkpoint['config']['dataset'], 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            decode = lambda l: ''.join([itos[i] for i in l])
            print(f"Tokenizer loaded, vocab size: {len(stoi)}", file=sys.stderr)
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
            print(f"[{file_idx+1}/{len(input_files)}] SKIP: {basename}", file=sys.stderr)
            files_skipped += 1
            continue

        # Load sequences
        sequences, ids = load_fasta_file(input_file)

        if not sequences:
            print(f"[{file_idx+1}/{len(input_files)}] EMPTY: {basename}", file=sys.stderr)
            continue

        print(f"[{file_idx+1}/{len(input_files)}] {basename}: {len(sequences):,} seqs...",
              file=sys.stderr, end='', flush=True)

        file_start = time.time()

        # Create dataset and dataloader
        dataset = ProteinDataset(sequences, ids, stoi)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_left_pad,
            pin_memory=True,
            drop_last=False
        )

        results = []

        with ctx:
            for batch in loader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                batch_ids = batch['ids']
                lengths = batch['lengths']

                # Generate
                outputs = batched_generate(
                    model, input_ids, max_new_tokens,
                    temperature=temperature, top_k=top_k
                )

                # Decode and extract generated portion
                for i, (seq_id, orig_len) in enumerate(zip(batch_ids, lengths)):
                    full_output = outputs[i].tolist()
                    # The original sequence is at positions [max_len - orig_len : max_len]
                    # Generated tokens start at position max_len
                    max_len = input_ids.size(1)
                    generated_tokens = full_output[max_len:]  # Only new tokens
                    generated_text = decode(generated_tokens)
                    results.append((seq_id, generated_text))

        # Write results
        with open(output_path, 'w') as f:
            for seq_id, classification in results:
                f.write(f"{seq_id}\t{classification}\n")

        file_time = time.time() - file_start
        rate = len(sequences) / file_time if file_time > 0 else 0
        print(f" {rate:.0f} seq/s", file=sys.stderr)

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
