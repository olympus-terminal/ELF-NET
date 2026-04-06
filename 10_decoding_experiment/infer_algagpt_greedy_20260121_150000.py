#!/usr/bin/env python3
"""
infer_algagpt_greedy_20260121_150000.py — AlgaGPT inference with GREEDY decoding

This script is a modified version of infer_TI-inc-algaGPT.py that uses greedy
decoding (argmax) instead of multinomial sampling.

Key difference from baseline:
- Baseline: torch.multinomial(probs, num_samples=1) with temp=0.1, top_k=10
- This script: torch.argmax(probs, dim=-1, keepdim=True) - deterministic

This modification is for the decoding strategy controlled experiment to test
whether the difference in recall between AlgaGPT and AlKhidr is due to
decoding strategy rather than model architecture.

Provenance:
  Original: tools/la4sr/infer_TI-inc-algaGPT.py
  Modified: 2026-01-21
  Experiment: decoding_experiment (Task 5)
"""

import os
import sys
import argparse
import pickle
import random
from datetime import datetime
from typing import Iterator, Tuple

# ---------------------------------------------------------------------------
# Python < 3.7 compatibility: provide a fallback for contextlib.nullcontext
# ---------------------------------------------------------------------------
try:
    from contextlib import nullcontext  # Python >=3.7
except ImportError:
    class _NullContext:
        def __init__(self, result=None):
            self.result = result
        def __enter__(self):
            return self.result
        def __exit__(self, *exc):
            return False
    nullcontext = _NullContext

import torch
import torch.nn.functional as F

# Add the tools/la4sr directory to path to import model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LA4SR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))), 'tools', 'la4sr')
sys.path.insert(0, LA4SR_DIR)

from model import GPTConfig, GPT

# --------------------------------------------------------------------------
#                             FASTA reader                                    #
# --------------------------------------------------------------------------

def stream_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (header, sequence) tuples, collapsing wrapped lines."""
    header, seq_chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq_chunks)
                header = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, ''.join(seq_chunks)

# --------------------------------------------------------------------------
#                    GREEDY GENERATE (KEY MODIFICATION)                       #
# --------------------------------------------------------------------------

@torch.no_grad()
def generate_greedy(model, idx, max_new_tokens):
    """
    Greedy decoding: Take conditioning sequence idx and complete it by always
    selecting the highest probability token (argmax).

    This is DETERMINISTIC - running twice produces identical output.

    Key difference from model.generate():
    - Original: idx_next = torch.multinomial(probs, num_samples=1)  [stochastic]
    - Here:     idx_next = torch.argmax(probs, dim=-1, keepdim=True) [deterministic]

    Args:
        model: The GPT model
        idx: Input token indices (LongTensor of shape (b, t))
        max_new_tokens: Number of tokens to generate

    Returns:
        Extended sequence with generated tokens
    """
    for _ in range(max_new_tokens):
        # Crop context if too long
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

        # Forward pass to get logits
        logits, _ = model(idx_cond)

        # Get logits for the last position (no temperature scaling for greedy)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities (not strictly needed for argmax, but kept for consistency)
        probs = F.softmax(logits, dim=-1)

        # GREEDY: Select the token with highest probability
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# --------------------------------------------------------------------------
#                          argument parsing                                   #
# --------------------------------------------------------------------------

def get_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AlgaGPT GREEDY decoding inference (FASTA to TSV)",
        epilog="For decoding strategy controlled experiment"
    )
    # model/runtime
    p.add_argument('--init_from', default='resume',
                   choices=['resume', 'gpt2', 'gpt2-medium', 'gpt2-large'],
                   help='Model source; "resume" = local ckpt.pt')
    p.add_argument('--out_dir', default=None,
                   help='Directory with ckpt.pt (default: tools/la4sr)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='float16',
                   choices=['float32', 'bfloat16', 'float16'])
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--compile', action='store_true')
    # generation knobs
    p.add_argument('--max_new_tokens', type=int, default=14)
    # NOTE: temperature and top_k are NOT used in greedy decoding
    # I/O
    p.add_argument('fasta_in', help='Input FASTA')
    p.add_argument('-o', '--tsv_out', required=True,
                   help='Output TSV file path')
    return p.parse_args()

# --------------------------------------------------------------------------
#                                 MAIN                                        #
# --------------------------------------------------------------------------

def main():
    args = get_cli()

    # Default out_dir to LA4SR directory
    if args.out_dir is None:
        args.out_dir = LA4SR_DIR

    print(f"=" * 70)
    print(f"AlgaGPT GREEDY Decoding Inference")
    print(f"=" * 70)
    print(f"Decoding strategy: GREEDY (argmax, deterministic)")
    print(f"Input FASTA: {args.fasta_in}")
    print(f"Output TSV: {args.tsv_out}")
    print(f"Model dir: {args.out_dir}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"=" * 70)

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device and dtype setup
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype_map = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }
    ptdtype = ptdtype_map[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    ###########################################################################
    #                           model loading                                 #
    ###########################################################################

    if args.init_from == 'resume':
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        print(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        # Strip DDP prefixes if present
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model'].items()}
        model.load_state_dict(state_dict)
        print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    else:
        model = GPT.from_pretrained(args.init_from, dict(dropout=0.0))

    model.to(args.device).eval()
    if args.compile:
        model = torch.compile(model)

    ###########################################################################
    #                     encoding / decoding setup                           #
    ###########################################################################

    # Check for meta.pkl in out_dir
    meta_path = os.path.join(args.out_dir, 'meta.pkl')

    if os.path.exists(meta_path):
        print(f"Loading vocabulary from: {meta_path}")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        UNK_ID = stoi.get('<unk>', 0)
        encode = lambda s: [stoi.get(c, UNK_ID) for c in s]
        decode = lambda l: ''.join(itos[i] for i in l)
        print(f"Vocabulary size: {len(stoi)}")
    else:
        print("[WARN] No meta.pkl found, using tiktoken GPT2 encoding")
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
        encode = lambda s: enc.encode(s, allowed_special={""})
        decode = lambda l: enc.decode(l)

    ###########################################################################
    #                            inference loop                               #
    ###########################################################################

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.tsv_out)), exist_ok=True)

    n_seqs = 0
    start_time = datetime.now()

    with open(args.tsv_out, 'w') as tsv, torch.no_grad(), ctx:
        # Write header with provenance
        tsv.write('# Provenance:\n')
        tsv.write(f'#   Script: {os.path.abspath(__file__)}\n')
        tsv.write(f'#   Input: {os.path.abspath(args.fasta_in)}\n')
        tsv.write(f'#   Date: {start_time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        tsv.write(f'#   Decoding: GREEDY (argmax)\n')
        tsv.write(f'#   Seed: {args.seed}\n')
        tsv.write('record_id\tsequence\tmodel_output\n')

        for rid, seq in stream_fasta(args.fasta_in):
            if not seq:
                print(f"[WARN] empty sequence for {rid}; skipping", file=sys.stderr)
                continue

            x = torch.tensor(encode(seq), dtype=torch.long, device=args.device).unsqueeze(0)

            try:
                # Use GREEDY generation
                y = generate_greedy(model, x, args.max_new_tokens)
                cont = decode(y[0].tolist())
            except Exception as e:
                print(f"[ERR] generation failed on {rid}: {e}", file=sys.stderr)
                cont = ''

            tsv.write(f"{rid}\t{seq}\t{cont}\n")
            n_seqs += 1

            if n_seqs % 50 == 0:
                print(f"  Processed {n_seqs} sequences...")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 70}")
    print(f"Completed: {n_seqs} sequences in {elapsed:.1f}s ({n_seqs/elapsed:.2f} seq/s)")
    print(f"Output saved to: {args.tsv_out}")
    print(f"{'=' * 70}")

if __name__ == '__main__':
    main()
