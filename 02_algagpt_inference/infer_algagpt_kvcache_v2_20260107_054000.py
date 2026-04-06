#!/usr/bin/env python3
"""
algaGPT KV-Cached Inference v2 - Per-sequence KV caching (no padding issues)

Processes sequences individually with KV caching for correct output.
Each sequence gets ~5x speedup from caching, no batching artifacts.

Usage:
    python infer_algagpt_kvcache_v2.py <model_dir> <output_dir> <file1.fa> [file2.fa ...]
"""
import sys
import os
import time
import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from contextlib import nullcontext

MODEL_BASE = "/scratch/drn2/PROJECTS/AI/KouroshModels/algaGPT2-S_set4_running_interactive_David-clone2-ht"

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttentionKV(nn.Module):
    """Attention with KV caching - for single sequence processing"""

    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, kv_cache=None):
        B, T, C = x.size()
        head_dim = C // self.n_head

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        new_kv_cache = (k, v)

        # For single token generation (T=1), no causal mask needed
        # For initial encoding (T>1), use causal mask
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None and T > 1))
        else:
            scale = 1.0 / math.sqrt(k.size(-1))
            att = (q @ k.transpose(-2, -1)) * scale
            if kv_cache is None and T > 1:
                causal_mask = torch.tril(torch.ones(T, T, device=x.device))
                att = att.masked_fill(causal_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, new_kv_cache

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class BlockKV(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionKV(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.ln_1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache

class GPTKV(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([BlockKV(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

    def forward(self, idx, kv_caches=None, start_pos=0):
        B, T = idx.size()
        pos = torch.arange(start_pos, start_pos + T, dtype=torch.long, device=idx.device)

        x = self.wte(idx) + self.wpe(pos)

        new_caches = []
        for i, block in enumerate(self.h):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = block(x, cache)
            new_caches.append(new_cache)

        logits = self.lm_head(self.ln_f(x))
        return logits, new_caches

    @classmethod
    def from_pretrained_nanoGPT(cls, checkpoint_path, device='cuda'):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = GPTConfig(**checkpoint['model_args'])
        model = cls(config)

        state_dict = {}
        for k, v in checkpoint['model'].items():
            if k.startswith('_orig_mod.'):
                k = k[len('_orig_mod.'):]
            if k.startswith('transformer.'):
                k = k[len('transformer.'):]
            state_dict[k] = v

        model.load_state_dict(state_dict)
        model.eval().to(device)
        return model, checkpoint

@torch.inference_mode()
def generate_single_kv(model, tokens, max_new_tokens, temperature=0.1, top_k=10):
    """Generate for a single sequence with KV caching"""
    # Initial forward - encode full sequence
    idx = tokens.unsqueeze(0)  # (1, T)
    logits, kv_caches = model(idx, kv_caches=None, start_pos=0)
    current_pos = idx.size(1)

    generated = []
    for _ in range(max_new_tokens):
        next_logits = logits[0, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[-1]] = -float('Inf')

        probs = F.softmax(next_logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        generated.append(idx_next.item())

        # Forward only new token with cache
        logits, kv_caches = model(idx_next.unsqueeze(0), kv_caches=kv_caches, start_pos=current_pos)
        current_pos += 1

    return generated

def load_fasta_file(file_path):
    sequences, ids = [], []
    current_header, seq_buffer = None, []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('>'):
                if current_header and seq_buffer:
                    sequences.append("".join(seq_buffer) + "<")
                    ids.append(current_header)
                current_header = line[1:].split()[0]
                seq_buffer = []
            else:
                seq_buffer.append(line)

        if current_header and seq_buffer:
            sequences.append("".join(seq_buffer) + "<")
            ids.append(current_header)

    return sequences, ids

def get_output_filename(input_basename, output_dir):
    out_basename = input_basename.replace('.fa', '').replace('.algal', '') + '_algagpt.tsv'
    return os.path.join(output_dir, out_basename)

@torch.inference_mode()
def run_inference():
    if len(sys.argv) < 4:
        print("Usage: python infer_algagpt_kvcache_v2.py <model_dir> <output_dir> <file1.fa> [...]")
        sys.exit(1)

    model_dir = sys.argv[1]
    output_dir = sys.argv[2]
    input_files = sys.argv[3:]

    max_new_tokens = 14
    temperature = 0.1
    top_k = 10
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

    print(f"=" * 70, file=sys.stderr)
    print(f"algaGPT KV-Cached Inference v2 (TURBO)", file=sys.stderr)
    print(f"Model: {model_dir}", file=sys.stderr)
    print(f"Input files: {len(input_files)}", file=sys.stderr)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", file=sys.stderr)
    print(f"=" * 70, file=sys.stderr)

    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)

    print("Loading KV-cached model...", file=sys.stderr)
    ckpt_path = os.path.join(model_dir, 'ckpt.pt')
    model, checkpoint = GPTKV.from_pretrained_nanoGPT(ckpt_path, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters", file=sys.stderr)

    print("Loading tokenizer...", file=sys.stderr)
    meta_path = os.path.join(MODEL_BASE, 'data', checkpoint['config']['dataset'], 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    print(f"Tokenizer loaded, vocab size: {len(stoi)}", file=sys.stderr)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing files...", file=sys.stderr)
    start_time = time.time()
    total_seqs, files_processed, files_skipped = 0, 0, 0

    for file_idx, input_file in enumerate(input_files):
        basename = os.path.basename(input_file)
        output_path = get_output_filename(basename, output_dir)

        if os.path.exists(output_path):
            print(f"[{file_idx+1}/{len(input_files)}] SKIP: {basename}", file=sys.stderr)
            files_skipped += 1
            continue

        sequences, ids = load_fasta_file(input_file)
        if not sequences:
            print(f"[{file_idx+1}/{len(input_files)}] EMPTY: {basename}", file=sys.stderr)
            continue

        print(f"[{file_idx+1}/{len(input_files)}] {basename}: {len(sequences):,} seqs...",
              file=sys.stderr, end='', flush=True)

        file_start = time.time()
        results = []

        with ctx:
            for seq, seq_id in zip(sequences, ids):
                tokens = torch.tensor(encode(seq), dtype=torch.long, device=device)
                generated_tokens = generate_single_kv(
                    model, tokens, max_new_tokens, temperature, top_k
                )
                generated_text = decode(generated_tokens)
                results.append((seq_id, generated_text))

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

    print(f"\n" + "=" * 70, file=sys.stderr)
    print(f"COMPLETE: {files_processed} files, {total_seqs:,} seqs", file=sys.stderr)
    print(f"Skipped: {files_skipped}, Time: {total_time:.1f}s, Rate: {avg_rate:.0f} seq/s", file=sys.stderr)
    print(f"=" * 70, file=sys.stderr)

if __name__ == "__main__":
    run_inference()
