#!/usr/bin/env python3
"""
AlKhidr Sampling Inference Script for Decoding Strategy Experiment

This script modifies the original AlKhidr greedy decoding to use multinomial sampling
with the same parameters as AlgaGPT (temperature=0.1, top_k=10).

Original: tools/la4sr/TI-free-la4sr/infer-ByT5tok-attn-fastaParser.py
Modification: Line 39 model.generate() now uses do_sample=True, temperature=0.1, top_k=10

Purpose: Test whether AlKhidr recall improves when using sampling decoding instead of greedy.

Usage:
    python infer_alkhidr_sampling_20260121_160000.py <model_path> <input_fasta> [-o output.tsv] [--seed SEED]

Output format: TSV with columns: record_id, sequence, model_output
Classification markers: @ = microalgae, ! = contamination

Provenance:
    Script: /media/drn2/External/TARA-Oceans/03_analyses/decoding_experiment/scripts/infer_alkhidr_sampling_20260121_160000.py
    Based on: /media/drn2/External/TARA-Oceans/tools/la4sr/TI-free-la4sr/infer-ByT5tok-attn-fastaParser.py
    Date: 2026-01-21
    Experiment: Decoding Strategy Controlled Experiment
"""

import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_fasta(file_path):
    """
    Simple FASTA parser: yields (seq_id, full_sequence) pairs,
    collapsing wrapped lines and skipping headers.
    """
    header = None
    seq_chunks = []
    with open(file_path, 'r') as f:
        for line in f:
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
        # yield last record
        if header is not None:
            yield header, ''.join(seq_chunks)

def generate_output_sampling(input_text, model, tokenizer, temperature=0.1, top_k=10):
    """
    Generate output using multinomial SAMPLING decoding.

    Key difference from original greedy:
    - do_sample=True enables multinomial sampling
    - temperature=0.1 makes distribution sharper (closer to greedy but with some variance)
    - top_k=10 restricts sampling to top 10 tokens

    These parameters match AlgaGPT's sampling configuration.
    """
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to('cuda')
    attention_mask = inputs["attention_mask"].to('cuda')

    # Generate output with SAMPLING (not greedy)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=15,
            do_sample=True,           # Enable multinomial sampling
            temperature=temperature,   # Control distribution sharpness
            top_k=top_k               # Restrict to top-k tokens
        )

    # Decode the output tokens back to text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(
        description='AlKhidr inference with SAMPLING decoding (temp=0.1, top_k=10)'
    )
    parser.add_argument('model_path', help='Path to AlKhidr model checkpoint')
    parser.add_argument('input_fasta', help='Input FASTA file')
    parser.add_argument('-o', '--output', default=None, help='Output TSV file (default: stdout)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--temperature', type=float, default=0.1, help='Sampling temperature (default: 0.1)')
    parser.add_argument('--top_k', type=int, default=10, help='Top-k sampling (default: 10)')

    args = parser.parse_args()

    # Set random seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        print(f"# Random seed set to: {args.seed}", file=sys.stderr)

    print(f"# AlKhidr Sampling Inference", file=sys.stderr)
    print(f"# Model: {args.model_path}", file=sys.stderr)
    print(f"# Input: {args.input_fasta}", file=sys.stderr)
    print(f"# Decoding: do_sample=True, temperature={args.temperature}, top_k={args.top_k}", file=sys.stderr)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        "hmbyt5/byt5-small-english", use_fast=True, padding_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, is_decoder=True
    ).to('cuda')

    model.eval()

    # Open output file if specified
    out_file = open(args.output, 'w') if args.output else sys.stdout

    # Write header
    print("record_id\tsequence\tmodel_output", file=out_file)

    # Iterate through FASTA records and run inference on each sequence
    count = 0
    for seq_id, sequence in parse_fasta(args.input_fasta):
        output_text = generate_output_sampling(
            sequence, model, tokenizer,
            temperature=args.temperature,
            top_k=args.top_k
        )
        print(f"{seq_id}\t{sequence}\t{output_text}", file=out_file)
        count += 1

        if count % 50 == 0:
            print(f"# Processed {count} sequences...", file=sys.stderr)

    print(f"# Complete: {count} sequences processed", file=sys.stderr)

    if args.output:
        out_file.close()
        print(f"# Output written to: {args.output}", file=sys.stderr)

if __name__ == "__main__":
    main()
