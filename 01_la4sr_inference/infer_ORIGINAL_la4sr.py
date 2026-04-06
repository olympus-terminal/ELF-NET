#!/usr/bin/env python3
import sys
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

def generate_output(input_text, model, tokenizer):
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to('cuda')
    attention_mask = inputs["attention_mask"].to('cuda')

    # Generate output
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=15)

    # Decode the output tokens back to text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <model_name_or_path> <input_fasta>")
        sys.exit(1)

    model_name_or_path = sys.argv[1]
    input_file = sys.argv[2]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        "hmbyt5/byt5-small-english", use_fast=True, padding_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, is_decoder=True
    ).to('cuda')

    # Iterate through FASTA records and run inference on each sequence
    for seq_id, sequence in parse_fasta(input_file):
        output_text = generate_output(sequence, model, tokenizer)
        print(f"{seq_id}\t{output_text}")

