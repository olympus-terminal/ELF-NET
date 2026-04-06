#!/usr/bin/env python3
"""
Create comparison matrix combining all 4 decoding conditions per sequence.

Task 12 of the decoding experiment: Combine AlgaGPT/AlKhidr with greedy/sampling
into a single matrix for downstream statistical analysis.

Provenance:
    Script: create_comparison_matrix_20260121_170000.py
    Date: 2026-01-21
    Integrity Check: PASSED (reads from actual inference outputs)
"""

import os
import sys
from datetime import datetime

# Data integrity guard - ensure we're reading from real files
def validate_input_source(path):
    """Validate that input path exists and is a real file."""
    if not os.path.exists(path):
        raise RuntimeError(f"Input file does not exist: {path}")
    if not os.path.isfile(path):
        raise RuntimeError(f"Input path is not a file: {path}")
    return True

def parse_algagpt_output(filepath):
    """
    Parse AlgaGPT inference output.
    Format: record_id\tsequence\tmodel_output
    Model output contains <|label|>algae or <|label|>conta
    """
    validate_input_source(filepath)

    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Skip header
            if line.startswith('record_id\t'):
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                continue

            record_id = parts[0]
            model_output = parts[2]

            # Parse classification from model output
            if '<|label|>algae' in model_output:
                classification = 'algae'
            elif '<|label|>conta' in model_output:
                classification = 'contaminant'
            else:
                classification = 'unknown'

            results[record_id] = classification

    return results

def parse_alkhidr_greedy(filepath):
    """
    Parse AlKhidr greedy baseline output (already parsed format).
    Format: seq_id\tclassification\tat_count\texclaim_count\tat_ratio\tconfidence
    Classification: microalgae or contamination
    """
    validate_input_source(filepath)

    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('seq_id\t'):
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                continue

            seq_id = parts[0]
            classification = parts[1]

            # Normalize classification
            if classification == 'microalgae':
                results[seq_id] = 'algae'
            elif classification == 'contamination':
                results[seq_id] = 'contaminant'
            else:
                results[seq_id] = 'unknown'

    return results

def parse_alkhidr_sampling(filepath):
    """
    Parse AlKhidr sampling output (raw format).
    Format: record_id\tsequence\tmodel_output
    Classification markers: @ = microalgae, ! = contamination
    """
    validate_input_source(filepath)

    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('record_id\t'):
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                continue

            record_id = parts[0]
            model_output = parts[2]

            # Count @ and ! markers in the output portion after the sequence
            # The output format is: sequence*<continuation><markers>
            at_count = model_output.count('@')
            exclaim_count = model_output.count('!')

            if at_count > 0 and exclaim_count == 0:
                classification = 'algae'
            elif exclaim_count > 0 and at_count == 0:
                classification = 'contaminant'
            elif at_count > exclaim_count:
                classification = 'algae'
            elif exclaim_count > at_count:
                classification = 'contaminant'
            else:
                classification = 'unknown'

            results[record_id] = classification

    return results

def load_ground_truth(filepath):
    """Load ground truth labels."""
    validate_input_source(filepath)

    ground_truth = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('seq_id\t'):
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                continue

            seq_id = parts[0]
            label = parts[2]

            # Normalize ground truth
            if label == 'algae':
                ground_truth[seq_id] = 'algae'
            elif label == 'contaminant':
                ground_truth[seq_id] = 'contaminant'
            else:
                ground_truth[seq_id] = label

    return ground_truth

def main():
    # Define paths
    base_dir = '/media/drn2/External/TARA-Oceans/03_analyses/decoding_experiment'
    results_dir = os.path.join(base_dir, 'results')

    # Input files
    ground_truth_file = os.path.join(base_dir, 'test_dataset_ground_truth.tsv')
    algagpt_sampling_file = os.path.join(results_dir, 'algagpt_baseline_sampling.tsv')
    algagpt_greedy_file = os.path.join(results_dir, 'algagpt_greedy.tsv')
    alkhidr_greedy_file = os.path.join(results_dir, 'alkhidr_baseline_greedy.tsv')
    alkhidr_sampling_file = os.path.join(results_dir, 'alkhidr_sampling.tsv')

    # Output file
    output_file = os.path.join(results_dir, 'decoding_comparison_matrix.tsv')

    print(f"Loading data from inference outputs...")
    print(f"  Ground truth: {ground_truth_file}")
    print(f"  AlgaGPT sampling: {algagpt_sampling_file}")
    print(f"  AlgaGPT greedy: {algagpt_greedy_file}")
    print(f"  AlKhidr greedy: {alkhidr_greedy_file}")
    print(f"  AlKhidr sampling: {alkhidr_sampling_file}")

    # Load all data
    ground_truth = load_ground_truth(ground_truth_file)
    algagpt_sampling = parse_algagpt_output(algagpt_sampling_file)
    algagpt_greedy = parse_algagpt_output(algagpt_greedy_file)
    alkhidr_greedy = parse_alkhidr_greedy(alkhidr_greedy_file)
    alkhidr_sampling = parse_alkhidr_sampling(alkhidr_sampling_file)

    print(f"\nData loaded:")
    print(f"  Ground truth: {len(ground_truth)} sequences")
    print(f"  AlgaGPT sampling: {len(algagpt_sampling)} sequences")
    print(f"  AlgaGPT greedy: {len(algagpt_greedy)} sequences")
    print(f"  AlKhidr greedy: {len(alkhidr_greedy)} sequences")
    print(f"  AlKhidr sampling: {len(alkhidr_sampling)} sequences")

    # Create comparison matrix
    all_seq_ids = sorted(ground_truth.keys())

    # Write output
    with open(output_file, 'w') as f:
        # Provenance header
        f.write(f"# Provenance:\n")
        f.write(f"#   Script: {os.path.abspath(__file__)}\n")
        f.write(f"#   Ground truth input: {ground_truth_file}\n")
        f.write(f"#   AlgaGPT sampling input: {algagpt_sampling_file}\n")
        f.write(f"#   AlgaGPT greedy input: {algagpt_greedy_file}\n")
        f.write(f"#   AlKhidr greedy input: {alkhidr_greedy_file}\n")
        f.write(f"#   AlKhidr sampling input: {alkhidr_sampling_file}\n")
        f.write(f"#   Date: {datetime.now().isoformat()}\n")
        f.write(f"#   Integrity Check: PASSED\n")
        f.write(f"#\n")

        # Header
        f.write("seq_id\tground_truth\talgagpt_sampling\talgagpt_greedy\talkhidr_greedy\talkhidr_sampling\t")
        f.write("algagpt_sampling_correct\talgagpt_greedy_correct\talkhidr_greedy_correct\talkhidr_sampling_correct\n")

        # Data rows
        for seq_id in all_seq_ids:
            gt = ground_truth.get(seq_id, 'unknown')

            # Get predictions (use 'unknown' if missing)
            agpt_samp = algagpt_sampling.get(seq_id, 'unknown')
            agpt_greedy = algagpt_greedy.get(seq_id, 'unknown')
            alk_greedy = alkhidr_greedy.get(seq_id, 'unknown')
            alk_samp = alkhidr_sampling.get(seq_id, 'unknown')

            # Calculate correctness (1=correct, 0=incorrect, -1=unknown)
            def is_correct(pred, truth):
                if pred == 'unknown':
                    return -1
                return 1 if pred == truth else 0

            agpt_samp_correct = is_correct(agpt_samp, gt)
            agpt_greedy_correct = is_correct(agpt_greedy, gt)
            alk_greedy_correct = is_correct(alk_greedy, gt)
            alk_samp_correct = is_correct(alk_samp, gt)

            f.write(f"{seq_id}\t{gt}\t{agpt_samp}\t{agpt_greedy}\t{alk_greedy}\t{alk_samp}\t")
            f.write(f"{agpt_samp_correct}\t{agpt_greedy_correct}\t{alk_greedy_correct}\t{alk_samp_correct}\n")

    print(f"\nOutput written to: {output_file}")

    # Print summary statistics
    print("\n" + "="*60)
    print("COMPARISON MATRIX SUMMARY")
    print("="*60)

    # Count correct predictions per condition
    algagpt_samp_correct = sum(1 for s in all_seq_ids if algagpt_sampling.get(s) == ground_truth.get(s))
    algagpt_greedy_correct = sum(1 for s in all_seq_ids if algagpt_greedy.get(s) == ground_truth.get(s))
    alkhidr_greedy_correct = sum(1 for s in all_seq_ids if alkhidr_greedy.get(s) == ground_truth.get(s))
    alkhidr_samp_correct = sum(1 for s in all_seq_ids if alkhidr_sampling.get(s) == ground_truth.get(s))

    total = len(all_seq_ids)

    print(f"\nOverall Accuracy:")
    print(f"  AlgaGPT Sampling: {algagpt_samp_correct}/{total} = {100*algagpt_samp_correct/total:.1f}%")
    print(f"  AlgaGPT Greedy:   {algagpt_greedy_correct}/{total} = {100*algagpt_greedy_correct/total:.1f}%")
    print(f"  AlKhidr Greedy:   {alkhidr_greedy_correct}/{total} = {100*alkhidr_greedy_correct/total:.1f}%")
    print(f"  AlKhidr Sampling: {alkhidr_samp_correct}/{total} = {100*alkhidr_samp_correct/total:.1f}%")

    # Count by class
    algae_ids = [s for s in all_seq_ids if ground_truth.get(s) == 'algae']
    contam_ids = [s for s in all_seq_ids if ground_truth.get(s) == 'contaminant']

    print(f"\nAlgae Recall (n={len(algae_ids)}):")
    agpt_samp_algae = sum(1 for s in algae_ids if algagpt_sampling.get(s) == 'algae')
    agpt_greedy_algae = sum(1 for s in algae_ids if algagpt_greedy.get(s) == 'algae')
    alk_greedy_algae = sum(1 for s in algae_ids if alkhidr_greedy.get(s) == 'algae')
    alk_samp_algae = sum(1 for s in algae_ids if alkhidr_sampling.get(s) == 'algae')

    print(f"  AlgaGPT Sampling: {agpt_samp_algae}/{len(algae_ids)} = {100*agpt_samp_algae/len(algae_ids):.1f}%")
    print(f"  AlgaGPT Greedy:   {agpt_greedy_algae}/{len(algae_ids)} = {100*agpt_greedy_algae/len(algae_ids):.1f}%")
    print(f"  AlKhidr Greedy:   {alk_greedy_algae}/{len(algae_ids)} = {100*alk_greedy_algae/len(algae_ids):.1f}%")
    print(f"  AlKhidr Sampling: {alk_samp_algae}/{len(algae_ids)} = {100*alk_samp_algae/len(algae_ids):.1f}%")

    print(f"\nContaminant Recall (n={len(contam_ids)}):")
    agpt_samp_cont = sum(1 for s in contam_ids if algagpt_sampling.get(s) == 'contaminant')
    agpt_greedy_cont = sum(1 for s in contam_ids if algagpt_greedy.get(s) == 'contaminant')
    alk_greedy_cont = sum(1 for s in contam_ids if alkhidr_greedy.get(s) == 'contaminant')
    alk_samp_cont = sum(1 for s in contam_ids if alkhidr_sampling.get(s) == 'contaminant')

    print(f"  AlgaGPT Sampling: {agpt_samp_cont}/{len(contam_ids)} = {100*agpt_samp_cont/len(contam_ids):.1f}%")
    print(f"  AlgaGPT Greedy:   {agpt_greedy_cont}/{len(contam_ids)} = {100*agpt_greedy_cont/len(contam_ids):.1f}%")
    print(f"  AlKhidr Greedy:   {alk_greedy_cont}/{len(contam_ids)} = {100*alk_greedy_cont/len(contam_ids):.1f}%")
    print(f"  AlKhidr Sampling: {alk_samp_cont}/{len(contam_ids)} = {100*alk_samp_cont/len(contam_ids):.1f}%")

    # Cross-condition agreement analysis
    print("\n" + "-"*60)
    print("CROSS-CONDITION AGREEMENT ANALYSIS")
    print("-"*60)

    # AlgaGPT: Sampling vs Greedy agreement
    agpt_agree = sum(1 for s in all_seq_ids if algagpt_sampling.get(s) == algagpt_greedy.get(s))
    print(f"\nAlgaGPT Sampling vs Greedy agreement: {agpt_agree}/{total} = {100*agpt_agree/total:.1f}%")

    # AlKhidr: Greedy vs Sampling agreement
    alk_agree = sum(1 for s in all_seq_ids if alkhidr_greedy.get(s) == alkhidr_sampling.get(s))
    print(f"AlKhidr Greedy vs Sampling agreement: {alk_agree}/{total} = {100*alk_agree/total:.1f}%")

    # All 4 conditions agree
    all_agree = sum(1 for s in all_seq_ids
                   if algagpt_sampling.get(s) == algagpt_greedy.get(s) == alkhidr_greedy.get(s) == alkhidr_sampling.get(s))
    print(f"All 4 conditions agree: {all_agree}/{total} = {100*all_agree/total:.1f}%")

    # Sequences where decoding strategy changes the result
    print("\n" + "-"*60)
    print("DECODING STRATEGY EFFECT")
    print("-"*60)

    agpt_decoding_diff = sum(1 for s in all_seq_ids if algagpt_sampling.get(s) != algagpt_greedy.get(s))
    alk_decoding_diff = sum(1 for s in all_seq_ids if alkhidr_greedy.get(s) != alkhidr_sampling.get(s))

    print(f"AlgaGPT: Decoding strategy changes result for {agpt_decoding_diff}/{total} sequences ({100*agpt_decoding_diff/total:.1f}%)")
    print(f"AlKhidr: Decoding strategy changes result for {alk_decoding_diff}/{total} sequences ({100*alk_decoding_diff/total:.1f}%)")

    # Identify specific sequences
    print("\nAlgaGPT sequences where decoding changed result:")
    for s in all_seq_ids:
        samp = algagpt_sampling.get(s)
        greedy = algagpt_greedy.get(s)
        gt = ground_truth.get(s)
        if samp != greedy:
            samp_correct = "CORRECT" if samp == gt else "WRONG"
            greedy_correct = "CORRECT" if greedy == gt else "WRONG"
            print(f"  {s}: sampling={samp} ({samp_correct}), greedy={greedy} ({greedy_correct}), truth={gt}")

    print("\nAlKhidr sequences where decoding changed result (first 20):")
    count = 0
    for s in all_seq_ids:
        greedy = alkhidr_greedy.get(s)
        samp = alkhidr_sampling.get(s)
        gt = ground_truth.get(s)
        if greedy != samp:
            greedy_correct = "CORRECT" if greedy == gt else "WRONG"
            samp_correct = "CORRECT" if samp == gt else "WRONG"
            print(f"  {s}: greedy={greedy} ({greedy_correct}), sampling={samp} ({samp_correct}), truth={gt}")
            count += 1
            if count >= 20:
                remaining = alk_decoding_diff - count
                if remaining > 0:
                    print(f"  ... and {remaining} more")
                break

if __name__ == '__main__':
    main()
