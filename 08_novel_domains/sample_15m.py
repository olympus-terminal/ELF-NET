#!/usr/bin/env python3
"""
Sample ~15 million dark proteome sequences and ~15 million annotated
sequences for LM distillation, each split 90/10 into train/holdout.

Source of truth for annotated/dark classification:
  hmmsearch_results/*.aa.hmmsearch.tbl
  Protein has a hit → annotated.  No hit → dark.

Does NOT use dark_ids/ or pfam_results/ (both are deprecated/inconsistent).

Annotated sequences are delivered WITH their Pfam hmmsearch results
so the collaborator has both FASTA and domain annotations.

Output (in kourosh_subsets/):
  dark_15m_train.fa                  (~13.5M seqs)
  dark_15m_holdout.fa                (~1.5M seqs)
  annotated_15m_train.fa             (~13.5M seqs)
  annotated_15m_holdout.fa           (~1.5M seqs)
  annotated_15m_train.hmmsearch.tbl  (Pfam hits for train seqs)
  annotated_15m_holdout.hmmsearch.tbl(Pfam hits for holdout seqs)

Usage:
  python sample_15m.py [--target-seqs 15000000] [--seed 42] [--count-only]
"""

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment detection (works on both HPC and local)
# ---------------------------------------------------------------------------
def get_base_dir(project_name: str) -> Path:
    hpc_path = Path(f"/scratch/drn2/PROJECTS/{project_name}")
    if hpc_path.exists():
        return hpc_path
    return Path(f"/media/drn2/External/{project_name}")

BASE = get_base_dir("TARA-LA4SR") / "03_analyses"
ALGAE_DIR = BASE / "algae_proteins"
HMMSEARCH_DIR = BASE / "hmmsearch_results"  # source of truth for annotation
SAMPLE_LIST = BASE / "novel_domains" / "sample_list.txt"
OUT_DIR = BASE / "novel_domains" / "kourosh_subsets"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_tbl(sample: str) -> Path | None:
    """Find the hmmsearch tblout file for a sample in hmmsearch_results/."""
    primary = HMMSEARCH_DIR / f"{sample}.aa.hmmsearch.tbl"
    if primary.exists():
        return primary
    for pat in [f"{sample}*.hmmsearch.tbl", f"{sample}*.tbl"]:
        hits = sorted(HMMSEARCH_DIR.glob(pat))
        if hits:
            return hits[0]
    return None

def load_annotated_ids(tbl_path: Path) -> set[str]:
    """Get the set of protein IDs that have any hmmsearch hit."""
    ids = set()
    with open(tbl_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) >= 1:
                ids.add(fields[0])
    return ids

def load_hmmsearch_by_protein(tbl_path: Path) -> dict[str, list[str]]:
    """Parse hmmsearch tblout into {protein_id: [raw_lines]}."""
    hits = defaultdict(list)
    with open(tbl_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                continue
            hits[fields[0]].append(line)
    return hits

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Sample 15M dark + 15M annotated sequences with 90/10 holdout split"
)
parser.add_argument("--target-seqs", type=int, default=15_000_000,
                    help="Target number of sequences per class (default: 15M)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--count-only", action="store_true",
                    help="Only count available sequences; do not write FASTAs")
parser.add_argument("--holdout-frac", type=float, default=0.10,
                    help="Fraction held out for validation (default: 0.10)")
args = parser.parse_args()

TARGET = args.target_seqs
SEED = args.seed
HOLDOUT_FRAC = args.holdout_frac

OUT_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()

# ===================================================================
# Phase 0: Count annotated and dark proteins using hmmsearch_results
# ===================================================================
print("=" * 70)
print("Phase 0: Counting proteins using hmmsearch_results as source of truth")
print("=" * 70)
print(f"  hmmsearch dir: {HMMSEARCH_DIR}")
print(f"  algae dir    : {ALGAE_DIR}")

samples = [line.strip() for line in open(SAMPLE_LIST) if line.strip()]
print(f"  Assemblies in sample list: {len(samples)}")

total_annotated = 0
total_dark = 0
total_proteins = 0
samples_valid = 0
samples_skipped = 0
samples_no_tbl = 0

for sample in samples:
    fasta_file = ALGAE_DIR / f"{sample}.algae.fa"
    if not fasta_file.exists():
        samples_skipped += 1
        continue

    tbl = find_tbl(sample)
    if tbl is None:
        samples_no_tbl += 1
        continue

    # Count total proteins in FASTA
    n_total = sum(1 for line in open(fasta_file) if line.startswith(">"))

    # Count annotated proteins (those with hmmsearch hits)
    annotated_ids = load_annotated_ids(tbl)
    n_annot = len(annotated_ids)
    n_dark = n_total - n_annot

    total_annotated += n_annot
    total_dark += n_dark
    total_proteins += n_total
    samples_valid += 1

    if samples_valid % 200 == 0:
        elapsed = time.time() - t0
        print(f"  Counted {samples_valid}/{len(samples)} assemblies: "
              f"{total_annotated:,} annotated, {total_dark:,} dark  "
              f"({total_proteins:,} total, {elapsed:.0f}s)")

elapsed = time.time() - t0
print(f"\nCount complete in {elapsed:.0f}s")
print(f"  Valid assemblies      : {samples_valid}")
print(f"  Skipped (no FASTA)    : {samples_skipped}")
print(f"  Missing hmmsearch tbl : {samples_no_tbl}")
print(f"  Total proteins        : {total_proteins:,}")
print(f"  Total annotated       : {total_annotated:,}")
print(f"  Total dark            : {total_dark:,}")
if total_proteins > 0:
    print(f"  Annotated frac        : {total_annotated / total_proteins * 100:.2f}%")

# Check feasibility
dark_target = min(TARGET, total_dark)
annot_target = min(TARGET, total_annotated)

if total_annotated < TARGET:
    print(f"\n  WARNING: Only {total_annotated:,} annotated proteins available "
          f"(requested {TARGET:,}). Will take ALL annotated proteins.")
if total_dark < TARGET:
    print(f"\n  WARNING: Only {total_dark:,} dark proteins available "
          f"(requested {TARGET:,}). Will take ALL dark proteins.")

print(f"\n  Dark target     : {dark_target:,}")
print(f"  Annotated target: {annot_target:,}")

if args.count_only:
    print("\n--count-only mode; exiting without writing FASTAs.")
    sys.exit(0)

# ===================================================================
# Compute sampling rates
# ===================================================================
sample_rate_dark = min((dark_target / total_dark) * 1.10, 1.0)
sample_rate_annot = min((annot_target / total_annotated) * 1.10, 1.0)

holdout_mod = max(1, round(1.0 / HOLDOUT_FRAC))  # every Nth → holdout

# ===================================================================
# Main pass: one sweep through all assemblies, routing proteins
# to dark or annotated outputs based on hmmsearch hits
# ===================================================================
print()
print("=" * 70)
print("Sampling dark and annotated in a single pass per assembly")
print("=" * 70)
print(f"  Dark sampling rate     : {sample_rate_dark:.6f}")
print(f"  Annotated sampling rate: {sample_rate_annot:.6f}")
if sample_rate_annot >= 1.0:
    print("  (Taking ALL annotated proteins)")

dark_train_path = OUT_DIR / "dark_15m_train.fa"
dark_holdout_path = OUT_DIR / "dark_15m_holdout.fa"
annot_train_path = OUT_DIR / "annotated_15m_train.fa"
annot_holdout_path = OUT_DIR / "annotated_15m_holdout.fa"
tbl_train_path = OUT_DIR / "annotated_15m_train.hmmsearch.tbl"
tbl_holdout_path = OUT_DIR / "annotated_15m_holdout.hmmsearch.tbl"

random.seed(SEED)
t1 = time.time()

# Counters
dark_train_n = 0
dark_holdout_n = 0
dark_selected = 0
annot_train_n = 0
annot_holdout_n = 0
annot_selected = 0
tbl_lines_train = 0
tbl_lines_holdout = 0
assemblies_processed = 0

TBL_HEADER = (
    "# Pfam-A hmmsearch results for sampled annotated proteins\n"
    "# Source: hmmsearch_results/*.aa.hmmsearch.tbl (Jan 14 2026)\n"
    "# Extracted by sample_15m.py\n"
    "# Format: HMMER3 tblout (--tblout)\n"
)

with open(dark_train_path, "w") as dk_train, \
     open(dark_holdout_path, "w") as dk_hold, \
     open(annot_train_path, "w") as an_train, \
     open(annot_holdout_path, "w") as an_hold, \
     open(tbl_train_path, "w") as tbl_train, \
     open(tbl_holdout_path, "w") as tbl_hold:

    tbl_train.write(TBL_HEADER)
    tbl_hold.write(TBL_HEADER)

    dark_done = False
    annot_done = False

    for sample in samples:
        if dark_done and annot_done:
            break

        fasta_file = ALGAE_DIR / f"{sample}.algae.fa"
        tbl_file = find_tbl(sample)
        if not fasta_file.exists() or tbl_file is None:
            continue

        # Load annotated IDs and hmmsearch hits for this assembly
        hmm_hits = load_hmmsearch_by_protein(tbl_file)
        annotated_ids = set(hmm_hits.keys())

        # Stream FASTA and route each protein
        include = False
        current_record = []
        current_seq_id = None
        is_annot = False

        with open(fasta_file) as fin:
            for line in fin:
                if line.startswith(">"):
                    # Flush previous record
                    if include and current_record and current_seq_id:
                        record_text = "".join(current_record)
                        if is_annot and not annot_done:
                            annot_selected += 1
                            if annot_selected % holdout_mod == 0:
                                an_hold.write(record_text)
                                annot_holdout_n += 1
                                for hl in hmm_hits.get(current_seq_id, []):
                                    tbl_hold.write(hl)
                                    tbl_lines_holdout += 1
                            else:
                                an_train.write(record_text)
                                annot_train_n += 1
                                for hl in hmm_hits.get(current_seq_id, []):
                                    tbl_train.write(hl)
                                    tbl_lines_train += 1
                            if annot_train_n + annot_holdout_n >= annot_target:
                                annot_done = True
                        elif not is_annot and not dark_done:
                            dark_selected += 1
                            if dark_selected % holdout_mod == 0:
                                dk_hold.write(record_text)
                                dark_holdout_n += 1
                            else:
                                dk_train.write(record_text)
                                dark_train_n += 1
                            if dark_train_n + dark_holdout_n >= dark_target:
                                dark_done = True

                    seq_id = line[1:].strip().split()[0]
                    is_annot = seq_id in annotated_ids

                    # Decide whether to include this protein
                    if is_annot and not annot_done:
                        include = random.random() < sample_rate_annot
                    elif not is_annot and not dark_done:
                        include = random.random() < sample_rate_dark
                    else:
                        include = False

                    current_seq_id = seq_id if include else None
                    current_record = [line] if include else []
                elif include:
                    current_record.append(line)

            # Last record in file
            if include and current_record and current_seq_id:
                record_text = "".join(current_record)
                if is_annot and not annot_done:
                    annot_selected += 1
                    if annot_selected % holdout_mod == 0:
                        an_hold.write(record_text)
                        annot_holdout_n += 1
                        for hl in hmm_hits.get(current_seq_id, []):
                            tbl_hold.write(hl)
                            tbl_lines_holdout += 1
                    else:
                        an_train.write(record_text)
                        annot_train_n += 1
                        for hl in hmm_hits.get(current_seq_id, []):
                            tbl_train.write(hl)
                            tbl_lines_train += 1
                    if annot_train_n + annot_holdout_n >= annot_target:
                        annot_done = True
                elif not is_annot and not dark_done:
                    dark_selected += 1
                    if dark_selected % holdout_mod == 0:
                        dk_hold.write(record_text)
                        dark_holdout_n += 1
                    else:
                        dk_train.write(record_text)
                        dark_train_n += 1
                    if dark_train_n + dark_holdout_n >= dark_target:
                        dark_done = True

        assemblies_processed += 1
        if assemblies_processed % 200 == 0:
            elapsed = time.time() - t1
            print(f"  {assemblies_processed}/{len(samples)} assemblies: "
                  f"dark {dark_train_n + dark_holdout_n:,} "
                  f"({dark_train_n:,}+{dark_holdout_n:,}), "
                  f"annot {annot_train_n + annot_holdout_n:,} "
                  f"({annot_train_n:,}+{annot_holdout_n:,}), "
                  f"{elapsed:.0f}s")

elapsed = time.time() - t1
print(f"\nSampling complete ({elapsed:.0f}s, {assemblies_processed} assemblies)")

# ===================================================================
# Summary
# ===================================================================
print()
print("=" * 70)
print("Summary")
print("=" * 70)
print(f"  Dark train     : {dark_train_n:>12,} seqs")
print(f"  Dark holdout   : {dark_holdout_n:>12,} seqs")
print(f"  Dark total     : {dark_train_n + dark_holdout_n:>12,} / {dark_target:,} target")
print(f"  Annot train    : {annot_train_n:>12,} seqs")
print(f"  Annot holdout  : {annot_holdout_n:>12,} seqs")
print(f"  Annot total    : {annot_train_n + annot_holdout_n:>12,} / {annot_target:,} target")
print(f"  HMM hits train : {tbl_lines_train:>12,}")
print(f"  HMM hits hold  : {tbl_lines_holdout:>12,}")
print()

for label, path in [
    ("Dark train", dark_train_path),
    ("Dark holdout", dark_holdout_path),
    ("Annotated train", annot_train_path),
    ("Annotated holdout", annot_holdout_path),
    ("HMM hits train", tbl_train_path),
    ("HMM hits holdout", tbl_holdout_path),
]:
    if path.exists():
        size = path.stat().st_size
        if size > 1024**3:
            size_str = f"{size / 1024**3:.2f} GB"
        elif size > 1024**2:
            size_str = f"{size / 1024**2:.1f} MB"
        else:
            size_str = f"{size / 1024:.1f} KB"
        print(f"  {label:20s}: {size_str:>10s}  ({path.name})")

print(f"\nOutput directory: {OUT_DIR}")
print(f"Total wall time : {time.time() - t0:.0f}s")
print("Done!")
