#!/usr/bin/env python3
"""
Extract Form II RuBisCO sequences and cluster at 97% identity to
estimate species-level diversity.

Strategy:
  1. Read combined TSV to find samples with Form II hits
  2. For each sample, re-run hmmsearch with formII_rbcL.hmm to get protein IDs
  3. Extract matching sequences from the proteome FASTAs
  4. Cluster with cd-hit at 97% identity, 80% coverage
  5. Report cluster count as species-level estimate

Usage:
    python3 cluster_formII_rubisco.py

Author: drn
Date: 2026-03-21
"""

import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict

# === PATHS ===
BASE = Path("/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses")
RUBISCO_DIR = BASE / "rubisco_hmms"
HMM_DIR = RUBISCO_DIR / "hmms"
EXTENDED_TSV = RUBISCO_DIR / "rubisco_extended_combined_20260218_081615.tsv"
PROTEIN_FILELIST = RUBISCO_DIR / "protein_filelist_extended.txt"
OUTDIR = RUBISCO_DIR / "formII_clustering"

# Form II broad HMM
FORMII_HMM = HMM_DIR / "formII_rbcL.hmm"

BROAD_EVALUE = 1e-7
COMBINED_FASTA = OUTDIR / "formII_rubisco_all.fasta"
SUMMARY_FILE = OUTDIR / "formII_species_estimate.txt"

# Find binaries - use full paths since modules may not propagate to subprocesses
HMMSEARCH = shutil.which("hmmsearch")
CDHIT = shutil.which("cd-hit")

if not HMMSEARCH:
    # Try known spack path
    candidate = "/share/apps/NYUAD5/spack/0.16.2/opt/spack/linux-centos8-zen2/gcc-9.2.0/hmmer-3.3-25fpdage2fhfo3tohfl4wfpjm4yg6neb/bin/hmmsearch"
    if os.path.exists(candidate):
        HMMSEARCH = candidate

if not CDHIT:
    # Search common paths
    for p in ["/share/apps/NYUAD5/spack/0.16.2/opt/spack/linux-centos8-zen2/gcc-9.2.0/cd-hit-4.8.1-*/bin/cd-hit"]:
        import glob as g
        matches = g.glob(p)
        if matches:
            CDHIT = matches[0]
            break

def load_protein_filelist():
    """Load sample_id -> protein file path mapping."""
    mapping = {}
    with open(PROTEIN_FILELIST) as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            # The sample_id in the TSV is the filename stem minus the final .fa
            # e.g., path = .../proteins/MGYA00123456.aa.fa
            #        stem after removing .fa = MGYA00123456.aa
            basename = os.path.basename(path)
            # Remove trailing .fa or .fasta
            if basename.endswith(".fa"):
                key = basename[:-3]
            elif basename.endswith(".fasta"):
                key = basename[:-6]
            else:
                key = basename
            mapping[key] = path
    return mapping

def run_hmmsearch(protein_file, hmm_file, evalue):
    """Run hmmsearch, return set of hit protein IDs."""
    hits = set()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tbl', delete=False) as tbl:
        tbl_file = tbl.name
    try:
        subprocess.run(
            [HMMSEARCH, '--tblout', tbl_file, '-E', str(evalue),
             '--cpu', '1', '--noali', hmm_file, protein_file],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1800
        )
        with open(tbl_file) as f:
            for line in f:
                if not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 5:
                        hits.add(parts[0])
    except Exception as e:
        print(f"  WARNING: hmmsearch error: {e}", file=sys.stderr)
    finally:
        if os.path.exists(tbl_file):
            os.unlink(tbl_file)
    return hits

def extract_sequences(fasta_path, target_ids):
    """Extract sequences with IDs in target_ids from a FASTA file."""
    sequences = {}
    current_id = None
    current_seq = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                if current_id and current_id in target_ids:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_id and current_id in target_ids:
            sequences[current_id] = "".join(current_seq)
    return sequences

def main():
    # Validate tools
    print(f"hmmsearch: {HMMSEARCH}")
    print(f"cd-hit:    {CDHIT}")
    if not HMMSEARCH or not os.path.exists(HMMSEARCH):
        print("ERROR: hmmsearch not found!")
        sys.exit(1)
    if not CDHIT or not os.path.exists(CDHIT):
        print("ERROR: cd-hit not found!")
        sys.exit(1)
    if not FORMII_HMM.exists():
        print(f"ERROR: HMM not found: {FORMII_HMM}")
        sys.exit(1)

    os.makedirs(OUTDIR, exist_ok=True)

    # Load protein file mapping
    print(f"\nLoading protein filelist...")
    filelist_map = load_protein_filelist()
    print(f"  {len(filelist_map)} entries")

    # Debug: show first few mappings
    items = list(filelist_map.items())[:3]
    for k, v in items:
        print(f"  key='{k}' -> {v}")

    # Read combined TSV to find samples with Form II hits
    print(f"\nReading {EXTENDED_TSV}...")
    formII_samples = {}
    with open(EXTENDED_TSV) as f:
        for line in f:
            if line.startswith("#") or line.startswith("sample_id"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            sample_id = parts[0]
            formII_count = int(parts[5])
            if formII_count > 0:
                formII_samples[sample_id] = formII_count

    print(f"  {len(formII_samples)} samples with Form II hits")
    print(f"  Expected total: {sum(formII_samples.values()):,} sequences")

    # Debug: show first few sample IDs and check mapping
    sample_list = sorted(formII_samples.keys())[:5]
    print(f"\n  First few sample IDs from TSV:")
    for s in sample_list:
        found = s in filelist_map
        print(f"    '{s}' -> mapped={found}")

    # Extract Form II sequences
    print(f"\nExtracting Form II sequences...")
    total_extracted = 0
    missing = 0
    hmm_str = str(FORMII_HMM)

    with open(COMBINED_FASTA, "w") as out_f:
        for i, (sample, expected) in enumerate(sorted(formII_samples.items())):
            if (i + 1) % 200 == 0 or i == 0:
                print(f"  [{i+1}/{len(formII_samples)}] extracted={total_extracted:,} ...")

            # Find protein file
            protein_file = filelist_map.get(sample)
            if protein_file is None or not os.path.exists(protein_file or ""):
                missing += 1
                if missing <= 5:
                    print(f"  MISS: sample='{sample}' not in filelist", file=sys.stderr)
                continue

            # Run hmmsearch
            hit_ids = run_hmmsearch(protein_file, hmm_str, BROAD_EVALUE)
            if not hit_ids:
                continue

            # Extract sequences
            seqs = extract_sequences(protein_file, hit_ids)
            for pid, seq in seqs.items():
                out_f.write(f">{sample}|{pid}\n{seq}\n")
                total_extracted += 1

    print(f"\n  Total extracted: {total_extracted:,}")
    print(f"  Missing FASTAs: {missing}")

    if total_extracted == 0:
        print("ERROR: No sequences extracted!")
        sys.exit(1)

    # Cluster with cd-hit at 97% identity
    cdhit_out = str(OUTDIR / "formII_cdhit97")
    print(f"\nClustering {total_extracted:,} sequences at 97% identity with cd-hit...")
    cmd = [
        CDHIT,
        '-i', str(COMBINED_FASTA),
        '-o', cdhit_out,
        '-c', '0.97',       # 97% identity
        '-aL', '0.8',       # 80% alignment coverage of longer seq
        '-aS', '0.8',       # 80% alignment coverage of shorter seq
        '-M', '80000',      # 80 GB memory limit
        '-T', '16',         # threads
        '-d', '0',          # no description length limit
        '-g', '1',          # accurate mode
    ]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  cd-hit FAILED (rc={result.returncode})")
        print(f"  stderr: {result.stderr[:1000]}")
        sys.exit(1)

    # Parse cd-hit .clstr output
    n_clusters = 0
    sizes = []
    current_size = 0
    with open(cdhit_out + ".clstr") as f:
        for line in f:
            if line.startswith(">Cluster"):
                if current_size > 0:
                    sizes.append(current_size)
                current_size = 0
                n_clusters += 1
            else:
                current_size += 1
        if current_size > 0:
            sizes.append(current_size)

    # Actually n_clusters = len(sizes) since we count on >Cluster lines
    n_clusters = len(sizes)
    singletons = sum(1 for s in sizes if s == 1)
    sizes.sort()
    median_size = sizes[len(sizes)//2] if sizes else 0
    max_size = max(sizes) if sizes else 0
    mean_size = sum(sizes) / len(sizes) if sizes else 0

    # Write summary
    with open(SUMMARY_FILE, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("FORM II RuBisCO SPECIES-LEVEL ESTIMATE\n")
        f.write("(97% protein sequence identity clustering)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: 2026-03-21\n")
        f.write(f"Script: cluster_formII_rubisco.py\n")
        f.write(f"HMM: {FORMII_HMM}\n")
        f.write(f"E-value: {BROAD_EVALUE}\n")
        f.write(f"Clustering: cd-hit -c 0.97 -aL 0.8 -aS 0.8\n\n")
        f.write(f"Input sequences: {total_extracted:,}\n")
        f.write(f"Samples with Form II: {len(formII_samples)}\n")
        f.write(f"Samples with missing FASTAs: {missing}\n\n")
        f.write(f"CLUSTERS (97% OTUs): {n_clusters:,}\n")
        f.write(f"Singletons: {singletons:,} ({100*singletons/n_clusters:.1f}%)\n")
        f.write(f"Median cluster size: {median_size}\n")
        f.write(f"Mean cluster size: {mean_size:.1f}\n")
        f.write(f"Max cluster size: {max_size}\n\n")

        f.write("Cluster size distribution:\n")
        bins = [(1,1),(2,5),(6,10),(11,50),(51,100),(101,500),(501,float('inf'))]
        for lo, hi in bins:
            count = sum(1 for s in sizes if lo <= s <= hi)
            label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi != float('inf') else f">{lo-1}")
            f.write(f"  {label:>8s}: {count:>6,} clusters ({100*count/n_clusters:.1f}%)\n")

        f.write(f"\n{'='*70}\n")
        f.write(f"FORM II SPECIES ESTIMATE: ~{round(n_clusters, -2):,}\n")
        f.write(f"\nForm I taxa (original): ~21,500\n")
        combined = n_clusters + 21500
        f.write(f"COMBINED ESTIMATE: ~{round(combined, -3):,}\n")
        f.write(f"{'='*70}\n")

    print(f"\n{'='*60}")
    print(f"RESULT: {n_clusters:,} clusters at 97% identity")
    print(f"Form I (~21,500) + Form II ({n_clusters:,}) = ~{round(n_clusters + 21500, -3):,}")
    print(f"{'='*60}")
    print(f"Summary: {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
