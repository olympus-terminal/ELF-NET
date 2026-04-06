#!/usr/bin/env python3
"""
AlgaGPT vs DIAMOND BLAST Comparative Analysis
===============================================
Compares LLM-based protein classification (algaGPT: algae/conta) against
DIAMOND BLASTp hits (NR database, e-value < 1e-9) on the original unfiltered
protein set (02_processed_data/proteins/).

For each of the 2,044 assemblies:
  - algaGPT classified every protein as "algae" or "conta" (contaminant)
  - DIAMOND BLASTp searched every protein against NCBI NR

Cross-tabulation reveals how LLM classification relates to sequence homology:
  - algae + hit:   LLM calls algal AND has NR homolog
  - algae + nohit: LLM calls algal BUT no NR homolog (novel/dark algal protein)
  - conta + hit:   LLM calls contaminant AND has NR homolog
  - conta + nohit: LLM calls contaminant AND no NR homolog

Provenance:
  Script: scripts/algagpt_vs_diamond_comparison.py
  Date: 2026-03-15
  AlgaGPT results: 03_analyses/algagpt_results_fixed/
  DIAMOND results: 03_analyses/dark_proteome_v2/diamond_raw/ + diamond_remaining/results/
  Input proteins: 02_processed_data/proteins/*.aa.fa
  DIAMOND DB: NCBI NR (nr-diamond.dmnd), e-value < 1e-9, very-sensitive
"""

import os
import sys
import socket
import logging
import glob
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

# =============================================================================
# Environment detection
# =============================================================================

def get_base_dir(project_name: str) -> Path:
    hostname = socket.gethostname()
    if "cn" in hostname or "dn" in hostname or "gpu" in hostname or "jubail" in hostname:
        return Path(f"/scratch/drn2/PROJECTS/{project_name}")
    return Path(f"/media/drn2/External/{project_name}")

BASE_DIR = get_base_dir("TARA-LA4SR")

# =============================================================================
# Configuration
# =============================================================================

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

ALGAGPT_DIR = BASE_DIR / "03_analyses" / "algagpt_results_fixed"
DIAMOND_RAW_DIR = BASE_DIR / "03_analyses" / "dark_proteome_v2" / "diamond_raw"
DIAMOND_REM_DIR = BASE_DIR / "03_analyses" / "dark_proteome_v2" / "diamond_remaining" / "results"
OUTPUT_DIR = BASE_DIR / "03_analyses" / "algagpt_vs_diamond"

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def agpt_name_to_diamond_variants(agpt_basename: str) -> list:
    """
    Generate all possible DIAMOND result basenames for a given algaGPT sample.

    Naming differences between algaGPT and DIAMOND results:
    - Species.AAC.aa (algaGPT) → Species.AAC.fa.aa (diamond_raw)
                                  or Species.AAC.aa (diamond_remaining)
    - MMETSP*.Trinitysta.transdecoder.aa → MMETSP*.Trinity.fasta.transdecoder.aa
    - MGYA*_assembly.aa → same (no change needed)
    - euglenozoa_*.aa → euglenozoa_*.fa.aa or euglenozoa_*.aa

    Returns list of names to try, in priority order.
    """
    name = agpt_basename.replace("_algagpt.tsv", "")
    variants = [name]  # always try exact match first

    # MMETSP: Trinitysta → Trinity.fasta
    if "Trinitysta" in name:
        fasta_name = name.replace("Trinitysta", "Trinity.fasta")
        variants.append(fasta_name)

    # Insert .fa before final .aa (for samples where protein file is *.aa.fa)
    if name.endswith(".aa") and not name.endswith(".fa.aa"):
        fa_name = name[:-3] + ".fa.aa"
        variants.append(fa_name)
        # Also try Trinitysta→Trinity.fasta + .fa insertion
        if "Trinitysta" in name:
            variants.append(fa_name.replace("Trinitysta", "Trinity.fasta"))

    return variants

def _read_hit_ids(filepath: Path) -> set:
    """Read hit IDs from a file."""
    with open(filepath) as f:
        return {line.strip() for line in f if line.strip()}

def find_diamond_hits(agpt_basename: str) -> set:
    """
    Find DIAMOND hit IDs for a sample. Tries multiple name variants
    across diamond_raw/ and diamond_remaining/results/ directories.
    Returns set of protein IDs with NR hits, or None if not found.
    """
    variants = agpt_name_to_diamond_variants(agpt_basename)

    for name in variants:
        # Check diamond_raw
        hit_file = DIAMOND_RAW_DIR / f"{name}.hit_ids.txt"
        if hit_file.exists():
            return _read_hit_ids(hit_file)

        # Check diamond_remaining (whole file)
        hit_file = DIAMOND_REM_DIR / f"{name}.hit_ids.txt"
        if hit_file.exists():
            return _read_hit_ids(hit_file)

        # Check diamond_remaining (chunked)
        chunk_pattern = str(DIAMOND_REM_DIR / f"{name}_chunk*.hit_ids.txt")
        chunk_files = sorted(glob.glob(chunk_pattern))
        if chunk_files:
            hits = set()
            for cf in chunk_files:
                hits.update(_read_hit_ids(Path(cf)))
            return hits

    return None  # Not found

def process_sample(agpt_file: Path) -> dict:
    """
    Process a single sample: read algaGPT labels and DIAMOND hits,
    cross-tabulate, and return stats dict.
    """
    agpt_basename = agpt_file.name
    sample_id = agpt_basename.replace("_algagpt.tsv", "")

    # Read algaGPT classifications
    labels = {}
    with open(agpt_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            pid = parts[0]
            label = "algae" if "algae" in parts[1] else "conta"
            labels[pid] = label

    if not labels:
        return None

    # Find DIAMOND hits (tries multiple name variants)
    diamond_hits = find_diamond_hits(agpt_basename)
    if diamond_hits is None:
        return {"sample": sample_id, "status": "diamond_missing"}

    total = len(labels)
    n_algae = sum(1 for v in labels.values() if v == "algae")
    n_conta = total - n_algae
    n_diamond = len(diamond_hits)

    # Cross-tabulation
    algae_hit = 0
    conta_hit = 0
    for pid, lab in labels.items():
        if pid in diamond_hits:
            if lab == "algae":
                algae_hit += 1
            else:
                conta_hit += 1

    algae_nohit = n_algae - algae_hit
    conta_nohit = n_conta - conta_hit

    # Diamond hits not in algaGPT (unexpected proteins)
    diamond_only = len(diamond_hits - set(labels.keys()))

    return {
        "sample": sample_id,
        "status": "ok",
        "total_proteins": total,
        "algae_count": n_algae,
        "conta_count": n_conta,
        "diamond_hits": n_diamond,
        "diamond_unique_in_agpt": algae_hit + conta_hit,
        "diamond_only": diamond_only,
        "algae_hit": algae_hit,
        "algae_nohit": algae_nohit,
        "conta_hit": conta_hit,
        "conta_nohit": conta_nohit,
        "pct_algae": 100 * n_algae / total if total > 0 else 0,
        "pct_diamond": 100 * n_diamond / total if total > 0 else 0,
        "pct_algae_of_hits": 100 * algae_hit / n_diamond if n_diamond > 0 else np.nan,
        "pct_conta_of_hits": 100 * conta_hit / n_diamond if n_diamond > 0 else np.nan,
        "diamond_rate_algae": 100 * algae_hit / n_algae if n_algae > 0 else 0,
        "diamond_rate_conta": 100 * conta_hit / n_conta if n_conta > 0 else 0,
    }

def classify_source_type(sample_name: str) -> str:
    """Classify sample source: MMETSP, MGYA, GCA/GCF, or reference."""
    if sample_name.startswith("MMETSP"):
        return "MMETSP"
    elif sample_name.startswith("MGYA"):
        return "MGYA"
    elif sample_name.startswith("GCA_") or sample_name.startswith("GCF_"):
        return "RefGenome"
    elif sample_name.startswith("euglenozoa_"):
        return "RefGenome"
    else:
        return "Reference"

def main():
    logger.info("=" * 70)
    logger.info("AlgaGPT vs DIAMOND BLAST Comparative Analysis")
    logger.info(f"Timestamp: {TIMESTAMP}")
    logger.info("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Add file handler
    log_file = OUTPUT_DIR / f"comparison_{TIMESTAMP}.log"
    fh = logging.FileHandler(str(log_file))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # Enumerate algaGPT result files
    agpt_files = sorted(ALGAGPT_DIR.glob("*_algagpt.tsv"))
    logger.info(f"AlgaGPT result files: {len(agpt_files)}")

    # Process all samples
    results = []
    missing = []
    errors = []

    for i, agpt_file in enumerate(agpt_files):
        if (i + 1) % 100 == 0:
            logger.info(f"  Processing {i + 1}/{len(agpt_files)}...")

        try:
            result = process_sample(agpt_file)
            if result is None:
                errors.append(agpt_file.name)
            elif result.get("status") == "diamond_missing":
                missing.append(result["sample"])
            else:
                results.append(result)
        except Exception as e:
            logger.warning(f"  Error processing {agpt_file.name}: {e}")
            errors.append(agpt_file.name)

    logger.info(f"\nProcessed: {len(results)}, Missing DIAMOND: {len(missing)}, Errors: {len(errors)}")

    if not results:
        logger.error("No results to analyze!")
        return 1

    # Build DataFrame
    df = pd.DataFrame(results)
    df["source_type"] = df["sample"].apply(classify_source_type)

    # =========================================================================
    # Per-sample results
    # =========================================================================
    per_sample_file = OUTPUT_DIR / f"algagpt_vs_diamond_per_sample_{TIMESTAMP}.tsv"
    with open(per_sample_file, 'w') as f:
        f.write(f"# AlgaGPT vs DIAMOND BLAST Per-Sample Comparison\n")
        f.write(f"# Date: {datetime.now().isoformat()}\n")
        f.write(f"# Script: {os.path.abspath(__file__)}\n")
        f.write(f"# AlgaGPT dir: {ALGAGPT_DIR}\n")
        f.write(f"# DIAMOND dir: {DIAMOND_RAW_DIR}\n")
        f.write(f"# Samples: {len(df)}\n")
        df.to_csv(f, sep='\t', index=False)
    logger.info(f"Saved: {per_sample_file}")

    # =========================================================================
    # Aggregate statistics
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 70)

    total_proteins = df["total_proteins"].sum()
    total_algae = df["algae_count"].sum()
    total_conta = df["conta_count"].sum()
    total_diamond = df["diamond_hits"].sum()
    total_algae_hit = df["algae_hit"].sum()
    total_algae_nohit = df["algae_nohit"].sum()
    total_conta_hit = df["conta_hit"].sum()
    total_conta_nohit = df["conta_nohit"].sum()
    total_diamond_only = df["diamond_only"].sum()

    logger.info(f"\nSamples analyzed: {len(df):,}")
    logger.info(f"Total proteins: {total_proteins:,}")
    logger.info(f"")
    logger.info(f"AlgaGPT classification:")
    logger.info(f"  Algae:        {total_algae:>14,} ({100*total_algae/total_proteins:.2f}%)")
    logger.info(f"  Contaminant:  {total_conta:>14,} ({100*total_conta/total_proteins:.2f}%)")
    logger.info(f"")
    logger.info(f"DIAMOND NR hits:")
    logger.info(f"  With hit:     {total_diamond:>14,} ({100*total_diamond/total_proteins:.2f}%)")
    logger.info(f"  No hit (dark):{total_proteins - total_diamond:>14,} ({100*(total_proteins-total_diamond)/total_proteins:.2f}%)")
    logger.info(f"")
    logger.info(f"Cross-tabulation (2x2 confusion matrix):")
    logger.info(f"{'':>20} {'DIAMOND hit':>14} {'No hit':>14} {'Total':>14}")
    logger.info(f"{'AlgaGPT algae':>20} {total_algae_hit:>14,} {total_algae_nohit:>14,} {total_algae:>14,}")
    logger.info(f"{'AlgaGPT conta':>20} {total_conta_hit:>14,} {total_conta_nohit:>14,} {total_conta:>14,}")
    logger.info(f"{'Total':>20} {total_algae_hit+total_conta_hit:>14,} {total_algae_nohit+total_conta_nohit:>14,} {total_proteins:>14,}")

    logger.info(f"")
    logger.info(f"Key metrics:")
    if total_diamond > 0:
        logger.info(f"  Of DIAMOND hits: {100*total_algae_hit/total_diamond:.1f}% called algae, "
                     f"{100*total_conta_hit/total_diamond:.1f}% called conta")
    if total_algae > 0:
        logger.info(f"  DIAMOND hit rate in algae proteins:  {100*total_algae_hit/total_algae:.2f}%")
    if total_conta > 0:
        logger.info(f"  DIAMOND hit rate in conta proteins:  {100*total_conta_hit/total_conta:.2f}%")

    hit_rate_algae = total_algae_hit / total_algae if total_algae > 0 else 0
    hit_rate_conta = total_conta_hit / total_conta if total_conta > 0 else 0
    if hit_rate_algae > 0:
        enrichment = hit_rate_conta / hit_rate_algae
        logger.info(f"  Conta/algae DIAMOND enrichment ratio: {enrichment:.2f}x")

    logger.info(f"  DIAMOND-only proteins (not in algaGPT): {total_diamond_only:,}")

    # =========================================================================
    # Per source-type breakdown
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info(f"BY SOURCE TYPE")
    logger.info(f"{'=' * 70}")

    for src_type in ["MMETSP", "MGYA", "RefGenome", "Reference"]:
        sub = df[df["source_type"] == src_type]
        if len(sub) == 0:
            continue

        st = sub["total_proteins"].sum()
        sa = sub["algae_count"].sum()
        sc = sub["conta_count"].sum()
        sd = sub["diamond_hits"].sum()
        sah = sub["algae_hit"].sum()
        sch = sub["conta_hit"].sum()

        logger.info(f"\n  {src_type} ({len(sub)} samples, {st:,} proteins):")
        logger.info(f"    AlgaGPT algae: {100*sa/st:.1f}%, conta: {100*sc/st:.1f}%")
        logger.info(f"    DIAMOND hit rate: {100*sd/st:.2f}%")
        if sd > 0:
            logger.info(f"    Of hits: {100*sah/sd:.1f}% algae, {100*sch/sd:.1f}% conta")
        if sa > 0 and sc > 0:
            rate_a = sah / sa
            rate_c = sch / sc
            if rate_a > 0:
                logger.info(f"    Conta/algae enrichment: {rate_c/rate_a:.2f}x")

    # =========================================================================
    # Distribution statistics
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info(f"DISTRIBUTION OF PER-SAMPLE METRICS")
    logger.info(f"{'=' * 70}")

    for col, desc in [
        ("pct_algae", "% proteins called algae"),
        ("pct_diamond", "% proteins with DIAMOND hit"),
        ("pct_algae_of_hits", "% of DIAMOND hits called algae"),
        ("diamond_rate_algae", "DIAMOND hit rate in algae proteins"),
        ("diamond_rate_conta", "DIAMOND hit rate in conta proteins"),
    ]:
        vals = df[col].dropna()
        if len(vals) > 0:
            logger.info(f"\n  {desc}:")
            logger.info(f"    Mean: {vals.mean():.2f}%, Median: {vals.median():.2f}%")
            logger.info(f"    Q25: {vals.quantile(0.25):.2f}%, Q75: {vals.quantile(0.75):.2f}%")
            logger.info(f"    Min: {vals.min():.2f}%, Max: {vals.max():.2f}%")

    # =========================================================================
    # Correlation: algaGPT algae% vs DIAMOND hit rate
    # =========================================================================
    from scipy.stats import spearmanr, pearsonr

    valid = df.dropna(subset=["pct_algae", "pct_diamond"])
    if len(valid) > 10:
        rho, pval = spearmanr(valid["pct_algae"], valid["pct_diamond"])
        logger.info(f"\n  Spearman(pct_algae, pct_diamond): rho={rho:.4f}, p={pval:.2e}")

        rho2, pval2 = spearmanr(valid["diamond_rate_algae"], valid["diamond_rate_conta"])
        logger.info(f"  Spearman(diamond_rate_algae, diamond_rate_conta): rho={rho2:.4f}, p={pval2:.2e}")

    # =========================================================================
    # Summary file
    # =========================================================================
    summary_file = OUTPUT_DIR / f"algagpt_vs_diamond_summary_{TIMESTAMP}.tsv"
    summary_data = {
        "metric": [
            "samples_analyzed",
            "total_proteins",
            "algagpt_algae",
            "algagpt_conta",
            "diamond_hits",
            "diamond_no_hit",
            "algae_with_hit",
            "algae_no_hit",
            "conta_with_hit",
            "conta_no_hit",
            "pct_algae",
            "pct_diamond_hit",
            "pct_algae_of_diamond_hits",
            "diamond_rate_in_algae",
            "diamond_rate_in_conta",
            "conta_algae_enrichment",
        ],
        "value": [
            len(df),
            total_proteins,
            total_algae,
            total_conta,
            total_diamond,
            total_proteins - total_diamond,
            total_algae_hit,
            total_algae_nohit,
            total_conta_hit,
            total_conta_nohit,
            round(100 * total_algae / total_proteins, 4),
            round(100 * total_diamond / total_proteins, 4),
            round(100 * total_algae_hit / total_diamond, 4) if total_diamond > 0 else "NA",
            round(100 * total_algae_hit / total_algae, 4) if total_algae > 0 else "NA",
            round(100 * total_conta_hit / total_conta, 4) if total_conta > 0 else "NA",
            round(hit_rate_conta / hit_rate_algae, 4) if hit_rate_algae > 0 else "NA",
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    with open(summary_file, 'w') as f:
        f.write(f"# AlgaGPT vs DIAMOND BLAST Summary Statistics\n")
        f.write(f"# Date: {datetime.now().isoformat()}\n")
        f.write(f"# Script: {os.path.abspath(__file__)}\n")
        summary_df.to_csv(f, sep='\t', index=False)
    logger.info(f"\nSaved: {summary_file}")

    # =========================================================================
    # Missing samples report
    # =========================================================================
    if missing:
        missing_file = OUTPUT_DIR / f"diamond_missing_samples_{TIMESTAMP}.txt"
        with open(missing_file, 'w') as f:
            f.write(f"# Samples with algaGPT results but no DIAMOND results\n")
            f.write(f"# Count: {len(missing)}\n")
            for s in sorted(missing):
                f.write(f"{s}\n")
        logger.info(f"Missing DIAMOND samples: {len(missing)} (saved to {missing_file})")

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Analysis complete. Output: {OUTPUT_DIR}")
    logger.info(f"{'=' * 70}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
