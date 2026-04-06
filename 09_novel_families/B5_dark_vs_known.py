#!/usr/bin/env python3
"""
B5: Statistical Comparison — Dark vs Known R² Distributions

Provenance:
    Script: scripts/novel_families/B5_dark_vs_known.py
    Generated: 2026-02-21
    Pipeline: Novel Domain Discovery — Track B

Purpose:
    Same three-test framework as A9, applied to dark proteome families.
    Key hypothesis: Are dark families enriched for high-R² domains compared
    to annotated Pfam families?

Input:
    - novel_families/results/track_b/dark_family_xgboost_forward_r2*.tsv
    - supplement/TableS12_spatial_block_cv_*.tsv

Output:
    - novel_families/results/track_b/dark_vs_known_r2_comparison.tsv
    - novel_families/results/track_b/dark_vs_known_statistics.md
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import socket

def get_base_dir(project_name: str) -> Path:
    hostname = socket.gethostname()
    if os.path.isdir("/scratch/drn2") or "dn" in hostname or "cn" in hostname or "gpu" in hostname or "jubail" in hostname:
        return Path(f"/scratch/drn2/PROJECTS/{project_name}")
    return Path(f"/media/drn/External1/{project_name}")

def rank_biserial(U, n1, n2):
    return 1 - 2 * U / (n1 * n2)

def main():
    BASE = get_base_dir("TARA-LA4SR")
    RESULTS_DIR = BASE / "novel_families" / "results" / "track_b"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT = BASE / "MANUSCRIPT"

    print("=" * 60)
    print("  B5: Dark vs Known R² Comparison")
    print("=" * 60)
    print()

    # Concatenate chunk results if needed
    chunk_files = sorted(RESULTS_DIR.glob("dark_family_xgboost_forward_r2_chunk*.tsv"))
    single_file = RESULTS_DIR / "dark_family_xgboost_forward_r2.tsv"

    if chunk_files:
        print(f"  Concatenating {len(chunk_files)} chunk files...")
        dfs = [pd.read_csv(f, sep="\t") for f in chunk_files]
        dark_df = pd.concat(dfs, ignore_index=True)
    elif single_file.exists():
        dark_df = pd.read_csv(single_file, sep="\t")
    else:
        print("  ERROR: No dark family R² results found")
        sys.exit(1)

    dark_r2 = dark_df["r2_overall"].dropna().values
    print(f"  Dark families: {len(dark_r2)}")
    print(f"  Dark R² mean: {dark_r2.mean():.4f}")

    # Load known Pfam R²
    ralph4_path = MANUSCRIPT / "ralph4_statistical_reanalysis" / "forward_r2_all_pfams.tsv"
    s12_files = sorted(glob.glob(str(MANUSCRIPT / "supplement" / "TableS12_spatial_block_cv_*.tsv")))

    known_r2 = None
    if ralph4_path.exists():
        kdf = pd.read_csv(ralph4_path, sep="\t")
        if "r2_overall" in kdf.columns:
            known_r2 = kdf["r2_overall"].dropna().values
    elif s12_files:
        kdf = pd.read_csv(s12_files[-1], sep="\t", comment="#")
        fwd = kdf[(kdf["direction"] == "forward") &
                  (kdf["analysis"] == "spatial_block_cv_summary")]
        if len(fwd) > 0:
            known_r2 = fwd["r2"].dropna().values

    if known_r2 is None:
        print("  ERROR: No known Pfam R² found")
        sys.exit(1)

    print(f"  Known Pfam: {len(known_r2)} domains")
    print(f"  Known R² mean: {known_r2.mean():.4f}")

    # Test 1: Mann-Whitney U
    U, p = stats.mannwhitneyu(dark_r2, known_r2, alternative="two-sided")
    rbc = rank_biserial(U, len(dark_r2), len(known_r2))
    med_diff = np.median(dark_r2) - np.median(known_r2)

    print(f"\n  Mann-Whitney U: U={U:.0f}, p={p:.2e}, rbc={rbc:.4f}")
    print(f"  Median diff: {med_diff:.4f}")

    # Save
    stats_path = RESULTS_DIR / "dark_vs_known_statistics.md"
    with open(stats_path, "w") as f:
        f.write("# Dark vs Known Domain R² Comparison\n\n")
        f.write(f"- Script: {os.path.abspath(__file__)}\n")
        f.write(f"- Date: {__import__('datetime').datetime.now()}\n\n")
        f.write(f"| Group | N | Mean R² | Median R² |\n|---|---|---|---|\n")
        f.write(f"| Dark families | {len(dark_r2)} | {dark_r2.mean():.4f} | "
                f"{np.median(dark_r2):.4f} |\n")
        f.write(f"| Known Pfam | {len(known_r2)} | {known_r2.mean():.4f} | "
                f"{np.median(known_r2):.4f} |\n\n")
        f.write(f"Mann-Whitney U = {U:.0f}, p = {p:.2e}\n")
        f.write(f"Rank-biserial r = {rbc:.4f}\n")
        f.write(f"Median difference = {med_diff:.4f}\n")

    comp_df = pd.DataFrame([{
        "test": "mann_whitney",
        "U": U, "p_value": p,
        "rank_biserial": rbc,
        "median_diff": med_diff,
        "n_dark": len(dark_r2),
        "n_known": len(known_r2),
    }])
    comp_df.to_csv(RESULTS_DIR / "dark_vs_known_r2_comparison.tsv",
                   sep="\t", index=False)

    print(f"\n  Written: {stats_path}")
    print(f"  Done: {__import__('datetime').datetime.now()}")

if __name__ == "__main__":
    main()
