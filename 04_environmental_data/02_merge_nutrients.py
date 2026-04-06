#!/usr/bin/env python3
"""
02_merge_nutrients.py — Merge extracted nutrient values into existing GEE+PFAM dataset.

Joins nutrients_at_sample_coordinates.tsv with the existing merged dataset on
assembly_id, appending 5 new columns (nitrate, phosphate, silicate, oxygen, MLD).

Input:
    - {BASE}/03_analyses/woa23_nutrients/nutrients_at_sample_coordinates.tsv
    - {BASE}/03_analyses/ALGAGPT-based-analyses/
      algagpt_gee_pfam_merged_GPS_RECOVERED_20260124_114445.tsv

Output:
    - {BASE}/03_analyses/ALGAGPT-based-analyses/
      algagpt_gee_pfam_nutrients_merged_{timestamp}.tsv

Usage:
    python3 scripts/nutrients/02_merge_nutrients.py

Author: TARA-OMEN analysis pipeline
Date: 2026-03-20
"""

import socket
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_base_dir():
    hostname = socket.gethostname()
    if any(x in hostname for x in ['cn', 'gpu', 'dn', 'jubail', 'login', 'fast']):
        return Path("/scratch/drn2/PROJECTS/TARA-LA4SR")
    else:
        script_dir = Path(__file__).resolve().parent
        return script_dir.parent.parent / "data"

BASE = get_base_dir()

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{ts}] 02_merge_nutrients.py starting on {socket.gethostname()}")

    # --- Load nutrient data ---
    if "scratch" in str(BASE):
        nutrient_path = BASE / "03_analyses" / "woa23_nutrients" / "nutrients_at_sample_coordinates.tsv"
    else:
        nutrient_path = BASE / "woa23_nutrients" / "nutrients_at_sample_coordinates.tsv"

    if not nutrient_path.exists():
        print(f"ERROR: Nutrient file not found: {nutrient_path}")
        print("       Run 01_extract_nutrients.py first.")
        sys.exit(1)

    print(f"Loading nutrients: {nutrient_path}")
    nutrients = pd.read_csv(nutrient_path, sep='\t', comment='#')
    print(f"  {len(nutrients)} samples, columns: {list(nutrients.columns)}")

    # Keep only the nutrient columns (not lat/lon, which already exist in merged)
    nutrient_cols = ['nitrate_umol_l', 'phosphate_umol_l', 'silicate_umol_l',
                     'oxygen_umol_l', 'mld_m']
    merge_cols = ['assembly_id'] + [c for c in nutrient_cols if c in nutrients.columns]
    nutrients_slim = nutrients[merge_cols].copy()

    # --- Load existing merged dataset ---
    if "scratch" in str(BASE):
        merged_dir = BASE / "03_analyses" / "ALGAGPT-based-analyses"
    else:
        merged_dir = BASE

    # Search multiple locations for the merged dataset
    import glob as glob_mod
    merged_path = None
    search_patterns = [
        str(merged_dir / "algagpt_gee_pfam_merged_GPS_RECOVERED_*.tsv"),
        str(merged_dir / "algagpt_gee_pfam_merged_SMART_*.tsv"),
        str(merged_dir / "algagpt_gee_pfam_merged_*.tsv"),
    ]
    if "scratch" in str(BASE):
        search_patterns.extend([
            str(BASE / "03_analyses" / "algagpt_gee_pfam_merged_SMART_*.tsv"),
            str(BASE / "03_analyses" / "algagpt_gee_pfam_merged_*.tsv"),
            str(BASE / "algagpt_gee_pfam_merged_*.tsv"),
        ])
    for pat in search_patterns:
        found = sorted(glob_mod.glob(pat))
        # Skip any nutrients-merged files (that's our output)
        found = [f for f in found if 'nutrients' not in f]
        if found:
            merged_path = Path(found[-1])
            break
    if merged_path is None:
        print(f"ERROR: Merged dataset not found")
        sys.exit(1)

    print(f"Loading merged dataset: {merged_path}")

    # Read provenance header
    header_lines = []
    with open(merged_path) as f:
        for line in f:
            if line.startswith('#'):
                header_lines.append(line.rstrip())
            else:
                break

    df = pd.read_csv(merged_path, sep='\t', comment='#', low_memory=False)
    print(f"  {len(df)} samples, {len(df.columns)} columns")

    # Check for existing nutrient columns (avoid duplicates)
    existing_nutrient_cols = [c for c in nutrient_cols if c in df.columns]
    if existing_nutrient_cols:
        print(f"  WARNING: Nutrient columns already exist: {existing_nutrient_cols}")
        print(f"  Dropping existing columns before merge.")
        df = df.drop(columns=existing_nutrient_cols)

    # --- Merge ---
    merged = df.merge(nutrients_slim, on='assembly_id', how='left')
    print(f"  Merged: {len(merged)} samples, {len(merged.columns)} columns")

    # Coverage stats
    for col in nutrient_cols:
        if col in merged.columns:
            n_valid = merged[col].notna().sum()
            pct = 100.0 * n_valid / len(merged)
            print(f"  {col}: {n_valid}/{len(merged)} ({pct:.1f}%) non-NaN")

    # --- Write output ---
    out_name = f"algagpt_gee_pfam_nutrients_merged_{ts}.tsv"
    out_path = merged_dir / out_name

    # Build provenance header
    new_header = header_lines + [
        f"# --- Nutrient data appended: {datetime.now().isoformat()} ---",
        f"# Source: WOA23 annual climatology (1° grid, surface depth)",
        f"# MLD source: de Boyer Montégut climatology (annual mean)",
        f"# Nutrient extraction script: scripts/nutrients/01_extract_nutrients.py",
        f"# Merge script: scripts/nutrients/02_merge_nutrients.py",
        f"# Added columns: {', '.join(nutrient_cols)}",
    ]

    with open(out_path, 'w') as f:
        for line in new_header:
            f.write(line + '\n')
        merged.to_csv(f, sep='\t', index=False)

    print(f"\nOutput: {out_path}")
    print(f"  Total columns: {len(merged.columns)}")

    # Also create a symlink for easier reference
    link_path = merged_dir / "algagpt_gee_pfam_nutrients_merged_LATEST.tsv"
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(out_path.name)
    print(f"  Symlink: {link_path} -> {out_name}")

if __name__ == '__main__':
    main()
