#!/usr/bin/env python3
"""
Dunaliella-only fix: compute Spearman rho between the Dunaliella novel domain
abundance and bathymetry. Standalone so we don't re-run the 27-min CV portion
(FTR1 + CA results from job 16343493 are already correct).

Fixes vs the first run (which gave n=0):
  - novel matrix assembly_id has a trailing ".aa" suffix; strip it before join
  - use the FULL bathymetry-available sample set (main.tex n=1,523), not the
    SST-key-filtered n=1,279 subset.
No synthetic data. Values reported as computed (main.tex claims rho=0.566).
"""
import socket
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def base():
    h = socket.gethostname()
    p = Path("/scratch/drn2/PROJECTS/TARA-LA4SR")
    return p if p.exists() else Path("/media/drn2/External/TARA-LA4SR")


B = base()
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
MERGED = B / "03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_nutrients_merged_20260320_090002.tsv"
NOVEL = B / "03_analyses/novel_domains/results/novel_domain_count_matrix.tsv"
OUT = B / "MANUSCRIPT/source_data/panel_regen"
OUT.mkdir(parents=True, exist_ok=True)
DOM = "NOVEL_33041-Dunaliella_ROIL-10xG.AAC.fa.2"

# only need assembly_id + bathymetry from the (huge) merged TSV
df = pd.read_csv(MERGED, sep="\t", comment="#", usecols=["assembly_id", "bathymetry_m"],
                 low_memory=False)
df["assembly_id"] = df["assembly_id"].astype(str)
df["bathymetry_m"] = pd.to_numeric(df["bathymetry_m"], errors="coerce")
full_b = df.dropna(subset=["bathymetry_m"])
print(f"bathymetry-available samples: {len(full_b)}")

nd = pd.read_csv(NOVEL, sep="\t", usecols=lambda c: c in ("assembly_id", DOM))
nd["assembly_id"] = nd["assembly_id"].astype(str).str.replace(r"\.aa$", "", regex=True)
print(f"novel matrix has domain column: {DOM in nd.columns}")

merged = full_b.merge(nd, on="assembly_id", how="inner").dropna(subset=[DOM, "bathymetry_m"])
print(f"joined n = {len(merged)}")

rho, pval = spearmanr(merged[DOM], merged["bathymetry_m"])
fp = OUT / f"panel_dunaliella_bathy_FIXED_{TS}.tsv"
merged.rename(columns={DOM: "dunaliella_domain_count"}).to_csv(fp, sep="\t", index=False)

prov = OUT / f"panel_dunaliella_provenance_FIXED_{TS}.md"
with open(prov, "w") as f:
    f.write("# Dunaliella novel domain vs bathymetry — FIXED join\n\n")
    f.write(f"- Host: {socket.gethostname()}  Date: {datetime.now()}\n")
    f.write(f"- Domain: {DOM}\n")
    f.write(f"- Inputs: {MERGED} ; {NOVEL}\n")
    f.write("- Fix: stripped '.aa' suffix from novel IDs; full bathymetry set\n")
    f.write(f"- **rho = {rho:.4f}, p = {pval:.3e}, n = {len(merged)}**\n")
    f.write("- main.tex claims rho=0.566, n=1,523 — compare\n")
print(f"RESULT: rho={rho:.4f} p={pval:.3e} n={len(merged)}")
print(f"Wrote {fp}\nWrote {prov}")
