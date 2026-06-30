#!/usr/bin/env python3
"""
Extract the Dunaliella novel-domain CLR-abundance vs bathymetry scatter vector
for the Figure-6 proof-of-concept panel (Shady review #2), reproducing EXACTLY
the method of scripts/novel_domains/08_env_correlation.py that produced the
authoritative rho=0.566 in novel_domain_env_correlations.tsv.

Method (from 08_env_correlation.py):
  - load novel_domain_count_matrix.tsv (index=assembly_id, cols=domains)
  - strip trailing ".aa" from sample index
  - CLR transform over the FULL domain matrix (pseudocount +1, log, subtract
    per-sample geometric mean across ALL domains) -- cannot be done on one column
  - intersect with env samples; Spearman(CLR[domain], bathymetry)

SELF-CHECK: asserts the recomputed rho is within 0.01 of the published 0.5659.
If it does not reproduce, the script FAILS loudly rather than emit a wrong panel.
No synthetic data.
"""
import socket
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def base():
    p = Path("/scratch/drn2/PROJECTS/TARA-LA4SR")
    return p if p.exists() else Path("/media/drn2/External/TARA-LA4SR")


B = base()
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
NOVEL = B / "03_analyses/novel_domains/results/novel_domain_count_matrix.tsv"
MERGED = B / "03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_nutrients_merged_20260320_090002.tsv"
OUT = B / "MANUSCRIPT/source_data/panel_regen"
OUT.mkdir(parents=True, exist_ok=True)
DOM = "NOVEL_33041-Dunaliella_ROIL-10xG.AAC.fa.2"
PUBLISHED_RHO = 0.5659049513838765


def clr_transform(count_matrix):
    X = count_matrix + 1.0
    log_X = np.log(X)
    geo_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geo_mean


print(f"[{datetime.now()}] host={socket.gethostname()}")
print(f"Loading novel matrix {NOVEL} ...")
dom_df = pd.read_csv(NOVEL, sep="\t", index_col=0)
dom_df.index = dom_df.index.str.replace(r"\.aa$", "", regex=True)
print(f"  novel matrix: {dom_df.shape[0]} samples x {dom_df.shape[1]} domains")

print(f"Loading bathymetry from {MERGED} ...")
env = pd.read_csv(MERGED, sep="\t", comment="#", usecols=["assembly_id", "bathymetry_m"])
env["assembly_id"] = env["assembly_id"].astype(str)
env = env.set_index("assembly_id")
env["bathymetry_m"] = pd.to_numeric(env["bathymetry_m"], errors="coerce")

shared = dom_df.index.intersection(env.index)
print(f"  shared samples: {len(shared)}")

# CLR over the FULL matrix on shared samples (exactly as 08_env_correlation.py)
clr = clr_transform(dom_df.loc[shared].values)
clr_df = pd.DataFrame(clr, index=shared, columns=dom_df.columns)

bathy = env.loc[shared, "bathymetry_m"].values
valid = ~np.isnan(bathy)
dom_clr = clr_df.loc[shared, DOM].values[valid]
bathy_v = bathy[valid]

rho, pval = spearmanr(dom_clr, bathy_v)
n = int(valid.sum())
print(f"RECOMPUTED rho = {rho:.4f}  (published {PUBLISHED_RHO:.4f})  n={n}")

# self-check
assert abs(rho - PUBLISHED_RHO) < 0.01, (
    f"FAILED to reproduce published rho: got {rho:.4f}, expected {PUBLISHED_RHO:.4f}. "
    "Do NOT use this output for a panel."
)
print("SELF-CHECK PASSED: reproduces published 0.566")

out = pd.DataFrame({
    "assembly_id": np.array(shared)[valid],
    "dunaliella_clr_abundance": dom_clr,
    "bathymetry_m": bathy_v,
})
fp = OUT / f"panel_dunaliella_clr_scatter_{TS}.tsv"
out.to_csv(fp, sep="\t", index=False)

prov = OUT / f"panel_dunaliella_clr_provenance_{TS}.md"
with open(prov, "w") as f:
    f.write("# Dunaliella CLR-abundance vs bathymetry scatter — VERIFIED\n\n")
    f.write(f"- Host {socket.gethostname()}  Date {datetime.now()}\n")
    f.write(f"- Domain {DOM}\n")
    f.write(f"- Method: full-matrix CLR (08_env_correlation.py), strip .aa, Spearman\n")
    f.write(f"- **rho = {rho:.4f}, p = {pval:.3e}, n = {n}** (published 0.5659; self-check PASSED)\n")
    f.write(f"- Scatter vector: {fp}\n")
print(f"Wrote {fp}\nWrote {prov}")
