#!/usr/bin/env python3
"""
Biological proof-of-concept panels for Shady review #2:
  Panel A: FTR1 (PF03239) iron permease — observed vs predicted CLR abundance
           (forward env->domain, 5-fold CV), R^2 = 0.40
  Panel B: Dunaliella novel domain — CLR abundance vs ocean bathymetry,
           Spearman rho = 0.566 (the tightest single domain-environment coupling)

Both plot REAL per-sample vectors (no synthetic data):
  - FTR1:  source_data/panel_regen/panel_FTR1_obs_pred_*.tsv  (HPC job 16343493)
  - Duna:  source_data/panel_regen/panel_dunaliella_clr_scatter_*.tsv (HPC job 16344168,
           verified to reproduce published rho=0.566 before emission)

Style: FIGURE_PROTOCOL.md — 6pt Arial, tara_style + Earth-from-Space palette,
hexbin OCEAN_CMAP, matches existing Figure5 calibration panels.
"""
import glob
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "figures"))
from tara_style import apply_tara_style, save_figure  # noqa: E402
from palette import OCEAN_CMAP  # noqa: E402

apply_tara_style()
SD = ROOT / "source_data" / "panel_regen"
TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def latest(pattern):
    hits = sorted(glob.glob(str(SD / pattern)))
    return hits[-1] if hits else None


fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))

# ---- Panel A: FTR1 obs vs pred ----
ftr1_fp = latest("panel_FTR1_obs_pred_*.tsv")
axA = axes[0]
if ftr1_fp:
    d = pd.read_csv(ftr1_fp, sep="\t")
    o, p = d["observed_clr"].values, d["predicted_clr"].values
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - o.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    hb = axA.hexbin(o, p, gridsize=25, cmap=OCEAN_CMAP, mincnt=1, linewidths=0)
    lo, hi = min(o.min(), p.min()), max(o.max(), p.max())
    axA.plot([lo, hi], [lo, hi], "--", color="0.4", lw=0.8)
    axA.set_xlabel("Observed FTR1 abundance (CLR)", fontsize=6)
    axA.set_ylabel("Predicted FTR1 abundance (CLR)", fontsize=6)
    axA.text(0.05, 0.93, f"$R^2$ = {r2:.2f}\nn = {len(d):,}", transform=axA.transAxes,
             fontsize=6, va="top")
    axA.set_title("FTR1 iron permease (PF03239)", fontsize=6, fontweight="bold")
    cb = fig.colorbar(hb, ax=axA, shrink=0.8, pad=0.02)
    cb.ax.tick_params(labelsize=5)
    cb.set_label("samples", fontsize=5)
else:
    axA.text(0.5, 0.5, "FTR1 vector not found", ha="center", transform=axA.transAxes, fontsize=6)
axA.tick_params(labelsize=6)
axA.text(-0.18, 1.02, "A", transform=axA.transAxes, fontsize=8, fontweight="bold")

# ---- Panel B: Dunaliella CLR abundance vs bathymetry ----
duna_fp = latest("panel_dunaliella_clr_scatter_*.tsv")
axB = axes[1]
if duna_fp:
    d = pd.read_csv(duna_fp, sep="\t")
    x = d["bathymetry_m"].values
    y = d["dunaliella_clr_abundance"].values
    # rho on the full per-sample data (matches published 0.566) — annotated, not plotted raw
    rho, pval = spearmanr(y, x)
    # Bin by depth zone to make the trend legible despite zero-inflation.
    # Bathymetry is negative for ocean depth; bin from deep (most negative) to shallow/land.
    edges = np.array([-6000, -4000, -3000, -2000, -1000, -200, 0, 5000])
    labels = ["<-4000", "-4000\nto\n-3000", "-3000\nto\n-2000", "-2000\nto\n-1000",
              "-1000\nto\n-200", "-200\nto 0", ">0"]
    idx = np.digitize(x, edges[1:-1], right=False)
    med, q1, q3, ns = [], [], [], []
    for b in range(len(labels)):
        yv = y[idx == b]
        if len(yv):
            med.append(np.median(yv)); q1.append(np.percentile(yv, 25))
            q3.append(np.percentile(yv, 75)); ns.append(len(yv))
        else:
            med.append(np.nan); q1.append(np.nan); q3.append(np.nan); ns.append(0)
    pos = np.arange(len(labels))
    med = np.array(med); q1 = np.array(q1); q3 = np.array(q3)
    axB.fill_between(pos, q1, q3, color="#7FB7BE", alpha=0.45, lw=0, label="IQR")
    axB.plot(pos, med, "-o", color="#1F4E5F", lw=1.0, ms=3.0, label="median")
    axB.set_xticks(pos)
    axB.set_xticklabels(labels, fontsize=5)
    axB.set_xlabel("Ocean bathymetry zone (m)", fontsize=6)
    axB.set_ylabel("Dunaliella domain abundance (CLR)", fontsize=6)
    ptxt = "p < 1e-100" if pval < 1e-100 else f"p = {pval:.1e}"
    axB.text(0.04, 0.93, f"Spearman $\\rho$ = {rho:.3f}\n{ptxt}\nn = {len(d):,}",
             transform=axB.transAxes, fontsize=6, va="top")
    axB.set_title("Dunaliella dark domain (no Pfam/InterPro/PDB)", fontsize=6, fontweight="bold")
    axB.legend(fontsize=5, loc="upper right", frameon=False)
else:
    axB.text(0.5, 0.5, "Dunaliella vector pending\n(HPC job 16344168)",
             ha="center", va="center", transform=axB.transAxes, fontsize=6)
axB.tick_params(labelsize=6)
axB.text(-0.18, 1.02, "B", transform=axB.transAxes, fontsize=8, fontweight="bold")

fig.tight_layout()
name = f"biological_poc_panels_{TS}"
save_figure(fig, name, output_dir=str(ROOT / "figures"))
print(f"Saved figures/{name}.(pdf/svg)")
print(f"  FTR1 source: {ftr1_fp}")
print(f"  Duna source: {duna_fp}")
