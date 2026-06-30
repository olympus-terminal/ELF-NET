#!/usr/bin/env python3
"""
Figure S1 sub-block: CCA row 1 (canonical correlations, permutation test, shared
variance) rendered as a wide HALF-HEIGHT strip for inclusion at the bottom of the
restructured Figure S1.

These are the former Figure S2 panels A-C, relocated to S1 per the new layout
(trees -> half-height UMAP -> this CCA row). Rendered natively at 6pt from the
same cached CCA results the full S2 figure uses (no recomputation).

Inputs (all cached, real data):
  - cca_summary_20260122_104244.json        (observed canonical correlations)
  - archive/ralph4_statistical_reanalysis/cca_permutation.tsv (permutation null)

Output: figures/FigureS1_ccarow_<TS>.pdf/.svg
Generated: 2026-06-27
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = Path(__file__).parent
MANUSCRIPT_DIR = SCRIPT_DIR.parent
BASE_DIR = Path("/media/drn2/External/TARA-Oceans")
CCA_REPORT_DIR = BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "env_pfam_manifold" / "reports"
PERM_PATH = MANUSCRIPT_DIR / "archive" / "ralph4_statistical_reanalysis" / "cca_permutation.tsv"

sys.path.insert(0, str(SCRIPT_DIR))
from palette import DEEP_OCEAN, COASTAL_BLUE, PALE_AQUA, CLAY, SIENNA

mpl.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 6, 'axes.labelsize': 6, 'axes.titlesize': 6,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
    'axes.linewidth': 0.25, 'xtick.major.width': 0.25, 'ytick.major.width': 0.25,
    'xtick.major.size': 2, 'ytick.major.size': 2,
    'axes.labelpad': 1, 'xtick.major.pad': 1, 'ytick.major.pad': 1,
})

TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Load cached data ──
with open(CCA_REPORT_DIR / "cca_summary_20260122_104244.json") as f:
    observed_ccs = np.array(json.load(f)['canonical_correlations'])
perm_df = pd.read_csv(PERM_PATH, sep='\t', comment='#')
null_max = perm_df['null_max'].max()
null_mean = perm_df['null_mean'].mean()
perm_cc1 = perm_df[perm_df['component'] == 'CC1'].iloc[0]
r2_per_cc = observed_ccs ** 2
cumulative_prop = np.cumsum(r2_per_cc) / np.sum(r2_per_cc)
x_a = np.arange(10)

# ── Wide half-height strip: 1 row x 3 panels ──
fig = plt.figure(figsize=(7.0, 1.9))
gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.0, 0.8, 1.0],
                       wspace=0.42, left=0.07, right=0.95, top=0.84, bottom=0.22)

# Panel A: canonical correlations + permutation null lines
ax_a = fig.add_subplot(gs[0, 0])
ax_a.bar(x_a, observed_ccs, color=[DEEP_OCEAN if i < 5 else COASTAL_BLUE for i in range(10)],
         edgecolor='none', width=0.7, alpha=0.9)
ax_a.axhline(null_max, color=SIENNA, linewidth=0.7, linestyle='--', label=f'Null max ({null_max:.3f})')
ax_a.axhline(null_mean, color=CLAY, linewidth=0.7, linestyle=':', label=f'Null mean ({null_mean:.3f})')
ax_a.set_xticks(x_a)
ax_a.set_xticklabels([f'CC{i+1}' for i in range(10)], fontsize=5, rotation=45, ha='right')
ax_a.set_ylabel('Canonical correlation')
ax_a.set_title('Canonical correlations (n = 995)', fontsize=6, fontweight='bold')
ax_a.set_ylim(0, 0.95)
ax_a.legend(loc='upper right', frameon=False, fontsize=4.5)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.text(-0.18, 1.12, 'L', transform=ax_a.transAxes, fontsize=6, fontweight='bold', va='top')

# Panel B: CC1 permutation test (normal approximation of null + observed line)
ax_b = fig.add_subplot(gs[0, 1])
null_mu = perm_cc1['null_mean']
null_sd = perm_cc1['null_std']
obs_cc1 = perm_cc1['observed_cc']
x_null = np.linspace(null_mu - 4*null_sd, null_mu + 4*null_sd, 200)
y_null = (1 / (null_sd * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_null - null_mu)/null_sd)**2)
ax_b.fill_between(x_null, y_null, color=PALE_AQUA, alpha=0.7)
ax_b.plot(x_null, y_null, color=COASTAL_BLUE, linewidth=0.7)
# observed-CC1 marker line removed: it overlapped the legend box; the observed
# value is already reported in the annotation box below.
ax_b.set_xlabel('CC1 correlation')
ax_b.set_ylabel('Density (normal approx.)')
ax_b.set_title('Permutation test CC1', fontsize=6, fontweight='bold')
ax_b.text(0.97, 0.95,
          f'Observed: {obs_cc1:.3f}\nNull: {null_mu:.3f} ({null_sd:.3f})\nz = {perm_cc1["effect_size_z"]:.1f}\np < 0.001 (0/1000)',
          transform=ax_b.transAxes, fontsize=4.5, va='top', ha='right',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=(0.7,0.7,0.7), alpha=0.9, linewidth=0.25))
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.text(-0.20, 1.12, 'M', transform=ax_b.transAxes, fontsize=6, fontweight='bold', va='top')

# Panel C: shared variance decomposition
ax_c = fig.add_subplot(gs[0, 2])
ax_c.bar(x_a, r2_per_cc, color=COASTAL_BLUE, edgecolor='none', width=0.6, alpha=0.8)
ax_c2 = ax_c.twinx()
ax_c2.plot(x_a, cumulative_prop, 'o-', color=SIENNA, linewidth=0.8, markersize=3)
ax_c.set_xticks(x_a)
ax_c.set_xticklabels([f'CC{i+1}' for i in range(10)], fontsize=5, rotation=45, ha='right')
ax_c.set_ylabel('Shared variance (r2)', color=COASTAL_BLUE)
ax_c2.set_ylabel('Cumulative proportion', color=SIENNA)
ax_c.set_title('Per-component shared variance', fontsize=6, fontweight='bold')
ax_c2.set_ylim(0, 1.1)
ax_c.spines['top'].set_visible(False)
ax_c2.spines['top'].set_visible(False)
ax_c2.tick_params(axis='y', labelsize=5, colors=SIENNA)
ax_c.tick_params(axis='y', labelsize=5, colors=COASTAL_BLUE)
ax_c.text(-0.18, 1.12, 'N', transform=ax_c.transAxes, fontsize=6, fontweight='bold', va='top')

out_dir = MANUSCRIPT_DIR / "figures"
for fmt in ('pdf', 'svg'):
    p = out_dir / f"FigureS1_ccarow_{TS}.{fmt}"
    fig.savefig(p, format=fmt, bbox_inches='tight', transparent=True, edgecolor='none')
    print(f"Saved: {p}")
fig.savefig(out_dir / f"FigureS1_ccarow_{TS}_check.png", dpi=300, bbox_inches='tight', transparent=False)
plt.close(fig)
print("Done.")
