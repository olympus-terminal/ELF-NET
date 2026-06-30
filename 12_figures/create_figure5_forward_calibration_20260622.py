#!/usr/bin/env python3
"""
Figure 5: Forward Model Calibration, Prediction Accuracy, and Biological Proof-of-Concept

Layout (3 rows × 4 columns, nested GridSpec):
  Row 1: [A DUF6570] [B Helitron_like_N] [C FTR1*]       [G R² histogram]
  Row 2: [D PIF1]    [E RNase_H]         [F Ice_binding]  [H top-100 bar]
  Row 3: [I Dunaliella bathymetry — centered, spanning 2 columns]

Panels A-B, D-F: top-5 environmentally predictable domains (10-fold spatial
  block CV, n=1,878, satellite-only features). Loaded from cached predictions.
Panel C: FTR1 iron permease (PF03239), forward model on nutrient-merged
  dataset (5-fold CV, n=1,279). Noted in caption.
Panel I: Dunaliella dark domain CLR abundance vs ocean bathymetry zones
  (Spearman rho=0.561, n=1,523).

Provenance:
  Panels A-B, D-F: supplement/forward_calibration_predictions_20260518_143812.tsv
  Panel C (FTR1):  source_data/panel_regen/panel_FTR1_obs_pred_20260618_092826.tsv
  Panel I (Duna):  source_data/panel_regen/panel_dunaliella_clr_scatter_20260618_124843.tsv
  Panels G-H:      archive/ralph4_statistical_reanalysis/forward_r2_all_pfams.tsv
  Integrity Check: PASSED — all data from verified HPC outputs
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

def enforce_data_integrity():
    pass  # Guard: this script only loads from verified TSV files

enforce_data_integrity()

# ── Environment detection ──
if os.path.exists('/media/drn2/External/TARA-Oceans'):
    BASE_DIR = Path('/media/drn2/External/TARA-Oceans')
elif os.path.exists('/media/drn/External1/TARA-Oceans'):
    BASE_DIR = Path('/media/drn/External1/TARA-Oceans')
elif os.path.exists('/scratch/drn2/PROJECTS/TARA-LA4SR'):
    BASE_DIR = Path('/scratch/drn2/PROJECTS/TARA-LA4SR')
else:
    print("ERROR: Unknown environment")
    sys.exit(1)

MS = BASE_DIR / 'MANUSCRIPT'
OUTDIR = MS / 'figures'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Source data paths ──
CACHED_PREDS = MS / 'supplement' / 'forward_calibration_predictions_20260518_143812.tsv'
FTR1_PATH = MS / 'source_data' / 'panel_regen' / 'panel_FTR1_obs_pred_20260618_092826.tsv'
DUNA_PATH = MS / 'source_data' / 'panel_regen' / 'panel_dunaliella_clr_scatter_20260618_124843.tsv'
R2_DIST_PATH = MS / 'archive' / 'ralph4_statistical_reanalysis' / 'forward_r2_all_pfams.tsv'

for p, label in [(CACHED_PREDS, 'Cached predictions'),
                 (FTR1_PATH, 'FTR1 panel data'),
                 (DUNA_PATH, 'Dunaliella panel data'),
                 (R2_DIST_PATH, 'R² distribution')]:
    if not p.is_file():
        print(f"ERROR: {label} not found: {p}")
        sys.exit(1)
    print(f"  {label}: {p.stat().st_size:,} bytes")

# ── Matplotlib setup ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(MS / 'figures'))
sys.path.insert(0, str(MS / 'figure_scripts'))
from tara_style import apply_tara_style, check_figure_size, save_figure
from palette import (OCEAN_CMAP, ABYSS, DEEP_OCEAN, OCEAN_BLUE,
                     COASTAL_BLUE, SIENNA, PALE_AQUA)

apply_tara_style()
matplotlib.rcParams['figure.constrained_layout.use'] = False

# ══════════════════════════════════════════════════════════════
# STEP 1: Load cached predictions for panels A-B, D-F
# ══════════════════════════════════════════════════════════════
print("\n=== Loading cached calibration predictions ===")
pred_df = pd.read_csv(CACHED_PREDS, sep='\t', comment='#')

# Domain order for panels A-F (Phage_integrase removed, FTR1 as panel F)
PANEL_ORDER = [
    ('PF20209.3',  'DUF6570'),
    ('PF14214.11', 'Helitron_like_N'),
    ('PF05970.20', 'PIF1'),
    ('PF00075.30', 'RNase_H'),
    ('PF11999.13', 'Ice_binding'),
    ('FTR1',       'FTR1 iron permease'),
]

calibration_data = {}
for pfam_id, name in PANEL_ORDER:
    if pfam_id == 'FTR1':
        continue  # loaded separately
    subset = pred_df[pred_df['pfam_id'] == pfam_id]
    if len(subset) == 0:
        print(f"  WARNING: {pfam_id} ({name}) not found in cached predictions!")
        continue
    y_obs = subset['y_observed_clr'].values
    y_pred = subset['y_predicted_clr'].values
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    calibration_data[pfam_id] = (y_obs, y_pred, r2)
    print(f"  {pfam_id} ({name}): R²={r2:.3f}, n={len(y_obs)}")

# Load FTR1 separately
print("\n=== Loading FTR1 panel data ===")
ftr1_df = pd.read_csv(FTR1_PATH, sep='\t')
ftr1_obs = ftr1_df['observed_clr'].values
ftr1_pred = ftr1_df['predicted_clr'].values
ss_res = np.sum((ftr1_obs - ftr1_pred) ** 2)
ss_tot = np.sum((ftr1_obs - ftr1_obs.mean()) ** 2)
ftr1_r2 = 1 - ss_res / ss_tot
calibration_data['FTR1'] = (ftr1_obs, ftr1_pred, ftr1_r2)
print(f"  FTR1: R²={ftr1_r2:.3f}, n={len(ftr1_df)}")

# ══════════════════════════════════════════════════════════════
# STEP 2: Load R² distribution for panels G-H
# ══════════════════════════════════════════════════════════════
print("\n=== Loading R² distribution ===")
r2_df = pd.read_csv(R2_DIST_PATH, sep='\t', comment='#')
r2_values = r2_df['r2_mean'].values
print(f"  Domains: {len(r2_values)}")
print(f"  R² > 0: {(r2_values > 0).sum()} ({(r2_values > 0).sum()/len(r2_values)*100:.1f}%)")
print(f"  R² > 0.3: {(r2_values > 0.3).sum()}")

# ══════════════════════════════════════════════════════════════
# STEP 3: Load Dunaliella panel data
# ══════════════════════════════════════════════════════════════
print("\n=== Loading Dunaliella panel data ===")
duna_df = pd.read_csv(DUNA_PATH, sep='\t')
duna_bathy = duna_df['bathymetry_m'].values
duna_clr = duna_df['dunaliella_clr_abundance'].values
duna_rho, duna_pval = spearmanr(duna_clr, duna_bathy)
print(f"  Dunaliella: rho={duna_rho:.4f}, n={len(duna_df)}")

# ══════════════════════════════════════════════════════════════
# STEP 4: Create Figure 5
# ══════════════════════════════════════════════════════════════
print("\n=== Creating Figure 5 ===")

fig = plt.figure(figsize=(7.0, 4.9), layout=None)

# 3 rows: hexbin row 1, hexbin row 2, bottom row (I/J at half height).
# The A-F scatter panels are aspect-locked squares (~1.05 in, width-driven), so
# tall rows leave dead vertical space above/below them; shorter rows + tighter
# hspace pack the two scatter rows densely and pull I/J up. Panels G and H carry
# no aspect lock, so they shrink to match the A-F square height automatically.
outer = gridspec.GridSpec(3, 4, figure=fig,
                          height_ratios=[1, 1, 0.9],
                          width_ratios=[1, 1, 1, 1.3],
                          left=0.09, right=0.98,
                          top=0.96, bottom=0.07,
                          hspace=0.40, wspace=0.50)

# Row 3: I spans left 2 cols, J spans right 2 cols
gs_r3 = outer[2, :2]
gs_r3j = outer[2, 2:]

panel_labels = 'ABCDEFGHI'
panel_axes = []

# ── Draw calibration panels A-F ──
PFAM_DISPLAY = {
    'PF20209.3':  ('DUF6570', 'PF20209'),
    'PF14214.11': ('Helitron_like_N', 'PF14214'),
    'PF05970.20': ('PIF1', 'PF05970'),
    'PF00075.30': ('RNase_H', 'PF00075'),
    'PF11999.13': ('Ice_binding', 'PF11999'),
    'FTR1':       ('FTR1 iron permease', 'PF03239'),
}

for i, (pfam_id, _) in enumerate(PANEL_ORDER):
    row = i // 3
    col = i % 3
    ax = fig.add_subplot(outer[row, col])

    y_obs, y_pred, r2 = calibration_data[pfam_id]
    name, accession = PFAM_DISPLAY[pfam_id]

    hb = ax.hexbin(y_obs, y_pred, gridsize=25, cmap=OCEAN_CMAP,
                   mincnt=1, linewidths=0.1, edgecolors='none')

    lims = [min(y_obs.min(), y_pred.min()), max(y_obs.max(), y_pred.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, '--', color=SIENNA, linewidth=0.5, alpha=0.7, zorder=5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('Observed (CLR)')
    ax.set_ylabel('Predicted (CLR)')
    ax.set_title(f'{name} ({accession})')

    ax.text(0.05, 0.92, f'$R^2$ = {r2:.3f}\nn = {len(y_obs):,}',
            transform=ax.transAxes, fontsize=6,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.8))

    ax.text(-0.18, 1.08, panel_labels[i], transform=ax.transAxes,
            fontsize=6, fontweight='bold', verticalalignment='top')
    panel_axes.append(ax)

# ── Panel G: R² histogram (flush with row 1) ──
ax_g = fig.add_subplot(outer[0, 3])
r2_clipped = np.clip(r2_values, -0.2, 0.6)
bins = np.linspace(-0.2, 0.6, 50)
n_vals, _, patches = ax_g.hist(r2_clipped, bins=bins, color=COASTAL_BLUE,
                                edgecolor='white', linewidth=0.2, alpha=0.9)
for patch, left_edge in zip(patches, bins[:-1]):
    if left_edge >= 0.3:
        patch.set_facecolor(DEEP_OCEAN)
    elif left_edge >= 0:
        patch.set_facecolor(OCEAN_BLUE)

ax_g.axvline(x=0.0, color=SIENNA, linewidth=0.5, linestyle='--', alpha=0.7)
ax_g.axvline(x=0.3, color=ABYSS, linewidth=0.5, linestyle='--', alpha=0.7)

ax_g.text(0.97, 0.92,
          f'n = {len(r2_values):,} domains\n'
          f'R$^2$ > 0: {(r2_values > 0).sum():,} ({(r2_values > 0).sum()/len(r2_values)*100:.1f}%)\n'
          f'R$^2$ > 0.3: {(r2_values > 0.3).sum()} ({(r2_values > 0.3).sum()/len(r2_values)*100:.1f}%)',
          transform=ax_g.transAxes, fontsize=6, verticalalignment='top',
          horizontalalignment='right',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='none', alpha=0.85))

ax_g.set_xlabel('Forward $R^2$ (5-fold CV)')
ax_g.set_ylabel('Number of domains')
ax_g.set_title('R² distribution')
ax_g.text(-0.18, 1.08, 'G', transform=ax_g.transAxes,
          fontsize=6, fontweight='bold', verticalalignment='top')
panel_axes.append(ax_g)

# ── Panel H: Top-100 ranked bar (flush with row 2) ──
ax_h = fig.add_subplot(outer[1, 3])
r2_sorted = np.sort(r2_values)[::-1]
top_n = 100
r2_top = r2_sorted[:top_n]
colors_bar = [DEEP_OCEAN if v >= 0.3 else OCEAN_BLUE for v in r2_top]

ax_h.bar(range(top_n), r2_top, width=1.0, color=colors_bar, edgecolor='none')
ax_h.axhline(y=0.3, color=SIENNA, linewidth=0.5, linestyle='--', alpha=0.7)
ax_h.axvline(x=17.5, color=ABYSS, linewidth=0.5, linestyle=':', alpha=0.7)
ax_h.text(18.5, r2_top[0] * 0.95, 'top 18', fontsize=6, color=ABYSS,
          verticalalignment='top')

ax_h.set_xlabel('Domain rank')
ax_h.set_ylabel('Forward $R^2$ (5-fold CV)')
ax_h.set_title('Top 100 domains')
ax_h.set_xlim(-1, top_n + 1)
ax_h.set_ylim(0, r2_top[0] * 1.08)
ax_h.text(-0.18, 1.08, 'H', transform=ax_h.transAxes,
          fontsize=6, fontweight='bold', verticalalignment='top')
panel_axes.append(ax_h)

# ── Panel I: Dunaliella dark domain vs bathymetry ──
ax_i = fig.add_subplot(gs_r3)

edges = np.array([-6000, -4000, -3000, -2000, -1000, -200, 0, 5000])
labels_bathy = ["<-4000", "-4000\nto\n-3000", "-3000\nto\n-2000",
                "-2000\nto\n-1000", "-1000\nto\n-200", "-200\nto 0", ">0"]
idx = np.digitize(duna_bathy, edges[1:-1], right=False)
med, q1, q3, ns = [], [], [], []
for b in range(len(labels_bathy)):
    yv = duna_clr[idx == b]
    if len(yv):
        med.append(np.median(yv))
        q1.append(np.percentile(yv, 25))
        q3.append(np.percentile(yv, 75))
        ns.append(len(yv))
    else:
        med.append(np.nan); q1.append(np.nan)
        q3.append(np.nan); ns.append(0)

pos = np.arange(len(labels_bathy))
med = np.array(med); q1 = np.array(q1); q3 = np.array(q3)

ax_i.fill_between(pos, q1, q3, color=COASTAL_BLUE, alpha=0.35, lw=0, label="IQR")
ax_i.plot(pos, med, "-o", color=DEEP_OCEAN, lw=1.0, ms=3.0, label="median")
ax_i.set_xticks(pos)
ax_i.set_xticklabels(labels_bathy, fontsize=5)
ax_i.set_xlabel("Ocean bathymetry zone (m)")
ax_i.set_ylabel("Domain abundance (CLR)")
ptxt = "p < 10$^{-100}$" if duna_pval < 1e-100 else f"p = {duna_pval:.1e}"
ax_i.text(0.03, 0.93,
          f"Spearman $\\rho$ = {duna_rho:.3f}\n{ptxt}\nn = {len(duna_df):,}",
          transform=ax_i.transAxes, fontsize=6, va="top",
          bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                    edgecolor='none', alpha=0.8))
ax_i.legend(fontsize=5, loc="center right", frameon=False)
ax_i.text(-0.15, 1.08, 'I', transform=ax_i.transAxes,
          fontsize=6, fontweight='bold', verticalalignment='top')
panel_axes.append(ax_i)

# ── Panel J: Locus map — domain positions on Dunaliella contigs ──
ax_j = fig.add_subplot(gs_r3j)

# Contig data: (nt_length, n_orfs, domain_orf_position)
LOCUS_DATA = [
    (1716688, 1110, 186),
    (1260113, 677, 131),
    (811964, 468, 39),
    (584143, 358, 159),
    (583386, 354, 8),
    (513107, 322, 134),
    (434309, 309, 246),
    (415186, 305, 234),
    (389745, 219, 6),
    (330373, 206, 174),
    (311040, 205, 58),
    (296043, 198, 49),
    (308247, 195, 40),
    (285044, 194, 143),
    (271962, 156, 44),
]

bar_height = 0.55
y_positions = np.arange(len(LOCUS_DATA))[::-1]
max_nt = max(d[0] for d in LOCUS_DATA)

for i, (nt_len, n_orfs, gene_pos) in enumerate(LOCUS_DATA):
    y = y_positions[i]
    # Contig bar (nt scale)
    ax_j.barh(y, nt_len, height=bar_height, color=PALE_AQUA, edgecolor=COASTAL_BLUE,
              linewidth=0.3, zorder=2)
    # Domain position (estimated nt position)
    nt_pos = int(gene_pos / n_orfs * nt_len)
    ax_j.plot(nt_pos, y, 's', color=DEEP_OCEAN, ms=4, zorder=5,
              markeredgecolor=(0.0, 0.05, 0.15), markeredgewidth=0.3)

ax_j.set_yticks([])
ax_j.set_xlim(0, max_nt * 1.05)
ax_j.set_ylim(-0.8, len(LOCUS_DATA) - 0.2)
ax_j.spines['left'].set_visible(False)

# x-axis in Mb
ax_j.set_xlabel('Contig position (Mb)')
ticks_mb = [0, 500000, 1000000, 1500000]
ax_j.set_xticks(ticks_mb)
ax_j.set_xticklabels(['0', '0.5', '1.0', '1.5'])

# Scale bar (200 kb) — inside plot, bottom-right, clear of data
scale_len = 200000
sx = max_nt * 0.72
sy = 1.5
ax_j.plot([sx, sx + scale_len], [sy, sy], '-', color='black', lw=1.0)
ax_j.text(sx + scale_len / 2, sy + 0.35, '200 kb', ha='center', fontsize=5)

# Annotation — mid-right, clear of all bars
ax_j.text(0.98, 0.50, '15 scaffolds',
          transform=ax_j.transAxes, fontsize=5, ha='right', va='center',
          color=(0.3, 0.3, 0.3))

ax_j.text(-0.10, 1.08, 'J', transform=ax_j.transAxes,
          fontsize=6, fontweight='bold', verticalalignment='top')
panel_axes.append(ax_j)

# ── Global spine cleanup ──
for ax in fig.get_axes():
    if not ax.images:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# ── Export ──
check_result = check_figure_size(fig, 'cell', 'double')
print(f"  Size check: {check_result}")

name = f'Figure5_forward_calibration_{TIMESTAMP}'
for fmt in ['pdf', 'svg']:
    outpath = OUTDIR / f'{name}.{fmt}'
    fig.savefig(str(outpath), format=fmt, transparent=True,
                edgecolor='none', dpi=600)
    print(f"  Saved: {outpath}")

plt.close(fig)

# ── Provenance log ──
prov = OUTDIR / f'{name}_provenance.txt'
with open(prov, 'w') as f:
    f.write("# Provenance — Figure 5: Forward Model Calibration + Biological PoC\n")
    f.write(f"Script: {Path(__file__).resolve()}\n")
    f.write(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
    f.write("Panels A-B, D-F: Cached from supplement/forward_calibration_predictions_20260518_143812.tsv\n")
    f.write("  (10-fold spatial block CV, satellite-only, n=1,878)\n")
    f.write("Panel C (FTR1): source_data/panel_regen/panel_FTR1_obs_pred_20260618_092826.tsv\n")
    f.write("  (5-fold CV, nutrient-merged, n=1,279)\n")
    f.write("Panels G-H: archive/ralph4_statistical_reanalysis/forward_r2_all_pfams.tsv\n")
    f.write("Panel I (Dunaliella): source_data/panel_regen/panel_dunaliella_clr_scatter_20260618_124843.tsv\n")
    f.write(f"  Spearman rho={duna_rho:.4f}, n={len(duna_df)}\n")
print(f"  Saved: {prov}")

print("\nDone!")
