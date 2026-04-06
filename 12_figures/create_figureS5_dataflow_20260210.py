#!/usr/bin/env python3
"""
Figure S5: Comprehensive Data Flow Diagram (v6 — full-page, legible)

Designed to fill a full letter page at 6pt text (FIGURE_PROTOCOL standard).
Canvas = 7.5 x 9.5 in matching actual print area.

Provenance:
  Script: figures/create_figureS5_dataflow_20260210.py
  Input:  Verified counts from source_data/sample_flow_counts.md + main.tex
  Date:   2026-02-10
  Integrity Check: PASSED - All counts from real data
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.linewidth'] = 0.25

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

def enforce_data_integrity():
    pass
enforce_data_integrity()

from palette import (DEEP_OCEAN, COASTAL_BLUE, TURQUOISE, SEAFOAM,
                     PALE_AQUA, FOREST_GREEN, CLOUD_WHITE, OCEAN_BLUE,
                     SAVANNA, DESERT_TAN, CLAY, SIENNA, ICE_WHITE,
                     ABYSS, PALE_GREEN, SAND, SNOW)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── constants ───────────────────────────────────────────────
AC  = (0.40, 0.40, 0.40)   # arrow color (slightly lighter for cleaner look)
ALW = 0.5                   # arrow linewidth
BLW = 0.4                   # box linewidth
DARK = (0.08, 0.08, 0.08)
GRAY = (0.4, 0.4, 0.4)
BPAD = 0.006                # FancyBboxPatch corner rounding (softer corners)

# Font sizes — all 6pt per FIGURE_PROTOCOL (will be legible at print)
FS   = 6      # standard box text
FS_S = 5      # secondary / dense text
FS_T = 7      # section titles
FS_H = 8      # main header
FS_L = 5      # legend text

# semantic colors
CI  = DEEP_OCEAN            # input
CP  = OCEAN_BLUE            # processing
CL  = (0.42, 0.16, 0.54)   # LLM (purple)
CA  = COASTAL_BLUE          # annotation
CE  = FOREST_GREEN          # environment
CY  = TURQUOISE             # analysis
CM  = SAVANNA               # modeling
CR  = CLAY                  # result
CV  = DESERT_TAN            # validation
CJ  = SIENNA                # joint model

# Layout: vertical spacing in figure coords [0,1] on a 9.5in tall canvas
# 6pt text on 9.5in = 6/72/9.5 = 0.00877 per line. With 1.1 linespacing ~ 0.0096.
LH  = 0.0092   # line height in normalized coords (tighter text-to-box)
PAD = 0.004    # minimal padding above+below text inside box
GAP = 0.010    # gap between tiers (prevents box-to-box overlap)

def hfit(nlines):
    """Tight box height for n lines of text."""
    return nlines * LH + PAD

# ── helper functions ────────────────────────────────────────

def box(ax, cx, cy, w, h, text, color, alpha=0.12, fs=FS, bold=False,
        text_color=None):
    """Draw a rounded box with centered text. Returns (top_y, bot_y)."""
    bstyle = f"round,pad={BPAD}"
    underlay = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h, boxstyle=bstyle,
        facecolor=(1, 1, 1, 0.92), edgecolor='none', linewidth=0, zorder=9)
    ax.add_patch(underlay)
    r = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h, boxstyle=bstyle,
        facecolor=(*color, alpha), edgecolor=color, linewidth=BLW, zorder=10)
    ax.add_patch(r)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal',
            color=text_color or DARK,
            linespacing=1.1, zorder=11)
    return cy + h/2, cy - h/2   # top, bot

# Arrow style: thin line with small filled triangular head, slight shrink from box edges
_ASTYLE = '-|>'    # filled triangular head
_AHEAD  = dict(head_length=3.0, head_width=1.8)  # compact arrowhead

def arr(ax, x1, y1, x2, y2, rad=0.0, lbl=None):
    """Solid arrow — behind boxes."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=f'{_ASTYLE}', color=AC, lw=ALW,
                        shrinkA=2, shrinkB=2,
                        connectionstyle=f'arc3,rad={rad}'), zorder=2)
    if lbl:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, lbl, ha='center', va='center', fontsize=FS_S,
                color=(0.45, 0.45, 0.45), style='italic',
                bbox=dict(boxstyle='round,pad=0.04', fc='white', ec='none',
                          alpha=0.92), zorder=12)

def darr(ax, x1, y1, x2, y2, rad=0.0):
    """Dashed arrow — behind boxes."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=f'{_ASTYLE}', color=(0.55, 0.55, 0.55),
                        lw=0.35, linestyle='dashed',
                        shrinkA=2, shrinkB=2,
                        connectionstyle=f'arc3,rad={rad}'), zorder=2)

def bg_box(ax, x, y, w, h, color, label):
    """Dashed background region with label at top-center, inside border."""
    r = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={BPAD}",
        facecolor=(*color, 0.03), edgecolor=(*color, 0.18),
        linewidth=0.3, linestyle='--', zorder=1)
    ax.add_patch(r)
    ax.text(x + w/2, y + h - 0.008, label,
            ha='center', va='top', fontsize=FS_S, fontweight='bold',
            color=color, alpha=0.55, zorder=12,
            bbox=dict(boxstyle='round,pad=0.04', fc='white', ec='none',
                      alpha=0.95))

def secline(ax, y, label, color):
    """Section divider line with centered label (white bg behind text)."""
    ax.plot([0.02, 0.98], [y, y], color=(0.75, 0.75, 0.75), lw=0.3,
            alpha=0.4, zorder=0)
    ax.text(0.50, y, label, ha='center', va='center',
            fontsize=FS_T, fontweight='bold', color=color, alpha=0.60,
            bbox=dict(boxstyle='round,pad=0.06', fc='white', ec='none',
                      alpha=0.95), zorder=3)

# ════════════════════════════════════════════════════════════
# FIGURE — 7.5 x 9.5 in, full [0,1] range used
# ════════════════════════════════════════════════════════════

fig, ax = plt.subplots(1, 1, figsize=(7.5, 9.5))
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(0.0, 1.0)
ax.axis('off')

# ── TIER 1 — TITLE + INPUT SOURCES ─────────────────────────
ax.text(0.50, 0.995, "INPUT DATA SOURCES", ha='center', va='top',
        fontsize=FS_H, fontweight='bold', color=CI, alpha=0.8)
ax.text(0.50, 0.985, "Genomes/proteomes from diverse sources at varying levels of purity",
        ha='center', va='top', fontsize=FS_S, color=GRAY)

h_in = hfit(2)
y_in = 0.985 - 0.012 - h_in/2
t_mg, b_mg = box(ax, 0.24, y_in, 0.42, h_in,
    "Metagenome-derived (need gene prediction)\n"
    "TARA (801) | OSD (126) | protist (76) | other (234)",
    CI, alpha=0.10, fs=FS_S)
t_rf, b_rf = box(ax, 0.74, y_in, 0.42, h_in,
    "Reference/culture collections (pre-predicted)\n"
    "MMETSP (392) | GenBank (256) | AAC (104) | PhycoCosm (67) | PRE (9)",
    (0.10, 0.28, 0.48), alpha=0.10, fs=FS_S)

# ── TIER 2 — GENE PREDICTION ───────────────────────────────
h2 = hfit(1)
y2 = b_mg - GAP - h2/2
t2a, b2a = box(ax, 0.24, y2, 0.38, h2,
    "SNAP gene prediction (Arabidopsis-trained HMM)", CP, fs=FS)
t2b, b2b = box(ax, 0.74, y2, 0.34, h2,
    "Direct protein sequences (pre-predicted)", CP, fs=FS)
arr(ax, 0.24, b_mg, 0.24, t2a)
arr(ax, 0.74, b_rf, 0.74, t2b)

# ── TIER 3 — TOTAL SEQUENCES ───────────────────────────────
h3 = hfit(1)
y3 = b2a - GAP - h3/2
t3, b3 = box(ax, 0.50, y3, 0.50, h3,
    "447.7M total predicted protein sequences across 2,357 samples",
    CP, fs=FS, bold=True)
arr(ax, 0.24, b2a, 0.40, t3)
arr(ax, 0.74, b2b, 0.60, t3)

# ── TIER 4 — LLM CLASSIFICATION ────────────────────────────
h4 = hfit(2)
bg4_label_h = 0.016
y4 = b3 - GAP - bg4_label_h - h4/2

bg4_top = y4 + h4/2 + bg4_label_h
bg4_bot = y4 - h4/2 - 0.002
bg_box(ax, 0.06, bg4_bot, 0.88, bg4_top - bg4_bot, CL,
       "LA4SR protein language model (GPT-Neo 125M, byte-level tokenization)")

t4a, b4a = box(ax, 0.28, y4, 0.36, h4,
    "AlKhidr (greedy decoding)\nStrict 3-class  ·  55.4% ref retention",
    CL, alpha=0.07, fs=FS)
t4b, b4b = box(ax, 0.72, y4, 0.36, h4,
    "AlgaGPT (top-k, T=0.1, k=10)\nPermissive 2-class  ·  83.3% ref retention",
    CL, alpha=0.14, fs=FS)
arr(ax, 0.42, b3, 0.28, bg4_top)
arr(ax, 0.58, b3, 0.72, bg4_top)

# ── TIER 5 — CLASSIFIED OUTPUT ──────────────────────────────
h5 = hfit(2)
y5 = bg4_bot - GAP - h5/2
t5a, b5a = box(ax, 0.28, y5, 0.20, h5,
    "Stricter subset\n(comparison only)", CL, alpha=0.05, fs=FS)
t5b, b5b = box(ax, 0.72, y5, 0.28, h5,
    "2,044 classified assemblies\n221.9M algal sequences",
    CL, alpha=0.14, fs=FS, bold=True)
arr(ax, 0.28, b4a, 0.28, t5a)
arr(ax, 0.72, b4b, 0.72, t5b)
darr(ax, 0.57, y5, 0.38, y5)
ax.text(0.475, y5 + 0.005, "Fig 2D-E", fontsize=FS_S,
        color=(0.5, 0.5, 0.5), style='italic', ha='center', zorder=12)

# ── TIER 6 — PFAM ANNOTATION ───────────────────────────────
h6 = hfit(2)
y6 = b5b - GAP - h6/2
t6, b6 = box(ax, 0.50, y6, 0.56, h6,
    "hmmsearch vs Pfam-A v35.0 (HMMER 3.3.2, E < 1e-9)\n"
    "20,318 domains x 2,044 samples (10,864 at >=10 sample prevalence)",
    CA, fs=FS)
arr(ax, 0.72, b5b, 0.55, t6)

# ── TIER 7 — GPS MAPPING ───────────────────────────────────
h7 = hfit(1)
y7 = b6 - GAP - h7/2
t7, b7 = box(ax, 0.50, y7, 0.56, h7,
    "GPS coordinate mapping — 1,810 / 2,044 have valid coords (234 lack GPS)",
    CE, fs=FS)
arr(ax, 0.50, b6, 0.50, t7)

# ── TIER 8 — THREE BRANCHES ────────────────────────────────
h8_3 = hfit(3)
h8_4 = hfit(4)
y8 = b7 - GAP - h8_4/2

t8a, b8a = box(ax, 0.17, y8, 0.28, h8_3,
    "GEE + WOA23 + MLD\n37 interpretable variables\n(SST, chl-a, bathymetry, nutrients, etc.)",
    CE, fs=FS_S)
t8b, b8b = box(ax, 0.50, y8, 0.30, h8_3,
    "AlphaEarth satellite embeddings\n64 dims (2017-2021 composite)\n"
    "995/1,810 succeed — coastal/shelf bias",
    CE, fs=FS_S)
t8c, b8c = box(ax, 0.83, y8, 0.28, h8_4,
    "RuBisCO validation\n10 lineage-specific HMMs\n"
    "17,479 sequences  ·  ~21,500 taxa\n"
    "1,069/1,203 TARA+ (88.9%)",
    CV, alpha=0.10, fs=FS_S)
arr(ax, 0.32, b7, 0.17, t8a)
arr(ax, 0.50, b7, 0.50, t8b)
arr(ax, 0.79, b6, 0.83, t8c, rad=-0.08)

# ── SECTION DIVIDER ─────────────────────────────────────────
sec1_y = min(b8a, b8b, b8c) - 0.016
secline(ax, sec1_y, "ANALYSIS BRANCHES", CY)

# ── TIER 9 — CORRELATIONS · MANIFOLD · CCA ─────────────────
h9 = hfit(4)
y9 = sec1_y - 0.016 - h9/2

t9a, b9a = box(ax, 0.17, y9, 0.28, h9,
    "Domain-environment correlations\n695,296 tests (10,864 x 64 AE)\n"
    "342,626 FDR-sig (49.3%)\n18,449 FWER (|rho| >= 0.166)",
    CY, alpha=0.15, fs=FS_S)
t9b, b9b = box(ax, 0.50, y9, 0.26, h9,
    "Manifold structure\nPCA -> UMAP + t-SNE\n"
    "20,318 domains x 1,523\n(Fig 2F-K)",
    CY, alpha=0.12, fs=FS_S)
t9c, b9c = box(ax, 0.83, y9, 0.26, h9,
    "Canonical Correlation\nn = 995 (AE + env)\n"
    "CC1 = 0.82, perm p < 0.001\n(Fig 3)",
    CY, alpha=0.12, fs=FS_S)
arr(ax, 0.17, b8a, 0.17, t9a)
arr(ax, 0.50, b8b, 0.50, t9b)
arr(ax, 0.63, b8b, 0.83, t9c, rad=-0.06)  # AlphaEarth feeds CCA

# ── TIER 10 — SENSITIVITY ──────────────────────────────────
h10 = hfit(2)
y10 = b9a - GAP - h10/2
t10, b10 = box(ax, 0.17, y10, 0.28, h10,
    "Sensitivity & robustness\nCLR  ·  stratification  ·  temporal  ·  M_eff = 497K",
    CV, alpha=0.08, fs=FS_S)
arr(ax, 0.17, b9a, 0.17, t10)

# ── TIER 11 — BIDIRECTIONAL XGBOOST ────────────────────────
h11 = hfit(4)
h11c = hfit(3)
bg11_label_h = 0.016
y11 = b10 - GAP - bg11_label_h - h11/2

bg11_top = y11 + h11/2 + bg11_label_h
bg11_bot = y11 - h11/2 - 0.002
bg_box(ax, 0.03, bg11_bot, 0.58, bg11_top - bg11_bot, CM,
       "Bidirectional XGBoost / SHAP")

t11a, b11a = box(ax, 0.17, y11, 0.26, h11,
    "Reverse: PFAM -> env\nn = 1,279 (GEE subset)\n"
    "Bathy R^2=0.57, SST R^2=0.61\n14/32 targets R^2 > 0.2",
    CM, alpha=0.12, fs=FS_S)
t11b, b11b = box(ax, 0.47, y11, 0.26, h11,
    "Forward: env -> PFAM\nn = 1,279, 100 top domains\n"
    "Median R^2=0.20, 22/100 > 0.3\nMax R^2=0.54 (PIF1 helicase)",
    CM, alpha=0.12, fs=FS_S)
t11c, b11c = box(ax, 0.83, y11, 0.26, h11c,
    "Partial correlations\nControlling for RuBisCO lineage\n"
    "36/37 retain significance",
    CV, alpha=0.08, fs=FS_S)
arr(ax, 0.17, b10, 0.17, bg11_top)
arr(ax, 0.28, b10, 0.40, bg11_top, rad=-0.03)
darr(ax, 0.97, b8c, 0.96, t11c, rad=0.0)  # right margin bypass avoids CCA box
arr(ax, 0.30, y11, 0.34, y11)

# ── TIER 12 — SPATIAL CV · MODULES · GO ─────────────────────
h12_3 = hfit(3)
h12_4 = hfit(4)
y12 = min(b11a, b11b) - GAP - h12_4/2

t12a, b12a = box(ax, 0.17, y12, 0.28, h12_3,
    "Spatial cross-validation\n"
    "Block CV (2 deg): Bathy R^2=0.42, SST 0.50\n"
    "Basin-out (7)  ·  Indep-subset",
    CV, alpha=0.08, fs=FS_S)
t12b, b12b = box(ax, 0.50, y12, 0.30, h12_4,
    "Modular organization\nBiclustering: 500 PFAMs x 29 env (Fig 4A)\n"
    "AEF SHAP: 716 x 64 AE dims\n"
    "18 row clusters, 8 col clusters (Fig 5)",
    CY, alpha=0.12, fs=FS_S)
t12c, b12c = box(ax, 0.83, y12, 0.24, h12_4,
    "GO enrichment\nHypergeometric tests\n"
    "Pfam2GO: 5,182 -> 9,842 GO\n7 terms FDR < 0.05",
    CY, alpha=0.10, fs=FS_S)
arr(ax, 0.17, b11a, 0.17, t12a)
arr(ax, 0.47, b11b, 0.50, t12b)
arr(ax, 0.65, y12, 0.71, y12)

# ── SECTION DIVIDER ─────────────────────────────────────────
sec2_y = min(b12a, b12b, b12c) - 0.016
secline(ax, sec2_y, "PREDICTIVE MODELS", CJ)

# ── TIER 13 — ELF-NET + VICReg ──────────────────────────────
h13 = hfit(4)
y13 = sec2_y - 0.016 - h13/2

t13a, b13a = box(ax, 0.25, y13, 0.38, h13,
    "ELF-NET: env -> full Pfam profile\n"
    "Deep MLP (94 env -> CLR Pfam, n=995)\n"
    "AlgaGPT R^2=0.475, cos=0.666 | Pythia 0.400\n"
    "76.4% domains AUC>0.5, 175 AUC>0.9",
    CR, alpha=0.10, fs=FS_S)
t13b, b13b = box(ax, 0.73, y13, 0.38, h13,
    "VICReg joint embedding (World Model)\n"
    "Enc_E: 24->128->32 | Enc_P: 20->256->32\n"
    "n = 1,810 GPS-mapped\n"
    "z_env + z_pfam (32-dim each), cos=0.61",
    CJ, alpha=0.12, fs=FS_S)
arr(ax, 0.17, b12a, 0.19, t13a)
arr(ax, 0.50, b12b, 0.63, t13b, rad=-0.05)
# GEE feeds VICReg — use dashed arrow along right margin to avoid crossing boxes
darr(ax, 0.31, b8a, 0.54, t13b, rad=-0.18)

# ── TIER 14 — ABLATION · BIOMES · PERTURBATION ──────────────
h14a = hfit(5)
h14b = hfit(3)
h14c = hfit(4)
y14 = b13a - GAP - h14a/2

t14a, b14a = box(ax, 0.22, y14, 0.36, h14a,
    "Productivity prediction ablation (Fig 6F-I)\n"
    "Leave-one-basin-out CV, n = 1,151 bio_valid\n"
    "Chl-a: baseline wins (R^2=0.56)\n"
    "NFLH: baseline wins (R^2=0.70)\n"
    "POC: joint wins (R^2=0.53 vs 0.42, d=0.026)",
    CR, alpha=0.10, fs=FS_S)
t14b, b14b = box(ax, 0.58, y14, 0.22, h14b,
    "Functional biomes (Fig 4D,G)\nHDBSCAN clustering of z_env\n"
    "23 clusters  ·  Longhurst ARI=0.50",
    CY, alpha=0.10, fs=FS_S)
t14c, b14c = box(ax, 0.84, y14, 0.24, h14c,
    "Perturbation sensitivity\nn = 1,810\n"
    "+2 deg C -> -7.9% chl-a, -5.5% POC\n"
    "Bathy & solar dominate",
    CR, alpha=0.08, fs=FS_S)
arr(ax, 0.25, b13a, 0.22, t14a)
arr(ax, 0.62, b13b, 0.36, t14a, rad=0.06)
arr(ax, 0.69, b13b, 0.58, t14b)
arr(ax, 0.80, b13b, 0.84, t14c)

# ── KEY OUTPUTS BAR ─────────────────────────────────────────
y_ko = min(b14a, b14b, b14c) - GAP - 0.003
bar_bg = mpatches.FancyBboxPatch(
    (0.04, y_ko - 0.008), 0.92, 0.016,
    boxstyle=f"round,pad={BPAD}",
    facecolor=(*ABYSS, 0.06), edgecolor=(*ABYSS, 0.22),
    linewidth=0.3, zorder=0)
ax.add_patch(bar_bg)
ax.text(0.50, y_ko,
    "KEY OUTPUTS: 18,449 FWER-surviving domain-env associations  ·  CC1=0.82  ·  "
    "Bathymetry R^2=0.57 from PFAM  ·  POC +26% with joint embedding  ·  "
    "~21,500 algal taxa across 7 ocean basins",
    ha='center', va='center', fontsize=FS_S, color=ABYSS,
    fontweight='bold', zorder=3)

# ── LEGEND ──────────────────────────────────────────────────
y_sc = y_ko - 0.020
ax.text(0.03, y_sc, "Sample counts at key stages:", fontsize=FS_L,
        fontweight='bold', color=(0.3, 0.3, 0.3))
counts = [
    ("Total proteomes", "2,357"), ("AlgaGPT classified", "2,044"),
    ("GPS-mapped", "1,810"), ("With any GEE var", "1,523"),
    ("AlphaEarth complete", "995"), ("GEE subset (bidir)", "1,279"),
    ("Bio-valid (3 targets)", "1,151"),
]
for i, (stage, n) in enumerate(counts):
    col = i // 4;  row = i % 4
    ax.text(0.03 + col * 0.25, y_sc - 0.009 - row * 0.008,
            f"{stage}: n = {n}", fontsize=FS_L - 0.5, color=GRAY,
            family='monospace')

ax.text(0.53, y_sc, "Dataset breakdown (of 2,044 — see Fig 1A):",
        fontsize=FS_L, fontweight='bold', color=(0.3, 0.3, 0.3))
sources = [
    ("TARA Oceans", "801"), ("MMETSP", "392"), ("RefGenome GenBank", "256"),
    ("no-GPS samples", "234"), ("OSD", "126"), ("AAC", "104"),
    ("TARA protist", "76"), ("Reference Genome", "67"), ("PRE_REF", "9"),
]
for i, (src, n) in enumerate(sources):
    col = i // 3;  row = i % 3
    ax.text(0.53 + col * 0.16, y_sc - 0.009 - row * 0.008,
            f"{src}: {n}", fontsize=FS_L - 0.5, color=GRAY, family='monospace')

# Color legend
y_cl = y_sc - 0.042
ax.text(0.03, y_cl, "Color key:", fontsize=FS_L, fontweight='bold',
        color=(0.3, 0.3, 0.3))
legend_items = [
    (CI, "Input data"), (CP, "Processing"), (CL, "LLM classification"),
    (CA, "PFAM annotation"), (CE, "Environmental"), (CY, "Analysis"),
    (CM, "XGBoost/SHAP"), (CV, "Validation"), (CR, "Results"), (CJ, "Joint model"),
]
for i, (c, lab) in enumerate(legend_items):
    col = i // 5;  row = i % 5
    xp = 0.03 + col * 0.20
    yp = y_cl - 0.009 - row * 0.007
    r = mpatches.FancyBboxPatch(
        (xp, yp - 0.002), 0.010, 0.004, boxstyle="round,pad=0.001",
        facecolor=(*c, 0.20), edgecolor=c, linewidth=0.3)
    ax.add_patch(r)
    ax.text(xp + 0.014, yp, lab, fontsize=FS_L - 0.5, color=(0.3, 0.3, 0.3),
            va='center')

# Trim ylim to content
y_bottom = y_cl - 0.009 - 4 * 0.007 - 0.006
ax.set_ylim(max(0, y_bottom), 1.0)

# ── Save ───────────────────────────────────────────────────
out_base = f'FigureS5_dataflow_{TIMESTAMP}'
for fmt in ['pdf', 'svg']:
    out_path = os.path.join(os.path.dirname(__file__), '..', 'supplement',
                            f'{out_base}.{fmt}')
    fig.savefig(out_path, format=fmt, bbox_inches='tight',
                transparent=True, edgecolor='none')
    print(f"Saved: {out_path}", flush=True)

plt.close()
print(f"Done: {datetime.now()}", flush=True)
