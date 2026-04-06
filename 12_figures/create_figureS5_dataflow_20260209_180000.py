#!/usr/bin/env python3
"""
Figure S5: Comprehensive Data Flow Diagram (v3)

Shows the complete project pipeline from diverse input genomes through
LLM classification (both modes), PFAM annotation, environmental
characterization, and all major analysis branches.

Provenance:
  Script: figures/create_figureS5_dataflow_20260209_180000.py
  Input: Verified counts from source_data/sample_flow_counts.md + main.tex
  Date: 2026-02-09
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
AC = (0.35, 0.35, 0.35)  # arrow color
ALW = 0.4                 # arrow linewidth
BLW = 0.5                 # box linewidth
T4 = 4.0; T45 = 4.5; T5 = 5.0; T55 = 5.5  # font sizes
DARK = (0.08, 0.08, 0.08)
GRAY = (0.4, 0.4, 0.4)

# semantic colors
CI = DEEP_OCEAN           # input
CP = OCEAN_BLUE           # processing
CL = (0.42, 0.16, 0.54)  # LLM (purple)
CA = COASTAL_BLUE         # annotation
CE = FOREST_GREEN         # environment
CY = TURQUOISE            # analysis
CM = SAVANNA              # modeling
CR = CLAY                 # result
CV = DESERT_TAN           # validation
CJ = SIENNA               # joint model

def box(ax, cx, cy, w, h, text, color, alpha=0.12, fs=T45, bold=False):
    r = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h, boxstyle="round,pad=0.005",
        facecolor=(*color, alpha), edgecolor=color, linewidth=BLW, zorder=2)
    ax.add_patch(r)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color=DARK,
            linespacing=1.15, zorder=3)
    return cy + h/2, cy - h/2  # top, bot

def arr(ax, x1, y1, x2, y2, rad=0.0, lbl=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=AC, lw=ALW,
                       connectionstyle=f'arc3,rad={rad}'), zorder=1)
    if lbl:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, lbl, ha='center', va='center', fontsize=T4,
                color=(0.45, 0.45, 0.45), style='italic',
                bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='none',
                         alpha=0.85), zorder=4)

def darr(ax, x1, y1, x2, y2, rad=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=(0.6, 0.6, 0.6), lw=0.3,
                       linestyle='dashed', connectionstyle=f'arc3,rad={rad}'),
        zorder=1)

def secline(ax, y, label, color):
    ax.plot([0.02, 0.98], [y, y], color=(0.75, 0.75, 0.75), lw=0.3,
            alpha=0.4, zorder=0)
    ax.text(0.50, y + 0.006, label, ha='center', va='bottom',
            fontsize=T55, fontweight='bold', color=color, alpha=0.60)

def bg_box(ax, x, y, w, h, color, label):
    """Dashed background with label INSIDE at top, not overlapping child boxes."""
    r = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004",
        facecolor=(*color, 0.03), edgecolor=(*color, 0.18),
        linewidth=0.3, linestyle='--', zorder=0)
    ax.add_patch(r)
    # Place label just inside the top of the background box
    ax.text(x + w/2, y + h - 0.003, label,
            ha='center', va='top', fontsize=T4, fontweight='bold',
            color=color, alpha=0.60)

# ════════════════════════════════════════════════════════════
# LAYOUT — use full page, ~7×10.5, ylim [0,1]
# Each row gets explicit y-center with guaranteed gap
# ════════════════════════════════════════════════════════════

fig, ax = plt.subplots(1, 1, figsize=(7.0, 10.5))
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.185, 1.0)
ax.axis('off')

# ── R0: Section title ──────────────────────────────────────
ax.text(0.50, 0.993, "INPUT DATA SOURCES", ha='center', va='top',
        fontsize=T55, fontweight='bold', color=CI, alpha=0.8)
ax.text(0.50, 0.984, "Genomes/proteomes from diverse sources at varying levels of purity",
        ha='center', va='top', fontsize=T4, color=GRAY)

# ── R1: Input sources — TWO ROWS matching Fig 1 legend ─────
# Row 1: metagenome / field projects (need SNAP gene prediction)
y1a = 0.964; h1 = 0.020; bw1 = 0.215
CI2 = (0.10, 0.28, 0.48)  # slightly lighter input color for row 2

r1_meta = [  # metagenome-origin projects → SNAP
    (0.12, "TARA Oceans\nmetagenomes (801)", CI),
    (0.36, "OSD\nOcean Sampling Day (126)", CI),
    (0.62, "TARA protist\nprotist fraction (76)", CI),
    (0.88, "Other metagenomes\n+ no-GPS (234)", CI),
]
bots_meta = []
for xi, li, c in r1_meta:
    _, b = box(ax, xi, y1a, bw1, h1, li, c, fs=T4)
    bots_meta.append(b)

# Row 2: reference/culture collections (pre-predicted proteins)
y1b = 0.936; bw1b = 0.185
r1_ref = [
    (0.10, "MMETSP\ntranscriptomes (392)", CI2),
    (0.30, "RefGenome GenBank\nNCBI refs (256)", CI2),
    (0.50, "AAC\nalgae cultures (104)", CI2),
    (0.70, "Reference Genome\nPhycoCosm etc. (67)", CI2),
    (0.90, "RefGenome PRE_REF\npre-release (9)", CI2),
]
bots_ref = []
for xi, li, c in r1_ref:
    _, b = box(ax, xi, y1b, bw1b, h1, li, c, fs=T4)
    bots_ref.append(b)

# ── R2: Gene prediction (y=0.908) ──────────────────────────
y2 = 0.908; h2 = 0.016
t2a, b2a = box(ax, 0.24, y2, 0.34, h2,
    "SNAP gene prediction (Arabidopsis-trained HMM)", CP, fs=T45)
t2b, b2b = box(ax, 0.74, y2, 0.32, h2,
    "Direct protein sequences (pre-predicted)", CP, fs=T45)

# Metagenome row → SNAP
for bm, xi in zip(bots_meta, [0.12, 0.36, 0.62, 0.88]):
    arr(ax, xi, bm, 0.24, t2a)
# Reference row → direct
for br, xi in zip(bots_ref, [0.10, 0.30, 0.50, 0.70, 0.90]):
    arr(ax, xi, br, 0.74, t2b)

# ── R3: Total sequences (y=0.884) ──────────────────────────
y3 = 0.884; h3 = 0.016
t3, b3 = box(ax, 0.50, y3, 0.48, h3,
    "447.7M total predicted protein sequences across 2,357 samples",
    CP, fs=T45, bold=True)
arr(ax, 0.24, b2a, 0.40, t3)
arr(ax, 0.74, b2b, 0.60, t3)

# ── R4: LLM section (y=0.848) ──────────────────────────────
y4 = 0.848; h4 = 0.024

# Background: extend upward to create a label band above child boxes
bg_box(ax, 0.04, y4 - h4/2 - 0.003, 0.92, h4 + 0.016, CL,
       "LA4SR protein language model (GPT-Neo 125M, byte-level tokenization)")

t4a, b4a = box(ax, 0.27, y4, 0.38, h4,
    "AlKhidr (greedy decoding)\nStrict 3-class · 55.4% ref retention",
    CL, alpha=0.07, fs=T45)
t4b, b4b = box(ax, 0.73, y4, 0.38, h4,
    "AlgaGPT (top-k, T=0.1, k=10)\nPermissive 2-class · 83.3% ref retention",
    CL, alpha=0.14, fs=T45)

arr(ax, 0.42, b3, 0.27, t4a)
arr(ax, 0.58, b3, 0.73, t4b)

# ── R5: Classified output (y=0.812) ────────────────────────
y5 = 0.812; h5 = 0.020
t5a, b5a = box(ax, 0.27, y5, 0.24, h5,
    "Stricter subset\n(comparison only)", CL, alpha=0.05, fs=T45)
t5b, b5b = box(ax, 0.73, y5, 0.30, h5,
    "2,044 classified assemblies\n221.9M algal sequences",
    CL, alpha=0.14, fs=T45, bold=True)

arr(ax, 0.27, b4a, 0.27, t5a)
arr(ax, 0.73, b4b, 0.73, t5b)
darr(ax, 0.58, y5, 0.39, y5)
ax.text(0.49, y5 + 0.005, "Fig 2D-E", fontsize=3.5,
        color=(0.5, 0.5, 0.5), style='italic', ha='center')

# ── R6: PFAM annotation (y=0.778) ──────────────────────────
y6 = 0.778; h6 = 0.022
t6, b6 = box(ax, 0.50, y6, 0.56, h6,
    "hmmsearch vs Pfam-A v35.0 (HMMER 3.3.2, E < 1e-9)\n"
    "20,318 domains × 2,044 samples (10,864 at ≥10 sample prevalence)",
    CA, fs=T45)
arr(ax, 0.73, b5b, 0.55, t6)

# ── R7: GPS mapping (y=0.745) ──────────────────────────────
y7 = 0.745; h7 = 0.018
t7, b7 = box(ax, 0.25, y7, 0.38, h7,
    "GPS coordinate mapping — 1,810/2,044 have valid coords (234 lack GPS)",
    CE, fs=T45)
arr(ax, 0.38, b6, 0.25, t7)

# ── R8: Env branches + RuBisCO (y=0.708) ───────────────────
y8 = 0.708; h8 = 0.026

t8a, b8a = box(ax, 0.15, y8, 0.26, h8,
    "GEE + WOA23 + MLD\n37 interpretable variables\n(SST, chl-a, bathymetry, nutrients, Rrs, etc.)",
    CE, fs=4.3)
t8b, b8b = box(ax, 0.46, y8, 0.30, h8,
    "AlphaEarth satellite embeddings\n64 dims (2017-2021 composite)\n"
    "995/1,810 succeed — coastal/shelf bias",
    CE, fs=4.3)
t8c, b8c = box(ax, 0.82, y8, 0.30, h8 + 0.008,
    "RuBisCO validation\n10 lineage-specific HMMs\n"
    "17,479 sequences · ~21,500 taxa\n"
    "1,069/1,203 TARA+ (88.9%)",
    CV, alpha=0.10, fs=4.3)

arr(ax, 0.16, b7, 0.15, t8a)
arr(ax, 0.34, b7, 0.40, t8b, rad=-0.06)
arr(ax, 0.73, b5b, 0.82, t8c, rad=0.15)

# ── DIVIDER: ANALYSIS BRANCHES (y=0.664) ───────────────────
secline(ax, 0.664, "ANALYSIS BRANCHES", CY)

# ── R9: Correlations · Manifold · CCA (y=0.648) ────────────
y9 = 0.648; h9 = 0.032

t9a, b9a = box(ax, 0.15, y9, 0.26, h9,
    "Domain-env correlations\n695,296 tests (10,864 × 64 AE)\n"
    "342,626 FDR-sig (49.3%)\n18,449 FWER (|ρ| ≥ 0.166)",
    CY, alpha=0.15, fs=4.3)
t9b, b9b = box(ax, 0.44, y9, 0.22, h9,
    "Manifold structure\nPCA → UMAP + t-SNE\n"
    "20,318 domains × 1,523\n(Fig 2F-K)",
    CY, alpha=0.12, fs=4.3)
t9c, b9c = box(ax, 0.72, y9, 0.24, h9,
    "Canonical Correlation\nn = 995 (AE + env)\n"
    "CC1 = 0.82, perm p < 0.001\n(Fig 3)",
    CY, alpha=0.12, fs=4.3)

arr(ax, 0.15, b8a, 0.15, t9a)
arr(ax, 0.42, b6, 0.15, t9a, rad=0.10)
arr(ax, 0.28, b8a, 0.40, t9b, rad=-0.06)
arr(ax, 0.48, b6, 0.44, t9b)
arr(ax, 0.52, b8b, 0.68, t9c, rad=-0.06)
arr(ax, 0.56, b6, 0.72, t9c, rad=-0.10)

# ── R10: Sensitivity (y=0.608) ─────────────────────────────
y10 = 0.608; h10 = 0.020
t10, b10 = box(ax, 0.15, y10, 0.26, h10,
    "Sensitivity & robustness\nCLR · stratification · temporal · M_eff=497K",
    CV, alpha=0.08, fs=T4)
arr(ax, 0.15, b9a, 0.15, t10)

# ── R11: Bidirectional + Partial (y=0.568) ─────────────────
y11 = 0.568; h11 = 0.034

# Background: extend upward to create a label band above child boxes
bg_box(ax, 0.03, y11 - h11/2 - 0.002, 0.56, h11 + 0.014, CM,
       "Bidirectional XGBoost / SHAP")

t11a, b11a = box(ax, 0.16, y11, 0.24, h11,
    "Reverse: PFAM → env\nn = 1,279 (GEE subset)\n"
    "Bathy R²=0.57, SST R²=0.61\n14/32 targets R² > 0.2",
    CM, alpha=0.12, fs=T4)
t11b, b11b = box(ax, 0.44, y11, 0.24, h11,
    "Forward: env → PFAM\nn = 1,279, 100 top domains\n"
    "Median R²=0.20, 22/100 > 0.3\nMax R²=0.54 (PIF1 helicase)",
    CM, alpha=0.12, fs=T4)
t11c, b11c = box(ax, 0.78, y11, 0.26, 0.024,
    "Partial correlations\nControlling for RuBisCO lineage\n"
    "36/37 retain significance",
    CV, alpha=0.08, fs=T4)

arr(ax, 0.15, b10, 0.16, t11a)
arr(ax, 0.15, b8a, 0.35, t11b, rad=-0.15)
arr(ax, 0.82, b8c, 0.78, t11c)
arr(ax, 0.28, y11, 0.66, y11, rad=-0.03)

# ── R12: Spatial CV · Modules · GO (y=0.516) ───────────────
y12 = 0.516; h12 = 0.028

t12a, b12a = box(ax, 0.16, y12, 0.26, h12,
    "Spatial cross-validation\n"
    "Block CV (2°): Bathy R²=0.42, SST 0.50\n"
    "Basin-out (7) · Indep-subset",
    CV, alpha=0.08, fs=T4)
t12b, b12b = box(ax, 0.48, y12, 0.28, h12,
    "Modular organization\nBiclustering: 500 PFAMs × 29 env (Fig 4A)\n"
    "AEF SHAP: 716 × 64 AE dims\n"
    "18 row clusters, 8 col clusters (Fig 5)",
    CY, alpha=0.12, fs=T4)
t12c, b12c = box(ax, 0.82, y12, 0.22, h12,
    "GO enrichment\nHypergeometric tests\n"
    "Pfam2GO: 5,182→9,842 GO\n7 terms FDR < 0.05",
    CY, alpha=0.10, fs=T4)

arr(ax, 0.16, b11a, 0.16, t12a)
arr(ax, 0.44, b11b, 0.48, t12b, lbl="SHAP")
arr(ax, 0.62, y12, 0.71, y12)

# ── DIVIDER: PREDICTIVE MODELS (y=0.482) ───────────────────
secline(ax, 0.482, "PREDICTIVE MODELS", CJ)

# ── R13: ELF-NET + VICReg (y=0.445) ────────────────────────
y13 = 0.445; h13 = 0.034

t13a, b13a = box(ax, 0.24, y13, 0.40, h13,
    "ELF-NET: environment → full Pfam profile\n"
    "Deep MLP (94 env → CLR Pfam, n=995)\n"
    "AlgaGPT: R²=0.475, cos=0.666 | Pythia: R²=0.400\n"
    "76.4% domains AUC > 0.5, 175 domains AUC > 0.9",
    CR, alpha=0.10, fs=T4)
t13b, b13b = box(ax, 0.74, y13, 0.40, h13,
    "VICReg joint embedding (World Model)\n"
    "Enc_E: 24→128→32 | Enc_P: 20→256→128→32\n"
    "n = 1,810 GPS-mapped\n"
    "z_env + z_pfam (32-dim each), cos sim = 0.61",
    CJ, alpha=0.12, fs=T4)

arr(ax, 0.16, b12a, 0.18, t13a)
arr(ax, 0.48, b12b, 0.62, t13b, rad=-0.05)
arr(ax, 0.15, b8a, 0.58, t13b, rad=-0.18)

# ── R14: Ablation · Biomes · Perturbation (y=0.388) ────────
y14 = 0.388; h14 = 0.036

t14a, b14a = box(ax, 0.22, y14, 0.36, h14,
    "Productivity prediction ablation (Fig 6F-I)\n"
    "Leave-one-basin-out CV, n = 1,151 bio_valid\n"
    "Chl-a: baseline wins (R²=0.56)\n"
    "NFLH: baseline wins (R²=0.70)\n"
    "POC: joint wins (R²=0.53 vs 0.42, d=0.026)",
    CR, alpha=0.10, fs=T4)
t14b, b14b = box(ax, 0.58, y14, 0.22, 0.026,
    "Functional biomes (Fig 4D,G)\nHDBSCAN clustering of z_env\n"
    "23 clusters · Longhurst ARI=0.50",
    CY, alpha=0.10, fs=T4)
t14c, b14c = box(ax, 0.84, y14, 0.24, 0.030,
    "Perturbation sensitivity\nn = 1,810\n"
    "+2°C → −7.9% chl-a, −5.5% POC\n"
    "Bathy & solar dominate",
    CR, alpha=0.08, fs=T4)

arr(ax, 0.24, b13a, 0.22, t14a)
arr(ax, 0.62, b13b, 0.36, t14a, rad=0.08)
arr(ax, 0.70, b13b, 0.58, t14b)
arr(ax, 0.82, b13b, 0.84, t14c)

# ── KEY OUTPUTS BAR (y=0.340) ──────────────────────────────
y18 = 0.340
bar_bg = mpatches.FancyBboxPatch(
    (0.04, y18 - 0.011), 0.92, 0.022,
    boxstyle="round,pad=0.003",
    facecolor=(*ABYSS, 0.06), edgecolor=(*ABYSS, 0.22),
    linewidth=0.3, zorder=0)
ax.add_patch(bar_bg)
ax.text(0.50, y18,
    "KEY OUTPUTS: 18,449 FWER-surviving domain-env associations · CC1=0.82 · "
    "Bathymetry R²=0.57 from PFAM · POC +26% with joint embedding · "
    "~21,500 algal taxa across 7 ocean basins",
    ha='center', va='center', fontsize=T4, color=ABYSS,
    fontweight='bold', zorder=3)

# ── LEGEND ─────────────────────────────────────────────────
y19 = 0.312
ax.text(0.03, y19, "Sample counts at key stages:", fontsize=3.8,
        fontweight='bold', color=(0.3, 0.3, 0.3))
counts = [
    ("Total proteomes", "2,357"),
    ("AlgaGPT classified", "2,044"),
    ("GPS-mapped", "1,810"),
    ("With any GEE var", "1,523"),
    ("AlphaEarth complete", "995"),
    ("GEE subset (bidir)", "1,279"),
    ("Bio-valid (3 targets)", "1,151"),
]
for i, (stage, n) in enumerate(counts):
    col = i // 4; row = i % 4
    ax.text(0.03 + col * 0.27, y19 - 0.012 - row * 0.010,
            f"{stage}: n = {n}", fontsize=3.5, color=GRAY, family='monospace')

ax.text(0.55, y19, "Dataset breakdown (of 2,044 — see Fig 1A):", fontsize=3.8,
        fontweight='bold', color=(0.3, 0.3, 0.3))
sources = [
    ("TARA Oceans", "801"), ("MMETSP", "392"), ("RefGenome GenBank", "256"),
    ("no-GPS samples", "234"), ("OSD", "126"), ("AAC", "104"),
    ("TARA protist", "76"), ("Reference Genome", "67"), ("PRE_REF", "9"),
]
for i, (src, n) in enumerate(sources):
    col = i // 3; row = i % 3
    ax.text(0.55 + col * 0.16, y19 - 0.012 - row * 0.010,
            f"{src}: {n}", fontsize=3.5, color=GRAY, family='monospace')

# ── COLOR LEGEND (bottom) ──────────────────────────────────
y20 = 0.240
ax.text(0.03, y20, "Color key:", fontsize=3.8, fontweight='bold',
        color=(0.3, 0.3, 0.3))
legend_items = [
    (CI, "Input data"), (CP, "Processing"), (CL, "LLM classification"),
    (CA, "PFAM annotation"), (CE, "Environmental"), (CY, "Analysis"),
    (CM, "XGBoost/SHAP"), (CV, "Validation"), (CR, "Results"), (CJ, "Joint model"),
]
for i, (c, lab) in enumerate(legend_items):
    col = i // 5; row = i % 5
    xp = 0.03 + col * 0.22
    yp = y20 - 0.012 - row * 0.009
    r = mpatches.FancyBboxPatch(
        (xp, yp - 0.003), 0.012, 0.006, boxstyle="round,pad=0.001",
        facecolor=(*c, 0.20), edgecolor=c, linewidth=0.3)
    ax.add_patch(r)
    ax.text(xp + 0.016, yp, lab, fontsize=3.5, color=(0.3, 0.3, 0.3),
            va='center')

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
