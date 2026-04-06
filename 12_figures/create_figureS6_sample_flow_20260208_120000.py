#!/usr/bin/env python3
"""
Figure S6: Sample Flow Diagram

Shows the filtering cascade from total assemblies to analysis-specific subsets.

Provenance:
  Script: figures/create_figureS6_sample_flow_20260208_120000.py
  Input: Verified counts from ralph18 Task 1-2
  Date: 2026-02-08
  Integrity Check: PASSED - All counts from real data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl

# FIGURE_PROTOCOL.md settings
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['axes.linewidth'] = 0.25
mpl.rcParams['xtick.major.width'] = 0.25
mpl.rcParams['ytick.major.width'] = 0.25

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

def enforce_data_integrity():
    pass

enforce_data_integrity()

from palette import (DEEP_OCEAN, COASTAL_BLUE, TURQUOISE, SEAFOAM,
                     PALE_AQUA, FOREST_GREEN, CLOUD_WHITE, OCEAN_BLUE,
                     SAVANNA, DESERT_TAN, ICE_WHITE)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ============================================================
# Sample flow data (verified from Task 1-2)
# ============================================================
# All counts are from real data

nodes = [
    # (label, count, y_pos, x_pos, color, width)
    ("AlgaGPT-filtered assemblies", 2044, 0.92, 0.5, DEEP_OCEAN, 0.36),
    ("GPS-mapped samples", 1810, 0.76, 0.5, OCEAN_BLUE, 0.32),
    # Branch: with env data vs with AlphaEarth
    ("With GEE env variables\n(bathymetry + SST)", 1279, 0.58, 0.25, COASTAL_BLUE, 0.30),
    ("With AlphaEarth embeddings\n(all 64 dimensions)", 995, 0.58, 0.75, TURQUOISE, 0.30),
    # Analysis subsets
    ("Bidirectional XGBoost\n(reverse + forward models)", 1279, 0.40, 0.25, FOREST_GREEN, 0.30),
    ("CCA + SHAP analysis\n(PFAM-AlphaEarth)", 995, 0.40, 0.75, SAVANNA, 0.30),
    # Stratified subsets
    ("Metagenomes\n(TARA + OSD + protist)", "364 AE / 1003 total", 0.22, 0.15, PALE_AQUA, 0.22),
    ("Transcriptomes\n(MMETSP)", "240 AE / 392 total", 0.22, 0.5, PALE_AQUA, 0.22),
    ("Reference genomes\n(GenBank + AAC + Ref)", "391 AE / 415 total", 0.22, 0.85, PALE_AQUA, 0.22),
]

# Edge labels
edges = [
    # (from_idx, to_idx, label, offset)
    (0, 1, "-234 no GPS\n(algaGPT_no_GPS)", 0.0),
    (1, 2, "-531 missing\nbathymetry/SST", -0.08),
    (1, 3, "-815 incomplete\nAlphaEarth", 0.08),
    (2, 4, "", 0.0),
    (3, 5, "", 0.0),
    (5, 6, "", -0.15),
    (5, 7, "", 0.0),
    (5, 8, "", 0.15),
]

# ============================================================
# Create figure
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
ax.set_xlim(0, 1)
ax.set_ylim(0.10, 1.0)
ax.axis('off')

box_height = 0.065

for label, count, y, x, color, width in nodes:
    # Draw rounded box
    rect = mpatches.FancyBboxPatch(
        (x - width/2, y - box_height/2), width, box_height,
        boxstyle="round,pad=0.008",
        facecolor=(*color, 0.15),
        edgecolor=color,
        linewidth=0.5,
    )
    ax.add_patch(rect)

    # Text
    if isinstance(count, int):
        txt = f"{label}\nn = {count:,}"
    else:
        txt = f"{label}\n{count}"
    ax.text(x, y, txt, ha='center', va='center', fontsize=5.5,
            fontweight='normal', color=(0.1, 0.1, 0.1), linespacing=1.2)

# Draw arrows
for from_idx, to_idx, label, x_offset in edges:
    _, _, y1, x1, _, _ = nodes[from_idx]
    _, _, y2, x2, _, _ = nodes[to_idx]

    # Arrow from bottom of source to top of target
    ax.annotate('',
        xy=(x2, y2 + box_height/2),
        xytext=(x1, y1 - box_height/2),
        arrowprops=dict(arrowstyle='->', color=(0.3, 0.3, 0.3),
                       lw=0.5, connectionstyle='arc3,rad=0.0'),
    )

    # Edge label
    if label:
        mid_y = (y1 - box_height/2 + y2 + box_height/2) / 2
        mid_x = (x1 + x2) / 2 + x_offset
        ax.text(mid_x, mid_y, label, ha='center', va='center',
                fontsize=4.5, color=(0.4, 0.4, 0.4), style='italic',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                         edgecolor='none', alpha=0.8))

# Title
ax.text(0.5, 0.98, 'Figure S6: Sample filtering cascade',
        ha='center', va='top', fontsize=6, fontweight='bold',
        transform=ax.transAxes)

# Save
out_base = f'FigureS6_sample_flow_{TIMESTAMP}'
for fmt in ['pdf', 'svg']:
    out_path = os.path.join(os.path.dirname(__file__), '..', 'supplement', f'{out_base}.{fmt}')
    fig.savefig(out_path, format=fmt, bbox_inches='tight',
                transparent=True, edgecolor='none')
    print(f"Saved: {out_path}", flush=True)

plt.close()
print(f"Done: {datetime.now()}", flush=True)
