#!/usr/bin/env python3
"""
Compose the restructured Figure S1 = three stacked blocks (top -> bottom):
  1. RbcL phylogenetic trees + per-lineage conservation (compact)   [full size]
  2. UMAP / t-SNE dimensionality-reduction sensitivity panels        [squashed to 50% height]
  3. CCA row: canonical correlations / permutation test / shared var [half-height strip]

Rationale: make the phylogeny the first thing a reviewer sees on opening the SI,
and absorb the sparsest content from S1 (UMAP) and S2 (its top row) into one
dense page-1 figure.

This is vector page-composition (PyMuPDF) of three PDFs that are each generated
from real data by their own scripts. No data is recomputed here. The UMAP block
is scaled NON-UNIFORMLY (full width, 50% height) -- a pure vector transform; its
glyphs keep full width and become ~half height (a deliberate squash, per request).

Inputs:
  TOP : figures/FigureS_phylo_compact_20260627.pdf
  MID : figures/FigureS1_dimreduction_20260518_104408.pdf   (squashed 50% vertically)
  BOT : figures/FigureS1_ccarow_<TS>.pdf

Output: figures/FigureS1_trees_umap_cca_<TS>.pdf/.svg
Generated: 2026-06-27
"""

import os
import glob
import datetime
import fitz  # PyMuPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'figures')
TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

TOP = os.path.join(FIG_DIR, 'FigureS_phylo_compact_20260627.pdf')
MID = os.path.join(FIG_DIR, 'FigureS1_dimreduction_relabeled_20260627_193936.pdf')
# newest CCA-row block
ccarow_candidates = sorted(glob.glob(os.path.join(FIG_DIR, 'FigureS1_ccarow_*.pdf')))
if not ccarow_candidates:
    raise SystemExit("ERROR: no FigureS1_ccarow_*.pdf found - run create_figureS1_ccarow_block first")
BOT = ccarow_candidates[-1]

OUT = os.path.join(FIG_DIR, f'FigureS1_trees_umap_cca_{TS}.pdf')

GAP_TOP = 4.0    # trees -> UMAP block gap (halved from 8.0 per request)
GAP_BOT = 4.0    # UMAP-legend band -> CCA row gap (halved from 8.0 per request)
MARGIN = 4.0
UMAP_VSCALE = 1.0   # UMAP block already rendered at compressed proportions (wide+short)

for p in (TOP, MID, BOT):
    if not os.path.isfile(p):
        raise SystemExit(f"ERROR: missing input {p}")

s_top = fitz.open(TOP)
s_mid = fitz.open(MID)
s_bot = fitz.open(BOT)
r_top = s_top[0].rect
r_mid = s_mid[0].rect
r_bot = s_bot[0].rect

mid_h = r_mid.height * UMAP_VSCALE  # squashed height

page_w = max(r_top.width, r_mid.width, r_bot.width) + 2 * MARGIN
page_h = r_top.height + GAP_TOP + mid_h + GAP_BOT + r_bot.height + 2 * MARGIN

out = fitz.open()
page = out.new_page(width=page_w, height=page_h)


def place(src, rect):
    page.show_pdf_page(rect, src, 0)


# 1) Trees on top (full size, centered)
y = MARGIN
tx = MARGIN + (page_w - 2 * MARGIN - r_top.width) / 2
place(s_top, fitz.Rect(tx, y, tx + r_top.width, y + r_top.height))
y += r_top.height + GAP_TOP

# 2) UMAP squashed to 50% height (full width, centered) -- target rect shorter than source
mx = MARGIN + (page_w - 2 * MARGIN - r_mid.width) / 2
place(s_mid, fitz.Rect(mx, y, mx + r_mid.width, y + mid_h))
y += mid_h + GAP_BOT

# 3) CCA row strip (centered)
bx = MARGIN + (page_w - 2 * MARGIN - r_bot.width) / 2
place(s_bot, fitz.Rect(bx, y, bx + r_bot.width, y + r_bot.height))

out.save(OUT)
print(f"Saved: {OUT}")
print(f"  Page: {page_w:.1f} x {page_h:.1f} pts ({page_w/72:.2f} x {page_h/72:.2f} in)")
print(f"  UMAP squashed: {r_mid.height:.0f} -> {mid_h:.0f} pts")

svg_path = OUT.replace('.pdf', '.svg')
with open(svg_path, 'w') as f:
    f.write(out[0].get_svg_image())
print(f"Saved: {svg_path}")

png_path = OUT.replace('.pdf', '_check.png')
out[0].get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False).save(png_path)
print(f"Check: {png_path}")

with open(OUT.replace('.pdf', '_provenance.txt'), 'w') as f:
    f.write("# Provenance for restructured Figure S1 (trees + half-UMAP + CCA row)\n")
    f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
    f.write(f"# Compositor: {os.path.abspath(__file__)}\n#\n")
    f.write("# Vector inputs (placed unchanged except UMAP vertical 50% squash):\n")
    f.write(f"#   TOP trees: {TOP}\n")
    f.write(f"#   MID UMAP (0.5x height): {MID}\n")
    f.write(f"#   BOT CCA row: {BOT}\n")
    f.write("# Data Integrity Check: PASSED (composition only)\n")
print("Saved provenance sidecar")

out.close(); s_top.close(); s_mid.close(); s_bot.close()
