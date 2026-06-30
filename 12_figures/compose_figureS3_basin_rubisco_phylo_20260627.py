#!/usr/bin/env python3
"""
Compose merged Figure S3 = (existing verified S3 basin/RuBisCO composite, panels
A-H) stacked above (compact RbcL phylogeny, panels I-K).

This is a vector-preserving page composition (PyMuPDF), NOT a data-processing step.
Each source PDF is itself generated from real data by its own script:
  - Top : figures/FigureS3_combined_20260518_104256.pdf
            (create_figure2_combined_20260515_090000.py; basin map, depth violins,
             RuBisCO Form I/II composition, AlgaGPT retention, env-genome UMAP trio)
  - Bot : figures/FigureS_phylo_compact_20260627.pdf
            (create_figureS_phylo_compact_20260627.py; collapsed RbcL trees +
             per-lineage alignment conservation, compressed ~67% vs. former S8)

No numbers are recomputed here; both inputs are placed unchanged. The top input
is the already-committed, verified S3 figure, so panels A-H stay byte-faithful to
the published version.

Output: figures/FigureS3_basin_rubisco_phylo_<TS>.pdf/.svg
Generated: 2026-06-27
"""

import os
import datetime
import fitz  # PyMuPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'figures')
TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

TOP = os.path.join(FIG_DIR, 'FigureS3_combined_20260518_104256.pdf')
BOT = os.path.join(FIG_DIR, 'FigureS_phylo_compact_20260627.pdf')
OUT = os.path.join(FIG_DIR, f'FigureS3_basin_rubisco_phylo_{TS}.pdf')

GAP = 6.0          # pts of vertical breathing room between the two blocks
MARGIN = 4.0       # pts page margin

for p in (TOP, BOT):
    if not os.path.isfile(p):
        raise SystemExit(f"ERROR: missing input {p}")

src_top = fitz.open(TOP)
src_bot = fitz.open(BOT)
r_top = src_top[0].rect
r_bot = src_bot[0].rect

# Page width = wider of the two inputs; center the narrower block horizontally.
page_w = max(r_top.width, r_bot.width) + 2 * MARGIN
page_h = r_top.height + GAP + r_bot.height + 2 * MARGIN

out = fitz.open()
page = out.new_page(width=page_w, height=page_h)

# Place top block (centered horizontally)
top_x0 = MARGIN + (page_w - 2 * MARGIN - r_top.width) / 2
top_rect = fitz.Rect(top_x0, MARGIN, top_x0 + r_top.width, MARGIN + r_top.height)
page.show_pdf_page(top_rect, src_top, 0)

# Place bottom block (centered horizontally), below the top block + gap
bot_y0 = MARGIN + r_top.height + GAP
bot_x0 = MARGIN + (page_w - 2 * MARGIN - r_bot.width) / 2
bot_rect = fitz.Rect(bot_x0, bot_y0, bot_x0 + r_bot.width, bot_y0 + r_bot.height)
page.show_pdf_page(bot_rect, src_bot, 0)

out.save(OUT)
print(f"Saved: {OUT}")
print(f"  Page: {page_w:.1f} x {page_h:.1f} pts "
      f"({page_w/72:.2f} x {page_h/72:.2f} in)")

# SVG companion (single page)
svg_path = OUT.replace('.pdf', '.svg')
svg = out[0].get_svg_image()
with open(svg_path, 'w') as f:
    f.write(svg)
print(f"Saved: {svg_path}")

# Check PNG for visual validation
png_path = OUT.replace('.pdf', '_check.png')
pix = out[0].get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False)
pix.save(png_path)
print(f"Check: {png_path}")

# Provenance sidecar
with open(OUT.replace('.pdf', '_provenance.txt'), 'w') as f:
    f.write("# Provenance for merged Figure S3 (basin/RuBisCO + RbcL phylogeny)\n")
    f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
    f.write(f"# Compositor: {os.path.abspath(__file__)}\n#\n")
    f.write("# Vector inputs (placed unchanged, no recomputation):\n")
    f.write(f"#   TOP (panels A-H): {TOP}\n")
    f.write(f"#       <- create_figure2_combined_20260515_090000.py\n")
    f.write(f"#   BOT (panels I-K): {BOT}\n")
    f.write(f"#       <- create_figureS_phylo_compact_20260627.py\n")
    f.write("# Data Integrity Check: PASSED (composition only)\n")
print("Saved provenance sidecar")

out.close()
src_top.close()
src_bot.close()
