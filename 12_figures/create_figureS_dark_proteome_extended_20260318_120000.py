#!/usr/bin/env python3
"""
Supplemental Figure: Dark Proteome Extended
# --------------------------------------------------------------------------

Page 1 (7.5 x 7.0 in, 2 panels):
  Panel A — Per-category dark fraction (grouped bar: median + mean ± SD)
  Panel B — Novel domain prevalence by source type (violin + box)

Page 2 (7.5 x 10.0 in):
  Panel C — Contig neighborhood diagrams for 8 novel families (rasterized
            from A10 pipeline PDF, arranged in 4×2 grid)

Data:
  - source_data/pfam_density_summary.tsv
  - source_data/dark_proteome/novel_domain_summary.tsv
  - figures/novel_families/contig_neighborhood_diagrams.pdf

Created: 2026-03-18
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import datetime
import subprocess, tempfile, os, sys
import warnings
warnings.filterwarnings('ignore')

_palette_dir = str(Path(__file__).resolve().parent)
if _palette_dir not in sys.path:
    sys.path.insert(0, _palette_dir)

from palette import (
    DEEP_OCEAN, TURQUOISE, DESERT_TAN, OCEAN_BLUE, COASTAL_BLUE,
    FOREST_GREEN, CLAY, CLOUD_WHITE, PALE_AQUA, SAND, SIENNA,
)

# ── rcParams ─────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 6, 'axes.labelsize': 6, 'axes.titlesize': 6,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 5.5,
    'axes.linewidth': 0.25,
    'xtick.major.width': 0.25, 'ytick.major.width': 0.25,
    'xtick.major.size': 2, 'ytick.major.size': 2,
    'axes.labelpad': 1, 'xtick.major.pad': 1, 'ytick.major.pad': 1,
})

SCRIPT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = SCRIPT_DIR.parent
DATA_DIR = MANUSCRIPT_DIR / 'source_data' / 'dark_proteome'
OUT_DIR = SCRIPT_DIR
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
NEIGHBORHOOD_PDF = SCRIPT_DIR / 'novel_families' / 'contig_neighborhood_diagrams.pdf'

# Source-type colors (shared with Figure 7)
SRC_COLORS = {
    'Metagenome': TURQUOISE,
    'Cultured reference': FOREST_GREEN,
    'NCBI genome': DESERT_TAN,
    'Other': COASTAL_BLUE,
}
SRC_ORDER = ['Metagenome', 'NCBI genome', 'Cultured reference', 'Other']

def classify_source(name):
    if '-NODE-' in name or 'MGYA' in name:
        return 'Metagenome'
    elif '.PRE.fa' in name or 'MMETSP' in name:
        return 'Cultured reference'
    elif 'GCA_' in name or 'GCF_' in name:
        return 'NCBI genome'
    return 'Other'

# --------------------------------------------------------------------------
# PANEL A — Per-category dark fraction
# --------------------------------------------------------------------------

def draw_panel_a(ax):
    df = pd.read_csv(MANUSCRIPT_DIR / 'source_data' / 'pfam_density_summary.tsv',
                     sep='\t')
    cats = df[df['category'] != 'GLOBAL'].sort_values('n_samples', ascending=False)

    labels = {
        'Metagenome ORFs': 'Meta',
        'Transcriptome ORFs': 'Trans',
        'NCBI genome': 'NCBI',
        'Other cultured reference': 'Cultured',
    }
    xlabels = [labels.get(c, c) for c in cats['category']]
    n = len(xlabels)
    x = np.arange(n)
    medians = cats['pct_dark_median'].values
    means = cats['pct_dark_mean'].values
    stds = cats['pct_dark_std'].values
    ns = cats['n_samples'].values
    colors = [TURQUOISE, COASTAL_BLUE, DESERT_TAN, CLAY]
    w = 0.35

    ax.bar(x - w/2, medians, w, label='Median', color=colors, alpha=0.85,
           edgecolor='white', linewidth=0.2)
    ax.bar(x + w/2, means, w, yerr=stds, label='Mean ± SD',
           color=[(*c[:3], 0.45) for c in colors],
           edgecolor=colors, linewidth=0.3,
           error_kw=dict(lw=0.3, capsize=1.5, capthick=0.3, color='#555555'))

    for i in range(n):
        ax.text(x[i], max(medians[i], means[i] + stds[i]) + 2,
                f'{ns[i]:,}', ha='center', va='bottom',
                fontsize=4.5, color='#555555')

    glob = df[df['category'] == 'GLOBAL']
    if len(glob):
        gm = float(glob['pct_dark_median'].iloc[0])
        ax.axhline(gm, color=DEEP_OCEAN, lw=0.4, ls='--', alpha=0.7)
        ax.text(n - 0.3, gm + 1.5, f'{gm:.1f}%',
                fontsize=4.5, color=DEEP_OCEAN, ha='right', va='bottom')

    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=5)
    ax.set_ylabel('Dark fraction (%)')
    ax.set_ylim(0, 108)
    ax.legend(fontsize=4, frameon=False, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('A', loc='left', fontweight='bold', fontsize=8, pad=2)

# --------------------------------------------------------------------------
# PANEL B — Domain prevalence by source type (violin + box)
# --------------------------------------------------------------------------

def draw_panel_b(ax):
    """Violin + strip plot: prevalence (n_samples) distribution for novel
    domains, grouped by source type."""
    df = pd.read_csv(DATA_DIR / 'novel_domain_summary.tsv', sep='\t')
    df['source'] = df['domain'].apply(classify_source)

    # Prepare data in source order
    data = []
    positions = []
    colors_list = []
    tick_labels = []
    for i, src in enumerate(SRC_ORDER):
        sub = df[df['source'] == src]
        if len(sub) == 0:
            continue
        data.append(sub['n_samples'].values)
        positions.append(i)
        colors_list.append(SRC_COLORS[src])
        tick_labels.append(f'{src}\n(n = {len(sub):,})')

    # Violin plots
    parts = ax.violinplot(data, positions=positions, widths=0.7,
                          showmeans=False, showmedians=False,
                          showextrema=False)
    for pc, color in zip(parts['bodies'], colors_list):
        pc.set_facecolor(color)
        pc.set_alpha(0.35)
        pc.set_edgecolor(color)
        pc.set_linewidth(0.3)

    # Box plots overlaid
    bp = ax.boxplot(data, positions=positions, widths=0.25,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='white', lw=0.5),
                    whiskerprops=dict(color='#666666', lw=0.3),
                    capprops=dict(color='#666666', lw=0.3))
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('white')
        patch.set_linewidth(0.2)

    # Median annotations
    for i, d in enumerate(data):
        med = np.median(d)
        ax.text(positions[i], med + 50, f'{med:.0f}',
                ha='center', va='bottom', fontsize=4.5, fontweight='bold',
                color='#444444')

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=4.5)
    ax.set_ylabel('Assemblies detected (of 2,044)')
    ax.set_ylim(0, 2150)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('B', loc='left', fontweight='bold', fontsize=8, pad=2)

# --------------------------------------------------------------------------
# PAGE 2 — Contig neighborhood diagrams (rasterized)
# --------------------------------------------------------------------------

def make_page2():
    """Full page: 8 novel families in 4×2 grid, rasterized from PDF."""
    from PIL import Image

    if not NEIGHBORHOOD_PDF.exists():
        fig2 = plt.figure(figsize=(7.5, 10.0))
        ax = fig2.add_subplot(111)
        ax.text(0.5, 0.5, 'Panel C: Contig neighborhoods\n(PDF not found)',
                ha='center', va='center', fontsize=10, color='#888888')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        return fig2

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_base = tmp.name.replace('.png', '')
    subprocess.run(['pdftoppm', '-png', '-r', '400',
                    str(NEIGHBORHOOD_PDF), tmp_base],
                   check=True, capture_output=True)
    png_path = tmp_base + '-1.png'
    img = Image.open(png_path)
    w, h = img.size
    print(f'  Neighborhoods image: {w}×{h}')

    fig2 = plt.figure(figsize=(7.5, 10.0))
    gs2 = gridspec.GridSpec(4, 2, figure=fig2, hspace=0.06, wspace=0.04,
                            left=0.02, right=0.98, top=0.96, bottom=0.02)
    fig2.text(0.03, 0.99, 'C', fontsize=10, fontweight='bold',
              va='top', ha='left')
    fig2.text(0.5, 0.99, 'Contig neighborhood diagrams for 8 novel families',
              fontsize=7, ha='center', va='top', color='#555555')

    strip_h = h // 8
    for i in range(8):
        row, col = i % 4, i // 4
        strip = img.crop((0, i * strip_h, w, min((i+1) * strip_h, h)))
        ax = fig2.add_subplot(gs2[row, col])
        ax.imshow(np.array(strip), aspect='auto', interpolation='lanczos')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.25); sp.set_color('#cccccc')

    os.unlink(png_path)
    return fig2

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    print('=' * 70)
    print('Supplemental Figure: Dark Proteome Extended')
    print(f'  {datetime.now().isoformat()}')
    print('=' * 70)

    # Page 1
    fig1 = plt.figure(figsize=(2.7, 1.5))
    gs = gridspec.GridSpec(1, 2, figure=fig1, wspace=0.55)

    print('Panel A (dark fraction by category)...')
    draw_panel_a(fig1.add_subplot(gs[0]))

    print('Panel B (prevalence by source type)...')
    draw_panel_b(fig1.add_subplot(gs[1]))

    # Page 2
    print('Panel C (contig neighborhoods)...')
    fig2 = make_page2()

    # Export
    pdf_out = OUT_DIR / f'FigureS_dark_proteome_extended_{TIMESTAMP}.pdf'
    with PdfPages(str(pdf_out)) as pdf:
        pdf.savefig(fig1, bbox_inches='tight', transparent=True)
        pdf.savefig(fig2, bbox_inches='tight', transparent=True)
    print(f'Saved: {pdf_out}')

    svg_out = OUT_DIR / f'FigureS_dark_proteome_extended_{TIMESTAMP}.svg'
    fig1.savefig(str(svg_out), format='svg', bbox_inches='tight',
                 transparent=True, edgecolor='none')
    print(f'Saved: {svg_out}')

    prov = OUT_DIR / f'FigureS_dark_proteome_extended_{TIMESTAMP}_provenance.txt'
    with open(prov, 'w') as f:
        f.write(f'# Provenance — Supplemental Figure: Dark Proteome Extended\n')
        f.write(f'Script: {Path(__file__).resolve()}\n')
        f.write(f'Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n')
        f.write('Panel A: pfam_density_summary.tsv (dark fraction by category)\n')
        f.write('Panel B: novel_domain_summary.tsv (prevalence violin by source)\n')
        f.write('Panel C: novel_families/contig_neighborhood_diagrams.pdf (rasterized)\n')
        f.write(f'\nPage 1: 7.5 × 4.0 in (Panels A, B side-by-side)\n')
        f.write(f'Page 2: 7.5 × 10.0 in (Panel C, 4×2 grid)\n')
    print(f'Saved: {prov}')

    plt.close(fig1); plt.close(fig2)
    print('DONE')

if __name__ == '__main__':
    main()
