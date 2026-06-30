#!/usr/bin/env python3
"""
Single combined RuBisCO phylogenetic figure with collapsed clades and
per-lineage alignment conservation.

Three panels (Form IB, Form ID, Form II), each showing:
  - Left: collapsed tree with weighted triangles (height ∝ n_tips)
  - Right: per-lineage alignment conservation heatmap (from lineage-specific .aln)

Output: figures/FigureS8_phylo_collapsed_20260529.pdf/.svg
"""

import os, sys, re, warnings
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from Bio import Phylo, AlignIO

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
MANUSCRIPT_DIR = SCRIPT_DIR.parent
TREE_DIR = Path("/media/drn2/External/TARA-Oceans/03_analyses/rubisco_hmms/trees")
SEQ_DIR  = Path("/media/drn2/External/TARA-Oceans/03_analyses/rubisco_hmms/sequences")
ALN_DIR  = Path("/media/drn2/External/TARA-Oceans/03_analyses/rubisco_hmms/alignments")

sys.path.insert(0, str(MANUSCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
from palette import (FOREST_GREEN, SAVANNA, PALE_GREEN, SEAFOAM,
                     DEEP_OCEAN, COASTAL_BLUE, TURQUOISE, OCEAN_BLUE,
                     DESERT_TAN, SAND)

mpl.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 6, 'axes.linewidth': 0.25,
})

# ═══════════════════════════════════════════════════════════════════════════════
# Lineage definitions: (display_name, color, fasta_key, aln_file)
# ═══════════════════════════════════════════════════════════════════════════════

FORM_IB = [
    ('Chlorellaceae',    FOREST_GREEN,          'chlorellaceae',    'chlorellaceae_rbcL.aln'),
    ('Mamiellophyceae',  SAVANNA,               'mamiellophyceae',  'mamiellophyceae_rbcL.aln'),
    ('Prasinophyceae',   PALE_GREEN,            'prasinophyceae',   'prasinophyceae_rbcL.aln'),
    ('Pyramimonadales',  SEAFOAM,               'pyramimonadales',  'pyramimonadales_rbcL.aln'),
    ('Scenedesmaceae',   (0.31, 0.49, 0.31),    'scenedesmaceae',   'scenedesmaceae_rbcL.aln'),
    ('Trebouxiophyceae', DEEP_OCEAN,            'trebouxiophyceae', 'trebouxiophyceae_rbcL.aln'),
]

FORM_ID = [
    ('Bolidophyceae',  COASTAL_BLUE,          'bolidophyceae',  'bolidophyceae_rbcL.aln'),
    ('Cryptophyta',    (0.725, 0.875, 0.839), 'cryptophyta',    'cryptophyta_rbcL.aln'),
    ('Haptophyta',     TURQUOISE,             'haptophyta',     'haptophyta_rbcL.aln'),
    ('Pelagophyceae',  OCEAN_BLUE,            'pelagophyceae',  'pelagophyceae_rbcL.aln'),
]

FORM_II = [
    ('Chromerida',      SAND,                  'chromerida',      'chromerida_rbcL.aln'),
    ('Gonyaulacales',   (0.62, 0.42, 0.258),   'gonyaulacales',   'gonyaulacales_rbcL.aln'),
    ('Peridiniales',    (0.678, 0.486, 0.306),  'peridiniales',    'peridiniales_rbcL.aln'),
    ('Prorocentrales',  (0.824, 0.659, 0.447),  'prorocentrales',  'prorocentrales_rbcL.aln'),
    ('Symbiodiniaceae', (0.545, 0.353, 0.200),  'symbiodiniaceae', 'symbiodiniaceae_rbcL.aln'),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _acc_from_header(line):
    """Extract accession from a FASTA header line."""
    hdr = line[1:].strip()
    if '|' in hdr:
        return hdr.split('|')[1]
    return hdr.split()[0]


def get_accs(fasta_key):
    """Return set of accessions for a lineage."""
    for suf in ('_rbcL_nr.fasta', '_rbcL_filtered.fasta', '_rbcL.fasta'):
        fp = SEQ_DIR / f'{fasta_key}{suf}'
        if fp.exists():
            accs = set()
            with open(fp) as f:
                for line in f:
                    if line.startswith('>'):
                        accs.add(_acc_from_header(line))
            return accs
    return set()


def tip_acc(tip):
    """Extract accession from a Bio.Phylo terminal node."""
    name = tip.name or ''
    if '|' in name:
        return name.split('|')[1]
    return name.split()[0] if name else ''


def compute_depths(tree):
    depths = {}
    def _r(clade, d):
        depths[id(clade)] = d
        for ch in clade.clades:
            _r(ch, d + (ch.branch_length or 0))
    _r(tree.root, 0)
    return depths


def find_mrca_depth(tree, depths, tip_ids):
    """Walk from root; the MRCA is the deepest node whose descendant-tip set
    is a superset of tip_ids."""
    target = set(tip_ids)
    best_depth = 0

    def _desc_tips(clade):
        if clade.is_terminal():
            return {id(clade)}
        s = set()
        for ch in clade.clades:
            s |= _desc_tips(ch)
        return s

    def _search(clade):
        nonlocal best_depth
        desc = _desc_tips(clade)
        if target <= desc:
            d = depths[id(clade)]
            if d > best_depth:
                best_depth = d
            for ch in clade.clades:
                _search(ch)

    _search(tree.root)
    return best_depth


def clamp_branches(tree, pct=97):
    bls = [c.branch_length for c in tree.find_clades()
           if c.branch_length and c.branch_length > 0]
    if not bls:
        return
    thr = np.percentile(bls, pct)
    for c in tree.find_clades():
        if c.branch_length and c.branch_length > thr:
            c.branch_length = thr


def conservation_from_aln(aln_path):
    """Per-column fraction of most-common residue (ignoring gaps)."""
    aln = AlignIO.read(str(aln_path), 'fasta')
    n = len(aln)
    L = aln.get_alignment_length()
    cons = np.zeros(L)
    for i in range(L):
        col = aln[:, i].replace('-', '').replace('X', '').replace('.', '')
        if col:
            cons[i] = Counter(col).most_common(1)[0][1] / n
    return cons, L


# ═══════════════════════════════════════════════════════════════════════════════
# Collapse a tree
# ═══════════════════════════════════════════════════════════════════════════════

def collapse_tree(tree_path, lineage_defs):
    """
    Returns (lineage_data_list, max_depth).
    Each entry: {name, color, key, aln_file, n_tips, mrca_depth, min_depth,
                 max_depth, med_depth}.
    Tip counts come from the lineage-specific FASTA (ground truth), not from
    matching against the overview tree.
    """
    tree = Phylo.read(str(tree_path), 'newick')
    clamp_branches(tree)
    depths = compute_depths(tree)

    # Build per-lineage accession sets and match to tree tips
    results = []
    for display, color, key, aln_file in lineage_defs:
        accs = get_accs(key)
        # Match tips
        matched = [t for t in tree.get_terminals() if tip_acc(t) in accs]
        # Tip count from the FASTA (authoritative), not from tree matching
        n_tips = len(accs)

        if matched:
            tip_depths = [depths[id(t)] for t in matched]
            mrca_d = find_mrca_depth(tree, depths, {id(t) for t in matched})
        else:
            tip_depths = [0]
            mrca_d = 0

        results.append({
            'name': display, 'color': color, 'key': key,
            'aln_file': aln_file,
            'n_tips': n_tips,
            'mrca_depth': mrca_d,
            'min_depth': min(tip_depths),
            'max_depth': max(tip_depths),
            'med_depth': float(np.median(tip_depths)),
        })

    results.sort(key=lambda r: r['mrca_depth'])
    return results, max(depths.values())


# ═══════════════════════════════════════════════════════════════════════════════
# Draw one panel
# ═══════════════════════════════════════════════════════════════════════════════

def draw_panel(ax_tree, ax_aln, lineage_data, max_depth,
               title, scale_val=0.05):

    n_lin = len(lineage_data)
    total_tips = sum(d['n_tips'] for d in lineage_data)

    # Vertical layout — height proportional to sqrt(n_tips) for readability.
    # Compressed ~75% vs. the original standalone S8: each triangle only needs
    # to be tall enough to carry its italic label and matching conservation
    # strip. Height multiplier 6 -> 1.5, gap 0.4 -> 0.22, min height 0.6 -> 0.42.
    weights = [max(np.sqrt(d['n_tips']), 2) for d in lineage_data]
    total_w = sum(weights)
    gap = 0.22

    y_cursor = 0.0
    positions = []  # (y_top, y_mid, y_bot)
    for w in weights:
        h = w / total_w * n_lin * 1.5
        h = max(h, 0.42)
        positions.append((y_cursor, y_cursor + h / 2, y_cursor + h))
        y_cursor += h + gap
    total_h = y_cursor - gap

    x_scale = 0.7 / max_depth if max_depth > 0 else 1

    # Draw each lineage
    _labels = []
    for i, d in enumerate(lineage_data):
        y_top, y_mid, y_bot = positions[i]
        mrca_x = d['mrca_depth'] * x_scale
        tip_x  = d['med_depth']  * x_scale

        # Backbone → MRCA horizontal
        ax_tree.plot([0, mrca_x], [y_mid, y_mid],
                     color='black', lw=0.6, solid_capstyle='butt')

        # Triangle
        tri = Polygon(
            [(mrca_x, y_mid), (tip_x, y_top + 0.03), (tip_x, y_bot - 0.03)],
            closed=True, fc=d['color'], ec='black', lw=0.4, alpha=0.85, zorder=2)
        ax_tree.add_patch(tri)

        t = ax_tree.text(tip_x + 0.015, y_mid,
                         f"{d['name']} (n={d['n_tips']})",
                         fontsize=6, va='center', ha='left', fontstyle='italic')
        _labels.append(t)

    # Vertical backbone connecting lineage stems
    if n_lin > 1:
        y_first = positions[0][1]
        y_last  = positions[-1][1]
        ax_tree.plot([0, 0], [y_first, y_last],
                     color='black', lw=0.6, solid_capstyle='butt')

    # Scale bar (offsets tightened to match compressed row heights)
    bar_len = scale_val * x_scale
    ax_tree.plot([0, bar_len], [total_h + 0.18, total_h + 0.18],
                 color='black', lw=0.5)
    ax_tree.text(bar_len / 2, total_h + 0.40, str(scale_val),
                 fontsize=6, ha='center', va='top')

    # Set xlim so the rightmost italic label sits flush against the box edge,
    # removing the dead whitespace between the tree and the alignment column.
    ax_tree.figure.canvas.draw()
    rend = ax_tree.figure.canvas.get_renderer()
    inv = ax_tree.transData.inverted()
    max_label_right = max(
        inv.transform((t.get_window_extent(renderer=rend).x1, 0))[0]
        for t in _labels
    )
    ax_tree.set_xlim(-0.05, max_label_right)
    ax_tree.set_ylim(total_h + 0.58, -0.35)
    ax_tree.axis('off')
    ax_tree.set_title(title, fontsize=6, fontweight='bold', pad=3)

    # ── Per-lineage alignment conservation heatmap ────────────────────────────
    if ax_aln is None:
        return

    # Compute conservation for each lineage from its own alignment
    cons_data = []
    max_len = 0
    for d in lineage_data:
        aln_path = ALN_DIR / d['aln_file']
        if aln_path.exists():
            cons, L = conservation_from_aln(aln_path)
            cons_data.append((cons, L))
            if L > max_len:
                max_len = L
        else:
            cons_data.append((np.zeros(1), 1))

    # Build a 2D array: rows = lineages, cols = alignment positions
    # Pad shorter alignments with NaN
    n_rows = n_lin
    heatmap = np.full((n_rows, max_len), np.nan)
    for i, (cons, L) in enumerate(cons_data):
        heatmap[i, :L] = cons

    # Create a masked colormap (NaN = white)
    cmap = mpl.cm.Greens.copy()
    cmap.set_bad('white')

    # Map rows to vertical positions matching the tree
    # Each row spans the same y-range as its triangle
    for i in range(n_rows):
        y_top, y_mid, y_bot = positions[i]
        row = heatmap[i:i+1, :]
        ax_aln.imshow(row, aspect='auto',
                      extent=[0, max_len, y_bot, y_top],
                      cmap=cmap, vmin=0, vmax=1, interpolation='nearest')

    ax_aln.set_xlim(0, max_len)
    ax_aln.set_ylim(total_h + 0.45, -0.35)
    ax_aln.set_xlabel(f'Alignment position (aa)', fontsize=6, labelpad=2)
    ax_aln.tick_params(axis='x', labelsize=5, pad=1)
    ax_aln.tick_params(axis='y', left=False, labelleft=False)
    for sp in ('top', 'right', 'left'):
        ax_aln.spines[sp].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Creating collapsed RuBisCO phylogenetic figure...")

    fig = plt.figure(figsize=(7, 3.0), layout='constrained')
    subfigs = fig.subfigures(3, 1, height_ratios=[1.2, 0.8, 1.0], hspace=0.04)

    # Panel A — Form IB
    ax_a = subfigs[0].subplots(1, 2, width_ratios=[0.85, 1],
                                gridspec_kw={'wspace': 0.04})
    data_ib, md_ib = collapse_tree(TREE_DIR / 'green_rbcL.treefile', FORM_IB)
    draw_panel(ax_a[0], ax_a[1], data_ib, md_ib,
               'Form IB — Chlorophyte RbcL', scale_val=0.05)
    ax_a[0].text(-0.05, 1.05, 'A', transform=ax_a[0].transAxes,
                 fontsize=6, fontweight='bold', va='bottom')

    # Panel B — Form ID
    ax_b = subfigs[1].subplots(1, 2, width_ratios=[0.85, 1],
                                gridspec_kw={'wspace': 0.04})
    data_id, md_id = collapse_tree(TREE_DIR / 'red_rbcL.treefile', FORM_ID)
    draw_panel(ax_b[0], ax_b[1], data_id, md_id,
               'Form ID — Red/cryptophyte RbcL', scale_val=0.05)
    ax_b[0].text(-0.05, 1.05, 'B', transform=ax_b[0].transAxes,
                 fontsize=6, fontweight='bold', va='bottom')

    # Panel C — Form II
    ax_c = subfigs[2].subplots(1, 2, width_ratios=[0.85, 1],
                                gridspec_kw={'wspace': 0.04})
    data_ii, md_ii = collapse_tree(TREE_DIR / 'formII_rbcL.nwk', FORM_II)
    draw_panel(ax_c[0], ax_c[1], data_ii, md_ii,
               'Form II — Dinoflagellate RbcL', scale_val=0.1)
    ax_c[0].text(-0.05, 1.05, 'C', transform=ax_c[0].transAxes,
                 fontsize=6, fontweight='bold', va='bottom')

    out_dir = MANUSCRIPT_DIR / 'figures'
    for fmt in ('pdf', 'svg'):
        p = out_dir / f'FigureS_phylo_compact_20260627.{fmt}'
        fig.savefig(str(p), format=fmt, transparent=True, edgecolor='none')
        print(f'  Saved: {p}')
    fig.savefig(str(out_dir / 'FigureS_phylo_compact_20260627_check.png'),
                dpi=300, transparent=False)
    plt.close(fig)
    print("Done.")


if __name__ == '__main__':
    os.chdir(str(SCRIPT_DIR))
    main()
