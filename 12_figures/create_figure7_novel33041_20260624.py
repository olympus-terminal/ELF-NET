#!/usr/bin/env python3
"""
Figure 7: A conserved uncharacterized protein family across photosynthetic eukaryotes

Layout:
  Row 1 (45%): [A Phylogenetic tree]  [B ESMFold structure]
  Row 2 (20%): [C Contig context diagram — full width]
  Row 3 (35%): [D Copy-number bar chart]  [E Conserved-position alignment]

All data from source_data/novel33041/.
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from Bio import Phylo
from io import StringIO

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / 'figures'))
from tara_style import apply_tara_style, check_figure_size, save_figure
from palette import (DEEP_OCEAN, TURQUOISE, FOREST_GREEN, COASTAL_BLUE,
                     DESERT_TAN, CLAY, OCEAN_BLUE, SIENNA, OCEAN_CMAP,
                     SEAFOAM, PALE_AQUA, SAND, ABYSS)

apply_tara_style()
matplotlib.rcParams['figure.constrained_layout.use'] = False

DATA = ROOT / 'source_data' / 'novel33041'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Lineage colors ──
LINEAGE_COLORS = {
    'Chlorophyceae': DEEP_OCEAN,
    'Sphaeropleales': OCEAN_BLUE,
    'Trebouxiophyceae': COASTAL_BLUE,
    'Stramenopiles': DESERT_TAN,
    'Haptophyta': CLAY,
    'Euglenozoa': FOREST_GREEN,
    'Charophyta': TURQUOISE,
    'Other': (0.5, 0.5, 0.5),
}

TAXON_LINEAGE = {
    'Dunaliella_Volv': 'Chlorophyceae',
    'Dunaliella_sal': 'Chlorophyceae',
    'Chloromonas': 'Chlorophyceae',
    'Chlamy_reinh': 'Chlorophyceae',
    'Chlamy_spp': 'Chlorophyceae',
    'Other_Chlamy': 'Chlorophyceae',
    'Volvox': 'Chlorophyceae',
    'Colonial_Volv': 'Chlorophyceae',
    'Haematococcus': 'Chlorophyceae',
    'Chlorococcum': 'Chlorophyceae',
    'Scenedesmus': 'Sphaeropleales',
    'Chlorella': 'Trebouxiophyceae',
    'Pelagomonas': 'Stramenopiles',
    'Tisochrysis': 'Haptophyta',
    'Euglenozoa': 'Euglenozoa',
    'Euglena': 'Euglenozoa',
    'Klebsormidium': 'Charophyta',
    'MMETSP_trans': 'Chlorophyceae',
    'Metagenome': 'Other',
}

TAXON_DISPLAY = {
    'Dunaliella_Volv': 'Dunaliella (ROIL/M2)',
    'Dunaliella_sal': 'Dunaliella salina',
    'Chloromonas': 'Chloromonas',
    'Chlamy_reinh': 'C. reinhardtii',
    'Chlamy_spp': 'Chlamydomonas spp.',
    'Other_Chlamy': 'Chlamy. (GCA)',
    'Volvox': 'Volvox carterii',
    'Colonial_Volv': 'Colonial Volv.',
    'Haematococcus': 'Haematococcus',
    'Chlorococcum': 'Chlorococcum',
    'Scenedesmus': 'Scenedesmus/Desm.',
    'Chlorella': 'Chlorella/Parachl.',
    'Pelagomonas': 'Pelagomonas calc.',
    'Tisochrysis': 'Tisochrysis lutea',
    'Euglenozoa': 'Euglenozoa (annot.)',
    'Euglena': 'Euglena viridis',
    'Klebsormidium': 'Klebsormidium nitens',
    'MMETSP_trans': 'MMETSP transcriptomes',
    'Metagenome': 'Metagenome',
}

# Copy numbers per collapsed group (from count matrix analysis)
COPY_NUMBERS = {
    'Dunaliella (ROIL/M2)': 448,
    'Dunaliella salina': 65,
    'Volvox carterii': 117,
    'C. reinhardtii': 14,
    'Chlamydomonas spp.': 25,
    'Chlamy. (GCA)': 23,
    'Colonial Volv.': 47,
    'Chloromonas': 3,
    'Haematococcus': 2,
    'Chlorococcum': 8,
    'Scenedesmus/Desm.': 36,
    'Chlorella/Parachl.': 15,
    'Pelagomonas calc.': 11,
    'Tisochrysis lutea': 5,
    'Euglenozoa (annot.)': 13,
    'Euglena viridis': 1,
    'Klebsormidium nitens': 1,
    'MMETSP transcriptomes': 6,
    'Metagenome': 2,
}


def draw_panel_A(ax):
    """Phylogenetic tree colored by lineage."""
    tree_file = DATA / 'novel33041_reps_tree.treefile'
    tree = Phylo.read(str(tree_file), 'newick')

    def get_color(name):
        if name is None:
            return 'black'
        short = name.split('|')[0]
        lin = TAXON_LINEAGE.get(short, 'Other')
        return LINEAGE_COLORS.get(lin, (0.5, 0.5, 0.5))

    def get_display(name):
        if name is None:
            return ''
        short = name.split('|')[0]
        return TAXON_DISPLAY.get(short, short)

    # Custom label function: display name with lineage color
    def label_func(clade):
        if clade.is_terminal():
            return get_display(clade.name)
        return ''

    # Build label->color mapping
    _label_color_map = {}
    for clade in tree.get_terminals():
        display = get_display(clade.name)
        _label_color_map[display] = get_color(clade.name)

    def color_func(label):
        return _label_color_map.get(label, 'black')

    Phylo.draw(tree, axes=ax, do_show=False, show_confidence=False,
               label_func=label_func, label_colors=color_func)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.set_xlabel('Substitutions/site')

    # Lineage legend — place outside tree area
    used_lineages = []
    for short, lin in TAXON_LINEAGE.items():
        if lin not in [u for u, _ in used_lineages]:
            used_lineages.append((lin, LINEAGE_COLORS[lin]))
    handles = [mpatches.Patch(color=c, label=l) for l, c in used_lineages]
    # Legend drawn externally on panel B to avoid tree overlap
    ax._lineage_handles = handles


def draw_panel_B(ax):
    """ESMFold structure colored by pLDDT — render as per-residue pLDDT profile."""
    pdb_file = DATA / 'protein2_novel33041_esmfold.pdb'
    residues, plddts = [], []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith('ATOM') and ' CA ' in line:
                resnum = int(line[22:26].strip())
                plddt = float(line[60:66].strip()) * 100  # 0-1 scale -> 0-100
                residues.append(resnum)
                plddts.append(plddt)

    residues = np.array(residues)
    plddts = np.array(plddts)

    # Color by pLDDT confidence bands
    for i in range(len(residues) - 1):
        color = DEEP_OCEAN if plddts[i] >= 70 else COASTAL_BLUE if plddts[i] >= 50 else DESERT_TAN if plddts[i] >= 30 else CLAY
        ax.plot(residues[i:i+2], plddts[i:i+2], '-', color=color, lw=1.0)

    ax.axhline(70, color=SIENNA, ls='--', lw=0.5, alpha=0.7)
    ax.text(len(residues) * 0.98, 72, 'pLDDT 70', fontsize=5, ha='right',
            color=SIENNA, va='bottom')

    # Mark conserved cysteines
    seq = 'MGGVRATRVAAWGKCVADAIALARGGLASSITPPTRRLMEQLQARVKRVWELRWGNRWKEVWRLLLHGDKGAGGHGWAWAGGKTCVCGWQPTMDTDAPTRAFQQRAHVFWGCPCAQAVVQCVHERVAGVSVLPVQLWLLGPSPLDVGQCEWGGSEQVKIKTRALLDLSWQDFLPLSPSLPLFLA'
    cys_pos = [i + 1 for i, c in enumerate(seq) if c == 'C']
    trp_pos = [i + 1 for i, c in enumerate(seq) if c == 'W']
    for cp in cys_pos:
        if cp <= len(plddts):
            ax.plot(cp, plddts[cp-1], 'v', color=FOREST_GREEN, ms=3, zorder=6)
    for wp in trp_pos:
        if wp <= len(plddts):
            ax.plot(wp, plddts[wp-1], '^', color=CLAY, ms=2.5, zorder=6)

    ax.set_xlabel('Residue position')
    ax.set_ylabel('pLDDT')
    ax.set_ylim(15, 85)
    ax.set_xlim(0, len(residues) + 5)

    # Cys/Trp markers in annotation text instead of legend
    ax.text(0.98, 0.98, 'v Cys   ^ Trp', transform=ax.transAxes,
            fontsize=4, ha='right', va='top', color=(0.4, 0.4, 0.4))


def draw_panel_C(ax):
    """Contig context diagram: 3 proteins on Dunaliella contig 33041."""
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 2.5)

    proteins = [
        {'x': 0.5, 'w': 2.5, 'label': 'Protein 1\n(149 aa)', 'sublabel': 'No database match',
         'color': PALE_AQUA, 'edgecolor': DEEP_OCEAN},
        {'x': 3.5, 'w': 3.0, 'label': 'Protein 2\n(184 aa)', 'sublabel': 'NOVEL_33041\n7 Cys, 12 Trp',
         'color': DEEP_OCEAN, 'edgecolor': ABYSS},
        {'x': 7.0, 'w': 2.5, 'label': 'Protein 3\n(141 aa)', 'sublabel': 'Antifreeze type I\n(IPR000104)',
         'color': TURQUOISE, 'edgecolor': FOREST_GREEN},
    ]

    for p in proteins:
        rect = FancyBboxPatch((p['x'], 0.2), p['w'], 1.2,
                              boxstyle='round,pad=0.1',
                              facecolor=p['color'], edgecolor=p['edgecolor'],
                              linewidth=0.5)
        ax.add_patch(rect)
        txt_color = 'white' if p['color'] == DEEP_OCEAN else 'black'
        ax.text(p['x'] + p['w']/2, 0.95, p['label'], ha='center', va='center',
                fontsize=5, fontweight='normal', color=txt_color)
        ax.text(p['x'] + p['w']/2, -0.3, p['sublabel'], ha='center', va='top',
                fontsize=4.5, color=(0.3, 0.3, 0.3), style='italic')

    # Contig line
    ax.plot([0.3, 9.7], [0.8, 0.8], '-', color=(0.7, 0.7, 0.7), lw=2, zorder=0)

    # Arrow heads showing reading frame direction
    for p in proteins:
        cx = p['x'] + p['w'] - 0.15
        cy = 0.8
        ax.annotate('', xy=(cx + 0.2, cy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color=p['edgecolor'], lw=0.8))

    ax.text(5.0, 2.1, 'Dunaliella ROIL contig 33041', ha='center', fontsize=6,
            fontweight='bold')
    ax.text(5.0, 1.75, 'All 3 proteins: 0 BLAST hits in nr (128M seqs)',
            ha='center', fontsize=5, color=(0.4, 0.4, 0.4))

    ax.set_axis_off()


def draw_panel_D(ax):
    """Copy-number bar chart per organism."""
    # Sort by copy number, top 12
    sorted_items = sorted(COPY_NUMBERS.items(), key=lambda x: -x[1])[:12]
    labels = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    # Map to lineage colors
    display_to_lineage = {}
    for short, display in TAXON_DISPLAY.items():
        display_to_lineage[display] = TAXON_LINEAGE.get(short, 'Other')

    colors = []
    for lab in labels:
        lin = display_to_lineage.get(lab, 'Other')
        colors.append(LINEAGE_COLORS.get(lin, (0.5, 0.5, 0.5)))

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, edgecolor='none', height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=5)
    ax.invert_yaxis()
    ax.set_xlabel('Domain copies per genome')
    ax.set_xlim(0, max(values) * 1.15)

    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.02, i, str(v), va='center', fontsize=5)


def draw_panel_E(ax):
    """Conserved position summary from alignment."""
    # Parse alignment to find conservation at Cys/Trp positions
    aln_file = DATA / 'novel33041_reps_trimmed_fixed.afa'
    seqs_dict = {}
    current = None
    with open(aln_file) as f:
        for line in f:
            if line.startswith('>'):
                current = line[1:].strip().split('|')[0]
                seqs_dict[current] = ''
            elif current:
                seqs_dict[current] += line.strip()

    if not seqs_dict:
        ax.text(0.5, 0.5, 'No alignment', ha='center', transform=ax.transAxes)
        return

    aln_matrix = np.array([list(s) for s in seqs_dict.values()])
    n_seq, n_col = aln_matrix.shape

    # Conservation per column
    conservation = []
    for j in range(n_col):
        col = aln_matrix[:, j]
        non_gap = col[col != '-']
        if len(non_gap) == 0:
            conservation.append(0)
        else:
            from collections import Counter
            counts = Counter(non_gap)
            most_common_frac = counts.most_common(1)[0][1] / len(non_gap)
            conservation.append(most_common_frac)

    conservation = np.array(conservation)

    # Find which columns are Cys or Trp in the reference (first seq)
    ref = list(seqs_dict.values())[0]
    cys_cols = [j for j, c in enumerate(ref) if c == 'C']
    trp_cols = [j for j, c in enumerate(ref) if c == 'W']

    # Smooth conservation with rolling window
    window = 5
    smoothed = np.convolve(conservation, np.ones(window)/window, mode='same')

    ax.fill_between(range(n_col), smoothed, alpha=0.25, color=COASTAL_BLUE, lw=0)
    ax.plot(range(n_col), smoothed, '-', color=DEEP_OCEAN, lw=0.6)

    # Mark conserved Cys and Trp
    for j in cys_cols:
        if conservation[j] > 0.5:
            ax.plot(j, min(conservation[j] + 0.05, 1.0), 'v', color=FOREST_GREEN,
                    ms=3.5, zorder=6, clip_on=False)
    for j in trp_cols:
        if conservation[j] > 0.5:
            ax.plot(j, min(conservation[j] + 0.05, 1.0), '^', color=CLAY,
                    ms=2.5, zorder=6, clip_on=False)

    ax.set_xlabel('Alignment position')
    ax.set_ylabel('Cons.')
    ax.set_ylim(0, 1.10)
    ax.set_xlim(0, n_col)

    n_conserved_cys = len([j for j in cys_cols if conservation[j] > 0.5])
    n_conserved_trp = len([j for j in trp_cols if conservation[j] > 0.5])
    ax.plot([], [], 'v', color=FOREST_GREEN, ms=3, label=f'Cys (n={n_conserved_cys})')
    ax.plot([], [], '^', color=CLAY, ms=2.5, label=f'Trp (n={n_conserved_trp})')
    ax.legend(fontsize=4, loc='upper left', frameon=False, borderpad=0.2)


# ══════════════════════════════════════════════════════════════
# BUILD FIGURE
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(7.0, 8.0), layout=None)

outer = gridspec.GridSpec(3, 1, figure=fig,
                          height_ratios=[0.45, 0.20, 0.35],
                          left=0.12, right=0.97,
                          top=0.97, bottom=0.05,
                          hspace=0.45)

# Row 1: A (tree) + B (structure)
gs_r1 = outer[0].subgridspec(1, 2, wspace=0.30, width_ratios=[1.7, 1])
ax_a = fig.add_subplot(gs_r1[0, 0])
ax_b = fig.add_subplot(gs_r1[0, 1])

# Row 2: C (contig) — full width
gs_r2 = outer[1].subgridspec(1, 1)
ax_c = fig.add_subplot(gs_r2[0, 0])

# Row 3: D (copy number) + E (conservation)
gs_r3 = outer[2].subgridspec(1, 2, wspace=0.40, width_ratios=[1.2, 1])
ax_d = fig.add_subplot(gs_r3[0, 0])
ax_e = fig.add_subplot(gs_r3[0, 1])

print("Drawing panels...")
draw_panel_A(ax_a)
draw_panel_B(ax_b)

# Add lineage legend to panel B lower-left (avoid tree overlap)
if hasattr(ax_a, '_lineage_handles'):
    ax_b.legend(handles=ax_a._lineage_handles, fontsize=4, loc='lower left',
                frameon=False, handlelength=0.8, handleheight=0.6, borderpad=0.2,
                bbox_to_anchor=(0.0, 0.0))
draw_panel_C(ax_c)
draw_panel_D(ax_d)
draw_panel_E(ax_e)

# Panel labels
panel_axes = [ax_a, ax_b, ax_c, ax_d, ax_e]
for ax, letter in zip(panel_axes, 'ABCDE'):
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes,
            fontsize=6, fontweight='bold', va='top', ha='left')

# Spine cleanup
for ax in [ax_a, ax_b, ax_d, ax_e]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── Validate ──
def validate_figure(fig):
    renderer = fig.canvas.get_renderer()
    issues = []
    all_texts = list(fig.texts)
    for ax in fig.get_axes():
        all_texts.extend(ax.texts)
        all_texts.append(ax.title)
        all_texts.append(ax.xaxis.label)
        all_texts.append(ax.yaxis.label)
        all_texts.extend(ax.get_xticklabels())
        all_texts.extend(ax.get_yticklabels())
    all_texts = [t for t in all_texts if t.get_text().strip()]
    bboxes = []
    for t in all_texts:
        try:
            bb = t.get_window_extent(renderer=renderer)
            if bb.width > 0 and bb.height > 0:
                bboxes.append((t, bb))
        except Exception:
            pass
    for i, (t1, bb1) in enumerate(bboxes):
        for t2, bb2 in bboxes[i+1:]:
            if bb1.overlaps(bb2):
                issues.append(f"OVERLAP: '{t1.get_text()[:25]}' x '{t2.get_text()[:25]}'")
    for t in all_texts:
        size = t.get_fontsize()
        if size > 6.5:
            issues.append(f"FONT SIZE {size}pt on '{t.get_text()[:20]}'")
    if issues:
        print(f"VALIDATION: {len(issues)} issue(s):")
        for issue in issues[:10]:
            print(f"  x {issue}")
    else:
        print("VALIDATION PASSED")
    return issues

print("\n=== Validation ===")
validate_figure(fig)

# ── Export ──
check_figure_size(fig, 'cell', 'double')
name = f'Figure7_novel33041_{TIMESTAMP}'
save_figure(fig, name, output_dir=str(ROOT / 'figures'))
plt.close(fig)
print(f"\nDone: figures/{name}")
