#!/usr/bin/env python3
"""
Figure S9: Structural Characterization of Top Novel Domains
# --------------------------------------------------------------------------

Three-panel supplementary figure:

Panel A — Per-residue pLDDT profiles for all 10 predicted structures
          (line plots, colored by source type)
Panel B — Foldseek structural homology summary (table/heatmap of best
          TM-scores per domain × database)
Panel C — Domain selection summary: prevalence vs max |rho|, sized by
          cluster size, colored by source type

Data:
  source_data/dark_proteome/top10_selection_summary.tsv  (rsynced from HPC)
  source_data/dark_proteome/plddt_summary.tsv
  source_data/dark_proteome/foldseek_best_hits.tsv
  source_data/dark_proteome/plddt_per_residue/  (one .tsv per domain)

Created: 2026-03-20
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

_palette_dir = str(Path(__file__).resolve().parent)
if _palette_dir not in sys.path:
    sys.path.insert(0, _palette_dir)

from palette import (
    DEEP_OCEAN, TURQUOISE, DESERT_TAN, OCEAN_BLUE, COASTAL_BLUE,
    FOREST_GREEN, CLAY, CLOUD_WHITE, PALE_AQUA, SAND, SIENNA,
    EARTH_CATEGORICAL, OCEAN_CMAP,
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

# ── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "source_data" / "dark_proteome"
PLDDT_DIR = DATA_DIR / "plddt_per_residue"

# Source-type colors
SOURCE_COLORS = {
    'metagenomic': DEEP_OCEAN,
    'green_alga': FOREST_GREEN,
    'red_alga': CLAY,
    'haptophyte': TURQUOISE,
    'diatom': OCEAN_BLUE,
    'dinoflagellate': DESERT_TAN,
    'ncbi_genome': COASTAL_BLUE,
    'other': (0.6, 0.6, 0.6),
}

def shorten_domain_name(name: str, max_len: int = 30) -> str:
    """Create a readable short label from a NOVEL_... domain ID."""
    s = name.replace("NOVEL_", "")
    # If it has a recognizable organism, extract that
    for org in ["Dunaliella", "Chlamydomonas", "Chondrus", "Chrysochromulina",
                "Characiochloris", "Micromonas", "Emiliania", "Galdieria"]:
        if org in s:
            # Extract org + suffix
            idx = s.index(org)
            short = s[idx:idx+len(org)+15]
            return short[:max_len]
    # Metagenomic: use ERZ or first part
    if "ERZ" in s:
        return s[:max_len]
    if "KAL" in s:
        return s[:max_len]
    if "GCA_" in s:
        return s[:max_len]
    return s[:max_len]

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load data ────────────────────────────────────────────────────────────
    selection_file = DATA_DIR / "top10_selection_summary.tsv"
    plddt_file = DATA_DIR / "plddt_summary.tsv"
    foldseek_file = DATA_DIR / "foldseek_best_hits.tsv"

    for f in [selection_file, plddt_file, foldseek_file]:
        if not f.exists():
            print(f"WARNING: {f} not found — using placeholder data")

    # Selection summary
    if selection_file.exists():
        sel_df = pd.read_csv(selection_file, sep="\t")
    else:
        print("ERROR: top10_selection_summary.tsv required")
        sys.exit(1)

    # pLDDT summary
    if plddt_file.exists():
        plddt_df = pd.read_csv(plddt_file, sep="\t")
    else:
        plddt_df = None

    # Foldseek best hits
    if foldseek_file.exists():
        foldseek_df = pd.read_csv(foldseek_file, sep="\t")
    else:
        foldseek_df = None

    # ── Create figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.5, 6.5))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[1.2, 1],
                           hspace=0.35, wspace=0.35)

    # Panel A: pLDDT profiles (top, spans full width)
    ax_a = fig.add_subplot(gs[0, :])

    # Panel B: Foldseek summary table (bottom left)
    ax_b = fig.add_subplot(gs[1, 0])

    # Panel C: Prevalence vs rho scatter (bottom right)
    ax_c = fig.add_subplot(gs[1, 1])

    # ── Panel A: Per-residue pLDDT profiles ──────────────────────────────────
    ax_a.text(-0.02, 1.05, 'A', transform=ax_a.transAxes, fontsize=8,
              fontweight='bold', va='top', ha='right')

    if plddt_df is not None:
        # Try to load per-residue profiles
        per_res_loaded = False
        if PLDDT_DIR.exists():
            for i, row in sel_df.iterrows():
                domain = row['domain']
                safe_name = domain.replace("/", "_").replace(" ", "_")
                plddt_path = PLDDT_DIR / f"{safe_name}_plddt.tsv"
                if plddt_path.exists():
                    per_res_loaded = True
                    residue_plddt = pd.read_csv(plddt_path, sep="\t")
                    color = SOURCE_COLORS.get(row.get('source', 'other'), (0.5, 0.5, 0.5))
                    label = shorten_domain_name(domain, 25)
                    ax_a.plot(residue_plddt['residue'], residue_plddt['plddt'],
                              color=color, linewidth=0.5, alpha=0.8, label=label)

        if not per_res_loaded:
            # Fall back to bar chart of mean pLDDT per domain
            merged = sel_df.merge(plddt_df, on='domain', how='left')
            colors = [SOURCE_COLORS.get(row.get('source', 'other'), (0.5, 0.5, 0.5))
                      for _, row in merged.iterrows()]
            labels = [shorten_domain_name(d, 20) for d in merged['domain']]
            x = np.arange(len(merged))
            bars = ax_a.bar(x, merged['mean_plddt'], color=colors, edgecolor='white',
                            linewidth=0.3, width=0.7)

            # Add confidence bands
            ax_a.axhline(y=70, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax_a.axhline(y=50, color='gray', linestyle=':', linewidth=0.5, alpha=0.4)
            ax_a.text(len(merged) - 0.3, 71, 'confident', fontsize=4.5,
                      color='gray', va='bottom', ha='right')
            ax_a.text(len(merged) - 0.3, 51, 'low confidence', fontsize=4.5,
                      color='gray', va='bottom', ha='right')

            ax_a.set_xticks(x)
            ax_a.set_xticklabels(labels, rotation=45, ha='right', fontsize=5)
            ax_a.set_ylabel('Mean pLDDT')
            ax_a.set_title('ESMFold confidence for top 10 novel domains')
            ax_a.set_ylim(0, 100)

            # Add pct>70 annotation on bars
            for xi, (_, row) in zip(x, merged.iterrows()):
                pct = row.get('pct_above_70', 0)
                if pct > 0 and not np.isnan(pct):
                    ax_a.text(xi, row['mean_plddt'] + 1.5, f'{pct:.0f}%',
                              fontsize=4, ha='center', va='bottom', color='gray')
    else:
        ax_a.text(0.5, 0.5, 'pLDDT data not yet available\n(awaiting ESMFold results)',
                  transform=ax_a.transAxes, ha='center', va='center', fontsize=8,
                  color='gray')
        ax_a.set_title('ESMFold confidence for top 10 novel domains')

    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # ── Panel B: Foldseek summary ────────────────────────────────────────────
    ax_b.text(-0.02, 1.05, 'B', transform=ax_b.transAxes, fontsize=8,
              fontweight='bold', va='top', ha='right')
    ax_b.set_title('Structural homology search (Foldseek)')

    if foldseek_df is not None and len(foldseek_df) > 0:
        # Pivot: domain × database → TM-score
        domains = sel_df['domain'].tolist()
        short_labels = [shorten_domain_name(d, 20) for d in domains]
        databases = ['AFDB', 'PDB']

        # Build TM-score matrix
        tm_matrix = np.zeros((len(domains), len(databases)))
        hit_labels = [[''] * len(databases) for _ in range(len(domains))]

        for i, domain in enumerate(domains):
            for j, db in enumerate(databases):
                match = foldseek_df[(foldseek_df['domain'] == domain) &
                                     (foldseek_df['database'] == db)]
                if len(match) > 0:
                    tm = float(match.iloc[0]['tm_score'])
                    tm_matrix[i, j] = tm
                    target = str(match.iloc[0]['best_target'])
                    if target != 'no_significant_hit':
                        hit_labels[i][j] = f'{tm:.2f}'
                    else:
                        hit_labels[i][j] = '—'
                else:
                    hit_labels[i][j] = '—'

        im = ax_b.imshow(tm_matrix, cmap=OCEAN_CMAP, aspect='auto',
                          vmin=0, vmax=1)
        ax_b.set_xticks([0, 1])
        ax_b.set_xticklabels(['AlphaFold DB\n(Swiss-Prot)', 'PDB'], fontsize=5)
        ax_b.set_yticks(range(len(domains)))
        ax_b.set_yticklabels(short_labels, fontsize=4.5)

        # Add text annotations
        for i in range(len(domains)):
            for j in range(len(databases)):
                ax_b.text(j, i, hit_labels[i][j], ha='center', va='center',
                          fontsize=4.5, color='white' if tm_matrix[i, j] > 0.5 else 'black')

        cbar = plt.colorbar(im, ax=ax_b, shrink=0.6, pad=0.02)
        cbar.set_label('TM-score', fontsize=5)
        cbar.ax.tick_params(labelsize=4.5)
    else:
        ax_b.text(0.5, 0.5, 'Foldseek results\nnot yet available',
                  transform=ax_b.transAxes, ha='center', va='center',
                  fontsize=8, color='gray')
        ax_b.set_xticks([])
        ax_b.set_yticks([])

    # ── Panel C: Prevalence vs max |rho| ─────────────────────────────────────
    ax_c.text(-0.02, 1.05, 'C', transform=ax_c.transAxes, fontsize=8,
              fontweight='bold', va='top', ha='right')

    if 'n_samples' in sel_df.columns and 'max_abs_rho' in sel_df.columns:
        sources = sel_df.get('source', pd.Series(['other'] * len(sel_df)))
        colors = [SOURCE_COLORS.get(s, (0.5, 0.5, 0.5)) for s in sources]
        sizes = sel_df.get('cluster_size', pd.Series([50] * len(sel_df)))
        # Scale sizes for visibility
        size_scaled = np.clip(sizes / sizes.max() * 200, 30, 300)

        ax_c.scatter(sel_df['n_samples'], sel_df['max_abs_rho'],
                     s=size_scaled, c=colors, edgecolors='white',
                     linewidths=0.3, alpha=0.85, zorder=3)

        # Label each point
        for _, row in sel_df.iterrows():
            label = shorten_domain_name(row['domain'], 15)
            ax_c.annotate(label, (row['n_samples'], row['max_abs_rho']),
                          fontsize=3.5, xytext=(3, 3),
                          textcoords='offset points', alpha=0.7)

        ax_c.set_xlabel('Prevalence (n samples)')
        ax_c.set_ylabel('Max |Spearman ρ|')
        ax_c.set_title('Domain selection landscape')
        ax_c.spines['top'].set_visible(False)
        ax_c.spines['right'].set_visible(False)

        # Legend for source types
        seen = set()
        handles = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=SOURCE_COLORS.get(s, (0.5, 0.5, 0.5)),
                                          markersize=4, label=s.replace('_', ' ')))
        ax_c.legend(handles=handles, loc='lower right', fontsize=4.5,
                    frameon=True, fancybox=False, edgecolor='gray',
                    framealpha=0.9)
    else:
        ax_c.text(0.5, 0.5, 'Selection data\nnot yet available',
                  transform=ax_c.transAxes, ha='center', va='center',
                  fontsize=8, color='gray')

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(__file__).resolve().parent / f"FigureS9_structure_predictions_{timestamp}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
