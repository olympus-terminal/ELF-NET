#!/usr/bin/env python3
"""
A10: Contig Neighborhood Arrow Diagrams

Provenance:
    Script: scripts/novel_families/A10_neighborhood_diagrams.py
    Generated: 2026-02-21
    Pipeline: Novel Domain Discovery — Track A

Purpose:
    For top 5-10 most recurrent novel families, draw arrow diagrams showing
    3-5 representative contig neighborhoods. Color scheme:
    - Green: anchor gene (photosynthetic)
    - Gray: annotated neighbor (known Pfam)
    - Orange/red: novel family member (unannotated)

Input:
    - novel_families/data/anchor_neighborhoods/photosynthetic_neighbors.tsv.gz (A3)
    - novel_families/data/novel_families_filtered.tsv (A6)
    - novel_families/data/mmseqs_clusters/cluster_membership.tsv (A5)
    - novel_families/results/track_a/novel_family_xgboost_forward_r2.tsv (A8)

Output:
    - novel_families/figures/contig_neighborhood_diagrams.pdf

Usage:
    python3 scripts/novel_families/A10_neighborhood_diagrams.py [--top-n 8]
"""

import argparse
import gzip
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import socket

def get_base_dir(project_name: str) -> Path:
    hostname = socket.gethostname()
    if os.path.isdir("/scratch/drn2") or "dn" in hostname or "cn" in hostname or "gpu" in hostname or "jubail" in hostname:
        return Path(f"/scratch/drn2/PROJECTS/{project_name}")
    return Path(f"/media/drn/External1/{project_name}")

def main():
    parser = argparse.ArgumentParser(description="Draw contig neighborhood diagrams")
    parser.add_argument("--top-n", type=int, default=8,
                        help="Number of top families to visualize")
    parser.add_argument("--examples-per-family", type=int, default=4,
                        help="Number of example contigs per family")
    args = parser.parse_args()

    BASE = get_base_dir("TARA-LA4SR")
    RESULTS_DIR = BASE / "novel_families" / "results" / "track_a"
    FIG_DIR = BASE / "novel_families" / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  A10: Contig Neighborhood Diagrams")
    print("=" * 60)
    print()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import FancyArrow
    except ImportError:
        print("  ERROR: matplotlib required")
        sys.exit(1)

    # Load R² results to select top families
    r2_path = RESULTS_DIR / "novel_family_xgboost_forward_r2.tsv"
    if not r2_path.exists():
        print(f"  ERROR: {r2_path} not found. Run A8 first.")
        sys.exit(1)

    import pandas as pd
    r2_df = pd.read_csv(r2_path, sep="\t")
    r2_df = r2_df.dropna(subset=["r2_overall"]).sort_values("r2_overall", ascending=False)
    top_families = r2_df.head(args.top_n)["family_id"].tolist()
    top_r2 = dict(zip(r2_df["family_id"], r2_df["r2_overall"]))

    print(f"  Top {args.top_n} families by R²:")
    for fid in top_families:
        print(f"    {fid}: R² = {top_r2[fid]:.4f}")
    print()

    # Load filtered families → representative map
    families_path = BASE / "novel_families" / "data" / "novel_families_filtered.tsv"
    fam_df = pd.read_csv(families_path, sep="\t")
    rep_to_family = dict(zip(fam_df["representative_protein_id"], fam_df["family_id"]))
    family_to_rep = dict(zip(fam_df["family_id"], fam_df["representative_protein_id"]))

    # Load cluster membership → protein → family_id
    membership_path = BASE / "novel_families" / "data" / "mmseqs_clusters" / "cluster_membership.tsv"
    protein_to_family = {}
    with open(membership_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                if rep in rep_to_family:
                    protein_to_family[member] = rep_to_family[rep]

    # Load neighborhood data
    print("  Loading neighborhood data...")
    neighbor_path = (BASE / "novel_families" / "data" / "anchor_neighborhoods" /
                     "photosynthetic_neighbors.tsv.gz")

    neighborhoods = defaultdict(list)  # family_id -> [(contig_key, records)]
    contig_records = defaultdict(list)  # (anchor_id, contig_id) -> [neighbor_records]

    with gzip.open(neighbor_path, "rt") as f:
        header = f.readline().strip().split("\t")
        col_idx = {h: i for i, h in enumerate(header)}

        for line in f:
            parts = line.strip().split("\t")
            neighbor_id = parts[col_idx["neighbor_protein_id"]]
            fam = protein_to_family.get(neighbor_id)
            if fam and fam in top_families:
                anchor_id = parts[col_idx["anchor_protein_id"]]
                contig_id = parts[col_idx["contig_id"]]
                assembly_id = parts[col_idx["assembly_id"]]
                key = (anchor_id, contig_id, assembly_id)

                record = {
                    "anchor_id": anchor_id,
                    "anchor_pfam": parts[col_idx["anchor_pfam"]],
                    "anchor_name": parts[col_idx["anchor_pfam_name"]],
                    "neighbor_id": neighbor_id,
                    "gene_index": int(parts[col_idx["neighbor_gene_index"]]),
                    "status": parts[col_idx["neighbor_annotation_status"]],
                    "pfam_id": parts[col_idx["neighbor_pfam_id"]],
                    "family_id": fam,
                    "assembly_id": assembly_id,
                    "contig_id": contig_id,
                }
                contig_records[key].append(record)
                neighborhoods[fam].append(key)

    print(f"  Loaded neighborhoods for {len(neighborhoods)} families")

    # Select representative contigs for each family
    family_examples = {}
    for fam in top_families:
        if fam not in neighborhoods:
            continue
        # Get unique contig contexts
        unique_keys = list(set(neighborhoods[fam]))
        # Sort by number of family members visible
        unique_keys.sort(key=lambda k: len(contig_records[k]), reverse=True)
        family_examples[fam] = unique_keys[:args.examples_per_family]

    # ── Draw diagrams ──
    print("  Drawing diagrams...")

    n_families_to_draw = sum(1 for f in top_families if f in family_examples)
    fig_height = 1.5 * n_families_to_draw * args.examples_per_family + 2
    fig, axes = plt.subplots(n_families_to_draw, 1,
                             figsize=(14, fig_height),
                             squeeze=False)

    colors = {
        "anchor": "#2ca02c",        # green
        "annotated": "#aaaaaa",     # gray
        "novel_member": "#d62728",  # red/orange
    }

    row = 0
    for fam in top_families:
        if fam not in family_examples:
            continue

        ax = axes[row, 0]
        ax.set_title(f"{fam} (R² = {top_r2.get(fam, 0):.3f}, "
                     f"n_members = {fam_df.loc[fam_df['family_id']==fam, 'n_members'].values[0]})",
                     fontsize=11, fontweight="bold", loc="left")

        y_pos = 0
        for key in family_examples[fam]:
            records = contig_records[key]
            anchor_id = key[0]
            assembly_id = key[2]

            # Build gene list for this contig context
            # Find all neighbors + the anchor
            all_genes = []

            # Add anchor gene
            if records:
                anchor_rec = records[0]
                # Estimate anchor gene index from neighbors
                neighbor_indices = [r["gene_index"] for r in records]
                anchor_index = int(np.median(neighbor_indices))
                all_genes.append({
                    "id": anchor_id,
                    "index": anchor_index,
                    "type": "anchor",
                    "label": anchor_rec["anchor_name"],
                })

            # Add neighbor genes
            for r in records:
                gene_type = "novel_member" if r["family_id"] == fam else "annotated"
                if r["status"] == "annotated" and r["family_id"] != fam:
                    gene_type = "annotated"
                label = fam if gene_type == "novel_member" else r["pfam_id"]
                if label == "NA":
                    label = ""
                all_genes.append({
                    "id": r["neighbor_id"],
                    "index": r["gene_index"],
                    "type": gene_type,
                    "label": label,
                })

            # Sort by gene index
            all_genes.sort(key=lambda g: g["index"])

            # Draw arrows
            for i, gene in enumerate(all_genes):
                x = i * 1.5
                color = colors.get(gene["type"], "#cccccc")

                # Draw gene arrow
                arrow = patches.FancyArrowPatch(
                    (x, y_pos), (x + 1.0, y_pos),
                    arrowstyle="->,head_length=0.3,head_width=0.3",
                    linewidth=2, color=color,
                    mutation_scale=15,
                )
                ax.add_patch(arrow)

                # Background rectangle
                rect = patches.Rectangle(
                    (x, y_pos - 0.15), 1.0, 0.3,
                    facecolor=color, alpha=0.3, edgecolor=color)
                ax.add_patch(rect)

                # Label
                if gene["label"]:
                    label = gene["label"][:12]
                    ax.text(x + 0.5, y_pos + 0.25, label,
                            ha="center", va="bottom", fontsize=6,
                            rotation=30)

            # Assembly label
            ax.text(-0.5, y_pos, assembly_id[:20],
                    ha="right", va="center", fontsize=7, color="gray")

            y_pos -= 0.8

        # Format axes
        ax.set_xlim(-3, args.examples_per_family * 10 * 1.5)
        ax.set_ylim(y_pos - 0.5, 0.8)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        row += 1

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors["anchor"], lw=4, label="Photosynthetic anchor"),
        Line2D([0], [0], color=colors["annotated"], lw=4, label="Known Pfam domain"),
        Line2D([0], [0], color=colors["novel_member"], lw=4, label="Novel family member"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=10, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)

    fig_path = FIG_DIR / "contig_neighborhood_diagrams.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n  Written: {fig_path}")
    print(f"  Done: {__import__('datetime').datetime.now()}")

if __name__ == "__main__":
    main()
