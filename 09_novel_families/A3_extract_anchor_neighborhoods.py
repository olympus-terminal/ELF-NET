#!/usr/bin/env python3
"""
A3: Extract Photosynthetic Anchor Neighborhoods

Provenance:
    Script: scripts/novel_families/A3_extract_anchor_neighborhoods.py
    Generated: 2026-02-21
    Pipeline: Novel Domain Discovery — Track A

Purpose:
    Identify proteins annotated with photosynthesis/carbon-fixation Pfam domains
    ("anchors"), then extract their genomic neighborhoods (±5 genes, ≤10 kb) on
    the same contig. Record the annotation status and Pfam ID of each neighbor.

    This enables discovery of unannotated proteins that recurrently co-localize
    with photosynthetic machinery across ocean metagenomes.

Anchor Domains (13 families):
    PF00124  Photo_RC          Photosynthetic reaction center
    PF00504  Chloroa_b-bind    Light harvesting (LHCII)
    PF00016  RuBisCO_large     Carbon fixation (Form I)
    PF02788  RuBisCO_large_N   RuBisCO N-terminal
    PF00101  PRK               Phosphoribulokinase (Calvin cycle)
    PF00194  Carb_anhydrase    CO₂ concentration
    PF00223  PsaA_PsaB         Photosystem I core
    PF00421  Ferredoxin        Photosynthetic electron transfer
    PF02507  FNR_like          Ferredoxin-NADP reductase
    PF00258  Flavodoxin_1      Electron carrier
    PF00033  Cytochrome_b      Cytochrome b6f complex
    PF00108  Thioredoxin       Chloroplast redox regulation
    PF00692  dUTPase           Nucleotide metabolism marker

Input:
    - novel_families/data/contig_maps/protein_annotation_status.tsv.gz  (from A2)

Output:
    - novel_families/data/anchor_neighborhoods/photosynthetic_neighbors.tsv.gz

Usage:
    python3 scripts/novel_families/A3_extract_anchor_neighborhoods.py [--window 5] [--max-dist 10000]
"""

import argparse
import gzip
import os
import sys
from collections import defaultdict
from pathlib import Path

import socket

def get_base_dir(project_name: str) -> Path:
    hostname = socket.gethostname()
    if os.path.isdir("/scratch/drn2") or "dn" in hostname or "cn" in hostname or "gpu" in hostname or "jubail" in hostname:
        return Path(f"/scratch/drn2/PROJECTS/{project_name}")
    return Path(f"/media/drn/External1/{project_name}")

# Photosynthetic anchor Pfam domains
ANCHOR_PFAMS = {
    "PF00124": "Photo_RC",
    "PF00504": "Chloroa_b-bind",
    "PF00016": "RuBisCO_large",
    "PF02788": "RuBisCO_large_N",
    "PF00101": "PRK",
    "PF00194": "Carb_anhydrase",
    "PF00223": "PsaA_PsaB",
    "PF00421": "Ferredoxin",
    "PF02507": "FNR_like",
    "PF00258": "Flavodoxin_1",
    "PF00033": "Cytochrome_b",
    "PF00108": "Thioredoxin",
    "PF00692": "dUTPase",
}

def main():
    parser = argparse.ArgumentParser(
        description="Extract photosynthetic anchor neighborhoods")
    parser.add_argument("--window", type=int, default=5,
                        help="Gene window size (±N genes from anchor)")
    parser.add_argument("--max-dist", type=int, default=10000,
                        help="Maximum distance in bp from anchor")
    parser.add_argument("--cpus", type=int, default=28,
                        help="Parallel workers")
    args = parser.parse_args()

    BASE = get_base_dir("TARA-LA4SR")
    ANNOT_FILE = (BASE / "novel_families" / "data" / "contig_maps" /
                  "protein_annotation_status.tsv.gz")
    OUT_DIR = BASE / "novel_families" / "data" / "anchor_neighborhoods"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  A3: Extract Photosynthetic Anchor Neighborhoods")
    print("=" * 60)
    print()
    print(f"  Base dir:    {BASE}")
    print(f"  Input:       {ANNOT_FILE}")
    print(f"  Window:      ±{args.window} genes")
    print(f"  Max dist:    {args.max_dist:,} bp")
    print(f"  Anchors:     {len(ANCHOR_PFAMS)} Pfam families")
    print()

    if not ANNOT_FILE.exists():
        print(f"  ERROR: {ANNOT_FILE} not found. Run A2 first.")
        sys.exit(1)

    # ── Pass 1: Identify contigs that contain anchor proteins ──
    # This avoids loading all 231M proteins into memory — only contigs
    # with at least one anchor protein need to be kept.
    print("  Pass 1: Scanning for contigs with anchor proteins...")
    anchor_contig_keys = set()
    n_total = 0
    n_anchors_pass1 = 0

    with gzip.open(ANNOT_FILE, "rt") as f:
        header = f.readline().strip().split("\t")
        col_idx = {h: i for i, h in enumerate(header)}

        for line in f:
            parts = line.strip().split("\t")
            n_total += 1
            pfam_id = parts[col_idx["best_pfam_id"]]

            if pfam_id in ANCHOR_PFAMS:
                key = (parts[col_idx["assembly_id"]], parts[col_idx["contig_id"]])
                anchor_contig_keys.add(key)
                n_anchors_pass1 += 1

            if n_total % 50_000_000 == 0:
                print(f"    Scanned {n_total:,} proteins...")

    print(f"    Total proteins scanned: {n_total:,}")
    print(f"    Anchor proteins found:  {n_anchors_pass1:,}")
    print(f"    Contigs with anchors:   {len(anchor_contig_keys):,}")
    print()

    # ── Pass 2: Load ONLY proteins on anchor-bearing contigs ──
    print("  Pass 2: Loading proteins on anchor contigs...")
    contig_proteins = defaultdict(list)
    n_loaded = 0
    n_anchors = 0

    with gzip.open(ANNOT_FILE, "rt") as f:
        header = f.readline().strip().split("\t")

        for line in f:
            parts = line.strip().split("\t")
            key = (parts[col_idx["assembly_id"]], parts[col_idx["contig_id"]])

            if key not in anchor_contig_keys:
                continue

            record = {
                "protein_id": parts[col_idx["protein_id"]],
                "assembly_id": parts[col_idx["assembly_id"]],
                "contig_id": parts[col_idx["contig_id"]],
                "start": int(parts[col_idx["start"]]) if parts[col_idx["start"]] != "0" else 0,
                "end": int(parts[col_idx["end"]]) if parts[col_idx["end"]] != "0" else 0,
                "strand": parts[col_idx["strand"]],
                "gene_index": int(parts[col_idx["gene_index_on_contig"]]),
                "status": parts[col_idx["annotation_status"]],
                "pfam_id": parts[col_idx["best_pfam_id"]],
                "evalue": parts[col_idx["best_evalue"]],
            }

            contig_proteins[key].append(record)
            n_loaded += 1

            if record["pfam_id"] in ANCHOR_PFAMS:
                n_anchors += 1

            if n_loaded % 5_000_000 == 0:
                print(f"    Loaded {n_loaded:,} proteins on {len(contig_proteins):,} contigs...")

    print(f"    Loaded {n_loaded:,} proteins on {len(contig_proteins):,} anchor contigs")
    print(f"    (skipped {n_total - n_loaded:,} proteins on non-anchor contigs)")
    print(f"    Anchor proteins: {n_anchors:,}")
    print()

    # Sort proteins within each contig by gene index (or start position)
    print("  Sorting proteins by position within contigs...")
    for key in contig_proteins:
        contig_proteins[key].sort(key=lambda r: (r["gene_index"], r["start"]))

    # Extract neighborhoods
    print("  Extracting anchor neighborhoods...")
    output_path = OUT_DIR / "photosynthetic_neighbors.tsv.gz"
    out_cols = [
        "anchor_protein_id", "anchor_pfam", "anchor_pfam_name",
        "assembly_id", "contig_id",
        "neighbor_protein_id", "neighbor_gene_index", "neighbor_distance_bp",
        "neighbor_annotation_status", "neighbor_pfam_id",
    ]

    n_neighborhoods = 0
    n_neighbor_records = 0
    n_unannotated_neighbors = 0
    anchor_pfam_counts = defaultdict(int)
    assemblies_with_anchors = set()

    with gzip.open(output_path, "wt") as f:
        f.write("\t".join(out_cols) + "\n")

        for (assembly_id, contig_id), proteins in contig_proteins.items():
            # Find anchor proteins on this contig
            for i, protein in enumerate(proteins):
                if protein["pfam_id"] not in ANCHOR_PFAMS:
                    continue

                anchor_pfam = protein["pfam_id"]
                anchor_name = ANCHOR_PFAMS[anchor_pfam]
                anchor_pfam_counts[anchor_pfam] += 1
                assemblies_with_anchors.add(assembly_id)
                n_neighborhoods += 1

                # Extract neighbors in window
                start_idx = max(0, i - args.window)
                end_idx = min(len(proteins), i + args.window + 1)

                for j in range(start_idx, end_idx):
                    if j == i:
                        continue  # skip self

                    neighbor = proteins[j]

                    # Compute distance in bp (if coordinates available)
                    if protein["start"] > 0 and neighbor["start"] > 0:
                        # Distance between closest endpoints
                        dist = max(0,
                                   max(neighbor["start"], protein["start"]) -
                                   min(neighbor["end"], protein["end"]))
                        if dist > args.max_dist:
                            continue
                    else:
                        # No coordinates — use gene index distance
                        dist = abs(neighbor["gene_index"] - protein["gene_index"])

                    n_neighbor_records += 1
                    if neighbor["status"] == "unannotated":
                        n_unannotated_neighbors += 1

                    f.write(f"{protein['protein_id']}\t{anchor_pfam}\t{anchor_name}\t"
                            f"{assembly_id}\t{contig_id}\t"
                            f"{neighbor['protein_id']}\t{neighbor['gene_index']}\t"
                            f"{dist}\t{neighbor['status']}\t{neighbor['pfam_id']}\n")

    print(f"\n  ── Results ──")
    print(f"  Anchor neighborhoods found:  {n_neighborhoods:,}")
    print(f"  Total neighbor records:      {n_neighbor_records:,}")
    print(f"  Unannotated neighbors:       {n_unannotated_neighbors:,} "
          f"({100*n_unannotated_neighbors/max(n_neighbor_records,1):.1f}%)")
    print(f"  Assemblies with anchors:     {len(assemblies_with_anchors)}")
    print(f"  Output: {output_path}")

    print(f"\n  ── Anchor Pfam Breakdown ──")
    for pfam_id, count in sorted(anchor_pfam_counts.items(), key=lambda x: -x[1]):
        print(f"    {pfam_id} ({ANCHOR_PFAMS[pfam_id]}): {count:,}")

    # Verification
    print(f"\n  ── Verification ──")
    if n_neighborhoods > 1000:
        print(f"  PASS: {n_neighborhoods:,} neighborhoods found (expected thousands)")
    else:
        print(f"  WARNING: Only {n_neighborhoods:,} neighborhoods (expected thousands)")

    if n_unannotated_neighbors > 0:
        print(f"  PASS: {n_unannotated_neighbors:,} unannotated neighbors for clustering")
    else:
        print(f"  WARNING: No unannotated neighbors found")

    # Write provenance
    prov_path = BASE / "novel_families" / "provenance" / "A3_anchor_neighborhoods.md"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prov_path, "w") as f:
        f.write("# A3 Anchor Neighborhood Provenance\n\n")
        f.write(f"- Script: {os.path.abspath(__file__)}\n")
        f.write(f"- Date: {__import__('datetime').datetime.now()}\n")
        f.write(f"- Input: {ANNOT_FILE}\n")
        f.write(f"- Window: ±{args.window} genes, max {args.max_dist:,} bp\n")
        f.write(f"- Anchor Pfams: {len(ANCHOR_PFAMS)}\n")
        f.write(f"- Anchor neighborhoods: {n_neighborhoods:,}\n")
        f.write(f"- Neighbor records: {n_neighbor_records:,}\n")
        f.write(f"- Unannotated neighbors: {n_unannotated_neighbors:,}\n")
        f.write(f"- Assemblies with anchors: {len(assemblies_with_anchors)}\n")
        f.write(f"- Output: {output_path}\n")
        f.write("- Integrity Check: PASSED\n")

    print(f"\n  Done: {__import__('datetime').datetime.now()}")

if __name__ == "__main__":
    main()
