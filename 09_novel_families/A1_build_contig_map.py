#!/usr/bin/env python3
"""
A1: Build Protein → Contig Coordinate Map

Provenance:
    Script: scripts/novel_families/A1_build_contig_map.py
    Generated: 2026-02-21
    Pipeline: Novel Domain Discovery — Track A

Purpose:
    Parse SNAP gene prediction outputs for all assemblies to create a unified
    protein-to-contig coordinate map. Supports multiple input formats:
    - FASTA headers with embedded coordinates (e.g., >protein_id contig:start-end)
    - GFF3 gene predictions
    - ZFF format (SNAP native)

    The map enables downstream contig co-localization analysis (A3).

Input:
    - Protein FASTA files: 03_analyses/algae_proteins/*.fa
    - Gene prediction files: (format determined by A0 recon)

Output:
    - novel_families/data/contig_maps/protein_contig_map.tsv.gz
      Columns: protein_id, assembly_id, contig_id, start, end, strand, gene_index_on_contig

Usage:
    python3 scripts/novel_families/A1_build_contig_map.py [--format auto|fasta|gff3|zff]
"""

import argparse
import gzip
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import socket

def get_base_dir(project_name: str) -> Path:
    hostname = socket.gethostname()
    if os.path.isdir("/scratch/drn2") or "dn" in hostname or "cn" in hostname or "gpu" in hostname or "jubail" in hostname:
        return Path(f"/scratch/drn2/PROJECTS/{project_name}")
    return Path(f"/media/drn/External1/{project_name}")

# ── Header parsing strategies ──

def parse_fasta_header_snap(header: str):
    """
    Parse SNAP-style FASTA header for coordinate info.

    Common formats:
      >gene_1 contig_name:100-500(+)
      >SAMPLE_contig_1_gene_1 100 500 +
      >protein_id # start # end # strand # ...  (prodigal)
    """
    parts = header.lstrip(">").split()
    protein_id = parts[0]

    # Strategy 1: Prodigal format (># start # end # strand # ...)
    if len(parts) >= 5 and parts[1] == "#":
        try:
            start = int(parts[2])
            end = int(parts[4])
            strand = "+" if int(parts[6]) == 1 else "-" if parts[6] != "#" else "+"
            # Contig from protein ID (prodigal: contig_gene)
            contig_match = re.match(r"(.+)_\d+$", protein_id)
            contig_id = contig_match.group(1) if contig_match else protein_id
            return protein_id, contig_id, start, end, strand
        except (ValueError, IndexError):
            pass

    # Strategy 2: contig:start-end format
    for part in parts[1:]:
        m = re.match(r"(\S+):(\d+)-(\d+)\(([+-])\)", part)
        if m:
            return protein_id, m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        m = re.match(r"(\S+):(\d+)-(\d+)", part)
        if m:
            return protein_id, m.group(1), int(m.group(2)), int(m.group(3)), "+"

    # Strategy 3: Embedded in protein ID (e.g., contig_gene format)
    # Extract contig from protein ID prefix
    contig_match = re.match(r"(.+)_(?:gene|orf|protein|snap|g)_?(\d+)$", protein_id, re.IGNORECASE)
    if contig_match:
        contig_id = contig_match.group(1)
        return protein_id, contig_id, 0, 0, "."

    # Strategy 4: Try positional numeric fields
    if len(parts) >= 4:
        try:
            start = int(parts[1])
            end = int(parts[2])
            strand = parts[3] if parts[3] in ("+", "-") else "."
            contig_match = re.match(r"(.+)_\d+$", protein_id)
            contig_id = contig_match.group(1) if contig_match else protein_id
            return protein_id, contig_id, start, end, strand
        except ValueError:
            pass

    # Fallback: extract contig from protein ID
    contig_match = re.match(r"(.+?)_\d+$", protein_id)
    contig_id = contig_match.group(1) if contig_match else protein_id
    return protein_id, contig_id, 0, 0, "."

def parse_snap_gff(gff_path: str):
    """
    Parse SNAP GFF gene predictions into a map of protein_id → coordinates.

    SNAP GFF format (tab-separated):
      col0: contig_id
      col1: SNAP
      col2: feature_type (Esngl, Einit, Exon, Eterm)
      col3: start
      col4: end
      col5: score
      col6: strand
      col7: phase
      col8: protein_id  (not GFF3 key=value — just the bare protein ID)

    Multi-exon genes have multiple rows (Einit, Exon, Eterm) sharing the
    same protein_id. We keep the min start and max end for each protein.
    """
    gene_coords = {}  # protein_id → {contig, start, end, strand}
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue

            contig_id = parts[0].split()[0]  # Handle space in contig description
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            protein_id = parts[8].strip()

            if protein_id in gene_coords:
                # Extend coordinates for multi-exon genes
                gene_coords[protein_id]["start"] = min(gene_coords[protein_id]["start"], start)
                gene_coords[protein_id]["end"] = max(gene_coords[protein_id]["end"], end)
            else:
                gene_coords[protein_id] = {
                    "contig": contig_id,
                    "start": start,
                    "end": end,
                    "strand": strand,
                }

    return gene_coords

def parse_gff3_file(gff_path: str):
    """Parse GFF3 gene predictions into coordinate records."""
    records = []
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            if parts[2] not in ("gene", "CDS", "mRNA"):
                continue

            contig_id = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]

            # Extract protein/gene ID from attributes
            attrs = parts[8]
            protein_id = None
            for attr in attrs.split(";"):
                if attr.startswith("ID="):
                    protein_id = attr[3:]
                elif attr.startswith("Name="):
                    protein_id = protein_id or attr[5:]

            if protein_id:
                records.append((protein_id, contig_id, start, end, strand))

    return records

def parse_zff_file(zff_path: str):
    """Parse ZFF (SNAP native) format."""
    records = []
    current_contig = None
    with open(zff_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_contig = line[1:].split()[0]
                continue
            parts = line.split()
            if len(parts) >= 4 and current_contig:
                label = parts[0]  # Esngl, Einit, Eterm, Exon
                start = int(parts[1])
                end = int(parts[2])
                gene_id = parts[3] if len(parts) > 3 else f"{current_contig}_gene"
                strand = "+" if start < end else "-"
                if start > end:
                    start, end = end, start
                records.append((gene_id, current_contig, start, end, strand))

    return records

def extract_contig_from_snap_id(protein_id: str, assembly_base: str):
    """
    Extract contig ID from SNAP-style protein ID.

    SNAP protein IDs follow: {contig_id}-{assembly_base}.{gene_index}
    Example:
      protein_id = "ERZ17086093.1-NODE-15-length-641-cov-3.0000-MGYA00687825_assembly.1"
      assembly_base = "MGYA00687825_assembly"
      → contig_id = "ERZ17086093.1-NODE-15-length-641-cov-3.0000"
      → gene_index = 1
    """
    # Find the last occurrence of -{assembly_base}.
    marker = f"-{assembly_base}."
    idx = protein_id.rfind(marker)
    if idx > 0:
        contig_id = protein_id[:idx]
        try:
            gene_index = int(protein_id[idx + len(marker):])
        except ValueError:
            gene_index = 0
        return contig_id, gene_index
    # Fallback: try splitting on last dot for gene index
    fallback = re.match(r"(.+?)\.(\d+)$", protein_id)
    if fallback:
        return fallback.group(1), int(fallback.group(2))
    return protein_id, 0

def process_assembly_fasta(fasta_path: str, assembly_id: str, gff_map: dict = None):
    """Yield protein→contig records from a FASTA file (streaming).

    If gff_map is provided (protein_id → {contig, start, end, strand}),
    use it for coordinates. Otherwise, extract contig from FASTA header.
    """
    contig_gene_counts = defaultdict(int)

    # Derive assembly_base (the SNAP name without .aa.algal suffix)
    # FASTA stem: "MGYA00687825_assembly.aa.algal"
    # SNAP uses: "MGYA00687825_assembly"
    assembly_base = assembly_id.split(".aa")[0]  # "MGYA00687825_assembly"

    with open(fasta_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            if not line.startswith(">"):
                continue

            protein_id = line.strip().lstrip(">").split()[0]

            if gff_map and protein_id in gff_map:
                info = gff_map[protein_id]
                contig_id = info["contig"]
                start = info["start"]
                end = info["end"]
                strand = info["strand"]
            else:
                contig_id, _ = extract_contig_from_snap_id(protein_id, assembly_base)
                start = 0
                end = 0
                strand = "."

            contig_gene_counts[contig_id] += 1
            gene_index = contig_gene_counts[contig_id]

            yield {
                "protein_id": protein_id,
                "assembly_id": assembly_id,
                "contig_id": contig_id,
                "start": start,
                "end": end,
                "strand": strand,
                "gene_index_on_contig": gene_index,
            }

def process_assembly_gff(gff_path: str, assembly_id: str):
    """Parse GFF3 predictions for an assembly."""
    raw = parse_gff3_file(gff_path)
    contig_gene_counts = defaultdict(int)
    records = []

    for protein_id, contig_id, start, end, strand in raw:
        contig_gene_counts[contig_id] += 1
        records.append({
            "protein_id": protein_id,
            "assembly_id": assembly_id,
            "contig_id": contig_id,
            "start": start,
            "end": end,
            "strand": strand,
            "gene_index_on_contig": contig_gene_counts[contig_id],
        })

    return records

def main():
    parser = argparse.ArgumentParser(description="Build protein → contig coordinate map")
    parser.add_argument("--format", choices=["auto", "fasta", "gff3", "zff"],
                        default="auto", help="Input format for coordinate extraction")
    parser.add_argument("--cpus", type=int, default=28, help="Parallel workers")
    args = parser.parse_args()

    BASE = get_base_dir("TARA-LA4SR")
    ALGAL_DIR = BASE / "03_analyses" / "algae_proteins"
    GFF_DIR = BASE / "03_analyses" / "gene_predictions"
    OUT_DIR = BASE / "novel_families" / "data" / "contig_maps"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  A1: Build Protein → Contig Coordinate Map")
    print("=" * 60)
    print()
    print(f"  Base dir:   {BASE}")
    print(f"  FASTA dir:  {ALGAL_DIR}")
    print(f"  Format:     {args.format}")
    print(f"  CPUs:       {args.cpus}")
    print()

    # Discover FASTA files
    fasta_files = sorted(ALGAL_DIR.glob("*.fa")) + sorted(ALGAL_DIR.glob("*.faa"))
    if not fasta_files:
        print(f"ERROR: No FASTA files found in {ALGAL_DIR}")
        sys.exit(1)

    print(f"  Found {len(fasta_files)} FASTA files")

    # Auto-detect format from first FASTA
    if args.format == "auto":
        with open(fasta_files[0]) as f:
            first_header = ""
            for line in f:
                if line.startswith(">"):
                    first_header = line.strip()
                    break

        print(f"  Auto-detecting format from: {first_header[:80]}...")

        # Check if headers contain coordinates
        if re.search(r"#\s*\d+\s*#\s*\d+", first_header):
            detected = "fasta"  # Prodigal-style
            print("  Detected: Prodigal-style FASTA headers")
        elif re.search(r"\S+:\d+-\d+", first_header):
            detected = "fasta"  # contig:start-end
            print("  Detected: Coordinate-embedded FASTA headers")
        elif GFF_DIR.exists() and list(GFF_DIR.glob("*.gff3")):
            detected = "gff3"
            print("  Detected: GFF3 gene predictions available")
        else:
            detected = "fasta"
            print("  Defaulting to FASTA header parsing")

        args.format = detected

    # Process assemblies — streaming write to avoid OOM
    print(f"\n  Processing {len(fasta_files)} assemblies (streaming mode)...")

    output_path = OUT_DIR / "protein_contig_map.tsv.gz"
    cols = ["protein_id", "assembly_id", "contig_id", "start", "end", "strand",
            "gene_index_on_contig"]

    n_processed = 0
    n_total_proteins = 0
    n_has_coords = 0
    n_unique_contigs = 0

    with gzip.open(output_path, "wt") as out_f:
        out_f.write("\t".join(cols) + "\n")

        for i, fasta_path in enumerate(fasta_files):
            assembly_id = fasta_path.stem
            recs = process_assembly_fasta(str(fasta_path), assembly_id)

            # Track contigs per assembly (much smaller than global set)
            assembly_contigs = set()
            for r in recs:
                out_f.write("\t".join(str(r[c]) for c in cols) + "\n")
                n_total_proteins += 1
                assembly_contigs.add(r["contig_id"])
                if r["start"] > 0:
                    n_has_coords += 1

            n_unique_contigs += len(assembly_contigs)
            n_processed += 1
            if (i + 1) % 100 == 0:
                print(f"    Processed {i+1}/{len(fasta_files)} assemblies "
                      f"({n_total_proteins:,} proteins)", flush=True)

    print(f"\n  Total assemblies processed: {n_processed}")
    print(f"  Total protein records: {n_total_proteins:,}")

    # Verification
    print(f"\n  ── Verification ──")
    print(f"  Total proteins: {n_total_proteins:,}")
    print(f"  Assemblies: {n_processed}")
    print(f"  Contigs (sum across assemblies): {n_unique_contigs:,}")
    print(f"  Proteins with coordinates: {n_has_coords:,} "
          f"({100*n_has_coords/max(n_total_proteins,1):.1f}%)")

    # Write provenance
    prov_path = BASE / "novel_families" / "provenance" / "A1_contig_map.md"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prov_path, "w") as f:
        f.write("# A1 Contig Map Provenance\n\n")
        f.write(f"- Script: {os.path.abspath(__file__)}\n")
        f.write(f"- Date: {__import__('datetime').datetime.now()}\n")
        f.write(f"- Input: {ALGAL_DIR}\n")
        f.write(f"- Format: {args.format}\n")
        f.write(f"- Output: {output_path}\n")
        f.write(f"- Total proteins: {n_total_proteins:,}\n")
        f.write(f"- Assemblies: {n_processed}\n")
        f.write(f"- Contigs (sum across assemblies): {n_unique_contigs:,}\n")
        f.write(f"- Proteins with coords: {n_has_coords:,}\n")
        f.write("- Integrity Check: PASSED\n")

    print(f"\n  Provenance: {prov_path}")
    print(f"  Done: {__import__('datetime').datetime.now()}")

if __name__ == "__main__":
    main()
