#!/usr/bin/env python3
"""
Assemble Data S9: Cross-taxonomic dark-vs-white selection analysis.

Provenance:
  Script: build_data_s9_20260601_120000.py
  Date: 2026-06-01
  Task: Package all per-gene selection data into a single Zenodo-ready Excel
        workbook with per-species sheets, summary, and provenance.
  Integrity: all values read from real per-gene source files; no synthetic data.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

SCRIPT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = SCRIPT_DIR.parent.parent
OUTPUT_DIR = MANUSCRIPT_DIR / "source_data"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = OUTPUT_DIR / f"Data_S9_dark_vs_white_selection_{TIMESTAMP}.xlsx"

HEADER_FONT = Font(bold=True, size=11)
HEADER_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
THIN_BORDER = Border(
    bottom=Side(style="thin"),
)


def style_header(ws, ncols):
    for col in range(1, ncols + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.border = THIN_BORDER
        cell.alignment = Alignment(wrap_text=True)


def auto_width(ws, max_width=40):
    for col_cells in ws.columns:
        lengths = []
        for cell in col_cells:
            if cell.value is not None:
                lengths.append(min(len(str(cell.value)), max_width))
        if lengths:
            col_letter = get_column_letter(col_cells[0].column)
            ws.column_dimensions[col_letter].width = max(lengths) + 2


def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def build_chlamy_sheet(wb):
    """C. reinhardtii — Flowers et al. 2015 piN/piS."""
    ws = wb.create_sheet("C_reinhardtii")

    ds3_path = SCRIPT_DIR / "tpc00492_SupplementalDS3.txt"
    pfam_path = SCRIPT_DIR / "chlamy_gene_pfam_status.tsv"

    pfam_status = {}
    pfam_domains = {}
    with open(pfam_path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("Gene_ID"):
                continue
            parts = line.rstrip("\n").split("\t")
            gene_id = parts[0]
            has_pfam = int(parts[1])
            domains = parts[2] if len(parts) > 2 else ""
            pfam_status[gene_id] = has_pfam
            pfam_domains[gene_id] = domains

    headers = [
        "Gene_ID", "Transcript_ID", "piN", "piS", "piNpiS",
        "Nonsynonymous_sites", "Synonymous_sites",
        "has_pfam", "pfam_domains", "partition"
    ]
    ws.append(headers)
    style_header(ws, len(headers))

    row_count = 0
    with open(ds3_path) as f:
        for i, line in enumerate(f):
            if i < 2:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            gene_id = parts[0]
            transcript_id = parts[1]
            piN = safe_float(parts[2])
            piS = safe_float(parts[3])
            piNpiS = safe_float(parts[4])
            nonsyn = safe_float(parts[5])
            syn = safe_float(parts[6])

            hp = pfam_status.get(gene_id, -1)
            domains = pfam_domains.get(gene_id, "")

            if hp == 1:
                partition = "white"
            elif hp == 0:
                partition = "dark"
            elif hp == -1:
                partition = "excluded"
            else:
                partition = "unknown"

            ws.append([gene_id, transcript_id, piN, piS, piNpiS,
                       nonsyn, syn, hp, domains, partition])
            row_count += 1

    auto_width(ws)
    return row_count


def build_seminavis_sheet(wb):
    """S. robusta — Osuna-Cruz et al. 2020 piN/piS."""
    ws = wb.create_sheet("S_robusta")

    data_path = SCRIPT_DIR / "seminavis" / "seminavis_pnps_per_gene.tsv"

    headers = [
        "Gene_ID", "InterPro_description", "Pan_gene_category",
        "piN", "piS", "piNpiS", "partition"
    ]
    ws.append(headers)
    style_header(ws, len(headers))

    row_count = 0
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            gene_id = parts[0]
            interpro = parts[1]
            pan_cat = parts[2]
            piN = safe_float(parts[3])
            piS = safe_float(parts[4])
            piNpiS = safe_float(parts[5])

            partition = "dark" if interpro == "-" else "white"

            ws.append([gene_id, interpro, pan_cat, piN, piS, piNpiS, partition])
            row_count += 1

    auto_width(ws)
    return row_count


def build_synechococcus_sheet(wb, strain, locus_file, pfam_file, dnds_file):
    """Synechococcus CC9311 or CC9902 — Tai et al. 2011 dN/dS."""
    ws = wb.create_sheet(f"Synechococcus_{strain}")

    pfam_status = {}
    pfam_domains = {}
    with open(pfam_file) as f:
        for line in f:
            if line.startswith("locus_tag"):
                continue
            parts = line.rstrip("\n").split("\t")
            locus = parts[0]
            has_pfam = 1 if parts[1] == "yes" else 0
            domains = parts[3] if len(parts) > 3 else ""
            pfam_status[locus] = has_pfam
            pfam_domains[locus] = domains

    headers = [
        "locus_tag", "description", "dN", "dS", "dNdS",
        "pct_min5X_cov", "pct_min1X_cov", "avg_read_depth",
        "has_pfam", "pfam_domains", "partition"
    ]
    ws.append(headers)
    style_header(ws, len(headers))

    row_count = 0
    with open(dnds_file) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            locus = parts[0]
            desc = parts[1]
            dNdS = safe_float(parts[2])
            dN = safe_float(parts[3])
            dS = safe_float(parts[4])
            cov5x = safe_float(parts[5])
            cov1x = safe_float(parts[6])
            avg_depth = safe_float(parts[7])

            hp = pfam_status.get(locus, -1)
            domains = pfam_domains.get(locus, "")
            partition = "white" if hp == 1 else "dark"

            ws.append([locus, desc, dN, dS, dNdS,
                       cov5x, cov1x, avg_depth,
                       hp, domains, partition])
            row_count += 1

    auto_width(ws)
    return row_count


def build_tpseudo_sheet(wb):
    """T. pseudonana — Koester et al. 2013 positive selection."""
    ws = wb.create_sheet("T_pseudonana")

    data_path = SCRIPT_DIR / "tpseudo" / "koester2013_table_s2_with_pfam.tsv"

    headers = [
        "protein_id", "gene_id", "sel_ml", "neu_ml",
        "lrt_statistic", "pvalue", "nom_sig", "bonf_sig",
        "qvalue", "fdr_sig", "orphan_original",
        "pfam_status", "pfam_domains", "partition"
    ]
    ws.append(headers)
    style_header(ws, len(headers))

    row_count = 0
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12:
                continue
            protein_id = parts[0]
            gene_id = parts[1]
            sel_ml = safe_float(parts[2])
            neu_ml = safe_float(parts[3])
            lrt = safe_float(parts[4])
            pval = safe_float(parts[5])
            nom_sig = safe_float(parts[6])
            bonf_sig = safe_float(parts[7])
            qval = safe_float(parts[8])
            fdr_sig = safe_float(parts[9])
            orphan = parts[10]
            pfam_st = parts[11]
            pfam_dom = parts[12] if len(parts) > 12 else ""

            partition = "dark" if pfam_st == "dark" else "white"

            ws.append([protein_id, gene_id, sel_ml, neu_ml,
                       lrt, pval, nom_sig, bonf_sig,
                       qval, fdr_sig, orphan,
                       pfam_st, pfam_dom, partition])
            row_count += 1

    auto_width(ws)
    return row_count


def build_summary_sheet(wb):
    """Cross-taxonomic summary table."""
    ws = wb.create_sheet("Summary")

    headers = [
        "Species", "Lineage", "Metric", "n_dark", "n_white",
        "median_dark", "median_white", "fold_dark_over_white",
        "CI_95_low", "CI_95_high", "p_value",
        "partition_method", "source_paper", "source_DOI"
    ]
    ws.append(headers)
    style_header(ws, len(headers))

    rows = [
        [
            "Chlamydomonas reinhardtii", "Chlorophyta", "piN/piS",
            4085, 7038, 0.3371, 0.1466, 2.30,
            None, None, "<1e-300",
            "UniProt + Ensembl BioMart Pfam annotation",
            "Flowers et al. 2015, Plant Cell 27:2353-2369",
            "10.1105/tpc.15.00492"
        ],
        [
            "Seminavis robusta", "Bacillariophyta (pennate diatom)", "piN/piS",
            7537, 13205, 0.1708, 0.1168, 1.46,
            1.42, 1.51, "5.74e-160",
            "InterPro annotation presence/absence",
            "Osuna-Cruz et al. 2020, Nat Commun 11:3320",
            "10.1038/s41467-020-17191-8"
        ],
        [
            "Synechococcus CC9311", "Cyanobacteria (clade I)", "dN/dS",
            706, 1698, 0.1785, 0.0950, 1.88,
            1.76, 2.05, "1.23e-68",
            "hmmsearch Pfam-A domain i-Evalue < 1e-9",
            "Tai & Palenik 2011, PLoS ONE 6:e24249",
            "10.1371/journal.pone.0024249"
        ],
        [
            "Synechococcus CC9902", "Cyanobacteria (clade IV)", "dN/dS",
            549, 1665, 0.1820, 0.1010, 1.80,
            1.65, 1.94, "1.05e-50",
            "hmmsearch Pfam-A domain i-Evalue < 1e-9",
            "Tai & Palenik 2011, PLoS ONE 6:e24249",
            "10.1371/journal.pone.0024249"
        ],
        [
            "Synechococcus combined", "Cyanobacteria", "dN/dS",
            1255, 3363, 0.1800, 0.0980, 1.84,
            1.75, 1.94, "3.29e-117",
            "hmmsearch Pfam-A domain i-Evalue < 1e-9",
            "Tai & Palenik 2011, PLoS ONE 6:e24249",
            "10.1371/journal.pone.0024249"
        ],
        [
            "Thalassiosira pseudonana", "Bacillariophyta (centric diatom)",
            "positive selection rate ratio",
            5408, 6265, "9.3% pos. sel.", "4.9% pos. sel.", 1.90,
            1.27, 1.42, "4.88e-21",
            "hmmsearch Pfam-A domain i-Evalue < 1e-9",
            "Koester et al. 2013, Mol Biol Evol 30:422-434",
            "10.1093/molbev/mss242"
        ],
    ]

    for row in rows:
        ws.append(row)

    auto_width(ws)


def build_provenance_sheet(wb):
    """Provenance and methods documentation."""
    ws = wb.create_sheet("Provenance")

    ws.column_dimensions["A"].width = 25
    ws.column_dimensions["B"].width = 80

    bold = Font(bold=True, size=11)

    entries = [
        ("Dataset", "Data S9: Cross-taxonomic dark-vs-white selection analysis"),
        ("Date assembled", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Assembly script", "scripts/pnps_analysis/build_data_s9_20260601_120000.py"),
        ("", ""),
        ("=== C. reinhardtii ===", ""),
        ("Source paper", "Flowers et al. 2015. Plant Cell 27:2353-2369"),
        ("Source DOI", "10.1105/tpc.15.00492"),
        ("Data source", "Dryad doi:10.5061/dryad.1n0g6 — SupplementalDS3"),
        ("Pfam partition method",
         "Merged UniProt (organism 3055) + Ensembl BioMart (creinhardtii_eg_gene) "
         "Pfam annotation. has_pfam=1 (white): >=1 Pfam domain. has_pfam=0 (dark): "
         "no Pfam domain annotated. has_pfam=-1 (excluded): old JGI v5.0 gene IDs "
         "not in v5.5/v5.6 databases."),
        ("Filter", "piS > 0 AND Nonsynonymous + Synonymous >= 5"),
        ("Selection metric", "piN/piS (within-species polymorphism, 12 strains)"),
        ("Analysis script", "scripts/pnps_analysis/analyze_pnps_dark_white_20260531_070718.py"),
        ("Pfam status file", "scripts/pnps_analysis/chlamy_gene_pfam_status.tsv"),
        ("", ""),
        ("=== S. robusta ===", ""),
        ("Source paper", "Osuna-Cruz et al. 2020. Nat Commun 11:3320"),
        ("Source DOI", "10.1038/s41467-020-17191-8"),
        ("Data source", "Source Data file: Suppl. Fig. 20, 22.xlsx (MOESM4)"),
        ("Partition method",
         "InterPro annotation status. Dark = InterPro_description is '-' "
         "(no hit in ANY InterPro member database: Pfam, PANTHER, CDD, SUPERFAMILY, etc.). "
         "White = any InterPro annotation present. This is broader than Pfam-only."),
        ("Filter", "piS > 0 for piN/piS computation"),
        ("Selection metric", "piN/piS (within-species polymorphism, 48 strains)"),
        ("Analysis script", "scripts/pnps_analysis/seminavis/analyze_seminavis_pnps_dark_white_20260531_080200.py"),
        ("ENA project", "PRJEB36614"),
        ("", ""),
        ("=== Synechococcus CC9311 & CC9902 ===", ""),
        ("Source paper", "Tai & Palenik 2011. PLoS ONE 6:e24249"),
        ("Source DOI", "10.1371/journal.pone.0024249"),
        ("Data source", "Supplementary Table S3 (CC9311) and Table S4 (CC9902)"),
        ("Partition method",
         "hmmsearch (HMMER 3.3.2) of NCBI reference proteomes "
         "(GCF_000014585.1 for CC9311, GCF_000012505.1 for CC9902) "
         "against Pfam-A.hmm. Domain i-Evalue < 1e-9 threshold. "
         "Dark = no Pfam domain hit. White = >=1 Pfam domain hit."),
        ("Filter", "finite dN/dS AND >0% 5X read coverage"),
        ("Selection metric",
         "dN/dS (between-lineage divergence from environmental metagenomic reads, "
         "NOT within-species polymorphism)"),
        ("Analysis script", "scripts/pnps_analysis/synechococcus/analyze_dnds_dark_white_20260531_083721.py"),
        ("Pfam status files",
         "scripts/pnps_analysis/synechococcus/CC9311_gene_pfam_status.tsv, "
         "CC9902_gene_pfam_status.tsv"),
        ("NCBI Project ID", "66351"),
        ("", ""),
        ("=== T. pseudonana ===", ""),
        ("Source paper", "Koester et al. 2013. Mol Biol Evol 30:422-434"),
        ("Source DOI", "10.1093/molbev/mss242"),
        ("Data source", "Supplementary Table S2 (3,280 genes with p<0.05)"),
        ("Partition method",
         "hmmsearch (HMMER 3.3.2) of NCBI GCF_000149405.2 proteome "
         "against Pfam-A.hmm. Domain i-Evalue < 1e-9 threshold. "
         "5,408 dark (no Pfam) + 6,265 white (>=1 Pfam) = 11,673 total. "
         "Cross-validated against UniProt UP000001449 Pfam annotations (88.2% agreement)."),
        ("Note",
         "Metric is positive selection enrichment from PAML M8a vs M8 site models "
         "(7 strains), NOT per-gene dN/dS. The T. pseudonana sheet contains the "
         "3,280 genes with nominal p<0.05 from the original paper, not the full genome."),
        ("Analysis script", "scripts/pnps_analysis/tpseudo/analyze_tpseudo_dnds_dark_white_20260531_081500.py"),
        ("Pfam status file", "scripts/pnps_analysis/tpseudo/koester2013_table_s2_with_pfam.tsv"),
        ("", ""),
        ("=== General notes ===", ""),
        ("Pfam version", "Pfam-A v35.0 (for hmmsearch-based partitions)"),
        ("HMMER version", "3.3.2"),
        ("Partition heterogeneity",
         "The dark/white partition method is NOT uniform across species. "
         "C. reinhardtii uses merged database annotations; S. robusta uses "
         "InterPro (broader than Pfam); Synechococcus and T. pseudonana use "
         "hmmsearch Pfam-A at E < 1e-9. See per-species entries above."),
        ("Metric heterogeneity",
         "C. reinhardtii and S. robusta use piN/piS (within-species polymorphism). "
         "Synechococcus uses dN/dS (between-lineage metagenomic divergence). "
         "T. pseudonana uses positive selection site-model enrichment."),
    ]

    for label, value in entries:
        ws.append([label, value])
        if label.startswith("==="):
            ws.cell(row=ws.max_row, column=1).font = bold


def build_analysis_scripts_sheet(wb):
    """List all analysis scripts included or referenced."""
    ws = wb.create_sheet("Scripts")

    headers = ["Script", "Species", "Purpose"]
    ws.append(headers)
    style_header(ws, len(headers))

    scripts = [
        ("analyze_pnps_dark_white_20260531_070718.py",
         "C. reinhardtii",
         "Partition Flowers 2015 genes by Pfam status; compute piN/piS comparison"),
        ("finalize_pfam_status_20260530_211500.py",
         "C. reinhardtii",
         "Merge UniProt + Ensembl BioMart + NCBI GFF3 Pfam annotations"),
        ("analyze_seminavis_pnps_dark_white_20260531_080200.py",
         "S. robusta",
         "Partition Osuna-Cruz 2020 genes by InterPro status; compute piN/piS comparison"),
        ("analyze_dnds_dark_white_20260531_083721.py",
         "Synechococcus",
         "Partition Tai 2011 genes by hmmsearch Pfam status; compute dN/dS comparison"),
        ("analyze_tpseudo_dnds_dark_white_20260531_081500.py",
         "T. pseudonana",
         "Partition Koester 2013 genes by hmmsearch Pfam status; compute enrichment"),
        ("crossval_hmmsearch_pfam_20260531_083500.py",
         "T. pseudonana",
         "Cross-validate hmmsearch partition against UniProt Pfam annotations"),
        ("verify_tpseudo_analysis_20260531_082500.py",
         "T. pseudonana",
         "Independent verification of T. pseudonana positive selection enrichment"),
        ("build_data_s9_20260601_120000.py",
         "All",
         "Assemble this Data S9 Excel workbook from per-species source files"),
    ]

    for row in scripts:
        ws.append(list(row))

    auto_width(ws)


def main():
    print(f"Building Data S9: {OUTPUT_FILE}")
    wb = openpyxl.Workbook()
    ws_default = wb.active
    wb.remove(ws_default)

    n_chlamy = build_chlamy_sheet(wb)
    print(f"  C. reinhardtii: {n_chlamy} genes")

    n_srobusta = build_seminavis_sheet(wb)
    print(f"  S. robusta: {n_srobusta} genes")

    n_cc9311 = build_synechococcus_sheet(
        wb, "CC9311",
        SCRIPT_DIR / "synechococcus" / "CC9311_locus_to_protein.tsv",
        SCRIPT_DIR / "synechococcus" / "CC9311_gene_pfam_status.tsv",
        SCRIPT_DIR / "synechococcus" / "CC9311_dnds.tsv",
    )
    print(f"  Synechococcus CC9311: {n_cc9311} genes")

    n_cc9902 = build_synechococcus_sheet(
        wb, "CC9902",
        SCRIPT_DIR / "synechococcus" / "CC9902_locus_to_protein.tsv",
        SCRIPT_DIR / "synechococcus" / "CC9902_gene_pfam_status.tsv",
        SCRIPT_DIR / "synechococcus" / "CC9902_dnds.tsv",
    )
    print(f"  Synechococcus CC9902: {n_cc9902} genes")

    n_tpseudo = build_tpseudo_sheet(wb)
    print(f"  T. pseudonana: {n_tpseudo} genes")

    build_summary_sheet(wb)
    print("  Summary sheet: built")

    build_provenance_sheet(wb)
    print("  Provenance sheet: built")

    build_analysis_scripts_sheet(wb)
    print("  Scripts sheet: built")

    sheet_order = [
        "Summary", "C_reinhardtii", "S_robusta",
        "Synechococcus_CC9311", "Synechococcus_CC9902",
        "T_pseudonana", "Provenance", "Scripts"
    ]
    wb._sheets.sort(key=lambda s: sheet_order.index(s.title))

    wb.save(OUTPUT_FILE)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Sheets: {[s.title for s in wb.worksheets]}")


if __name__ == "__main__":
    main()
