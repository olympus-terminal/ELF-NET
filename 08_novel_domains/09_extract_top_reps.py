#!/usr/bin/env python3
"""
09_extract_top_reps.py — Select top 10 novel domains for structural characterization.

Selection strategy:
  Composite ranking = rank(n_samples) + rank(max |rho|) + rank(total_abundance)
  Top 10 by composite rank. No artificial taxonomic diversity constraint —
  the paper's argument is about broadly distributed novel domains.

Inputs:
  novel_domains/results/novel_domain_summary.tsv
  novel_domains/results/novel_domain_env_correlations.tsv
  novel_domains/results/clusters_30/clusters_30_rep_seq.fasta

Outputs:
  novel_domains/results/top10_for_structure.fasta
  novel_domains/results/top10_selection_summary.tsv
"""

import sys
from pathlib import Path
import csv
import re

BASE = Path("/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains")
RESULTS = BASE / "results"

SUMMARY_FILE = RESULTS / "novel_domain_summary.tsv"
CORR_FILE = RESULTS / "novel_domain_env_correlations.tsv"
REP_FASTA = BASE / "clusters_30" / "clusters_30_rep_seq.fasta"
OUT_FASTA = RESULTS / "top10_for_structure.fasta"
OUT_SUMMARY = RESULTS / "top10_selection_summary.tsv"

def classify_source(domain_id: str) -> str:
    """Classify domain source from the anchor sequence ID."""
    d = domain_id.replace("NOVEL_", "")
    # Metagenomic: ERZ accessions or MGYA
    if re.search(r"ERZ\d+|MGYA\d+|NODE-", d):
        return "metagenomic"
    # Known green algae
    greens = ["Dunaliella", "Chlamydomonas", "Chlorella", "Characiochloris",
              "Micromonas", "Ostreococcus", "Bathycoccus", "Coccomyxa",
              "Volvox", "Gonium", "Ulva"]
    for g in greens:
        if g.lower() in d.lower():
            return "green_alga"
    # Red algae
    reds = ["Chondrus", "Gracilaria", "Porphyra", "Pyropia", "Palmaria",
            "Galdieria", "Cyanidioschyzon"]
    for r in reds:
        if r.lower() in d.lower():
            return "red_alga"
    # Haptophytes
    hapts = ["Chrysochromulina", "Emiliania", "Isochrysis", "Prymnesium",
             "Phaeocystis", "Gephyrocapsa", "Tisochrysis"]
    for h in hapts:
        if h.lower() in d.lower():
            return "haptophyte"
    # Diatoms
    diatoms = ["Thalassiosira", "Phaeodactylum", "Pseudo-nitzschia",
               "Fragilariopsis", "Skeletonema", "Chaetoceros"]
    for dt in diatoms:
        if dt.lower() in d.lower():
            return "diatom"
    # Dinoflagellates
    dinos = ["Symbiodinium", "Amphidinium", "Alexandrium", "Karlodinium",
             "Prorocentrum"]
    for di in dinos:
        if di.lower() in d.lower():
            return "dinoflagellate"
    # NCBI genome accessions (GCA_, RKSO, KAL, etc.)
    if re.search(r"GCA_|RKSO|KAL\d+", d):
        return "ncbi_genome"
    return "other"

def main():
    # ---- Load summary ----
    print("Loading domain summary...")
    domains = {}
    with open(SUMMARY_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            d = row["domain"]
            domains[d] = {
                "cluster_size": int(row["cluster_size"]),
                "n_samples": int(row["n_samples"]),
                "total_abundance": int(row["total_abundance"]),
                "mean_abundance": float(row["mean_abundance_when_present"]),
            }
    print(f"  {len(domains)} domains loaded from summary")

    # ---- Load max |rho| per domain from correlations ----
    print("Loading environmental correlations...")
    max_rho = {}
    best_env = {}
    best_rho_signed = {}
    n_corr = 0
    with open(CORR_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_corr += 1
            d = row["domain"]
            rho = abs(float(row["rho"]))
            if d not in max_rho or rho > max_rho[d]:
                max_rho[d] = rho
                best_env[d] = row["env_variable"]
                best_rho_signed[d] = float(row["rho"])
    print(f"  {n_corr} correlations loaded, {len(max_rho)} domains with env data")

    # ---- Compute composite rank ----
    # Only consider domains present in both summary AND correlations
    common = set(domains.keys()) & set(max_rho.keys())
    print(f"  {len(common)} domains with both summary and correlation data")

    records = []
    for d in common:
        records.append({
            "domain": d,
            "n_samples": domains[d]["n_samples"],
            "total_abundance": domains[d]["total_abundance"],
            "cluster_size": domains[d]["cluster_size"],
            "mean_abundance": domains[d]["mean_abundance"],
            "max_abs_rho": max_rho[d],
            "best_env": best_env[d],
            "best_rho": best_rho_signed[d],
            "source": classify_source(d),
        })

    # Rank: lower is better
    records.sort(key=lambda x: x["n_samples"], reverse=True)
    for i, r in enumerate(records):
        r["rank_prevalence"] = i + 1

    records.sort(key=lambda x: x["max_abs_rho"], reverse=True)
    for i, r in enumerate(records):
        r["rank_rho"] = i + 1

    records.sort(key=lambda x: x["total_abundance"], reverse=True)
    for i, r in enumerate(records):
        r["rank_abundance"] = i + 1

    for r in records:
        r["composite_rank"] = r["rank_prevalence"] + r["rank_rho"] + r["rank_abundance"]

    records.sort(key=lambda x: x["composite_rank"])

    # ---- Select top 10 by composite rank ----
    # Pure composite ranking — no artificial diversity constraint.
    # The paper's argument is about broadly distributed novel domains
    # across the ocean, so selection should reflect that directly.
    selected = records[:10]

    print(f"\n{'='*80}")
    print(f"  TOP 10 NOVEL DOMAINS FOR STRUCTURAL CHARACTERIZATION")
    print(f"{'='*80}")
    for i, s in enumerate(selected, 1):
        print(f"  {i:2d}. {s['domain'][:70]}")
        print(f"      source={s['source']}, samples={s['n_samples']}, "
              f"|rho|={s['max_abs_rho']:.3f} ({s['best_env']}), "
              f"abundance={s['total_abundance']}, cluster={s['cluster_size']}")
        print(f"      composite_rank={s['composite_rank']} "
              f"(prev={s['rank_prevalence']}, rho={s['rank_rho']}, abund={s['rank_abundance']})")

    # ---- Extract sequences from representative FASTA ----
    print(f"\nExtracting sequences from {REP_FASTA}...")
    target_ids = set()
    for s in selected:
        # The rep sequence ID in the FASTA should match the domain name without NOVEL_ prefix
        # or match the full domain name — we'll try both
        target_ids.add(s["domain"])
        target_ids.add(s["domain"].replace("NOVEL_", ""))

    extracted = {}
    current_id = None
    current_seq = []
    writing = False

    with open(REP_FASTA) as f:
        for line in f:
            if line.startswith(">"):
                if writing and current_id:
                    extracted[current_id] = "".join(current_seq)
                header = line.strip().lstrip(">")
                seq_id = header.split()[0]
                # Check if this ID matches any of our targets
                writing = (seq_id in target_ids or
                           f"NOVEL_{seq_id}" in target_ids)
                if writing:
                    # Use the NOVEL_ form as the canonical ID
                    if seq_id.startswith("NOVEL_"):
                        current_id = seq_id
                    else:
                        current_id = f"NOVEL_{seq_id}" if f"NOVEL_{seq_id}" in target_ids else seq_id
                current_seq = []
            elif writing:
                current_seq.append(line.strip())
        if writing and current_id:
            extracted[current_id] = "".join(current_seq)

    print(f"  Extracted {len(extracted)} sequences")

    if len(extracted) == 0:
        print("WARNING: No sequences extracted! Checking FASTA header format...")
        # Show first 5 headers for debugging
        with open(REP_FASTA) as f:
            for i, line in enumerate(f):
                if line.startswith(">"):
                    print(f"  Header: {line.strip()[:100]}")
                    if i > 10:
                        break

    # ---- Write output FASTA ----
    with open(OUT_FASTA, "w") as f:
        for s in selected:
            domain = s["domain"]
            # Try to find the sequence under various ID forms
            seq = extracted.get(domain) or extracted.get(domain.replace("NOVEL_", ""))
            if seq:
                f.write(f">{domain} source={s['source']} samples={s['n_samples']} "
                        f"max_rho={s['max_abs_rho']:.4f} env={s['best_env']} "
                        f"cluster_size={s['cluster_size']}\n")
                # Write 80-char lines
                for j in range(0, len(seq), 80):
                    f.write(seq[j:j+80] + "\n")
            else:
                print(f"  WARNING: sequence not found for {domain}")
    print(f"  Written: {OUT_FASTA}")

    # ---- Write selection summary TSV ----
    with open(OUT_SUMMARY, "w") as f:
        fields = ["rank", "domain", "source", "n_samples", "total_abundance",
                  "mean_abundance", "cluster_size", "max_abs_rho", "best_env",
                  "best_rho", "composite_rank", "seq_length"]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for i, s in enumerate(selected, 1):
            seq = extracted.get(s["domain"]) or extracted.get(s["domain"].replace("NOVEL_", ""))
            writer.writerow({
                "rank": i,
                "domain": s["domain"],
                "source": s["source"],
                "n_samples": s["n_samples"],
                "total_abundance": s["total_abundance"],
                "mean_abundance": f"{s['mean_abundance']:.1f}",
                "cluster_size": s["cluster_size"],
                "max_abs_rho": f"{s['max_abs_rho']:.4f}",
                "best_env": s["best_env"],
                "best_rho": f"{s['best_rho']:.4f}",
                "composite_rank": s["composite_rank"],
                "seq_length": len(seq) if seq else 0,
            })
    print(f"  Written: {OUT_SUMMARY}")

    print(f"\nDone. {len(extracted)} of {len(selected)} sequences extracted.")

if __name__ == "__main__":
    main()
