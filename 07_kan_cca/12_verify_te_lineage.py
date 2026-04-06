#!/usr/bin/env python3
"""
12_verify_te_lineage.py — Verify TE domain enrichment across algal lineages.

Computes mean counts of the 10 TE-related domains (from sparse CCA selection)
per taxonomic group (dinoflagellate, chlorophyte, diatom, haptophyte, other)
using reference genome rows in the merged Pfam dataset.

This validates the biological claim that dinoflagellate genomes carry elevated
TE loads relative to other algal lineages, providing empirical support for
interpreting sparse CCA TE domain selection as a dinoflagellate compositional
proxy.

Output:
  - TSV with per-lineage TE domain counts
  - Summary statistics to stdout
"""

import json
import socket
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

BASE_DIR = Path('/scratch/drn2/PROJECTS/TARA-LA4SR')
ARCHIVE_DIR = Path('/archive/drn2/TARA-Oceans')
OUTPUT_DIR = BASE_DIR / '03_analyses' / 'kan_cca'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def find_data(relative_path):
    for base in [BASE_DIR, ARCHIVE_DIR]:
        p = base / relative_path
        if p.exists():
            return p
    raise FileNotFoundError(f"Not found in scratch or archive: {relative_path}")

# ---------------------------------------------------------------------------
# Domain definitions
# ---------------------------------------------------------------------------

# 13 sparse CCA-selected domains (union across 3 components)
SPARSE_CCA_DOMAINS = [
    'PF00078', 'PF00665', 'PF01576', 'PF05380', 'PF05970',
    'PF13927', 'PF14214', 'PF14529', 'PF17917', 'PF17919',
    'PF17921', 'PF18701', 'PF20209',
]

# 10 of 13 classified as TE machinery (verified via InterPro API 2026-02-22)
TE_DOMAINS = {
    'PF00078': 'RVT_1 — Reverse transcriptase (RNA-dependent DNA polymerase)',
    'PF00665': 'rve — Integrase core domain',
    'PF05380': 'Peptidase_A17 — Pao retrotransposon peptidase',
    'PF14214': 'Helitron_like_N — Rolling-circle DNA transposase',
    'PF14529': 'Exo_endo_phos_2 — Endonuclease in retrotransposons',
    'PF17917': 'RT_RNaseH — RNase H domain in reverse transcriptase',
    'PF17919': 'RT_RNaseH_2 — RNase H-like domain in reverse transcriptase',
    'PF17921': 'Integrase_H2C2 — Zinc-binding domain in integrases',
    'PF18701': 'DUF5641 — Domain in retrotransposon polyproteins',
    'PF20209': 'DUF6570 — Domain in eukaryotic transposon proteins',
}

# Non-TE domains in the sparse CCA set
NON_TE_DOMAINS = {
    'PF01576': 'Myosin_tail_1 — Myosin heavy chain coiled-coil tail',
    'PF05970': 'PIF1 — PIF1-like helicase',
    'PF13927': 'Ig_3 — Immunoglobulin-like domain',
}

# ---------------------------------------------------------------------------
# Lineage classification by genus
# ---------------------------------------------------------------------------

DINOFLAGELLATE_GENERA = {
    'Alexandrium', 'Amphidinium', 'Breviolum', 'Ceratium',
    'Cladocopium', 'Crypthecodinium', 'Dinophysis', 'Durusdinium',
    'Fugacium', 'Gambierdiscus', 'Gymnodinium', 'Gyrodinium',
    'Hematodinium', 'Heterocapsa', 'Karenia', 'Karlodinium',
    'Lepidodinium', 'Lingulodinium', 'Noctiluca', 'Oxyrrhis',
    'Pelagodinium', 'Peridinium', 'Polarella', 'Prorocentrum',
    'Protoceratium', 'Pyrocystis', 'Scrippsiella', 'Symbiodinium',
    'Togula', 'Zooxanthella', 'Azadinium', 'Amoebophrya',
    'Perkinsus', 'Brandtodinium',
}

DIATOM_GENERA = {
    'Chaetoceros', 'Coscinodiscus', 'Cyclotella', 'Cylindrotheca',
    'Ditylum', 'Fragilariopsis', 'Leptocylindrus', 'Licmophora',
    'Nitzschia', 'Odontella', 'Phaeodactylum', 'Pseudo-nitzschia',
    'Skeletonema', 'Thalassionema', 'Thalassiosira', 'Corethron',
    'Extubocellulus', 'Minutocellus', 'Attheya', 'Asterionellopsis',
    'Eucampia', 'Guinardia', 'Rhizosolenia', 'Stephanopyxis',
}

HAPTOPHYTE_GENERA = {
    'Chrysochromulina', 'Emiliania', 'Gephyrocapsa', 'Isochrysis',
    'Pavlova', 'Phaeocystis', 'Pleurochrysis', 'Prymnesium',
    'Tisochrysis', 'Coccolithus', 'Calcidiscus', 'Diacronema',
    'Exanthemachrysis',
}

CHLOROPHYTE_GENERA = {
    'Bathycoccus', 'Botryococcus', 'Chlamydomonas', 'Chlorella',
    'Chloropicon', 'Coccomyxa', 'Dunaliella', 'Haematococcus',
    'Micromonas', 'Nannochloropsis', 'Ostreococcus', 'Picochlorum',
    'Pyramimonas', 'Tetraselmis', 'Ulva', 'Volvox',
    'Auxenochlorella', 'Gonium', 'Prasinoderma', 'Pycnococcus',
    'Mantoniella', 'Nephroselmis', 'Mychonastes', 'Mamiella',
    'Parachlorella', 'Trebouxia', 'Scenedesmus', 'Monoraphidium',
}

def classify_lineage(assembly_id):
    """Classify assembly by genus → lineage."""
    genus = assembly_id.split('_')[0]
    if genus in DINOFLAGELLATE_GENERA:
        return 'Dinoflagellata'
    elif genus in DIATOM_GENERA:
        return 'Bacillariophyta'
    elif genus in HAPTOPHYTE_GENERA:
        return 'Haptophyta'
    elif genus in CHLOROPHYTE_GENERA:
        return 'Chlorophyta'
    else:
        return 'Other'

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{ts}] 12_verify_te_lineage.py starting on {socket.gethostname()}")

    # 1. Load merged TSV
    data_path = find_data(
        '03_analyses/ALGAGPT-based-analyses/'
        'algagpt_gee_pfam_merged_SMART_20260119_100639.tsv'
    )
    print(f"Loading: {data_path}")
    df = pd.read_csv(data_path, sep='\t', comment='#', low_memory=False)
    print(f"Total rows: {len(df)}")

    # 2. Identify Pfam columns matching our 13 CCA domains
    pfam_cols = [c for c in df.columns if c.startswith('PF')]
    print(f"Total Pfam columns: {len(pfam_cols)}")

    # Map versioned column names to accessions
    acc_to_col = {}
    for col in pfam_cols:
        acc = col.split('.')[0]
        if acc in SPARSE_CCA_DOMAINS:
            acc_to_col[acc] = col

    found = sorted(acc_to_col.keys())
    missing = sorted(set(SPARSE_CCA_DOMAINS) - set(found))
    print(f"Found {len(found)}/13 sparse CCA domains in data")
    if missing:
        print(f"  Missing: {missing}")

    # 3. Identify reference genome rows (AAC/MMETSP suffix)
    ref_mask = df['assembly_id'].str.contains(r'\.(AAC|MMETSP)$', na=False)
    df_ref = df[ref_mask].copy()
    print(f"Reference genome rows: {len(df_ref)}")

    # Also try species column if available
    if 'species' in df.columns:
        named_mask = df['species'].notna() & (df['species'] != '')
        print(f"Rows with species annotation: {named_mask.sum()}")

    # 4. Classify lineage
    df_ref['lineage'] = df_ref['assembly_id'].apply(classify_lineage)
    lineage_counts = df_ref['lineage'].value_counts()
    print(f"\nLineage distribution (reference genomes):")
    for lin, cnt in lineage_counts.items():
        print(f"  {lin}: {cnt}")

    # 5. Compute per-lineage domain counts
    results = []
    for acc in SPARSE_CCA_DOMAINS:
        col = acc_to_col.get(acc)
        if col is None:
            continue

        is_te = acc in TE_DOMAINS
        annotation = TE_DOMAINS.get(acc, NON_TE_DOMAINS.get(acc, 'Unknown'))

        for lineage in ['Dinoflagellata', 'Bacillariophyta', 'Haptophyta',
                        'Chlorophyta', 'Other']:
            mask = df_ref['lineage'] == lineage
            vals = pd.to_numeric(df_ref.loc[mask, col], errors='coerce').fillna(0)
            results.append({
                'pfam_accession': acc,
                'annotation': annotation,
                'is_te': is_te,
                'lineage': lineage,
                'n_genomes': mask.sum(),
                'mean_count': vals.mean(),
                'median_count': vals.median(),
                'std_count': vals.std(),
                'max_count': vals.max(),
                'nonzero_frac': (vals > 0).mean(),
            })

    results_df = pd.DataFrame(results)

    # 6. Save results
    out_path = OUTPUT_DIR / f'te_lineage_verification_{ts}.tsv'
    results_df.to_csv(out_path, sep='\t', index=False)
    print(f"\nResults saved to: {out_path}")

    # 7. Summary: TE domains only, dinoflagellate vs others
    te_results = results_df[results_df['is_te']]

    print("\n" + "=" * 80)
    print("TE DOMAIN ENRICHMENT BY LINEAGE (mean count per genome)")
    print("=" * 80)

    pivot = te_results.pivot_table(
        index='pfam_accession', columns='lineage',
        values='mean_count', aggfunc='first'
    )
    col_order = ['Dinoflagellata', 'Bacillariophyta', 'Haptophyta',
                 'Chlorophyta', 'Other']
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
    print(pivot.round(2).to_string())

    # Per-lineage aggregate (sum of mean TE domain counts)
    print("\n--- Aggregate TE load (sum of mean counts across 10 TE domains) ---")
    for lineage in col_order:
        subset = te_results[te_results['lineage'] == lineage]
        if len(subset) > 0:
            total = subset['mean_count'].sum()
            n = subset['n_genomes'].iloc[0]
            print(f"  {lineage}: total TE load = {total:.1f} (n = {n} genomes)")

    # Fold-enrichment: dinoflagellate vs mean of other lineages
    dino_total = te_results[te_results['lineage'] == 'Dinoflagellata']['mean_count'].sum()
    other_totals = []
    for lin in ['Bacillariophyta', 'Haptophyta', 'Chlorophyta']:
        t = te_results[te_results['lineage'] == lin]['mean_count'].sum()
        if t > 0:
            other_totals.append(t)

    if other_totals and dino_total > 0:
        mean_other = np.mean(other_totals)
        fold = dino_total / mean_other if mean_other > 0 else float('inf')
        print(f"\n  Dinoflagellata / mean(other 3 lineages) = {fold:.1f}-fold enrichment")

    # Non-TE domains for comparison
    non_te = results_df[~results_df['is_te']]
    print("\n--- Non-TE domain counts by lineage ---")
    for _, row in non_te.iterrows():
        print(f"  {row['pfam_accession']} ({row['annotation']}): "
              f"{row['lineage']} = {row['mean_count']:.2f}")

    print(f"\nTimestamp: {ts}")
    print("Done.")

if __name__ == '__main__':
    main()
