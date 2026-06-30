#!/usr/bin/env python3
"""
Synechococcus dark-vs-white dN/dS comparison.

Source data: Tai et al. 2011 (PLoS ONE 6:e24249)
  - Table S3: CC9311 per-gene dN/dS
  - Table S4: CC9902 per-gene dN/dS
Pfam annotation: hmmsearch (HMMER 3.3.2) vs Pfam-A.hmm, domain i-Evalue < 1e-9
  - CC9311 proteome: NCBI GCF_000014585.1
  - CC9902 proteome: NCBI GCF_000012505.1
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVALUE_THRESHOLD = 1e-9
N_BOOTSTRAP = 10000
SEED = 42


def parse_domtbl(path):
    """Parse hmmsearch --domtblout, return set of proteins with Pfam at E < threshold."""
    proteins = set()
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) >= 13:
                i_evalue = float(fields[12])
                if i_evalue <= EVALUE_THRESHOLD:
                    proteins.add(fields[0])
    return proteins


def parse_dnds_table(path):
    """Parse Tai et al. supplementary dN/dS table (Excel)."""
    df = pd.read_excel(path, header=None, skiprows=3)
    df.columns = [
        'Locus_Tag', 'Description', 'dN_dS', 'dN', 'dS',
        'pct_min5Xcov', 'pct_min1Xcov', 'avg_read_depth',
        'dN_dS_vs_ref', 'dN_dS_70_70', 'dN_dS_PAML'
    ]
    for col in ['dN_dS', 'dN', 'dS', 'pct_min5Xcov']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def analyze_strain(label, xls_path, domtbl_path):
    """Run dark-vs-white dN/dS comparison for one strain."""
    pfam_set = parse_domtbl(domtbl_path)
    df = parse_dnds_table(xls_path)

    df['has_pfam'] = df['Locus_Tag'].isin(pfam_set)
    mask = df['dN_dS'].notna() & np.isfinite(df['dN_dS']) & (df['pct_min5Xcov'] > 0)
    df = df[mask].copy()

    dark = df[~df['has_pfam']]['dN_dS']
    white = df[df['has_pfam']]['dN_dS']

    print(f'\n{"="*60}')
    print(f'{label}')
    print(f'{"="*60}')
    print(f'Dark (no Pfam):  n={len(dark)}, median={dark.median():.4f}, '
          f'IQR=[{dark.quantile(0.25):.4f}, {dark.quantile(0.75):.4f}]')
    print(f'White (Pfam):    n={len(white)}, median={white.median():.4f}, '
          f'IQR=[{white.quantile(0.25):.4f}, {white.quantile(0.75):.4f}]')

    fold = dark.median() / white.median()
    u_stat, p_val = stats.mannwhitneyu(dark, white, alternative='greater')
    print(f'Fold (dark/white): {fold:.2f}x')
    print(f'Mann-Whitney U: {u_stat:.0f}, p = {p_val:.2e}')

    dark_genes = df[~df['has_pfam']]
    white_genes = df[df['has_pfam']]
    agg_dark = dark_genes['dN'].sum() / dark_genes['dS'].sum()
    agg_white = white_genes['dN'].sum() / white_genes['dS'].sum()
    print(f'Aggregate dN/dS: dark={agg_dark:.4f}, white={agg_white:.4f}')

    dark_pos = (dark > 1).sum()
    white_pos = (white > 1).sum()
    print(f'Positively selected (dN/dS > 1): dark={dark_pos}/{len(dark)} '
          f'({100*dark_pos/len(dark):.1f}%), white={white_pos}/{len(white)} '
          f'({100*white_pos/len(white):.1f}%)')

    table = [[dark_pos, len(dark) - dark_pos], [white_pos, len(white) - white_pos]]
    odds, fp = stats.fisher_exact(table, alternative='greater')
    print(f'Fisher exact (dark enrichment): OR={odds:.1f}, p={fp:.2e}')

    np.random.seed(SEED)
    boot_ratios = []
    for _ in range(N_BOOTSTRAP):
        d = np.random.choice(dark.values, size=len(dark), replace=True)
        w = np.random.choice(white.values, size=len(white), replace=True)
        if np.median(w) > 0:
            boot_ratios.append(np.median(d) / np.median(w))
    ci = np.percentile(boot_ratios, [2.5, 97.5])
    print(f'Bootstrap 95% CI on fold: [{ci[0]:.2f}x, {ci[1]:.2f}x]')

    return dark.values, white.values


if __name__ == '__main__':
    d1, w1 = analyze_strain(
        'CC9311 (Clade I)',
        os.path.join(SCRIPT_DIR, 'table_s3_cc9311_dnds.xls'),
        os.path.join(SCRIPT_DIR, 'cc9311_pfam_domtbl.txt')
    )
    d2, w2 = analyze_strain(
        'CC9902 (Clade IV)',
        os.path.join(SCRIPT_DIR, 'table_s4_cc9902_dnds.xls'),
        os.path.join(SCRIPT_DIR, 'cc9902_pfam_domtbl.txt')
    )

    all_dark = np.concatenate([d1, d2])
    all_white = np.concatenate([w1, w2])
    print(f'\n{"="*60}')
    print(f'COMBINED')
    print(f'{"="*60}')
    print(f'Dark: n={len(all_dark)}, median={np.median(all_dark):.4f}')
    print(f'White: n={len(all_white)}, median={np.median(all_white):.4f}')
    print(f'Fold: {np.median(all_dark)/np.median(all_white):.2f}x')
    u, p = stats.mannwhitneyu(all_dark, all_white, alternative='greater')
    print(f'Mann-Whitney: U={u:.0f}, p={p:.2e}')
