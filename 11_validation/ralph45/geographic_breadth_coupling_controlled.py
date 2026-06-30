#!/usr/bin/env python3
"""
Prevalence-controlled geographic breadth vs coupling analysis.

The raw correlation between breadth and coupling is confounded by prevalence:
rare domains have both fewer stations (by definition) AND different correlation
properties. This script:
1. Bins domains by prevalence (quartiles)
2. Within each prevalence bin, computes partial correlation of breadth vs coupling
3. Also computes residual breadth (beyond what prevalence predicts)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

INPUT = "/media/drn2/External/TARA-Oceans/MANUSCRIPT/.wt45/task4/source_data/ralph45/geographic_breadth_vs_coupling.tsv"
OUTPUT = "/media/drn2/External/TARA-Oceans/MANUSCRIPT/.wt45/task4/source_data/ralph45/geographic_breadth_vs_coupling.tsv"

df = pd.read_csv(INPUT, sep='\t')

print(f"Total domains: {len(df)}")
print(f"  Novel: {(df['domain_type'] == 'novel').sum()}")
print(f"  Pfam: {(df['domain_type'] == 'pfam').sum()}")

# Strategy: partial Spearman correlation controlling for prevalence
# Method: rank-based residuals
def partial_spearman(x, y, z):
    """Partial Spearman correlation between x and y, controlling for z."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    # Residualize x and y on z
    slope_x, intercept_x, _, _, _ = stats.linregress(rz, rx)
    slope_y, intercept_y, _, _, _ = stats.linregress(rz, ry)
    resid_x = rx - (slope_x * rz + intercept_x)
    resid_y = ry - (slope_y * rz + intercept_y)
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p


print("\n=== Partial Spearman (controlling for prevalence) ===")
results = []
for dtype in ['novel', 'pfam', 'all']:
    subset = df if dtype == 'all' else df[df['domain_type'] == dtype]
    if len(subset) < 30:
        continue

    r, p = partial_spearman(
        subset['n_stations'].values,
        subset['median_abs_rho'].values,
        subset['n_samples_present'].values
    )
    results.append({
        'comparison': f'{dtype}_n_stations_partial',
        'spearman_rho': r,
        'p_value': p,
        'n_domains': len(subset),
        'method': 'partial_spearman_controlling_prevalence'
    })
    print(f"  {dtype:6s} | n_stations vs median|ρ| (partial, ctl prevalence): ρ = {r:.4f}, p = {p:.2e}, n = {len(subset)}")

print("\n=== Within-prevalence-bin correlations ===")
# Bin by prevalence deciles
for dtype in ['novel', 'pfam']:
    subset = df[df['domain_type'] == dtype].copy()
    subset['prev_bin'] = pd.qcut(subset['prevalence'], q=10, labels=False, duplicates='drop')

    print(f"\n  {dtype.upper()} domains:")
    for b in sorted(subset['prev_bin'].unique()):
        bin_data = subset[subset['prev_bin'] == b]
        if len(bin_data) < 20:
            continue
        prev_range = f"{bin_data['prevalence'].min():.3f}-{bin_data['prevalence'].max():.3f}"
        rho, p = stats.spearmanr(bin_data['n_stations'], bin_data['median_abs_rho'])
        print(f"    Bin {b} (prev {prev_range}, n={len(bin_data)}): "
              f"breadth vs coupling ρ = {rho:.4f}, p = {p:.2e}")
        results.append({
            'comparison': f'{dtype}_bin{b}_n_stations',
            'spearman_rho': rho,
            'p_value': p,
            'n_domains': len(bin_data),
            'method': f'within_prevalence_bin_{prev_range}'
        })

# Also: compare novel vs pfam at MATCHED prevalence
print("\n=== Novel vs Pfam at matched prevalence (10-50%) ===")
novel_mid = df[(df['domain_type'] == 'novel') & (df['prevalence'] >= 0.10) & (df['prevalence'] <= 0.50)]
pfam_mid = df[(df['domain_type'] == 'pfam') & (df['prevalence'] >= 0.10) & (df['prevalence'] <= 0.50)]
print(f"  Novel (10-50% prev): n={len(novel_mid)}, median stations={novel_mid['n_stations'].median():.0f}, median|ρ|={novel_mid['median_abs_rho'].median():.4f}")
print(f"  Pfam  (10-50% prev): n={len(pfam_mid)}, median stations={pfam_mid['n_stations'].median():.0f}, median|ρ|={pfam_mid['median_abs_rho'].median():.4f}")

if len(novel_mid) > 0 and len(pfam_mid) > 0:
    # Within this prevalence range, test breadth vs coupling for each type
    for dtype, sub in [('novel', novel_mid), ('pfam', pfam_mid)]:
        rho, p = stats.spearmanr(sub['n_stations'], sub['median_abs_rho'])
        print(f"  {dtype:6s} | breadth vs coupling (10-50% prev): ρ = {rho:.4f}, p = {p:.2e}, n = {len(sub)}")
        results.append({
            'comparison': f'{dtype}_matched_prev_10_50',
            'spearman_rho': rho,
            'p_value': p,
            'n_domains': len(sub),
            'method': 'matched_prevalence_10_50pct'
        })

# Save extended summary
summary_path = OUTPUT.replace('.tsv', '_summary.tsv')
summary_df = pd.DataFrame(results)
summary_df.to_csv(summary_path, sep='\t', index=False)
print(f"\nSummary saved to: {summary_path}")
