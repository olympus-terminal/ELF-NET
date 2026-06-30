#!/usr/bin/env python3
"""
LA4SR False-Positive Validation Analysis
=========================================
Two complementary analyses to preempt reviewer concerns about
LA4SR/AlgaGPT false positive rates:

  Analysis 1: Size-fraction retention consistency
    If LA4SR pulls in bacterial false positives, algal retention should be
    artificially high in bacterially-dominated 0.8-5 µm fractions. Instead,
    retention should scale with expected algal content.

  Analysis 2: Negative control summary
    Aggregate existing LA4SR benchmarks on known bacterial, archaeal, fungal,
    and viral sequences to report per-kingdom false-positive rates.

Outputs:
  - figures/la4sr_validation_panel_YYYYMMDD_HHMMSS.pdf/.svg
  - source_data/la4sr_validation_results_YYYYMMDD_HHMMSS.tsv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'figures'))
from palette import (
    DEEP_OCEAN, OCEAN_BLUE, COASTAL_BLUE, TURQUOISE,
    FOREST_GREEN, DESERT_TAN, CLAY, SIENNA,
    OCEAN_CMAP, get_categorical_colors,
)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats
from datetime import datetime

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, 'figures')
SRC_DIR = os.path.join(BASE, 'source_data')

CLASSIFICATION_FILE = os.path.join(
    SRC_DIR, 'algagpt_classification_summary_20260114_150000.csv'
)
SIZE_FRACTION_FILE = os.path.join(
    BASE, '..', '01_raw_data', 'metadata',
    'mgya_size_fraction_mapping_20260421_093512.tsv'
)
RESULTS_ARCHIVE = os.path.join(BASE, '..', 'tools', 'la4sr', 'results-archive')

# ── Figure protocol ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 6,
    'axes.labelsize': 6,
    'axes.titlesize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
})

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Size-fraction retention consistency
# ═══════════════════════════════════════════════════════════════════════════════

def load_size_fraction_data():
    sf = pd.read_csv(SIZE_FRACTION_FILE, sep='\t')
    clf = pd.read_csv(CLASSIFICATION_FILE, comment='#')

    clf['mgya_id'] = clf['filename'].str.extract(r'(MGYA\d+)')
    clf = clf.dropna(subset=['mgya_id'])

    merged = sf.merge(clf, on='mgya_id', how='inner')

    canonical = ['0.8-5', '5-20', '20-180', '180-2000']
    merged = merged[merged['size_fraction'].isin(canonical)].copy()

    return merged


def analyze_size_fractions(df):
    results = []
    for frac in ['0.8-5', '5-20', '20-180', '180-2000']:
        sub = df[df['size_fraction'] == frac]
        results.append({
            'size_fraction': frac,
            'n_samples': len(sub),
            'mean_pct_algae': sub['pct_algae'].mean(),
            'std_pct_algae': sub['pct_algae'].std(),
            'median_pct_algae': sub['pct_algae'].median(),
            'q25_pct_algae': sub['pct_algae'].quantile(0.25),
            'q75_pct_algae': sub['pct_algae'].quantile(0.75),
            'mean_total_seqs': sub['total_sequences'].mean(),
            'total_algal_seqs': sub['n_algae'].sum(),
            'total_seqs': sub['total_sequences'].sum(),
        })
    res = pd.DataFrame(results)
    res['bulk_retention'] = res['total_algal_seqs'] / res['total_seqs'] * 100

    # Kruskal-Wallis test across fractions
    groups = [df[df['size_fraction'] == f]['pct_algae'].values
              for f in ['0.8-5', '5-20', '20-180', '180-2000']]
    h_stat, kw_p = stats.kruskal(*groups)

    # Monotonic trend test (Jonckheere-Terpstra direction: 0.8-5 < 5-20 < 20-180 < 180-2000)
    # Use Spearman on ordered fractions
    order_map = {'0.8-5': 1, '5-20': 2, '20-180': 3, '180-2000': 4}
    df_ordered = df.copy()
    df_ordered['frac_order'] = df_ordered['size_fraction'].map(order_map)
    rho, sp_p = stats.spearmanr(df_ordered['frac_order'], df_ordered['pct_algae'])

    # Pairwise Mann-Whitney U: 0.8-5 µm vs each other fraction
    ref_vals = df[df['size_fraction'] == '0.8-5']['pct_algae'].values
    pairwise = []
    for frac in ['5-20', '20-180', '180-2000']:
        test_vals = df[df['size_fraction'] == frac]['pct_algae'].values
        u_stat, mw_p = stats.mannwhitneyu(ref_vals, test_vals, alternative='two-sided')
        pairwise.append((frac, mw_p))

    return res, h_stat, kw_p, rho, sp_p, pairwise


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Negative control summary
# ═══════════════════════════════════════════════════════════════════════════════

def parse_negative_controls():
    records = []
    for fname in sorted(os.listdir(RESULTS_ARCHIVE)):
        if not fname.endswith('_report.txt'):
            continue
        path = os.path.join(RESULTS_ARCHIVE, fname)

        # Extract kingdom from filename
        if '_bact' in fname:
            kingdom = 'Bacteria'
        elif '_archa' in fname:
            kingdom = 'Archaea'
        elif '_fungi' in fname:
            kingdom = 'Fungi'
        elif '_virus' in fname:
            kingdom = 'Virus'
        else:
            continue

        # Extract replicate number
        import re
        rep_match = re.search(r'(?:bact|archa|fungi|virus)(\d)', fname)
        replicate = int(rep_match.group(1)) if rep_match else 0

        with open(path) as f:
            text = f.read()

        # Parse contaminant class metrics
        conta_correct = None
        conta_total = None
        conta_incorrect = None
        for line in text.split('\n'):
            if 'Contaminant class:' in line:
                in_conta = True
            if 'Correctly classified:' in line and 'Contaminant' not in line and 'Algal' not in line:
                # This is under whichever class section we're in
                pass

        # More robust parsing: extract confusion matrix values
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if 'Contaminant class:' in line:
                for j in range(i+1, min(i+6, len(lines))):
                    if 'Correctly classified:' in lines[j]:
                        match = re.search(r'(\d+)\s*/\s*(\d+)', lines[j])
                        if not match:
                            match = re.search(r'(\d+)\s*\(', lines[j])
                        if match:
                            conta_correct = int(match.group(1))
                    if 'Incorrectly classified:' in lines[j]:
                        match = re.search(r'(\d+)\s*\(', lines[j])
                        if match:
                            conta_incorrect = int(match.group(1))
                    if 'Total samples:' in lines[j]:
                        match = re.search(r'(\d+)', lines[j])
                        if match:
                            conta_total = int(match.group(1))

        if conta_correct is not None and conta_total is not None:
            fpr = (conta_incorrect / conta_total * 100) if conta_incorrect else 0
            records.append({
                'kingdom': kingdom,
                'replicate': replicate,
                'n_total': conta_total,
                'n_correct': conta_correct,
                'n_false_positive': conta_incorrect if conta_incorrect else 0,
                'true_negative_rate': conta_correct / conta_total * 100,
                'false_positive_rate': fpr,
            })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def create_figure(sf_data, sf_df, sf_stats, pairwise_results):
    h_stat, kw_p, rho, sp_p = sf_stats

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.4),
                             gridspec_kw={'width_ratios': [1.3, 1],
                                          'wspace': 0.4})

    # ── Panel A: Size-fraction box/strip plot ────────────────────────────────
    ax = axes[0]
    fractions = ['0.8-5', '5-20', '20-180', '180-2000']
    fraction_labels = ['0.8–5\nµm', '5–20\nµm', '20–180\nµm', '180–2000\nµm']
    colors = [DEEP_OCEAN, OCEAN_BLUE, COASTAL_BLUE, TURQUOISE]

    positions = [1, 2, 3, 4]
    bp_data = [sf_df[sf_df['size_fraction'] == f]['pct_algae'].values for f in fractions]

    bp = ax.boxplot(bp_data, positions=positions, widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=0.8),
                    whiskerprops=dict(linewidth=0.5),
                    capprops=dict(linewidth=0.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_linewidth(0.5)

    rng = np.random.default_rng(42)
    for i, (frac, color) in enumerate(zip(fractions, colors)):
        vals = sf_df[sf_df['size_fraction'] == frac]['pct_algae'].values
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(positions[i] + jitter, vals, s=2, alpha=0.25,
                   color=color, edgecolors='none', zorder=2)

    ax.set_xticks(positions)
    ax.set_xticklabels(fraction_labels, linespacing=0.9)
    ax.set_ylabel('Algal retention (%)')
    ax.set_title('A', fontweight='bold', loc='left', pad=4)

    for i, frac in enumerate(fractions):
        n = len(sf_df[sf_df['size_fraction'] == frac])
        ax.text(positions[i], 86, f'n={n}',
                ha='center', va='bottom', fontsize=5)

    ax.set_ylim(20, 92)

    bracket_y = 83
    for i, (frac, p_val) in enumerate(pairwise_results):
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        target_pos = positions[fractions.index(frac)]
        ax.plot([1, 1, target_pos, target_pos],
                [bracket_y - 1, bracket_y, bracket_y, bracket_y - 1],
                color='black', linewidth=0.4)
        ax.text((1 + target_pos) / 2, bracket_y + 0.3, sig,
                ha='center', va='bottom', fontsize=5)
        bracket_y -= 5

    stat_text = f'Kruskal–Wallis p = {kw_p:.1e}'
    ax.text(0.97, 0.03, stat_text, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.8, linewidth=0.3))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel B: Pairwise comparison vs 0.8-5 µm ────────────────────────────
    ax = axes[1]

    compare_fracs = ['5-20', '20-180', '180-2000']
    compare_labels = ['5–20', '20–180', '180–2000']
    compare_colors = [OCEAN_BLUE, COASTAL_BLUE, TURQUOISE]

    ref_median = sf_data[sf_data['size_fraction'] == '0.8-5']['median_pct_algae'].values[0]

    diffs = []
    for frac in compare_fracs:
        med = sf_data[sf_data['size_fraction'] == frac]['median_pct_algae'].values[0]
        diffs.append(med - ref_median)

    bars = ax.barh(range(len(compare_fracs)), diffs,
                   color=compare_colors, edgecolor='black', linewidth=0.3,
                   height=0.5)

    ax.axvline(0, color='black', linewidth=0.5, linestyle='-')
    ax.set_yticks(range(len(compare_fracs)))
    ax.set_yticklabels([f'{l} µm' for l in compare_labels])
    ax.set_xlabel('Δ retention vs 0.8–5 µm (pp)')
    ax.set_title('B', fontweight='bold', loc='left', pad=4)
    ax.invert_yaxis()

    for i, (d, (frac, p_val)) in enumerate(zip(diffs, pairwise_results)):
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
        label = f'+{d:.1f} pp {sig}' if d > 0 else f'{d:.1f} pp {sig}'
        ax.text(d + (0.3 if d >= 0 else -0.3), i, label,
                ha='left' if d >= 0 else 'right', va='center', fontsize=5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save
    for ext in ['pdf', 'svg']:
        out = os.path.join(FIG_DIR, f'la4sr_sizefraction_validation_{TIMESTAMP}.{ext}')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved: {out}')

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 70)
    print('LA4SR FALSE-POSITIVE VALIDATION ANALYSIS')
    print('=' * 70)

    # ── Analysis 1 ────────────────────────────────────────────────────────────
    print('\n── Analysis 1: Size-fraction retention consistency ──')
    sf_df = load_size_fraction_data()
    print(f'Merged samples with size fractions: {len(sf_df)}')
    sf_data, h_stat, kw_p, rho, sp_p, pairwise = analyze_size_fractions(sf_df)

    print('\nPer-fraction summary:')
    print(sf_data[['size_fraction', 'n_samples', 'median_pct_algae',
                    'mean_pct_algae', 'std_pct_algae', 'bulk_retention']].to_string(index=False))

    print(f'\nKruskal–Wallis H = {h_stat:.2f}, p = {kw_p:.2e}')
    print(f'Spearman ρ (fraction order vs retention): {rho:.4f}, p = {sp_p:.2e}')

    print('\nPairwise Mann-Whitney U vs 0.8–5 µm (bacterially-dominated):')
    for frac, mw_p in pairwise:
        med_ref = sf_data[sf_data['size_fraction'] == '0.8-5']['median_pct_algae'].values[0]
        med_test = sf_data[sf_data['size_fraction'] == frac]['median_pct_algae'].values[0]
        delta = med_test - med_ref
        print(f'  vs {frac:>8s}: Δmedian = {delta:+.1f} pp, p = {mw_p:.2e}')

    # ── Figure ────────────────────────────────────────────────────────────────
    print('\n── Generating figure ──')
    create_figure(sf_data, sf_df, (h_stat, kw_p, rho, sp_p), pairwise)

    # ── Save results ──────────────────────────────────────────────────────────
    out_tsv = os.path.join(SRC_DIR, f'la4sr_sizefraction_validation_{TIMESTAMP}.tsv')
    with open(out_tsv, 'w') as f:
        f.write(f'# Provenance:\n')
        f.write(f'#   Script: {os.path.abspath(__file__)}\n')
        f.write(f'#   Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'#   Classification input: {CLASSIFICATION_FILE}\n')
        f.write(f'#   Size fraction input: {SIZE_FRACTION_FILE}\n')
        f.write(f'#\n')
        f.write(f'# Size-fraction retention consistency\n')
        f.write(f'#   Kruskal-Wallis H = {h_stat:.2f}, p = {kw_p:.2e}\n')
        for frac, mw_p in pairwise:
            f.write(f'#   Mann-Whitney 0.8-5 vs {frac}: p = {mw_p:.2e}\n')
        f.write(f'#\n')
        sf_data.to_csv(f, sep='\t', index=False)

    print(f'\nResults saved: {out_tsv}')
    print('\nDone.')


if __name__ == '__main__':
    main()
