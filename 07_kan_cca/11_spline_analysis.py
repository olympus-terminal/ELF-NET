#!/usr/bin/env python3
"""
11_spline_analysis.py — Analyze KAN spline activations for nonlinearity.

Reads spline activation data from kan_cca_spline_activations TSV,
computes curvature (2nd derivative) metrics, identifies most nonlinear
features, and outputs interpretability results.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent.parent / 'kan_cca_results'
SPLINE_FILE = RESULTS_DIR / 'kan_cca_spline_activations_20260222_190340.tsv'

def compute_nonlinearity(x_vals, y_vals):
    """Compute nonlinearity metrics for a single spline.

    Returns:
        linearity_r2: R^2 of linear fit (1.0 = perfectly linear)
        max_curvature: max absolute 2nd derivative
        mean_curvature: mean absolute 2nd derivative
        amplitude: max(y) - min(y)
    """
    x = np.array(x_vals)
    y = np.array(y_vals)

    # Linear fit
    coeffs = np.polyfit(x, y, 1)
    y_linear = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_linear) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0

    # 2nd derivative (curvature proxy)
    dx = np.diff(x)
    dy = np.diff(y)
    d1 = dy / dx  # first derivative
    d2 = np.diff(d1) / ((dx[:-1] + dx[1:]) / 2)  # second derivative

    max_curv = np.max(np.abs(d2)) if len(d2) > 0 else 0.0
    mean_curv = np.mean(np.abs(d2)) if len(d2) > 0 else 0.0
    amplitude = np.max(y) - np.min(y)

    return r2, max_curv, mean_curv, amplitude

def main():
    print("=== KAN Spline Activation Analysis ===\n")

    # Read spline data
    df = pd.read_csv(SPLINE_FILE, sep='\t')
    print(f"Loaded {len(df)} spline records")
    print(f"Columns: {list(df.columns)}")
    print(f"Folds: {sorted(df['fold'].unique())}")
    print(f"Branches: {sorted(df['branch'].unique())}")

    # Parse x_vals and y_vals
    records = []
    for _, row in df.iterrows():
        x_vals = [float(v) for v in row['x_vals'].split(',')]
        y_vals = [float(v) for v in row['y_vals'].split(',')]
        r2, max_curv, mean_curv, amplitude = compute_nonlinearity(x_vals, y_vals)

        records.append({
            'fold': row['fold'],
            'branch': row['branch'],
            'input_idx': row['input_idx'],
            'feature_name': row['feature_name'],
            'output_idx': row['output_idx'],
            'linearity_r2': r2,
            'max_curvature': max_curv,
            'mean_curvature': mean_curv,
            'amplitude': amplitude,
        })

    metrics = pd.DataFrame(records)

    # Average across folds (for each branch, feature, output)
    avg = metrics.groupby(['branch', 'feature_name', 'output_idx']).agg({
        'linearity_r2': 'mean',
        'max_curvature': 'mean',
        'mean_curvature': 'mean',
        'amplitude': 'mean',
    }).reset_index()

    # Separate domain and env
    for branch in ['domain', 'env']:
        branch_data = avg[avg['branch'] == branch].copy()

        # Aggregate across output indices (take mean across 3 components)
        feat_avg = branch_data.groupby('feature_name').agg({
            'linearity_r2': 'mean',
            'max_curvature': 'mean',
            'mean_curvature': 'mean',
            'amplitude': 'mean',
        }).reset_index()

        # Sort by nonlinearity (lowest R^2 = most nonlinear)
        feat_avg = feat_avg.sort_values('linearity_r2')

        print(f"\n=== {branch.upper()} Features — Nonlinearity Ranking ===")
        print(f"{'Feature':<25s} {'R²(lin)':>8s} {'MaxCurv':>10s} {'MeanCurv':>10s} {'Amplitude':>10s}")
        for _, row in feat_avg.iterrows():
            print(f"{row['feature_name']:<25s} {row['linearity_r2']:8.4f} {row['max_curvature']:10.4f} {row['mean_curvature']:10.4f} {row['amplitude']:10.4f}")

        # Top-5 most nonlinear
        top5 = feat_avg.head(5)
        print(f"\nTop-5 most nonlinear {branch} features:")
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            classification = "highly nonlinear" if row['linearity_r2'] < 0.5 else \
                           "moderately nonlinear" if row['linearity_r2'] < 0.8 else \
                           "near-linear"
            print(f"  {i}. {row['feature_name']}: R²={row['linearity_r2']:.4f} ({classification}), "
                  f"mean curvature={row['mean_curvature']:.4f}, amplitude={row['amplitude']:.4f}")

    # Per-component analysis for output_idx=0 (component 1)
    print("\n\n=== Per-Component Nonlinearity (domain branch) ===")
    domain_data = avg[avg['branch'] == 'domain']
    for comp in sorted(domain_data['output_idx'].unique()):
        comp_data = domain_data[domain_data['output_idx'] == comp].sort_values('linearity_r2')
        mean_r2 = comp_data['linearity_r2'].mean()
        n_nonlinear = (comp_data['linearity_r2'] < 0.8).sum()
        n_highly = (comp_data['linearity_r2'] < 0.5).sum()
        print(f"\n  Component {comp+1}: mean R²={mean_r2:.4f}, "
              f"{n_nonlinear}/{len(comp_data)} nonlinear (R²<0.8), "
              f"{n_highly}/{len(comp_data)} highly nonlinear (R²<0.5)")
        for _, row in comp_data.head(3).iterrows():
            print(f"    {row['feature_name']}: R²={row['linearity_r2']:.4f}, "
                  f"curv={row['mean_curvature']:.4f}, amp={row['amplitude']:.4f}")

    # Cross-fold stability of nonlinearity
    print("\n\n=== Cross-Fold Stability of Nonlinearity ===")
    domain_metrics = metrics[metrics['branch'] == 'domain']
    for feat in sorted(domain_metrics['feature_name'].unique()):
        feat_data = domain_metrics[domain_metrics['feature_name'] == feat]
        r2_vals = feat_data.groupby('fold')['linearity_r2'].mean()
        print(f"  {feat}: R²={r2_vals.mean():.4f} ± {r2_vals.std():.4f} "
              f"(range [{r2_vals.min():.4f}, {r2_vals.max():.4f}])")

    # Summary statistics
    all_domain = avg[avg['branch'] == 'domain']
    all_env = avg[avg['branch'] == 'env']
    print(f"\n\n=== SUMMARY ===")
    print(f"Domain splines: {len(all_domain)} (features × components)")
    print(f"  Mean linearity R²: {all_domain['linearity_r2'].mean():.4f}")
    print(f"  Fraction nonlinear (R²<0.8): {(all_domain['linearity_r2'] < 0.8).sum()}/{len(all_domain)}")
    print(f"  Fraction highly nonlinear (R²<0.5): {(all_domain['linearity_r2'] < 0.5).sum()}/{len(all_domain)}")
    print(f"Env splines: {len(all_env)} (features × components)")
    print(f"  Mean linearity R²: {all_env['linearity_r2'].mean():.4f}")
    print(f"  Fraction nonlinear (R²<0.8): {(all_env['linearity_r2'] < 0.8).sum()}/{len(all_env)}")
    print(f"  Fraction highly nonlinear (R²<0.5): {(all_env['linearity_r2'] < 0.5).sum()}/{len(all_env)}")

if __name__ == '__main__':
    main()
