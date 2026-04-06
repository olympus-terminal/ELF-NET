#!/usr/bin/env python3
"""
Profile the SHAP importance distribution to justify the top-N domain cutoff
used in Figure 4A.

Outputs:
  - Total number of Pfam domains with SHAP scores
  - Distribution statistics (mean, median, percentiles)
  - Cumulative SHAP mass captured by top N domains
  - Location of natural breakpoints (elbow detection)
  - Saves a diagnostic plot to figures/shap_distribution_profile.png
"""

import socket
from pathlib import Path
import pandas as pd
import numpy as np

def get_base_dir(project_name: str) -> Path:
    hostname = socket.gethostname()
    if "cn" in hostname or "gpu" in hostname or "jubail" in hostname:
        return Path(f"/scratch/drn2/PROJECTS/{project_name}")
    return Path(f"/media/drn2/External/{project_name}")

BASE = get_base_dir("TARA-Oceans")
SHAP_FILE = BASE / "03_analyses" / "ALGAGPT-based-analyses" / "validations" / "results" / "shap_dependence_gee_20260119_194303.tsv"
OUT_DIR = Path(__file__).resolve().parent.parent / "source_data"
OUT_DIR.mkdir(exist_ok=True)

print(f"Reading: {SHAP_FILE}")
shap = pd.read_csv(SHAP_FILE, sep='\t', comment='#')
print(f"  {len(shap):,} rows, {shap['pfam'].nunique():,} unique Pfam domains, {shap['variable'].nunique()} variables")
print(f"  Columns: {list(shap.columns)}")

# Reproduce the exact normalization from the figure script
shap['norm_shap'] = shap.groupby('variable')['mean_abs_shap'].transform(
    lambda x: x / x.max() if x.max() > 0 else 0
)
pfam_shap = shap.groupby('pfam')['norm_shap'].mean().sort_values(ascending=False)

total_domains = len(pfam_shap)
print(f"\n=== Distribution of mean normalized |SHAP| across {total_domains:,} Pfam domains ===")
print(f"  Max:    {pfam_shap.iloc[0]:.6f}")
print(f"  Mean:   {pfam_shap.mean():.6f}")
print(f"  Median: {pfam_shap.median():.6f}")
print(f"  Min:    {pfam_shap.iloc[-1]:.6f}")
print(f"  Std:    {pfam_shap.std():.6f}")

# Percentiles
for p in [99, 95, 90, 75, 50, 25, 10, 5, 1]:
    val = pfam_shap.quantile(1 - p/100)  # top p%
    n_above = (pfam_shap >= val).sum()
    print(f"  Top {p:>2d}%: >= {val:.6f} ({n_above:,} domains)")

# Cumulative mass analysis
cumsum = pfam_shap.cumsum()
total_mass = cumsum.iloc[-1]
print(f"\n=== Cumulative SHAP mass ===")
print(f"  Total mass: {total_mass:.4f}")
for n in [50, 100, 200, 300, 500, 750, 1000, 1500, 2000]:
    if n <= total_domains:
        mass_n = cumsum.iloc[min(n-1, total_domains-1)]
        pct = 100 * mass_n / total_mass
        print(f"  Top {n:>5d}: {mass_n:.4f} ({pct:.1f}% of total)")

# Elbow detection using second derivative
vals = pfam_shap.values
if len(vals) > 10:
    # Smooth with window for stability
    window = max(5, len(vals) // 100)
    smoothed = pd.Series(vals).rolling(window, center=True, min_periods=1).mean().values
    d1 = np.diff(smoothed)
    d2 = np.diff(d1)
    # Find where second derivative is most negative (sharpest bend)
    elbow_idx = np.argmin(d2) + 2  # offset for diff
    print(f"\n=== Elbow detection ===")
    print(f"  Sharpest elbow at rank {elbow_idx} (SHAP = {vals[min(elbow_idx, len(vals)-1)]:.6f})")

    # Also find where domains drop below 1% of max
    threshold_1pct = vals[0] * 0.01
    below_1pct = np.where(vals < threshold_1pct)[0]
    if len(below_1pct) > 0:
        print(f"  Domains drop below 1% of max at rank {below_1pct[0]} (SHAP = {vals[below_1pct[0]]:.6f})")

    threshold_5pct = vals[0] * 0.05
    below_5pct = np.where(vals < threshold_5pct)[0]
    if len(below_5pct) > 0:
        print(f"  Domains drop below 5% of max at rank {below_5pct[0]} (SHAP = {vals[below_5pct[0]]:.6f})")

# Save rank table for reference
rank_df = pd.DataFrame({
    'rank': range(1, total_domains + 1),
    'pfam': pfam_shap.index,
    'mean_norm_shap': pfam_shap.values,
    'cumulative_mass': cumsum.values,
    'cumulative_pct': 100 * cumsum.values / total_mass
})
outfile = OUT_DIR / "shap_importance_rank_profile.tsv"
rank_df.to_csv(outfile, sep='\t', index=False)
print(f"\nRank table saved to: {outfile}")

# Try to make a plot (may fail on headless HPC nodes without display)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: Ranked SHAP values
    ax = axes[0]
    ax.plot(range(1, total_domains + 1), vals, 'k-', linewidth=0.5)
    ax.axvline(500, color='red', linestyle='--', alpha=0.7, label='N=500 (current)')
    if len(below_1pct) > 0:
        ax.axvline(below_1pct[0], color='blue', linestyle=':', alpha=0.7, label=f'1% of max (N={below_1pct[0]})')
    ax.set_xlabel('Domain rank')
    ax.set_ylabel('Mean normalized |SHAP|')
    ax.set_title('Ranked domain importance')
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(2000, total_domains))

    # Panel 2: Cumulative mass
    ax = axes[1]
    ax.plot(range(1, total_domains + 1), 100 * cumsum.values / total_mass, 'k-', linewidth=1)
    ax.axvline(500, color='red', linestyle='--', alpha=0.7, label='N=500')
    mass_at_500 = 100 * cumsum.iloc[min(499, total_domains-1)] / total_mass
    ax.axhline(mass_at_500, color='red', linestyle=':', alpha=0.3)
    ax.set_xlabel('Top N domains')
    ax.set_ylabel('Cumulative % of total SHAP mass')
    ax.set_title(f'Cumulative importance (500 → {mass_at_500:.1f}%)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(2000, total_domains))

    # Panel 3: Zoomed log-scale
    ax = axes[2]
    ax.semilogy(range(1, total_domains + 1), vals, 'k-', linewidth=0.5)
    ax.axvline(500, color='red', linestyle='--', alpha=0.7, label='N=500')
    ax.set_xlabel('Domain rank')
    ax.set_ylabel('Mean normalized |SHAP| (log)')
    ax.set_title('Log-scale importance decay')
    ax.legend(fontsize=8)

    plt.tight_layout()
    figpath = Path(__file__).resolve().parent.parent / "figures" / "shap_distribution_profile.png"
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {figpath}")
except Exception as e:
    print(f"Plot skipped ({e})")

print("\nDone.")
