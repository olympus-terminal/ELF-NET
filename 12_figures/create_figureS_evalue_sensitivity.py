#!/usr/bin/env python3
"""
Create Figure S10: E-value Threshold Sensitivity Analysis

Provenance:
    Script: figures/create_figureS_evalue_sensitivity.py
    Generated: 2026-03-20

Purpose:
    Generate a 2x2 panel supplemental figure showing that the enhanced
    environmental coupling of novel domains is robust to E-value threshold.

Panels:
    (A) Domains retained at each threshold (total and prevalent)
    (B) Significance rates (% FDR < 0.05) for Pfam + 3 novel thresholds
    (C) Effect size distributions (|rho|) as violin/box plots
    (D) Fold-enrichment vs Pfam at each threshold

Input:
    source_data/dark_proteome/evalue_sensitivity_comparison.tsv
    (+ per_threshold_details/ for panel C violin data)

Output:
    figures/FigureS10_evalue_sensitivity_YYYYMMDD_HHMMSS.pdf

Usage:
    python3 figures/create_figureS_evalue_sensitivity.py
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas required")
    sys.exit(1)

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SOURCE_DATA = PROJECT_DIR / "source_data" / "dark_proteome"
DETAIL_DIR = SOURCE_DATA / "sensitivity_per_threshold_details"

# Try palette import
sys.path.insert(0, str(SCRIPT_DIR))
try:
    from palette import (
        DEEP_OCEAN, TURQUOISE, COASTAL_BLUE, OCEAN_BLUE,
        FOREST_GREEN, DESERT_TAN, CLAY,
    )
except ImportError:
    DEEP_OCEAN = (0.031, 0.188, 0.420)
    TURQUOISE = (0.369, 0.678, 0.651)
    COASTAL_BLUE = (0.255, 0.573, 0.612)
    OCEAN_BLUE = (0.129, 0.400, 0.545)
    FOREST_GREEN = (0.180, 0.420, 0.239)
    DESERT_TAN = (0.824, 0.659, 0.447)
    CLAY = (0.678, 0.486, 0.306)

# Styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
})

# Colors for bars
PFAM_COLOR = DESERT_TAN
NOVEL_COLORS = {
    "Novel_1e-05": DEEP_OCEAN,
    "Novel_1e-07": COASTAL_BLUE,
    "Novel_1e-09": TURQUOISE,
}
ALL_COLORS = [PFAM_COLOR, DEEP_OCEAN, COASTAL_BLUE, TURQUOISE]
ALL_LABELS = ["Pfam\n(E < 10$^{-9}$)", "Novel\n(E < 10$^{-5}$)",
              "Novel\n(E < 10$^{-7}$)", "Novel\n(E < 10$^{-9}$)"]

NOVEL_ONLY_COLORS = [DEEP_OCEAN, COASTAL_BLUE, TURQUOISE]
NOVEL_ONLY_LABELS = ["E < 10$^{-5}$", "E < 10$^{-7}$", "E < 10$^{-9}$"]

def load_summary():
    """Load the sensitivity comparison TSV."""
    path = SOURCE_DATA / "evalue_sensitivity_comparison.tsv"
    if not path.exists():
        print(f"ERROR: {path} not found")
        print("Run 09_evalue_sensitivity.py first and copy results to source_data/")
        sys.exit(1)
    df = pd.read_csv(path, sep="\t")
    return df

def load_correlation_details():
    """Load per-threshold correlation detail files for violin plots."""
    detail_data = {}

    # Try project-local paths first, then HPC-synced paths
    for label, fname in [
        ("Pfam", "correlations_pfam_baseline.tsv"),
        ("Novel_1e-05", "correlations_1e-05.tsv"),
        ("Novel_1e-07", "correlations_1e-07.tsv"),
        ("Novel_1e-09", "correlations_1e-09.tsv"),
    ]:
        path = DETAIL_DIR / fname
        if path.exists():
            df = pd.read_csv(path, sep="\t")
            detail_data[label] = df["rho"].abs().values
        else:
            print(f"  Warning: {path} not found, skipping violin for {label}")

    return detail_data

def panel_a(ax, summary):
    """Panel A: Domains retained at each threshold."""
    novel = summary[summary["threshold"].str.startswith("Novel")]
    x = np.arange(len(novel))
    width = 0.35

    bars_total = ax.bar(
        x - width / 2, novel["n_domains_total"].values, width,
        color=[c + (0.5,) if len(c) == 3 else c for c in NOVEL_ONLY_COLORS],
        edgecolor=[c for c in NOVEL_ONLY_COLORS], linewidth=0.5,
        label="Total with hits",
    )
    bars_prev = ax.bar(
        x + width / 2, novel["n_domains_prevalent"].values, width,
        color=NOVEL_ONLY_COLORS, edgecolor="black", linewidth=0.5,
        label=f"Prevalent (≥{10} samples)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(NOVEL_ONLY_LABELS)
    ax.set_ylabel("Number of domains")
    ax.set_title("Domains retained by threshold")
    ax.legend(fontsize=5, frameon=False)

    # Add count labels on bars
    for bar in bars_total:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{int(h):,}",
                ha="center", va="bottom", fontsize=5)
    for bar in bars_prev:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{int(h):,}",
                ha="center", va="bottom", fontsize=5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def panel_b(ax, summary):
    """Panel B: Significance rates across thresholds."""
    x = np.arange(len(summary))
    bars = ax.bar(x, summary["pct_significant"].values,
                  color=ALL_COLORS, edgecolor="black", linewidth=0.5)

    # Pfam baseline dashed line
    pfam_pct = summary.loc[summary["threshold"] == "Pfam_1e-9", "pct_significant"].values
    if len(pfam_pct) > 0:
        ax.axhline(pfam_pct[0], color=PFAM_COLOR, linestyle="--", linewidth=0.8,
                   alpha=0.7, zorder=0)
        ax.text(len(summary) - 0.5, pfam_pct[0] + 1, f"Pfam: {pfam_pct[0]:.1f}%",
                ha="right", va="bottom", fontsize=5, color=CLAY)

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_LABELS)
    ax.set_ylabel("Significant correlations (%)")
    ax.set_title("Proportion significant (FDR < 0.05)")
    ax.set_ylim(0, 100)

    # Add percentage labels
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def panel_c(ax, summary, detail_data):
    """Panel C: Effect size distributions as box/violin plots."""
    if detail_data:
        # Violin + box plots from raw data
        labels_used = []
        data_used = []
        colors_used = []

        for label, color, display in [
            ("Pfam", PFAM_COLOR, "Pfam\n(E<10$^{-9}$)"),
            ("Novel_1e-05", DEEP_OCEAN, "Novel\n(E<10$^{-5}$)"),
            ("Novel_1e-07", COASTAL_BLUE, "Novel\n(E<10$^{-7}$)"),
            ("Novel_1e-09", TURQUOISE, "Novel\n(E<10$^{-9}$)"),
        ]:
            if label in detail_data and len(detail_data[label]) > 0:
                data_used.append(detail_data[label])
                labels_used.append(display)
                colors_used.append(color)

        if data_used:
            parts = ax.violinplot(data_used, positions=range(len(data_used)),
                                  showextrema=False, widths=0.7)
            for i, body in enumerate(parts["bodies"]):
                body.set_facecolor(colors_used[i])
                body.set_alpha(0.3)
                body.set_edgecolor(colors_used[i])

            bp = ax.boxplot(data_used, positions=range(len(data_used)),
                           widths=0.3, showfliers=False,
                           medianprops=dict(color="black", linewidth=1),
                           boxprops=dict(linewidth=0.5),
                           whiskerprops=dict(linewidth=0.5),
                           capprops=dict(linewidth=0.5))

            ax.set_xticks(range(len(labels_used)))
            ax.set_xticklabels(labels_used)

            # Add median annotations
            for i, d in enumerate(data_used):
                med = np.median(d)
                ax.text(i, med + 0.01, f"{med:.3f}", ha="center", va="bottom",
                        fontsize=5, fontweight="bold")
    else:
        # Fallback: bar chart from summary
        med_values = summary["median_abs_rho"].values
        x = np.arange(len(summary))
        ax.bar(x, med_values, color=ALL_COLORS, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_LABELS)
        for i, v in enumerate(med_values):
            if not np.isnan(v):
                ax.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=5)

    ax.set_ylabel("$|\\rho|$ (Spearman)")
    ax.set_title("Effect size distributions")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def panel_d(ax, summary):
    """Panel D: Fold-enrichment vs Pfam at each threshold."""
    novel = summary[summary["threshold"].str.startswith("Novel")].copy()

    x = np.arange(len(novel))
    width = 0.35

    bars_med = ax.bar(x - width / 2, novel["fold_vs_pfam_median"].values, width,
                      color=NOVEL_ONLY_COLORS, edgecolor="black", linewidth=0.5,
                      label="Median |ρ| fold")
    bars_sig = ax.bar(x + width / 2, novel["fold_vs_pfam_pct_sig"].values, width,
                      color=[c + (0.5,) if len(c) == 3 else c for c in NOVEL_ONLY_COLORS],
                      edgecolor=[c for c in NOVEL_ONLY_COLORS], linewidth=0.5,
                      label="% significant fold")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5, zorder=0)
    ax.text(-0.3, 1.05, "Pfam baseline", fontsize=5, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(NOVEL_ONLY_LABELS)
    ax.set_ylabel("Fold-enrichment vs Pfam")
    ax.set_title("Enrichment robustness across thresholds")
    ax.legend(fontsize=5, frameon=False)

    # Add value labels
    for bar in bars_med:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.1f}×", ha="center", va="bottom", fontsize=5)
    for bar in bars_sig:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.1f}×", ha="center", va="bottom", fontsize=5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def main():
    print("Creating Figure S10: E-value Sensitivity Analysis")
    print()

    summary = load_summary()
    print("Summary data:")
    print(summary.to_string(index=False))
    print()

    detail_data = load_correlation_details()

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    panel_a(axes[0, 0], summary)
    panel_b(axes[0, 1], summary)
    panel_c(axes[1, 0], summary, detail_data)
    panel_d(axes[1, 1], summary)

    # Panel labels
    for ax, label in zip(axes.flat, ["A", "B", "C", "D"]):
        ax.text(-0.12, 1.08, label, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")

    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SCRIPT_DIR / f"FigureS10_evalue_sensitivity_{timestamp}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print()

if __name__ == "__main__":
    main()
