#!/usr/bin/env python3
"""
Phase 6: Canonical Correlation Analysis (CCA) Manifold Learning for algaGPT.

This script uses CCA to find the shared manifold between environmental
variables and PFAM domain abundances - discovering the joint latent space
that captures the environment-genome relationship.

Key insights from CCA:
- Canonical correlations measure env-PFAM coupling strength
- CCA loadings reveal which variables drive the shared structure
- Joint embedding shows samples in the coupled env-genome space

Improvements:
- Permutation testing for statistical significance
- Better interpretation of loadings
- Multiple visualization approaches

Provenance:
  Input: data/*.npy from Phase 1
  Output: data/cca_*.npy, reports/cca_*.json, figures/cca_*.pdf
  Date: 2026-01-22
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import json
import sys

# Data integrity enforcement
sys.path.insert(0, str(Path(__file__).parent))
from DataIntegrityGuard import enforce_data_integrity
enforce_data_integrity()

# ============================================================================
# FIGURE PROTOCOL
# ============================================================================

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 5
mpl.rcParams['axes.linewidth'] = 0.25
mpl.rcParams['xtick.major.width'] = 0.25
mpl.rcParams['ytick.major.width'] = 0.25

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# CCA parameters
N_CCA_COMPONENTS = 10
N_PFAM_PCS = 100  # Reduce PFAM to this many PCs before CCA
N_PERMUTATIONS = 100  # For significance testing

def load_processed_data():
    """Load processed data from Phase 1."""
    print("=" * 70)
    print("LOADING PROCESSED DATA")
    print("=" * 70)

    env_files = sorted(DATA_DIR.glob("env_matrix_*.npy"))
    pfam_files = sorted(DATA_DIR.glob("pfam_matrix_*.npy"))
    coord_files = sorted(DATA_DIR.glob("coordinates_*.npy"))

    if not env_files or not pfam_files:
        raise FileNotFoundError("Processed data not found. Run Phase 1 first.")

    env_matrix = np.load(env_files[-1])
    pfam_matrix = np.load(pfam_files[-1])
    coordinates = np.load(coord_files[-1])

    env_cols_file = sorted(DATA_DIR.glob("env_columns_*.txt"))[-1]
    with open(env_cols_file) as f:
        env_cols = [line.strip() for line in f]

    pfam_cols_file = sorted(DATA_DIR.glob("pfam_columns_*.txt"))[-1]
    with open(pfam_cols_file) as f:
        pfam_cols = [line.strip() for line in f]

    print(f"  Environmental: {env_matrix.shape}")
    print(f"  PFAM: {pfam_matrix.shape}")
    print(f"  Coordinates: {coordinates.shape}")

    return env_matrix, pfam_matrix, coordinates, env_cols, pfam_cols

def reduce_pfam_dimensions(pfam_matrix, n_components=N_PFAM_PCS):
    """Reduce PFAM dimensions for CCA."""
    print("\n" + "=" * 70)
    print(f"REDUCING PFAM TO {n_components} PCs")
    print("=" * 70)

    n_components = min(n_components, min(pfam_matrix.shape) - 1)
    pca = PCA(n_components=n_components, random_state=42)
    pfam_pca = pca.fit_transform(pfam_matrix)

    print(f"  Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    return pfam_pca, pca

def run_cca(env_matrix, pfam_pca, n_components=N_CCA_COMPONENTS):
    """Run Canonical Correlation Analysis."""
    print("\n" + "=" * 70)
    print(f"RUNNING CCA ({n_components} components)")
    print("=" * 70)

    # Handle NaN: drop columns that are all NaN or have any NaN
    valid_cols = ~np.any(np.isnan(env_matrix), axis=0)
    env_clean = env_matrix[:, valid_cols]
    print(f"  Dropped {(~valid_cols).sum()} env columns with NaN")
    print(f"  Using {env_clean.shape[1]} env variables")

    # CCA requires n_components <= min features
    n_components = min(n_components, env_clean.shape[1], pfam_pca.shape[1])

    cca = CCA(n_components=n_components)
    env_cca, pfam_cca = cca.fit_transform(env_clean, pfam_pca)

    # Calculate canonical correlations
    canonical_corrs = []
    for i in range(n_components):
        corr = np.corrcoef(env_cca[:, i], pfam_cca[:, i])[0, 1]
        canonical_corrs.append(corr)

    print(f"\n  Canonical Correlations:")
    for i, corr in enumerate(canonical_corrs):
        print(f"    CC{i+1}: {corr:.4f}")

    return cca, env_cca, pfam_cca, canonical_corrs, valid_cols

def permutation_test(env_matrix, pfam_pca, n_permutations=N_PERMUTATIONS):
    """Test significance of canonical correlations via permutation."""
    print("\n" + "=" * 70)
    print(f"PERMUTATION TESTING ({n_permutations} permutations)")
    print("=" * 70)

    # Handle NaN
    valid_cols = ~np.any(np.isnan(env_matrix), axis=0)
    env_clean = env_matrix[:, valid_cols]

    # Get observed correlations
    n_comp = min(5, env_clean.shape[1], pfam_pca.shape[1])
    cca = CCA(n_components=n_comp)
    env_cca, pfam_cca = cca.fit_transform(env_clean, pfam_pca)

    observed_corrs = []
    for i in range(n_comp):
        corr = np.corrcoef(env_cca[:, i], pfam_cca[:, i])[0, 1]
        observed_corrs.append(corr)

    # Permutation null distribution
    null_corrs = np.zeros((n_permutations, n_comp))
    print(f"  Running permutations...", end=" ", flush=True)

    for perm in range(n_permutations):
        # Shuffle samples in PFAM matrix
        perm_idx = np.random.permutation(len(pfam_pca))
        pfam_perm = pfam_pca[perm_idx]

        cca_perm = CCA(n_components=n_comp)
        env_perm, pfam_perm_cca = cca_perm.fit_transform(env_clean, pfam_perm)

        for i in range(n_comp):
            null_corrs[perm, i] = np.corrcoef(env_perm[:, i], pfam_perm_cca[:, i])[0, 1]

    print("done")

    # Calculate p-values
    p_values = []
    for i in range(n_comp):
        p_val = (null_corrs[:, i] >= observed_corrs[i]).mean()
        p_values.append(p_val)
        print(f"    CC{i+1}: r={observed_corrs[i]:.4f}, p={p_val:.4f}")

    return observed_corrs, p_values, null_corrs

def analyze_cca_loadings(cca, env_cols, valid_cols=None, n_show=5):
    """Analyze which variables contribute to each canonical component."""
    print("\n" + "=" * 70)
    print("CCA LOADING ANALYSIS")
    print("=" * 70)

    env_loadings = cca.x_loadings_

    # Filter env_cols to match valid columns
    if valid_cols is not None:
        filtered_env_cols = [env_cols[i] for i in range(len(env_cols)) if valid_cols[i]]
    else:
        filtered_env_cols = env_cols

    n_show = min(n_show, env_loadings.shape[1])
    loading_df = pd.DataFrame(
        env_loadings[:, :n_show],
        index=filtered_env_cols,
        columns=[f'CC{i+1}' for i in range(n_show)]
    )

    # Top contributors for each CC
    for cc in range(min(3, env_loadings.shape[1])):
        print(f"\n  CC{cc+1} - Top environmental contributors:")
        sorted_loadings = loading_df[f'CC{cc+1}'].abs().sort_values(ascending=False)
        for var in sorted_loadings.head(5).index:
            val = loading_df.loc[var, f'CC{cc+1}']
            direction = '+' if val > 0 else '-'
            print(f"    {direction} {var}: {abs(val):.4f}")

    return loading_df

def save_figure(fig, name):
    """Save as PDF and SVG with transparent background."""
    for fmt in ['pdf', 'svg']:
        fig.savefig(FIGURES_DIR / f"{name}_{TIMESTAMP}.{fmt}",
                   format=fmt, bbox_inches='tight', transparent=True, edgecolor='none')
    print(f"  Saved: {name}_{TIMESTAMP}.pdf/.svg")

def plot_cca_results(env_cca, pfam_cca, canonical_corrs, coordinates, env_matrix, env_cols):
    """Create comprehensive CCA visualizations."""
    print("\n" + "=" * 70)
    print("CREATING CCA VISUALIZATIONS")
    print("=" * 70)

    # Joint embedding (average of env and pfam projections)
    joint_embedding = (env_cca[:, :2] + pfam_cca[:, :2]) / 2

    fig = plt.figure(figsize=(7, 6))

    # Panel A: Canonical correlations bar chart
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.bar(range(1, len(canonical_corrs) + 1), canonical_corrs,
           color='steelblue', width=0.7, edgecolor='none')
    ax1.set_xlabel('Canonical Component')
    ax1.set_ylabel('Canonical Correlation')
    ax1.set_xticks(range(1, len(canonical_corrs) + 1))
    ax1.set_ylim(0, 1)
    ax1.text(-0.15, 1.05, 'A', transform=ax1.transAxes, fontweight='bold', fontsize=7)

    # Panel B: Joint embedding by latitude
    ax2 = fig.add_subplot(2, 2, 2)
    sc = ax2.scatter(joint_embedding[:, 0], joint_embedding[:, 1],
                    c=coordinates[:, 0], cmap='coolwarm', s=2, alpha=0.7)
    cb = plt.colorbar(sc, ax=ax2, shrink=0.8, aspect=15, pad=0.02)
    cb.set_label('Latitude', fontsize=5)
    cb.ax.tick_params(labelsize=5, width=0.25, length=1)
    ax2.set_xlabel('CC1 (Joint)')
    ax2.set_ylabel('CC2 (Joint)')
    ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes, fontweight='bold', fontsize=7)

    # Panel C: Env vs PFAM on CC1
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(env_cca[:, 0], pfam_cca[:, 0], s=2, alpha=0.5, color='steelblue')
    lims = [min(env_cca[:, 0].min(), pfam_cca[:, 0].min()),
            max(env_cca[:, 0].max(), pfam_cca[:, 0].max())]
    ax3.plot(lims, lims, 'r--', lw=0.5, alpha=0.7)
    ax3.set_xlabel('Environment CC1')
    ax3.set_ylabel('PFAM CC1')
    ax3.set_title(f'CC1: r={canonical_corrs[0]:.3f}', fontsize=6)
    ax3.text(-0.15, 1.05, 'C', transform=ax3.transAxes, fontweight='bold', fontsize=7)

    # Panel D: Joint embedding by SST (if available)
    ax4 = fig.add_subplot(2, 2, 4)
    if 'sst_mean_c' in env_cols:
        idx = env_cols.index('sst_mean_c')
        values = env_matrix[:, idx]
        mask = np.isfinite(values)
        sc = ax4.scatter(joint_embedding[mask, 0], joint_embedding[mask, 1],
                        c=values[mask], cmap='viridis', s=2, alpha=0.7)
        cb = plt.colorbar(sc, ax=ax4, shrink=0.8, aspect=15, pad=0.02)
        cb.set_label('SST (C)', fontsize=5)
        cb.ax.tick_params(labelsize=5, width=0.25, length=1)
        ax4.set_title('Joint Manifold by SST', fontsize=6)
    else:
        ax4.text(0.5, 0.5, 'SST not available', ha='center', va='center',
                transform=ax4.transAxes, fontsize=6)
    ax4.set_xlabel('CC1')
    ax4.set_ylabel('CC2')
    ax4.text(-0.15, 1.05, 'D', transform=ax4.transAxes, fontweight='bold', fontsize=7)

    plt.tight_layout()
    save_figure(fig, 'cca_results')
    plt.close()

    return joint_embedding

def save_cca_results(cca, env_cca, pfam_cca, canonical_corrs, loading_df, joint_embedding, p_values=None):
    """Save all CCA results."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save embeddings
    np.save(DATA_DIR / f"cca_env_embedding_{TIMESTAMP}.npy", env_cca)
    np.save(DATA_DIR / f"cca_pfam_embedding_{TIMESTAMP}.npy", pfam_cca)
    np.save(DATA_DIR / f"cca_joint_embedding_{TIMESTAMP}.npy", joint_embedding)
    print(f"  Saved CCA embeddings")

    # Save loadings
    loading_df.to_csv(REPORTS_DIR / f"cca_env_loadings_{TIMESTAMP}.csv")
    print(f"  Saved CCA loadings")

    # Summary
    summary = {
        'timestamp': TIMESTAMP,
        'n_components': len(canonical_corrs),
        'canonical_correlations': [float(c) for c in canonical_corrs],
        'cc1': float(canonical_corrs[0]),
        'cc2': float(canonical_corrs[1]) if len(canonical_corrs) > 1 else None,
        'cc3': float(canonical_corrs[2]) if len(canonical_corrs) > 2 else None,
        'mean_canonical_corr': float(np.mean(canonical_corrs)),
        'total_shared_variance_top3': float(sum([c**2 for c in canonical_corrs[:3]])),
    }

    if p_values is not None:
        summary['p_values'] = [float(p) for p in p_values]
        summary['significant_ccs'] = int(sum([p < 0.05 for p in p_values]))

    with open(REPORTS_DIR / f"cca_summary_{TIMESTAMP}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved summary")

    # Print key results
    print(f"\n  Key Results:")
    print(f"    CC1: {canonical_corrs[0]:.4f}")
    print(f"    CC2: {canonical_corrs[1]:.4f}" if len(canonical_corrs) > 1 else "")
    print(f"    Mean CC: {np.mean(canonical_corrs):.4f}")
    if p_values is not None:
        print(f"    Significant CCs (p<0.05): {summary['significant_ccs']}/{len(p_values)}")

def main():
    print("=" * 70)
    print("PHASE 6: CCA MANIFOLD LEARNING (algaGPT)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load data
    env_matrix, pfam_matrix, coordinates, env_cols, pfam_cols = load_processed_data()

    # Reduce PFAM dimensions
    pfam_pca, pca_model = reduce_pfam_dimensions(pfam_matrix)

    # Run CCA
    cca, env_cca, pfam_cca, canonical_corrs, valid_cols = run_cca(env_matrix, pfam_pca)

    # Permutation test
    observed_corrs, p_values, null_corrs = permutation_test(env_matrix, pfam_pca)

    # Analyze loadings
    loading_df = analyze_cca_loadings(cca, env_cols, valid_cols)

    # Create visualizations
    joint_embedding = plot_cca_results(env_cca, pfam_cca, canonical_corrs,
                                       coordinates, env_matrix, env_cols)

    # Save results
    save_cca_results(cca, env_cca, pfam_cca, canonical_corrs, loading_df, joint_embedding, p_values)

    print("\n" + "=" * 70)
    print("PHASE 6 COMPLETE")
    print("=" * 70)

    return cca, env_cca, pfam_cca, canonical_corrs

if __name__ == "__main__":
    cca, env_cca, pfam_cca, canonical_corrs = main()
