#!/usr/bin/env python3
"""
Phase 6: Manifold Learning with Canonical Correlation Analysis.

This script uses CCA to find the shared manifold between environmental
variables and PFAM domain abundances - the joint latent space.

Provenance:
  Input: Processed data from Phase 1
  Date: 2026-01-13
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import json

# Data integrity enforcement (CLAUDE.md requirement)
from DataIntegrityGuard import enforce_data_integrity
enforce_data_integrity()

# Paths
BASE_DIR = Path("/media/drn/External1/TARA-Oceans/03_analyses/env_pfam_manifold")
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    pfam_cols_file = sorted(DATA_DIR.glob("pfam_columns_*.txt"))[-1]

    with open(env_cols_file) as f:
        env_cols = [line.strip() for line in f]

    with open(pfam_cols_file) as f:
        pfam_cols = [line.strip() for line in f]

    print(f"  Environmental matrix: {env_matrix.shape}")
    print(f"  PFAM matrix: {pfam_matrix.shape}")
    print(f"  Environmental variables: {len(env_cols)}")
    print(f"  PFAM domains: {len(pfam_cols)}")

    return env_matrix, pfam_matrix, coordinates, env_cols, pfam_cols

def reduce_pfam_dimensions(pfam_matrix, n_components=100):
    """Reduce PFAM dimensions for CCA (CCA works better with fewer dimensions)."""
    print("\n" + "=" * 70)
    print(f"REDUCING PFAM TO {n_components} PCs")
    print("=" * 70)

    pca = PCA(n_components=n_components, random_state=42)
    pfam_pca = pca.fit_transform(pfam_matrix)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  Variance explained: {explained_var*100:.2f}%")

    return pfam_pca, pca

def run_cca(env_matrix, pfam_pca, n_components=10):
    """Run Canonical Correlation Analysis."""
    print("\n" + "=" * 70)
    print(f"RUNNING CCA WITH {n_components} COMPONENTS")
    print("=" * 70)

    cca = CCA(n_components=n_components)
    env_cca, pfam_cca = cca.fit_transform(env_matrix, pfam_pca)

    # Calculate canonical correlations
    canonical_corrs = []
    for i in range(n_components):
        corr = np.corrcoef(env_cca[:, i], pfam_cca[:, i])[0, 1]
        canonical_corrs.append(corr)

    print(f"\n  Canonical Correlations:")
    for i, corr in enumerate(canonical_corrs):
        print(f"    CC{i+1}: {corr:.4f}")

    return cca, env_cca, pfam_cca, canonical_corrs

def analyze_cca_loadings(cca, env_cols, n_components=5):
    """Analyze which variables contribute to each canonical component."""
    print("\n" + "=" * 70)
    print("CCA LOADING ANALYSIS")
    print("=" * 70)

    # Environmental loadings (x_loadings_)
    env_loadings = cca.x_loadings_

    print("\n  Environmental Variable Loadings on Top Canonical Components:")
    loading_df = pd.DataFrame(
        env_loadings[:, :n_components],
        index=env_cols,
        columns=[f'CC{i+1}' for i in range(n_components)]
    )

    # For each CC, show top contributors
    for cc in range(min(3, n_components)):
        print(f"\n  CC{cc+1} - Top environmental contributors:")
        sorted_loadings = loading_df[f'CC{cc+1}'].abs().sort_values(ascending=False)
        for var in sorted_loadings.head(5).index:
            val = loading_df.loc[var, f'CC{cc+1}']
            direction = '+' if val > 0 else '-'
            print(f"    {direction} {var}: {abs(val):.4f}")

    return loading_df

def plot_cca_results(env_cca, pfam_cca, canonical_corrs, coordinates, env_cols, output_dir):
    """Create comprehensive CCA visualizations."""
    print("\n" + "=" * 70)
    print("CREATING CCA VISUALIZATIONS")
    print("=" * 70)

    # 1. Canonical correlation bar plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.bar(range(1, len(canonical_corrs)+1), canonical_corrs, color='steelblue')
    ax.set_xlabel('Canonical Component')
    ax.set_ylabel('Canonical Correlation')
    ax.set_title('Canonical Correlations between Environment and PFAM')
    ax.set_xticks(range(1, len(canonical_corrs)+1))

    # 2. Joint CCA embedding (CC1 vs CC2)
    ax = axes[0, 1]
    # Average of env and pfam projections
    joint_embedding = (env_cca[:, :2] + pfam_cca[:, :2]) / 2
    scatter = ax.scatter(joint_embedding[:, 0], joint_embedding[:, 1],
                        c=coordinates[:, 0], cmap='coolwarm', s=10, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label='Latitude')
    ax.set_xlabel('CC1 (Joint)')
    ax.set_ylabel('CC2 (Joint)')
    ax.set_title('Joint CCA Embedding Colored by Latitude')

    # 3. Env vs PFAM projections on CC1
    ax = axes[1, 0]
    ax.scatter(env_cca[:, 0], pfam_cca[:, 0], alpha=0.5, s=10)
    ax.plot([env_cca[:, 0].min(), env_cca[:, 0].max()],
            [env_cca[:, 0].min(), env_cca[:, 0].max()], 'r--', alpha=0.5)
    ax.set_xlabel('Environment CC1')
    ax.set_ylabel('PFAM CC1')
    ax.set_title(f'CC1: r={canonical_corrs[0]:.3f}')

    # 4. Env vs PFAM projections on CC2
    ax = axes[1, 1]
    ax.scatter(env_cca[:, 1], pfam_cca[:, 1], alpha=0.5, s=10)
    ax.plot([env_cca[:, 1].min(), env_cca[:, 1].max()],
            [env_cca[:, 1].min(), env_cca[:, 1].max()], 'r--', alpha=0.5)
    ax.set_xlabel('Environment CC2')
    ax.set_ylabel('PFAM CC2')
    ax.set_title(f'CC2: r={canonical_corrs[1]:.3f}')

    plt.tight_layout()
    plt.savefig(output_dir / f"cca_results_{TIMESTAMP}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cca_results_{TIMESTAMP}.png")

    # 5. Joint manifold with multiple environmental colorings
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    key_env_vars = ['sst_mean_c', 'chl_mean_mg_m3', 'bathymetry_m',
                    'solar_rad_mj_m2', 'distance_to_coast_km', 'modis_sst_mean_c']

    # Load env matrix for coloring
    env_files = sorted(DATA_DIR.glob("env_matrix_*.npy"))
    env_matrix = np.load(env_files[-1])

    for i, var in enumerate(key_env_vars):
        ax = axes[i]
        if var in env_cols:
            idx = env_cols.index(var)
            values = env_matrix[:, idx]
            scatter = ax.scatter(joint_embedding[:, 0], joint_embedding[:, 1],
                               c=values, cmap='viridis', s=10, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label=var)
            ax.set_xlabel('CC1')
            ax.set_ylabel('CC2')
            ax.set_title(f'Joint Manifold: {var}')

    plt.tight_layout()
    plt.savefig(output_dir / f"cca_manifold_env_{TIMESTAMP}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cca_manifold_env_{TIMESTAMP}.png")

    return joint_embedding

def save_cca_results(cca, env_cca, pfam_cca, canonical_corrs, loading_df, joint_embedding):
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
        'cc2': float(canonical_corrs[1]),
        'cc3': float(canonical_corrs[2]),
        'mean_canonical_corr': float(np.mean(canonical_corrs)),
        'total_shared_variance_top3': float(sum([c**2 for c in canonical_corrs[:3]]))
    }

    with open(REPORTS_DIR / f"cca_summary_{TIMESTAMP}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved summary")

def main():
    print("=" * 70)
    print("PHASE 6: MANIFOLD LEARNING WITH CCA")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load data
    env_matrix, pfam_matrix, coordinates, env_cols, pfam_cols = load_processed_data()

    # Reduce PFAM dimensions
    pfam_pca, pca_model = reduce_pfam_dimensions(pfam_matrix, n_components=100)

    # Run CCA
    cca, env_cca, pfam_cca, canonical_corrs = run_cca(env_matrix, pfam_pca, n_components=10)

    # Analyze loadings
    loading_df = analyze_cca_loadings(cca, env_cols)

    # Plot results
    joint_embedding = plot_cca_results(
        env_cca, pfam_cca, canonical_corrs, coordinates, env_cols, FIGURES_DIR
    )

    # Save results
    save_cca_results(cca, env_cca, pfam_cca, canonical_corrs, loading_df, joint_embedding)

    print("\n" + "=" * 70)
    print("PHASE 6 COMPLETE")
    print("=" * 70)

    return cca, env_cca, pfam_cca, canonical_corrs

if __name__ == "__main__":
    cca, env_cca, pfam_cca, canonical_corrs = main()
