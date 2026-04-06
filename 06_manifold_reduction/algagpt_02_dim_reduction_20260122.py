#!/usr/bin/env python3
"""
Phase 2: Dimensionality Reduction and Visualization (algaGPT).

This script performs PCA, UMAP, and t-SNE on the PFAM data to visualize
how samples cluster in the reduced-dimension space, colored by environmental
variables.

Improvements over alkhidr version:
- Publication-quality figures following FIGURE_PROTOCOL.md
- Multiple embedding methods for robustness
- Systematic exploration of hyperparameters
- Correlation of embedding dimensions with environment

Provenance:
  Input: data/*.npy from Phase 1
  Output: data/*_embedding_*.npy, figures/*.pdf
  Date: 2026-01-22
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("WARNING: umap-learn not installed. UMAP will be skipped.")

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import sys

# Data integrity enforcement
sys.path.insert(0, str(Path(__file__).parent))
from DataIntegrityGuard import enforce_data_integrity
enforce_data_integrity()

# ============================================================================
# FIGURE PROTOCOL - STRICT 6pt ARIAL, 0.25pt LINES
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
mpl.rcParams['xtick.major.size'] = 1.5
mpl.rcParams['ytick.major.size'] = 1.5
mpl.rcParams['axes.labelpad'] = 1
mpl.rcParams['xtick.major.pad'] = 1
mpl.rcParams['ytick.major.pad'] = 1

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# PCA parameters
N_PCA_COMPONENTS = 50

# UMAP parameters to explore
UMAP_N_NEIGHBORS = [15, 30, 50]
UMAP_MIN_DIST = 0.1

# t-SNE parameters
TSNE_PERPLEXITIES = [5, 30, 50]

# Key environmental variables for coloring
KEY_ENV_VARS = [
    'sst_mean_c', 'chl_mean_mg_m3', 'bathymetry_m',
    'solar_rad_mj_m2', 'distance_to_coast_km', 'modis_sst_mean_c'
]

def load_processed_data():
    """Load processed data from Phase 1."""
    print("=" * 70)
    print("LOADING PROCESSED DATA")
    print("=" * 70)

    # Find most recent files
    env_files = sorted(DATA_DIR.glob("env_matrix_*.npy"))
    pfam_files = sorted(DATA_DIR.glob("pfam_matrix_*.npy"))
    coord_files = sorted(DATA_DIR.glob("coordinates_*.npy"))
    id_files = sorted(DATA_DIR.glob("sample_ids_*.npy"))

    if not env_files or not pfam_files:
        raise FileNotFoundError("Processed data not found. Run Phase 1 first.")

    env_matrix = np.load(env_files[-1])
    pfam_matrix = np.load(pfam_files[-1])
    coordinates = np.load(coord_files[-1])
    sample_ids = np.load(id_files[-1], allow_pickle=True)

    # Load column names
    env_cols_file = sorted(DATA_DIR.glob("env_columns_*.txt"))[-1]
    with open(env_cols_file) as f:
        env_cols = [line.strip() for line in f]

    print(f"  Environmental matrix: {env_matrix.shape}")
    print(f"  PFAM matrix: {pfam_matrix.shape}")
    print(f"  Coordinates: {coordinates.shape}")
    print(f"  Sample IDs: {len(sample_ids)}")
    print(f"  Environmental variables: {len(env_cols)}")

    return env_matrix, pfam_matrix, coordinates, sample_ids, env_cols

def run_pca(pfam_matrix, n_components=N_PCA_COMPONENTS):
    """Run PCA on PFAM data."""
    print("\n" + "=" * 70)
    print("RUNNING PCA")
    print("=" * 70)

    n_components = min(n_components, min(pfam_matrix.shape) - 1)
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(pfam_matrix)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print(f"  Components: {pca_result.shape[1]}")
    print(f"  PC1 variance: {explained_var[0]*100:.2f}%")
    print(f"  PC1-5 variance: {cumulative_var[4]*100:.2f}%")
    print(f"  PC1-10 variance: {cumulative_var[9]*100:.2f}%")
    print(f"  PCs for 50% variance: {np.argmax(cumulative_var >= 0.50) + 1}")
    print(f"  PCs for 80% variance: {np.argmax(cumulative_var >= 0.80) + 1}")

    return pca_result, pca, explained_var

def run_umap(pfam_matrix, n_neighbors_list=UMAP_N_NEIGHBORS):
    """Run UMAP with multiple n_neighbors settings."""
    if not UMAP_AVAILABLE:
        print("\n  UMAP skipped (not installed)")
        return {}

    print("\n" + "=" * 70)
    print("RUNNING UMAP")
    print("=" * 70)

    umap_results = {}

    for n_neighbors in n_neighbors_list:
        print(f"  n_neighbors={n_neighbors}...", end=" ", flush=True)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=UMAP_MIN_DIST,
            n_components=2,
            metric='euclidean',
            random_state=42,
            verbose=False
        )
        umap_result = reducer.fit_transform(pfam_matrix)
        umap_results[n_neighbors] = umap_result
        print(f"done. Shape: {umap_result.shape}")

    return umap_results

def run_tsne(pfam_matrix, perplexity_list=TSNE_PERPLEXITIES, pca_result=None):
    """Run t-SNE with multiple perplexity settings."""
    print("\n" + "=" * 70)
    print("RUNNING t-SNE")
    print("=" * 70)

    # Use PCA-reduced data for faster t-SNE
    if pca_result is not None and pca_result.shape[1] >= 50:
        input_data = pca_result[:, :50]
        print(f"  Using first 50 PCA components")
    else:
        input_data = pfam_matrix

    tsne_results = {}

    for perplexity in perplexity_list:
        # Adjust perplexity if too large
        max_perplexity = (input_data.shape[0] - 1) // 3
        actual_perplexity = min(perplexity, max_perplexity)

        print(f"  perplexity={actual_perplexity}...", end=" ", flush=True)
        tsne = TSNE(
            n_components=2,
            perplexity=actual_perplexity,
            random_state=42,
            max_iter=1000,
            init='pca'
        )
        tsne_result = tsne.fit_transform(input_data)
        tsne_results[perplexity] = tsne_result
        print(f"done. Shape: {tsne_result.shape}")

    return tsne_results

def compute_embedding_correlations(embedding, env_matrix, env_cols):
    """Compute correlations between embedding dimensions and environment."""
    from scipy.stats import spearmanr

    correlations = {}

    for dim in range(embedding.shape[1]):
        dim_corrs = {}
        for i, env_var in enumerate(env_cols):
            mask = np.isfinite(env_matrix[:, i])
            if mask.sum() > 10:
                rho, _ = spearmanr(embedding[mask, dim], env_matrix[mask, i])
                dim_corrs[env_var] = rho
        correlations[f'Dim{dim+1}'] = dim_corrs

    return correlations

def save_figure(fig, name):
    """Save as PDF and SVG with transparent background."""
    for fmt in ['pdf', 'svg']:
        fig.savefig(FIGURES_DIR / f"{name}_{TIMESTAMP}.{fmt}",
                   format=fmt, bbox_inches='tight', transparent=True, edgecolor='none')
    print(f"  Saved: {name}_{TIMESTAMP}.pdf/.svg")

def plot_pca_variance(explained_var):
    """Plot PCA variance explained - publication quality."""
    print("\n  Creating PCA variance figure...")

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))

    # Scree plot
    ax = axes[0]
    n_show = min(30, len(explained_var))
    ax.bar(range(1, n_show + 1), explained_var[:n_show] * 100,
           color='steelblue', width=0.8, edgecolor='none')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_xlim(0.5, n_show + 0.5)
    ax.set_xticks([1, 10, 20, 30])
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontweight='bold', fontsize=7)

    # Cumulative variance
    ax = axes[1]
    cumvar = np.cumsum(explained_var) * 100
    ax.plot(range(1, len(cumvar) + 1), cumvar, 'b-', lw=0.75)
    ax.axhline(50, color='#CC0000', ls='--', lw=0.5, alpha=0.7)
    ax.axhline(80, color='#00CC00', ls='--', lw=0.5, alpha=0.7)
    ax.text(len(cumvar) - 5, 52, '50%', fontsize=5, color='#CC0000')
    ax.text(len(cumvar) - 5, 82, '80%', fontsize=5, color='#00CC00')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance (%)')
    ax.set_xlim(1, len(cumvar))
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontweight='bold', fontsize=7)

    plt.tight_layout()
    save_figure(fig, 'pca_variance')
    plt.close()

def plot_embedding_grid(embeddings_dict, env_matrix, env_cols, coordinates, title_prefix):
    """
    Create a grid of embeddings colored by key environmental variables.
    Publication-quality multi-panel figure.
    """
    print(f"\n  Creating {title_prefix} embedding grid...")

    # Filter to key env vars that exist
    available_vars = [v for v in KEY_ENV_VARS if v in env_cols][:6]

    if not available_vars:
        print(f"    No key environmental variables found")
        return

    # Get one embedding (use n=30 for UMAP or p=30 for t-SNE as default)
    if 30 in embeddings_dict:
        embedding = embeddings_dict[30]
    else:
        embedding = list(embeddings_dict.values())[0]

    n_vars = len(available_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 2.5 * n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    for i, var in enumerate(available_vars):
        ax = axes[i]
        idx = env_cols.index(var)
        values = env_matrix[:, idx]
        mask = np.isfinite(values)

        if mask.sum() < 10:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=6)
            ax.set_title(var)
            continue

        sc = ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=values[mask], cmap='viridis', s=1, alpha=0.7)
        cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=15, pad=0.02)
        cb.ax.tick_params(labelsize=5, width=0.25, length=1)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(var.replace('_', ' ').replace(' c', ' (C)').replace(' m3', '/m3')
                    .replace('mg ', 'mg/'), fontsize=6)

    # Hide unused axes
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    save_figure(fig, f'{title_prefix.lower()}_env_grid')
    plt.close()

def plot_geographic_embedding(embedding, coordinates, method_name):
    """Plot embedding colored by geographic coordinates."""
    print(f"\n  Creating {method_name} geographic figure...")

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))

    # By latitude
    ax = axes[0]
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                   c=coordinates[:, 0], cmap='coolwarm', s=1, alpha=0.7)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=15, pad=0.02)
    cb.set_label('Latitude', fontsize=5)
    cb.ax.tick_params(labelsize=5, width=0.25, length=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{method_name} by Latitude', fontsize=6)
    ax.text(-0.08, 1.05, 'A', transform=ax.transAxes, fontweight='bold', fontsize=7)

    # By longitude
    ax = axes[1]
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                   c=coordinates[:, 1], cmap='twilight', s=1, alpha=0.7)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=15, pad=0.02)
    cb.set_label('Longitude', fontsize=5)
    cb.ax.tick_params(labelsize=5, width=0.25, length=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{method_name} by Longitude', fontsize=6)
    ax.text(-0.08, 1.05, 'B', transform=ax.transAxes, fontweight='bold', fontsize=7)

    plt.tight_layout()
    save_figure(fig, f'{method_name.lower()}_geographic')
    plt.close()

def plot_method_comparison(pca_result, umap_results, tsne_results, env_matrix, env_cols):
    """Create a comparison figure showing all three methods side by side."""
    print("\n  Creating method comparison figure...")

    # Use SST for coloring
    if 'sst_mean_c' in env_cols:
        color_var = 'sst_mean_c'
        idx = env_cols.index(color_var)
        values = env_matrix[:, idx]
    else:
        values = np.zeros(len(pca_result))
        color_var = 'none'

    mask = np.isfinite(values)

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    # PCA
    ax = axes[0]
    sc = ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                   c=values[mask], cmap='viridis', s=1, alpha=0.7)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA', fontsize=6)
    ax.text(-0.12, 1.05, 'A', transform=ax.transAxes, fontweight='bold', fontsize=7)

    # UMAP
    ax = axes[1]
    if umap_results and 30 in umap_results:
        umap_emb = umap_results[30]
        sc = ax.scatter(umap_emb[mask, 0], umap_emb[mask, 1],
                       c=values[mask], cmap='viridis', s=1, alpha=0.7)
        ax.set_title('UMAP', fontsize=6)
    else:
        ax.text(0.5, 0.5, 'UMAP not available', ha='center', va='center',
               transform=ax.transAxes, fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(-0.08, 1.05, 'B', transform=ax.transAxes, fontweight='bold', fontsize=7)

    # t-SNE
    ax = axes[2]
    if tsne_results and 30 in tsne_results:
        tsne_emb = tsne_results[30]
        sc = ax.scatter(tsne_emb[mask, 0], tsne_emb[mask, 1],
                       c=values[mask], cmap='viridis', s=1, alpha=0.7)
        cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=15, pad=0.02)
        cb.set_label(color_var.replace('_', ' '), fontsize=5)
        cb.ax.tick_params(labelsize=5, width=0.25, length=1)
        ax.set_title('t-SNE', fontsize=6)
    else:
        ax.text(0.5, 0.5, 't-SNE not available', ha='center', va='center',
               transform=ax.transAxes, fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(-0.08, 1.05, 'C', transform=ax.transAxes, fontweight='bold', fontsize=7)

    plt.tight_layout()
    save_figure(fig, 'method_comparison')
    plt.close()

def save_embeddings(pca_result, pca_model, umap_results, tsne_results, env_matrix, env_cols):
    """Save embeddings and metadata."""
    print("\n" + "=" * 70)
    print("SAVING EMBEDDINGS")
    print("=" * 70)

    # PCA
    np.save(DATA_DIR / f"pca_embedding_{TIMESTAMP}.npy", pca_result)
    np.save(DATA_DIR / f"pca_components_{TIMESTAMP}.npy", pca_model.components_)
    np.save(DATA_DIR / f"pca_explained_variance_{TIMESTAMP}.npy", pca_model.explained_variance_ratio_)
    print(f"  Saved PCA: {pca_result.shape}")

    # UMAP
    for n_neighbors, embedding in umap_results.items():
        np.save(DATA_DIR / f"umap_n{n_neighbors}_{TIMESTAMP}.npy", embedding)
        print(f"  Saved UMAP n={n_neighbors}: {embedding.shape}")

    # t-SNE
    for perplexity, embedding in tsne_results.items():
        np.save(DATA_DIR / f"tsne_p{perplexity}_{TIMESTAMP}.npy", embedding)
        print(f"  Saved t-SNE p={perplexity}: {embedding.shape}")

    # Save summary
    summary = {
        'timestamp': TIMESTAMP,
        'pca': {
            'n_components': int(pca_result.shape[1]),
            'variance_pc1': float(pca_model.explained_variance_ratio_[0]),
            'variance_pc1_5': float(np.sum(pca_model.explained_variance_ratio_[:5])),
            'variance_pc1_10': float(np.sum(pca_model.explained_variance_ratio_[:10])),
        },
        'umap': {
            'n_neighbors_tested': list(umap_results.keys()),
            'available': len(umap_results) > 0
        },
        'tsne': {
            'perplexities_tested': list(tsne_results.keys())
        }
    }

    # Compute embedding-environment correlations for UMAP
    if umap_results and 30 in umap_results:
        corrs = compute_embedding_correlations(umap_results[30], env_matrix, env_cols)
        # Find strongest correlations
        top_corrs = {}
        for dim, dim_corrs in corrs.items():
            sorted_corrs = sorted(dim_corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            top_corrs[dim] = {k: float(v) for k, v in sorted_corrs if np.isfinite(v)}
        summary['umap_env_correlations'] = top_corrs

    with open(REPORTS_DIR / f"dimensionality_reduction_summary_{TIMESTAMP}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved summary")

def main():
    print("=" * 70)
    print("PHASE 2: DIMENSIONALITY REDUCTION (algaGPT)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load data
    env_matrix, pfam_matrix, coordinates, sample_ids, env_cols = load_processed_data()

    # Run PCA
    pca_result, pca_model, explained_var = run_pca(pfam_matrix)

    # Run UMAP
    umap_results = run_umap(pfam_matrix)

    # Run t-SNE (using PCA for speed)
    tsne_results = run_tsne(pfam_matrix, pca_result=pca_result)

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    plot_pca_variance(explained_var)

    # PCA embeddings
    plot_embedding_grid({'pca': pca_result[:, :2]}, env_matrix, env_cols, coordinates, 'PCA')

    # UMAP embeddings
    if umap_results:
        plot_embedding_grid(umap_results, env_matrix, env_cols, coordinates, 'UMAP')
        if 30 in umap_results:
            plot_geographic_embedding(umap_results[30], coordinates, 'UMAP')

    # t-SNE embeddings
    if tsne_results:
        plot_embedding_grid(tsne_results, env_matrix, env_cols, coordinates, 'tSNE')

    # Method comparison
    plot_method_comparison(pca_result, umap_results, tsne_results, env_matrix, env_cols)

    # Save results
    save_embeddings(pca_result, pca_model, umap_results, tsne_results, env_matrix, env_cols)

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)

    return pca_result, umap_results, tsne_results

if __name__ == "__main__":
    pca_result, umap_results, tsne_results = main()
