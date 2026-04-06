#!/usr/bin/env python3
"""
Phase 2: Dimensionality Reduction and Visualization.

This script performs PCA, UMAP, and t-SNE on the PFAM data,
colored by environmental variables.

Provenance:
  Input: Processed data from Phase 1
  Date: 2026-01-13
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Data integrity enforcement (CLAUDE.md requirement)
from DataIntegrityGuard import enforce_data_integrity
enforce_data_integrity()

# Paths
BASE_DIR = Path("/media/drn/External1/TARA-Oceans/03_analyses/env_pfam_manifold")
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def load_processed_data():
    """Load processed data from Phase 1."""
    print("=" * 70)
    print("LOADING PROCESSED DATA")
    print("=" * 70)

    # Find the most recent processed files
    env_files = sorted(DATA_DIR.glob("env_matrix_*.npy"))
    pfam_files = sorted(DATA_DIR.glob("pfam_matrix_*.npy"))
    coord_files = sorted(DATA_DIR.glob("coordinates_*.npy"))
    id_files = sorted(DATA_DIR.glob("sample_ids_*.npy"))

    if not env_files or not pfam_files:
        raise FileNotFoundError("Processed data not found. Run Phase 1 first.")

    # Load most recent
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

    return env_matrix, pfam_matrix, coordinates, sample_ids, env_cols

def run_pca(pfam_matrix, n_components=50):
    """Run PCA on PFAM data."""
    print("\n" + "=" * 70)
    print("RUNNING PCA")
    print("=" * 70)

    pca = PCA(n_components=min(n_components, min(pfam_matrix.shape)))
    pca_result = pca.fit_transform(pfam_matrix)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print(f"  Components: {pca_result.shape[1]}")
    print(f"  Variance explained by PC1: {explained_var[0]*100:.2f}%")
    print(f"  Variance explained by PC1-5: {cumulative_var[4]*100:.2f}%")
    print(f"  Variance explained by PC1-10: {cumulative_var[9]*100:.2f}%")
    print(f"  Components for 80% variance: {np.argmax(cumulative_var >= 0.80) + 1}")
    print(f"  Components for 90% variance: {np.argmax(cumulative_var >= 0.90) + 1}")

    return pca_result, pca, explained_var

def run_umap(pfam_matrix, n_neighbors_list=[15, 30, 50]):
    """Run UMAP with different n_neighbors settings."""
    print("\n" + "=" * 70)
    print("RUNNING UMAP")
    print("=" * 70)

    umap_results = {}

    for n_neighbors in n_neighbors_list:
        print(f"  Running UMAP with n_neighbors={n_neighbors}...")
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=42
        )
        umap_result = reducer.fit_transform(pfam_matrix)
        umap_results[n_neighbors] = umap_result
        print(f"    Done. Shape: {umap_result.shape}")

    return umap_results

def run_tsne(pfam_matrix, perplexity_list=[5, 30, 50], pca_result=None):
    """Run t-SNE with different perplexity settings."""
    print("\n" + "=" * 70)
    print("RUNNING t-SNE")
    print("=" * 70)

    # Use PCA-reduced data for faster t-SNE
    if pca_result is not None and pca_result.shape[1] > 50:
        input_data = pca_result[:, :50]
        print(f"  Using first 50 PCA components for t-SNE input")
    else:
        input_data = pca_result if pca_result is not None else pfam_matrix

    tsne_results = {}

    for perplexity in perplexity_list:
        # Adjust perplexity if too large for sample size
        max_perplexity = (input_data.shape[0] - 1) // 3
        actual_perplexity = min(perplexity, max_perplexity)

        print(f"  Running t-SNE with perplexity={actual_perplexity}...")
        tsne = TSNE(
            n_components=2,
            perplexity=actual_perplexity,
            random_state=42,
            max_iter=1000,
            init='pca'
        )
        tsne_result = tsne.fit_transform(input_data)
        tsne_results[perplexity] = tsne_result
        print(f"    Done. Shape: {tsne_result.shape}")

    return tsne_results

def plot_pca_variance(explained_var, output_path):
    """Plot PCA variance explained."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scree plot
    ax = axes[0]
    ax.bar(range(1, min(31, len(explained_var)+1)), explained_var[:30] * 100)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('PCA Scree Plot')

    # Cumulative variance
    ax = axes[1]
    cumvar = np.cumsum(explained_var)
    ax.plot(range(1, len(cumvar)+1), cumvar * 100, 'b-')
    ax.axhline(y=80, color='r', linestyle='--', label='80% threshold')
    ax.axhline(y=90, color='g', linestyle='--', label='90% threshold')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance Explained (%)')
    ax.set_title('Cumulative Variance Explained')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def plot_embedding_grid(embedding, env_matrix, env_cols, title_prefix, output_path):
    """Plot embedding colored by multiple environmental variables."""
    # Select interesting environmental variables
    key_vars = ['sst_mean_c', 'chl_mean_mg_m3', 'salinity_psu_est',
                'bathymetry_m', 'solar_rad_mj_m2', 'distance_to_coast_km']

    # Find indices of available key variables
    available_vars = []
    available_indices = []
    for var in key_vars:
        if var in env_cols:
            idx = env_cols.index(var)
            available_vars.append(var)
            available_indices.append(idx)

    if len(available_vars) == 0:
        print(f"  No key environmental variables found for plotting")
        return

    # Create figure
    n_vars = min(6, len(available_vars))
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    for i, (var, idx) in enumerate(zip(available_vars[:n_vars], available_indices[:n_vars])):
        ax = axes[i]
        values = env_matrix[:, idx]

        # Handle NaN/Inf
        mask = np.isfinite(values)
        if not mask.any():
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(var)
            continue

        # Plot
        scatter = ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=values[mask], cmap='viridis', s=10, alpha=0.7
        )
        plt.colorbar(scatter, ax=ax, label=var)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f'{title_prefix} colored by {var}')

    # Hide unused axes
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def plot_geographic_map(embedding, coordinates, output_path):
    """Plot embedding with geographic coordinates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Embedding colored by latitude
    ax = axes[0]
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=coordinates[:, 0], cmap='coolwarm', s=10, alpha=0.7
    )
    plt.colorbar(scatter, ax=ax, label='Latitude')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Embedding colored by Latitude')

    # Embedding colored by longitude
    ax = axes[1]
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=coordinates[:, 1], cmap='twilight', s=10, alpha=0.7
    )
    plt.colorbar(scatter, ax=ax, label='Longitude')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Embedding colored by Longitude')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def save_embeddings(pca_result, umap_results, tsne_results, sample_ids):
    """Save all embeddings for downstream analysis."""
    print("\n" + "=" * 70)
    print("SAVING EMBEDDINGS")
    print("=" * 70)

    # PCA
    np.save(DATA_DIR / f"pca_embedding_{TIMESTAMP}.npy", pca_result)
    print(f"  Saved PCA embedding: {pca_result.shape}")

    # UMAP
    for n_neighbors, embedding in umap_results.items():
        np.save(DATA_DIR / f"umap_n{n_neighbors}_{TIMESTAMP}.npy", embedding)
        print(f"  Saved UMAP (n={n_neighbors}) embedding: {embedding.shape}")

    # t-SNE
    for perplexity, embedding in tsne_results.items():
        np.save(DATA_DIR / f"tsne_p{perplexity}_{TIMESTAMP}.npy", embedding)
        print(f"  Saved t-SNE (p={perplexity}) embedding: {embedding.shape}")

def main():
    print("=" * 70)
    print("PHASE 2: DIMENSIONALITY REDUCTION")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load data
    env_matrix, pfam_matrix, coordinates, sample_ids, env_cols = load_processed_data()

    # Run PCA
    pca_result, pca_model, explained_var = run_pca(pfam_matrix)

    # Run UMAP
    umap_results = run_umap(pfam_matrix)

    # Run t-SNE (using PCA-reduced data)
    tsne_results = run_tsne(pfam_matrix, pca_result=pca_result)

    # Visualization
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    # PCA variance plot
    plot_pca_variance(explained_var, FIGURES_DIR / f"pca_variance_{TIMESTAMP}.png")

    # PCA embedding plots
    plot_embedding_grid(pca_result[:, :2], env_matrix, env_cols, 'PCA',
                       FIGURES_DIR / f"pca_env_grid_{TIMESTAMP}.png")

    # UMAP plots (using n_neighbors=30 as default)
    if 30 in umap_results:
        plot_embedding_grid(umap_results[30], env_matrix, env_cols, 'UMAP',
                           FIGURES_DIR / f"umap_env_grid_{TIMESTAMP}.png")
        plot_geographic_map(umap_results[30], coordinates,
                           FIGURES_DIR / f"umap_geographic_{TIMESTAMP}.png")

    # t-SNE plots (using perplexity=30 as default)
    if 30 in tsne_results:
        plot_embedding_grid(tsne_results[30], env_matrix, env_cols, 't-SNE',
                           FIGURES_DIR / f"tsne_env_grid_{TIMESTAMP}.png")

    # Save embeddings
    save_embeddings(pca_result, umap_results, tsne_results, sample_ids)

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)

    return pca_result, umap_results, tsne_results

if __name__ == "__main__":
    pca_result, umap_results, tsne_results = main()
