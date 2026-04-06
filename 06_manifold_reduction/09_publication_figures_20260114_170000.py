#!/usr/bin/env python3
"""
Publication-Quality Figure Generation (FIGURE_PROTOCOL.md Compliant)

This script regenerates ALL analysis figures as data-dense vector graphics
following the journal figure protocol:
  - ALL text: 6pt Arial
  - Vector formats: PDF + SVG (NO PNG)
  - Transparent backgrounds
  - Maximized data density
  - No overlapping text

Figures Generated:
  1. PCA variance (scree plot + cumulative)
  2. Dimensionality reduction embeddings (PCA, UMAP, t-SNE) colored by environment
  3. Correlation heatmap (env x PFAM)
  4. Predictive modeling comparison (RF vs XGBoost)
  5. Feature importance (aggregated)
  6. Reverse modeling results
  7. CCA manifold visualization
  8. SHAP importance plots

Provenance:
  Input: data/*.npy, reports/*.csv
  Output: figures/*.pdf, figures/*.svg
  Date: 2026-01-14
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# Data integrity enforcement
import sys
sys.path.insert(0, str(Path(__file__).parent))
from DataIntegrityGuard import enforce_data_integrity
enforce_data_integrity()

# ============================================================================
# FIGURE PROTOCOL CONFIGURATION (STRICT COMPLIANCE)
# ============================================================================

# CRITICAL: Journal compatibility - TrueType fonts
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# Font configuration - ALL 6pt Arial
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6

# Line weights (0.25pt for publication)
mpl.rcParams['axes.linewidth'] = 0.25
mpl.rcParams['xtick.major.width'] = 0.25
mpl.rcParams['ytick.major.width'] = 0.25
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2

# Minimal padding
mpl.rcParams['axes.labelpad'] = 1
mpl.rcParams['xtick.major.pad'] = 1
mpl.rcParams['ytick.major.pad'] = 1

# Paths
BASE_DIR = Path("/media/drn/External1/TARA-Oceans/03_analyses/env_pfam_manifold")
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# COLOR SCHEMES (FIGURE_PROTOCOL.md)
# ============================================================================

# Blackbody-inspired for abundance data
ABUNDANCE_COLORS = [
    (1.0, 1.0, 1.0),    # White for NaN/zero
    (0.95, 0.95, 0.95), # Very light gray
    (0.1, 0.1, 0.1),    # Near black
    (0.4, 0.0, 0.0),    # Dark red
    (0.7, 0.0, 0.0),    # Red
    (0.9, 0.2, 0.0),    # Orange-red
    (1.0, 0.5, 0.0),    # Orange
    (1.0, 0.7, 0.0),    # Yellow-orange
    (1.0, 0.9, 0.2),    # Yellow
]
ABUNDANCE_CMAP = LinearSegmentedColormap.from_list('abundance', ABUNDANCE_COLORS)

# Blue-white-red diverging for correlations
DIVERGING_COLORS = [
    (0.0, 0.3, 0.7),    # Deep blue (negative)
    (0.3, 0.5, 0.9),    # Medium blue
    (0.7, 0.8, 0.95),   # Light blue
    (0.95, 0.95, 0.95), # Near white (zero)
    (0.95, 0.8, 0.7),   # Light red
    (0.9, 0.4, 0.3),    # Medium red
    (0.7, 0.1, 0.1),    # Deep red (positive)
]
DIVERGING_CMAP = LinearSegmentedColormap.from_list('diverging', DIVERGING_COLORS)

def save_figure(fig, name):
    """Save figure in PDF and SVG formats with transparent background."""
    for fmt in ['pdf', 'svg']:
        filepath = FIGURES_DIR / f"{name}_{TIMESTAMP}.{fmt}"
        fig.savefig(filepath, format=fmt,
                   bbox_inches='tight',
                   transparent=True,
                   edgecolor='none')
    print(f"  Saved: {name}_{TIMESTAMP}.pdf/.svg")

def load_data():
    """Load all required data files."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    data = {}

    # Environmental and PFAM matrices
    env_files = sorted(DATA_DIR.glob("env_matrix_*.npy"))
    pfam_files = sorted(DATA_DIR.glob("pfam_matrix_*.npy"))
    coord_files = sorted(DATA_DIR.glob("coordinates_*.npy"))

    data['env_matrix'] = np.load(env_files[-1])
    data['pfam_matrix'] = np.load(pfam_files[-1])
    data['coordinates'] = np.load(coord_files[-1])

    # Column names
    env_cols_file = sorted(DATA_DIR.glob("env_columns_*.txt"))[-1]
    pfam_cols_file = sorted(DATA_DIR.glob("pfam_columns_*.txt"))[-1]

    with open(env_cols_file) as f:
        data['env_cols'] = [line.strip() for line in f]
    with open(pfam_cols_file) as f:
        data['pfam_cols'] = [line.strip() for line in f]

    # Embeddings
    pca_files = sorted(DATA_DIR.glob("pca_embedding_*.npy"))
    umap_files = sorted(DATA_DIR.glob("umap_n30_*.npy"))
    tsne_files = sorted(DATA_DIR.glob("tsne_p30_*.npy"))
    cca_joint_files = sorted(DATA_DIR.glob("cca_joint_embedding_*.npy"))
    cca_env_files = sorted(DATA_DIR.glob("cca_env_embedding_*.npy"))
    cca_pfam_files = sorted(DATA_DIR.glob("cca_pfam_embedding_*.npy"))

    if pca_files:
        data['pca'] = np.load(pca_files[-1])
    if umap_files:
        data['umap'] = np.load(umap_files[-1])
    if tsne_files:
        data['tsne'] = np.load(tsne_files[-1])
    if cca_joint_files:
        data['cca_joint'] = np.load(cca_joint_files[-1])
    if cca_env_files:
        data['cca_env'] = np.load(cca_env_files[-1])
    if cca_pfam_files:
        data['cca_pfam'] = np.load(cca_pfam_files[-1])

    # Results from reports
    corr_files = sorted(REPORTS_DIR.glob("correlation_matrix_*.csv"))
    top_corr_files = sorted(REPORTS_DIR.glob("top_correlations_*.csv"))
    rf_files = sorted(REPORTS_DIR.glob("rf_results_*.csv"))
    xgb_files = sorted(REPORTS_DIR.glob("xgb_results_*.csv"))
    reverse_rf_files = sorted(REPORTS_DIR.glob("reverse_rf_results_*.csv"))
    reverse_xgb_files = sorted(REPORTS_DIR.glob("reverse_xgb_results_*.csv"))
    cca_loadings_files = sorted(REPORTS_DIR.glob("cca_env_loadings_*.csv"))

    if corr_files:
        data['corr_matrix'] = pd.read_csv(corr_files[-1], index_col=0)
    if top_corr_files:
        data['top_corr'] = pd.read_csv(top_corr_files[-1])
    if rf_files:
        data['rf_results'] = pd.read_csv(rf_files[-1])
    if xgb_files:
        data['xgb_results'] = pd.read_csv(xgb_files[-1])
    if reverse_rf_files:
        data['reverse_rf'] = pd.read_csv(reverse_rf_files[-1])
    if reverse_xgb_files:
        data['reverse_xgb'] = pd.read_csv(reverse_xgb_files[-1])
    if cca_loadings_files:
        data['cca_loadings'] = pd.read_csv(cca_loadings_files[-1], index_col=0)

    # SHAP results
    shap_reverse_files = sorted(REPORTS_DIR.glob("xgboost_reverse_full_*.csv"))
    shap_forward_files = sorted(REPORTS_DIR.glob("xgboost_forward_full_*.csv"))
    shap_importance_files = sorted(REPORTS_DIR.glob("shap_reverse_importance_*.csv"))

    if shap_reverse_files:
        data['shap_reverse'] = pd.read_csv(shap_reverse_files[-1])
    if shap_forward_files:
        data['shap_forward'] = pd.read_csv(shap_forward_files[-1])
    if shap_importance_files:
        data['shap_importance'] = pd.read_csv(shap_importance_files[-1])

    print(f"  Samples: {data['env_matrix'].shape[0]}")
    print(f"  Environmental variables: {len(data['env_cols'])}")
    print(f"  PFAM domains: {len(data['pfam_cols'])}")

    return data

# ============================================================================
# FIGURE 1: PCA VARIANCE
# ============================================================================

def figure_pca_variance(data):
    """
    Figure 1: PCA Scree Plot and Cumulative Variance

    Two-panel figure showing variance explained by principal components.
    """
    print("\n" + "=" * 70)
    print("FIGURE 1: PCA VARIANCE")
    print("=" * 70)

    from sklearn.decomposition import PCA

    # Fit PCA to get variance explained
    pca = PCA(n_components=min(50, min(data['pfam_matrix'].shape)))
    pca.fit(data['pfam_matrix'])
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # Create figure - 7 inch width (double column)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

    # Panel A: Scree plot
    ax = axes[0]
    n_show = 30
    ax.bar(range(1, n_show+1), explained_var[:n_show] * 100,
           color='steelblue', edgecolor='none', width=0.8)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_xlim(0.5, n_show + 0.5)
    ax.set_xticks([1, 10, 20, 30])
    ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontweight='bold',
            va='top', ha='left')

    # Panel B: Cumulative variance
    ax = axes[1]
    ax.plot(range(1, len(cumulative_var)+1), cumulative_var * 100,
            'b-', linewidth=0.5)
    ax.axhline(y=80, color='#CC0000', linestyle='--', linewidth=0.25)
    ax.axhline(y=90, color='#006600', linestyle='--', linewidth=0.25)

    # Find PCs for 80% and 90% (if achievable)
    if cumulative_var[-1] >= 0.80:
        pc_80 = np.argmax(cumulative_var >= 0.80) + 1
        ax.annotate(f'80% (PC{pc_80})', xy=(pc_80, 80), xytext=(pc_80+5, 75),
                    fontsize=6, color='#CC0000',
                    arrowprops=dict(arrowstyle='->', color='#CC0000', lw=0.25))
    else:
        # Show max achieved variance
        ax.text(0.95, 0.15, f'Max: {cumulative_var[-1]*100:.1f}%\nat PC{len(cumulative_var)}',
                transform=ax.transAxes, fontsize=6, ha='right', va='bottom')

    if cumulative_var[-1] >= 0.90:
        pc_90 = np.argmax(cumulative_var >= 0.90) + 1
        ax.annotate(f'90% (PC{pc_90})', xy=(pc_90, 90), xytext=(pc_90+5, 85),
                    fontsize=6, color='#006600',
                    arrowprops=dict(arrowstyle='->', color='#006600', lw=0.25))

    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance (%)')
    ax.set_xlim(0, min(50, len(cumulative_var)))
    ax.set_ylim(0, 100)
    ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontweight='bold',
            va='top', ha='left')

    plt.tight_layout(w_pad=1)
    save_figure(fig, 'fig1_pca_variance')
    plt.close()

# ============================================================================
# FIGURE 2: DIMENSIONALITY REDUCTION EMBEDDINGS
# ============================================================================

def figure_embeddings(data):
    """
    Figure 2: Multi-panel dimensionality reduction embeddings

    3x3 grid showing PCA, UMAP, t-SNE colored by key environmental variables.
    High data density - 9 plots in one figure.
    """
    print("\n" + "=" * 70)
    print("FIGURE 2: DIMENSIONALITY REDUCTION EMBEDDINGS")
    print("=" * 70)

    # Key environmental variables for coloring
    key_vars = ['sst_mean_c', 'chl_mean_mg_m3', 'bathymetry_m']
    var_labels = ['SST (C)', 'Chl (mg/m3)', 'Depth (m)']

    embeddings = {
        'PCA': data.get('pca', None),
        'UMAP': data.get('umap', None),
        't-SNE': data.get('tsne', None)
    }

    # Filter to available embeddings
    embeddings = {k: v for k, v in embeddings.items() if v is not None}

    if not embeddings:
        print("  No embeddings available, skipping figure.")
        return

    n_methods = len(embeddings)
    n_vars = len(key_vars)

    # Create figure
    fig, axes = plt.subplots(n_methods, n_vars, figsize=(7, 7))
    if n_methods == 1:
        axes = axes.reshape(1, -1)

    for i, (method_name, embedding) in enumerate(embeddings.items()):
        for j, (var, label) in enumerate(zip(key_vars, var_labels)):
            ax = axes[i, j]

            # Get environmental values
            if var in data['env_cols']:
                idx = data['env_cols'].index(var)
                values = data['env_matrix'][:, idx]

                # Handle NaN
                mask = np.isfinite(values)

                if mask.any():
                    scatter = ax.scatter(
                        embedding[mask, 0], embedding[mask, 1],
                        c=values[mask], cmap='viridis', s=1, alpha=0.7
                    )

                    # Minimal colorbar
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
                    cbar.ax.tick_params(labelsize=5, width=0.25, length=1)
                    cbar.outline.set_linewidth(0.25)

            # Labels
            if i == 0:
                ax.set_title(label)
            if j == 0:
                ax.set_ylabel(method_name)

            ax.set_xticks([])
            ax.set_yticks([])

            # Panel label
            panel_idx = i * n_vars + j
            panel_letter = chr(65 + panel_idx)  # A, B, C, ...
            ax.text(0.02, 0.98, panel_letter, transform=ax.transAxes,
                   fontweight='bold', va='top', ha='left')

    plt.tight_layout(h_pad=0.3, w_pad=0.3)
    save_figure(fig, 'fig2_embeddings')
    plt.close()

# ============================================================================
# FIGURE 3: CORRELATION HEATMAP
# ============================================================================

def figure_correlation_heatmap(data):
    """
    Figure 3: High-density correlation heatmap with clustering

    Shows correlations between environmental variables and top PFAM domains.
    Includes dendrograms on both axes for hierarchical structure.
    """
    print("\n" + "=" * 70)
    print("FIGURE 3: CORRELATION HEATMAP")
    print("=" * 70)

    if 'corr_matrix' not in data:
        print("  Correlation matrix not available, skipping.")
        return

    corr_df = data['corr_matrix']

    # Select top 50 PFAM domains by max absolute correlation
    max_abs_corr = np.abs(corr_df.values).max(axis=0)
    top_idx = np.argsort(max_abs_corr)[-50:]
    corr_subset = corr_df.iloc[:, top_idx]

    # Handle NaN for clustering
    corr_clean = corr_subset.fillna(0).values

    # Compute linkages for dendrograms
    row_linkage = linkage(pdist(corr_clean), method='ward')
    col_linkage = linkage(pdist(corr_clean.T), method='ward')

    # Get ordering from dendrograms
    row_order = dendrogram(row_linkage, no_plot=True)['leaves']
    col_order = dendrogram(col_linkage, no_plot=True)['leaves']

    # Reorder data
    corr_ordered = corr_clean[row_order, :][:, col_order]
    row_labels = [corr_df.index[i] for i in row_order]
    col_labels = [corr_subset.columns[i].split('.')[0] for i in col_order]

    # Create figure - no dendrograms, cleaner layout for publication
    fig, ax = plt.subplots(figsize=(7, 4.5))

    im = ax.imshow(corr_ordered, aspect='auto', cmap=DIVERGING_CMAP,
                   vmin=-0.5, vmax=0.5)

    # Labels - abbreviate PFAM names
    col_labels_short = [c.replace('PF', '') for c in col_labels]
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels_short, rotation=90, ha='center')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Spearman rho')
    cbar.ax.tick_params(labelsize=5, width=0.25, length=1)
    cbar.outline.set_linewidth(0.25)

    plt.tight_layout()
    save_figure(fig, 'fig3_correlation_heatmap')
    plt.close()

# ============================================================================
# FIGURE 4: MODEL COMPARISON
# ============================================================================

def figure_model_comparison(data):
    """
    Figure 4: Predictive modeling comparison

    Three panels showing:
    A) R2 distribution histogram (RF vs XGBoost)
    B) RF vs XGBoost scatter
    C) Top 20 most predictable PFAM domains
    """
    print("\n" + "=" * 70)
    print("FIGURE 4: MODEL COMPARISON")
    print("=" * 70)

    if 'rf_results' not in data or 'xgb_results' not in data:
        print("  Model results not available, skipping.")
        return

    rf_df = data['rf_results']
    xgb_df = data['xgb_results']

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    # Panel A: R2 distribution
    ax = axes[0]
    bins = np.linspace(-0.2, 0.8, 30)
    ax.hist(rf_df['r2_test'], bins=bins, alpha=0.7, label='RF',
            color='steelblue', edgecolor='none')
    ax.hist(xgb_df['r2_test'], bins=bins, alpha=0.7, label='XGB',
            color='darkorange', edgecolor='none')
    ax.set_xlabel('Test R2')
    ax.set_ylabel('Count')
    ax.legend(loc='upper right', frameon=False)
    ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontweight='bold',
            va='top', ha='left')

    # Panel B: RF vs XGBoost scatter
    ax = axes[1]
    ax.scatter(rf_df['r2_test'], xgb_df['r2_test'],
               s=3, alpha=0.5, color='steelblue')
    ax.plot([-0.2, 0.8], [-0.2, 0.8], 'k--', linewidth=0.25, alpha=0.5)
    ax.set_xlabel('RF R2')
    ax.set_ylabel('XGBoost R2')
    ax.set_xlim(-0.2, 0.8)
    ax.set_ylim(-0.2, 0.8)
    ax.set_aspect('equal')
    ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontweight='bold',
            va='top', ha='left')

    # Panel C: Top 20 most predictable
    ax = axes[2]
    top_rf = rf_df.nlargest(20, 'r2_test')
    y_pos = range(len(top_rf))
    ax.barh(y_pos, top_rf['r2_test'], color='steelblue', height=0.7,
            edgecolor='none')
    ax.set_yticks(y_pos)
    # Abbreviate PFAM names
    labels = [p.split('.')[0].replace('PF', '') for p in top_rf['pfam']]
    ax.set_yticklabels(labels)
    ax.set_xlabel('Test R2')
    ax.invert_yaxis()
    ax.text(0.02, 0.02, 'C', transform=ax.transAxes, fontweight='bold',
            va='bottom', ha='left')

    plt.tight_layout(w_pad=0.5)
    save_figure(fig, 'fig4_model_comparison')
    plt.close()

# ============================================================================
# FIGURE 5: REVERSE MODELING RESULTS
# ============================================================================

def figure_reverse_modeling(data):
    """
    Figure 5: Reverse modeling (PFAM -> Environment)

    Shows predictability of environmental variables from PFAM profiles.
    """
    print("\n" + "=" * 70)
    print("FIGURE 5: REVERSE MODELING")
    print("=" * 70)

    if 'reverse_rf' not in data:
        print("  Reverse modeling results not available, skipping.")
        return

    rf_df = data['reverse_rf'].sort_values('r2_test', ascending=True)

    fig, ax = plt.subplots(figsize=(3.5, 3))

    y_pos = range(len(rf_df))
    colors = ['steelblue' if r > 0.2 else 'lightgray'
              for r in rf_df['r2_test']]

    ax.barh(y_pos, rf_df['r2_test'], color=colors, height=0.7, edgecolor='none')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rf_df['env_var'])
    ax.set_xlabel('Test R2')
    ax.axvline(0.2, color='red', linestyle='--', linewidth=0.25, alpha=0.5)

    # Add threshold annotation
    ax.text(0.21, 0.02, 'R2=0.2', transform=ax.transAxes, fontsize=5,
            color='red', va='bottom')

    plt.tight_layout()
    save_figure(fig, 'fig5_reverse_modeling')
    plt.close()

# ============================================================================
# FIGURE 6: CCA MANIFOLD
# ============================================================================

def figure_cca_manifold(data):
    """
    Figure 6: Canonical Correlation Analysis manifold

    Shows joint environment-PFAM manifold from CCA, colored by different variables.
    """
    print("\n" + "=" * 70)
    print("FIGURE 6: CCA MANIFOLD")
    print("=" * 70)

    if 'cca_joint' not in data:
        print("  CCA results not available, skipping.")
        return

    joint = data['cca_joint']

    # Key environmental variables for coloring
    key_vars = ['sst_mean_c', 'chl_mean_mg_m3', 'bathymetry_m',
                'distance_to_coast_km', 'solar_rad_mj_m2', 'modis_sst_mean_c']
    var_labels = ['SST', 'Chlorophyll', 'Bathymetry',
                  'Coast Distance', 'Solar Rad', 'MODIS SST']

    # Filter to available variables
    available = [(v, l) for v, l in zip(key_vars, var_labels)
                 if v in data['env_cols']]

    n_vars = min(6, len(available))
    n_cols = 3
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 4.5))
    axes = axes.flatten()

    for i, (var, label) in enumerate(available[:n_vars]):
        ax = axes[i]
        idx = data['env_cols'].index(var)
        values = data['env_matrix'][:, idx]

        mask = np.isfinite(values)
        if mask.any():
            scatter = ax.scatter(joint[mask, 0], joint[mask, 1],
                               c=values[mask], cmap='viridis', s=2,
                               alpha=0.7)
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=15)
            cbar.ax.tick_params(labelsize=5, width=0.25, length=1)
            cbar.outline.set_linewidth(0.25)

        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

        # Panel label
        panel_letter = chr(65 + i)
        ax.text(0.02, 0.98, panel_letter, transform=ax.transAxes,
               fontweight='bold', va='top', ha='left')

    # Hide unused axes
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(h_pad=0.3, w_pad=0.3)
    save_figure(fig, 'fig6_cca_manifold')
    plt.close()

# ============================================================================
# FIGURE 7: SHAP FEATURE IMPORTANCE
# ============================================================================

def figure_shap_importance(data):
    """
    Figure 7: SHAP feature importance from XGBoost models

    Two panels: forward modeling (env -> PFAM) and reverse (PFAM -> env)
    """
    print("\n" + "=" * 70)
    print("FIGURE 7: SHAP FEATURE IMPORTANCE")
    print("=" * 70)

    has_forward = 'shap_forward' in data
    has_reverse = 'shap_reverse' in data or 'shap_importance' in data

    if not (has_forward or has_reverse):
        print("  SHAP results not available, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    # Panel A: Reverse modeling (PFAM -> Env) - aggregate importance
    ax = axes[0]
    if 'shap_importance' in data:
        shap_df = data['shap_importance']

        # Aggregate by feature across all targets
        agg = shap_df.groupby('feature')['mean_abs_shap'].mean().sort_values(ascending=False)
        top_features = agg.head(25)

        y_pos = range(len(top_features))
        ax.barh(y_pos, top_features.values, color='steelblue',
                height=0.7, edgecolor='none')
        ax.set_yticks(y_pos)
        labels = [f.split('.')[0].replace('PF', '') for f in top_features.index]
        ax.set_yticklabels(labels)
        ax.set_xlabel('Mean |SHAP|')
        ax.invert_yaxis()
        ax.set_title('PFAM -> Environment')
    else:
        ax.text(0.5, 0.5, 'Not available', ha='center', va='center',
               transform=ax.transAxes)
    ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontweight='bold',
            va='top', ha='left')

    # Panel B: Forward modeling results (R2 performance)
    ax = axes[1]
    if 'shap_forward' in data:
        forward_df = data['shap_forward'].sort_values('r2_test', ascending=False)
        top_20 = forward_df.head(20)

        y_pos = range(len(top_20))
        ax.barh(y_pos, top_20['r2_test'], color='darkorange',
                height=0.7, edgecolor='none')
        ax.set_yticks(y_pos)
        labels = [p.split('.')[0].replace('PF', '') for p in top_20['pfam']]
        ax.set_yticklabels(labels)
        ax.set_xlabel('Test R2')
        ax.invert_yaxis()
        ax.set_title('Environment -> PFAM')
    else:
        ax.text(0.5, 0.5, 'Not available', ha='center', va='center',
               transform=ax.transAxes)
    ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontweight='bold',
            va='top', ha='left')

    plt.tight_layout(w_pad=1)
    save_figure(fig, 'fig7_shap_importance')
    plt.close()

# ============================================================================
# FIGURE 8: GEOGRAPHIC EMBEDDING
# ============================================================================

def figure_geographic(data):
    """
    Figure 8: Geographic distribution of samples in embedding space

    Shows UMAP embedding colored by latitude and longitude.
    """
    print("\n" + "=" * 70)
    print("FIGURE 8: GEOGRAPHIC EMBEDDING")
    print("=" * 70)

    if 'umap' not in data:
        print("  UMAP embedding not available, skipping.")
        return

    umap = data['umap']
    coords = data['coordinates']

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Panel A: Latitude
    ax = axes[0]
    scatter = ax.scatter(umap[:, 0], umap[:, 1], c=coords[:, 0],
                        cmap='coolwarm', s=2, alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Latitude')
    cbar.ax.tick_params(labelsize=5, width=0.25, length=1)
    cbar.outline.set_linewidth(0.25)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('UMAP by Latitude')
    ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontweight='bold',
            va='top', ha='left')

    # Panel B: Longitude
    ax = axes[1]
    scatter = ax.scatter(umap[:, 0], umap[:, 1], c=coords[:, 1],
                        cmap='twilight', s=2, alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Longitude')
    cbar.ax.tick_params(labelsize=5, width=0.25, length=1)
    cbar.outline.set_linewidth(0.25)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('UMAP by Longitude')
    ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontweight='bold',
            va='top', ha='left')

    plt.tight_layout(w_pad=0.5)
    save_figure(fig, 'fig8_geographic')
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PUBLICATION-QUALITY FIGURE GENERATION")
    print(f"Following FIGURE_PROTOCOL.md")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    print("\nProtocol compliance:")
    print(f"  Font: {mpl.rcParams['font.sans-serif'][0]} {mpl.rcParams['font.size']}pt")
    print(f"  Line width: {mpl.rcParams['axes.linewidth']}pt")
    print(f"  Output formats: PDF, SVG (vector only)")
    print(f"  Background: Transparent")

    # Load data
    data = load_data()

    # Generate all figures
    figure_pca_variance(data)
    figure_embeddings(data)
    figure_correlation_heatmap(data)
    figure_model_comparison(data)
    figure_reverse_modeling(data)
    figure_cca_manifold(data)
    figure_shap_importance(data)
    figure_geographic(data)

    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
