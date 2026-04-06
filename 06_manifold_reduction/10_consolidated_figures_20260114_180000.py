#!/usr/bin/env python3
"""
Consolidated Publication Figures - Maximum Data Density

Two multi-panel figures following FIGURE_PROTOCOL.md:
  Figure 1: Manifold Analysis (PCA, UMAP, t-SNE, CCA, Geographic)
  Figure 2: Predictive Modeling (Correlations, RF/XGB, Reverse, SHAP)

Provenance:
  Input: data/*.npy, reports/*.csv
  Output: figures/Figure1_manifold_*.pdf, figures/Figure2_modeling_*.pdf
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
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Data integrity enforcement
import sys
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
mpl.rcParams['legend.handlelength'] = 1
mpl.rcParams['legend.handletextpad'] = 0.3
mpl.rcParams['legend.borderpad'] = 0.2
mpl.rcParams['legend.borderaxespad'] = 0.2

# Paths
BASE_DIR = Path("/media/drn/External1/TARA-Oceans/03_analyses/env_pfam_manifold")
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Color schemes
DIVERGING_COLORS = [
    (0.0, 0.3, 0.7), (0.3, 0.5, 0.9), (0.7, 0.8, 0.95),
    (0.95, 0.95, 0.95), (0.95, 0.8, 0.7), (0.9, 0.4, 0.3), (0.7, 0.1, 0.1)
]
DIVERGING_CMAP = LinearSegmentedColormap.from_list('diverging', DIVERGING_COLORS)

def save_figure(fig, name):
    """Save as PDF and SVG with transparent background."""
    for fmt in ['pdf', 'svg']:
        fig.savefig(FIGURES_DIR / f"{name}_{TIMESTAMP}.{fmt}",
                   format=fmt, bbox_inches='tight', transparent=True, edgecolor='none')
    print(f"  Saved: {name}_{TIMESTAMP}.pdf/.svg")

def load_data():
    """Load all data."""
    print("Loading data...")
    data = {}

    # Matrices
    data['env_matrix'] = np.load(sorted(DATA_DIR.glob("env_matrix_*.npy"))[-1])
    data['pfam_matrix'] = np.load(sorted(DATA_DIR.glob("pfam_matrix_*.npy"))[-1])
    data['coordinates'] = np.load(sorted(DATA_DIR.glob("coordinates_*.npy"))[-1])

    # Column names
    with open(sorted(DATA_DIR.glob("env_columns_*.txt"))[-1]) as f:
        data['env_cols'] = [l.strip() for l in f]
    with open(sorted(DATA_DIR.glob("pfam_columns_*.txt"))[-1]) as f:
        data['pfam_cols'] = [l.strip() for l in f]

    # Embeddings
    for name, pattern in [('pca', 'pca_embedding_*.npy'), ('umap', 'umap_n30_*.npy'),
                          ('tsne', 'tsne_p30_*.npy'), ('cca_joint', 'cca_joint_embedding_*.npy')]:
        files = sorted(DATA_DIR.glob(pattern))
        if files:
            data[name] = np.load(files[-1])

    # Reports
    for name, pattern in [('corr_matrix', 'correlation_matrix_*.csv'),
                          ('rf_results', 'rf_results_*.csv'),
                          ('xgb_results', 'xgb_results_*.csv'),
                          ('reverse_rf', 'reverse_rf_results_*.csv'),
                          ('shap_reverse', 'xgboost_reverse_full_*.csv'),
                          ('shap_forward', 'xgboost_forward_full_*.csv'),
                          ('shap_importance', 'shap_reverse_importance_*.csv')]:
        files = sorted(REPORTS_DIR.glob(pattern))
        if files:
            data[name] = pd.read_csv(files[-1], index_col=0 if 'matrix' in name else None)

    print(f"  {data['env_matrix'].shape[0]} samples, {len(data['env_cols'])} env vars, {len(data['pfam_cols'])} PFAMs")
    return data

def figure1_manifold(data):
    """
    Figure 1: Manifold Analysis - Full page, maximum density

    Layout (7 x 9 inches):
    Row 0: PCA variance (2 panels) | UMAP geographic (2 panels)
    Rows 1-3: 3x3 embedding grid (PCA/UMAP/tSNE x SST/Chl/Depth)
    """
    print("\n" + "=" * 60)
    print("FIGURE 1: MANIFOLD ANALYSIS")
    print("=" * 60)

    fig = plt.figure(figsize=(7, 8))

    # Tight grid layout - increase spacing between row 0 and rows 1-3 to prevent overlap
    gs = gridspec.GridSpec(4, 6, figure=fig, height_ratios=[0.75, 1, 1, 1],
                          hspace=0.35, wspace=0.15)

    # =========== ROW 0: PCA Variance + Geographic ===========

    # Panel A: PCA Scree
    ax_a = fig.add_subplot(gs[0, 0:2])
    pca = PCA(n_components=30)
    pca.fit(data['pfam_matrix'])
    exp_var = pca.explained_variance_ratio_ * 100
    ax_a.bar(range(1, 31), exp_var, color='steelblue', width=0.8, edgecolor='none')
    ax_a.set_xlabel('Principal Component')
    ax_a.set_ylabel('Variance (%)')
    ax_a.set_xlim(0.5, 30.5)
    ax_a.set_xticks([1, 10, 20, 30])
    ax_a.set_title('Scree Plot', fontsize=6)
    ax_a.text(-0.12, 1.08, 'A', transform=ax_a.transAxes, fontweight='bold', fontsize=7)

    # Panel B: Cumulative variance
    ax_b = fig.add_subplot(gs[0, 2:4])
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax_b.plot(range(1, 31), cumvar, 'b-', lw=0.75)
    ax_b.axhline(50, color='#CC0000', ls='--', lw=0.5)
    ax_b.text(22, 52, '50%', fontsize=5, color='#CC0000')
    ax_b.set_xlabel('Number of PCs')
    ax_b.set_ylabel('Cumulative Var. (%)')
    ax_b.set_xlim(1, 30)
    ax_b.set_ylim(20, 70)
    ax_b.set_title('Cumulative Variance', fontsize=6)
    ax_b.text(-0.12, 1.08, 'B', transform=ax_b.transAxes, fontweight='bold', fontsize=7)

    # Panel C: UMAP by Latitude
    ax_c = fig.add_subplot(gs[0, 4])
    if 'umap' in data:
        sc = ax_c.scatter(data['umap'][:, 0], data['umap'][:, 1],
                         c=data['coordinates'][:, 0], cmap='coolwarm', s=0.8, alpha=0.8)
        cb = plt.colorbar(sc, ax=ax_c, shrink=0.8, aspect=12, pad=0.02)
        cb.set_label('Latitude (°)', fontsize=5)
        cb.ax.tick_params(labelsize=4, width=0.25, length=1)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.set_title('UMAP by Lat', fontsize=6)
    ax_c.text(-0.08, 1.08, 'C', transform=ax_c.transAxes, fontweight='bold', fontsize=7)

    # Panel D: UMAP by Longitude
    ax_d = fig.add_subplot(gs[0, 5])
    if 'umap' in data:
        sc = ax_d.scatter(data['umap'][:, 0], data['umap'][:, 1],
                         c=data['coordinates'][:, 1], cmap='twilight', s=0.8, alpha=0.8)
        cb = plt.colorbar(sc, ax=ax_d, shrink=0.8, aspect=12, pad=0.02)
        cb.set_label('Longitude (°)', fontsize=5)
        cb.ax.tick_params(labelsize=4, width=0.25, length=1)
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    ax_d.set_title('UMAP by Lon', fontsize=6)
    ax_d.text(-0.08, 1.08, 'D', transform=ax_d.transAxes, fontweight='bold', fontsize=7)

    # =========== ROW 1-3: 3x3 Embedding Grid ===========

    embeddings = [('PCA', data.get('pca')), ('UMAP', data.get('umap')), ('t-SNE', data.get('tsne'))]
    env_vars = [('sst_mean_c', 'SST (°C)'), ('chl_mean_mg_m3', 'Chl-a (mg/m³)'), ('bathymetry_m', 'Depth (m)')]

    panel_labels = ['E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    panel_idx = 0

    for row, (method_name, emb) in enumerate(embeddings):
        for col, (var, label) in enumerate(env_vars):
            ax = fig.add_subplot(gs[1 + row, col * 2:col * 2 + 2])

            if emb is not None and var in data['env_cols']:
                idx = data['env_cols'].index(var)
                vals = data['env_matrix'][:, idx]
                mask = np.isfinite(vals)
                if mask.any():
                    sc = ax.scatter(emb[mask, 0], emb[mask, 1], c=vals[mask],
                                   cmap='viridis', s=0.8, alpha=0.8)
                    cb = plt.colorbar(sc, ax=ax, shrink=0.7, aspect=12, pad=0.02)
                    cb.ax.tick_params(labelsize=4, width=0.25, length=1)

            ax.set_xticks([])
            ax.set_yticks([])

            # Column headers (top row only)
            if row == 0:
                ax.set_title(label, fontsize=6)

            # Row labels (left column only)
            if col == 0:
                ax.set_ylabel(method_name, fontsize=6)

            # Axis labels for bottom row
            if row == 2:
                ax.set_xlabel('Dim 1', fontsize=5)

            ax.text(-0.05, 1.05, panel_labels[panel_idx], transform=ax.transAxes,
                   fontweight='bold', fontsize=7)
            panel_idx += 1

    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05)
    save_figure(fig, 'Figure1_manifold')
    plt.close()

def figure2_modeling(data):
    """
    Figure 2: Predictive Modeling - Full page

    Layout (7 x 9 inches):
    Row 0: Biclustered correlation heatmap with dendrograms
    Row 1: Model comparison (3 panels)
    Row 2: Reverse modeling + SHAP importance
    """
    print("\n" + "=" * 60)
    print("FIGURE 2: PREDICTIVE MODELING")
    print("=" * 60)

    fig = plt.figure(figsize=(7, 8.5))

    # Main grid with tighter spacing
    gs_main = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.4, 0.9, 1.1],
                                hspace=0.25)

    # =========== ROW 0: Biclustered Correlation Heatmap ===========

    if 'corr_matrix' in data:
        corr_df = data['corr_matrix']

        # Top 50 PFAMs by max absolute correlation
        max_corr = np.abs(corr_df.values).max(axis=0)
        top_idx = np.argsort(max_corr)[-50:]
        corr_sub = corr_df.iloc[:, top_idx]

        # Clean data for clustering
        corr_clean = np.nan_to_num(corr_sub.values, 0)

        # BICLUSTERING: Cluster both rows and columns
        row_link = linkage(pdist(corr_clean), method='ward')
        col_link = linkage(pdist(corr_clean.T), method='ward')
        row_order = dendrogram(row_link, no_plot=True)['leaves']
        col_order = dendrogram(col_link, no_plot=True)['leaves']

        corr_ordered = corr_clean[row_order, :][:, col_order]

        # Create sub-gridspec for heatmap with dendrograms
        # Use 3 columns: dendrogram | y-labels space | heatmap
        gs_heat = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_main[0],
                                                    height_ratios=[0.12, 1],
                                                    width_ratios=[0.06, 0.18, 1],
                                                    hspace=0.02, wspace=0.01)

        # Top dendrogram
        ax_dend_top = fig.add_subplot(gs_heat[0, 2])
        dendrogram(col_link, ax=ax_dend_top, no_labels=True,
                   color_threshold=0, above_threshold_color='#333333')
        ax_dend_top.set_xticks([])
        ax_dend_top.set_yticks([])
        for spine in ax_dend_top.spines.values():
            spine.set_visible(False)

        # Left dendrogram
        ax_dend_left = fig.add_subplot(gs_heat[1, 0])
        dendrogram(row_link, ax=ax_dend_left, orientation='left', no_labels=True,
                   color_threshold=0, above_threshold_color='#333333')
        ax_dend_left.set_xticks([])
        ax_dend_left.set_yticks([])
        for spine in ax_dend_left.spines.values():
            spine.set_visible(False)

        # Main heatmap
        ax_a = fig.add_subplot(gs_heat[1, 2])
        im = ax_a.imshow(corr_ordered, aspect='auto', cmap='viridis', vmin=-0.5, vmax=0.5)

        # Labels
        row_labels = [corr_df.index[i] for i in row_order]
        col_labels = [corr_df.columns[top_idx[i]].split('.')[0].replace('PF', '') for i in col_order]

        ax_a.set_yticks(range(len(row_labels)))
        ax_a.set_yticklabels(row_labels, fontsize=4)
        ax_a.set_xticks(range(len(col_labels)))
        ax_a.set_xticklabels(col_labels, rotation=90, fontsize=4, ha='center')
        ax_a.set_xlabel('PFAM Domain', fontsize=6)
        ax_a.yaxis.set_ticks_position('left')
        ax_a.tick_params(axis='y', pad=1)

        # Colorbar
        cb = plt.colorbar(im, ax=ax_a, shrink=0.6, aspect=15, pad=0.02)
        cb.set_label('Spearman ρ', fontsize=5)
        cb.ax.tick_params(labelsize=4, width=0.25, length=1)

        ax_a.text(-0.25, 1.02, 'A', transform=ax_a.transAxes, fontweight='bold', fontsize=7)

    # =========== ROW 1: Model Comparison (3 panels) ===========

    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[1],
                                                wspace=0.3)

    # Panel B: R² histogram
    ax_b = fig.add_subplot(gs_row1[0])
    if 'rf_results' in data and 'xgb_results' in data:
        rf_r2 = data['rf_results']['r2_test']
        xgb_r2 = data['xgb_results']['r2_test']
        bins = np.linspace(-0.1, 0.7, 25)
        ax_b.hist(rf_r2, bins=bins, alpha=0.7, label='RF', color='steelblue', edgecolor='none')
        ax_b.hist(xgb_r2, bins=bins, alpha=0.7, label='XGB', color='darkorange', edgecolor='none')
        ax_b.set_xlabel('Test R²')
        ax_b.set_ylabel('Count')
        ax_b.set_title('R² Distribution', fontsize=6)
        ax_b.legend(loc='upper right', frameon=False, fontsize=5)
    ax_b.text(-0.15, 1.08, 'B', transform=ax_b.transAxes, fontweight='bold', fontsize=7)

    # Panel C: RF vs XGB scatter
    ax_c = fig.add_subplot(gs_row1[1])
    if 'rf_results' in data and 'xgb_results' in data:
        ax_c.scatter(rf_r2, xgb_r2, s=3, alpha=0.6, color='steelblue')
        ax_c.plot([-0.1, 0.7], [-0.1, 0.7], 'k--', lw=0.5)
        ax_c.set_xlabel('Random Forest R²')
        ax_c.set_ylabel('XGBoost R²')
        ax_c.set_xlim(-0.1, 0.7)
        ax_c.set_ylim(-0.1, 0.7)
        ax_c.set_aspect('equal')
        ax_c.set_title('Model Comparison', fontsize=6)
    ax_c.text(-0.15, 1.08, 'C', transform=ax_c.transAxes, fontweight='bold', fontsize=7)

    # Panel D: Top predictable PFAMs
    ax_d = fig.add_subplot(gs_row1[2])
    if 'rf_results' in data:
        top_rf = data['rf_results'].nlargest(15, 'r2_test')
        y_pos = range(len(top_rf))
        ax_d.barh(y_pos, top_rf['r2_test'], color='steelblue', height=0.7, edgecolor='none')
        ax_d.set_yticks(y_pos)
        ax_d.set_yticklabels([p.replace('PF', '').split('.')[0] for p in top_rf['pfam']], fontsize=5)
        ax_d.set_xlabel('Test R²')
        ax_d.set_title('Top Predictable PFAMs', fontsize=6)
        ax_d.invert_yaxis()
    ax_d.text(-0.15, 1.08, 'D', transform=ax_d.transAxes, fontweight='bold', fontsize=7)

    # =========== ROW 2: Reverse Modeling + SHAP ===========

    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[2],
                                                wspace=0.35)

    # Panel E: Reverse modeling (PFAM → Env)
    ax_e = fig.add_subplot(gs_row2[0])
    if 'reverse_rf' in data:
        rev_df = data['reverse_rf'].sort_values('r2_test', ascending=True)
        y_pos = range(len(rev_df))
        colors = ['steelblue' if r > 0.2 else 'lightgray' for r in rev_df['r2_test']]
        ax_e.barh(y_pos, rev_df['r2_test'], color=colors, height=0.7, edgecolor='none')
        ax_e.set_yticks(y_pos)
        ax_e.set_yticklabels(rev_df['env_var'], fontsize=4)
        ax_e.set_xlabel('Test R²')
        ax_e.axvline(0.2, color='red', ls='--', lw=0.5)
        ax_e.set_title('PFAM → Environment', fontsize=6)
    ax_e.text(-0.15, 1.05, 'E', transform=ax_e.transAxes, fontweight='bold', fontsize=7)

    # Panel F: SHAP importance (reverse)
    ax_f = fig.add_subplot(gs_row2[1])
    if 'shap_importance' in data:
        shap_df = data['shap_importance']
        agg = shap_df.groupby('feature')['mean_abs_shap'].mean().sort_values(ascending=False)
        top_shap = agg.head(20)
        y_pos = range(len(top_shap))
        ax_f.barh(y_pos, top_shap.values, color='teal', height=0.7, edgecolor='none')
        ax_f.set_yticks(y_pos)
        ax_f.set_yticklabels([f.replace('PF', '').split('.')[0] for f in top_shap.index], fontsize=5)
        ax_f.set_xlabel('Mean |SHAP|')
        ax_f.set_title('Top PFAM Features (SHAP)', fontsize=6)
        ax_f.invert_yaxis()
    ax_f.text(-0.15, 1.05, 'F', transform=ax_f.transAxes, fontweight='bold', fontsize=7)

    # Panel G: Forward modeling top results
    ax_g = fig.add_subplot(gs_row2[2])
    if 'shap_forward' in data:
        fwd_df = data['shap_forward'].sort_values('r2_test', ascending=False).head(20)
        y_pos = range(len(fwd_df))
        ax_g.barh(y_pos, fwd_df['r2_test'], color='darkorange', height=0.7, edgecolor='none')
        ax_g.set_yticks(y_pos)
        ax_g.set_yticklabels([p.replace('PF', '').split('.')[0] for p in fwd_df['pfam']], fontsize=5)
        ax_g.set_xlabel('Test R²')
        ax_g.set_title('Environment → PFAM', fontsize=6)
        ax_g.invert_yaxis()
    ax_g.text(-0.15, 1.05, 'G', transform=ax_g.transAxes, fontweight='bold', fontsize=7)

    plt.subplots_adjust(left=0.15, right=0.95, top=0.97, bottom=0.05)
    save_figure(fig, 'Figure2_modeling')
    plt.close()

def main():
    print("=" * 60)
    print("CONSOLIDATED PUBLICATION FIGURES")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    data = load_data()
    figure1_manifold(data)
    figure2_modeling(data)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
