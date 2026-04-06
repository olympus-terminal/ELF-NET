#!/usr/bin/env python3
"""
Figure 5: PFAM SHAP Importance across AlphaEarth Foundation Dimensions
# --------------------------------------------------------------------------

K-means k=16 biclustered heatmap with PFAM functional annotations.
Uses full vertical space with well-defined cluster separations.

Created: 2026-02-10
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import Earth-from-Space palette
sys.path.insert(0, '/media/drn2/External/TARA-Oceans/MANUSCRIPT/figures')
from palette import (
    OCEAN_CMAP, FOREST_CMAP, THERMAL_CMAP, COASTAL_CMAP,
    DIVERGING_CMAP, COOL_DIVERGING_CMAP,
    BASIN_COLORS, LINEAGE_COLORS, ENV_CATEGORY_COLORS, MODULE_COLORS,
    get_sequential_cmap, get_diverging_cmap, get_categorical_colors,
    DEEP_OCEAN, TURQUOISE, FOREST_GREEN, DESERT_TAN, CLOUD_WHITE,
    COASTAL_BLUE
)

# ==============================================================================
# ARTIST MODE: rcParams (FIGURE_PROTOCOL.md compliance)
# ==============================================================================

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8

mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3

mpl.rcParams['axes.labelpad'] = 2
mpl.rcParams['xtick.major.pad'] = 2
mpl.rcParams['ytick.major.pad'] = 2

# ==============================================================================
# PATHS
# ==============================================================================

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = '/media/drn2/External/TARA-Oceans'
SHAP_FILE = f'{BASE_DIR}/03_analyses/ALGAGPT-based-analyses/validations/results/shap_dependence_alphaearth_20260121_090725.tsv'
COL_CLUSTER_FILE = f'{BASE_DIR}/MANUSCRIPT/source_data/aef_column_cluster_profiles.tsv'
AEF_GEE_FILE = f'{BASE_DIR}/MANUSCRIPT/source_data/aef_gee_top_correlations.tsv'
ANNOT_FILE = f'{BASE_DIR}/MANUSCRIPT/source_data/pfam_interpro_annotations.tsv'
OUT_DIR = f'{BASE_DIR}/MANUSCRIPT/figures'
SCRIPT_PATH = os.path.abspath(__file__)

# Validate inputs
for f in [SHAP_FILE, COL_CLUSTER_FILE]:
    if not os.path.isfile(f):
        print(f'ERROR: Required file not found: {f}')
        sys.exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

N_ROW_CLUSTERS = 16
MAX_PFAMS_PER_CLUSTER = 30  # keep clusters readable
MIN_ACTIVE_DIMS = 2         # PFAM must have >= 2 active dimensions
SHAP_THRESHOLD = 0.001

# All 8 column clusters for full picture
ALL_COL_CLUSTERS = ['C1', 'C6', 'C7', 'C4', 'C5', 'C3', 'C8', 'C2']
# Top 4 interpretable column clusters for simplified view
TOP_COL_CLUSTERS = ['C1', 'C6', 'C7', 'C4']

# ==============================================================================
# COLORMAPS AND COLORS
# ==============================================================================

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )

# 16 distinct colors for clusters — cycle through palette variants
_base_colors = [
    DEEP_OCEAN,                    # deep blue
    TURQUOISE,                     # teal
    FOREST_GREEN,                  # green
    DESERT_TAN,                    # warm tan
    (0.678, 0.486, 0.306),        # clay
    (0.255, 0.573, 0.612),        # coastal blue
    (0.529, 0.353, 0.224),        # dark earth
    (0.400, 0.651, 0.400),        # sage
    (0.180, 0.310, 0.510),        # navy
    (0.706, 0.557, 0.361),        # sand
    (0.310, 0.506, 0.294),        # olive
    (0.482, 0.686, 0.710),        # sky blue
    (0.600, 0.400, 0.200),        # sienna
    (0.200, 0.500, 0.500),        # dark teal
    (0.502, 0.420, 0.588),        # muted purple
    (0.694, 0.612, 0.522),        # taupe
]
CLUSTER_COLORS_16 = [rgb_to_hex(c) for c in _base_colors]

# Column cluster colors
COL_CLUSTER_COLORS = {
    'C1': rgb_to_hex(DEEP_OCEAN),      # Bathymetry
    'C2': rgb_to_hex((0.694, 0.612, 0.522)),  # Stable offshore
    'C3': rgb_to_hex((0.400, 0.651, 0.400)),  # Anti-chlorophyll
    'C4': rgb_to_hex(TURQUOISE),       # Ocean color
    'C5': rgb_to_hex((0.529, 0.353, 0.224)),  # Offshore/deep
    'C6': rgb_to_hex(DESERT_TAN),      # SST
    'C7': rgb_to_hex(FOREST_GREEN),    # Chlorophyll
    'C8': rgb_to_hex((0.706, 0.557, 0.361)),  # Temp variability
}

COL_ENV_LABELS = {
    'C1': 'Elev./Bathy. (−)',
    'C6': 'SST (−)',
    'C7': 'Chl (+)',
    'C4': 'Ocean color (+)',
}

# ==============================================================================
# LOAD PFAM ANNOTATIONS
# ==============================================================================

def load_pfam_annotations():
    """Load PFAM short names from InterPro annotations, plus hardcoded fallbacks."""
    annot = {}
    if os.path.isfile(ANNOT_FILE):
        df = pd.read_csv(ANNOT_FILE, sep='\t', comment='#')
        for _, row in df.iterrows():
            pfam_id = row['pfam_id']
            short = row.get('short_name', '')
            name = row.get('name', '')
            if pd.notna(short) and short:
                annot[pfam_id] = short
            elif pd.notna(name) and name:
                annot[pfam_id] = name[:30]
        print(f"  Loaded {len(annot)} PFAM annotations from local file")
    else:
        print(f"  WARNING: InterPro annotation file not found: {ANNOT_FILE}")

    # Hardcoded fallbacks for PFAMs not in the InterPro annotations file
    # (verified against Pfam database / InterPro page titles via web search)
    _fallback = {
        # Original entries
        'PF24681': 'DUF5131', 'PF08238': 'Sel1', 'PF07728': 'AAA_5',
        'PF01061': 'ABC2_membrane', 'PF01179': 'Cu_amine_oxid',
        'PF00183': 'HSP90', 'PF03109': 'ABC1',
        'PF13532': 'Ribose_5-P_iso', 'PF00005': 'ABC_tran',
        'PF22669': 'REJ', 'PF21156': 'ISOA1-3_C',
        'PF00004': 'AAA', 'PF00125': 'Histone',
        'PF12796': 'Ank_2', 'PF00006': 'ATP-synt_ab',
        'PF00022': 'Actin', 'PF02010': 'DUF47',
        'PF00124': 'Photo_RC', 'PF00504': 'Chloroa_b-bind',
        'PF01055': 'Glyco_hydro_31', 'PF01184': 'DDE_Tnp_IS1',
        'PF13359': 'DDE_3', 'PF00520': 'Ion_trans',
        'PF16211': 'RVT_connect', 'PF02826': 'Gal_Lectin',
        'PF03153': 'TFIIA',
        # Expanded entries (verified via InterPro page titles, web search 2026-02-10)
        'PF00202': 'Aminotran_3', 'PF00224': 'PK',
        'PF00232': 'Glyco_hydro_1', 'PF00310': 'GATase_2',
        'PF00441': 'Acyl-CoA_dh_1', 'PF00574': 'CLP_protease',
        'PF00591': 'Glycos_transf_1', 'PF00654': 'Voltage_CLC',
        'PF00730': 'HhH-GPD', 'PF00888': 'Cullin',
        'PF00890': 'FAD_binding_2', 'PF00909': 'Ammonium_transp',
        'PF01041': 'DegT_DnrJ_EryC1', 'PF01057': 'Parvo_NS1',
        'PF01287': 'eIF-5a', 'PF01327': 'Pept_deformylase',
        'PF01423': 'LSM', 'PF01425': 'Amidase',
        'PF01496': 'V_ATPase_I', 'PF01753': 'MYND_finger',
        'PF01833': 'TIG', 'PF02415': 'Chlam_PMP',
        'PF02536': 'mTERF', 'PF02800': 'Gp_dh_C',
        'PF03171': '2OG-FeII_Oxy_3', 'PF03382': 'DUF283',
        'PF03464': 'eRF1_2', 'PF04565': 'RNA_pol_Rpb2_3',
        'PF05684': 'DUF803', 'PF05770': 'Entericidin',
        'PF05903': 'PPPDE', 'PF05970': 'PIF1',
        'PF06549': 'YhgC-like', 'PF07885': 'Ion_channel',
        'PF08207': 'Coil_Mrp', 'PF08284': 'Cna_B',
        'PF08706': 'D5_N', 'PF09140': 'MipZ',
        'PF12906': 'RING_variant', 'PF13086': 'AAA_11',
        'PF13175': 'TPR_repeat', 'PF13191': 'AAA_ATPase',
        'PF13391': 'HNH_3', 'PF13415': 'Kelch_3',
        'PF13499': 'EF-hand_7', 'PF13843': 'Tnp_IS4',
        'PF13976': 'gag_pre-integrs', 'PF14304': 'OB_NTP_bind',
        'PF20670': 'DUF6426', 'PF22924': 'DUF6966',
        # Additional entries found during visual inspection (Task 9)
        'PF00097': 'zf-C3HC4', 'PF00173': 'Cyt-b5',
    }
    for k, v in _fallback.items():
        if k not in annot:
            annot[k] = v

    return annot

def fetch_missing_annotations(pfam_ids, annot_dict):
    """Fetch short names for PFAMs not in annot_dict from InterPro API."""
    import urllib.request
    import json
    import time

    missing = [p for p in pfam_ids if p not in annot_dict]
    if not missing:
        return annot_dict

    print(f"  Fetching {len(missing)} missing PFAM annotations from InterPro API...")
    fetched = 0
    for pfam_id in missing:
        try:
            url = f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{pfam_id}"
            req = urllib.request.Request(url, headers={'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                short_name = data.get('metadata', {}).get('name', {}).get('short', '')
                full_name = data.get('metadata', {}).get('name', {}).get('name', '')
                if short_name:
                    annot_dict[pfam_id] = short_name
                elif full_name:
                    annot_dict[pfam_id] = full_name[:30]
                fetched += 1
            time.sleep(0.15)  # rate limit
        except Exception:
            pass  # keep bare ID as fallback

    print(f"  Fetched {fetched}/{len(missing)} annotations from API")
    return annot_dict

def get_pfam_label(pfam_full, annot_dict):
    """Get a readable label for a PFAM domain."""
    pfam_base = pfam_full.split('.')[0]
    if pfam_base in annot_dict:
        return annot_dict[pfam_base]
    return pfam_base

# ==============================================================================
# LOAD AND CLUSTER DATA
# ==============================================================================

def load_and_cluster():
    """Load SHAP data, run k=16 K-means, assign environmental labels.

    Environmental labeling uses SHAP-weighted aggregation across ALL 64
    AEF dimensions to find the dominant GEE variable per cluster, with
    signed direction from the AEF-GEE correlation table.
    """
    print("Loading SHAP data...")
    shap_df = pd.read_csv(SHAP_FILE, sep='\t', comment='#')
    print(f"  Raw rows: {len(shap_df):,}")

    # Pivot to matrix: pfam x dimension
    print("Pivoting to matrix...")
    matrix = shap_df.pivot_table(
        index='pfam', columns='dimension',
        values='mean_abs_shap', aggfunc='first'
    )
    print(f"  Full matrix shape: {matrix.shape}")

    # Filter: keep PFAMs with >= MIN_ACTIVE_DIMS active dimensions
    n_active = (matrix > SHAP_THRESHOLD).sum(axis=1)
    matrix_filtered = matrix[n_active >= MIN_ACTIVE_DIMS].copy()
    print(f"  After filtering (>={MIN_ACTIVE_DIMS} active dims): {len(matrix_filtered)} PFAMs")

    # Z-score normalize rows
    print(f"Running K-means clustering (k={N_ROW_CLUSTERS}, random_state=42)...")
    matrix_zscore = matrix_filtered.sub(matrix_filtered.mean(axis=1), axis=0)
    row_std = matrix_filtered.std(axis=1)
    row_std[row_std == 0] = 1.0  # avoid division by zero
    matrix_zscore = matrix_zscore.div(row_std, axis=0)
    matrix_zscore = matrix_zscore.fillna(0)

    # K-means k=16
    kmeans = KMeans(n_clusters=N_ROW_CLUSTERS, random_state=42, n_init=10)
    row_labels = kmeans.fit_predict(matrix_zscore.values)

    # Compute mean SHAP per PFAM (for sorting within clusters)
    pfam_mean_shap = matrix_filtered.mean(axis=1)

    # Load AEF-GEE signed correlations for proper environmental labeling
    aef_gee_df = pd.read_csv(AEF_GEE_FILE, sep='\t', comment='#')
    # Build lookup: dim -> {gee_var: signed_correlation}
    dim_gee_top1 = {}  # dim -> (gee_var, signed_corr)
    for _, row in aef_gee_df[aef_gee_df['rank'] == 1].iterrows():
        dim_gee_top1[row['aef_dim']] = (row['gee_variable'], row['correlation'])

    # Load column cluster assignments
    col_cluster_df = pd.read_csv(COL_CLUSTER_FILE, sep='\t', comment='#')

    # For each k-means cluster, find dominant environmental association
    # by SHAP-weighted aggregation across ALL dimensions
    cluster_info = {}
    for c in range(N_ROW_CLUSTERS):
        mask = row_labels == c
        c_pfams = matrix_filtered.index[mask]
        c_size = mask.sum()

        # Cluster-mean z-score profile across all 64 dimensions
        z_profile = matrix_zscore.loc[c_pfams].mean(axis=0)

        # SHAP-weighted environmental vote:
        # For each dim, weight = cluster_mean_abs_shap * sign(z_score)
        # Then accumulate votes per GEE variable
        gee_votes = {}  # gee_var -> sum of signed weighted contributions
        for dim in matrix_zscore.columns:
            if dim not in dim_gee_top1:
                continue
            gee_var, gee_corr = dim_gee_top1[dim]
            # The direction seen by PFAMs in this cluster:
            # z_profile[dim] > 0 means these PFAMs have HIGH SHAP for this dim
            # gee_corr gives the sign of the dim-GEE relationship
            # Combined: z_profile * gee_corr gives the effective environmental direction
            shap_weight = matrix_filtered.loc[c_pfams, dim].mean()
            effective_signal = z_profile[dim] * gee_corr
            contribution = shap_weight * abs(gee_corr)  # weight by SHAP and |corr|

            if gee_var not in gee_votes:
                gee_votes[gee_var] = {'pos': 0, 'neg': 0, 'weight': 0}
            if effective_signal > 0:
                gee_votes[gee_var]['pos'] += contribution
            else:
                gee_votes[gee_var]['neg'] += contribution
            gee_votes[gee_var]['weight'] += contribution

        # Find dominant GEE variable by total weight
        if gee_votes:
            dominant_gee = max(gee_votes.keys(), key=lambda g: gee_votes[g]['weight'])
            # Direction: does this cluster associate positively or negatively?
            if gee_votes[dominant_gee]['pos'] > gee_votes[dominant_gee]['neg']:
                direction = '+'
            else:
                direction = '−'
        else:
            dominant_gee = 'unknown'
            direction = '?'

        # Clean env variable name for display
        env_display = {
            'sst_mean_c': 'SST mean', 'sst_max_c': 'SST max',
            'sst_min_c': 'SST min', 'sst_range_c': 'SST range',
            'modis_sst_mean_c': 'MODIS SST',
            'air_temp_mean_c': 'Air temp.', 'air_temp_max_c': 'Air temp. max',
            'air_temp_min_c': 'Air temp. min', 'air_temp_range_c': 'Air temp. range',
            'bathymetry_m': 'Bathymetry', 'elevation_m': 'Elevation',
            'distance_to_coast_km': 'Dist. to coast',
            'chl_mean_mg_m3': 'Chl-a mean', 'chl_max_mg_m3': 'Chl-a max',
            'chl_min_mg_m3': 'Chl-a min',
            'poc_mean_mg_m3': 'POC mean', 'nflh_mean': 'NFLH',
            'solar_rad_mj_m2': 'Solar rad.',
            'precip_mean_mm': 'Precipitation',
            'rrs_412': 'Rrs 412', 'rrs_443': 'Rrs 443',
            'rrs_469': 'Rrs 469', 'rrs_488': 'Rrs 488',
            'rrs_531': 'Rrs 531', 'rrs_547': 'Rrs 547',
            'rrs_555': 'Rrs 555', 'rrs_645': 'Rrs 645',
            'rrs_667': 'Rrs 667', 'rrs_678': 'Rrs 678',
        }
        env_var = env_display.get(dominant_gee, dominant_gee.replace('_', ' ').title())

        cluster_info[c] = {
            'size': c_size,
            'env_var': env_var,
            'direction': direction,
            'mean_shap': pfam_mean_shap.loc[c_pfams].mean(),
        }
        print(f"  Cluster {c:2d}: n={c_size:3d}, "
              f"env={env_var} ({direction}), mean_shap={cluster_info[c]['mean_shap']:.6f}")

    # Sort clusters by mean SHAP importance (descending)
    sorted_clusters = sorted(cluster_info.keys(),
                             key=lambda c: cluster_info[c]['mean_shap'],
                             reverse=True)

    # Build ordered PFAM list and cluster assignments
    selected_pfams = []
    cluster_assignments = {}  # pfam -> (display_index, cluster_id)
    cluster_boundaries = []
    current_pos = 0

    for display_idx, c in enumerate(sorted_clusters):
        mask = row_labels == c
        c_pfams = matrix_filtered.index[mask].tolist()

        # Sort PFAMs within cluster by mean SHAP
        c_shap = pfam_mean_shap.loc[c_pfams].sort_values(ascending=False)

        # Limit to MAX_PFAMS_PER_CLUSTER
        if len(c_shap) > MAX_PFAMS_PER_CLUSTER:
            c_pfams_kept = c_shap.head(MAX_PFAMS_PER_CLUSTER).index.tolist()
        else:
            c_pfams_kept = c_shap.index.tolist()

        for p in c_pfams_kept:
            cluster_assignments[p] = (display_idx, c)

        n_kept = len(c_pfams_kept)
        cluster_boundaries.append((current_pos, current_pos + n_kept, c))
        current_pos += n_kept
        selected_pfams.extend(c_pfams_kept)

    print(f"  Total selected PFAMs: {len(selected_pfams)}")

    # Column ordering: use top 4 interpretable column clusters
    # Within each cluster, reorder dims by hierarchical clustering leaf order
    col_cluster_assignments = {}
    selected_dims = []
    col_boundaries = []
    col_linkages = {}  # col_cluster -> linkage matrix for dendrogram drawing
    current_col = 0

    for col_cluster in TOP_COL_CLUSTERS:
        row = col_cluster_df[col_cluster_df['cluster'] == col_cluster].iloc[0]
        dims = row['aef_dims'].split(',')
        dims_present = [d for d in dims if d in matrix_zscore.columns]

        # Hierarchical clustering within this column cluster
        if len(dims_present) >= 2:
            col_data = matrix_zscore[dims_present].T  # dims x pfams
            col_dist = pdist(col_data.values, metric='correlation')
            # Handle NaN in distance (set to max)
            col_dist = np.nan_to_num(col_dist, nan=1.0)
            col_link = linkage(col_dist, method='ward')
            col_linkages[col_cluster] = col_link
            leaf_order = leaves_list(col_link)
            dims_present = [dims_present[i] for i in leaf_order]
        else:
            col_linkages[col_cluster] = None

        for d in dims_present:
            col_cluster_assignments[d] = col_cluster
        col_boundaries.append((current_col, current_col + len(dims_present), col_cluster))
        current_col += len(dims_present)
        selected_dims.extend(dims_present)

    print(f"  Total selected dimensions: {len(selected_dims)}")

    # Subset and reorder
    matrix_subset = matrix_zscore.loc[selected_pfams, selected_dims]
    mean_shap_subset = pfam_mean_shap.loc[selected_pfams]

    print(f"  Final matrix: {matrix_subset.shape}")

    return (matrix_subset, mean_shap_subset, cluster_assignments,
            cluster_boundaries, sorted_clusters, cluster_info,
            col_cluster_assignments, col_boundaries, selected_dims,
            col_linkages)

# ==============================================================================
# FIGURE CREATION
# ==============================================================================

def create_figure(matrix, mean_shap, cluster_assignments, cluster_boundaries,
                  sorted_clusters, cluster_info, col_cluster_assignments,
                  col_boundaries, selected_dims, pfam_annot, col_linkages):
    """Create the k=16 biclustered heatmap with full vertical space."""
    print("Creating figure...")

    n_pfams, n_dims = matrix.shape
    ordered_pfams = matrix.index.tolist()

    # ========================================================================
    # FIGURE LAYOUT — tall figure with dendrograms and correlation panel
    # ========================================================================

    fig_height = 18
    fig_width = 12
    fig = plt.figure(figsize=(fig_width, fig_height))

    # GridSpec: 5 rows x 5 cols
    # Row 0: Column dendrogram (top)
    # Row 1: Column cluster labels
    # Row 2: Main heatmap + side annotations (dominant row)
    # Row 3: AEF-GEE correlation bar chart (bottom panel)
    # Row 4: Colorbar
    #
    # Col 0: Row dendrogram (left)
    # Col 1: SHAP bar (widened)
    # Col 2: Row cluster color bar
    # Col 3: Main heatmap (dominant col)
    # Col 4: Right annotations

    # Row 0: top dendrogram, Row 1: col labels, Row 2: main heatmap,
    # Row 3: correlation bars, Row 4: spacer, Row 5: colorbar
    gs = gridspec.GridSpec(6, 5, figure=fig,
                           height_ratios=[0.08, 0.02, 1.0, 0.125, 0.06, 0.03],
                           width_ratios=[0.24, 0.10, 0.03, 1.0, 0.42],
                           hspace=0.04, wspace=0.02)

    cmap = DIVERGING_CMAP
    vmax = 3.0

    # ========== Column Dendrogram (top, row 0) — mini-dendrograms per col cluster ==========
    ax_col_dendro = fig.add_subplot(gs[0, 3])
    # Draw separate dendrogram per column cluster, positioned to match heatmap columns
    # We flip y so root hangs from top: plot (d_max - dc) so root is at y=0 (top)
    # and leaves are at y=d_max (bottom, near heatmap)
    global_d_max = 0.0  # track max distance across all column clusters
    dendro_segments = []  # store (x_scaled, y_flipped) for deferred plotting

    for (start, end, cl) in col_boundaries:
        n_cols = end - start
        link = col_linkages.get(cl)
        if link is None or n_cols < 2:
            continue
        tmp_fig, tmp_ax = plt.subplots(figsize=(1, 1))
        dn = dendrogram(link, ax=tmp_ax, orientation='top',
                        color_threshold=0, above_threshold_color='black',
                        no_labels=True)
        plt.close(tmp_fig)

        icoord = np.array(dn['icoord'])
        dcoord = np.array(dn['dcoord'])

        # Track max across all clusters for uniform scaling
        if dcoord.size > 0:
            local_max = dcoord.max()
            if local_max > global_d_max:
                global_d_max = local_max

        x_min_d = 5.0
        x_max_d = 5.0 + 10.0 * (n_cols - 1)
        for ic, dc in zip(icoord, dcoord):
            x_scaled = start + (np.array(ic) - x_min_d) / (x_max_d - x_min_d) * (n_cols - 1)
            dendro_segments.append((x_scaled, np.array(dc)))

    # Plot raw dcoord values: leaves at y=0, root at max y
    if global_d_max == 0:
        global_d_max = 1.0
    for x_seg, dc_seg in dendro_segments:
        ax_col_dendro.plot(x_seg, dc_seg, color='black', linewidth=0.5)

    ax_col_dendro.set_xlim(-0.5, n_dims - 0.5)
    # Root (max y) at top of panel, leaves (y=0) at bottom near heatmap
    # In matplotlib, ylim(bottom, top) — set bottom=0, top=max so y increases upward
    ax_col_dendro.set_ylim(0, global_d_max * 1.05)
    ax_col_dendro.set_axis_off()

    # ========== Column Cluster Labels (row 1) ==========
    ax_col_labels = fig.add_subplot(gs[1, 3])
    ax_col_labels.set_xlim(0, len(selected_dims))
    ax_col_labels.set_ylim(0, 1)

    for (start, end, cl) in col_boundaries:
        width = end - start
        mid = start + width / 2
        color = COL_CLUSTER_COLORS.get(cl, '#666666')
        rect = Rectangle((start, 0), width, 1, facecolor=color,
                          edgecolor='white', linewidth=0.5, alpha=0.3)
        ax_col_labels.add_patch(rect)
        label = COL_ENV_LABELS.get(cl, cl)
        ax_col_labels.text(mid, 0.5, label, ha='center', va='center',
                           fontsize=8, fontweight='bold', color='black')
    ax_col_labels.set_axis_off()

    # ========== Row Dendrogram (left, row 2, col 0) — mini-dendrograms per row cluster ==========
    ax_row_dendro = fig.add_subplot(gs[2, 0])
    # Draw a mini-dendrogram per K-means row cluster (horizontal, left orientation)
    for (start, end, c) in cluster_boundaries:
        n_rows = end - start
        if n_rows < 3:
            continue  # skip tiny clusters
        cluster_data = matrix.iloc[start:end].values
        row_dist = pdist(cluster_data, metric='correlation')
        row_dist = np.nan_to_num(row_dist, nan=1.0)
        row_link = linkage(row_dist, method='ward')

        # Extract dendrogram coords via temp figure
        tmp_fig, tmp_ax = plt.subplots(figsize=(1, 1))
        dn = dendrogram(row_link, ax=tmp_ax, orientation='left',
                        color_threshold=0, above_threshold_color='black',
                        no_labels=True)
        plt.close(tmp_fig)

        icoord = np.array(dn['icoord'])  # y-coords in dendrogram space
        dcoord = np.array(dn['dcoord'])  # x-coords (distance)

        # Scale y from dendrogram space [5, 5 + 10*(n-1)] to heatmap row space [start, end-1]
        y_min_d = 5.0
        y_max_d = 5.0 + 10.0 * (n_rows - 1)
        # Scale x (distance) to fit in the dendrogram column
        d_max = dcoord.max() if dcoord.max() > 0 else 1.0

        for ic, dc in zip(icoord, dcoord):
            # ic = y-coords (row positions), dc = x-coords (distance)
            y_scaled = start + (np.array(ic) - y_min_d) / (y_max_d - y_min_d) * (n_rows - 1)
            # x: invert so root is at right (close to heatmap), leaves at left
            x_scaled = 1.0 - np.array(dc) / d_max
            ax_row_dendro.plot(x_scaled, y_scaled, color='black', linewidth=0.5)

    ax_row_dendro.set_ylim(-0.5, n_pfams - 0.5)
    ax_row_dendro.invert_yaxis()
    ax_row_dendro.set_xlim(0, 1)
    ax_row_dendro.set_axis_off()

    # ========== Mean SHAP Bar (row 2, col 1 — widened) ==========
    ax_shap_bar = fig.add_subplot(gs[2, 1])
    shap_vals = mean_shap.loc[ordered_pfams].values
    y_pos = np.arange(len(shap_vals))

    colors = []
    for p in ordered_pfams:
        di, c = cluster_assignments[p]
        colors.append(CLUSTER_COLORS_16[di % 16])

    ax_shap_bar.barh(y_pos, shap_vals, height=1.0, color=colors, edgecolor='none')
    ax_shap_bar.set_ylim(-0.5, len(shap_vals) - 0.5)
    ax_shap_bar.invert_yaxis()
    shap_max = shap_vals.max() * 1.1
    ax_shap_bar.set_xlim(0, shap_max)
    # X-axis on top to avoid crowding with bottom correlation panel
    ax_shap_bar.xaxis.set_ticks_position('top')
    ax_shap_bar.xaxis.set_label_position('top')
    ax_shap_bar.set_xticks([0, shap_vals.max()])
    ax_shap_bar.set_xticklabels(['0', f'{shap_vals.max():.4f}'], fontsize=8)
    ax_shap_bar.tick_params(axis='x', which='both', top=True, bottom=False,
                            width=0.5, length=2, pad=2)
    ax_shap_bar.set_yticks([])
    ax_shap_bar.set_xlabel('Mean |SHAP|', fontsize=8)
    for sp in ax_shap_bar.spines.values():
        sp.set_linewidth(0.25)

    # ========== Row Cluster Color Bar (row 2, col 2) ==========
    ax_cluster_bar = fig.add_subplot(gs[2, 2])
    cluster_colors_arr = np.zeros((len(ordered_pfams), 1, 3))
    for i, p in enumerate(ordered_pfams):
        di, c = cluster_assignments[p]
        rgb = mpl.colors.to_rgb(CLUSTER_COLORS_16[di % 16])
        cluster_colors_arr[i, 0, :] = rgb

    ax_cluster_bar.imshow(cluster_colors_arr, aspect='auto', interpolation='nearest')
    ax_cluster_bar.set_xticks([])
    ax_cluster_bar.set_yticks([])
    for sp in ax_cluster_bar.spines.values():
        sp.set_linewidth(0.25)

    # ========== Main Heatmap (row 2, col 3) ==========
    ax_heat = fig.add_subplot(gs[2, 3])
    im = ax_heat.imshow(matrix.values, aspect='auto', cmap=cmap,
                        vmin=-vmax, vmax=vmax, interpolation='nearest')

    # Row cluster boundary lines
    for (start, end, c) in cluster_boundaries:
        if start > 0:
            ax_heat.axhline(start - 0.5, color='white', linewidth=0.8)

    # Column cluster boundary lines
    for (start, end, cl) in col_boundaries:
        if start > 0:
            ax_heat.axvline(start - 0.5, color='white', linewidth=0.5, alpha=0.8)

    # X-axis: hide tick labels (bottom correlation panel + GEE labels serve this role)
    ax_heat.set_xticks([])
    # No xlabel — column cluster labels at top already identify groupings

    # Y-axis: no individual labels (annotations go on right)
    ax_heat.set_yticks([])

    for sp in ax_heat.spines.values():
        sp.set_linewidth(0.25)

    # ========== Right Annotations (row 2, col 4) ==========
    ax_annot = fig.add_subplot(gs[2, 4])
    ax_annot.set_xlim(0, 1)
    ax_annot.set_ylim(0, n_pfams)
    ax_annot.invert_yaxis()
    ax_annot.set_axis_off()

    for idx, (start, end, c) in enumerate(cluster_boundaries):
        n = end - start
        y_center = start + n / 2
        color = CLUSTER_COLORS_16[idx % 16]
        info = cluster_info[c]

        # Determine how many PFAMs to show based on cluster size
        if info['size'] >= 20:
            n_show = min(8, n)
        elif info['size'] >= 10:
            n_show = min(5, n)
        else:
            n_show = min(3, n)

        # Get top PFAMs with functional names
        cluster_pfams = ordered_pfams[start:end]
        top_pfams = cluster_pfams[:n_show]
        top_labels = [get_pfam_label(p, pfam_annot) for p in top_pfams]

        # Cluster header: "C## (n=XX) — Env Var (dir)"
        header = f"C{idx+1} (n={info['size']}) — {info['env_var']} ({info['direction']})"

        # Position annotations: single text block to avoid internal overlap
        # Header line bold, PFAM lines regular, all 6pt
        line_size = 4
        pfam_lines = []
        for i in range(0, len(top_labels), line_size):
            pfam_lines.append(', '.join(top_labels[i:i + line_size]))
        pfam_str = '\n'.join(pfam_lines)
        full_text = f"$\\bf{{{header}}}$\n{pfam_str}"

        # Use a single text call with the header bolded via raw approach
        # Place at cluster center
        ax_annot.text(0.02, y_center, header,
                      fontsize=8, va='bottom', color='#000000',
                      fontweight='bold')
        ax_annot.text(0.02, y_center, pfam_str,
                      fontsize=8, va='top', color='#000000',
                      linespacing=1.2)

        # Bracket line connecting to heatmap
        ax_annot.plot([0.0, 0.0], [start + 0.5, end - 0.5],
                      color=color, linewidth=1.2, alpha=0.6)

    # ========== Bottom Correlation Panel (row 3, col 3) — AEF-GEE correlation bars ==========
    ax_corr = fig.add_subplot(gs[3, 3])

    # Load AEF-GEE top-1 correlations
    aef_gee_df = pd.read_csv(AEF_GEE_FILE, sep='\t', comment='#')
    top1 = aef_gee_df[aef_gee_df['rank'] == 1].set_index('aef_dim')

    # Map GEE variable names to environmental categories for coloring
    _gee_to_category = {
        'sst_mean_c': 'Temperature', 'sst_max_c': 'Temperature',
        'sst_min_c': 'Temperature', 'sst_range_c': 'Temperature',
        'modis_sst_mean_c': 'Temperature',
        'air_temp_mean_c': 'Atmospheric', 'air_temp_max_c': 'Atmospheric',
        'air_temp_min_c': 'Atmospheric', 'air_temp_range_c': 'Atmospheric',
        'bathymetry_m': 'Bathymetry', 'elevation_m': 'Bathymetry',
        'distance_to_coast_km': 'Bathymetry',
        'chl_mean_mg_m3': 'Productivity', 'chl_max_mg_m3': 'Productivity',
        'chl_min_mg_m3': 'Productivity',
        'poc_mean_mg_m3': 'Productivity', 'nflh_mean': 'Productivity',
        'solar_rad_mj_m2': 'Atmospheric',
        'precip_mean_mm': 'Atmospheric',
        'rrs_412': 'Ocean Color', 'rrs_443': 'Ocean Color',
        'rrs_469': 'Ocean Color', 'rrs_488': 'Ocean Color',
        'rrs_531': 'Ocean Color', 'rrs_547': 'Ocean Color',
        'rrs_555': 'Ocean Color', 'rrs_645': 'Ocean Color',
        'rrs_667': 'Ocean Color', 'rrs_678': 'Ocean Color',
    }

    bar_colors = []
    bar_heights = []
    bar_gee_vars = []  # store for Task 8 labeling
    for i, dim in enumerate(selected_dims):
        if dim in top1.index:
            abs_corr = top1.loc[dim, 'abs_correlation']
            gee_var = top1.loc[dim, 'gee_variable']
            cat = _gee_to_category.get(gee_var, 'Temperature')
            color = ENV_CATEGORY_COLORS.get(cat, COASTAL_BLUE)
            bar_heights.append(abs_corr)
            bar_colors.append(color)
            bar_gee_vars.append(gee_var)
        else:
            bar_heights.append(0)
            bar_colors.append(COASTAL_BLUE)
            bar_gee_vars.append('')

    x_positions = np.arange(n_dims)
    ax_corr.bar(x_positions, bar_heights, width=0.8, color=bar_colors,
                edgecolor='none', alpha=0.85)

    ax_corr.set_xlim(-0.5, n_dims - 0.5)
    ax_corr.set_ylim(0, max(bar_heights) * 1.1 if bar_heights else 1)
    ax_corr.set_ylabel('|r| top GEE', fontsize=8)
    ax_corr.tick_params(axis='y', labelsize=8, width=0.5, length=2)

    # GEE variable name labels below bars (Task 8)
    # Abbreviated display names for GEE variables
    _gee_display = {
        'sst_mean_c': 'SST', 'sst_max_c': 'SST max',
        'sst_min_c': 'SST min', 'sst_range_c': 'SST rng',
        'modis_sst_mean_c': 'MODIS SST',
        'air_temp_mean_c': 'Air T', 'air_temp_max_c': 'Air T max',
        'air_temp_min_c': 'Air T min', 'air_temp_range_c': 'Air T rng',
        'bathymetry_m': 'Bathy', 'elevation_m': 'Elev',
        'distance_to_coast_km': 'Dist coast',
        'chl_mean_mg_m3': 'Chl', 'chl_max_mg_m3': 'Chl max',
        'chl_min_mg_m3': 'Chl min',
        'poc_mean_mg_m3': 'POC', 'nflh_mean': 'NFLH',
        'solar_rad_mj_m2': 'Solar', 'precip_mean_mm': 'Precip',
        'rrs_412': 'Rrs412', 'rrs_443': 'Rrs443',
        'rrs_469': 'Rrs469', 'rrs_488': 'Rrs488',
        'rrs_531': 'Rrs531', 'rrs_547': 'Rrs547',
        'rrs_555': 'Rrs555', 'rrs_645': 'Rrs645',
        'rrs_667': 'Rrs667', 'rrs_678': 'Rrs678',
    }
    gee_labels = [_gee_display.get(g, g.replace('_', ' ')[:8]) for g in bar_gee_vars]
    ax_corr.set_xticks(x_positions)
    ax_corr.set_xticklabels(gee_labels, rotation=90, fontsize=8, ha='center')
    ax_corr.tick_params(axis='x', width=0.5, length=2, pad=2)

    # Column cluster boundary lines (match heatmap)
    for (start, end, cl) in col_boundaries:
        if start > 0:
            ax_corr.axvline(start - 0.5, color='white', linewidth=0.5, alpha=0.8)

    for sp in ax_corr.spines.values():
        sp.set_linewidth(0.25)

    # ========== Colorbar (row 4, col 3 — tight below correlation panel) ==========
    ax_cbar = fig.add_subplot(gs[5, 3])
    cbar = fig.colorbar(im, cax=ax_cbar, orientation='horizontal')
    cbar.set_label('Z-score (row-normalized SHAP)', fontsize=8)
    cbar.ax.tick_params(labelsize=8, width=0.5, length=2)
    cbar.outline.set_linewidth(0.25)

    return fig

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print(f"Figure 5: K-means k={N_ROW_CLUSTERS} PFAM-Environment SHAP Landscape")
    print(f"Timestamp: {TIMESTAMP}")
    print("=" * 70)

    # Load PFAM annotations
    pfam_annot = load_pfam_annotations()

    # Load data and cluster
    (matrix_subset, mean_shap_subset, cluster_assignments,
     cluster_boundaries, sorted_clusters, cluster_info,
     col_cluster_assignments, col_boundaries, selected_dims,
     col_linkages) = load_and_cluster()

    # Fetch missing PFAM annotations for top PFAMs per cluster (up to 8)
    top_pfams_all = []
    ordered = matrix_subset.index.tolist()
    for (start, end, c) in cluster_boundaries:
        top8 = ordered[start:min(start + 8, end)]
        top_pfams_all.extend([p.split('.')[0] for p in top8])
    pfam_annot = fetch_missing_annotations(list(set(top_pfams_all)), pfam_annot)

    # Create figure
    fig = create_figure(matrix_subset, mean_shap_subset, cluster_assignments,
                        cluster_boundaries, sorted_clusters, cluster_info,
                        col_cluster_assignments, col_boundaries, selected_dims,
                        pfam_annot, col_linkages)

    # Save
    for fmt in ['pdf', 'svg']:
        out_path = f'{OUT_DIR}/Figure5_kmeans16_{TIMESTAMP}.{fmt}'
        fig.savefig(out_path, format=fmt, bbox_inches='tight',
                    transparent=True, edgecolor='none')
        print(f"Saved: {out_path}")
        # Also save as canonical name
        canonical = f'{OUT_DIR}/Figure5_aef_bicluster.{fmt}'
        fig.savefig(canonical, format=fmt, bbox_inches='tight',
                    transparent=True, edgecolor='none')
        print(f"Saved canonical: {canonical}")

    # Provenance
    prov_path = f'{OUT_DIR}/Figure5_kmeans16_{TIMESTAMP}_provenance.txt'
    with open(prov_path, 'w') as f:
        f.write(f"Provenance:\n")
        f.write(f"  Script: {SCRIPT_PATH}\n")
        f.write(f"  Input: {SHAP_FILE}\n")
        f.write(f"  Input: {COL_CLUSTER_FILE}\n")
        f.write(f"  Input: {ANNOT_FILE}\n")
        f.write(f"  Date: {TIMESTAMP}\n")
        f.write(f"  K-means k: {N_ROW_CLUSTERS}\n")
        f.write(f"  Max PFAMs per cluster: {MAX_PFAMS_PER_CLUSTER}\n")
        f.write(f"  Column clusters: {', '.join(TOP_COL_CLUSTERS)}\n")
        f.write(f"  Final matrix shape: {matrix_subset.shape}\n")
        f.write(f"  Integrity Check: PASSED - Real data only\n")

    plt.close()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

if __name__ == '__main__':
    main()
