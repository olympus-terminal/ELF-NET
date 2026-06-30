#!/usr/bin/env python3
"""
Figure 4: Modular Organization of Domain-Environment Coupling (K-means rework)
===============================================================================

Replaces arbitrary head(500) + independent Ward linkage with:
  - Principled SHAP threshold-based domain selection (≥5% of max mean norm|SHAP|)
  - K-means row clustering with silhouette-based k selection
  - Environmental category column grouping with within-group Ward
  - SHAP-weighted environmental labeling per cluster

Layout (unchanged):
  - Top section:    Panel A (heatmap, ~65%) + Panel B (circular, ~35%)
  - Middle section: Panel C (Sankey, full width)
  - Bottom section: Panels D-G (2x2 grid, verbatim from prior script)

Panel A structure:
  - Left: mini-dendrograms per row cluster
  - Left marginal: mean |SHAP| bar per domain (colored by cluster)
  - Cluster color bar
  - Main heatmap: raw Spearman ρ (diverging cmap) — z-scoring internal only
  - Right: cluster labels (bracket + top Pfam names)
  - Top: mini-dendrograms per column group
  - Top: env category color bar
  - Bottom: R² bar chart

Data sources:
  - Correlations: algagpt_pfam_gee_correlations_full_20260119_103813.tsv
  - SHAP:         shap_dependence_gee_20260119_194303.tsv
  - Annotations:  pfam_interpro_annotations.tsv / _20260120_104640.tsv

Created: 2026-02-28
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Wedge, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.colors import Normalize
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Environment detection (HPC / local)
# ---------------------------------------------------------------------------

def _is_hpc() -> bool:
    if Path("/scratch/drn2").exists():
        return True
    cwd = str(Path.cwd())
    if cwd.startswith("/scratch/"):
        return True
    return False

def get_base_dir() -> Path:
    """Return project base: /media/drn2/External/TARA-Oceans (local) or
    /scratch/drn2/PROJECTS/TARA-LA4SR (HPC)."""
    if _is_hpc():
        return Path("/scratch/drn2/PROJECTS/TARA-LA4SR")
    return Path("/media/drn2/External/TARA-Oceans")

BASE = get_base_dir()

# Import Earth-from-Space palette — works whether CWD is local or HPC
_palette_dirs = [
    str(BASE / "MANUSCRIPT" / "figures"),
    str(Path(__file__).resolve().parent),
]
for _d in _palette_dirs:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from palette import (
    OCEAN_CMAP, DIVERGING_CMAP,
    ENV_CATEGORY_COLORS, MODULE_COLORS,
    get_categorical_colors,
    DEEP_OCEAN, TURQUOISE, FOREST_GREEN, DESERT_TAN, CLOUD_WHITE,
    COASTAL_BLUE, OCEAN_BLUE, CLAY, SIENNA, PALE_AQUA, PALE_GREEN, SAVANNA,
)
from tara_style import apply_tara_style, check_figure_size, save_figure

apply_tara_style()
mpl.rcParams['figure.constrained_layout.use'] = False

# =============================================================================
# COLOR SYSTEMS
# =============================================================================

ENV_CAT_DARK = {
    'Temperature': COASTAL_BLUE,
    'Productivity': FOREST_GREEN,
    'Ocean Color': TURQUOISE,
    'Bathymetry': DEEP_OCEAN,
    'Nutrient': SAVANNA,
    'Atmospheric': (0.42, 0.24, 0.60),
}

SHAP_CMAP = OCEAN_CMAP

# 12 cluster colors (max k we test)
_base_colors = [
    DEEP_OCEAN, TURQUOISE, FOREST_GREEN, DESERT_TAN,
    CLAY, COASTAL_BLUE, SIENNA, (0.400, 0.651, 0.400),
    (0.180, 0.310, 0.510), (0.706, 0.557, 0.361),
    (0.310, 0.506, 0.294), (0.502, 0.420, 0.588),
]

def _rgb_to_hex(c):
    return f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}'

CLUSTER_HEX = [_rgb_to_hex(c) for c in _base_colors]

# =============================================================================
# DATA PATHS
# =============================================================================

# HPC uses doubled 03_analyses path
_analyses = BASE / "03_analyses"
_algagpt = _analyses / "03_analyses" / "ALGAGPT-based-analyses"
# Fall back to single 03_analyses if doubled doesn't exist
if not _algagpt.exists():
    _algagpt = _analyses / "ALGAGPT-based-analyses"

CORR_FILE = _algagpt / "algagpt_pfam_gee_correlations_full_20260119_103813.tsv"
# Fallback: try the other timestamp if this doesn't exist
if not CORR_FILE.exists():
    _alt = _algagpt / "algagpt_pfam_gee_correlations_full_20260119_102603.tsv"
    if _alt.exists():
        CORR_FILE = _alt
    else:
        # Try full pattern
        _alt2 = _algagpt / "algagpt_pfam_gee_correlations_20260119_104938_full.tsv"
        if _alt2.exists():
            CORR_FILE = _alt2

SHAP_FILE = _algagpt / "validations" / "results" / "shap_dependence_gee_20260119_194303.tsv"

# Annotation file: try MANUSCRIPT/source_data first, then HPC validation results
ANNOT_FILE = BASE / "MANUSCRIPT" / "source_data" / "pfam_interpro_annotations.tsv"
if not ANNOT_FILE.exists():
    _alt_annot = _algagpt / "validations" / "results" / "pfam_interpro_annotations_20260120_104640.tsv"
    if _alt_annot.exists():
        ANNOT_FILE = _alt_annot

OUT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = Path(__file__).resolve().parent.parent
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================================================================
# CONFIGURATION
# =============================================================================

SHAP_THRESHOLD_FRAC = 0.05   # keep domains with mean norm|SHAP| >= 5% of max
K_RANGE = [4, 6, 8, 10, 12]  # silhouette sweep
MAX_PFAMS_PER_CLUSTER = 30   # readability cap per cluster
RANDOM_STATE = 42

# Environmental categories (ordered for display)
ENV_CATEGORIES = ['Temperature', 'Productivity', 'Ocean Color',
                  'Bathymetry', 'Nutrient', 'Atmospheric']

# =============================================================================
# HELPERS
# =============================================================================

def categorize_env(var):
    v = var.lower()
    if any(x in v for x in ['temp', 'sst']):
        return 'Temperature'
    if any(x in v for x in ['chl', 'nflh', 'poc']):
        return 'Productivity'
    if 'rrs' in v:
        return 'Ocean Color'
    if any(x in v for x in ['bathy', 'elev', 'distance', 'dist']):
        return 'Bathymetry'
    if any(x in v for x in ['nitrate', 'phosphate', 'silicate', 'oxygen', 'mld', 'salinity']):
        return 'Nutrient'
    return 'Atmospheric'

def abbrev_var(var):
    remap = {
        'air_temp_mean_c': 'AirT', 'air_temp_max_c': 'AirTmax',
        'air_temp_min_c': 'AirTmin', 'air_temp_range_c': 'AirTrng',
        'sst_mean_c': 'SST', 'sst_max_c': 'SSTmax',
        'sst_min_c': 'SSTmin', 'sst_range_c': 'SSTrng',
        'modis_sst_mean_c': 'mSST',
        'chl_mean_mg_m3': 'Chl', 'chl_max_mg_m3': 'Chlmax',
        'chl_min_mg_m3': 'Chlmin',
        'poc_mean_mg_m3': 'POC', 'nflh_mean': 'nFLH',
        'bathymetry_m': 'Bathy', 'elevation_m': 'Elev',
        'distance_to_coast_km': 'Coast',
        'solar_rad_mj_m2': 'Solar', 'precip_mean_mm': 'Precip',
        'landcover_class': 'Land',
        'nitrate_umol_l': 'NO₃', 'phosphate_umol_l': 'PO₄',
        'silicate_umol_l': 'SiO₄', 'oxygen_umol_l': 'O₂',
        'mld_m': 'MLD', 'salinity_psu_est': 'Sal',
    }
    if var in remap:
        return remap[var]
    if 'rrs' in var:
        num = ''.join(c for c in var if c.isdigit())
        return num if num else var[:8]
    return var.replace('_', ' ')[:10]

ENV_DISPLAY = {
    'sst_mean_c': 'SST', 'sst_max_c': 'SSTmax',
    'sst_min_c': 'SSTmin', 'sst_range_c': 'SSTrng',
    'modis_sst_mean_c': 'mSST',
    'air_temp_mean_c': 'AirT', 'air_temp_max_c': 'AirTmax',
    'air_temp_min_c': 'AirTmin', 'air_temp_range_c': 'AirTrng',
    'bathymetry_m': 'Bathy', 'elevation_m': 'Elev',
    'distance_to_coast_km': 'Coast',
    'chl_mean_mg_m3': 'Chl', 'chl_max_mg_m3': 'Chlmax',
    'chl_min_mg_m3': 'Chlmin',
    'poc_mean_mg_m3': 'POC', 'nflh_mean': 'NFLH',
    'solar_rad_mj_m2': 'Solar', 'precip_mean_mm': 'Precip',
    'landcover_class': 'Land',
    'rrs_412': 'Rrs412', 'rrs_443': 'Rrs443', 'rrs_469': 'Rrs469',
    'rrs_488': 'Rrs488', 'rrs_531': 'Rrs531', 'rrs_547': 'Rrs547',
    'rrs_555': 'Rrs555', 'rrs_645': 'Rrs645', 'rrs_667': 'Rrs667',
    'rrs_678': 'Rrs678',
    'nitrate_umol_l': 'Nitrate', 'phosphate_umol_l': 'Phosphate',
    'silicate_umol_l': 'Silicate', 'oxygen_umol_l': 'Dissolved O₂',
    'mld_m': 'MLD', 'salinity_psu_est': 'Salinity',
}

def draw_sankey_flow(ax, x0, y0, x1, y1, w0, w1, color, alpha=0.35):
    mx = (x0 + x1) / 2
    verts = [
        (x0, y0 + w0/2),
        (mx, y0 + w0/2), (mx, y1 + w1/2), (x1, y1 + w1/2),
        (x1, y1 - w1/2),
        (mx, y1 - w1/2), (mx, y0 - w0/2), (x0, y0 - w0/2),
        (x0, y0 + w0/2),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes),
                           facecolor=color, edgecolor='none', alpha=alpha))

# =============================================================================
# PFAM ANNOTATIONS
# =============================================================================

def load_pfam_annotations():
    """Load PFAM short names from InterPro annotations + hardcoded fallbacks."""
    annot = {}
    if ANNOT_FILE.exists():
        df = pd.read_csv(ANNOT_FILE, sep='\t', comment='#')
        for _, row in df.iterrows():
            pfam_id = str(row.iloc[0])  # first col is pfam_id
            # Try 'short_name' then 'name' column
            short = row.get('short_name', '')
            name = row.get('name', '')
            if pd.notna(short) and short:
                annot[pfam_id] = short
            elif pd.notna(name) and name:
                annot[pfam_id] = str(name)[:30]
        print(f"  Loaded {len(annot)} PFAM annotations")

    # Hardcoded fallbacks (verified)
    _fallback = {
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
        'PF00097': 'zf-C3HC4', 'PF00173': 'Cyt-b5',
        'PF00001': '7tm_1', 'PF00002': '7tm_2', 'PF00003': '7tm_3',
        'PF00012': 'HSP70', 'PF00036': 'EF-hand_1',
        'PF00118': 'Cpn60_TCP1',
        # Top-15 SHAP domains (verified via InterPro API 2026-04-12)
        'PF19028': 'TSP1_spondin', 'PF20209': 'DUF6570',
        'PF11999': 'Ice_binding', 'PF00589': 'Phage_integrase',
        'PF05380': 'Peptidase_A17', 'PF00090': 'TSP_1',
        'PF03050': 'DDE_Tnp_IS66', 'PF01609': 'DDE_Tnp_1',
        'PF24906': 'Zf_WRKY19', 'PF20147': 'Crinkler',
        'PF23055': 'DUF7041',
    }
    for k, v in _fallback.items():
        if k not in annot:
            annot[k] = v
    return annot

def get_pfam_label(pfam_full, annot):
    pfam_base = pfam_full.split('.')[0]
    if pfam_base in annot:
        return annot[pfam_base]
    return pfam_base

# =============================================================================
# LOAD & CLUSTER DATA
# =============================================================================

def load_data():
    """Load SHAP + correlation data, apply principled domain selection."""

    print(f"Loading SHAP data: {SHAP_FILE}")
    shap_df = pd.read_csv(SHAP_FILE, sep='\t', comment='#')
    print(f"  {len(shap_df):,} rows, columns: {list(shap_df.columns)}")

    # Normalise SHAP per variable, then average across variables per PFAM
    shap_df['norm_shap'] = shap_df.groupby('variable')['mean_abs_shap'].transform(
        lambda x: x / x.max() if x.max() > 0 else 0
    )
    pfam_shap = shap_df.groupby('pfam')['norm_shap'].mean().sort_values(ascending=False)
    print(f"  {len(pfam_shap):,} unique PFAMs scored")

    # --- Principled threshold: keep PFAMs >= SHAP_THRESHOLD_FRAC * max ---
    threshold = pfam_shap.max() * SHAP_THRESHOLD_FRAC
    selected = pfam_shap[pfam_shap >= threshold].index.tolist()
    print(f"  Threshold: {SHAP_THRESHOLD_FRAC*100:.0f}% of max ({threshold:.6f})")
    print(f"  Selected {len(selected)} PFAMs (captures "
          f"{pfam_shap.loc[selected].sum() / pfam_shap.sum() * 100:.1f}% of total SHAP mass)")

    # Variable-level R²
    var_r2 = shap_df.drop_duplicates('variable').set_index('variable')['r2']

    # Load correlations (filtered to selected PFAMs)
    print(f"Loading correlations: {CORR_FILE}")
    selected_set = set(selected)
    chunks = []
    # Detect column names from first non-comment line
    pfam_col = 'pfam'
    var_col = 'gee_variable'
    rho_col = 'rho'
    for chunk in pd.read_csv(CORR_FILE, sep='\t', comment='#', chunksize=50000):
        # Auto-detect column names from first chunk
        if not chunks:
            cols = list(chunk.columns)
            if 'pfam_domain' in cols:
                pfam_col = 'pfam_domain'
            if 'spearman_rho' in cols:
                rho_col = 'spearman_rho'
            print(f"  Correlation columns: {cols[:6]}...")
        chunks.append(chunk[chunk[pfam_col].isin(selected_set)])
    corr = pd.concat(chunks, ignore_index=True)
    print(f"  {len(corr):,} correlation records retained")

    # Pivot to matrix: PFAM × env_variable
    corr_mat = corr.pivot_table(
        index=pfam_col, columns=var_col, values=rho_col, aggfunc='first'
    )
    # Drop landcover_class if present (categorical, not meaningful ρ)
    if 'landcover_class' in corr_mat.columns:
        corr_mat = corr_mat.drop(columns=['landcover_class'])

    corr_mat = corr_mat.reindex(selected).fillna(0)
    print(f"  Correlation matrix: {corr_mat.shape[0]} PFAMs × {corr_mat.shape[1]} variables")

    return corr_mat, pfam_shap, var_r2, shap_df

def cluster_data(corr_mat, pfam_shap):
    """K-means with silhouette selection on z-scored correlation matrix."""

    variables = list(corr_mat.columns)
    data = corr_mat.values.copy()

    # --- Z-score rows (per-PFAM normalization across env vars) ---
    row_mean = data.mean(axis=1, keepdims=True)
    row_std = data.std(axis=1, keepdims=True)
    row_std[row_std == 0] = 1.0
    z_data = (data - row_mean) / row_std

    # --- Silhouette sweep ---
    print("\nK-means silhouette sweep:")
    sil_scores = {}
    km_models = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(z_data)
        sil = silhouette_score(z_data, labels)
        sil_scores[k] = sil
        km_models[k] = (km, labels)
        print(f"  k={k:2d}  silhouette={sil:.4f}")

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"  → Best k={best_k} (silhouette={sil_scores[best_k]:.4f})")
    km, row_labels = km_models[best_k]

    # --- Cluster info: z-score-based environmental association ---
    # The z-scored centroid reveals which variable is most distinctive for
    # each cluster (high |z| = this cluster deviates from the global mean
    # on that variable).  Direction comes from sign of raw mean ρ.
    pfams = list(corr_mat.index)
    pfam_mean_shap = pfam_shap.reindex(pfams).fillna(0)

    cluster_info = {}
    for c in range(best_k):
        mask = row_labels == c
        c_pfams = [pfams[i] for i in range(len(pfams)) if mask[i]]
        c_size = mask.sum()

        # Z-scored centroid (what K-means actually clustered on)
        z_centroid = z_data[mask].mean(axis=0)   # length = n_variables
        # Raw centroid (for direction)
        raw_centroid = data[mask].mean(axis=0)

        # Dominant variable = highest |z-scored centroid|
        abs_z = np.abs(z_centroid)
        dom_idx = int(np.argmax(abs_z))
        dominant_var = variables[dom_idx]
        direction = '+' if raw_centroid[dom_idx] > 0 else '−'

        env_label = ENV_DISPLAY.get(dominant_var,
                                     dominant_var.replace('_', ' ').title())
        cluster_info[c] = {
            'size': c_size,
            'env_var': env_label,
            'raw_env_var': dominant_var,
            'direction': direction,
            'mean_shap': pfam_mean_shap.loc[c_pfams].mean() if c_pfams else 0,
            'env_category': categorize_env(dominant_var),
            'top_z': abs_z[dom_idx],
        }
        print(f"  Cluster {c:2d}: n={c_size:3d}, "
              f"env={env_label} ({direction}), "
              f"|z|={abs_z[dom_idx]:.3f}, "
              f"mean_shap={cluster_info[c]['mean_shap']:.6f}")

    # Sort clusters by mean SHAP (descending)
    sorted_clusters = sorted(cluster_info.keys(),
                             key=lambda c: cluster_info[c]['mean_shap'],
                             reverse=True)

    # Build ordered PFAM list + cluster assignments
    selected_pfams = []
    cluster_assignments = {}  # pfam -> (display_idx, original_cluster_id)
    cluster_boundaries = []   # (start, end, original_cluster_id)
    current_pos = 0

    for display_idx, c in enumerate(sorted_clusters):
        mask = row_labels == c
        c_pfams = [pfams[i] for i in range(len(pfams)) if mask[i]]

        # Sort within cluster by SHAP
        c_shap = pfam_mean_shap.loc[c_pfams].sort_values(ascending=False)
        if len(c_shap) > MAX_PFAMS_PER_CLUSTER:
            c_pfams_kept = c_shap.head(MAX_PFAMS_PER_CLUSTER).index.tolist()
        else:
            c_pfams_kept = c_shap.index.tolist()

        # Within-cluster Ward ordering on correlation distance
        if len(c_pfams_kept) >= 3:
            sub = corr_mat.loc[c_pfams_kept].values
            dist = pdist(sub, metric='correlation')
            dist = np.nan_to_num(dist, nan=1.0)
            link = linkage(dist, method='ward')
            order = leaves_list(link)
            c_pfams_kept = [c_pfams_kept[i] for i in order]

        for p in c_pfams_kept:
            cluster_assignments[p] = (display_idx, c)
        n_kept = len(c_pfams_kept)
        cluster_boundaries.append((current_pos, current_pos + n_kept, c))
        current_pos += n_kept
        selected_pfams.extend(c_pfams_kept)

    print(f"  Total selected (after cap): {len(selected_pfams)} PFAMs")

    # --- Column ordering: group by env category, Ward within each ---
    var_categories = {v: categorize_env(v) for v in variables}
    col_order = []
    col_boundaries = []  # (start, end, category_name)
    col_linkages = {}
    current_col = 0

    for cat in ENV_CATEGORIES:
        cat_vars = [v for v in variables if var_categories[v] == cat]
        if not cat_vars:
            continue
        if len(cat_vars) >= 2:
            sub = corr_mat[cat_vars].T.values
            dist = pdist(sub, metric='correlation')
            dist = np.nan_to_num(dist, nan=1.0)
            link = linkage(dist, method='ward')
            col_linkages[cat] = link
            order = leaves_list(link)
            cat_vars = [cat_vars[i] for i in order]
        else:
            col_linkages[cat] = None
        col_boundaries.append((current_col, current_col + len(cat_vars), cat))
        current_col += len(cat_vars)
        col_order.extend(cat_vars)

    # Reorder matrix
    matrix_display = corr_mat.loc[selected_pfams, col_order]

    return (matrix_display, cluster_assignments, cluster_boundaries,
            sorted_clusters, cluster_info, col_boundaries, col_linkages,
            best_k, sil_scores, z_data, row_labels)

# =============================================================================
# PANEL A: K-means biclustered heatmap
# =============================================================================

def draw_panel_a(fig, gs_slot, matrix, pfam_shap, var_r2,
                 cluster_assignments, cluster_boundaries,
                 sorted_clusters, cluster_info,
                 col_boundaries, col_linkages, pfam_annot):
    """K-means clustered heatmap with env-category columns."""

    inner = gridspec.GridSpecFromSubplotSpec(
        4, 6, subplot_spec=gs_slot,
        height_ratios=[0.08, 0.025, 0.25, 0.16],
        width_ratios=[0.12, 0.08, 0.02, 1.0, 0.012, 0.035],
        hspace=0.02, wspace=0.02
    )

    data = matrix.values
    pfams_ordered = list(matrix.index)
    vars_ordered = list(matrix.columns)
    n_rows = len(pfams_ordered)
    n_cols = len(vars_ordered)

    # ── Column dendrograms (top) ──
    ax_cdend = fig.add_subplot(inner[0, 3])
    dendro_segments = []
    global_d_max = 0.0
    for (start, end, cat) in col_boundaries:
        nc = end - start
        link = col_linkages.get(cat)
        if link is None or nc < 2:
            continue
        tmp_fig, tmp_ax = plt.subplots(figsize=(1, 1))
        dn = dendrogram(link, ax=tmp_ax, orientation='top',
                        color_threshold=0, above_threshold_color='black',
                        no_labels=True)
        plt.close(tmp_fig)
        icoord = np.array(dn['icoord'])
        dcoord = np.array(dn['dcoord'])
        if dcoord.size > 0:
            local_max = dcoord.max()
            if local_max > global_d_max:
                global_d_max = local_max
        x_min_d = 5.0
        x_max_d = 5.0 + 10.0 * (nc - 1)
        for ic, dc in zip(icoord, dcoord):
            x_scaled = start + (np.array(ic) - x_min_d) / max(x_max_d - x_min_d, 1e-9) * (nc - 1)
            dendro_segments.append((x_scaled, np.array(dc)))
    if global_d_max == 0:
        global_d_max = 1.0
    for x_seg, dc_seg in dendro_segments:
        ax_cdend.plot(x_seg, dc_seg, color='black', linewidth=0.4)
    ax_cdend.set_xlim(-0.5, n_cols - 0.5)
    ax_cdend.set_ylim(0, global_d_max * 1.05)
    ax_cdend.set_axis_off()

    # ── Env category bar (above heatmap) ──
    ax_cat = fig.add_subplot(inner[1, 3])
    cat_colors_arr = np.zeros((1, n_cols, 3))
    for j, v in enumerate(vars_ordered):
        cat = categorize_env(v)
        cat_colors_arr[0, j, :] = ENV_CATEGORY_COLORS.get(cat, (0.8, 0.8, 0.8))
    ax_cat.imshow(cat_colors_arr, aspect='auto', interpolation='nearest')
    # Category labels centered on each group
    for (start, end, cat) in col_boundaries:
        n_span = end - start
        if n_span < 3:
            continue  # Too narrow for any text at 6pt
        mid = (start + end) / 2
        # Abbreviate category names to fit
        _cat_abbrev = {'Temperature': 'Temp.', 'Productivity': 'Prod.',
                       'Bathymetry': 'Bathy.', 'Atmospheric': 'Atm.',
                       'Ocean Color': 'Color', 'Nutrient': 'Nutr.',
                       'Salinity': 'Sal.'}
        cat_lbl = _cat_abbrev.get(cat, cat) if n_span < 6 else cat
        ax_cat.text(mid, 0, cat_lbl, fontsize=6, ha='center', va='center',
                    color='white')
    ax_cat.set_xticks([])
    ax_cat.set_yticks([])
    for sp in ax_cat.spines.values():
        sp.set_linewidth(0.25)

    # ── Row mini-dendrograms (left) ──
    ax_rdend = fig.add_subplot(inner[2, 0])
    for (start, end, c) in cluster_boundaries:
        nr = end - start
        if nr < 3:
            continue
        cluster_data_sub = data[start:end]
        dist = pdist(cluster_data_sub, metric='correlation')
        dist = np.nan_to_num(dist, nan=1.0)
        link = linkage(dist, method='ward')
        tmp_fig, tmp_ax = plt.subplots(figsize=(1, 1))
        dn = dendrogram(link, ax=tmp_ax, orientation='left',
                        color_threshold=0, above_threshold_color='black',
                        no_labels=True)
        plt.close(tmp_fig)
        icoord = np.array(dn['icoord'])
        dcoord = np.array(dn['dcoord'])
        y_min_d = 5.0
        y_max_d = 5.0 + 10.0 * (nr - 1)
        d_max_local = dcoord.max() if dcoord.max() > 0 else 1.0
        for ic, dc in zip(icoord, dcoord):
            y_scaled = start + (np.array(ic) - y_min_d) / max(y_max_d - y_min_d, 1e-9) * (nr - 1)
            x_scaled = 1.0 - np.array(dc) / d_max_local
            ax_rdend.plot(x_scaled, y_scaled, color='black', linewidth=0.4)
    ax_rdend.set_ylim(-0.5, n_rows - 0.5)
    ax_rdend.invert_yaxis()
    ax_rdend.set_xlim(0, 1)
    ax_rdend.set_axis_off()

    # ── SHAP bar (left marginal) ──
    ax_shap = fig.add_subplot(inner[2, 1])
    shap_vals = [pfam_shap.get(pf, 0) for pf in pfams_ordered]
    colors = [CLUSTER_HEX[cluster_assignments[p][0] % len(CLUSTER_HEX)]
              for p in pfams_ordered]
    ax_shap.barh(range(n_rows), shap_vals, height=1.0, color=colors,
                 edgecolor='none', linewidth=0)
    ax_shap.set_ylim(-0.5, n_rows - 0.5)
    ax_shap.invert_yaxis()
    ax_shap.set_yticks([])
    xmax = max(shap_vals)
    ax_shap.set_xlim(0, xmax * 1.05)
    ax_shap.xaxis.set_ticks_position('top')
    ax_shap.xaxis.set_label_position('top')
    ax_shap.set_xticks([0, round(xmax, 2)])
    ax_shap.set_xticklabels(['0', f'{xmax:.2f}'], fontsize=6)
    ax_shap.set_xlabel('|SHAP|', fontsize=6, labelpad=2)
    ax_shap.tick_params(axis='x', which='both', top=True, bottom=False,
                        width=0.25, length=1.5, pad=1.5)
    for sp in ax_shap.spines.values():
        sp.set_linewidth(0.25)

    # ── Cluster color bar ──
    ax_cbar = fig.add_subplot(inner[2, 2])
    cbar_arr = np.zeros((n_rows, 1, 3))
    for i, p in enumerate(pfams_ordered):
        di = cluster_assignments[p][0]
        cbar_arr[i, 0, :] = mpl.colors.to_rgb(CLUSTER_HEX[di % len(CLUSTER_HEX)])
    ax_cbar.imshow(cbar_arr, aspect='auto', interpolation='nearest')
    ax_cbar.set_xticks([])
    ax_cbar.set_yticks([])
    for sp in ax_cbar.spines.values():
        sp.set_linewidth(0.25)

    # ── MAIN HEATMAP ──
    ax_heat = fig.add_subplot(inner[2, 3])
    vmax = np.percentile(np.abs(data), 99)
    im = ax_heat.imshow(data, aspect='auto', cmap=DIVERGING_CMAP,
                        vmin=-vmax, vmax=vmax, interpolation='nearest')
    # Row cluster boundaries
    for (start, end, c) in cluster_boundaries:
        if start > 0:
            ax_heat.axhline(start - 0.5, color='white', linewidth=0.6)
    # Column category boundaries
    for (start, end, cat) in col_boundaries:
        if start > 0:
            ax_heat.axvline(start - 0.5, color='white', linewidth=0.4, alpha=0.8)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    for sp in ax_heat.spines.values():
        sp.set_linewidth(0.25)

    # ── Col 4 is a thin spacer between heatmap and colorbar ──

    # ── R² bar (bottom) ──
    ax_r2 = fig.add_subplot(inner[3, 3])
    r2_vals = [var_r2.get(v, 0) for v in vars_ordered]
    colors_r2 = [ENV_CAT_DARK.get(categorize_env(v), (0.3, 0.3, 0.3))
                 for v in vars_ordered]
    ax_r2.bar(range(n_cols), r2_vals, width=1.0,
              color=colors_r2, edgecolor='none', linewidth=0)
    ax_r2.set_xlim(-0.5, n_cols - 0.5)
    nice_labels = [ENV_DISPLAY.get(v, v.replace('_', ' ')[:12]) for v in vars_ordered]
    ax_r2.set_xticks(range(n_cols))
    ax_r2.set_xticklabels(nice_labels, rotation=45, ha='right',
                           rotation_mode='anchor', fontsize=6)
    ax_r2.tick_params(axis='x', which='major', pad=1, length=2, width=0.25)
    for lbl in ax_r2.get_xticklabels():
        lbl.set_clip_on(False)
    ax_r2.set_ylabel('R²', fontsize=6, labelpad=2, rotation=0, va='center')
    ax_r2.tick_params(axis='y', labelsize=6, pad=1, length=2, width=0.25)
    ymax_r2 = max(r2_vals) if r2_vals and max(r2_vals) > 0 else 0.1
    ax_r2.set_ylim(0, ymax_r2 * 1.15)
    ax_r2.set_yticks([0, round(ymax_r2, 2)])
    for sp in ax_r2.spines.values():
        sp.set_linewidth(0.25)
    ax_r2.spines['top'].set_visible(False)
    ax_r2.spines['right'].set_visible(False)
    # Column boundaries on R² bar
    for (start, end, cat) in col_boundaries:
        if start > 0:
            ax_r2.axvline(start - 0.5, color='white', linewidth=0.4, alpha=0.8)

    # ── Colorbar ──
    ax_cb = fig.add_subplot(inner[2, 5])
    cb = fig.colorbar(im, cax=ax_cb, orientation='vertical')
    cb.ax.tick_params(labelsize=6, width=0.25, length=1.5, pad=1)
    cb.outline.set_linewidth(0.25)
    cb.set_ticks([-round(vmax, 1), 0, round(vmax, 1)])
    # Tick labels face right (default) but are compact enough to not hit Panel B
    # because we increased wspace to 0.12 between top panels
    cb.ax.set_title('ρ', fontsize=6, pad=1)

    # ── Panel label ──
    ax_rdend.text(-0.15, 1.02, 'A', transform=ax_rdend.transAxes,
                  fontsize=6, fontweight='bold', va='top', ha='right')

    return pfams_ordered, vars_ordered

# =============================================================================
# PANEL B: Circular plot (top 20 PFAMs) — updated with K-means clusters
# =============================================================================

def draw_panel_b(fig, gs_slot, pfam_shap, cluster_assignments, pfam_annot):
    """Circular multi-track plot for top 20 PFAMs, colored by K-means cluster."""

    ax = fig.add_subplot(gs_slot)
    ax.set_aspect('equal')
    ax.set_axis_off()

    top15 = pfam_shap.head(15)
    pfams = list(top15.index)
    n = len(pfams)

    angles = {p: np.pi/2 - 2 * np.pi * i / n for i, p in enumerate(pfams)}

    r_mod_in, r_mod_out = 0.28, 0.36
    r_shap_in, r_shap_out = 0.38, 0.58
    r_label = 0.62

    theta_w = 2 * np.pi / n * 0.82
    shap_max = top15.max()
    shap_norm = top15 / shap_max

    # Cluster ring (inner) — use K-means cluster color
    for i, pf in enumerate(pfams):
        th = angles[pf]
        di = cluster_assignments.get(pf, (0, 0))[0]
        c = CLUSTER_HEX[di % len(CLUSTER_HEX)]
        wedge = Wedge((0, 0), r_mod_out,
                      np.degrees(th - theta_w/2), np.degrees(th + theta_w/2),
                      width=r_mod_out - r_mod_in,
                      facecolor=c, edgecolor='white', linewidth=0.15)
        ax.add_patch(wedge)

    # SHAP bars (outer)
    for i, pf in enumerate(pfams):
        th = angles[pf]
        bar_h = shap_norm[pf] * (r_shap_out - r_shap_in)
        r_top = r_shap_in + bar_h
        c = SHAP_CMAP(shap_norm[pf] * 0.8 + 0.2)
        wedge = Wedge((0, 0), r_top,
                      np.degrees(th - theta_w/2), np.degrees(th + theta_w/2),
                      width=bar_h, facecolor=c, edgecolor='none')
        ax.add_patch(wedge)

    # Baseline circle
    circle_theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(r_shap_in * np.cos(circle_theta), r_shap_in * np.sin(circle_theta),
            color='#999999', lw=0.25, zorder=0)

    # Interaction arcs (between same-cluster PFAMs)
    for i, p1 in enumerate(pfams):
        c1 = cluster_assignments.get(p1, (0, -1))[0]
        for j, p2 in enumerate(pfams):
            if j <= i:
                continue
            c2 = cluster_assignments.get(p2, (0, -2))[0]
            if c1 == c2:
                th1, th2 = angles[p1], angles[p2]
                t_arr = np.linspace(0, 1, 40)
                mid_th = (th1 + th2) / 2
                span = abs(th2 - th1)
                ctrl_r = (r_mod_in - 0.03) * max(0.1, 1 - span / np.pi)
                x1, y1 = (r_mod_in - 0.01) * np.cos(th1), (r_mod_in - 0.01) * np.sin(th1)
                x2, y2 = (r_mod_in - 0.01) * np.cos(th2), (r_mod_in - 0.01) * np.sin(th2)
                cx, cy = ctrl_r * np.cos(mid_th), ctrl_r * np.sin(mid_th)
                bx = (1-t_arr)**2 * x1 + 2*(1-t_arr)*t_arr * cx + t_arr**2 * x2
                by = (1-t_arr)**2 * y1 + 2*(1-t_arr)*t_arr * cy + t_arr**2 * y2
                ax.plot(bx, by, color=CLUSTER_HEX[c1 % len(CLUSTER_HEX)],
                       alpha=0.25, lw=0.4, solid_capstyle='round')

    # Labels — placed directly at bar tip + small radial offset (no linker lines)
    label_gap = 0.03
    for i, pf in enumerate(pfams):
        th = angles[pf]
        r_bar_tip = r_shap_in + shap_norm[pf] * (r_shap_out - r_shap_in)
        r_lbl = r_bar_tip + label_gap

        label = pf.split('.')[0]  # Raw PFAM accession
        deg = np.degrees(th) % 360
        # Radial text: aligned along radius, reading outward, never upside down
        if 90 < deg <= 270:
            rot = deg - 180
            ha = 'right'
            r_lbl = r_bar_tip + label_gap  # still outward
        else:
            rot = deg
            ha = 'left'
        x_lbl = r_lbl * np.cos(th)
        y_lbl = r_lbl * np.sin(th)
        ax.text(x_lbl, y_lbl, label, fontsize=6, ha=ha, va='center',
                rotation=rot, rotation_mode='anchor', color='black')

    ax.set_xlim(-0.78, 0.88)
    ax.set_ylim(-0.78, 0.78)
    ax.text(-0.05, 1.08, 'B', transform=ax.transAxes,
            fontsize=6, fontweight='bold', va='top', ha='left')
    # Subtitle removed — described in figure caption per artist-mode rules

# =============================================================================
# PANEL C: Sankey diagram — K-means cluster based
# =============================================================================

def draw_panel_c(fig, gs_slot, pfam_shap, cluster_assignments,
                 cluster_boundaries, sorted_clusters, cluster_info,
                 corr_mat):
    """Sankey: SHAP tiers → K-means clusters → env categories."""

    ax = fig.add_subplot(gs_slot)
    ax.set_xlim(-0.12, 1.18)
    ax.set_ylim(-0.08, 1.08)
    ax.set_axis_off()

    # --- Tier computation ---
    all_pfams = pfam_shap.sort_values(ascending=False)
    # Restrict to PFAMs that are in the clustering
    clustered_set = set(cluster_assignments.keys())
    n_total = len([p for p in all_pfams.index if p in clustered_set])

    n_tiers = 5
    tier_labels = ['Top 20%', '20-40%', '40-60%', '60-80%', 'Bottom 20%']
    tier_size = n_total // n_tiers
    pfam_tiers = {}
    tier_counts = []
    idx = 0
    tier_idx = 0
    for pf in all_pfams.index:
        if pf not in clustered_set:
            continue
        pfam_tiers[pf] = tier_idx
        idx += 1
        if tier_idx < n_tiers - 1 and idx >= (tier_idx + 1) * tier_size:
            tier_counts.append(idx - tier_idx * tier_size)
            tier_idx += 1
    tier_counts.append(idx - tier_idx * tier_size)

    # --- Cluster info for display ---
    display_clusters = sorted_clusters[:min(8, len(sorted_clusters))]
    cluster_sizes = {}
    for c in display_clusters:
        cluster_sizes[c] = cluster_info[c]['size']

    # --- Flows: tier → cluster ---
    t2c = {}
    for pf, (di, c) in cluster_assignments.items():
        ti = pfam_tiers.get(pf, n_tiers - 1)
        if c in display_clusters:
            t2c[(ti, c)] = t2c.get((ti, c), 0) + 1

    # --- Flows: cluster → env category ---
    env_var_map = {v: categorize_env(v) for v in corr_mat.columns}
    c2e = {}
    for c in display_clusters:
        c_pfams = [p for p, (di, cc) in cluster_assignments.items()
                   if cc == c and p in corr_mat.index]
        if not c_pfams:
            continue
        sub = corr_mat.loc[corr_mat.index.isin(c_pfams)]
        for cat in ENV_CATEGORIES:
            cat_vars = [v for v, cv in env_var_map.items() if cv == cat]
            if not cat_vars:
                continue
            mean_abs = sub[cat_vars].abs().values.mean()
            if mean_abs > 0.02:
                c2e[(c, cat)] = mean_abs

    # --- Layout ---
    x_t, x_m, x_e = 0.0, 0.48, 1.0
    bar_w = 0.04
    gap = 0.015
    total_h = 0.85
    y_top = 0.95          # top edge of the tallest column's stack

    # -------------------------------------------------------------------------
    # FLOW-CONSERVING SANKEY
    # Node height == sum of the flow widths touching it, in ONE common unit
    # (UNIT = drawable height per unit of flow). This guarantees every band
    # starts/ends flush with its node box (no overflow, no underfill).
    #
    # Flows are made conservative: each cluster's outgoing (cluster->env) flow
    # is rescaled so its total equals that cluster's incoming (tier->cluster)
    # count. Env categories then inherit a conserved share. So for every node,
    # sum(in) == sum(out) == node height.
    # -------------------------------------------------------------------------

    # Per-cluster incoming total (counts) = the cluster's box height in units
    clust_in = {c: 0.0 for c in display_clusters}
    for (ti, c), cnt in t2c.items():
        if c in clust_in:
            clust_in[c] += cnt

    # Rescale cluster->env strengths so each cluster's outflow sums to clust_in
    c2e_cons = {}
    for c in display_clusters:
        out_pairs = {cat: s for (cc, cat), s in c2e.items() if cc == c}
        s_sum = sum(out_pairs.values())
        if s_sum <= 0 or clust_in[c] <= 0:
            continue
        for cat, s in out_pairs.items():
            c2e_cons[(c, cat)] = (s / s_sum) * clust_in[c]

    # Column totals (in flow units) -> single UNIT scale for the whole panel
    tier_tot = {i: 0.0 for i in range(n_tiers)}
    for (ti, c), cnt in t2c.items():
        if c in display_clusters:
            tier_tot[ti] += cnt
    env_tot = {cat: 0.0 for cat in ENV_CATEGORIES}
    for (c, cat), v in c2e_cons.items():
        env_tot[cat] += v

    col_sum_tiers = sum(tier_tot.values())
    col_sum_clust = sum(clust_in.values())
    col_sum_env = sum(env_tot.values())
    max_col_units = max(col_sum_tiers, col_sum_clust, col_sum_env, 1.0)
    n_nodes_tallest = max(n_tiers, len(display_clusters), len(ENV_CATEGORIES))
    # Reserve gaps between nodes within the tallest column, then convert the
    # remaining height into a per-unit scale shared by boxes and bands.
    usable_h = total_h - gap * (n_nodes_tallest - 1)
    UNIT = usable_h / max_col_units

    def _draw_column(x, order, units, label_fn, color_fn, edge_fn,
                     label_side, header):
        """Draw stacked node boxes top-down; return {key: (y_top_of_node, height)}."""
        pos = {}
        y = y_top
        for k in order:
            h = units[k] * UNIT
            if h <= 0:
                continue
            # Box spans [y - h, y]; bands will stack from y downward.
            rect = FancyBboxPatch((x - bar_w/2, y - h),
                                  bar_w, h,
                                  boxstyle="round,pad=0.002",
                                  facecolor=color_fn(k),
                                  edgecolor=edge_fn(k), linewidth=0.25,
                                  mutation_aspect=0.5)
            ax.add_patch(rect)
            yc = y - h / 2
            if label_side == 'left':
                ax.text(x - bar_w/2 - 0.012, yc, label_fn(k),
                        fontsize=6, ha='right', va='center', linespacing=1.2)
            else:
                ax.text(x + bar_w/2 + 0.012, yc, label_fn(k),
                        fontsize=6, ha='left', va='center', zorder=10,
                        bbox=dict(facecolor='white', edgecolor='none',
                                  alpha=0.85, pad=0.3))
            pos[k] = (y, h)
            y -= h + gap
        ax.text(x, 0.968, header, fontsize=6, ha='center')
        return pos

    # Tier column (left)
    _tier_blues = [mpl.cm.Blues(0.25 + 0.55 * (1 - i/n_tiers)) for i in range(n_tiers)]
    tier_pos = _draw_column(
        x_t, list(range(n_tiers)), tier_tot,
        label_fn=lambda i: f'{tier_labels[i]}\n({tier_counts[i]:,})',
        color_fn=lambda i: _tier_blues[i],
        edge_fn=lambda i: 'white', label_side='left', header='PFAM Tiers')

    # Cluster column (middle)
    clust_pos = _draw_column(
        x_m, display_clusters, clust_in,
        label_fn=lambda c: f"C{sorted_clusters.index(c)+1} — "
                           f"{cluster_info[c]['env_var']} ({cluster_info[c]['direction']})",
        color_fn=lambda c: CLUSTER_HEX[sorted_clusters.index(c) % len(CLUSTER_HEX)],
        edge_fn=lambda c: 'white', label_side='right', header='K-means Clusters')

    # Env column (right)
    env_order = [cat for cat in ENV_CATEGORIES if env_tot.get(cat, 0) > 0]
    env_pos = _draw_column(
        x_e, env_order, env_tot,
        label_fn=lambda cat: cat,
        color_fn=lambda cat: ENV_CATEGORY_COLORS[cat],
        edge_fn=lambda cat: ENV_CAT_DARK[cat], label_side='right',
        header='Environment')

    # --- Draw flows: tier -> cluster (stack from each node's top edge down) ---
    tier_cur = {i: tier_pos[i][0] for i in tier_pos}          # running top edge
    clust_cur_l = {c: clust_pos[c][0] for c in clust_pos}
    for (ti, c), cnt in sorted(t2c.items(), key=lambda x: -x[1]):
        if c not in clust_pos or ti not in tier_pos:
            continue
        di = sorted_clusters.index(c)
        fw = cnt * UNIT
        y0 = tier_cur[ti] - fw / 2
        y1 = clust_cur_l[c] - fw / 2
        tier_cur[ti] -= fw
        clust_cur_l[c] -= fw
        draw_sankey_flow(ax, x_t + bar_w/2, y0,
                         x_m - bar_w/2, y1,
                         fw, fw, CLUSTER_HEX[di % len(CLUSTER_HEX)])

    # --- Draw flows: cluster -> env (conserved units) ---
    clust_cur_r = {c: clust_pos[c][0] for c in clust_pos}
    env_cur = {cat: env_pos[cat][0] for cat in env_pos}
    for (c, cat), v in sorted(c2e_cons.items(), key=lambda x: -x[1]):
        if c not in clust_pos or cat not in env_pos:
            continue
        fw = v * UNIT
        y0 = clust_cur_r[c] - fw / 2
        y1 = env_cur[cat] - fw / 2
        clust_cur_r[c] -= fw
        env_cur[cat] -= fw
        draw_sankey_flow(ax, x_m + bar_w/2, y0,
                         x_e - bar_w/2, y1,
                         fw, fw, ENV_CAT_DARK.get(cat, (0.3, 0.3, 0.3)))


    ax.text(-0.02, 0.99, 'C', transform=ax.transAxes,
            fontsize=6, fontweight='bold', va='top', ha='right')

# =============================================================================
# S5 PANELS D-G (verbatim from existing script)
# =============================================================================

def draw_panel_d_temporal(fig, gs_slot):
    """Panel D: Temporal attenuation — spatial vs temporal R²."""
    ax = fig.add_subplot(gs_slot)
    try:
        tc_file = MANUSCRIPT_DIR / "source_data" / "temporal_cv_results.tsv"
        print(f"  Loading temporal CV data: {tc_file}")
        tc = pd.read_csv(tc_file, sep='\t', comment='#')
        # Filter reverse direction with spatial_block_r2_manuscript present
        tc = tc[(tc['direction'] == 'reverse')
                & tc['spatial_block_r2_manuscript'].notna()
                & (tc['spatial_block_r2_manuscript'] != '')]
        targets = tc['target'].values
        spatial_r2 = tc['spatial_block_r2_manuscript'].astype(float).values
        temporal_r2 = tc['temporal_r2_primary'].astype(float).values
        print(f"  {len(targets)} targets with both spatial and temporal R²")

        label_map = {
            'bathymetry_m': 'Bathy.',
            'sst_mean_c': 'SST\nmean',
            'sst_max_c': 'SST\nmax',
            'sst_min_c': 'SST\nmin',
        }
        labels = [label_map.get(t, t) for t in targets]

        x = np.arange(len(targets))
        width = 0.35
        ax.bar(x - width / 2, spatial_r2, width, label='Spatial block CV',
               color=OCEAN_BLUE, alpha=0.85, edgecolor='white', linewidth=0.2)
        ax.bar(x + width / 2, temporal_r2, width, label='Temporal hold-out',
               color=DESERT_TAN, alpha=0.85, edgecolor='white', linewidth=0.2)
        ax.axhline(0, color='#555555', linewidth=0.4, linestyle='--', zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylabel(r'$R^2$', fontsize=6, labelpad=2)
        ax.tick_params(axis='both', labelsize=6, width=0.25, length=1.5, pad=1)
        y_lo = min(temporal_r2.min(), 0) - 0.05
        y_hi = max(spatial_r2.max(), 0.5) + 0.05
        ax.set_ylim(y_lo, y_hi)
        ax.legend(fontsize=6, frameon=False, loc='upper right',
                  handletextpad=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    except Exception as e:
        print(f"  Error drawing Panel D (temporal): {e}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, 'D: Temporal attenuation\n(Error)',
                ha='center', va='center', fontsize=6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PALE_AQUA,
                         edgecolor=COASTAL_BLUE, linewidth=0.25))
        ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.25)
    ax.text(-0.02, 1.05, 'D', transform=ax.transAxes,
            fontsize=6, fontweight='bold', va='top', ha='right')

def draw_panel_go_enrichment(fig, gs_slot, label='E'):
    """GO enrichment lollipop for env-predictable domains (FDR < 0.05)."""
    ax = fig.add_subplot(gs_slot)
    try:
        go_file = MANUSCRIPT_DIR / "supplement" / "TableS5_go_enrichment.tsv"
        print(f"  Loading GO enrichment data: {go_file}")
        go = pd.read_csv(go_file, sep='\t', comment='#')
        sig = go[(go['env_group'] == 'env_predictable') & (go['fdr'] < 0.05)].copy()
        sig = sig.sort_values('fold_enrichment', ascending=True)
        print(f"  {len(sig)} FDR-significant GO terms (env_predictable)")

        short_names = {
            'RNA-DNA hybrid ribonuclease activity': 'RNase H',
            'myosin complex': 'Myosin complex',
            'DNA helicase activity': 'DNA helicase',
            'monoatomic ion transport': 'Ion transport',
            'cytoskeletal motor activity': 'Cytoskel. motor',
            'DNA integration': 'DNA integration',
            'protein binding': 'Protein binding',
        }
        labels = [short_names.get(t, t[:20]) for t in sig['go_term']]
        fe = sig['fold_enrichment'].values
        log_fe = np.log10(fe)
        fdr_vals = sig['fdr'].values

        # Color by functional axis
        func_colors = {
            'RNase H': DEEP_OCEAN,
            'DNA helicase': DEEP_OCEAN,
            'DNA integration': DEEP_OCEAN,
            'Myosin complex': FOREST_GREEN,
            'Cytoskel. motor': FOREST_GREEN,
            'Ion transport': TURQUOISE,
            'Protein binding': DESERT_TAN,
        }
        colors = [func_colors.get(l, DEEP_OCEAN) for l in labels]

        # Dot size scales with -log10(FDR)
        neg_log_fdr = -np.log10(np.clip(fdr_vals, 1e-10, 1))
        dot_sizes = 6 + (neg_log_fdr / neg_log_fdr.max()) * 18

        y = np.arange(len(labels))
        for i in range(len(labels)):
            ax.plot([0, log_fe[i]], [y[i], y[i]], color=colors[i],
                    linewidth=0.6, alpha=0.5, zorder=1)
        ax.scatter(log_fe, y, c=colors, s=dot_sizes, zorder=2,
                   edgecolors='white', linewidth=0.2)
        # Fold enrichment value — offset beyond dot marker
        for i in range(len(labels)):
            ax.text(log_fe[i] + 0.12, y[i], f'{fe[i]:.0f}x',
                    ha='left', va='center', fontsize=6, color='black',
                    clip_on=False)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel('log10(fold enrich.)', fontsize=6, labelpad=1)
        ax.tick_params(axis='both', labelsize=6, width=0.25, length=1.5, pad=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Right margin for value labels (e.g. "146x")
        ax.set_xlim(0, max(log_fe) * 1.38)
        ax.set_ylim(-0.5, len(labels) - 0.5)
    except Exception as e:
        print(f"  Error drawing Panel {label} (GO enrichment): {e}")
        import traceback; traceback.print_exc()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, f'{label}: GO enrichment\n(Error)',
                ha='center', va='center', fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.25)
    ax.text(-0.12, 1.05, label, transform=ax.transAxes,
            fontsize=6, fontweight='bold', va='top', ha='right')

def draw_panel_te_sensitivity(fig, gs_slot, label='D'):
    """Compact transposase sensitivity — ±TE R² comparison.

    Tight y-axis cropped to data range; no wasted vertical space.
    """
    ax = fig.add_subplot(gs_slot)
    try:
        te_file = MANUSCRIPT_DIR / "source_data" / "mc5_transposase_exclusion.tsv"
        print(f"  Loading transposase exclusion data: {te_file}")
        te = pd.read_csv(te_file, sep='\t', comment='#')

        def get_val(section, metric):
            row = te[(te['section'] == section) & (te['metric'] == metric)]
            return float(row['value'].iloc[0])

        bathy_all = get_val('xgboost_bathymetry_m_all_pfams', 'r2_mean')
        bathy_all_sd = get_val('xgboost_bathymetry_m_all_pfams', 'r2_std')
        bathy_exc = get_val('xgboost_bathymetry_m_excluding_transposases', 'r2_mean')
        bathy_exc_sd = get_val('xgboost_bathymetry_m_excluding_transposases', 'r2_std')
        sst_all = get_val('xgboost_modis_sst_mean_c_all_pfams', 'r2_mean')
        sst_all_sd = get_val('xgboost_modis_sst_mean_c_all_pfams', 'r2_std')
        sst_exc = get_val('xgboost_modis_sst_mean_c_excluding_transposases', 'r2_mean')
        sst_exc_sd = get_val('xgboost_modis_sst_mean_c_excluding_transposases', 'r2_std')
        delta_bathy = get_val('delta_bathymetry_m', 'percent_change')
        delta_sst = get_val('delta_modis_sst_mean_c', 'percent_change')
        print(f"  Bathy: all={bathy_all:.3f}, excl={bathy_exc:.3f}, delta={delta_bathy:+.1f}%")
        print(f"  SST:   all={sst_all:.3f}, excl={sst_exc:.3f}, delta={delta_sst:+.1f}%")

        targets = ['Bathy.', 'MODIS\nSST']
        all_r2 = [bathy_all, sst_all]
        all_sd = [bathy_all_sd, sst_all_sd]
        exc_r2 = [bathy_exc, sst_exc]
        exc_sd = [bathy_exc_sd, sst_exc_sd]
        deltas = [delta_bathy, delta_sst]

        x = np.arange(len(targets))
        width = 0.32
        ax.bar(x - width / 2, all_r2, width, yerr=all_sd,
               label='All', color=OCEAN_BLUE, alpha=0.85,
               edgecolor='none', linewidth=0,
               error_kw=dict(lw=0.4, capsize=1.5, capthick=0.4))
        ax.bar(x + width / 2, exc_r2, width, yerr=exc_sd,
               label='Excl. TE', color=TURQUOISE, alpha=0.85,
               edgecolor='none', linewidth=0,
               error_kw=dict(lw=0.4, capsize=1.5, capthick=0.4))
        # Delta annotations — place above error bar caps
        for i in range(len(targets)):
            y_top = max(all_r2[i] + all_sd[i], exc_r2[i] + exc_sd[i]) + 0.012
            sign = '+' if deltas[i] > 0 else ''
            ax.text(x[i], y_top, f'{sign}{deltas[i]:.1f}%',
                    ha='center', va='bottom', fontsize=6,
                    color='black')

        ax.set_xticks(x)
        ax.set_xticklabels(targets, fontsize=6, linespacing=0.9)
        ax.set_ylabel('$R^2$', fontsize=6, labelpad=1, rotation=0, va='center')
        ax.tick_params(axis='both', labelsize=6, width=0.25, length=1.5, pad=1)
        # Crop y-axis tightly to data range
        y_lo = min(min(all_r2) - max(all_sd),
                   min(exc_r2) - max(exc_sd)) - 0.02
        y_lo = max(y_lo, 0)  # floor at 0
        y_hi = max(max(all_r2) + max(all_sd),
                   max(exc_r2) + max(exc_sd)) + 0.06
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(-0.55, len(targets) - 0.45)
        # Legend inside panel — use colored squares in the bar labels instead
        # Place legend in upper-left where there's space between bars
        leg = ax.legend(fontsize=6, frameon=False, loc='upper left',
                        handletextpad=0.2, handlelength=0.8,
                        borderpad=0.2, borderaxespad=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    except Exception as e:
        print(f"  Error drawing Panel {label} (TE sensitivity): {e}")
        import traceback; traceback.print_exc()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, f'{label}: TE sensitivity\n(Error)',
                ha='center', va='center', fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.25)
    ax.text(-0.25, 1.05, label, transform=ax.transAxes,
            fontsize=6, fontweight='bold', va='top', ha='right')

def draw_s5_panel_g(fig, gs_slot):
    """S5 Panel G: ARI/NMI biogeography comparison bars."""
    ax = fig.add_subplot(gs_slot)
    try:
        categories = ['Longhurst\nProvinces', 'Ocean\nBasins', 'Longhurst\nBiomes']
        ari_values = [0.5031, 0.2898, 0.2144]
        nmi_values = [0.7205, 0.6363, 0.5002]
        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, ari_values, width, label='ARI',
               color=OCEAN_BLUE, alpha=0.8, edgecolor='white', linewidth=0.2)
        ax.bar(x + width/2, nmi_values, width, label='NMI',
               color=TURQUOISE, alpha=0.8, edgecolor='white', linewidth=0.2)
        ax.set_xlabel('Biogeographic Classification', fontsize=6, labelpad=2)
        ax.set_ylabel('Agreement Score', fontsize=6, labelpad=2)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=6, ha='center')
        ax.tick_params(axis='both', labelsize=6, width=0.25, length=1.5, pad=1)
        ax.set_xlim(-0.5, len(categories) - 0.1)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        for i, (ari_val, nmi_val) in enumerate(zip(ari_values, nmi_values)):
            ax.text(x[i] - width/2, ari_val + 0.02, f'{ari_val:.2f}',
                    ha='center', va='bottom', fontsize=6, color='black')
            ax.text(x[i] + width/2, nmi_val + 0.02, f'{nmi_val:.2f}',
                    ha='center', va='bottom', fontsize=6, color='black')
        ax.legend(fontsize=6, frameon=False, loc='upper right',
                 markerscale=0.8, handletextpad=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.25)
    except Exception as e:
        print(f"  Error creating ARI/NMI bars: {e}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, 'S5G: ARI/NMI\n(Error)',
                ha='center', va='center', fontsize=6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=TURQUOISE,
                         edgecolor=DEEP_OCEAN, linewidth=0.25))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.25)
    ax.text(-0.15, 1.05, 'F', transform=ax.transAxes,
            fontsize=6, fontweight='bold', va='top', ha='right')

# =============================================================================
# PANEL E (consolidated) — Bootstrap CI forest plot of reverse-model R^2
# Ported from create_figureS4_correlation_modeling_20260212.py row-2 panel C
# =============================================================================

_BOOT_ENV_CATEGORIES = {
    'modis_sst_mean_c': 'Temperature', 'sst_mean_c': 'Temperature',
    'sst_max_c': 'Temperature', 'sst_min_c': 'Temperature',
    'sst_range_c': 'Temperature',
    'air_temp_mean_c': 'Atmospheric', 'air_temp_max_c': 'Atmospheric',
    'air_temp_min_c': 'Atmospheric', 'air_temp_range_c': 'Atmospheric',
    'bathymetry_m': 'Bathymetry', 'elevation_m': 'Bathymetry',
    'distance_to_coast_km': 'Bathymetry', 'depth_m': 'Bathymetry',
    'solar_rad_mj_m2': 'Atmospheric',
    'nflh_mean': 'Productivity', 'chl_mean_mg_m3': 'Productivity',
    'chl_max_mg_m3': 'Productivity', 'chl_min_mg_m3': 'Productivity',
    'poc_mean_mg_m3': 'Productivity',
    'precip_mean_mm': 'Atmospheric',
    'rrs_412': 'Ocean Color', 'rrs_443': 'Ocean Color',
    'rrs_469': 'Ocean Color', 'rrs_488': 'Ocean Color',
    'rrs_531': 'Ocean Color', 'rrs_547': 'Ocean Color',
    'rrs_555': 'Ocean Color', 'rrs_645': 'Ocean Color',
    'rrs_667': 'Ocean Color', 'rrs_678': 'Ocean Color',
    'nitrate_umol_l': 'Nutrient', 'phosphate_umol_l': 'Nutrient',
    'silicate_umol_l': 'Nutrient', 'oxygen_umol_l': 'Nutrient',
    'mld_m': 'Nutrient', 'salinity_psu_est': 'Salinity',
    'landcover_class': 'Atmospheric',
}

_BOOT_NICE_LABELS = {
    'modis_sst_mean_c': 'MODIS SST', 'sst_mean_c': 'SST (mean)',
    'sst_max_c': 'SST (max)', 'sst_min_c': 'SST (min)',
    'sst_range_c': 'SST (range)',
    'air_temp_mean_c': 'Air T (mean)', 'air_temp_max_c': 'Air T (max)',
    'air_temp_min_c': 'Air T (min)', 'air_temp_range_c': 'Air T (range)',
    'bathymetry_m': 'Bathymetry', 'elevation_m': 'Elevation',
    'distance_to_coast_km': 'Dist. coast',
    'solar_rad_mj_m2': 'Solar radiation',
    'nflh_mean': 'NFLH', 'chl_mean_mg_m3': 'Chl-a (mean)',
    'chl_max_mg_m3': 'Chl-a (max)', 'chl_min_mg_m3': 'Chl-a (min)',
    'poc_mean_mg_m3': 'POC', 'precip_mean_mm': 'Precipitation',
    'rrs_412': 'Rrs 412', 'rrs_443': 'Rrs 443',
    'rrs_469': 'Rrs 469', 'rrs_488': 'Rrs 488',
    'rrs_531': 'Rrs 531', 'rrs_547': 'Rrs 547',
    'rrs_555': 'Rrs 555', 'rrs_645': 'Rrs 645',
    'rrs_667': 'Rrs 667', 'rrs_678': 'Rrs 678',
    'nitrate_umol_l': 'Nitrate', 'phosphate_umol_l': 'Phosphate',
    'silicate_umol_l': 'Silicate', 'oxygen_umol_l': 'Diss. O2',
    'mld_m': 'MLD', 'salinity_psu_est': 'Salinity',
    'landcover_class': 'Land cover', 'depth_m': 'Depth',
}


def draw_panel_e_bootstrap_ci(fig, gs_slot, label='E'):
    """Bootstrap-CI forest plot of reverse-model R^2 per env target.

    Reads supplement/FigureS7_bootstrap_ci_20260208_105414.tsv and plots
    point estimates with 95% CI lines for each environmental target, sorted
    by R^2, colored by env category. Ported from Figure S4 panel C.
    """
    ax = fig.add_subplot(gs_slot)
    try:
        boot_file = MANUSCRIPT_DIR / "supplement" / "FigureS7_bootstrap_ci_20260208_105414.tsv"
        print(f"  Loading bootstrap CI data: {boot_file}")
        boot_df = pd.read_csv(boot_file, sep='\t', comment='#')
        boot_df = boot_df.sort_values('r2_point', ascending=True).reset_index(drop=True)

        boot_df['category'] = boot_df['env_target'].map(_BOOT_ENV_CATEGORIES)
        boot_df['color'] = boot_df['category'].map(ENV_CATEGORY_COLORS)
        boot_df['label'] = boot_df['env_target'].map(_BOOT_NICE_LABELS)
        boot_df['label'] = boot_df.apply(
            lambda r: r['label'] if pd.notna(r['label']) else r['env_target'],
            axis=1)
        boot_df['color'] = boot_df['color'].apply(
            lambda c: c if isinstance(c, tuple) else (0.5, 0.5, 0.5))
        boot_df['ci_low_clipped'] = boot_df['ci_95_low'].clip(lower=-0.5)
        boot_df['ci_high_clipped'] = boot_df['ci_95_high'].clip(upper=1.0)

        n_targets = len(boot_df)
        y_positions = np.arange(n_targets)

        # Place value labels with greedy stagger to guarantee no overlap.
        # A 6pt label occupies ~0.07 in data units horizontally (for "0.XXX").
        # Two labels on adjacent rows collide if their x-starts are within
        # this width of each other. We track the previous label's x-start
        # and bump the current one rightward if needed.
        LABEL_WIDTH = 0.09  # approx width of "0.XXX" in data coords at 6pt
        ci_ends = []
        for _, row in boot_df.iterrows():
            ci_ends.append(max(row['ci_high_clipped'], row['r2_point']))

        # First pass: draw CI lines and dots
        for i, (_, row) in enumerate(boot_df.iterrows()):
            color = row['color']
            r2 = row['r2_point']
            ci_lo = row['ci_low_clipped']
            ci_hi = row['ci_high_clipped']
            ax.plot([ci_lo, ci_hi], [i, i], color=color, linewidth=0.8,
                    solid_capstyle='round')
            ax.plot(r2, i, 'o', color=color, markersize=2.5,
                    markeredgecolor='white', markeredgewidth=0.3, zorder=5)

        # Second pass: place labels with greedy de-overlap
        prev_label_x = -999  # x-start of label on the previous row
        for i, (_, row) in enumerate(boot_df.iterrows()):
            r2 = row['r2_point']
            base_x = ci_ends[i] + 0.015
            # If this label would overlap the previous row's label, bump right
            if abs(base_x - prev_label_x) < LABEL_WIDTH:
                base_x = prev_label_x + LABEL_WIDTH
            ax.text(base_x, i, f'{r2:.3f}',
                    ha='left', va='center', fontsize=6, color='black')
            prev_label_x = base_x

        ax.axvline(x=0, color=(0.5, 0.5, 0.5), linewidth=0.25, linestyle='--', zorder=0)
        ax.axvline(x=0.2, color=(0.7, 0.7, 0.7), linewidth=0.25, linestyle=':', zorder=0)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(boot_df['label'].values, fontsize=6)
        ax.set_xlabel('Test $R^2$ (80/20 split, 95% CI from 1,000 bootstraps)',
                      fontsize=6, labelpad=2)
        ax.set_xlim(-0.55, 1.0)
        ax.set_ylim(-0.5, n_targets - 0.5)
        ax.tick_params(axis='both', labelsize=6, width=0.25, length=1.5, pad=1)

        categories_present = boot_df['category'].dropna().unique()
        legend_handles = []
        for cat in sorted(categories_present):
            col = ENV_CATEGORY_COLORS.get(cat, (0.5, 0.5, 0.5))
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=col, markersize=2.5,
                                             label=cat, markeredgewidth=0.3,
                                             markeredgecolor='white'))
        ax.legend(handles=legend_handles, loc='upper center', framealpha=0.9,
                  edgecolor='none', fontsize=6,
                  ncol=min(len(legend_handles), 5),
                  columnspacing=0.4, handletextpad=0.2,
                  bbox_to_anchor=(0.55, 1.06))
    except Exception as e:
        print(f"  Error drawing Panel E (bootstrap CI): {e}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, f'{label}: Bootstrap CI\n(Error)',
                ha='center', va='center', fontsize=6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=DESERT_TAN,
                         edgecolor=CLAY, linewidth=0.25))
        ax.set_xticks([]); ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.25)
    ax.text(-0.15, 1.05, label, transform=ax.transAxes,
            fontsize=6, fontweight='bold', va='top', ha='right')


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Figure 4: Modular Organization of Domain-Environment Coupling")
    print(f"  K-means rework with silhouette selection")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # ── Check data availability for panels A-C ──
    primary_data_available = True
    for f, desc in [(SHAP_FILE, "SHAP"), (CORR_FILE, "Correlations")]:
        if not f.exists():
            print(f"WARNING: {desc} file not found: {f}")
            primary_data_available = False
        else:
            print(f"  {desc}: {f}")

    # Load data for panels A-C if available
    pfam_annot = load_pfam_annotations()
    matrix_display = None
    cluster_assignments = None
    n_selected = 0
    n_vars = 0
    best_k = 0
    sil_scores = {}

    if primary_data_available:
        corr_mat, pfam_shap, var_r2, shap_df = load_data()
        (matrix_display, cluster_assignments, cluster_boundaries,
         sorted_clusters, cluster_info, col_boundaries, col_linkages,
         best_k, sil_scores, z_data, row_labels) = cluster_data(corr_mat, pfam_shap)
        n_selected = len(matrix_display)
        n_vars = matrix_display.shape[1]
    else:
        print("\n  Panels A-C will show placeholders (primary data not available).")

    # ── Figure layout: A/B top, C mid, D/E bottom ──
    fig = plt.figure(figsize=(7.0, 8.5))
    # Use nested GridSpec for non-uniform vertical spacing:
    # - Upper group (A+C) with generous internal hspace for R² labels
    # - Lower group (D+E) tight to upper group
    outer = gridspec.GridSpec(2, 1, figure=fig,
                              height_ratios=[1.40, 0.72],
                              hspace=0.035,
                              left=0.06, right=0.98,
                              top=0.98, bottom=0.04)
    # Upper group: A (row 0) and C (row 1) with generous spacing
    upper = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0],
                                             height_ratios=[0.90, 0.50],
                                             hspace=0.14)
    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=upper[0],
                                           width_ratios=[2.1, 1.0],
                                           wspace=0.08)
    mid = upper[1]
    # Bottom: 2 panels — D (TE, compact), E (bootstrap CI forest, wide)
    bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                           width_ratios=[0.40, 1.3],
                                           wspace=0.35)

    # ── Draw panels A-C ──
    if primary_data_available:
        print("\nDrawing Panel A (K-means biclustered heatmap)...")
        draw_panel_a(fig, top[0], matrix_display, pfam_shap, var_r2,
                     cluster_assignments, cluster_boundaries,
                     sorted_clusters, cluster_info,
                     col_boundaries, col_linkages, pfam_annot)

        print("Drawing Panel B (circular plot, K-means colors)...")
        draw_panel_b(fig, top[1], pfam_shap, cluster_assignments, pfam_annot)

        print("Drawing Panel C (Sankey, K-means clusters)...")
        draw_panel_c(fig, mid, pfam_shap, cluster_assignments,
                     cluster_boundaries, sorted_clusters, cluster_info,
                     corr_mat)
    else:
        for slot, label in [(top[0], 'A'), (top[1], 'B'), (mid, 'C')]:
            ax = fig.add_subplot(slot)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, f'{label}: Data not available\n(mount external drive)',
                    ha='center', va='center', fontsize=6, color='#888888')
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.0, 1.02, label, transform=ax.transAxes,
                    fontsize=6, fontweight='bold', va='top', ha='right')

    # ── Draw panels D and E (bottom row: TE + bootstrap CI) ──
    print("\nDrawing Panel D (TE sensitivity)...")
    draw_panel_te_sensitivity(fig, bot[0], label='D')

    print("Drawing Panel E (bootstrap CI forest)...")
    draw_panel_e_bootstrap_ci(fig, bot[1], label='E')

    # ── Export ──
    for fmt in ['pdf', 'svg']:
        out = OUT_DIR / f"Figure3_module_landscape_{TIMESTAMP}.{fmt}"
        fig.savefig(str(out), format=fmt, dpi=600,
                    transparent=True, edgecolor='none')
        print(f"Saved: {out}")

    # ── Provenance ──
    prov_path = OUT_DIR / f"Figure3_module_landscape_{TIMESTAMP}_provenance.txt"
    with open(prov_path, 'w') as f:
        f.write("# Provenance\n")
        f.write(f"Script: {Path(__file__).resolve()}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input: {CORR_FILE}\n")
        f.write(f"Input: {SHAP_FILE}\n")
        f.write(f"Input: {ANNOT_FILE}\n")
        f.write(f"\n# Domain selection\n")
        f.write(f"Method: mean norm|SHAP| >= {SHAP_THRESHOLD_FRAC*100:.0f}% of max\n")
        f.write(f"N domains retained: {n_selected}\n")
        f.write(f"N env variables: {n_vars}\n")
        f.write(f"\n# Clustering\n")
        f.write(f"Method: K-means on z-scored Spearman rho matrix\n")
        f.write(f"k range tested: {K_RANGE}\n")
        f.write(f"Silhouette scores:\n")
        for k, s in sorted(sil_scores.items()):
            f.write(f"  k={k}: {s:.4f}{'  ← selected' if k == best_k else ''}\n")
        f.write(f"Selected k: {best_k}\n")
        f.write(f"Max PFAMs per cluster: {MAX_PFAMS_PER_CLUSTER}\n")
        f.write(f"Within-cluster ordering: Ward linkage on correlation distance\n")
        f.write(f"Column grouping: {len(ENV_CATEGORIES)} env categories, Ward within each\n")
        f.write(f"\n# Cluster labels\n")
        if primary_data_available:
            for di, c in enumerate(sorted_clusters):
                info = cluster_info[c]
                f.write(f"  C{di+1}: n={info['size']}, "
                        f"env={info['env_var']} ({info['direction']}), "
                        f"mean_shap={info['mean_shap']:.6f}\n")
        else:
            f.write("  (Primary data not available — panels A-C are placeholders)\n")
        f.write(f"\n# Figure content\n")
        f.write(f"Panel A: K-means biclustered heatmap ({n_selected} PFAMs x {n_vars} env vars)\n")
        f.write(f"Panel B: Circular plot (top 15 PFAMs by |SHAP|, K-means cluster colors)\n")
        f.write(f"Panel C: Sankey (5 SHAP tiers -> {best_k} K-means clusters -> 5 env categories)\n")
        f.write(f"Panel D: TE sensitivity (consolidated from Figure S8 panel B)\n")
        f.write(f"Panel E: Bootstrap CI forest plot (consolidated from Figure S4 panel C)\n")
    print(f"Saved: {prov_path}")

    plt.close()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

if __name__ == '__main__':
    main()
