#!/usr/bin/env python3
"""
Ralph25 Task 4: Out-of-sample shrinkage visualization

Split-half cross-validation of CCA canonical correlations:
  1. Load CCA input data (env_matrix, pfam_matrix) and ocean basin assignments
  2. For each of 100 iterations:
     a. Stratified split by ocean basin into two halves (50/50)
     b. Fit PCA(100) + CCA(10) on half A
     c. Project half B through the fitted PCA + CCA
     d. Compute canonical correlations on the held-out half B
  3. Compare observed (full-sample) CCs with cross-validated distribution
  4. Create figure: observed CC bars + violin/box of CV distribution
  5. Report shrinkage per CC as (observed - median_CV) / observed × 100

Provenance:
  Input:  /media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data/*.npy
  Input:  /media/drn2/External/TARA-Oceans/03_analyses/WorldModelApp/data/ocean_basin_assignments.tsv
  Output: source_data/cca_shrinkage_cv_*.tsv
  Output: figures/FigureS9_cca_shrinkage_*.pdf/svg
  Date:   2026-02-12
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# ============================================================================
# Data integrity enforcement
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
MANUSCRIPT_DIR = SCRIPT_DIR.parent
BASE_DIR = Path("/media/drn2/External/TARA-Oceans")
CCA_DATA_DIR = BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "env_pfam_manifold" / "data"
BASIN_FILE = BASE_DIR / "03_analyses" / "WorldModelApp" / "data" / "ocean_basin_assignments.tsv"

sys.path.insert(0, str(BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "env_pfam_manifold"))
from DataIntegrityGuard import enforce_data_integrity, validate_input_source, create_provenance_header
enforce_data_integrity()

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# CCA parameters (must match original pipeline)
# ============================================================================
N_CCA_COMPONENTS = 10
N_PFAM_PCS = 100
N_CV_ITERATIONS = 100
RANDOM_SEED = 42

# ============================================================================
# Matplotlib configuration (FIGURE_PROTOCOL.md)
# ============================================================================
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['axes.linewidth'] = 0.25
mpl.rcParams['xtick.major.width'] = 0.25
mpl.rcParams['ytick.major.width'] = 0.25
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['axes.labelpad'] = 1
mpl.rcParams['xtick.major.pad'] = 1
mpl.rcParams['ytick.major.pad'] = 1

sys.path.insert(0, str(SCRIPT_DIR))
from palette import (
    OCEAN_CMAP, DEEP_OCEAN, COASTAL_BLUE, OCEAN_BLUE, TURQUOISE,
    DESERT_TAN, CLAY, SIENNA, CLOUD_WHITE, EARTH_CATEGORICAL
)

# ============================================================================
# Load data
# ============================================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

input_files = {
    'pfam_matrix': CCA_DATA_DIR / "pfam_matrix_20260122_101559.npy",
    'env_matrix': CCA_DATA_DIR / "env_matrix_20260122_101559.npy",
    'sample_ids': CCA_DATA_DIR / "sample_ids_20260122_101559.npy",
    'env_columns': CCA_DATA_DIR / "env_columns_20260122_101559.txt",
    'basin_assignments': BASIN_FILE,
}

for name, path in input_files.items():
    validate_input_source(path)
    print(f"  Validated: {name} ({path})")

pfam_matrix = np.load(input_files['pfam_matrix'])
env_matrix = np.load(input_files['env_matrix'])
sample_ids = np.load(input_files['sample_ids'], allow_pickle=True)

with open(input_files['env_columns']) as f:
    env_cols = [line.strip() for line in f]

print(f"\n  PFAM matrix: {pfam_matrix.shape}")
print(f"  Env matrix: {env_matrix.shape}")
print(f"  Samples: {len(sample_ids)}")

# ============================================================================
# Load ocean basin assignments and match to CCA samples
# ============================================================================
print("\n" + "=" * 70)
print("LOADING OCEAN BASIN ASSIGNMENTS")
print("=" * 70)

basin_df = pd.read_csv(BASIN_FILE, sep='\t', comment='#')
print(f"  Basin assignments loaded: {len(basin_df)} samples")
print(f"  Basins: {basin_df['ocean_basin'].value_counts().to_dict()}")

# Match sample IDs to basin assignments
sample_id_list = list(sample_ids)
basin_lookup = dict(zip(basin_df['assembly_id'], basin_df['ocean_basin']))

# Assign basins to CCA samples
sample_basins = []
for sid in sample_id_list:
    basin = basin_lookup.get(sid, 'unknown')
    sample_basins.append(basin)

sample_basins = np.array(sample_basins)
basin_counts = pd.Series(sample_basins).value_counts()
print(f"\n  CCA sample basin distribution:")
for basin, count in basin_counts.items():
    print(f"    {basin}: {count}")

n_matched = np.sum(sample_basins != 'unknown')
print(f"\n  Matched: {n_matched}/{len(sample_ids)} ({n_matched/len(sample_ids)*100:.1f}%)")

# ============================================================================
# Step 1: Compute observed (full-sample) CCA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: OBSERVED (FULL-SAMPLE) CCA")
print("=" * 70)

# Handle NaN in env matrix
valid_cols = ~np.any(np.isnan(env_matrix), axis=0)
env_clean = env_matrix[:, valid_cols]
print(f"  Env variables after NaN removal: {env_clean.shape[1]}")

# PCA reduction
n_pca = min(N_PFAM_PCS, min(pfam_matrix.shape) - 1)
pca_full = PCA(n_components=n_pca, random_state=RANDOM_SEED)
pfam_pca_full = pca_full.fit_transform(pfam_matrix)
print(f"  PCA variance explained: {pca_full.explained_variance_ratio_.sum()*100:.1f}%")

# CCA
n_cca = min(N_CCA_COMPONENTS, env_clean.shape[1], pfam_pca_full.shape[1])
cca_full = CCA(n_components=n_cca)
env_cca_full, pfam_cca_full = cca_full.fit_transform(env_clean, pfam_pca_full)

# Observed canonical correlations
observed_ccs = []
for i in range(n_cca):
    corr = np.corrcoef(env_cca_full[:, i], pfam_cca_full[:, i])[0, 1]
    observed_ccs.append(corr)
    print(f"  CC{i+1}: r = {corr:.4f}")

observed_ccs = np.array(observed_ccs)

# ============================================================================
# Step 2: Split-half cross-validation (stratified by ocean basin)
# ============================================================================
print("\n" + "=" * 70)
print(f"STEP 2: SPLIT-HALF CROSS-VALIDATION ({N_CV_ITERATIONS} iterations)")
print("=" * 70)

rng = np.random.RandomState(RANDOM_SEED)

# Store CV canonical correlations
cv_ccs = np.zeros((N_CV_ITERATIONS, n_cca))

# Use StratifiedShuffleSplit for basin-proportional splits
splitter = StratifiedShuffleSplit(
    n_splits=N_CV_ITERATIONS,
    test_size=0.5,
    random_state=RANDOM_SEED
)

for iteration, (train_idx, test_idx) in enumerate(splitter.split(env_clean, sample_basins)):
    if iteration % 20 == 0:
        print(f"  Iteration {iteration+1}/{N_CV_ITERATIONS}...")

    # Split data
    env_train = env_clean[train_idx]
    env_test = env_clean[test_idx]
    pfam_train = pfam_matrix[train_idx]
    pfam_test = pfam_matrix[test_idx]

    # Fit PCA on training half
    pca_cv = PCA(n_components=n_pca, random_state=RANDOM_SEED)
    pfam_pca_train = pca_cv.fit_transform(pfam_train)

    # Project test half through training PCA
    pfam_pca_test = pca_cv.transform(pfam_test)

    # Fit CCA on training half
    cca_cv = CCA(n_components=n_cca)
    cca_cv.fit(env_train, pfam_pca_train)

    # Project test half through fitted CCA
    env_cca_test, pfam_cca_test = cca_cv.transform(env_test, pfam_pca_test)

    # Compute canonical correlations on held-out test half
    for i in range(n_cca):
        corr = np.corrcoef(env_cca_test[:, i], pfam_cca_test[:, i])[0, 1]
        cv_ccs[iteration, i] = corr

print(f"\n  Cross-validation complete.")

# ============================================================================
# Step 3: Compute shrinkage statistics
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: SHRINKAGE STATISTICS")
print("=" * 70)

shrinkage_results = []
for i in range(n_cca):
    obs = observed_ccs[i]
    cv_median = np.median(cv_ccs[:, i])
    cv_mean = np.mean(cv_ccs[:, i])
    cv_std = np.std(cv_ccs[:, i])
    cv_q05 = np.percentile(cv_ccs[:, i], 5)
    cv_q25 = np.percentile(cv_ccs[:, i], 25)
    cv_q75 = np.percentile(cv_ccs[:, i], 75)
    cv_q95 = np.percentile(cv_ccs[:, i], 95)
    shrinkage_pct = (obs - cv_median) / obs * 100

    print(f"  CC{i+1}: observed={obs:.4f}, CV_median={cv_median:.4f}, "
          f"CV_IQR=[{cv_q25:.4f}, {cv_q75:.4f}], shrinkage={shrinkage_pct:.1f}%")

    shrinkage_results.append({
        'canonical_component': f'CC{i+1}',
        'observed_correlation': obs,
        'cv_median': cv_median,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'cv_q05': cv_q05,
        'cv_q25': cv_q25,
        'cv_q75': cv_q75,
        'cv_q95': cv_q95,
        'shrinkage_pct': shrinkage_pct,
        'n_cv_iterations': N_CV_ITERATIONS,
    })

shrinkage_df = pd.DataFrame(shrinkage_results)

# ============================================================================
# Step 4: Save source data
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: SAVING SOURCE DATA")
print("=" * 70)

# Save summary TSV
summary_path = MANUSCRIPT_DIR / "source_data" / f"cca_shrinkage_cv_{TIMESTAMP}.tsv"
provenance = create_provenance_header(
    script_path=__file__,
    input_path=str(CCA_DATA_DIR),
    output_description="CCA split-half cross-validation shrinkage analysis (100 iterations, stratified by ocean basin)"
)

with open(summary_path, 'w') as f:
    for line in provenance.strip().split('\n'):
        f.write(line + '\n')
    shrinkage_df.to_csv(f, sep='\t', index=False)

print(f"  Summary saved: {summary_path}")

# Save full CV distribution for reproducibility
cv_dist_path = MANUSCRIPT_DIR / "source_data" / f"cca_shrinkage_cv_distribution_{TIMESTAMP}.tsv"
cv_dist_df = pd.DataFrame(
    cv_ccs,
    columns=[f'CC{i+1}' for i in range(n_cca)]
)
cv_dist_df.insert(0, 'iteration', range(1, N_CV_ITERATIONS + 1))

with open(cv_dist_path, 'w') as f:
    for line in provenance.strip().split('\n'):
        f.write(line + '\n')
    cv_dist_df.to_csv(f, sep='\t', index=False)

print(f"  CV distribution saved: {cv_dist_path}")

# ============================================================================
# Step 5: Create figure
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: CREATING SHRINKAGE VISUALIZATION")
print("=" * 70)

fig, ax = plt.subplots(figsize=(3.5, 2.5))

cc_positions = np.arange(1, n_cca + 1)

# Draw observed bars (background)
bar_color = COASTAL_BLUE
cv_color = DEEP_OCEAN

bars = ax.bar(
    cc_positions, observed_ccs,
    width=0.6, color=bar_color, alpha=0.35,
    edgecolor=bar_color, linewidth=0.25,
    label='Observed (full sample)', zorder=2
)

# Overlay violin plots of CV distribution
parts = ax.violinplot(
    [cv_ccs[:, i] for i in range(n_cca)],
    positions=cc_positions,
    widths=0.5,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)

# Style violins
for pc in parts['bodies']:
    pc.set_facecolor(cv_color)
    pc.set_edgecolor(cv_color)
    pc.set_alpha(0.5)
    pc.set_linewidth(0.25)

# Add box-like percentile markers
for i in range(n_cca):
    pos = cc_positions[i]
    q25 = np.percentile(cv_ccs[:, i], 25)
    q75 = np.percentile(cv_ccs[:, i], 75)
    median = np.median(cv_ccs[:, i])
    q05 = np.percentile(cv_ccs[:, i], 5)
    q95 = np.percentile(cv_ccs[:, i], 95)

    # IQR box
    ax.vlines(pos, q25, q75, color='white', linewidth=1.5, zorder=4)
    ax.vlines(pos, q25, q75, color=cv_color, linewidth=0.75, zorder=4)

    # 5-95% whiskers
    ax.vlines(pos, q05, q25, color=cv_color, linewidth=0.25, zorder=4)
    ax.vlines(pos, q75, q95, color=cv_color, linewidth=0.25, zorder=4)

    # Median marker
    ax.scatter(pos, median, color='white', s=8, zorder=5, edgecolors=cv_color, linewidths=0.25)

# Add shrinkage percentage labels above each bar
for i in range(n_cca):
    shrink = shrinkage_df.iloc[i]['shrinkage_pct']
    y_pos = max(observed_ccs[i], np.percentile(cv_ccs[:, i], 95)) + 0.015
    ax.text(cc_positions[i], y_pos, f'{shrink:.0f}%',
            ha='center', va='bottom', fontsize=5, color=SIENNA)

# Formatting
ax.set_xlabel('Canonical Component')
ax.set_ylabel('Canonical Correlation')
ax.set_xticks(cc_positions)
ax.set_xticklabels([f'CC{i+1}' for i in range(n_cca)])
ax.set_ylim(0, 1.0)
ax.set_xlim(0.3, n_cca + 0.7)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=bar_color, alpha=0.35, edgecolor=bar_color,
          linewidth=0.25, label='Observed'),
    Patch(facecolor=cv_color, alpha=0.5, edgecolor=cv_color,
          linewidth=0.25, label=f'CV (n={N_CV_ITERATIONS})'),
]
ax.legend(handles=legend_elements, loc='upper right', frameon=False,
          handlelength=1, handletextpad=0.3)

# Add annotation for shrinkage label
ax.text(0.98, 0.82, 'shrinkage %', transform=ax.transAxes,
        ha='right', va='top', fontsize=5, fontstyle='italic', color=SIENNA)

plt.tight_layout()

# Save figure
fig_base = f"FigureS9_cca_shrinkage_{TIMESTAMP}"
for fmt in ['pdf', 'svg']:
    fig_path = SCRIPT_DIR / f"{fig_base}.{fmt}"
    fig.savefig(fig_path, format=fmt, bbox_inches='tight',
                transparent=True, edgecolor='none')
    print(f"  Saved: {fig_path}")

plt.close()

# ============================================================================
# Step 6: Print manuscript-ready summary
# ============================================================================
print("\n" + "=" * 70)
print("MANUSCRIPT-READY SUMMARY")
print("=" * 70)

cc1_shrink = shrinkage_df.iloc[0]['shrinkage_pct']
cc1_cv_median = shrinkage_df.iloc[0]['cv_median']
cc1_obs = shrinkage_df.iloc[0]['observed_correlation']

mean_shrinkage = shrinkage_df['shrinkage_pct'].mean()
max_shrinkage = shrinkage_df['shrinkage_pct'].max()
max_shrinkage_cc = shrinkage_df.loc[shrinkage_df['shrinkage_pct'].idxmax(), 'canonical_component']

# Check if any CV correlations are negative (sign of severe overfitting)
any_negative = np.any(cv_ccs < 0)
n_negative_total = np.sum(cv_ccs < 0)

print(f"\n  CC1: observed r={cc1_obs:.4f}, CV median r={cc1_cv_median:.4f}, shrinkage={cc1_shrink:.1f}%")
print(f"  Mean shrinkage across all CCs: {mean_shrinkage:.1f}%")
print(f"  Maximum shrinkage: {max_shrinkage:.1f}% ({max_shrinkage_cc})")
print(f"  Any negative CV correlations: {any_negative} (total: {n_negative_total}/{N_CV_ITERATIONS * n_cca})")
print(f"  Method: stratified split-half by ocean basin, {N_CV_ITERATIONS} iterations")
print(f"  N samples: {len(sample_ids)}, split into ~{len(sample_ids)//2} per half")

# How many CCs remain significant (CV median > 0.3)?
sig_ccs = shrinkage_df[shrinkage_df['cv_median'] > 0.3]
print(f"\n  CCs with CV median > 0.3: {len(sig_ccs)}/{n_cca}")
for _, row in sig_ccs.iterrows():
    print(f"    {row['canonical_component']}: CV median={row['cv_median']:.4f}")

print("\n" + "=" * 70)
print(f"TASK 4 COMPLETE — {datetime.now().isoformat()}")
print(f"  Summary: {summary_path}")
print(f"  Distribution: {cv_dist_path}")
print(f"  Figure: {SCRIPT_DIR / fig_base}.pdf")
print("=" * 70)
