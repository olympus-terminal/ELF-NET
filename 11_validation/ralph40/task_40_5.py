#!/usr/bin/env python3
"""
Ralph40 Task 40.5: CCA PCA component count sensitivity analysis.

Reviewer concern (R3.1): "Sensitivity to this threshold (e.g., 50, 150, 200
components) should be tested."

Action: Re-run CCA with 50, 100 (baseline), 150, 200 PCA components.
Report CC1-CC3 for each. Report variance retained by PCA at each level.

Provenance:
  Input:  /media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data/
  Output: source_data/ralph40/cca_pca_sensitivity.tsv
  Date:   2026-04-08
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# ============================================================================
# Data integrity enforcement
# ============================================================================
BASE_DIR = Path("/media/drn2/External/TARA-Oceans")
CCA_DATA_DIR = BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "env_pfam_manifold" / "data"
MANUSCRIPT_DIR = Path("/media/drn2/External/TARA-Oceans/MANUSCRIPT/.wt/a3")

sys.path.insert(0, str(BASE_DIR / "03_analyses" / "ALGAGPT-based-analyses" / "env_pfam_manifold"))
from DataIntegrityGuard import enforce_data_integrity, validate_input_source, create_provenance_header
enforce_data_integrity()

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# PCA component counts to test
# ============================================================================
PCA_COUNTS = [50, 100, 150, 200]
N_CCA_COMPONENTS = 10  # Must match original pipeline

# ============================================================================
# Load data
# ============================================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

input_files = {
    'pfam_matrix': CCA_DATA_DIR / "pfam_matrix_20260122_101559.npy",
    'env_matrix': CCA_DATA_DIR / "env_matrix_20260122_101559.npy",
    'pfam_columns': CCA_DATA_DIR / "pfam_columns_20260122_101559.txt",
    'env_columns': CCA_DATA_DIR / "env_columns_20260122_101559.txt",
    'sample_ids': CCA_DATA_DIR / "sample_ids_20260122_101559.npy",
}

for name, path in input_files.items():
    validate_input_source(path)
    print(f"  Validated: {name} ({path})")

# Load matrices
pfam_matrix = np.load(input_files['pfam_matrix'])
env_matrix = np.load(input_files['env_matrix'])
sample_ids = np.load(input_files['sample_ids'], allow_pickle=True)

with open(input_files['pfam_columns']) as f:
    pfam_cols = [line.strip() for line in f]

with open(input_files['env_columns']) as f:
    env_cols = [line.strip() for line in f]

print(f"\n  PFAM matrix: {pfam_matrix.shape}")
print(f"  Env matrix: {env_matrix.shape}")
print(f"  PFAM columns: {len(pfam_cols)}")
print(f"  Env columns: {len(env_cols)}")
print(f"  Samples: {len(sample_ids)}")

n_samples, n_pfam = pfam_matrix.shape

# Handle NaN in env matrix (same as original)
valid_cols = ~np.any(np.isnan(env_matrix), axis=0)
env_clean = env_matrix[:, valid_cols]
filtered_env_cols = [env_cols[i] for i in range(len(env_cols)) if valid_cols[i]]
print(f"  Dropped {(~valid_cols).sum()} env columns with NaN")
print(f"  Using {env_clean.shape[1]} env variables")

# ============================================================================
# Run CCA at each PCA component count
# ============================================================================
print("\n" + "=" * 70)
print("CCA PCA COMPONENT SENSITIVITY ANALYSIS")
print("=" * 70)

results = []

for n_pcs in PCA_COUNTS:
    print(f"\n--- PCA components = {n_pcs} ---")

    # Cap at min(n_samples, n_pfam) - 1
    actual_n_pcs = min(n_pcs, min(n_samples, n_pfam) - 1)
    if actual_n_pcs != n_pcs:
        print(f"  (Capped to {actual_n_pcs} due to matrix dimensions)")

    # Step 1: PCA
    pca = PCA(n_components=actual_n_pcs, random_state=42)
    pfam_pca = pca.fit_transform(pfam_matrix)

    variance_retained = pca.explained_variance_ratio_.sum()
    print(f"  PCA variance retained: {variance_retained*100:.2f}%")

    # Step 2: CCA
    n_cca = min(N_CCA_COMPONENTS, env_clean.shape[1], pfam_pca.shape[1])
    cca = CCA(n_components=n_cca)
    env_cca, pfam_cca = cca.fit_transform(env_clean, pfam_pca)

    # Compute all canonical correlations
    all_corrs = []
    for cc_idx in range(n_cca):
        corr = np.corrcoef(env_cca[:, cc_idx], pfam_cca[:, cc_idx])[0, 1]
        all_corrs.append(corr)

    # Report CC1-CC3
    for cc_idx in range(min(3, n_cca)):
        print(f"  CC{cc_idx+1}: r = {all_corrs[cc_idx]:.6f}")

    # Report all CCs
    print(f"  All CCs: ", end="")
    for cc_idx in range(n_cca):
        print(f"CC{cc_idx+1}={all_corrs[cc_idx]:.4f}", end="  ")
    print()

    # Store all CCs as results
    for cc_idx in range(n_cca):
        results.append({
            'n_pca_components': n_pcs,
            'actual_n_pca_components': actual_n_pcs,
            'pca_variance_retained': round(variance_retained, 6),
            'canonical_component': f"CC{cc_idx + 1}",
            'canonical_correlation': round(all_corrs[cc_idx], 6),
        })

# ============================================================================
# Save results
# ============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results_df = pd.DataFrame(results)

output_path = MANUSCRIPT_DIR / "source_data" / "ralph40" / "cca_pca_sensitivity.tsv"
output_path.parent.mkdir(parents=True, exist_ok=True)

provenance = create_provenance_header(
    script_path=__file__,
    input_path=str(CCA_DATA_DIR),
    output_description="CCA PCA component count sensitivity: CC1-CC10 at 50/100/150/200 PCA components"
)

with open(output_path, 'w') as f:
    for line in provenance.strip().split('\n'):
        f.write(line + '\n')
    results_df.to_csv(f, sep='\t', index=False)

print(f"  Saved: {output_path}")
print(f"  Total rows: {len(results_df)}")

# ============================================================================
# Summary table for manuscript
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY TABLE FOR MANUSCRIPT")
print("=" * 70)

# Pivot for CC1-CC3
cc13 = results_df[results_df['canonical_component'].isin(['CC1', 'CC2', 'CC3'])]
pivot = cc13.pivot(index='n_pca_components', columns='canonical_component', values='canonical_correlation')
var_retained = cc13.drop_duplicates('n_pca_components').set_index('n_pca_components')['pca_variance_retained']

print(f"\n{'PCA_n':<8} {'Var_%':<10} {'CC1':<10} {'CC2':<10} {'CC3':<10}")
print("-" * 48)
for n_pcs in PCA_COUNTS:
    vr = var_retained.loc[n_pcs] * 100
    cc1 = pivot.loc[n_pcs, 'CC1']
    cc2 = pivot.loc[n_pcs, 'CC2']
    cc3 = pivot.loc[n_pcs, 'CC3']
    print(f"{n_pcs:<8} {vr:<10.2f} {cc1:<10.4f} {cc2:<10.4f} {cc3:<10.4f}")

# Key finding
cc1_range = pivot['CC1'].max() - pivot['CC1'].min()
print(f"\nCC1 range across all PCA counts: {cc1_range:.4f}")
print(f"CC1 at 100 (baseline): {pivot.loc[100, 'CC1']:.4f}")

print(f"\n{'='*70}")
print(f"TASK 40.5 COMPLETE — {datetime.now().isoformat()}")
print(f"{'='*70}")
