#!/usr/bin/env python3
"""
Step 8: Environmental Correlation Analysis for Novel Domains

Provenance:
    Script: scripts/novel_domains/08_env_correlation.py
    Generated: 2026-02-21, updated 2026-03-15

Purpose:
    Apply the same Spearman correlation + FDR framework used for Pfam domains
    to novel domain families:
    1. CLR-normalize the novel domain count matrix
    2. Correlate with GEE environmental variables (SST, chl, POC, etc.)
    3. Compare effect sizes between novel and known Pfam domains
    4. Test whether novel domains add predictive power (gradient boosting)

Input:
    - 03_analyses/novel_domains/results/novel_domain_count_matrix.tsv
    - 03_analyses/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv
      (contains both GEE env variables and Pfam counts per assembly)

Output:
    - 03_analyses/novel_domains/results/novel_domain_env_correlations.tsv
    - 03_analyses/novel_domains/results/novel_domain_gbr_performance.tsv
    - 03_analyses/novel_domains/results/novel_vs_pfam_effect_sizes.tsv

Usage:
    python3 scripts/novel_domains/08_env_correlation.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas required. Install with: pip install pandas")
    sys.exit(1)

try:
    from scipy import stats as scipy_stats
    from scipy.stats import spearmanr
except ImportError:
    print("ERROR: scipy required. Install with: pip install scipy")
    sys.exit(1)

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None
    print("WARNING: statsmodels not available, FDR correction will use Bonferroni")

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not available, skipping gradient boosting analysis")

warnings.filterwarnings("ignore")

BASE = Path("/scratch/drn2/PROJECTS/TARA-LA4SR")
NOVEL_DIR = BASE / "03_analyses/novel_domains"
RESULTS_DIR = NOVEL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Minimum sample prevalence for a domain to be included
MIN_PREVALENCE = 10

def clr_transform(count_matrix):
    """Centered log-ratio transform for compositional data."""
    # Add pseudocount
    X = count_matrix + 1.0
    # Log transform
    log_X = np.log(X)
    # Subtract geometric mean per sample
    geo_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geo_mean

def load_novel_matrix():
    """Load novel domain count matrix and filter by prevalence."""
    path = RESULTS_DIR / "novel_domain_count_matrix.tsv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run 06b_build_count_matrix.py first.")
        sys.exit(1)

    df = pd.read_csv(path, sep="\t", index_col=0)
    # Strip .aa suffix so IDs match GEE merged file
    df.index = df.index.str.replace(r"\.aa$", "", regex=True)
    print(f"  Raw novel matrix: {df.shape[0]} samples x {df.shape[1]} domains")

    # Filter by prevalence
    prevalence = (df > 0).sum(axis=0)
    prevalent = prevalence[prevalence >= MIN_PREVALENCE].index
    df_filtered = df[prevalent]
    print(f"  After prevalence filter (>={MIN_PREVALENCE}): {df_filtered.shape[1]} domains")

    return df_filtered

# Try nutrients-merged dataset first, fall back to original
_nutrients_path = BASE / "03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_nutrients_merged_20260320_090002.tsv"
_original_path = BASE / "03_analyses/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
GEE_MERGED = _nutrients_path if _nutrients_path.exists() else _original_path

# GEE environmental columns to use (numeric, meaningful for correlation)
ENV_COLS = [
    "salinity_psu_est", "air_temp_mean_c", "air_temp_max_c", "air_temp_min_c",
    "air_temp_range_c", "precip_mean_mm", "solar_rad_mj_m2", "bathymetry_m",
    "distance_to_coast_km", "sst_mean_c", "sst_max_c", "sst_min_c",
    "sst_range_c", "chl_mean_mg_m3", "chl_max_mg_m3", "chl_min_mg_m3",
    "nflh_mean", "poc_mean_mg_m3", "modis_sst_mean_c",
    # WOA23 nutrients + MLD (added for nutrient integration)
    "nitrate_umol_l", "phosphate_umol_l", "silicate_umol_l",
    "oxygen_umol_l", "mld_m",
]

def load_gee_merged():
    """Load the GEE+Pfam merged file, returning env DataFrame and Pfam DataFrame."""
    if not GEE_MERGED.exists():
        print(f"ERROR: {GEE_MERGED} not found")
        sys.exit(1)

    df = pd.read_csv(GEE_MERGED, sep="\t", comment="#")
    df = df.set_index("assembly_id")
    print(f"  GEE merged file: {df.shape[0]} samples x {df.shape[1]} columns")

    # Environmental variables
    env_available = [c for c in ENV_COLS if c in df.columns]
    env_df = df[env_available].apply(pd.to_numeric, errors="coerce")
    print(f"  GEE env variables: {len(env_available)}")

    # Pfam columns
    pfam_cols = [c for c in df.columns if c.startswith("PF")]
    pfam_df = df[pfam_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Filter Pfam by prevalence
    prevalence = (pfam_df > 0).sum(axis=0)
    prevalent = prevalence[prevalence >= MIN_PREVALENCE].index
    pfam_df = pfam_df[prevalent]
    print(f"  Pfam domains (prevalence>={MIN_PREVALENCE}): {pfam_df.shape[1]}")

    return env_df, pfam_df

def load_environmental_data():
    """Load AlphaEarth embeddings and environmental variables."""
    env_data = {}

    # AlphaEarth embeddings
    ae_dir = BASE / "03_analyses/ALGAGPT-based-analyses/env_pfam_manifold/data"
    ae_files = sorted(ae_dir.glob("coordinates_*.npy"))
    if ae_files:
        coords = np.load(ae_files[-1])
        sample_files = sorted(ae_dir.glob("sample_ids_*.npy"))
        if sample_files:
            samples = np.load(sample_files[-1], allow_pickle=True)
            for i in range(coords.shape[1]):
                env_data[f"AE_dim{i+1}"] = dict(zip(samples, coords[:, i]))
            print(f"  AlphaEarth: {coords.shape[1]} dimensions, {len(samples)} samples")

    # Merged environmental data (prefer nutrients-merged if available)
    gee_dir = BASE / "03_analyses/ALGAGPT-based-analyses"
    nutrient_files = sorted(glob.glob(str(gee_dir / "algagpt_gee_pfam_nutrients_merged_*.tsv")))
    gee_files = nutrient_files if nutrient_files else sorted(glob.glob(str(gee_dir / "algagpt_gee_pfam_merged_*.tsv")))
    if gee_files:
        gee_df = pd.read_csv(gee_files[-1], sep="\t", comment="#", nrows=0)
        env_cols = [c for c in gee_df.columns
                    if not c.startswith("PF") and c not in ("assembly_id",)]
        if env_cols:
            gee_df = pd.read_csv(gee_files[-1], sep="\t", comment="#",
                                 usecols=["assembly_id"] + env_cols[:60])
            gee_df = gee_df.set_index("assembly_id")
            for col in gee_df.columns:
                if gee_df[col].dtype in (np.float64, np.int64, float, int):
                    env_data[col] = gee_df[col].dropna().to_dict()
            print(f"  Environmental vars: {len(env_cols)} columns")

    return env_data

def run_correlations(domain_df, env_data):
    """Run Spearman correlations between domains and environmental variables."""
    results = []

    # Shared samples
    shared = domain_df.index.intersection(env_df.index)
    if len(shared) < 20:
        print(f"  WARNING: Only {len(shared)} shared samples, need >=20")
        return pd.DataFrame()

    # CLR normalize domain counts
    clr_matrix = clr_transform(domain_df.loc[shared].values)
    clr_df = pd.DataFrame(clr_matrix, index=shared, columns=domain_df.columns)

    env_vars = list(env_df.columns)
    domains = list(clr_df.columns)

    print(f"  Shared samples: {len(shared)}")
    print(f"  Testing {len(domains)} domains x {len(env_vars)} env variables...")
    print(f"  Total tests: {len(domains) * len(env_vars):,}")

    for env_var in env_vars:
        env_vals = env_df.loc[shared, env_var].values
        # Drop NaN pairs
        valid = ~np.isnan(env_vals)
        if valid.sum() < 20:
            continue
        ev = env_vals[valid]
        if np.std(ev) < 1e-10:
            continue

        for domain in domains:
            dv = clr_df.loc[shared, domain].values[valid]
            rho, pval = spearmanr(dv, ev)

            if not np.isnan(rho):
                results.append({
                    "domain": domain,
                    "env_variable": env_var,
                    "rho": rho,
                    "p_value": pval,
                    "n_samples": int(valid.sum()),
                })

    if not results:
        print("  WARNING: No valid correlations computed")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # FDR correction
    if multipletests is not None:
        reject, qvals, _, _ = multipletests(
            results_df["p_value"].values, method="fdr_bh"
        )
        results_df["q_value"] = qvals
        results_df["significant"] = reject
    else:
        n_tests = len(results_df)
        results_df["q_value"] = np.minimum(results_df["p_value"] * n_tests, 1.0)
        results_df["significant"] = results_df["q_value"] < 0.05

    results_df = results_df.sort_values("p_value")

    n_sig = results_df["significant"].sum()
    print(f"  Significant correlations (FDR<0.05): {n_sig:,} / {len(results_df):,}")

    return results_df

MAX_GBR_FEATURES = 500  # Top features by variance for GBR (keeps runtime tractable)

def _select_top_features(X, n=MAX_GBR_FEATURES):
    """Select top-n columns by variance (unbiased feature reduction for GBR)."""
    if X.shape[1] <= n:
        return X
    variances = np.var(X, axis=0)
    top_idx = np.argsort(variances)[-n:]
    return X[:, top_idx]

def run_gbr_comparison(novel_df, pfam_df, env_df):
    """Compare predictive power: Pfam-only vs Pfam+novel using gradient boosting."""
    if not HAS_SKLEARN:
        return None

    print("  Running gradient boosting comparison...")

    # Shared samples across all three
    shared = novel_df.index.intersection(pfam_df.index).intersection(env_df.index)
    print(f"  Shared samples (novel+pfam+env): {len(shared)}")

    if len(shared) < 50:
        print("    Not enough shared samples for gradient boosting")
        return None

    results = []

    for target_var in env_df.columns:
        y = env_df.loc[shared, target_var].values
        valid = ~np.isnan(y)
        if valid.sum() < 50:
            continue
        y_valid = y[valid]
        if np.std(y_valid) < 1e-10:
            continue

        idx = shared[valid]

        # Pfam-only features (CLR, top-N by variance)
        X_pfam = _select_top_features(clr_transform(pfam_df.loc[idx].values))
        # Novel-only features (CLR, top-N by variance)
        X_novel = _select_top_features(clr_transform(novel_df.loc[idx].values))
        # Combined
        X_combined = np.hstack([X_pfam, X_novel])

        gbr = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42,
            subsample=0.8, learning_rate=0.1
        )

        try:
            r2_pfam = cross_val_score(gbr, X_pfam, y_valid, cv=5, scoring="r2").mean()
            r2_novel = cross_val_score(gbr, X_novel, y_valid, cv=5, scoring="r2").mean()
            r2_combined = cross_val_score(gbr, X_combined, y_valid, cv=5, scoring="r2").mean()
        except Exception as e:
            print(f"    Skipping {target_var}: {e}")
            continue

        results.append({
            "target": target_var,
            "r2_pfam_only": f"{r2_pfam:.4f}",
            "r2_novel_only": f"{r2_novel:.4f}",
            "r2_combined": f"{r2_combined:.4f}",
            "improvement": f"{r2_combined - r2_pfam:.4f}",
            "n_samples": int(valid.sum()),
            "n_pfam_features": X_pfam.shape[1],
            "n_novel_features": X_novel.shape[1],
        })
        print(f"    {target_var}: Pfam R²={r2_pfam:.3f}, "
              f"Novel R²={r2_novel:.3f}, Combined R²={r2_combined:.3f}")

    return pd.DataFrame(results) if results else None

def main():
    print("=" * 60)
    print("  Step 8: Environmental Correlation Analysis")
    print("=" * 60)
    print()

    # ---- Load data ----
    print("Loading novel domain matrix...")
    novel_df = load_novel_matrix()

    print("Loading GEE merged file (env + Pfam)...")
    env_df, pfam_df = load_gee_merged()
    print()

    # ---- Novel domain correlations ----
    print("=" * 40)
    print("  Novel Domain x GEE Correlations")
    print("=" * 40)

    corr_df = run_correlations(novel_df, env_df)

    if len(corr_df) > 0:
        corr_path = RESULTS_DIR / "novel_domain_env_correlations.tsv"
        corr_df.to_csv(corr_path, sep="\t", index=False)
        print(f"  Written: {corr_path}")

        # Effect size comparison with Pfam
        if pfam_df is not None and len(pfam_df.columns) > 0:
            print()
            print("  Running Pfam correlations for comparison...")
            pfam_corr = run_correlations(pfam_df, env_df)

            if len(pfam_corr) > 0:
                novel_abs_rho = corr_df["rho"].abs()
                pfam_abs_rho = pfam_corr["rho"].abs()

                eff_path = RESULTS_DIR / "novel_vs_pfam_effect_sizes.tsv"
                with open(eff_path, "w") as f:
                    f.write("metric\tpfam\tnovel\n")
                    f.write(f"n_correlations\t{len(pfam_corr)}\t{len(corr_df)}\n")
                    f.write(f"n_significant\t{pfam_corr['significant'].sum()}\t"
                            f"{corr_df['significant'].sum()}\n")
                    f.write(f"median_abs_rho\t{pfam_abs_rho.median():.4f}\t"
                            f"{novel_abs_rho.median():.4f}\n")
                    f.write(f"mean_abs_rho\t{pfam_abs_rho.mean():.4f}\t"
                            f"{novel_abs_rho.mean():.4f}\n")
                    f.write(f"max_abs_rho\t{pfam_abs_rho.max():.4f}\t"
                            f"{novel_abs_rho.max():.4f}\n")
                print(f"  Written: {eff_path}")
    print()

    # ---- Gradient Boosting ----
    if HAS_SKLEARN and pfam_df is not None and len(pfam_df.columns) > 0:
        print("=" * 40)
        print("  Gradient Boosting Predictive Power")
        print("=" * 40)

        gbr_df = run_gbr_comparison(novel_df, pfam_df, env_df)

        if gbr_df is not None and len(gbr_df) > 0:
            gbr_path = RESULTS_DIR / "novel_domain_gbr_performance.tsv"
            gbr_df.to_csv(gbr_path, sep="\t", index=False)
            print(f"  Written: {gbr_path}")
    print()

    # ---- Summary ----
    print("=" * 60)
    print("  Analysis Complete")
    print("=" * 60)
    print()
    print(f"  Results directory: {RESULTS_DIR}")
    for f in sorted(RESULTS_DIR.glob("*.tsv")):
        print(f"    {f.name}")
    print()

if __name__ == "__main__":
    main()
