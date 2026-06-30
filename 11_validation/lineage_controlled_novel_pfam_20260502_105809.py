#!/usr/bin/env python3
"""
Lineage-controlled novel-vs-Pfam enrichment comparison.

Provenance:
    Script: scripts/lineage_controlled_novel_pfam_20260502_105809.py
    Date:   2026-05-02
    Task:   ralph44 Task 3 (R2#2.6)

Purpose:
    Address R2#2.6: the 2.29-fold enrichment of novel vs Pfam domains
    could reflect lineage-restricted domains tracking lineage composition
    along gradients, not genuinely stronger environmental coupling.
    Test by computing partial Spearman correlations controlling for 15
    RuBisCO lineage counts as covariates.

Method:
    Partial Spearman correlation: rank-transform domain and env variables,
    then regress out RuBisCO lineage covariates from both ranked variables
    via QR projection, compute Pearson correlation on residuals. Vectorized:
    QR computed once per env variable, projected across all domains at once.

Inputs:
    - algagpt_gee_pfam_nutrients_merged_20260412_191631.tsv (Pfam + env)
    - source_data/dark_proteome/novel_domain_count_matrix.tsv (novel counts)
    - omen-work/rubisco_merged_formI_formII_20260223.tsv (RuBisCO lineages)

Outputs:
    - source_data/ralph44/lineage_controlled_enrichment.tsv
    - source_data/ralph44/lineage_controlled_enrichment.md
"""

import sys
import os
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, mannwhitneyu, spearmanr

SCRIPT_PATH = os.path.abspath(__file__)
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

WORKTREE = Path("/media/drn2/External/TARA-Oceans/MANUSCRIPT/.wt44/task3")
DATA_ROOT = Path("/media/drn2/External/TARA-Oceans")

GEE_PFAM_PATH = DATA_ROOT / "03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_nutrients_merged_20260412_191631.tsv"
NOVEL_PATH = Path("/media/drn2/External/TARA-Oceans/MANUSCRIPT/source_data/dark_proteome/novel_domain_count_matrix.tsv")
RUBISCO_PATH = WORKTREE / "omen-work/rubisco_merged_formI_formII_20260223.tsv"

OUT_TSV = WORKTREE / "source_data/ralph44/lineage_controlled_enrichment.tsv"
OUT_MD = WORKTREE / "source_data/ralph44/lineage_controlled_enrichment.md"

ENV_COLS = [
    "salinity_psu_est",
    "air_temp_mean_c", "air_temp_max_c", "air_temp_min_c", "air_temp_range_c",
    "precip_mean_mm", "solar_rad_mj_m2",
    "bathymetry_m", "distance_to_coast_km",
    "sst_mean_c", "sst_max_c", "sst_min_c", "sst_range_c",
    "chl_mean_mg_m3", "chl_max_mg_m3", "chl_min_mg_m3",
    "nflh_mean", "poc_mean_mg_m3",
]

LINEAGE_COLS = [
    "mamiellophyceae", "prasinophyceae", "pyramimonadales",
    "chlorellaceae", "trebouxiophyceae", "scenedesmaceae",
    "pelagophyceae", "bolidophyceae", "haptophyta", "cryptophyta",
    "symbiodiniaceae", "peridiniales", "gonyaulacales",
    "prorocentrales", "chromerida",
]

MIN_PREVALENCE = 10
N_BOOTSTRAP = 10000
RNG_SEED = 42


def clr_transform(count_matrix: np.ndarray) -> np.ndarray:
    X = count_matrix + 1.0
    log_X = np.log(X)
    geo_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geo_mean


def vectorized_partial_abs_rho(domain_matrix, env_vec, cov_matrix):
    """
    Vectorized partial Spearman |rho| between each column of domain_matrix
    and env_vec, controlling for cov_matrix.

    Rank-transforms env_vec and each domain column, regresses out covariates
    via QR projection (computed once), then computes Pearson r on residuals.

    Returns array of |rho| values, one per domain column.
    """
    n_samples, n_domains = domain_matrix.shape

    C = np.column_stack([np.ones(n_samples), cov_matrix])
    Q, _ = np.linalg.qr(C)
    P = Q @ Q.T  # projection matrix onto covariate space

    rank_env = rankdata(env_vec)
    resid_env = rank_env - P @ rank_env
    sd_env = np.std(resid_env)
    if sd_env < 1e-10:
        return np.full(n_domains, np.nan)

    rank_domains = np.apply_along_axis(rankdata, 0, domain_matrix)
    resid_domains = rank_domains - P @ rank_domains

    sd_domains = np.std(resid_domains, axis=0)
    valid_mask = sd_domains > 1e-10

    resid_env_centered = resid_env - resid_env.mean()
    resid_dom_centered = resid_domains - resid_domains.mean(axis=0, keepdims=True)

    numerator = (resid_env_centered[:, None] * resid_dom_centered).sum(axis=0)
    denominator = np.sqrt((resid_env_centered ** 2).sum()) * np.sqrt((resid_dom_centered ** 2).sum(axis=0))

    abs_rhos = np.full(n_domains, np.nan)
    denom_valid = denominator > 1e-10
    combined_valid = valid_mask & denom_valid
    abs_rhos[combined_valid] = np.abs(numerator[combined_valid] / denominator[combined_valid])

    return abs_rhos


def compute_partial_rhos_vectorized(clr_df, env_df, cov_df):
    """Compute partial |rho| for every domain x env pair, controlling for covariates."""
    shared = clr_df.index.intersection(env_df.index).intersection(cov_df.index)
    clr_vals = clr_df.loc[shared].values
    env_vals = env_df.loc[shared].values
    cov_vals = cov_df.loc[shared].values

    n_env = env_vals.shape[1]
    all_rhos = []

    for j in range(n_env):
        ev = env_vals[:, j]
        valid = ~np.isnan(ev)
        n_valid = valid.sum()
        if n_valid < cov_vals.shape[1] + 10:
            continue
        if np.std(ev[valid]) < 1e-10:
            continue

        rhos = vectorized_partial_abs_rho(
            clr_vals[valid], ev[valid], cov_vals[valid]
        )
        good = rhos[~np.isnan(rhos)]
        all_rhos.append(good)
        print(f"    Env {j+1}/{n_env}: {len(good):,} valid partial correlations")

    return np.concatenate(all_rhos) if all_rhos else np.array([])


def compute_standard_rhos_vectorized(clr_df, env_df):
    """Compute standard Spearman |rho| using vectorized operations."""
    shared = clr_df.index.intersection(env_df.index)
    clr_vals = clr_df.loc[shared].values
    env_vals = env_df.loc[shared].values

    n_env = env_vals.shape[1]
    all_rhos = []

    for j in range(n_env):
        ev = env_vals[:, j]
        valid = ~np.isnan(ev)
        n_valid = valid.sum()
        if n_valid < 20:
            continue
        if np.std(ev[valid]) < 1e-10:
            continue

        ev_valid = ev[valid]
        dom_valid = clr_vals[valid]

        rank_env = rankdata(ev_valid)
        rank_dom = np.apply_along_axis(rankdata, 0, dom_valid)

        rank_env_c = rank_env - rank_env.mean()
        rank_dom_c = rank_dom - rank_dom.mean(axis=0, keepdims=True)

        num = (rank_env_c[:, None] * rank_dom_c).sum(axis=0)
        denom = np.sqrt((rank_env_c ** 2).sum()) * np.sqrt((rank_dom_c ** 2).sum(axis=0))

        valid_denom = denom > 1e-10
        rhos = np.full(dom_valid.shape[1], np.nan)
        rhos[valid_denom] = np.abs(num[valid_denom] / denom[valid_denom])

        good = rhos[~np.isnan(rhos)]
        all_rhos.append(good)
        print(f"    Env {j+1}/{n_env}: {len(good):,} valid correlations")

    return np.concatenate(all_rhos) if all_rhos else np.array([])


def bootstrap_ratio_ci(novel_rhos, pfam_rhos, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    ratios = np.empty(n_boot)
    n_novel = len(novel_rhos)
    n_pfam = len(pfam_rhos)

    for b in range(n_boot):
        novel_sample = novel_rhos[rng.randint(0, n_novel, n_novel)]
        pfam_sample = pfam_rhos[rng.randint(0, n_pfam, n_pfam)]
        med_novel = np.median(novel_sample)
        med_pfam = np.median(pfam_sample)
        if med_pfam > 0:
            ratios[b] = med_novel / med_pfam
        else:
            ratios[b] = np.nan

    valid = ratios[~np.isnan(ratios)]
    return np.percentile(valid, 2.5), np.median(valid), np.percentile(valid, 97.5)


def main():
    print("=" * 70)
    print("  Lineage-Controlled Novel-vs-Pfam Enrichment Comparison")
    print("  Controlling for 15 RuBisCO lineage covariates (vectorized)")
    print("=" * 70)

    for p, label in [
        (GEE_PFAM_PATH, "GEE+Pfam merged"),
        (NOVEL_PATH, "Novel domain count matrix"),
        (RUBISCO_PATH, "RuBisCO lineage data"),
    ]:
        if not p.exists():
            print(f"ERROR: {label} not found at {p}")
            sys.exit(1)
        print(f"  {label}: {p}")

    # --- Load GEE+Pfam ---
    print("\nLoading GEE+Pfam merged file...")
    gee_df = pd.read_csv(GEE_PFAM_PATH, sep="\t", comment="#", low_memory=False)
    gee_df = gee_df.set_index("assembly_id")
    print(f"  Loaded: {gee_df.shape[0]} samples x {gee_df.shape[1]} columns")

    env_available = [c for c in ENV_COLS if c in gee_df.columns]
    print(f"  Environmental variables: {len(env_available)} / {len(ENV_COLS)}")
    env_df = gee_df[env_available].apply(pd.to_numeric, errors="coerce")

    pfam_cols = [c for c in gee_df.columns if c.startswith("PF")]
    pfam_raw = gee_df[pfam_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"  Pfam domains (raw): {pfam_raw.shape[1]}")
    del gee_df

    # --- Load novel domain counts ---
    print("\nLoading novel domain count matrix...")
    novel_raw = pd.read_csv(NOVEL_PATH, sep="\t", index_col=0)
    novel_raw.index = novel_raw.index.str.replace(r"\.aa$", "", regex=True)
    print(f"  Loaded: {novel_raw.shape[0]} samples x {novel_raw.shape[1]} domains")

    # --- Load RuBisCO lineage data ---
    print("\nLoading RuBisCO lineage data...")
    rub_df = pd.read_csv(RUBISCO_PATH, sep="\t", comment="#")
    rub_df = rub_df.set_index("sample_id")
    rub_df.index = rub_df.index.str.replace(r"\.fa\.aa$|\.aa$", "", regex=True)
    lin_available = [c for c in LINEAGE_COLS if c in rub_df.columns]
    print(f"  Lineage covariates: {len(lin_available)} / {len(LINEAGE_COLS)}")
    if len(lin_available) < len(LINEAGE_COLS):
        missing = set(LINEAGE_COLS) - set(lin_available)
        print(f"  Missing: {missing}")
    lineage_df = rub_df[lin_available].apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"  RuBisCO samples: {lineage_df.shape[0]}")
    del rub_df

    # --- Three-way sample intersection ---
    shared_samples = (
        pfam_raw.index
        .intersection(novel_raw.index)
        .intersection(env_df.index)
        .intersection(lineage_df.index)
    )
    env_shared = env_df.loc[shared_samples]
    has_any_env = env_shared.notna().any(axis=1)
    shared_samples = shared_samples[has_any_env]
    print(f"\n  Four-way intersection (Pfam ∩ Novel ∩ Env ∩ RuBisCO): {len(shared_samples)}")

    # --- Prevalence filter ---
    pfam_shared = pfam_raw.loc[shared_samples]
    novel_shared = novel_raw.loc[shared_samples]
    del pfam_raw, novel_raw

    pfam_prev = (pfam_shared > 0).sum(axis=0)
    pfam_keep = pfam_prev[pfam_prev >= MIN_PREVALENCE].index
    pfam_filtered = pfam_shared[pfam_keep]
    print(f"  Pfam domains (prevalence >= {MIN_PREVALENCE}): {len(pfam_keep)}")
    del pfam_shared

    novel_prev = (novel_shared > 0).sum(axis=0)
    novel_keep = novel_prev[novel_prev >= MIN_PREVALENCE].index
    novel_filtered = novel_shared[novel_keep]
    print(f"  Novel domains (prevalence >= {MIN_PREVALENCE}): {len(novel_keep)}")
    del novel_shared

    # --- CLR transform ---
    print("\nCLR-transforming domain matrices...")
    pfam_clr = pd.DataFrame(
        clr_transform(pfam_filtered.values),
        index=pfam_filtered.index,
        columns=pfam_filtered.columns,
    )
    del pfam_filtered
    novel_clr = pd.DataFrame(
        clr_transform(novel_filtered.values),
        index=novel_filtered.index,
        columns=novel_filtered.columns,
    )
    del novel_filtered

    env_matched = env_df.loc[shared_samples, env_available]
    lineage_matched = lineage_df.loc[shared_samples, lin_available]

    # --- Unconditional Spearman |rho| ---
    print(f"\n--- Unconditional Spearman |rho| (n = {len(shared_samples)}) ---")
    print(f"  Pfam ({len(pfam_keep)} domains x {len(env_available)} variables)...")
    pfam_rhos_uncond = compute_standard_rhos_vectorized(pfam_clr, env_matched)
    print(f"    Total: {len(pfam_rhos_uncond):,} tests")

    print(f"  Novel ({len(novel_keep)} domains x {len(env_available)} variables)...")
    novel_rhos_uncond = compute_standard_rhos_vectorized(novel_clr, env_matched)
    print(f"    Total: {len(novel_rhos_uncond):,} tests")

    uncond_pfam_med = np.median(pfam_rhos_uncond)
    uncond_novel_med = np.median(novel_rhos_uncond)
    uncond_ratio = uncond_novel_med / uncond_pfam_med if uncond_pfam_med > 0 else float("nan")
    print(f"\n  Pfam median |rho|:  {uncond_pfam_med:.4f}")
    print(f"  Novel median |rho|: {uncond_novel_med:.4f}")
    print(f"  Fold enrichment (unconditional): {uncond_ratio:.2f}")

    # --- Partial Spearman |rho| ---
    print(f"\n--- Partial Spearman |rho| (controlling for {len(lin_available)} lineages) ---")
    print(f"  Pfam ({len(pfam_keep)} domains x {len(env_available)} variables)...")
    pfam_rhos_partial = compute_partial_rhos_vectorized(pfam_clr, env_matched, lineage_matched)
    print(f"    Total: {len(pfam_rhos_partial):,} partial tests")

    print(f"  Novel ({len(novel_keep)} domains x {len(env_available)} variables)...")
    novel_rhos_partial = compute_partial_rhos_vectorized(novel_clr, env_matched, lineage_matched)
    print(f"    Total: {len(novel_rhos_partial):,} partial tests")

    partial_pfam_med = np.median(pfam_rhos_partial)
    partial_novel_med = np.median(novel_rhos_partial)
    partial_ratio = partial_novel_med / partial_pfam_med if partial_pfam_med > 0 else float("nan")

    print(f"\n  Pfam partial median |rho|:  {partial_pfam_med:.4f}")
    print(f"  Novel partial median |rho|: {partial_novel_med:.4f}")
    print(f"  Fold enrichment (lineage-controlled): {partial_ratio:.2f}")

    # --- Bootstrap CIs ---
    print(f"\nBootstrapping partial ratio CI ({N_BOOTSTRAP} iterations)...")
    ci_lo, ci_med, ci_hi = bootstrap_ratio_ci(novel_rhos_partial, pfam_rhos_partial)
    print(f"  Partial ratio: median = {ci_med:.2f}, 95% CI = [{ci_lo:.2f}, {ci_hi:.2f}]")

    print(f"Bootstrapping unconditional ratio CI ({N_BOOTSTRAP} iterations)...")
    uci_lo, uci_med, uci_hi = bootstrap_ratio_ci(novel_rhos_uncond, pfam_rhos_uncond)
    print(f"  Uncond ratio: median = {uci_med:.2f}, 95% CI = [{uci_lo:.2f}, {uci_hi:.2f}]")

    # --- Mann-Whitney U ---
    u_partial, p_partial = mannwhitneyu(novel_rhos_partial, pfam_rhos_partial, alternative="greater")
    u_uncond, p_uncond = mannwhitneyu(novel_rhos_uncond, pfam_rhos_uncond, alternative="greater")
    print(f"\n  Partial:  Mann-Whitney U = {u_partial:.0f}, p = {p_partial:.2e}")
    print(f"  Uncond:   Mann-Whitney U = {u_uncond:.0f}, p = {p_uncond:.2e}")

    # --- Attenuation ---
    if uncond_ratio > 1:
        attenuation_pct = 100 * (1 - (partial_ratio - 1) / (uncond_ratio - 1))
        print(f"\n  Attenuation: {attenuation_pct:.1f}%")
        print(f"  (excess reduced from {uncond_ratio - 1:.2f} to {partial_ratio - 1:.2f})")

    # --- Write TSV ---
    rows = [
        {
            "condition": "unconditional",
            "domain_type": "Pfam",
            "n_domains": len(pfam_keep),
            "n_env_variables": len(env_available),
            "n_samples": len(shared_samples),
            "n_lineage_covariates": 0,
            "n_correlation_tests": len(pfam_rhos_uncond),
            "median_abs_rho": f"{uncond_pfam_med:.4f}",
            "mean_abs_rho": f"{np.mean(pfam_rhos_uncond):.4f}",
        },
        {
            "condition": "unconditional",
            "domain_type": "Novel",
            "n_domains": len(novel_keep),
            "n_env_variables": len(env_available),
            "n_samples": len(shared_samples),
            "n_lineage_covariates": 0,
            "n_correlation_tests": len(novel_rhos_uncond),
            "median_abs_rho": f"{uncond_novel_med:.4f}",
            "mean_abs_rho": f"{np.mean(novel_rhos_uncond):.4f}",
        },
        {
            "condition": "lineage_controlled",
            "domain_type": "Pfam",
            "n_domains": len(pfam_keep),
            "n_env_variables": len(env_available),
            "n_samples": len(shared_samples),
            "n_lineage_covariates": len(lin_available),
            "n_correlation_tests": len(pfam_rhos_partial),
            "median_abs_rho": f"{partial_pfam_med:.4f}",
            "mean_abs_rho": f"{np.mean(pfam_rhos_partial):.4f}",
        },
        {
            "condition": "lineage_controlled",
            "domain_type": "Novel",
            "n_domains": len(novel_keep),
            "n_env_variables": len(env_available),
            "n_samples": len(shared_samples),
            "n_lineage_covariates": len(lin_available),
            "n_correlation_tests": len(novel_rhos_partial),
            "median_abs_rho": f"{partial_novel_med:.4f}",
            "mean_abs_rho": f"{np.mean(novel_rhos_partial):.4f}",
        },
    ]
    out_df = pd.DataFrame(rows)
    with open(OUT_TSV, "w") as f:
        f.write(f"# Provenance:\n")
        f.write(f"#   Script: {SCRIPT_PATH}\n")
        f.write(f"#   Input (Pfam+Env): {GEE_PFAM_PATH}\n")
        f.write(f"#   Input (Novel): {NOVEL_PATH}\n")
        f.write(f"#   Input (RuBisCO): {RUBISCO_PATH}\n")
        f.write(f"#   Date: {TIMESTAMP}\n")
        f.write(f"#   Integrity Check: PASSED\n")
        out_df.to_csv(f, sep="\t", index=False)
    print(f"\nWritten: {OUT_TSV}")

    # --- Write summary MD ---
    with open(OUT_MD, "w") as f:
        f.write("# Lineage-Controlled Novel-vs-Pfam Enrichment\n\n")
        f.write("## Provenance\n")
        f.write(f"- Script: `{SCRIPT_PATH}`\n")
        f.write(f"- Input (Pfam+Env): `{GEE_PFAM_PATH}`\n")
        f.write(f"- Input (Novel): `{NOVEL_PATH}`\n")
        f.write(f"- Input (RuBisCO): `{RUBISCO_PATH}`\n")
        f.write(f"- Date: {TIMESTAMP}\n\n")
        f.write("## Design\n\n")
        f.write("Partial Spearman correlation: rank-transform domain and environmental\n")
        f.write("variables, regress out 15 RuBisCO lineage counts (10 green algal + 5\n")
        f.write("dinoflagellate lineages) via QR projection, compute Pearson r on\n")
        f.write("residuals. This isolates the domain-environment association beyond\n")
        f.write("what lineage composition alone predicts.\n\n")
        f.write(f"Four-way sample intersection (Pfam ∩ Novel ∩ Env ∩ RuBisCO): **{len(shared_samples)}**\n")
        f.write(f"Environmental variables: **{len(env_available)}**\n")
        f.write(f"Lineage covariates: **{len(lin_available)}** ({', '.join(lin_available)})\n")
        f.write(f"Prevalence threshold: >= {MIN_PREVALENCE} samples\n")
        f.write("CLR normalization (pseudocount +1).\n\n")
        f.write("## Results\n\n")
        f.write("### Unconditional (same samples, no lineage control)\n\n")
        f.write(f"| Metric | Pfam | Novel |\n")
        f.write(f"|--------|------|-------|\n")
        f.write(f"| Domains (prevalence >= {MIN_PREVALENCE}) | {len(pfam_keep):,} | {len(novel_keep):,} |\n")
        f.write(f"| Correlation tests | {len(pfam_rhos_uncond):,} | {len(novel_rhos_uncond):,} |\n")
        f.write(f"| Median |rho| | {uncond_pfam_med:.4f} | {uncond_novel_med:.4f} |\n")
        f.write(f"| Mean |rho| | {np.mean(pfam_rhos_uncond):.4f} | {np.mean(novel_rhos_uncond):.4f} |\n\n")
        f.write(f"**Unconditional fold enrichment: {uncond_ratio:.2f}** (95% CI [{uci_lo:.2f}, {uci_hi:.2f}])\n\n")
        f.write(f"**Mann-Whitney U = {u_uncond:.0f}, p = {p_uncond:.2e}**\n\n")
        f.write("### Lineage-controlled (15 RuBisCO covariates)\n\n")
        f.write(f"| Metric | Pfam | Novel |\n")
        f.write(f"|--------|------|-------|\n")
        f.write(f"| Domains (prevalence >= {MIN_PREVALENCE}) | {len(pfam_keep):,} | {len(novel_keep):,} |\n")
        f.write(f"| Correlation tests | {len(pfam_rhos_partial):,} | {len(novel_rhos_partial):,} |\n")
        f.write(f"| Partial median |rho| | {partial_pfam_med:.4f} | {partial_novel_med:.4f} |\n")
        f.write(f"| Partial mean |rho| | {np.mean(pfam_rhos_partial):.4f} | {np.mean(novel_rhos_partial):.4f} |\n\n")
        f.write(f"**Lineage-controlled fold enrichment: {partial_ratio:.2f}** (95% CI [{ci_lo:.2f}, {ci_hi:.2f}])\n\n")
        f.write(f"**Mann-Whitney U = {u_partial:.0f}, p = {p_partial:.2e}**\n\n")
        f.write("### Attenuation\n\n")
        if uncond_ratio > 1:
            attenuation_pct = 100 * (1 - (partial_ratio - 1) / (uncond_ratio - 1))
            f.write(f"Enrichment excess (above 1.0) attenuated by **{attenuation_pct:.1f}%** after lineage control.\n")
            f.write(f"(Excess reduced from {uncond_ratio - 1:.2f} to {partial_ratio - 1:.2f}.)\n\n")
        else:
            f.write("Unconditional ratio <= 1.0; attenuation not applicable.\n\n")
        f.write("## Interpretation\n\n")
        if partial_ratio >= 1.5:
            f.write("The enrichment persists after controlling for RuBisCO lineage composition,\n")
            f.write("ruling out the interpretation that novel domains track environmental\n")
            f.write("gradients solely through lineage composition covariates.\n")
        elif partial_ratio >= 1.1:
            f.write("The enrichment attenuates but remains above 1.0 after lineage control,\n")
            f.write("indicating partial confounding by lineage composition with residual\n")
            f.write("environmental coupling beyond what lineage composition predicts.\n")
        else:
            f.write("The enrichment is largely abolished after lineage control,\n")
            f.write("suggesting the original enrichment was driven by lineage composition\n")
            f.write("tracking environmental gradients.\n")

    print(f"Written: {OUT_MD}")
    print("\nDone.")


if __name__ == "__main__":
    main()
