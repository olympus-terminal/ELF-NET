#!/usr/bin/env python3
"""
Regenerate per-sample observed-vs-predicted vectors for reviewer-requested
biological proof-of-concept panels (Shady Amin point #2):
  - FTR1 (PF03239.20)  iron permease   -> forward env->domain scatter
  - Carbonic anhydrase family (PF00194, PF00484, PF03746) -> settle the
    unverified "LCIB R^2=0.38" main.tex claim by computing the REAL R^2
  - Dunaliella novel domain (NOVEL_33041-Dunaliella_ROIL-10xG.AAC.fa.2)
    -> Spearman rho vs bathymetry (claimed rho=0.566)

Replicates EXACTLY the forward-screen CV design from
forward_r2_all_pfams_parallel_20260131_055300.py:
  5-fold KFold(shuffle=True, random_state=42),
  XGBoost(max_depth=6, n_estimators=50, learning_rate=0.1, random_state=42),
  CLR transform (pseudocount +1), >=5% presence filter, median imputation.

IMPORTANT (data integrity): the original screen ran on the now-deleted
algagpt_gee_pfam_merged_SMART_20260119 TSV. We run on its successor
nutrient-merged TSV (the dataset of record per main.tex line 225, n~1810).
We REPORT the actual R^2 on this dataset. We do NOT force a match to the
draft values (FTR1 0.40 / LCIB 0.38). No synthetic data is generated.

Outputs (timestamped) to OUT_DIR:
  panel_FTR1_obs_pred_<ts>.tsv          per-sample observed vs predicted (CV)
  panel_CA_r2_summary_<ts>.tsv          real R^2 for every CA pfam (LCIB check)
  panel_dunaliella_bathy_<ts>.tsv       abundance + bathymetry + rho
  panel_regen_provenance_<ts>.md        provenance + computed headline numbers
"""
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


def get_base_dir(project_name: str) -> Path:
    hostname = socket.gethostname()
    if any(t in hostname for t in ("cn", "gpu", "jubail", "login", "dn")):
        p = Path(f"/scratch/drn2/PROJECTS/{project_name}")
        if p.exists():
            return p
    p_local = Path(f"/media/drn2/External/{project_name}")
    if p_local.exists():
        return p_local
    # fall back to scratch even if hostname heuristic missed
    return Path(f"/scratch/drn2/PROJECTS/{project_name}")


PROJECT = "TARA-LA4SR"
BASE = get_base_dir(PROJECT)
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

MERGED_TSV = BASE / "03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_nutrients_merged_20260320_090002.tsv"
NOVEL_MATRIX = BASE / "03_analyses/novel_domains/results/novel_domain_count_matrix.tsv"
OUT_DIR = BASE / "MANUSCRIPT/source_data/panel_regen"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FTR1 = "PF03239.20"
CA_PFAMS = ["PF00194.26", "PF00484.24", "PF03746.21"]  # alpha, Pro_CA/beta, LCIB-type beta
DUNALIELLA_DOMAIN = "NOVEL_33041-Dunaliella_ROIL-10xG.AAC.fa.2"

# Exact env feature set + params from the original screen script
ENV_COLS = [
    "latitude", "longitude",
    "solar_rad_mj_m2", "bathymetry_m", "distance_to_coast_km",
    "sst_mean_c", "sst_max_c", "sst_min_c", "sst_range_c",
    "chl_mean_mg_m3", "chl_max_mg_m3", "chl_min_mg_m3", "nflh_mean", "poc_mean_mg_m3",
    "modis_sst_mean_c",
    "rrs_412", "rrs_443", "rrs_469", "rrs_488", "rrs_531",
    "rrs_547", "rrs_555", "rrs_645", "rrs_667", "rrs_678",
]
KEY_ENV_COLS = ["latitude", "longitude", "bathymetry_m", "modis_sst_mean_c", "solar_rad_mj_m2"]
XGB = dict(max_depth=6, n_estimators=50, learning_rate=0.1,
           random_state=42, n_jobs=4, verbosity=0)


def clr_transform(X):
    X_pseudo = X + 1
    log_X = np.log(X_pseudo)
    geom_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geom_mean


def cv_obs_pred(X, y):
    """5-fold CV; return per-sample out-of-fold predictions + per-fold R^2."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full(len(y), np.nan)
    fold_r2 = []
    for tr, te in kf.split(X):
        m = XGBRegressor(**XGB)
        m.fit(X[tr], y[tr], verbose=False)
        pred = m.predict(X[te])
        oof[te] = pred
        ss_res = np.sum((y[te] - pred) ** 2)
        ss_tot = np.sum((y[te] - np.mean(y[te])) ** 2)
        fold_r2.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    return oof, np.nanmean(fold_r2), np.nanstd(fold_r2)


def main():
    print(f"[{datetime.now()}] host={socket.gethostname()} BASE={BASE}")
    print(f"Loading {MERGED_TSV}")
    df = pd.read_csv(MERGED_TSV, sep="\t", comment="#", low_memory=False)
    print(f"Loaded {len(df)} rows x {len(df.columns)} cols")

    pfam_cols = [c for c in df.columns if c.startswith("PF")]
    env_avail = [c for c in ENV_COLS if c in df.columns]
    key_avail = [c for c in KEY_ENV_COLS if c in df.columns]
    print(f"{len(pfam_cols)} PFAM cols, {len(env_avail)} env cols")

    env = df[env_avail].apply(pd.to_numeric, errors="coerce")
    pf = df[pfam_cols].apply(pd.to_numeric, errors="coerce")
    sample_ids = df["assembly_id"].astype(str).values if "assembly_id" in df.columns else np.arange(len(df)).astype(str)

    valid = ~env[key_avail].isna().any(axis=1)
    env, pf = env[valid].reset_index(drop=True), pf[valid].reset_index(drop=True)
    sample_ids = sample_ids[valid.values]
    print(f"Samples with key env complete: {valid.sum()}")

    for c in env.columns:
        if env[c].isna().any():
            env[c] = env[c].fillna(env[c].median())

    presence = (pf > 0).mean()
    keep = presence[presence >= 0.05].index.tolist()
    pf = pf[keep]
    print(f"PFAMs >=5% presence: {len(keep)}")

    clr = pd.DataFrame(clr_transform(pf.values), columns=keep)
    X = env.values
    bathy = env["bathymetry_m"].values
    headline = {}

    # ---- FTR1 ----
    if FTR1 in clr.columns:
        oof, r2m, r2s = cv_obs_pred(X, clr[FTR1].values)
        out = pd.DataFrame({"assembly_id": sample_ids,
                            "observed_clr": clr[FTR1].values,
                            "predicted_clr": oof,
                            "bathymetry_m": bathy})
        fp = OUT_DIR / f"panel_FTR1_obs_pred_{TS}.tsv"
        out.to_csv(fp, sep="\t", index=False)
        headline["FTR1_PF03239_r2"] = f"{r2m:.4f} +/- {r2s:.4f} (n={len(out)})"
        print(f"FTR1 R2={r2m:.4f}; wrote {fp}")
    else:
        headline["FTR1_PF03239_r2"] = "PF03239 not present after presence filter"

    # ---- Carbonic anhydrase / LCIB check ----
    ca_rows = []
    for ca in CA_PFAMS:
        if ca in clr.columns:
            _, r2m, r2s = cv_obs_pred(X, clr[ca].values)
            ca_rows.append({"pfam": ca, "r2_mean": r2m, "r2_std": r2s, "present": True})
            print(f"CA {ca} R2={r2m:.4f}")
        else:
            ca_rows.append({"pfam": ca, "r2_mean": np.nan, "r2_std": np.nan, "present": False})
    ca_df = pd.DataFrame(ca_rows)
    ca_fp = OUT_DIR / f"panel_CA_r2_summary_{TS}.tsv"
    ca_df.to_csv(ca_fp, sep="\t", index=False)
    best_ca = ca_df.loc[ca_df["r2_mean"].idxmax()] if ca_df["r2_mean"].notna().any() else None
    headline["LCIB_CA_best_r2"] = (
        f"{best_ca['pfam']} R2={best_ca['r2_mean']:.4f}" if best_ca is not None else "no CA pfam present"
    )
    headline["LCIB_claim_check"] = "main.tex claims LCIB R2=0.38; compare to CA values above"

    # ---- Dunaliella novel domain vs bathymetry ----
    # NOTE: main.tex reports rho=0.566, n=1,523 (bathymetry-available samples),
    # NOT the SST-key-filtered n=1,279 set used for FTR1. So we join the novel
    # matrix against the FULL forward TSV bathymetry column, restricted to rows
    # that actually carry a bathymetry value (environmental samples; cultured
    # references lack bathymetry and are dropped by the dropna).
    # Also: novel matrix IDs carry a trailing ".aa" suffix absent in the forward
    # TSV assembly_id -> strip it before joining.
    try:
        nd = pd.read_csv(NOVEL_MATRIX, sep="\t",
                         usecols=lambda c: c in ("assembly_id", DUNALIELLA_DOMAIN))
        if DUNALIELLA_DOMAIN in nd.columns:
            nd["assembly_id"] = nd["assembly_id"].astype(str).str.replace(
                r"\.aa$", "", regex=True)
            # full (unfiltered) bathymetry per sample from the original df
            full_b = pd.DataFrame({
                "assembly_id": df["assembly_id"].astype(str).values,
                "bathymetry_m": pd.to_numeric(df["bathymetry_m"], errors="coerce").values,
            }).dropna(subset=["bathymetry_m"])
            merged = full_b.merge(nd, on="assembly_id", how="inner").dropna(
                subset=[DUNALIELLA_DOMAIN, "bathymetry_m"])
            rho, pval = spearmanr(merged[DUNALIELLA_DOMAIN], merged["bathymetry_m"])
            dfp = OUT_DIR / f"panel_dunaliella_bathy_{TS}.tsv"
            merged.to_csv(dfp, sep="\t", index=False)
            headline["Dunaliella_rho_bathymetry"] = f"rho={rho:.4f} p={pval:.2e} n={len(merged)}"
            print(f"Dunaliella rho={rho:.4f} (n={len(merged)}); wrote {dfp}")
        else:
            headline["Dunaliella_rho_bathymetry"] = f"{DUNALIELLA_DOMAIN} not a column in novel matrix"
    except Exception as e:
        headline["Dunaliella_rho_bathymetry"] = f"ERROR: {e}"

    # ---- provenance ----
    prov = OUT_DIR / f"panel_regen_provenance_{TS}.md"
    with open(prov, "w") as f:
        f.write("# Panel vector regeneration — provenance\n\n")
        f.write(f"- Script: {Path(__file__).resolve()}\n")
        f.write(f"- Host: {socket.gethostname()}\n")
        f.write(f"- Date: {datetime.now()}\n")
        f.write(f"- Input (forward): {MERGED_TSV}\n")
        f.write(f"- Input (novel): {NOVEL_MATRIX}\n")
        f.write("- CV: 5-fold KFold shuffle random_state=42; XGB depth6/n50/lr0.1\n")
        f.write("- CLR pseudocount +1; >=5% presence filter; median env imputation\n")
        f.write("- Integrity: real data only; values reported as computed (no draft-matching)\n\n")
        f.write("## Computed headline numbers\n\n")
        for k, v in headline.items():
            f.write(f"- **{k}**: {v}\n")
    print(f"Wrote {prov}")
    print("HEADLINE:", headline)


if __name__ == "__main__":
    main()
