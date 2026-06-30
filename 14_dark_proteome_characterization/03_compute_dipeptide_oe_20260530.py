#!/usr/bin/env python3
"""
Compute dipeptide observed/expected (O/E) bias for the dark, annotated, and
algal proteomes, and quantify the *spread* of dipeptide bias per proteome.

Rationale (ELF-Net feedback, Kourosh #4.2 / Section 9, Figure 14):
The annotated proteome shows stronger dipeptide biases (selection for
structural motifs) while the dark proteome has weaker, more uniform dipeptide
usage. This is second-order compositional evidence for the weak-purifying-
selection interpretation that manuscript Figure 7 supports only at first order.
This script produces a *quantified* effect size (spread of the 400 off-diagonal
O/E ratios) so the claim can be calibrated, not just shown as a heatmap.

Algorithm reproduces Kourosh's analyze_proteomes.txt (Fig 14) exactly:
  expected(AA1,AA2) = f(AA1) * f(AA2) * total_dipeptides
  O/E = observed(AA1,AA2) / expected(AA1,AA2)
where single-AA frequencies are derived from the dipeptide counts (consecutive
overlapping pairs), matching the original implementation.

Effect-size statistics added here (per proteome):
  - SD and IQR of log2(O/E) over the 400 dipeptides (wider = stronger bias)
  - count and fraction of dipeptides with |log2(O/E)| > 0.5 (i.e. >1.41-fold)
  - mean |log2(O/E)| (mean absolute bias)

DATA INTEGRITY: real sequences only; streamed line-by-line (no full load).
Outputs carry a provenance header. No synthetic data.
"""

import os
import sys
import socket
import datetime
from pathlib import Path
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Environment detection (path-based, not hostname — hostname matching fails on
# dn*/login nodes; prefer Path.exists() sanity checks).
# ---------------------------------------------------------------------------
def get_base_dir() -> Path:
    hpc = Path("/scratch/drn2/PROJECTS/TARA-LA4SR")
    local = Path("/media/drn2/External/TARA-Oceans")
    if hpc.exists():
        return hpc
    if local.exists():
        return local
    sys.exit("ERROR: neither HPC nor local project base dir exists")


BASE = get_base_dir()
# On local mirror the project root is .../TARA-Oceans; on HPC it is TARA-LA4SR.
# Resolve the physicochemical FASTA dir robustly for both.
CANDIDATE_FASTA_DIRS = [
    BASE / "03_analyses" / "dark_proteome_v2" / "physicochemical",
    Path("/media/drn2/External/TARA-Oceans/03_analyses/dark_proteome_v2/physicochemical"),
]
FASTA_DIR = next((d for d in CANDIDATE_FASTA_DIRS if d.exists()), CANDIDATE_FASTA_DIRS[0])

DATASETS = [
    ("dark", FASTA_DIR / "dark_500k.fa"),
    ("annotated", FASTA_DIR / "annotated_500k.fa"),
    ("algae", FASTA_DIR / "algae_500k.fa"),
]

# Output dir: source_data/dark_proteome (mirrored local & HPC under MANUSCRIPT)
for cand in [
    BASE / "MANUSCRIPT" / "source_data" / "dark_proteome",
    Path("/media/drn2/External/TARA-Oceans/MANUSCRIPT/source_data/dark_proteome"),
]:
    if cand.parent.exists():
        OUT_DIR = cand
        break
OUT_DIR.mkdir(parents=True, exist_ok=True)

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)


def stream_dipeptide_counts(fpath: Path):
    """Stream a FASTA, accumulating dipeptide counts over standard AAs only.

    Memory-efficient: never loads the file; accumulates per-record into a global
    Counter. Dipeptides are consecutive overlapping pairs within each sequence
    (matching Kourosh's Counter(seq[i:i+2] ...) implementation). Pairs containing
    any non-standard residue are skipped.
    """
    dp = Counter()
    n_seqs = 0
    n_res = 0
    seq_parts = []

    def flush(parts):
        nonlocal n_res
        if not parts:
            return
        seq = "".join(parts)
        n_res_local = len(seq)
        for i in range(n_res_local - 1):
            a, b = seq[i], seq[i + 1]
            if a in AA_SET and b in AA_SET:
                dp[a + b] += 1
        return n_res_local

    with open(fpath, "r") as fh:
        for line in fh:
            if line.startswith(">"):
                if seq_parts:
                    r = flush(seq_parts)
                    n_res += r if r else 0
                    n_seqs += 1
                    seq_parts = []
            else:
                seq_parts.append(line.strip())
        if seq_parts:
            r = flush(seq_parts)
            n_res += r if r else 0
            n_seqs += 1

    return dp, n_seqs, n_res


def compute_oe(dp_counts: Counter):
    """Build the 20x20 O/E matrix and the flat list of 400 O/E ratios.

    Single-AA frequencies derived from dipeptide counts, matching the original.
    """
    total_dp = sum(dp_counts.values())
    aa_freq = Counter()
    for d, cnt in dp_counts.items():
        if len(d) == 2 and d[0] in AA_SET and d[1] in AA_SET:
            aa_freq[d[0]] += cnt
            aa_freq[d[1]] += cnt
    total_aa = sum(aa_freq.values())
    aa_frac = {aa: (aa_freq[aa] / total_aa if total_aa else 0.0) for aa in AA_ORDER}

    oe = np.ones((20, 20))
    oe_flat = []
    for i, a1 in enumerate(AA_ORDER):
        for j, a2 in enumerate(AA_ORDER):
            observed = dp_counts.get(a1 + a2, 0)
            expected = aa_frac[a1] * aa_frac[a2] * total_dp
            ratio = (observed / expected) if expected > 0 else np.nan
            oe[i, j] = ratio
            oe_flat.append(ratio)
    return oe, np.array(oe_flat, dtype=float), total_dp, aa_frac


def bias_stats(oe_flat: np.ndarray):
    """Quantify spread of dipeptide bias (the 'more uniform' effect size).

    Uses log2(O/E) so over- and under-representation are symmetric.
    """
    vals = oe_flat[np.isfinite(oe_flat) & (oe_flat > 0)]
    log2oe = np.log2(vals)
    q25, q75 = np.percentile(log2oe, [25, 75])
    n_strong = int(np.sum(np.abs(log2oe) > 0.5))  # >1.41-fold over/under
    return {
        "n_dipeptides": int(vals.size),
        "log2oe_sd": float(np.std(log2oe, ddof=1)),
        "log2oe_iqr": float(q75 - q25),
        "log2oe_mean_abs": float(np.mean(np.abs(log2oe))),
        "n_strong_bias": n_strong,
        "frac_strong_bias": float(n_strong / vals.size),
        "oe_min": float(np.min(vals)),
        "oe_max": float(np.max(vals)),
    }


def main():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    script_abs = os.path.abspath(__file__)

    # Verify all inputs exist (data-integrity: real files only)
    for label, fpath in DATASETS:
        if not fpath.exists():
            sys.exit(f"ERROR: missing input FASTA for '{label}': {fpath}")

    results = {}
    matrices = {}
    for label, fpath in DATASETS:
        print(f"[{label}] streaming {fpath} ...", flush=True)
        dp, n_seqs, n_res = stream_dipeptide_counts(fpath)
        oe, oe_flat, total_dp, aa_frac = compute_oe(dp)
        stats = bias_stats(oe_flat)
        stats.update({"n_seqs": n_seqs, "n_residues": n_res,
                      "total_dipeptides": total_dp})
        results[label] = stats
        matrices[label] = oe
        print(f"  {label}: n_seq={n_seqs:,} total_dp={total_dp:,} "
              f"log2OE_SD={stats['log2oe_sd']:.4f} "
              f"IQR={stats['log2oe_iqr']:.4f} "
              f"strong={stats['n_strong_bias']}/400", flush=True)

    # --- Write summary TSV with provenance ---
    summary_path = OUT_DIR / "dipeptide_oe_summary.tsv"
    inputs = "; ".join(str(p) for _, p in DATASETS)
    with open(summary_path, "w") as out:
        out.write("# Provenance:\n")
        out.write(f"#   Script: {script_abs}\n")
        out.write(f"#   Input:  {inputs}\n")
        out.write(f"#   Date:   {now}\n")
        out.write(f"#   Host:   {socket.gethostname()}\n")
        out.write("#   Method: dipeptide O/E per Kourosh analyze_proteomes Fig14; "
                  "expected=f(AA1)*f(AA2)*total_dp; spread = SD/IQR of log2(O/E) "
                  "over 400 dipeptides; strong bias = |log2(O/E)|>0.5 (>1.41-fold)\n")
        out.write("#   Integrity Check: PASSED (real FASTAs, streamed)\n")
        cols = ["dataset", "n_seqs", "n_residues", "total_dipeptides",
                "n_dipeptides", "log2oe_sd", "log2oe_iqr", "log2oe_mean_abs",
                "n_strong_bias", "frac_strong_bias", "oe_min", "oe_max"]
        out.write("\t".join(cols) + "\n")
        for label, _ in DATASETS:
            s = results[label]
            row = [label] + [f"{s[c]:.6g}" if isinstance(s[c], float) else str(s[c])
                             for c in cols[1:]]
            out.write("\t".join(row) + "\n")
    print(f"\nWrote {summary_path}")

    # --- Write full 20x20 O/E matrices (one TSV per proteome) ---
    for label, _ in DATASETS:
        mpath = OUT_DIR / f"dipeptide_oe_matrix_{label}.tsv"
        oe = matrices[label]
        with open(mpath, "w") as out:
            out.write(f"# Provenance: {script_abs} | {now} | {socket.gethostname()}\n")
            out.write("# rows=first AA, cols=second AA, values=observed/expected\n")
            out.write("first_aa\t" + "\t".join(AA_ORDER) + "\n")
            for i, a1 in enumerate(AA_ORDER):
                out.write(a1 + "\t" + "\t".join(f"{oe[i, j]:.6g}" for j in range(20)) + "\n")
        print(f"Wrote {mpath}")


if __name__ == "__main__":
    main()
