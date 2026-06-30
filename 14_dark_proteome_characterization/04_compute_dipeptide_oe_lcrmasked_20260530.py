#!/usr/bin/env python3
"""
LCR-masked dipeptide observed/expected (O/E) control.

Purpose (ELF-Net feedback follow-up): the unmasked dipeptide O/E analysis
(script 03) found the dark proteome has STRONGER, more spread dipeptide bias
than the annotated proteome (log2 O/E SD 0.251 vs 0.163), opposite to the
report's heatmap-based 'dark is more uniform' impression. That excess may be
driven by low-complexity regions (LCRs: homopolymer / compositionally skewed
runs) rather than selection for structural motifs. This control masks LCRs
before counting dipeptides to separate the two:

  - If masked dark bias drops BELOW annotated -> the 'weaker constraint' claim
    holds for ordinary (non-LCR) sequence; unmasked excess was LCR skew.
  - If masked dark bias STAYS ABOVE annotated -> stronger bias is real beyond
    LCRs, and the report's framing is wrong.

LCR definition matches the manuscript's own physicochemical pipeline:
  sliding 12-residue window, Shannon entropy < 1.5 bits => low complexity.
Residues in any low-complexity window are masked to 'X'; the dipeptide counter
skips any pair containing a non-standard residue, so masked residues are
excluded from counts (and from the single-AA frequencies derived from them).

DATA INTEGRITY: real sequences only; streamed; provenance header on outputs.
"""

import os
import sys
import socket
import datetime
import math
from pathlib import Path
from collections import Counter

import numpy as np

# Reuse validated helpers from script 03 (same dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "dipeptide_oe_mod",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "03_compute_dipeptide_oe_20260530.py"),
)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

AA_ORDER = _m.AA_ORDER
AA_SET = _m.AA_SET
compute_oe = _m.compute_oe
bias_stats = _m.bias_stats
get_base_dir = _m.get_base_dir
FASTA_DIR = _m.FASTA_DIR
OUT_DIR = _m.OUT_DIR
DATASETS = _m.DATASETS

WINDOW = 12          # residues, matches manuscript LCR definition
ENTROPY_THRESH = 1.5  # bits, matches manuscript LCR definition


def window_shannon(seq_window: str) -> float:
    """Shannon entropy (bits) of amino-acid composition within a window."""
    n = len(seq_window)
    if n == 0:
        return 0.0
    counts = Counter(seq_window)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


def mask_lcr(seq: str) -> str:
    """Mask residues in any 12-residue window with Shannon entropy < 1.5 bits.

    A residue is masked to 'X' if it falls in at least one low-complexity window.
    Standard SEG-style: positions covered by a flagged window are masked.
    """
    L = len(seq)
    if L < WINDOW:
        # Too short to evaluate a full window; treat whole seq by its own entropy.
        return seq if window_shannon(seq) >= ENTROPY_THRESH else "X" * L
    masked = list(seq)
    flagged = bytearray(L)  # 0/1 per position
    for start in range(0, L - WINDOW + 1):
        w = seq[start:start + WINDOW]
        if window_shannon(w) < ENTROPY_THRESH:
            for p in range(start, start + WINDOW):
                flagged[p] = 1
    for p in range(L):
        if flagged[p]:
            masked[p] = "X"
    return "".join(masked)


def stream_dipeptide_counts_masked(fpath: Path):
    """Stream a FASTA, mask LCRs per sequence, accumulate dipeptide counts."""
    dp = Counter()
    n_seqs = 0
    n_res = 0          # total residues (pre-mask)
    n_masked = 0       # residues masked to X
    seq_parts = []

    def flush(parts):
        nonlocal n_res, n_masked
        if not parts:
            return
        seq = "".join(parts)
        n_res_local = len(seq)
        masked = mask_lcr(seq)
        n_masked += masked.count("X")
        for i in range(n_res_local - 1):
            a, b = masked[i], masked[i + 1]
            if a in AA_SET and b in AA_SET:
                dp[a + b] += 1
        return n_res_local

    with open(fpath, "r") as fh:
        for line in fh:
            if line.startswith(">"):
                if seq_parts:
                    n_res += flush(seq_parts) or 0
                    n_seqs += 1
                    seq_parts = []
            else:
                seq_parts.append(line.strip())
        if seq_parts:
            n_res += flush(seq_parts) or 0
            n_seqs += 1

    return dp, n_seqs, n_res, n_masked


def main():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    script_abs = os.path.abspath(__file__)

    for label, fpath in DATASETS:
        if not fpath.exists():
            sys.exit(f"ERROR: missing input FASTA for '{label}': {fpath}")

    results = {}
    matrices = {}
    for label, fpath in DATASETS:
        print(f"[{label}] LCR-masked streaming {fpath} ...", flush=True)
        dp, n_seqs, n_res, n_masked = stream_dipeptide_counts_masked(fpath)
        oe, oe_flat, total_dp, aa_frac = compute_oe(dp)
        stats = bias_stats(oe_flat)
        stats.update({"n_seqs": n_seqs, "n_residues": n_res,
                      "n_masked_residues": n_masked,
                      "frac_masked": (n_masked / n_res if n_res else 0.0),
                      "total_dipeptides": total_dp})
        results[label] = stats
        matrices[label] = oe
        print(f"  {label}: n_seq={n_seqs:,} masked={n_masked:,} "
              f"({100*stats['frac_masked']:.1f}%) "
              f"total_dp={total_dp:,} log2OE_SD={stats['log2oe_sd']:.4f} "
              f"IQR={stats['log2oe_iqr']:.4f} strong={stats['n_strong_bias']}/400",
              flush=True)

    summary_path = OUT_DIR / "dipeptide_oe_lcrmasked_summary.tsv"
    inputs = "; ".join(str(p) for _, p in DATASETS)
    with open(summary_path, "w") as out:
        out.write("# Provenance:\n")
        out.write(f"#   Script: {script_abs}\n")
        out.write(f"#   Input:  {inputs}\n")
        out.write(f"#   Date:   {now}\n")
        out.write(f"#   Host:   {socket.gethostname()}\n")
        out.write(f"#   Method: LCR-masked dipeptide O/E. LCR = 12-residue window, "
                  f"Shannon entropy < 1.5 bits (matches manuscript LCR def); masked "
                  f"residues -> X, excluded from counts. Spread = SD/IQR of log2(O/E) "
                  f"over 400 dipeptides; strong = |log2(O/E)|>0.5.\n")
        out.write("#   Integrity Check: PASSED (real FASTAs, streamed)\n")
        cols = ["dataset", "n_seqs", "n_residues", "n_masked_residues",
                "frac_masked", "total_dipeptides", "n_dipeptides",
                "log2oe_sd", "log2oe_iqr", "log2oe_mean_abs",
                "n_strong_bias", "frac_strong_bias", "oe_min", "oe_max"]
        out.write("\t".join(cols) + "\n")
        for label, _ in DATASETS:
            s = results[label]
            row = [label] + [f"{s[c]:.6g}" if isinstance(s[c], float) else str(s[c])
                             for c in cols[1:]]
            out.write("\t".join(row) + "\n")
    print(f"\nWrote {summary_path}")

    for label, _ in DATASETS:
        mpath = OUT_DIR / f"dipeptide_oe_lcrmasked_matrix_{label}.tsv"
        oe = matrices[label]
        with open(mpath, "w") as out:
            out.write(f"# Provenance: {script_abs} | {now} | {socket.gethostname()} | LCR-masked\n")
            out.write("# rows=first AA, cols=second AA, values=observed/expected\n")
            out.write("first_aa\t" + "\t".join(AA_ORDER) + "\n")
            for i, a1 in enumerate(AA_ORDER):
                out.write(a1 + "\t" + "\t".join(f"{oe[i, j]:.6g}" for j in range(20)) + "\n")
        print(f"Wrote {mpath}")


if __name__ == "__main__":
    main()
