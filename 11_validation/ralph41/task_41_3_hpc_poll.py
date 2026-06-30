#!/usr/bin/env python3
"""Ralph41 Task 41.3 — HPC job status poll.

Purpose
-------
Check for the presence of HPC-generated outputs needed by Phase D tasks
(41.15 SNAP vs MetaEuk, 41.16 InterProScan full, 41.17 Malaspina cross-
campaign, 41.18 LA4SR non-algal benchmark) and the HMM quality metrics
fallback (41.4). Informational only — does NOT block downstream phases.

For each candidate output file, mark as:

- READY    : file exists and size > 0 bytes
- RUNNING  : sentinel file (.running) exists in the parent directory
- MISSING  : neither file nor sentinel present

The script scans both the local mirror (`03_analyses/`) and any mounted HPC
scratch directory if present. Writes a human-readable markdown summary to
`source_data/ralph41/hpc_job_status.md`.

Provenance
----------
Script   : MANUSCRIPT/scripts/ralph41/task_41_3_hpc_poll.py
Output   : MANUSCRIPT/source_data/ralph41/hpc_job_status.md
Inputs   : filesystem scan under
           /media/drn2/External/TARA-Oceans/03_analyses/
           /media/drn2/External/TARA-Oceans/ (for mounted HPC dirs)
           /scratch/drn2/PROJECTS/TARA-LA4SR/ (if present)
Gate     : always_green (informational)
"""

from __future__ import annotations

import datetime as _dt
import os
import sys
from pathlib import Path

REPO_ROOT = Path("/media/drn2/External/TARA-Oceans")
MANUSCRIPT_ROOT = REPO_ROOT / "MANUSCRIPT"
OUTPUT_PATH = MANUSCRIPT_ROOT / "source_data/ralph41/hpc_job_status.md"

# (task_id, description, candidate file paths, parent-dir sentinels)
# Each candidate is checked in order; the first that exists as READY
# short-circuits the lookup. Sentinel check uses the directory portion of
# the LAST candidate (the canonical expected location).
CANDIDATES = [
    {
        "task_id": "41.4 / 41.5",
        "description": "HMM quality metrics (NSEQ/NEFF/LENG) for 33,950 novel domain HMMs",
        "files": [
            REPO_ROOT / "03_analyses/novel_domains/results/hmm_quality_metrics.tsv",
            REPO_ROOT / "03_analyses/novel_domains/hmm_quality_metrics.tsv",
            MANUSCRIPT_ROOT / "source_data/ralph41/hmm_quality_distribution.tsv",
        ],
        "hpc_path": "/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains/results/hmm_quality_metrics.tsv",
        "slurm_job": "14693147 (hmm_quality_metrics)",
    },
    {
        "task_id": "41.15 / 41.22",
        "description": "SNAP vs MetaEuk overlap on 20 TARA assemblies",
        "files": [
            REPO_ROOT / "03_analyses/snap_vs_metaeuk/overlap_summary.tsv",
            MANUSCRIPT_ROOT / "source_data/ralph41/snap_vs_metaeuk_overlap.tsv",
        ],
        "hpc_path": "/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/snap_vs_metaeuk/overlap_summary.tsv",
        "slurm_job": "(not yet submitted)",
    },
    {
        "task_id": "41.16 / 41.23",
        "description": "Full InterProScan output for 33,950 novel domains",
        "files": [
            REPO_ROOT / "03_analyses/novel_domains/results/interpro_full_33950.tsv",
            REPO_ROOT / "03_analyses/novel_domains/interpro_full_33950.tsv",
        ],
        "hpc_path": "/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains/results/interpro_full_33950.tsv",
        "slurm_job": "14688645 (11_interpro_api_scan.sbatch)",
    },
    {
        "task_id": "41.17 / 41.24",
        "description": "Cross-campaign Malaspina SST predictions (TARA-trained frozen model)",
        "files": [
            REPO_ROOT / "03_analyses/cross_campaign_validation/malaspina_predictions.tsv",
            MANUSCRIPT_ROOT / "source_data/ralph41/cross_campaign_malaspina.tsv",
        ],
        "hpc_path": "/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/cross_campaign_validation/malaspina_predictions.tsv",
        "slurm_job": "(not yet submitted)",
    },
    {
        "task_id": "41.18 / 41.25",
        "description": "LA4SR false-positive benchmark on 40 non-algal eukaryotes",
        "files": [
            REPO_ROOT / "03_analyses/la4sr_benchmark/results/fp_rates.tsv",
            MANUSCRIPT_ROOT / "source_data/ralph41/la4sr_benchmark_results.tsv",
        ],
        "hpc_path": "/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/la4sr_benchmark/results/fp_rates.tsv",
        "slurm_job": "(depends on 41.14 curation)",
    },
]


def poll_one(entry: dict) -> dict:
    """Return {status, hit_path, size_bytes, sentinel_path} for one candidate."""
    hit_path = None
    size = 0
    for path in entry["files"]:
        try:
            if path.is_file() and path.stat().st_size > 0:
                hit_path = path
                size = path.stat().st_size
                break
        except OSError:
            # Permission or missing-parent error — treat as missing and continue
            continue

    if hit_path is not None:
        return {
            "status": "READY",
            "hit_path": str(hit_path),
            "size_bytes": size,
            "sentinel_path": "",
        }

    # No file found → look for .running sentinel in the canonical directory
    canonical_parent = entry["files"][0].parent
    sentinel = canonical_parent / ".running"
    try:
        if sentinel.is_file():
            return {
                "status": "RUNNING",
                "hit_path": "",
                "size_bytes": 0,
                "sentinel_path": str(sentinel),
            }
    except OSError:
        pass

    return {
        "status": "MISSING",
        "hit_path": "",
        "size_bytes": 0,
        "sentinel_path": "",
    }


def write_report(results: list[tuple[dict, dict]], hpc_mount_present: bool) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Ralph41 Task 41.3 — HPC Job Status Poll")
    lines.append("")
    lines.append("<!--")
    lines.append(f"Script : MANUSCRIPT/scripts/ralph41/task_41_3_hpc_poll.py")
    lines.append(f"Output : MANUSCRIPT/source_data/ralph41/hpc_job_status.md")
    lines.append(f"Date   : {now}")
    lines.append(f"Gate   : always_green (informational)")
    lines.append(f"HPC mount detected : {'yes' if hpc_mount_present else 'no'}")
    lines.append("-->")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    ready = sum(1 for _, r in results if r["status"] == "READY")
    running = sum(1 for _, r in results if r["status"] == "RUNNING")
    missing = sum(1 for _, r in results if r["status"] == "MISSING")
    lines.append(f"- Total candidates polled: {len(results)}")
    lines.append(f"- READY   : {ready}")
    lines.append(f"- RUNNING : {running}")
    lines.append(f"- MISSING : {missing}")
    lines.append("")
    lines.append("**Informational only** — this audit does NOT block Phases B, C, or D.")
    lines.append("Phase D tasks (41.15–41.18) check for their outputs at runtime and")
    lines.append("are idempotent: if MISSING and no sentinel exists, the Phase D task")
    lines.append("will submit the corresponding HPC job on its next iteration.")
    lines.append("")
    lines.append("## Per-candidate status")
    lines.append("")
    lines.append("| Task | Description | Status | Evidence |")
    lines.append("|------|-------------|--------|----------|")
    for entry, r in results:
        if r["status"] == "READY":
            evidence = f"`{r['hit_path']}` ({r['size_bytes']:,} bytes)"
        elif r["status"] == "RUNNING":
            evidence = f"sentinel at `{r['sentinel_path']}`"
        else:
            evidence = f"no file at canonical `{entry['files'][0]}`"
        lines.append(f"| {entry['task_id']} | {entry['description']} | {r['status']} | {evidence} |")
    lines.append("")
    lines.append("## Candidate paths checked")
    lines.append("")
    for entry, r in results:
        lines.append(f"### {entry['task_id']} — {entry['description']}")
        lines.append("")
        lines.append(f"- Status: **{r['status']}**")
        lines.append(f"- HPC canonical path: `{entry['hpc_path']}`")
        lines.append(f"- SLURM job (if applicable): {entry['slurm_job']}")
        lines.append("- Local candidates (checked in order):")
        for path in entry["files"]:
            marker = " ← HIT" if str(path) == r["hit_path"] else ""
            lines.append(f"    - `{path}`{marker}")
        lines.append("")

    lines.append("## Operator notes")
    lines.append("")
    lines.append("- If HMM quality metrics remain MISSING, Task 41.4 must parse HMM")
    lines.append("  headers directly from `03_analyses/novel_domains/hmms/*.hmm` — but")
    lines.append("  that directory is itself absent locally (see input_audit.md), so")
    lines.append("  41.4 and 41.5 remain `blocked: true` until the HMM directory is")
    lines.append("  rsynced from HPC or the HPC job writes the summary TSV.")
    lines.append("- If InterProScan full results remain MISSING, Task 41.16 will")
    lines.append("  submit `scripts/novel_domains/11_interpro_api_scan.sbatch` on its")
    lines.append("  next run (or the operator may submit it manually).")
    lines.append("- SNAP vs MetaEuk (41.15), Malaspina (41.17), and LA4SR non-algal")
    lines.append("  benchmark (41.18) have no submitted SLURM jobs yet; each Phase D")
    lines.append("  task is responsible for generating and submitting its own sbatch")
    lines.append("  script on first run, writing a `.running` sentinel, and exiting.")
    lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    hpc_mount_candidates = [
        Path("/scratch/drn2/PROJECTS/TARA-LA4SR"),
        Path("/mnt/jubail/scratch/drn2/PROJECTS/TARA-LA4SR"),
    ]
    hpc_mount_present = any(p.is_dir() for p in hpc_mount_candidates)

    results: list[tuple[dict, dict]] = []
    for entry in CANDIDATES:
        results.append((entry, poll_one(entry)))

    write_report(results, hpc_mount_present)

    # Print a concise stdout summary for the operator
    for entry, r in results:
        print(f"{entry['task_id']:>14s}  {r['status']:>7s}  {entry['description']}")
    print(f"\nWrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
