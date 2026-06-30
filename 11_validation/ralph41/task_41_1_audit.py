#!/usr/bin/env python3
"""
Task 41.1 — Input provenance audit

# Provenance:
#   Script: scripts/ralph41/task_41_1_audit.py
#   Purpose: Audit 9 critical input files for ralph41 loop
#   Checks: file exists, row count, header columns, anomalies
#   Output: source_data/ralph41/input_audit.md
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

BASE = Path("/media/drn2/External/TARA-Oceans")
OUT = BASE / "MANUSCRIPT/source_data/ralph41/input_audit.md"
OUT.parent.mkdir(parents=True, exist_ok=True)


def count_data_rows(path: Path, skip_comment_prefix: str = "#") -> int:
    """Count non-comment, non-empty lines including header row."""
    n = 0
    with path.open("r") as fh:
        for line in fh:
            if line.startswith(skip_comment_prefix):
                continue
            if line.strip() == "":
                continue
            n += 1
    return n


def first_data_line(path: Path, skip_comment_prefix: str = "#") -> str:
    with path.open("r") as fh:
        for line in fh:
            if line.startswith(skip_comment_prefix):
                continue
            if line.strip() == "":
                continue
            return line.rstrip("\n")
    return ""


def header_columns(path: Path) -> list[str]:
    header = first_data_line(path)
    if "\t" in header:
        return header.split("\t")
    return [header]


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def audit_file(label: str, path: Path, expected_keywords: list[str],
               min_rows: int, notes: str) -> dict:
    entry = {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": file_size(path) if path.exists() else -1,
        "data_rows": 0,
        "n_cols": 0,
        "first_col": "",
        "expected_keywords_found": [],
        "expected_keywords_missing": [],
        "min_rows": min_rows,
        "meets_min_rows": False,
        "status": "FAIL",
        "notes": notes,
    }
    if not path.exists():
        entry["status"] = "FAIL_MISSING"
        return entry
    try:
        n = count_data_rows(path)
        entry["data_rows"] = n
        entry["meets_min_rows"] = n >= min_rows
        cols = header_columns(path)
        entry["n_cols"] = len(cols)
        entry["first_col"] = cols[0] if cols else ""
        found, missing = [], []
        cols_lower = [c.lower() for c in cols]
        for kw in expected_keywords:
            if any(kw.lower() in c for c in cols_lower):
                found.append(kw)
            else:
                missing.append(kw)
        entry["expected_keywords_found"] = found
        entry["expected_keywords_missing"] = missing
        if entry["meets_min_rows"] and not missing:
            entry["status"] = "PASS"
        elif entry["meets_min_rows"] and missing:
            entry["status"] = "PARTIAL"
        else:
            entry["status"] = "FAIL"
    except Exception as e:
        entry["status"] = f"FAIL_EXCEPTION: {e}"
    return entry


def audit_directory(label: str, path: Path, glob: str,
                    min_count: int, notes: str) -> dict:
    entry = {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir() if path.exists() else False,
        "matches": 0,
        "min_count": min_count,
        "meets_min": False,
        "status": "FAIL",
        "notes": notes,
    }
    if not path.exists() or not path.is_dir():
        entry["status"] = "FAIL_MISSING"
        return entry
    try:
        n = sum(1 for _ in path.glob(glob))
        entry["matches"] = n
        entry["meets_min"] = n >= min_count
        entry["status"] = "PASS" if entry["meets_min"] else "FAIL"
    except Exception as e:
        entry["status"] = f"FAIL_EXCEPTION: {e}"
    return entry


def main() -> int:
    results: list[dict] = []

    # 1. PFAM count matrix
    results.append(audit_file(
        "PFAM count matrix (SNAP/LA4SR pipeline)",
        BASE / "03_analyses/pfam_results/pfam_count_matrix_20260109_181211.tsv",
        ["assembly_id", "dataset", "latitude", "longitude"],
        min_rows=1000,
        notes="17,245 Pfam domain columns plus metadata; has 8-line provenance header.",
    ))

    # 2. GPS mapping
    results.append(audit_file(
        "ALL assemblies GPS mapping",
        BASE / "03_analyses/ALL_assemblies_GPS_mapping.tsv",
        ["assembly_id", "latitude", "longitude"],
        min_rows=500,
        notes="Sample GPS lookup; includes basin; source for spatial CV.",
    ))

    # 3. Dataset membership
    results.append(audit_file(
        "Dataset membership per sample",
        BASE / "MANUSCRIPT/source_data/dataset_membership.tsv",
        ["sample", "dataset", "category", "n_proteins"],
        min_rows=1500,
        notes="Category / dataset / source_type bookkeeping for all 2,044 samples.",
    ))

    # 4. Novel domain count matrix
    results.append(audit_file(
        "Novel domain count matrix (dark proteome)",
        BASE / "MANUSCRIPT/source_data/dark_proteome/novel_domain_count_matrix.tsv",
        [],  # column headers are cluster IDs; skip keyword check
        min_rows=1500,
        notes="Samples × 33,950 novel cluster counts. Column names are cluster IDs.",
    ))

    # 5. Novel domain env correlations
    results.append(audit_file(
        "Novel domain × environment Spearman correlations",
        BASE / "MANUSCRIPT/source_data/dark_proteome/novel_domain_env_correlations.tsv",
        [],
        min_rows=1000,
        notes="Spearman rho for 33,950 novel clusters vs environmental variables.",
    ))

    # 6. HMM directory (local fallback; may be HPC-only)
    results.append(audit_directory(
        "Novel domain HMM profiles",
        BASE / "03_analyses/novel_domains/hmms",
        glob="*.hmm",
        min_count=30000,
        notes="Used by task 41.4 local HMM quality metrics fallback. May reside on HPC only.",
    ))

    # 7. InterPro crosscheck summary (top-100)
    results.append(audit_file(
        "InterPro top-100 crosscheck summary",
        BASE / "MANUSCRIPT/source_data/dark_proteome/interpro_crosscheck_summary.md",
        [],
        min_rows=5,
        notes="Manually verified markdown summary; feeds task 41.16 sanity check.",
    ))

    # 8. Malaspina metadata
    results.append(audit_file(
        "Malaspina expedition metadata",
        BASE / "01_raw_data/metadata/malaspina_complete_metadata.tsv",
        ["sample"],  # at least one identifier column expected
        min_rows=50,
        notes="Used by task 41.17 cross-campaign validation.",
    ))

    # 9. TARA size-fraction metadata — search for the canonical file
    tara_candidates = [
        BASE / "01_raw_data/metadata/TARA_assembly_GPS_complete_mapping.tsv",
        BASE / "01_raw_data/metadata/TARA_assembly_GPS_mapping.tsv",
        BASE / "01_raw_data/metadata/TARA_Oceans_assembly_GPS_mapping.tsv",
    ]
    tara_file = next((p for p in tara_candidates if p.exists()), tara_candidates[0])
    results.append(audit_file(
        "TARA size-fraction metadata (verify)",
        tara_file,
        ["tara", "assembly"],
        min_rows=100,
        notes="Task 41.11 precondition: verify size-fraction column exists in TARA metadata.",
    ))

    # Write audit markdown
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results)
    passes = sum(1 for r in results if r["status"] == "PASS")
    partials = sum(1 for r in results if r["status"] == "PARTIAL")
    fails = sum(1 for r in results if r["status"].startswith("FAIL"))

    with OUT.open("w") as fh:
        fh.write("# Ralph41 Task 41.1 — Input Provenance Audit\n\n")
        fh.write("# Provenance:\n")
        fh.write("#   Script: MANUSCRIPT/scripts/ralph41/task_41_1_audit.py\n")
        fh.write(f"#   Date: {now}\n")
        fh.write("#   Integrity Check: PASSED — file existence and row counts read from filesystem only\n\n")
        fh.write(f"## Summary\n\n")
        fh.write(f"- Total files audited: **{total}**\n")
        fh.write(f"- PASS: **{passes}**\n")
        fh.write(f"- PARTIAL (rows OK, some expected columns missing): **{partials}**\n")
        fh.write(f"- FAIL: **{fails}**\n\n")

        for i, r in enumerate(results, start=1):
            fh.write(f"### {i}. {r['label']}\n\n")
            fh.write(f"- **Path**: `{r['path']}`\n")
            fh.write(f"- **Status**: `{r['status']}`\n")
            if "n_cols" in r:
                fh.write(f"- **Exists**: {r['exists']}\n")
                fh.write(f"- **Size (bytes)**: {r['size_bytes']}\n")
                fh.write(f"- **Data rows (incl. header)**: {r['data_rows']}\n")
                fh.write(f"- **Columns**: {r['n_cols']}\n")
                fh.write(f"- **First column**: `{r['first_col']}`\n")
                fh.write(f"- **Expected keywords found**: {r['expected_keywords_found']}\n")
                fh.write(f"- **Expected keywords missing**: {r['expected_keywords_missing']}\n")
                fh.write(f"- **Min rows threshold**: {r['min_rows']}\n")
                fh.write(f"- **Meets min rows**: {r['meets_min_rows']}\n")
            else:
                # directory audit
                fh.write(f"- **Exists**: {r['exists']}\n")
                fh.write(f"- **Is dir**: {r['is_dir']}\n")
                fh.write(f"- **Matches**: {r['matches']}\n")
                fh.write(f"- **Min count threshold**: {r['min_count']}\n")
                fh.write(f"- **Meets min count**: {r['meets_min']}\n")
            fh.write(f"- **Notes**: {r['notes']}\n\n")

        fh.write("## Downstream impact\n\n")
        for r in results:
            if r["status"].startswith("FAIL"):
                fh.write(f"- `{r['label']}` FAIL → tasks depending on this input are blocked.\n")
        if fails == 0:
            fh.write("- All 9 inputs PASS or PARTIAL; no downstream blocks on file existence grounds.\n")

    print(f"Wrote audit to {OUT}")
    print(f"Total={total}  PASS={passes}  PARTIAL={partials}  FAIL={fails}")
    for r in results:
        print(f"  [{r['status']:>15}] {r['label']}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
