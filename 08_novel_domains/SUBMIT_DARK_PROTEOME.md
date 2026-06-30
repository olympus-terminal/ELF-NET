# Dark Proteome Extraction — Jubail Submission Guide

## Overview

This guide covers Steps 0-2 of the novel domain discovery pipeline:
extracting the Pfam-dark proteome (proteins with zero Pfam-A annotation)
from the 2,318 TARA-OMEN metagenome assemblies.

**Goal:** Produce per-sample FASTA files of unannotated proteins, then
concatenate into a single `dark_proteome_all.fa` for downstream clustering.

**Estimated wall time:** ~1-2 hours for Step 0, ~2-4 hours for Steps 1-2
(2,318 array jobs each, throttled at 100 concurrent).

**Estimated output size:** Expect ~60-100 GB total in `dark_fasta/` based on
~70% of 221.9M proteins being Pfam-dark. Individual per-sample files will
vary from a few KB to several hundred MB depending on assembly size.

---

## Prerequisites

- SSH access to Jubail (`ssh jubail`)
- Project directory exists: `/scratch/drn2/PROJECTS/TARA-LA4SR`
- Manuscript repo synced: `MANUSCRIPT/scripts/novel_domains/` contains pipeline scripts
- Pfam hmmsearch results exist in `03_analyses/hmmsearch_results/*.aa.hmmsearch.tbl`
- Algal protein FASTA files exist in `03_analyses/algae_proteins/*.fa`

---

## Step-by-Step Instructions

### 1. Connect to Jubail and navigate to project

```bash
ssh jubail
cd /scratch/drn2/PROJECTS/TARA-LA4SR
```

### 2. Create log directory

```bash
mkdir -p logs/novel_domains
```

### 3. Run Step 0: HPC audit (interactive)

Run on the login node to verify all data is in place:

```bash
bash MANUSCRIPT/scripts/novel_domains/00_audit_hpc.sh 2>&1 | tee audit_novel_domains_$(date +%Y%m%d_%H%M%S).log
```

**What to check in the output:**

- `.tbl file count` should be ~2318 (one per sample)
- `.fa file count` should be ~2318
- `Samples with both .tbl and .fa` should match the total sample count
- `sample_list.txt` was generated at `03_analyses/novel_domains/sample_list.txt`
- Software check: `seqtk`, `mmseqs`, `hmmsearch`, `mafft` all found
- Disk space: verify at least ~200 GB free on `/scratch/drn2/`

**If .tbl files are zero or missing:** The Pfam hmmsearch has not been run for
some/all samples. Step 1 will treat those samples as 100% dark (all proteins
unannotated). This is noted in the stats output.

**Verify sample list:**

```bash
wc -l 03_analyses/novel_domains/sample_list.txt
# Expected: 2318
head -5 03_analyses/novel_domains/sample_list.txt
```

### 4. Submit Step 1: Extract dark protein IDs

```bash
sbatch MANUSCRIPT/scripts/novel_domains/01_extract_dark_ids.sbatch
```

This submits a 2,318-task array job (throttled at 100 concurrent). Each task:
- Reads one sample's FASTA and hmmsearch `.tbl`
- Computes set difference: all protein IDs minus Pfam-hit IDs
- Writes `dark_ids/{sample}.dark_ids.txt` and `dark_ids/{sample}.dark_stats.tsv`

**Monitor progress:**

```bash
squeue -u $(whoami) -n dark_ids
sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS -X | tail -20
```

**When complete, spot-check:**

```bash
# Count output files
ls 03_analyses/novel_domains/dark_ids/*.dark_ids.txt | wc -l
# Expected: 2318

# Check stats for a few samples
head 03_analyses/novel_domains/dark_ids/*.dark_stats.tsv | head -30

# Aggregate dark fraction across all samples
cat 03_analyses/novel_domains/dark_ids/*.dark_stats.tsv | grep -v "^sample" | \
  awk -F'\t' '{total+=$2; dark+=$4} END {printf "Total proteins: %d\nDark proteins: %d\nDark fraction: %.1f%%\n", total, dark, 100*dark/total}'

# Check for any failed jobs
sacct -j <JOB_ID> --format=JobID,State -X | grep -v COMPLETED | grep -v "^---"
```

**Expected results:**
- ~70% of proteins should be Pfam-dark (based on LA4SR precedent)
- All 2318 stats files should exist
- Stats header: `sample  total_seqs  pfam_hit_seqs  dark_seqs  pct_dark  note`

### 5. Submit Step 2: Extract dark FASTA sequences

```bash
sbatch MANUSCRIPT/scripts/novel_domains/02_extract_dark_fasta.sbatch
```

This submits another 2,318-task array job. Each task:
- Reads the dark ID list from Step 1
- Extracts matching sequences from the sample FASTA using `seqtk subseq`
- Writes `dark_fasta/{sample}.dark.fa`

**Monitor and spot-check (same as Step 1):**

```bash
squeue -u $(whoami) -n dark_fasta

# When complete:
ls 03_analyses/novel_domains/dark_fasta/*.dark.fa | wc -l
# Expected: 2318

# Check total size
du -sh 03_analyses/novel_domains/dark_fasta/

# Verify sequence counts match dark ID counts for a few samples
for f in $(ls 03_analyses/novel_domains/dark_fasta/*.dark.fa | head -5); do
  SAMPLE=$(basename "$f" .dark.fa)
  N_IDS=$(wc -l < "03_analyses/novel_domains/dark_ids/${SAMPLE}.dark_ids.txt")
  N_FA=$(grep -c "^>" "$f" 2>/dev/null || echo 0)
  echo "${SAMPLE}: IDs=${N_IDS} FASTA=${N_FA} $([ ${N_IDS} -eq ${N_FA} ] && echo OK || echo MISMATCH)"
done
```

### 6. Concatenate dark proteome for handoff

Once Step 2 is complete, concatenate all per-sample FASTA into a single file:

```bash
cat 03_analyses/novel_domains/dark_fasta/*.dark.fa > 03_analyses/novel_domains/dark_proteome_all.fa

# Verify
grep -c "^>" 03_analyses/novel_domains/dark_proteome_all.fa
# Should be sum of all per-sample dark counts

du -sh 03_analyses/novel_domains/dark_proteome_all.fa
```

### 7. Generate summary statistics

```bash
# Concatenate all per-sample stats into one file
head -1 $(ls 03_analyses/novel_domains/dark_ids/*.dark_stats.tsv | head -1) \
  > 03_analyses/novel_domains/dark_proteome_summary_stats.tsv
cat 03_analyses/novel_domains/dark_ids/*.dark_stats.tsv | grep -v "^sample" \
  >> 03_analyses/novel_domains/dark_proteome_summary_stats.tsv

wc -l 03_analyses/novel_domains/dark_proteome_summary_stats.tsv
# Expected: 2319 (header + 2318 samples)
```

---

## What to Hand Off

| File | Description |
|------|-------------|
| `03_analyses/novel_domains/dark_proteome_all.fa` | Concatenated dark proteome FASTA (all samples) |
| `03_analyses/novel_domains/dark_proteome_summary_stats.tsv` | Per-sample statistics (total, annotated, dark, % dark) |
| `03_analyses/novel_domains/dark_fasta/` | Per-sample dark FASTA files (2318 files) |
| `03_analyses/novel_domains/dark_ids/` | Per-sample dark protein ID lists (2318 files) |
| Audit log | `audit_novel_domains_*.log` in project root |

---

## Troubleshooting

**Array jobs stuck in PENDING:** Check `squeue -u $(whoami)` — jobs may be
queued behind other work. The `%100` throttle limits concurrency.

**Individual tasks failed:** Check logs:
```bash
ls logs/novel_domains/01_dark_ids_<JOB_ID>_*.err | head
cat logs/novel_domains/01_dark_ids_<JOB_ID>_<TASK_ID>.err
```
Common causes: missing FASTA for a sample, disk quota exceeded.

**Resubmit failed tasks only:**
```bash
# Find failed task IDs
sacct -j <JOB_ID> --format=JobID,State -X | grep FAILED | awk -F'_' '{print $2}'
# Resubmit specific tasks:
sbatch --array=<comma-separated-task-IDs> MANUSCRIPT/scripts/novel_domains/01_extract_dark_ids.sbatch
```

**seqtk not found in Step 2:** The script falls back to a Python extractor
automatically. Performance will be slower but results are identical.

---

## Next Steps (Phase B — not yet)

After dark proteome extraction is verified, the remaining pipeline steps are:
- Step 3: Filter by length (>=50 aa) and low-complexity, concatenate
- Step 4: MMseqs2 clustering at 30% and 50% identity
- Steps 5-8: HMM building, searching, characterization, environmental correlation

These are submitted via `submit_pipeline.sh` (the full orchestrator) or
individually. See `specs/pipeline_spec.md` for details.
