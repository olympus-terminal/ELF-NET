# Zenodo Deposit Checklist

**Date:** 2026-05-01 (supersedes 2026-04-06 version)
**Purpose:** Track all ELF-NET Data S1–S8 Zenodo deposits: status, filenames, and outstanding actions.

---

## Current State (verified 2026-05-01 via API)

All 8 deposits are **PUBLISHED**. New API token created 2026-05-01 (`zenodo-elfnet-api-key.txt`).

| Label | Deposit ID | DOI | Title Correct? | Files OK? | Notes |
|:---|:---|:---|:---|:---|:---|
| Data S1 | 18538439 (v1), 19342647 (v2) | `10.5281/zenodo.18538439` | v1: old title, v2: ✅ | ✅ 36 files, 1.15 GB | Manuscript cites v1 DOI; data identical across versions |
| Data S2 | 18728837 (v1), 19342648 (v2) | `10.5281/zenodo.18728837` | v1: old title, v2: ✅ | ✅ 1 archive, 25.8 GB | Manuscript cites v1 DOI; data identical across versions |
| Data S3 | 18786753 | `10.5281/zenodo.18786753` | ✅ | ⚠️ **Filename mismatch** | Archive named `DataS4_pfam_hmmsearch_results.tar.gz` (should be `DataS3_`) |
| Data S4 | 19514429 (v2.0) | `10.5281/zenodo.18786775` (concept) | ✅ | ✅ 2 files, 5.1 MB | Consolidated package; correct naming |
| Data S5 | 18786762 | `10.5281/zenodo.18786762` | ✅ | ✅ 1 TSV, 1.5 MB | Descriptive filename (no Sx prefix) — acceptable |
| Data S6 | 18786767 | `10.5281/zenodo.18786767` | ✅ | ⚠️ **Filename mismatch** | Archive named `DataS8_kan_cca_sparse_cca_results.tar.gz` (should be `DataS6_`) |
| Data S7 | 19493371 (v2) | `10.5281/zenodo.18786771` (concept) | ✅ | ⚠️ **Filename mismatch** | First archive named `data_s9_novel_domains_CLEAN_20260316.tar.gz` (should be `DataS7_`); second file `DataS7_structure_predictions_20260409.tar.gz` is correct |
| Data S8 | 19441356 | `10.5281/zenodo.19441356` | ✅ | ✅ 4 files, 454 KB | Descriptive filenames — fine |

### Deprecated Deposits (still visible on Zenodo, not deletable via API)

| Old Label | Deposit ID | Status | Notes |
|:---|:---|:---|:---|
| Data S4a (Form I) | 18786757 | Published, superseded | File: `DataS5_formI_rubisco_per_sample.tar.gz` (old name) |
| Data S4b (Form II) | 18786760 | Published, superseded | File: `DataS6_formII_rubisco_full_proteome.tar.gz` (old name) |
| Data S4c (HMMs) | 18786776 | Published, superseded | File: `DataS10_rubisco_hmm_profiles_and_alignments.tar.gz` (old name) |
| Al-Khidr | 18728839 | Published, removed from manuscript | Should be deleted via Zenodo web UI if possible |

---

## Outstanding Actions

### 1. Fix archive filenames on S3, S6, S7 (DONE — verified 2026-05-07)

All three were already fixed in their latest published versions:
- S3 (19948238): `DataS3_pfam_hmmsearch_results.tar.gz` ✅
- S6 (19948225): `DataS6_kan_cca_sparse_cca_results.tar.gz` ✅
- S7 (19948349): `DataS7_novel_domains_20260316.tar.gz` + `DataS7_structure_predictions_20260409.tar.gz` ✅

Old versions (18786753, 18786767, 19493371) still have wrong names but are superseded by concept DOI resolution. S3 v1 (18786753) title also fixed via API ("metatranscriptome" → "metagenome").

### 2. Update S1/S2 titles on Zenodo (DONE 2026-05-07)

Fixed titles on the v1 deposits via API metadata edit (no new version needed):
- S1 (18538439): "Marine microalgal..." → "ELF-NET Data S1: Domain-environment association and modeling results"
- S2 (18728837): "...metatranscriptome assemblies" → "ELF-NET Data S2: algaGPT-purified algal protein sequences from TARA-Oceans metagenome assemblies"

### 3. (Optional) Delete Al-Khidr deposit via Zenodo web UI

Deposit 18728839 — removed from manuscript, still visible.

### 4. SI terminology fix (DONE 2026-05-01)

Fixed 3 instances of "metatranscriptome" → "metagenome" in supplemental dataset descriptions (lines 791, 795, 823 of `supplemental_information.tex`).

### 5. Bibliography DOI + title fixes (DONE 2026-05-07)

- Fixed 4 wrong DOIs in `main.bbl`: S3 (18786753→18786751), S4 (18786776→18786775), S6 (18786767→18786766), S7 (18786772→18786771)
- Added missing S8 (`zenodo_diamond_blastp`) entry to `main.bbl`
- Removed 3 orphaned bbl entries: `zenodo_alkhidr_seqs`, `zenodo_formi_hmmsearch`, `zenodo_formii_hmmsearch`
- Fixed "metatranscriptome"→"metagenome" in S2 bbl entry and `references.bib`
- Updated all bbl titles to `{ELF-NET} Data {Sx}:` prefix format
- Fixed natexlab letter sequence (a–i)
- Fixed S7 enrichment fold "2.3-fold"→"2.29-fold" in `supplemental_information.tex`
- Harmonized S7 "related to" reference: added Figure~S5 to main.tex Data S7 description

---

## Numbering History

1. **Original (2026-02)**: S1–S10 script numbering, did not match manuscript
2. **First fix (2026-04-06)**: S1–S8 with S4a/b/c for RuBisCO. Al-Khidr was Data S3.
3. **Al-Khidr removal + renumber (2026-04-06)**: S4→S3, S5→S4, etc.
4. **S4 consolidation (2026-04-11)**: S4a/S4b/S4c merged into single S4 deposit (19514429 v2.0)
5. **S8 creation (2026-04-06)**: DIAMOND BLASTp added as Data S8 (19441356)
6. **Title fixes (2026-04-06)**: S1/S2 new versions published with `ELF-NET Data Sx:` prefix titles

## Management Scripts

| Script | Purpose |
|:---|:---|
| `scripts/zenodo_fix_filenames_20260501.sh` | Fix mismatched filenames on S3/S6/S7 |
| `scripts/zenodo_manage.sh` | Check status, publish, create new versions |
| `scripts/zenodo_update_titles.sh` | Fix titles (mostly done, may be stale) |
| `scripts/zenodo_update_descriptions.sh` | Push HTML descriptions |
| `scripts/zenodo_upload_s4_s10.sh` | Original upload script (old naming) |
