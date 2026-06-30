# Novel Protein Domain Discovery Pipeline — Specification

## Context

The TARA-OMEN project has 221.9M algal protein sequences annotated against Pfam-A
via hmmsearch (E-value 1e-9), yielding 17,245 known domains. The proteins with no
Pfam-A hit — the "Pfam-dark proteome" — have never been systematically clustered.
Based on the LA4SR paper's ~70% homology-inaccessible fraction, we estimate
100–150M+ proteins may lack Pfam annotation.

## Goals

1. Quantify the dark proteome (key manuscript number)
2. Discover recurrent novel protein families in ocean metagenomes
3. Enable environmental correlation analysis on novel domains alongside known Pfam domains
4. Strengthen the manuscript's "language models access the dark proteome" argument

## Pipeline Steps

| Step | Script | Purpose | SLURM? |
|------|--------|---------|--------|
| 0 | `00_audit_hpc.sh` | Verify HPC data availability | Interactive |
| 1 | `01_extract_dark_ids.sbatch` | Extract Pfam-dark protein IDs | Array 1-2318 |
| 2 | `02_extract_dark_fasta.sbatch` | Build dark proteome FASTA files | Array 1-2318 |
| 3 | `03_filter_concat.sbatch` | Filter (length, complexity) & concatenate | Single node |
| 4 | `04_mmseqs2_cluster.sbatch` | MMseqs2 linclust at 30% and 50% identity | Single node |
| 5 | `05_build_hmms.sbatch` | Build HMM profiles from large clusters | Array |
| 5b | `05b_concat_hmms.sh` | Concatenate and press HMM database | Interactive/short |
| 6 | `06_hmmsearch_novel.sbatch` | Search novel HMMs against full dataset | Array 1-2318 |
| 6b | `06b_build_count_matrix.py` | Build count matrix from hmmsearch results | Python |
| 7 | `07_characterize.py` + `.sbatch` | Characterize novel domain families | Single node |
| 8 | `08_env_correlation.py` + `.sbatch` | Environmental correlation analysis | Single node |
| master | `submit_pipeline.sh` | Chain all jobs with SLURM dependencies | Orchestrator |

## HPC Environment (Jubail)

- Scheduler: SLURM
- Partitions: `compute`
- Module system: `module load gcc/13.2.0`
- Conda: `/scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh`
- Conda env: `/scratch/drn2/software/conda-mamba_1`
- Base project: `/scratch/drn2/PROJECTS/TARA-LA4SR`
- Analysis output: `/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains/`
- Log dir: `logs/novel_domains/`
- HOME override: `/scratch/drn2/newhome`
- TMPDIR: `/scratch/drn2/tmp`

## Software Requirements

| Tool | Purpose | Install |
|------|---------|---------|
| MMseqs2 | Clustering at scale | conda install -c bioconda mmseqs2 |
| MAFFT | MSA for HMM building | likely already available |
| HMMER | hmmbuild, hmmsearch | already used for Pfam pipeline |
| seqtk | FASTA manipulation | conda install -c bioconda seqtk |

## Input Data Paths

- Per-sample FASTA: `03_analyses/algae_proteins/*.fa` (2318 files)
- Per-sample hmmsearch tblout: `03_analyses/hmmsearch_results/*.aa.hmmsearch.tbl` (2044 files)
- DIAMOND dark proteome: `03_analyses/dark_proteome_v2/diamond_raw/*.hit_ids.txt`
- GPS metadata: `01_raw_data/metadata/ALL_assemblies_GPS_mapping.tsv`
- LA4SR classification: `03_analyses/algagpt_classification_summary_*.csv`
- Pfam count matrix: existing from main pipeline

## Output Directory Structure

```
03_analyses/novel_domains/
├── dark_ids/                    # Step 1 output
├── dark_fasta/                  # Step 2 output
├── dark_proteome_filtered.fa    # Step 3 output
├── filter_stats.tsv             # Step 3 stats
├── clusters_30/                 # Step 4 output (30% identity)
├── clusters_50/                 # Step 4 output (50% identity)
├── hmms/                        # Step 5 output
│   ├── novel_domains.hmm
│   └── per_cluster/
├── cluster_alignments/          # Step 5 MSAs
├── novel_hits/                  # Step 6 output
├── novel_domain_count_matrix.tsv  # Step 6b output
└── results/                     # Steps 7-8 output
    ├── novel_domain_summary.tsv
    ├── novel_vs_pfam_comparison.tsv
    ├── novel_domain_env_correlations.tsv
    └── novel_domain_xgboost_performance.tsv
```

## Validation Checkpoints

1. Step 0 audit confirms data availability before committing compute
2. Step 1 cross-checks: sum of dark + annotated should equal total per sample
3. Step 3 filter stats should be inspected before proceeding to clustering
4. Step 4 cluster stats: if >99% singletons, parameters need tuning
5. Step 6 hmmsearch recovery rate validates cluster quality
6. Step 8 environmental correlations can be compared to known Pfam results for sanity

## SLURM Conventions (Jubail)

Every sbatch script MUST include:
```bash
set -euo pipefail
export HOME=/scratch/drn2/newhome
export TMPDIR=/scratch/drn2/tmp
mkdir -p $TMPDIR
export PYTHONUSERBASE=/scratch/drn2/newhome/.local
module purge
module load gcc/13.2.0
source /scratch/drn2/newhome/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/drn2/software/conda-mamba_1
```

Array jobs use sample list: `03_analyses/algae_proteins/*.fa` sorted, one per line.
