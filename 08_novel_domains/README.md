# Novel Protein Domain Discovery Pipeline

Discovers recurrent novel protein families in the TARA-OMEN Pfam-dark proteome
via de novo clustering, HMM profiling, and environmental correlation analysis.

## Prerequisites

- **HPC cluster**: Jubail (NYU Abu Dhabi), SLURM scheduler
- **Conda env**: `/scratch/drn2/software/conda-mamba_1`
- **Software** (in conda env):
  - MMseqs2 (linclust clustering)
  - MAFFT (multiple sequence alignment)
  - HMMER (hmmbuild, hmmsearch, hmmpress)
  - seqtk (FASTA extraction)
- **Data**: Algal protein sequences and Pfam hmmsearch results from the
  main TARA-LA4SR pipeline in `03_analyses/`

## Pipeline Steps

| Step | Script | Description | SLURM Type |
|------|--------|-------------|------------|
| 0 | `00_audit_hpc.sh` | Verify data availability, generate sample list | Interactive |
| 1 | `01_extract_dark_ids.sbatch` | Identify Pfam-dark protein IDs per sample | Array (2318) |
| 2 | `02_extract_dark_fasta.sbatch` | Extract dark proteome FASTA sequences | Array (2318) |
| 3 | `03_filter_concat.sbatch` | Filter (length/complexity) and concatenate | Single node |
| 4 | `04_mmseqs2_cluster.sbatch` | MMseqs2 linclust at 30% and 50% identity | Single node |
| 5 | `05_build_hmms.sbatch` | Build HMM profiles from large clusters | Array (batched) |
| 5b | `05b_concat_hmms.sh` | Concatenate and press HMM database | Short/single |
| 6 | `06_hmmsearch_novel.sbatch` | Search novel HMMs against full dataset | Array (2318) |
| 7 | `07_characterize.sbatch` | Build count matrix (6b) + characterize families | Single node |
| 8 | `08_env_correlation.sbatch` | Environmental correlation + XGBoost analysis | Single node |

## Usage

### Quick Start (full pipeline)

```bash
ssh jubail
cd /scratch/drn2/PROJECTS/TARA-LA4SR
mkdir -p logs/novel_domains

# Step 0: Interactive audit (verify data, generate sample_list.txt)
bash MANUSCRIPT/scripts/novel_domains/00_audit_hpc.sh 2>&1 | tee audit_novel_domains.log

# Review audit output, then submit full pipeline:
bash MANUSCRIPT/scripts/novel_domains/submit_pipeline.sh
```

### Step-by-step (manual submission)

```bash
cd /scratch/drn2/PROJECTS/TARA-LA4SR
mkdir -p logs/novel_domains

# Step 0: Audit
bash MANUSCRIPT/scripts/novel_domains/00_audit_hpc.sh

# Step 1: Extract dark IDs
JOB1=$(sbatch --parsable MANUSCRIPT/scripts/novel_domains/01_extract_dark_ids.sbatch)

# Step 2: Extract dark FASTA (after Step 1)
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} MANUSCRIPT/scripts/novel_domains/02_extract_dark_fasta.sbatch)

# Step 3: Filter & concatenate (after Step 2)
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} MANUSCRIPT/scripts/novel_domains/03_filter_concat.sbatch)

# Step 4: MMseqs2 clustering (after Step 3)
JOB4=$(sbatch --parsable --dependency=afterok:${JOB3} MANUSCRIPT/scripts/novel_domains/04_mmseqs2_cluster.sbatch)

# Step 5: Build HMMs (after Step 4)
JOB5=$(sbatch --parsable --dependency=afterok:${JOB4} MANUSCRIPT/scripts/novel_domains/05_build_hmms.sbatch)

# Step 5b: Concatenate & press HMMs (after Step 5)
# Wait for Step 5 to complete, then run:
bash MANUSCRIPT/scripts/novel_domains/05b_concat_hmms.sh

# Step 6: hmmsearch novel domains (after Step 5b)
sbatch MANUSCRIPT/scripts/novel_domains/06_hmmsearch_novel.sbatch

# Step 7: Characterize (after Step 6)
sbatch MANUSCRIPT/scripts/novel_domains/07_characterize.sbatch

# Step 8: Environmental correlation (after Step 7)
sbatch MANUSCRIPT/scripts/novel_domains/08_env_correlation.sbatch
```

## Expected Outputs

All outputs are written to `03_analyses/novel_domains/`:

```
novel_domains/
├── sample_list.txt                     # Sample names (from Step 0)
├── dark_ids/                           # Per-sample dark protein IDs (Step 1)
├── dark_fasta/                         # Per-sample dark FASTA (Step 2)
├── dark_proteome_filtered.fa           # Filtered concatenated FASTA (Step 3)
├── filter_stats.tsv                    # Filter statistics (Step 3)
├── clusters_30/                        # 30% identity clusters (Step 4)
│   ├── clusters_30_cluster.tsv
│   ├── cluster_sizes.tsv
│   └── large_clusters.txt
├── clusters_50/                        # 50% identity clusters (Step 4)
├── hmms/                               # HMM profiles (Step 5/5b)
│   ├── novel_domains.hmm
│   └── per_cluster/
├── cluster_alignments/                 # MSAs (Step 5)
├── novel_hits/                         # Per-sample hmmsearch results (Step 6)
└── results/                            # Analysis outputs (Steps 6b–8)
    ├── novel_domain_count_matrix.tsv
    ├── novel_domain_prevalence.tsv
    ├── novel_domain_summary.tsv
    ├── novel_vs_pfam_comparison.tsv
    ├── dark_proteome_overview.tsv
    ├── cluster_size_distribution.tsv
    ├── novel_domain_env_correlations.tsv
    ├── novel_domain_xgboost_performance.tsv
    └── novel_vs_pfam_effect_sizes.tsv
```

## Troubleshooting

- **Step 0 audit fails**: Verify the base path and data directories exist.
  Check `03_analyses/hmmsearch_results/` for hmmsearch `.tbl` files.
- **Step 5 array size**: The default array range (1-100) assumes up to 100,000
  large clusters. If Step 4 produces more, adjust:
  ```bash
  N=$(wc -l < 03_analyses/novel_domains/clusters_30/large_clusters.txt)
  NBATCHES=$(( (N + 999) / 1000 ))
  sbatch --array=1-${NBATCHES}%50 MANUSCRIPT/scripts/novel_domains/05_build_hmms.sbatch
  ```
- **Step 4 mostly singletons (>99%)**: Clustering parameters may need tuning.
  Try `--min-seq-id 0.2` or adjusting coverage threshold.
- **Memory errors in Steps 3/4**: Increase `--mem` in the sbatch header.
  The concatenated dark proteome can be very large (100M+ sequences).
- **Missing Python packages in Step 8**: The conda env needs `numpy`, `pandas`,
  `scipy`, `statsmodels`, and `scikit-learn`.
