# Dark Proteome Subsets for LM Distillation Experiment

Generated: 2026-02-21

## Files

| File | Size | Sequences | Description |
|------|------|-----------|-------------|
| `dark_proteome_100mb.fa` | 101 MB | 442,752 | Proteins with **no Pfam-A hits** (E < 1e-9) |
| `annotated_proteome_100mb.fa` | 101 MB | 341,803 | Proteins with **at least one Pfam-A hit** (positive control) |

## Source Data

- **Total proteome**: 2,044 algal assemblies from TARA-Oceans LA4SR pipeline
- **Total proteins**: ~462M across all assemblies
- **Dark fraction**: ~95.5% of proteins have no Pfam-A domain annotation
- **Source FASTAs**: `/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/algae_proteins/*.algae.fa`
- **Pfam results**: `/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/hmmsearch_results/*.aa.hmmsearch.tbl`

## Sampling Method

- Random subsampling with fixed seed (42) for reproducibility
- Dark set: sampled from `dark_proteome_filtered.fa` (50 GB, 442M sequences post length/complexity filtering)
- Annotated set: for each of the 2,044 samples, proteins NOT in the dark ID list were sampled
- Both sets sample across all 2,044 assemblies (not geographically stratified)
- Filtering applied to dark proteome before sampling: min 30 aa, max 5,000 aa, no low-complexity or poly-X runs

## Sequence Format

Standard FASTA. Headers follow the pattern:
```
>{protein_id}-{species}.{source}.fa.{contig}
```

Example:
```
>10000003-Alexandrium_andersonii.AAC.fa.1
MKLVTFSAAGLLL...
```

## Reproduction (100 MB subsets)

Scripts on Jubail HPC:
```
/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains/sample_subsets.py
/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains/sample_annotated_only.py
```

---

## 15M Sequence Delivery (2026-02-27)

Requested by Kourosh for scaled-up GPLM distillation training.
Each class is split 90% train / 10% holdout.

### Output Files

| File | Description |
|------|-------------|
| `dark_15m_train.fa` | ~13.5M dark sequences (no Pfam hits), train split |
| `dark_15m_holdout.fa` | ~1.5M dark sequences, holdout split |
| `annotated_15m_train.fa` | ~13.5M annotated sequences (with Pfam hits), train split |
| `annotated_15m_holdout.fa` | ~1.5M annotated sequences, holdout split |
| `annotated_15m_train.hmmsearch.tbl` | Pfam-A hmmsearch hits for train annotated seqs |
| `annotated_15m_holdout.hmmsearch.tbl` | Pfam-A hmmsearch hits for holdout annotated seqs |

### Where to Find Results

```
/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains/kourosh_subsets/
```

SLURM logs:
```
/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains/logs/slurm_*_sample_15m.out
```

### Sampling Parameters

- **Target**: 15,000,000 sequences per class
- **Holdout fraction**: 10% (every 10th selected sequence)
- **Random seed**: 42 (dark), 43 (annotated)
- **Dark source**: `dark_proteome_filtered.fa` (~442M seqs, pre-filtered 30–5000 aa, no low-complexity)
- **Annotated source**: all 2,044 assemblies, proteins NOT in dark ID lists
- **hmmsearch format**: HMMER3 tblout, lines starting with `#` are comments, column 1 = protein ID, column 3 = Pfam domain
- **Note**: If fewer than 15M annotated proteins exist, ALL are included and the actual count is reported in the SLURM log

### Reproduction

```bash
cd /scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains
sbatch sample_15m.sbatch
# or manually:
python sample_15m.py --target-seqs 15000000 --seed 42 --holdout-frac 0.10
```

Script source (git):
```
MANUSCRIPT/scripts/novel_domains/sample_15m.py
MANUSCRIPT/scripts/novel_domains/sample_15m.sbatch
```
