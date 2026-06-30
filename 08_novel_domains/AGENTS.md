# Novel Domains Pipeline — Agent Operations Guide

## Project Layout

- Pipeline scripts: `scripts/novel_domains/` (00-08 numbered steps)
- Spec: `scripts/novel_domains/specs/pipeline_spec.md`
- Plan: `scripts/novel_domains/IMPLEMENTATION_PLAN.md`
- Manuscript: `main.tex`, `supplemental_information.tex` (project root)
- Source data provenance: `source_data/*.md` (project root)
- Figures: `figures/` (project root)

## Build & Validate

```bash
# Shell/SLURM syntax check
bash -n scripts/novel_domains/*.sh scripts/novel_domains/*.sbatch

# Python syntax check
python3 -m py_compile scripts/novel_domains/06b_build_count_matrix.py
python3 -m py_compile scripts/novel_domains/07_characterize.py
python3 -m py_compile scripts/novel_domains/08_env_correlation.py

# LaTeX compilation
cd /home/drn/Documents/projects/TARA-OMEN/MANUSCRIPT && tectonic main.tex
```

## HPC Environment (Jubail)

- Base path: `/scratch/drn2/PROJECTS/TARA-LA4SR`
- Conda env: `/scratch/drn2/software/conda-mamba_1`
- Every .sbatch needs: `set -euo pipefail`, HOME/TMPDIR/PYTHONUSERBASE exports, `module purge && module load gcc/13.2.0`, conda activation
- Array jobs: 2318 samples from `03_analyses/algae_proteins/*.fa`

## Data Integrity Rules (from CLAUDE.md)

- NEVER fabricate statistics — compute from source files
- Write provenance to `source_data/*.md` before referencing in manuscript
- This is a methods paper — describe methodology, not biological findings

## Git Commit Convention

```
novel-domains: <brief description of what changed>
```

## Do NOT

- Run HPC scripts locally (they target `/scratch/drn2/` paths)
- Push to remote repositories
- Fabricate data or statistics
- Modify files outside `scripts/novel_domains/` without good reason
