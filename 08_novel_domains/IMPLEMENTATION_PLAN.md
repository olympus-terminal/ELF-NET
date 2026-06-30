# Novel Domains Pipeline — Implementation Plan

## Completed
- [x] Pipeline scripts written (Steps 00-08, helpers 05b/06b)
- [x] SLURM orchestrator (`submit_pipeline.sh`)
- [x] Spec written (`specs/pipeline_spec.md`)
- [x] Basic syntax validation (bash -n, py_compile)

## TODO (prioritized)

### High Priority — Pipeline Correctness
- [ ] Deep audit of all scripts: verify input/output path chains match across steps, array indexing is correct, edge cases handled (empty files, failed samples)
- [ ] Verify Step 1 (`01_extract_dark_ids.sbatch`): cross-check logic — dark IDs = total proteins minus Pfam-annotated proteins. Confirm file paths and ID extraction method are correct
- [ ] Verify Step 4 (`04_mmseqs2_cluster.sbatch`): MMseqs2 linclust parameters appropriate for marine metagenome protein clustering (30%/50% identity thresholds, coverage, sensitivity)
- [ ] Verify Step 5 (`05_build_hmms.sbatch`): cluster size thresholds for HMM building, MAFFT alignment parameters, hmmbuild options
- [ ] Verify Step 6b (`06b_build_count_matrix.py`): count matrix construction logic, E-value thresholds, output format compatible with downstream analysis
- [ ] Verify Step 7 (`07_characterize.py`): characterization metrics are meaningful (size distribution, taxonomy, functional annotation)
- [ ] Verify Step 8 (`08_env_correlation.py`): environmental correlation methodology matches the approach used for known Pfam domains in the main pipeline

### High Priority — Manuscript Integration
- [ ] Write `source_data/novel_domains_pipeline.md` provenance file documenting the pipeline design, parameters, and expected outputs
- [ ] Draft Methods section text for novel domain discovery (LaTeX paragraph for `main.tex`)
- [ ] Draft Results section text describing what the pipeline will produce
- [ ] Draft supplemental methods with full pipeline parameters

### Medium Priority — Analysis & Figures
- [ ] Design figure showing dark proteome size relative to annotated proteome
- [ ] Design figure or table for novel domain family characterization
- [ ] Plan integration of novel domain count matrix with existing Pfam-based analyses (XGBoost, CCA, etc.)

### Lower Priority — Robustness
- [ ] Add error recovery to `submit_pipeline.sh` (check for partial outputs from previous runs)
- [ ] Add intermediate checkpoints/validation between major steps
- [ ] Document expected runtime and resource requirements per step
