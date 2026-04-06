# ELF-NET

**Protein language model classification and satellite embeddings reveal microalgal genome-environment coupling on a global manifold**

Computational pipelines and analysis scripts accompanying the ELF-NET manuscript.

## Overview

ELF-NET integrates protein language model classification, PFAM domain annotation, satellite foundation model embeddings, and machine learning to characterize genome-environment coupling across 2,357 ocean samples and 221.9 million algal protein sequences.

## Repository Structure

| Directory | Description | Scripts |
|-----------|-------------|---------|
| `01_la4sr_inference/` | LA4SR (Pythia-based) protein language model inference | 6 |
| `02_algagpt_inference/` | AlgaGPT alternative LLM inference and post-processing | 8 |
| `03_pfam_annotation/` | PFAM hmmsearch, RuBisCO detection, DIAMOND BLASTp validation | 10 |
| `04_environmental_data/` | Google Earth Engine extraction, AlphaEarth embeddings, WOA23 nutrients | 7 |
| `05_xgboost_shap/` | Bidirectional XGBoost/SHAP modeling (domain-to-environment, environment-to-domain) | 14 |
| `06_manifold_reduction/` | UMAP, t-SNE, PCA dimensionality reduction pipelines | 14 |
| `07_kan_cca/` | Sparse CCA and Kolmogorov-Arnold Network CCA | 30 |
| `08_novel_domains/` | De novo domain discovery: MMseqs2 clustering, HMM construction, ESMFold, Foldseek | 38 |
| `09_novel_families/` | Novel protein family characterization (Track A/B pipelines) | 40 |
| `10_decoding_experiment/` | Greedy vs. top-k decoding strategy comparison | 6 |
| `11_validation/` | k-fold CV, permutation tests, feature stability, SHAP interactions | 24 |
| `12_figures/` | Publication figure generation scripts | 25 |
| `utilities/` | Shared modules: color palette, data integrity guard, world model architecture | 5 |

**Total: 227 scripts**

## Key Methods

### Protein Language Model Classification
- **LA4SR-Pythia**: Transformer-based classifier trained on amino acid representations to distinguish algal from non-algal sequences, bypassing homology dependence
- **AlgaGPT**: Alternative LLM architecture for proteome extraction comparison

### Satellite Foundation Model Integration
- **AlphaEarth**: 64-dimensional satellite embeddings fused with metagenomic PFAM domain profiles -- first integration of satellite foundation models with ocean metagenomics

### Bidirectional Genome-Environment Modeling
- XGBoost with SHAP decomposition for forward (domain-to-environment) and reverse (environment-to-domain) prediction under spatial block cross-validation

### Nonlinear Coupling Analysis
- Sparse CCA and KAN-CCA for detecting linear and nonlinear genome-environment coupling, including two-regime structure in harmful algal bloom monitoring bands

### De Novo Domain Discovery
- MMseqs2 clustering of 201 million unannotated proteins into 33,950 novel domain families
- ESMFold structure prediction and Foldseek structural similarity search

## Computational Environment

Scripts were developed and executed across two environments:
- **Local**: Script development and single-file testing
- **HPC (SLURM)**: Production batch processing with GPU-accelerated inference

SBATCH submission scripts (`.sbatch`) are included for HPC reproducibility.

## Dependencies

Core Python packages:
- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `xgboost`, `shap`
- `umap-learn`, `matplotlib`
- `torch` (PyTorch) -- for KAN-CCA and world model
- `transformers` -- for LA4SR/AlgaGPT inference
- `biopython` -- for sequence processing

External tools:
- HMMER 3.x (`hmmsearch`)
- MMseqs2
- ESMFold
- Foldseek
- SNAP (gene prediction)

## Data Availability

Input data and model outputs are deposited at Zenodo (accession numbers provided in the manuscript).

## License

See manuscript for terms of use.
