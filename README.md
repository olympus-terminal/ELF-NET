# ELF-NET

**Protein language model classification and satellite embeddings reveal microalgal genome-environment coupling on a global manifold**

Computational pipelines and analysis scripts accompanying the ELF-NET manuscript.

## Overview

ELF-NET integrates protein language model classification, PFAM domain annotation, satellite foundation model embeddings, and machine learning to characterize genome-environment coupling across 2,357 ocean samples and 221.9 million algal protein sequences.

## Repository Structure

| Module | Description | Scripts |
|--------|-------------|---------|
| `01_la4sr_inference/` | LA4SR (Pythia-based) protein language model inference | 6 |
| `02_algagpt_inference/` | AlgaGPT (nanoGPT-based) inference and post-processing | 8 |
| `03_pfam_annotation/` | Pfam hmmsearch, RuBisCO lineage detection, DIAMOND BLASTp validation | 10 |
| `04_environmental_data/` | Google Earth Engine extraction, AlphaEarth embeddings, WOA23 nutrients | 7 |
| `05_xgboost_shap/` | Bidirectional XGBoost/SHAP modeling (domain-to-environment, environment-to-domain) | 14 |
| `06_manifold_reduction/` | UMAP, t-SNE, PCA dimensionality reduction pipelines | 14 |
| `07_kan_cca/` | Sparse CCA and Kolmogorov-Arnold Network CCA | 30 |
| `08_novel_domains/` | De novo domain discovery: MMseqs2 clustering, HMM construction, ESMFold, Foldseek | 41 |
| `09_novel_families/` | Novel protein family characterization (Track A/B pipelines) | 40 |
| `10_decoding_experiment/` | Greedy vs. top-k decoding strategy comparison | 6 |
| `11_validation/` | Robustness testing: spatial block CV, bootstrap CI, permutation tests, sensitivity analyses, temporal stability, feature stability, SHAP interactions | 93 |
| `12_figures/` | Publication figure generation (main + supplemental) | 60 |
| `13_selection_analysis/` | Cross-taxonomic dark-vs-white selection analysis (pN/pS, dN/dS) for *C. reinhardtii*, *S. robusta*, *Synechococcus*, *T. pseudonana* | 13 |
| `14_dark_proteome_characterization/` | Physicochemical characterization, dipeptide O/E, AF3 structure analysis, rank-binned ESMFold | 20 |
| `15_gpt2_learnability/` | GPT-2 protein language model learnability comparison (five corpora) | 2 |
| `utilities/` | Shared modules: color palette, style, data integrity guard, VICReg world model architecture | 7 |

**Total: 371 scripts across 15 modules**

## Key Methods

### Protein Language Model Classification
- **LA4SR-Pythia**: Transformer-based classifier trained on amino acid representations to distinguish algal from non-algal sequences, bypassing homology dependence
- **AlgaGPT**: nanoGPT-based alternative architecture for proteome extraction

### Satellite Foundation Model Integration
- **AlphaEarth**: 64-dimensional satellite embeddings fused with metagenomic Pfam domain profiles -- first integration of satellite foundation models with ocean metagenomics

### Bidirectional Genome-Environment Modeling
- XGBoost with SHAP decomposition for forward (domain-to-environment) and reverse (environment-to-domain) prediction under spatial block cross-validation

### Nonlinear Coupling Analysis
- Sparse CCA and KAN-CCA for detecting linear and nonlinear genome-environment coupling, including two-regime structure in harmful algal bloom monitoring bands

### De Novo Domain Discovery
- MMseqs2 clustering of 201 million unannotated proteins into 33,950 novel domain families
- ESMFold and Boltz-2/AlphaFold 3 structure prediction with Foldseek structural similarity search

### Cross-Taxonomic Selection Analysis
- pN/pS and dN/dS comparison of dark (unannotated) vs. white (Pfam-annotated) genes across four lineages

### Protein Language Model Learnability
- GPT-2 Small trained from scratch on five protein-sequence corpora (white, algae, dark, composition-matched random, uniform random) to test whether dark proteome sequences carry learnable structure

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
- `torch` (PyTorch) -- for KAN-CCA, VICReg world model, and GPT-2 training
- `transformers` -- for LA4SR/AlgaGPT inference
- `biopython` -- for sequence processing

External tools:
- HMMER 3.x (`hmmsearch`)
- MMseqs2
- ESMFold / Boltz-2 / AlphaFold 3
- Foldseek
- SNAP (gene prediction)
- InterProScan 5.74-105.0
- nanoGPT

## Related Resources

### Trained Models (Hugging Face)
- [algaGPT](https://huggingface.co/GreenGenomicsLab/algaGPT) -- Protein classification model
- [TARA-XGBoost-Bidirectional](https://huggingface.co/GreenGenomicsLab/TARA-XGBoost-Bidirectional) -- Bidirectional XGBoost models
- [TARA-WorldModel-VICReg](https://huggingface.co/GreenGenomicsLab/TARA-WorldModel-VICReg) -- VICReg joint embedding checkpoints
- [dark-whiteGPLM](https://huggingface.co/SarahDaakour/dark-whiteGPLM) -- GPT-2 protein language model checkpoints
- [dark-whiteGPLM-data](https://huggingface.co/datasets/SarahDaakour/dark-whiteGPLM-data) -- Training data

### Data Deposits (Zenodo)
Input data and model outputs are deposited at Zenodo (accession numbers provided in the manuscript).

## License

See manuscript for terms of use.
