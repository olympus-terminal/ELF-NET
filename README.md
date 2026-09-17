# ELF-NET

**Coupling of oceanographic state to the dark proteome: a foundation for genome-informed marine productivity modeling**

Complete analysis pipeline (371 scripts, 15 modules) for the ELF-NET study, integrating protein language model classification, Pfam domain annotation, satellite foundation model embeddings, and machine learning to characterize genome-environment coupling. The input collection contained 2,357 ocean metagenomes, marine eukaryotic transcriptomes, and cultured reference proteomes; the 2,044-sample domain-analysis set contained 231.7 million proteins classified as algal.

## Repository Structure

| Module | Description | Scripts |
|--------|-------------|---------|
| `01_la4sr_inference/` | LA4SR (Pythia-based) protein language model inference | 6 |
| `02_algagpt_inference/` | algaGPT (nanoGPT-based) inference and post-processing | 8 |
| `03_pfam_annotation/` | Pfam hmmsearch, RuBisCO lineage detection, DIAMOND BLASTp validation | 10 |
| `04_environmental_data/` | Google Earth Engine extraction, AlphaEarth embeddings, WOA23 nutrients | 7 |
| `05_xgboost_shap/` | Bidirectional XGBoost/SHAP modeling (domain-to-environment, environment-to-domain) | 14 |
| `06_manifold_reduction/` | UMAP, t-SNE, PCA dimensionality reduction | 14 |
| `07_kan_cca/` | Sparse CCA and Kolmogorov-Arnold Network CCA | 30 |
| `08_novel_domains/` | De novo domain discovery: MMseqs2 clustering, HMM construction, ESMFold, Foldseek | 41 |
| `09_novel_families/` | Novel protein family characterization (Track A/B pipelines) | 40 |
| `10_decoding_experiment/` | Greedy vs. top-k decoding strategy comparison | 6 |
| `11_validation/` | Robustness: spatial block CV, bootstrap CI, permutation tests, sensitivity analyses, temporal stability | 93 |
| `12_figures/` | Publication figure generation (main + supplemental) | 60 |
| `13_selection_analysis/` | Cross-taxonomic dark-vs-white pN/pS and dN/dS (*C. reinhardtii*, *S. robusta*, *Synechococcus*, *T. pseudonana*) | 13 |
| `14_dark_proteome_characterization/` | Physicochemical characterization, dipeptide O/E, AF3/Boltz-2 structure analysis | 20 |
| `15_gpt2_learnability/` | GPT-2 protein language model learnability comparison (five corpora) | 2 |
| `utilities/` | Shared modules: color palette, style, data integrity guard, VICReg architecture | 7 |

## Key Methods

**Protein classification.** algaGPT, the primary classifier used in the study, is based on the GPT-2/nanoGPT architecture; the repository also preserves the Pythia-based LA4SR comparison workflow. Published whole-proteome tests produced predictions for >99% of sequences, while the matched algaGPT benchmark retained 83.3% of algal proteins and achieved >95% precision for the bacterial class. These are distinct performance measures. Inference was ~10,000-fold faster than BLASTp.

**Satellite-metagenome fusion.** AlphaEarth 64-dimensional satellite embeddings fused with Pfam domain profiles -- first integration of satellite foundation models with ocean metagenomics.

**Bidirectional modeling.** XGBoost with SHAP decomposition for forward (environment-to-domain) and reverse (domain-to-environment) prediction under spatial block cross-validation. Sea surface temperature predicted from Pfam composition at R² = 0.38.

**Nonlinear coupling.** Sparse CCA and KAN-CCA detect linear and nonlinear genome-environment coupling, including two-regime structure in harmful algal bloom monitoring bands.

**Dark proteome discovery.** MMseqs2 clustering of 201 million unannotated proteins into 33,950 novel domain families. Structure prediction via ESMFold, Boltz-2, and AlphaFold 3 with Foldseek homology search.

**Selection analysis.** pN/pS and dN/dS comparison of dark (unannotated) vs. white (Pfam-annotated) genes across four lineages shows dark proteins are under purifying selection comparable to annotated genes.

**Learnability test.** GPT-2 Small trained from scratch on five protein-sequence corpora ranks learnability as white > algae > dark > random, placing dark proteins between annotated proteins and random controls.

## Computational Environment

Scripts were developed and executed across two environments:
- **Local**: Script development and single-file testing
- **HPC (SLURM)**: Production batch processing with GPU-accelerated inference

SBATCH submission scripts (`.sbatch`) are included for HPC reproducibility.

**Note on paths:** Scripts contain hardcoded paths referencing the original HPC environment (`/scratch/drn2/PROJECTS/TARA-LA4SR/`) and local development machine (`/media/drn2/External/TARA-Oceans/`). Many scripts include hostname-based environment detection that switches paths automatically; others will need manual path adjustment. All input data referenced by these paths is available from the Zenodo deposits listed below.

## Dependencies

Install Python dependencies:

```bash
pip install -r requirements.txt
```

**External tools:** HMMER 3.x, MMseqs2, ESMFold, Boltz-2, AlphaFold 3, Foldseek, SNAP, InterProScan 5.74-105.0, nanoGPT

## Trained Models (Hugging Face)

| Model | Description | Link |
|-------|-------------|------|
| algaGPT | Protein classification (algal vs. contaminant) | [GreenGenomicsLab/algaGPT](https://huggingface.co/GreenGenomicsLab/algaGPT) |
| TARA-XGBoost-Bidirectional | Bidirectional XGBoost (env-domain coupling) | [GreenGenomicsLab/TARA-XGBoost-Bidirectional](https://huggingface.co/GreenGenomicsLab/TARA-XGBoost-Bidirectional) |
| TARA-WorldModel-VICReg | VICReg joint embedding (exploratory) | [GreenGenomicsLab/TARA-WorldModel-VICReg](https://huggingface.co/GreenGenomicsLab/TARA-WorldModel-VICReg) |
| dark-whiteGPLM | GPT-2 protein language model checkpoints | [SarahDaakour/dark-whiteGPLM](https://huggingface.co/SarahDaakour/dark-whiteGPLM) |
| dark-whiteGPLM-data | Training data for learnability comparison | [SarahDaakour/dark-whiteGPLM-data](https://huggingface.co/datasets/SarahDaakour/dark-whiteGPLM-data) |

## Data Deposits (Zenodo)

| Deposit | DOI |
|---------|-----|
| Data S1: Domain-environment association and modeling results | [10.5281/zenodo.18538438](https://doi.org/10.5281/zenodo.18538438) |
| Data S2: algaGPT-classified algal protein sequences | [10.5281/zenodo.18728836](https://doi.org/10.5281/zenodo.18728836) |
| Data S3: Pfam-A hmmsearch results | [10.5281/zenodo.18786751](https://doi.org/10.5281/zenodo.18786751) |
| Data S4: RuBisCO lineage analysis package | [10.5281/zenodo.18786775](https://doi.org/10.5281/zenodo.18786775) |
| Data S5: AlphaEarth satellite embedding matrix | [10.5281/zenodo.18786762](https://doi.org/10.5281/zenodo.18786762) |
| Data S6: KAN-CCA and sparse CCA results | [10.5281/zenodo.18786766](https://doi.org/10.5281/zenodo.18786766) |
| Data S7: Novel domain discovery results | [10.5281/zenodo.18786771](https://doi.org/10.5281/zenodo.18786771) |
| Data S8: DIAMOND BLASTp comparison | [10.5281/zenodo.19441356](https://doi.org/10.5281/zenodo.19441356) |
| Data S9: Cross-taxonomic selection analysis | [10.5281/zenodo.20486974](https://doi.org/10.5281/zenodo.20486974) |
| Data S10: Boltz-2 and AlphaFold 3 structure predictions | [10.5281/zenodo.20508681](https://doi.org/10.5281/zenodo.20508681) |
| Data S11: Protein language model learnability analysis | [10.5281/zenodo.20933333](https://doi.org/10.5281/zenodo.20933333) |

## Authors

David Roy Nelson, Maxence Plouviez, Sarah Daakour, Ashish Jaiswal, Weiqi Fu, Shady A. Amin, Kourosh Salehi-Ashtiani

Green Genomics Lab, New York University Abu Dhabi

## Citation

```bibtex
@article{nelson2026elfnet,
  title   = {Coupling of oceanographic state to the dark proteome: a foundation for genome-informed marine productivity modeling},
  author  = {Nelson, David Roy and Plouviez, Maxence and Daakour, Sarah and Jaiswal, Ashish and Fu, Weiqi and Amin, Shady A. and Salehi-Ashtiani, Kourosh},
  note    = {Manuscript submitted for publication},
  year    = {2026}
}
```

algaGPT was introduced in:

```bibtex
@article{nelson2025la4sr,
  title   = {Pan-microalgal dark proteome mapping via interpretable deep learning and synthetic chimeras},
  author  = {Nelson, David R. and Jaiswal, Ashish Kumar and Ismail, Noha Samir and Mystikou, Alexandra and Salehi-Ashtiani, Kourosh},
  journal = {Patterns},
  volume  = {6},
  pages   = {101373},
  year    = {2025},
  doi     = {10.1016/j.patter.2025.101373}
}
```

## Contact

Kourosh Salehi-Ashtiani -- ksa3@nyu.edu

## License

MIT License. See [LICENSE](LICENSE) for details.
