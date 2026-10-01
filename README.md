# Biological Signal in Image Embeddings

This repository evaluates whether raw PCA, normalized PCA, and brightfield (DINO) embeddings capture compound biology. It compares them with assay-feature and label-frequency baselines using pathway/target prediction, annotation and biological-activity retrieval, clustering, replicate consistency, and plate/batch confounding analyses.

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
```

Place the input Parquet dataset at `data/phh_prod_image_data_oasis_with_dmso.parquet`. It is not included in this repository and is not downloaded by the scripts. Required columns and assay features are listed in `analysis/annotation_model_utils.py`.

## Main analyses

Run commands from the repository root. Outputs are written under `results/`.

```bash
# Pathway/target prediction and embedding-cluster enrichment
uv run python analysis/compare_annotations_held_out_compound.py
uv run python analysis/compare_annotations_held_out_batch.py
uv run python analysis/compare_annotations_chemical_split.py

# Retrieval, stability, and technical confounding
uv run python analysis/evaluate_embedding_retrieval.py
uv run python analysis/evaluate_activity_text_retrieval.py
uv run python analysis/evaluate_normalized_pca_stability.py
uv run python analysis/evaluate_confounding.py

# Figures
uv run python plots/plot_annotation_results.py
uv run python plots/plot_normalized_pca_results.py
```

The compound split keeps all wells from a compound together. The batch split tests a held-out batch and removes test compounds from training. The chemical split holds out scaffold/similarity groups; it uses PubChem lookups for missing structures and caches them under `results/chemical_split/`. Uncached lookups need internet access. Compounds without usable structures are omitted from that split.

To create matched-compound/dose correction arrays, run `analysis/correct_features.py` first. Those corrections use profiles across the dataset, so results using them are transductive. Fold-safe Harmony can instead be evaluated on compound and chemical-group holdouts:

```bash
uv run python analysis/evaluate_harmony.py --protocols held_out_compound held_out_chemical_group
```

Harmony is fitted using training folds only. It cannot estimate a correction for a wholly unseen batch without calibration data from that batch.

## Additional evaluations

The drug-attribute benchmark predicts cellular measurements and public pathway/target annotations, and evaluates cross-plate replicate retrieval:

```bash
uv run python analysis/evaluate_drug_attributes.py
uv run python plots/plot_drug_attributes.py
```

Public annotation snapshots and source/matching notes are in `data/public_annotations/`. To refresh the local joins and run those benchmarks:

```bash
uv run python analysis/import_public_annotations.py --offline
uv run python analysis/evaluate_public_annotations.py
uv run python plots/plot_public_annotations.py
```

The Broad source files include a non-commercial-use notice; see `data/public_annotations/manifest.json` and check the source terms before redistribution or commercial use. Public annotation matches and all retrieval/enrichment results are associative evidence, not confirmation of mechanism or causality.

## Results and interpretation

Results are grouped by task under `results/`: `annotation_prediction/`, `annotation_retrieval/`, `activity_text_retrieval/`, `cluster_enrichment/`, `replicate_analysis/`, `confounding/`, `feature_processing/`, `drug_attributes/`, `public_annotations/`, and `chemical_split/`. Figures are generated from result tables by scripts in `plots/`.

Use fold variation and bootstrap intervals when comparing scores. Missing annotations are not necessarily negative labels. Chemical-group results cover only compounds with usable structures, and the supplied embeddings may have been preprocessed upstream. Batch correction, retrieval, and cluster enrichment can expose trade-offs; none alone establishes that a representation is biologically superior.

The local source dataset, large model files, and large neighbor tables are excluded from ordinary Git commits. Recreate them by running the analysis scripts. An exploratory notebook is available at `analysis/data_analysis.ipynb`.

## License

Code: [MIT](LICENSE). Data-source terms may differ; see the source manifest before redistributing cached annotations.
