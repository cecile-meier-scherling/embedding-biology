# Axiom Take-Home: Biological Signal in Image Embeddings

Analysis of PHH image embeddings and assay measurements to assess how well they capture compound pathway, target, and biological-activity annotations. The project compares raw PCA, normalized PCA, DINO embeddings (stored in the `brightfield` column), assay features, and combined representations.

The workflow includes exploratory analysis, multilabel annotation prediction, nearest-neighbor retrieval, cluster enrichment, and checks for plate/batch confounding.

## Setup

Use Python 3.12 (the version selected in `.python-version`) and [uv](https://docs.astral.sh/uv/). Run commands from the repository root:

```bash
uv sync --locked
```

Place the input dataset at:

```text
data/phh_prod_image_data_oasis_with_dmso.parquet
```

The dataset must be supplied separately if it is absent from your checkout; the analysis scripts do not download it. They expect compound identifiers, pathway/target/activity annotations, experimental metadata, the three embedding columns, and the assay columns listed in `analysis/annotation_model_utils.py`.

## Explore the data

```bash
uv run jupyter lab analysis/data_analysis.ipynb
```

The notebook loads the dataset using a path relative to `analysis/`; use that directory as the notebook kernel's working directory. It provides the exploratory starting point. The scripts below reproduce the model comparisons and figures. `main.py` is a placeholder, not a pipeline runner.

## Run the analysis

Run these steps in order. Scripts write to `results/` and replace existing outputs with the same names.

### 1. Generate corrected feature arrays

```bash
uv run python analysis/correct_features.py
```

Creates matched-compound/dose plate and batch corrections, plus Harmony integration variants when `harmonypy` is available. The output arrays are required by the compound/batch comparisons and retrieval scripts.

### 2. Compare pathway and target prediction

```bash
uv run python analysis/compare_annotations_held_out_compound.py
uv run python analysis/compare_annotations_held_out_batch.py
uv run python analysis/compare_annotations_chemical_split.py
```

The comparisons use per-label logistic regression and a label-frequency baseline. Replicate-well predictions are aggregated to the compound level, with macro/micro F1 for the top three predicted labels, fold scores, and bootstrap intervals.

| Validation protocol | What is held out |
| --- | --- |
| Compound | Five-fold grouped validation keeps all wells for a compound together. |
| Batch | Each batch is held out, and compounds present in that batch are excluded from training. |
| Chemical group | Five-fold grouped validation holds out entire scaffold groups; acyclic compounds are grouped using fingerprint similarity. |

Chemical splitting resolves compound structures through PubChem and caches them in `results/pubchem_structure_lookup.csv`. Uncached lookups require internet access. Optional corrections can be supplied in `data/compound_structure_overrides.csv` with `compound_id` and `smiles` columns, plus an optional `cid`. Compounds without usable structures are excluded from chemical-group evaluation; inspect `results/chemical_split_structure_audit.csv` for coverage.

The compound and batch runs also produce embedding-cluster enrichment tables. Comparison runs save fitted models and ranked predictions for unannotated compounds under `results/`.

### 3. Evaluate retrieval, stability, and confounding

```bash
uv run python analysis/evaluate_embedding_retrieval.py
uv run python analysis/evaluate_activity_text_retrieval.py
uv run python analysis/evaluate_normalized_pca_stability.py
uv run python analysis/evaluate_confounding.py
```

Annotation retrieval checks whether nearby training compounds share pathway or target labels. Activity retrieval measures similarity between biological-activity descriptions using training-fitted TF-IDF features. Both evaluate neighborhoods of 1, 3, 5, and 10 compounds across the three validation protocols.

The stability script requires the chemical-split audit from step 2. The confounding analysis probes technical information in the representations and compares it with the biological evaluation metrics.

An optional text-only annotation baseline is available separately:

```bash
uv run python analysis/predict_annotations_from_activity_text.py
```

### 4. Generate figures

```bash
uv run python plots/plot_annotation_results.py
uv run python plots/plot_normalized_pca_results.py
```

## Outputs and repository layout

| Location | Contents |
| --- | --- |
| `analysis/data_analysis.ipynb` | Exploratory notebook |
| `analysis/` | Feature correction, prediction, retrieval, and evaluation scripts |
| `plots/` | Figure generation from result tables |
| `results/annotation_model_comparison_*.csv` | Prediction metrics by representation and validation protocol |
| `results/annotation_prediction_stability_*.csv` | Fold scores and bootstrap intervals |
| `results/cluster_label_enrichment_*.csv` | Cluster enrichment with multiple-testing-adjusted q-values |
| `results/embedding_retrieval_*.csv` | Annotation retrieval metrics and neighbors |
| `results/biological_activity_retrieval_*.csv` | Activity-text retrieval metrics, neighbors, and per-query scores |
| `results/confounding_*.csv` | Technical-confounding probes and comparisons with biology scores |
| `results/models/` | Fitted annotation models |
| `results/unannotated_compound_*.csv` | Ranked annotation predictions and activity-text neighbors |
| `results/*.png` | Generated comparison figures |

## Interpretation

- Plate/batch corrections are estimated across the dataset without annotation labels. Because held-out wells contribute to these corrections, corrected scores are transductive comparisons, not strict prospective estimates for an entirely unseen plate or batch.
- Chemical-group evaluation uses the subset with resolved structures, so its scores may reflect a different compound population.
- Activity-text similarity and annotation enrichment measure agreement with existing descriptions and labels; they do not establish a compound's mechanism of action.
- Use the fold variation, bootstrap intervals, and enrichment q-values alongside aggregate scores when comparing representations.

## License

[MIT](LICENSE).
