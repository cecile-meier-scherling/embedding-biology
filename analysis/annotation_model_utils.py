"""Compare annotation prediction across features and validation schemes.

Run from any directory with:
    uv run python /path/to/axiom_takehome/analysis/compare_annotation_models.py

Writes prediction metrics plus held-out embedding-cluster enrichment tables to
the project's results/ directory. Clusters are discovered without labels; labels
are used only afterward to quantify enrichment.
"""
from __future__ import annotations

import ast
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/phh_prod_image_data_oasis_with_dmso.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"
N_SPLITS = 5
N_CLUSTERS = 10
MIN_LABEL_COMPOUNDS = N_SPLITS + 1
RANDOM_STATE = 42
CORRECTED_FEATURES_PATH = RESULTS_DIR / "feature_processing" / "corrected_feature_arrays.npz"

ASSAY_FEATURES = [
    "ldh_ridge_norm", "mtt_ridge_norm",
    "bioactivity_svc_tvn_pca_platenorm_coral_regnorm_all_n_components_256",
    "mean_nuclei_count", "mean_nuclei_count_ridge_norm",
    "mean_nuclei_count_ridge_norm_inv", "mean_nuclei_area_mean",
    "mean_nuclei_area_mean_ridge_norm_inv", "cyto_area_ratio",
    "cyto_area_ratio_ridge_norm", "vacuole_sum", "vacuole_ratio",
    "mito_puncta_ratio", "mito_puncta_ratio_ridge_norm_sub",
    "ros_stress_granules_sum_mean_norm_ridge_norm_corr",
]


def unpack_vector(value: object) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=np.float32)


def load_corrected_feature_arrays(source_row_indices: np.ndarray,
                                  feature_names: list[str]) -> dict[str, np.ndarray]:
    """Load well-aligned correction variants created by correct_features.py."""
    if not CORRECTED_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CORRECTED_FEATURES_PATH}; run analysis/correct_features.py first"
        )
    loaded: dict[str, np.ndarray] = {}
    with np.load(CORRECTED_FEATURES_PATH) as archive:
        for feature in feature_names:
            for variant in ("plate_corrected", "batch_corrected",
                            "plate_batch_corrected"):
                key = f"{feature}__{variant}"
                if key in archive:
                    loaded[f"{feature}_{variant}"] = archive[key][source_row_indices]
    return loaded


def parse_labels(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    return tuple(part.strip() for part in str(value).split(";") if part.strip())


def build_splits(df: pd.DataFrame) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Build compound CV and leave-one-batch-out folds with compound leakage removed."""
    groups = df["compound_id"].astype(str).to_numpy()
    group_splits = list(GroupKFold(n_splits=N_SPLITS).split(df, groups=groups))
    batch_splits = []
    batches = df["batch"].to_numpy()
    for batch in sorted(pd.unique(batches)):
        test = np.flatnonzero(batches == batch)
        test_compounds = set(groups[test])
        train = np.flatnonzero((batches != batch) & ~np.isin(groups, list(test_compounds)))
        if len(train) == 0 or len(test) == 0:
            continue
        batch_splits.append((train, test))
    return {"held_out_compound": group_splits, "held_out_batch": batch_splits}


def make_label_matrix(labels: list[tuple[str, ...]], groups: np.ndarray):
    compounds_by_label: dict[str, set[str]] = {}
    for compound, row_labels in zip(groups, labels):
        for label in row_labels:
            compounds_by_label.setdefault(label, set()).add(compound)
    keep = sorted(
        label for label, compounds in compounds_by_label.items()
        if len(compounds) >= MIN_LABEL_COMPOUNDS
    )
    keep_set = set(keep)
    filtered_labels = [tuple(label for label in row if label in keep_set) for row in labels]
    mlb = MultiLabelBinarizer(classes=keep)
    return mlb.fit_transform(filtered_labels), keep


def train_and_score(X_train: np.ndarray, Y_train: np.ndarray,
                    X_test: np.ndarray,
                    groups_train: np.ndarray | None = None,
                    label_frequency_baseline: bool = False) -> np.ndarray:
    """Return fold scores from logistic models or a label-frequency ranking."""
    if label_frequency_baseline:
        if groups_train is None:
            raise ValueError("groups_train is required for the label-frequency baseline")
        _, first_indices = np.unique(groups_train, return_index=True)
        compound_labels = Y_train[first_indices]
        frequencies = compound_labels.sum(axis=0).astype(np.int64)
        # Stable sorting means ties follow the already-sorted label names.
        order = np.argsort(-frequencies, kind="stable")
        scores_by_label = np.empty(Y_train.shape[1], dtype=np.float32)
        scores_by_label[order] = np.arange(
            Y_train.shape[1], 0, -1, dtype=np.float32
        )
        return np.tile(scores_by_label, (len(X_test), 1))

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train = scaler.fit_transform(imputer.fit_transform(X_train))
    test = scaler.transform(imputer.transform(X_test))
    def fit_one_label(j: int) -> tuple[int, np.ndarray]:
        y = Y_train[:, j]
        classes = np.unique(y)
        if len(classes) == 1:
            value = 1.0 if classes[0] == 1 else -1.0
            return j, np.full(len(X_test), value, dtype=np.float32)
        model = LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=2000,
            random_state=RANDOM_STATE,
        )
        model.fit(train, y)
        return j, model.decision_function(test).astype(np.float32)

    n_jobs = min(6, Y_train.shape[1])
    with threadpool_limits(limits=1):
        fitted = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(fit_one_label)(j) for j in range(Y_train.shape[1])
        )
    scores = np.zeros((len(X_test), Y_train.shape[1]), dtype=np.float32)
    for j, label_scores in fitted:
        scores[:, j] = label_scores
    return scores


def run_prediction(X: np.ndarray, Y: np.ndarray, groups: np.ndarray,
                   splits: list[tuple[np.ndarray, np.ndarray]],
                   task: str, feature_set: str, protocol: str,
                   stability_rows: list[dict] | None = None,
                   bootstrap_units: np.ndarray | None = None,
                   n_bootstrap: int = 1000,
                   label_frequency_baseline: bool = False):
    score_sums: dict[str, np.ndarray] = {}
    truth: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    fold_results = []
    for fold_idx, (train, test) in enumerate(splits, start=1):
        scores = train_and_score(
            X[train], Y[train], X[test], groups_train=groups[train],
            label_frequency_baseline=label_frequency_baseline,
        )
        # Score each split after combining replicate wells into compound-level
        # predictions. This exposes variation hidden by the pooled OOF score.
        fold_score_sums: dict[str, np.ndarray] = {}
        fold_truth: dict[str, np.ndarray] = {}
        fold_counts: dict[str, int] = {}
        for idx, score in zip(test, scores):
            compound = groups[idx]
            # A test compound may occur in multiple held-out batches; aggregate
            # its out-of-fold wells into one compound-level prediction.
            score_sums[compound] = score_sums.get(compound, np.zeros(Y.shape[1])) + score
            truth[compound] = Y[idx]
            counts[compound] = counts.get(compound, 0) + 1
            fold_score_sums[compound] = fold_score_sums.get(
                compound, np.zeros(Y.shape[1])
            ) + score
            fold_truth[compound] = Y[idx]
            fold_counts[compound] = fold_counts.get(compound, 0) + 1

        fold_compounds = sorted(fold_score_sums)
        fold_y_true = np.stack([fold_truth[c] for c in fold_compounds])
        fold_scores = np.stack([
            fold_score_sums[c] / fold_counts[c] for c in fold_compounds
        ])
        fold_y_pred = np.zeros_like(fold_y_true)
        for i, score in enumerate(fold_scores):
            fold_y_pred[i, np.argsort(score)[-min(3, score.size):]] = 1
        fold_results.append({
            "fold": fold_idx,
            "n_compounds": len(fold_compounds),
            "macro_f1_top3": f1_score(
                fold_y_true, fold_y_pred, average="macro", zero_division=0
            ),
            "micro_f1_top3": f1_score(
                fold_y_true, fold_y_pred, average="micro", zero_division=0
            ),
        })

    compounds = sorted(score_sums)
    y_true = np.stack([truth[c] for c in compounds])
    compound_scores = np.stack([score_sums[c] / counts[c] for c in compounds])
    y_pred = np.zeros_like(y_true)
    for i, score in enumerate(compound_scores):
        y_pred[i, np.argsort(score)[-min(3, score.size):]] = 1
    result = {
        "protocol": protocol,
        "task": task,
        "feature_set": feature_set,
        "n_compounds": len(compounds),
        "n_labels": Y.shape[1],
        "macro_f1_top3": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1_top3": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }
    records = {
        "compound_id": compounds,
        "truth": {c: truth[c] for c in compounds},
    }
    if stability_rows is not None:
        for fold_result in fold_results:
            stability_rows.append({
                "record_type": "fold", "protocol": protocol, "task": task,
                "feature_set": feature_set, **fold_result,
            })

        # Bootstrap already-generated out-of-fold predictions by independent
        # units; this quantifies test-sample uncertainty, not model refit noise.
        unit_by_compound = {}
        if bootstrap_units is not None:
            unit_values = np.asarray(bootstrap_units)
            for compound in compounds:
                first = np.flatnonzero(groups == compound)[0]
                unit_by_compound[compound] = str(unit_values[first])
        else:
            unit_by_compound = {compound: compound for compound in compounds}
        unique_units = sorted(set(unit_by_compound.values()))
        bootstrap_unit_name = (
            "compound" if set(unique_units) == set(compounds) else "chemical_group"
        )
        compound_indices_by_unit = {
            unit: np.asarray([i for i, compound in enumerate(compounds)
                              if unit_by_compound[compound] == unit], dtype=int)
            for unit in unique_units
        }
        rng = np.random.default_rng(RANDOM_STATE)
        boot_macro, boot_micro = [], []
        for _ in range(n_bootstrap):
            sampled_units = rng.choice(unique_units, size=len(unique_units), replace=True)
            sampled_indices = np.concatenate([
                compound_indices_by_unit[unit] for unit in sampled_units
            ])
            boot_true = y_true[sampled_indices]
            boot_pred = y_pred[sampled_indices]
            true_positive = np.sum(boot_true & boot_pred, axis=0)
            false_positive = np.sum((~boot_true) & boot_pred, axis=0)
            false_negative = np.sum(boot_true & (~boot_pred), axis=0)
            denominator = 2 * true_positive + false_positive + false_negative
            per_label_f1 = np.divide(
                2 * true_positive, denominator,
                out=np.zeros_like(denominator, dtype=float), where=denominator > 0,
            )
            boot_macro.append(float(np.mean(per_label_f1)))
            boot_micro.append(float(
                2 * true_positive.sum() / denominator.sum()
                if denominator.sum() else 0.0
            ))

        macro_folds = [row["macro_f1_top3"] for row in fold_results]
        micro_folds = [row["micro_f1_top3"] for row in fold_results]
        stability_rows.append({
            "record_type": "summary", "protocol": protocol, "task": task,
            "feature_set": feature_set, "n_compounds": len(compounds),
            "n_folds_or_batches": len(fold_results),
            "macro_f1_top3_fold_mean": float(np.mean(macro_folds)),
            "macro_f1_top3_fold_sd": float(np.std(macro_folds, ddof=1)) if len(macro_folds) > 1 else 0.0,
            "macro_f1_top3_fold_min": float(np.min(macro_folds)),
            "macro_f1_top3_fold_max": float(np.max(macro_folds)),
            "micro_f1_top3_fold_mean": float(np.mean(micro_folds)),
            "micro_f1_top3_fold_sd": float(np.std(micro_folds, ddof=1)) if len(micro_folds) > 1 else 0.0,
            "micro_f1_top3_fold_min": float(np.min(micro_folds)),
            "micro_f1_top3_fold_max": float(np.max(micro_folds)),
            "macro_f1_top3_bootstrap_ci_lower": float(np.percentile(boot_macro, 2.5)),
            "macro_f1_top3_bootstrap_ci_upper": float(np.percentile(boot_macro, 97.5)),
            "micro_f1_top3_bootstrap_ci_lower": float(np.percentile(boot_micro, 2.5)),
            "micro_f1_top3_bootstrap_ci_upper": float(np.percentile(boot_micro, 97.5)),
            "bootstrap_unit": bootstrap_unit_name,
            "n_bootstrap": n_bootstrap,
        })
    return result, records


def fit_models_and_predict_unannotated(source_df: pd.DataFrame | None = None) -> None:
    """Fit final compound-level models and rank predictions for missing labels.

    Models are fit independently for pathway and target, so compounds with one
    annotation can still train the other task. Biological-activity descriptions
    are returned as ranked nearest-neighbor examples rather than generated text.
    """
    if source_df is None:
        source_df = pd.read_parquet(DATA_PATH)
        source_df["_source_row_index"] = np.arange(len(source_df))
    df = source_df[source_df["compound_id"].notna()].copy()
    if df.empty:
        print("No compounds available for final annotation fitting.")
        return

    embeddings = {
        "pca_normalized": np.stack(df["pca_embedding_normalized"].map(unpack_vector)),
        "pca_raw": np.stack(df["pca_embedding_raw"].map(unpack_vector)),
        "dino": np.stack(df["brightfield"].map(unpack_vector)),
    }
    assays = df[ASSAY_FEATURES].to_numpy(dtype=np.float32)
    assays[~np.isfinite(assays)] = np.nan
    inputs = {**embeddings, "assay_features": assays}
    # Final unannotated-compound models receive the same corrected variants as
    # the cross-validation comparisons when the prepared arrays are present.
    if "_source_row_index" in df.columns:
        inputs.update(load_corrected_feature_arrays(
            df["_source_row_index"].to_numpy(dtype=int),
            ["pca_raw", "pca_normalized", "dino", "assay_features"],
        ))
    inputs.update({
        f"{name}_plus_assays": np.concatenate([X, assays], axis=1)
        for name, X in embeddings.items()
    })

    compound_ids = df["compound_id"].astype(str).to_numpy()
    unique_ids = sorted(set(compound_ids))
    # Average replicate wells to avoid giving compounds with more wells more
    # influence during training or prediction.
    features_by_compound: dict[str, dict[str, np.ndarray]] = {}
    for name, matrix in inputs.items():
        tmp = pd.DataFrame(matrix)
        tmp["compound_id"] = compound_ids
        means = tmp.groupby("compound_id").mean(numeric_only=True)
        features_by_compound[name] = {
            cid: means.loc[cid].to_numpy(dtype=np.float32) for cid in unique_ids
        }

    model_dir = RESULTS_DIR / "annotation_prediction" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict] = []
    fitted_count = 0
    for task, column in [("pathway", "compound_pathway"), ("target", "compound_target")]:
        labels_by_compound = (
            df.groupby(df["compound_id"].astype(str))[column]
            .apply(lambda values: tuple(sorted({label for value in values
                                                  for label in parse_labels(value)})))
            .to_dict()
        )
        label_counts: dict[str, set[str]] = {}
        for cid, labels in labels_by_compound.items():
            for label in labels:
                label_counts.setdefault(label, set()).add(cid)
        kept_labels = sorted(label for label, ids in label_counts.items()
                             if len(ids) >= MIN_LABEL_COMPOUNDS)
        if not kept_labels:
            continue

        labeled_ids = sorted(cid for cid in unique_ids
                             if labels_by_compound.get(cid))
        unannotated_ids = sorted(cid for cid in unique_ids
                                 if not labels_by_compound.get(cid))
        Y = np.array([[int(label in labels_by_compound.get(cid, ()))
                       for label in kept_labels] for cid in labeled_ids], dtype=np.uint8)

        for feature_set, by_compound in features_by_compound.items():
            X_train_raw = np.stack([by_compound[cid] for cid in labeled_ids])
            X_query_raw = np.stack([by_compound[cid] for cid in unannotated_ids]) \
                if unannotated_ids else np.empty((0, X_train_raw.shape[1]), dtype=np.float32)
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw))
            def fit_final_label(j: int):
                label = kept_labels[j]
                y = Y[:, j]
                if len(np.unique(y)) < 2:
                    return label, None
                model = LogisticRegression(
                    C=0.1, class_weight="balanced", max_iter=2000,
                    random_state=RANDOM_STATE,
                )
                model.fit(X_train, y)
                return label, model

            with threadpool_limits(limits=1):
                fitted = Parallel(n_jobs=min(6, Y.shape[1]), prefer="threads")(
                    delayed(fit_final_label)(j) for j in range(Y.shape[1])
                )
            models = {label: model for label, model in fitted if model is not None}

            model_path = model_dir / f"{task}_{feature_set}.joblib"
            joblib.dump({
                "task": task, "feature_set": feature_set,
                "labels": list(models), "models": models,
                "imputer": imputer, "scaler": scaler,
                "n_labeled_compounds": len(labeled_ids),
                "min_label_compounds": MIN_LABEL_COMPOUNDS,
            }, model_path)
            fitted_count += 1
            if len(unannotated_ids) == 0:
                continue

            X_query = scaler.transform(imputer.transform(X_query_raw))
            for cid, x in zip(unannotated_ids, X_query):
                scored = [(label, float(model.predict_proba(x.reshape(1, -1))[0, 1]))
                          for label, model in models.items()]
                scored.sort(key=lambda item: item[1], reverse=True)
                for rank, (label, score) in enumerate(scored, start=1):
                    prediction_rows.append({
                        "task": task, "feature_set": feature_set,
                        "compound_id": cid, "rank": rank,
                        "predicted_label": label, "score": score,
                    })

    predictions_path = RESULTS_DIR / "annotation_prediction" / "unannotated_compound_ranked_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    # For the free-text field, show activity descriptions from the most similar
    # annotated compounds. This is evidence for review, not generated text.
    text_rows: list[dict] = []
    text_by_compound = (
        df.groupby(df["compound_id"].astype(str))["compound_biological_activity"]
        .apply(lambda values: next((str(v).strip() for v in values
                                    if pd.notna(v) and str(v).strip()), ""))
        .to_dict()
    )
    text_train_ids = sorted(cid for cid, value in text_by_compound.items() if value)
    text_query_ids = sorted(cid for cid in unique_ids if not text_by_compound.get(cid))
    for feature_set in inputs:
        by_compound = features_by_compound[feature_set]
        if not text_train_ids or not text_query_ids:
            continue
        train = np.stack([by_compound[cid] for cid in text_train_ids])
        query = np.stack([by_compound[cid] for cid in text_query_ids])
        train_norm = np.linalg.norm(train, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        similarities = (query / np.maximum(query_norm, 1e-12)) @ (
            train / np.maximum(train_norm, 1e-12)
        ).T
        for cid, sims in zip(text_query_ids, similarities):
            for rank, idx in enumerate(np.argsort(sims)[::-1][:5], start=1):
                neighbor_id = text_train_ids[idx]
                text_rows.append({
                    "feature_set": feature_set, "compound_id": cid,
                    "rank": rank, "neighbor_compound_id": neighbor_id,
                    "cosine_similarity": float(sims[idx]),
                    "neighbor_biological_activity": text_by_compound[neighbor_id],
                })
    text_path = RESULTS_DIR / "activity_text_retrieval" / "unannotated_compound_activity_text_neighbors.csv"
    pd.DataFrame(text_rows).to_csv(text_path, index=False)
    print(f"Saved {fitted_count} fitted final models to {model_dir}")
    print(f"Saved ranked pathway/target predictions to {predictions_path}")
    print(f"Saved biological-activity text neighbors to {text_path}")


def bh_adjust(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    return np.minimum(adjusted, 1.0)


def cluster_enrichment(embedding_by_compound: dict[str, np.ndarray],
                       truth_by_compound: dict[str, np.ndarray], labels: list[str],
                       protocol: str, embedding_name: str, task: str) -> list[dict]:
    compounds = sorted(set(embedding_by_compound) & set(truth_by_compound))
    if len(compounds) < 3:
        return []
    X = np.stack([embedding_by_compound[c] for c in compounds]).astype(np.float32)
    Y = np.stack([truth_by_compound[c] for c in compounds]).astype(bool)
    # Normalize each feature before unsupervised PCA/K-means so scale and
    # embedding dimension do not dominate cluster assignment.
    X = StandardScaler().fit_transform(X)
    n_components = min(50, X.shape[1], X.shape[0] - 1)
    if n_components >= 2 and X.shape[1] > n_components:
        X = PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(X)
    n_clusters = min(N_CLUSTERS, len(compounds))
    cluster_ids = KMeans(n_clusters=n_clusters, n_init=10,
                         random_state=RANDOM_STATE).fit_predict(X)
    rows = []
    for cluster in range(n_clusters):
        in_cluster = cluster_ids == cluster
        size = int(in_cluster.sum())
        for j, label in enumerate(labels):
            hits = int(Y[in_cluster, j].sum())
            total_hits = int(Y[:, j].sum())
            if hits == 0 or total_hits == 0:
                continue
            p_value = float(hypergeom.sf(hits - 1, len(compounds), total_hits, size))
            rows.append({
                "protocol": protocol, "embedding": embedding_name, "task": task,
                "cluster": cluster, "label": label,
                "n_cluster_compounds": size, "n_with_label": hits,
                "cluster_prevalence": hits / size,
                "overall_prevalence": total_hits / len(compounds),
                "fold_enrichment": (hits / size) / (total_hits / len(compounds)),
                "p_value": p_value,
            })
    if rows:
        rows_df = pd.DataFrame(rows)
        rows_df["q_value"] = bh_adjust(rows_df["p_value"].tolist())
        rows = rows_df.to_dict("records")
    return rows


def run_analysis(protocol_names: list[str] | None = None,
                 fit_final_models: bool = True) -> None:
    """Run one or both validation protocols and save matching output files."""
    if protocol_names is None:
        protocol_names = ["held_out_compound", "held_out_batch"]
    df = pd.read_parquet(DATA_PATH)
    df["_source_row_index"] = np.arange(len(df))
    df = df[df["compound_id"].notna()].copy()
    df = df[df["compound_pathway"].notna() & df["compound_target"].notna()].copy()
    df = df.reset_index(drop=True)
    groups = df["compound_id"].astype(str).to_numpy()

    embeddings = {
        "pca_normalized": np.stack(df["pca_embedding_normalized"].map(unpack_vector)),
        "pca_raw": np.stack(df["pca_embedding_raw"].map(unpack_vector)),
        # 768-dimensional image embedding stored in the parquet's brightfield column.
        "dino": np.stack(df["brightfield"].map(unpack_vector)),
    }
    assays = df[ASSAY_FEATURES].to_numpy(dtype=np.float32)
    assays[~np.isfinite(assays)] = np.nan
    inputs = {**embeddings, "assay_features": assays}
    inputs.update(load_corrected_feature_arrays(
        df["_source_row_index"].to_numpy(dtype=int),
        ["pca_raw", "pca_normalized", "dino", "assay_features"],
    ))
    inputs.update({
        f"{name}_plus_assays": np.concatenate([X, assays], axis=1)
        for name, X in embeddings.items()
    })
    inputs["label_frequency_baseline"] = np.zeros((len(df), 1), dtype=np.float32)
    available_splits = build_splits(df)
    unknown = set(protocol_names) - set(available_splits)
    if unknown:
        raise ValueError(f"Unknown validation protocol(s): {sorted(unknown)}")
    splits_by_protocol = {
        protocol: available_splits[protocol] for protocol in protocol_names
    }

    metric_rows, enrichment_rows, stability_rows = [], [], []
    for task, column in [("pathway", "compound_pathway"), ("target", "compound_target")]:
        labels = df[column].map(parse_labels).tolist()
        Y, kept_labels = make_label_matrix(labels, groups)
        for protocol, splits in splits_by_protocol.items():
            protocol_truth = None
            for feature_set, X in inputs.items():
                result, records = run_prediction(
                    X, Y, groups, splits, task, feature_set, protocol,
                    stability_rows=stability_rows,
                    bootstrap_units=groups,
                    label_frequency_baseline=(feature_set == "label_frequency_baseline"),
                )
                metric_rows.append(result)
                if protocol_truth is None:
                    protocol_truth = records["truth"]
                print(
                    f"{protocol:18s} {task:8s} {feature_set:22s} "
                    f"macro-F1@3={result['macro_f1_top3']:.3f} "
                    f"micro-F1@3={result['micro_f1_top3']:.3f} "
                    f"({result['n_compounds']} compounds)"
                )

            # Cluster only held-out rows (including only the held-out batch rows
            # for the batch protocol), averaged to one vector per compound.
            heldout_indices = np.concatenate([test for _, test in splits])
            for embedding_name, matrix in embeddings.items():
                heldout = pd.DataFrame(matrix[heldout_indices])
                heldout["compound_id"] = groups[heldout_indices]
                embedding_means = heldout.groupby("compound_id").mean(numeric_only=True)
                embedding_map = {
                    c: embedding_means.loc[c].to_numpy(dtype=np.float32)
                    for c in protocol_truth if c in embedding_means.index
                }
                enrichment_rows.extend(cluster_enrichment(
                    embedding_map, protocol_truth, kept_labels,
                    protocol, embedding_name, task,
                ))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if len(protocol_names) == 1:
        suffix = protocol_names[0]
        metrics_path = RESULTS_DIR / "annotation_prediction" / f"annotation_model_comparison_{suffix}.csv"
        enrichment_path = RESULTS_DIR / "cluster_enrichment" / f"cluster_label_enrichment_{suffix}.csv"
    else:
        metrics_path = RESULTS_DIR / "annotation_prediction" / "annotation_model_comparison.csv"
        enrichment_path = RESULTS_DIR / "cluster_enrichment" / "cluster_label_enrichment.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(enrichment_rows).to_csv(enrichment_path, index=False)
    if len(protocol_names) == 1:
        stability_path = RESULTS_DIR / "annotation_prediction" / f"annotation_prediction_stability_{protocol_names[0]}.csv"
    else:
        stability_path = RESULTS_DIR / "annotation_prediction" / "annotation_prediction_stability.csv"
    pd.DataFrame(stability_rows).to_csv(stability_path, index=False)
    print(f"\nSaved prediction comparison to {metrics_path}")
    print(f"Saved cluster enrichment results to {enrichment_path}")
    print(f"Saved fold and bootstrap stability results to {stability_path}")
    print("Interpret enrichment with q_value (multiple-testing adjusted); "
          "fold_enrichment > 1 means the label is more common in that cluster.")
    if fit_final_models:
        fit_models_and_predict_unannotated()
