"""Evaluate whether embedding neighbors share known pathway/target annotations.

For each held-out query compound, retrieve the most similar *training*
compounds by cosine similarity between compound-averaged embeddings. Runs
compound-, batch-, and chemical-group-held-out validation for assay features,
normalized PCA, raw PCA, and DINO embeddings.

Run from the repository root:
    uv run python analysis/evaluate_embedding_retrieval.py

Outputs:
    results/annotation_retrieval/embedding_retrieval_metrics.csv
    results/annotation_retrieval/embedding_retrieval_neighbors.csv
"""
from __future__ import annotations

from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from annotation_model_utils import (
    ASSAY_FEATURES,
    DATA_PATH,
    MIN_LABEL_COMPOUNDS,
    N_SPLITS,
    RESULTS_DIR,
    load_corrected_feature_arrays,
    parse_labels,
    unpack_vector,
)
from compare_annotations_chemical_split import chemical_groups, get_structure_table


K_VALUES = (1, 3, 5, 10)
MAX_K = max(K_VALUES)
PROTOCOLS = ("held_out_compound", "held_out_batch", "held_out_chemical_group")


def build_splits(df: pd.DataFrame, protocol: str,
                 chemical_group_by_compound: dict[str, int] | None = None):
    compound_ids = df["compound_id"].astype(str).to_numpy()
    if protocol == "held_out_compound":
        return list(GroupKFold(n_splits=N_SPLITS).split(df, groups=compound_ids))
    if protocol == "held_out_chemical_group":
        if chemical_group_by_compound is None:
            raise ValueError("Chemical-group mapping is required for this protocol")
        groups = np.array([chemical_group_by_compound[c] for c in compound_ids])
        return list(GroupKFold(n_splits=N_SPLITS).split(df, groups=groups))
    if protocol == "held_out_batch":
        batches = df["batch"].to_numpy()
        splits = []
        for batch in sorted(pd.unique(batches)):
            test = np.flatnonzero(batches == batch)
            test_compounds = set(compound_ids[test])
            train = np.flatnonzero(
                (batches != batch) & ~np.isin(compound_ids, list(test_compounds))
            )
            if len(train) and len(test):
                splits.append((train, test))
        return splits
    raise ValueError(f"Unknown protocol: {protocol}")


def compound_label_sets(df: pd.DataFrame, column: str) -> tuple[dict[str, set[str]], set[str]]:
    by_compound: dict[str, set[str]] = {}
    for compound, values in df.groupby(df["compound_id"].astype(str))[column]:
        by_compound[compound] = set(chain.from_iterable(values.map(parse_labels)))
    counts: dict[str, int] = {}
    for labels in by_compound.values():
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    keep = {label for label, count in counts.items() if count >= MIN_LABEL_COMPOUNDS}
    filtered = {
        compound: labels & keep for compound, labels in by_compound.items()
    }
    return filtered, keep


def mean_profiles(indices: np.ndarray, X: np.ndarray,
                  compound_ids: np.ndarray) -> dict[str, np.ndarray]:
    profiles = {}
    for compound in np.unique(compound_ids[indices]):
        rows = indices[compound_ids[indices] == compound]
        profiles[str(compound)] = X[rows].mean(axis=0)
    return profiles


def make_queries(protocol: str, fold_id: str, test: np.ndarray,
                 df: pd.DataFrame, X: np.ndarray, compound_ids: np.ndarray):
    if protocol == "held_out_batch":
        queries = []
        test_batches = df.iloc[test]["batch"].to_numpy()
        for compound in np.unique(compound_ids[test]):
            for batch in sorted(pd.unique(test_batches[compound_ids[test] == compound])):
                rows = test[(compound_ids[test] == compound) & (df["batch"].to_numpy()[test] == batch)]
                queries.append({
                    "query_id": f"{fold_id}|{compound}|{batch}",
                    "compound_id": str(compound),
                    "batch": str(batch),
                    "profile": X[rows].mean(axis=0),
                })
        return queries

    profiles = mean_profiles(test, X, compound_ids)
    return [
        {
            "query_id": f"{fold_id}|{compound}",
            "compound_id": compound,
            "batch": "",
            "profile": profile,
        }
        for compound, profile in profiles.items()
    ]


def score_queries(queries: list[dict], gallery: dict[str, np.ndarray],
                  labels: dict[str, set[str]], protocol: str, fold_id: str,
                  embedding: str, task: str):
    gallery_ids = [c for c in gallery if labels.get(c)]
    if not queries or not gallery_ids:
        return [], []

    gallery_matrix = np.stack([gallery[c] for c in gallery_ids]).astype(np.float32)
    gallery_norms = np.linalg.norm(gallery_matrix, axis=1, keepdims=True)
    gallery_matrix = np.divide(
        gallery_matrix, gallery_norms, out=np.zeros_like(gallery_matrix),
        where=gallery_norms > 0,
    )
    metric_rows, neighbor_rows = [], []
    for query in queries:
        query_id = query["compound_id"]
        query_labels = labels.get(query_id, set())
        if not query_labels:
            continue
        q = query["profile"].astype(np.float32)
        qnorm = np.linalg.norm(q)
        if qnorm > 0:
            q = q / qnorm
        similarities = gallery_matrix @ q
        order = np.argsort(-similarities, kind="stable")[:MAX_K]
        neighbors = [gallery_ids[i] for i in order]
        neighbor_labels = [labels[c] for c in neighbors]
        relevant = [bool(query_labels & neighbor_label) for neighbor_label in neighbor_labels]
        recovered_labels: set[str] = set()

        for rank, (neighbor, sim, shared) in enumerate(
            zip(neighbors, similarities[order], neighbor_labels), start=1
        ):
            overlap = query_labels & shared
            neighbor_rows.append({
                "protocol": protocol,
                "fold": fold_id,
                "embedding": embedding,
                "task": task,
                "query_id": query["query_id"],
                "query_compound_id": query_id,
                "query_batch": query["batch"],
                "query_labels": "; ".join(sorted(query_labels)),
                "rank": rank,
                "neighbor_compound_id": neighbor,
                "cosine_similarity": float(sim),
                "neighbor_labels": "; ".join(sorted(shared)),
                "shared_labels": "; ".join(sorted(overlap)),
                "shares_annotation": bool(overlap),
            })

        total_relevant = sum(labels[c] & query_labels != set() for c in gallery_ids)
        for k in K_VALUES:
            n = min(k, len(neighbors))
            top_labels = set(chain.from_iterable(neighbor_labels[:n]))
            recovered_labels.update(top_labels & query_labels)
            n_relevant = sum(relevant[:n])
            precision = n_relevant / n if n else np.nan
            recall = len(top_labels & query_labels) / len(query_labels)
            ap_numerator = sum(
                sum(relevant[:rank]) / rank
                for rank in range(1, n + 1) if relevant[rank - 1]
            )
            average_precision = (
                ap_numerator / min(total_relevant, n)
                if total_relevant and n else np.nan
            )
            metric_rows.append({
                "protocol": protocol,
                "fold": fold_id,
                "embedding": embedding,
                "task": task,
                "k": k,
                "query_id": query["query_id"],
                "query_compound_id": query_id,
                "neighbor_precision_at_k": precision,
                "annotation_recall_at_k": recall,
                "map_at_k": average_precision,
                "n_query_labels": len(query_labels),
                "n_gallery_compounds": len(gallery_ids),
            })
    return metric_rows, neighbor_rows


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df["_source_row_index"] = np.arange(len(df))
    df = df[df["compound_id"].notna()].copy().reset_index(drop=True)
    compound_ids = df["compound_id"].astype(str).to_numpy()

    embeddings = {
        "pca_normalized": np.stack(df["pca_embedding_normalized"].map(unpack_vector)),
        "pca_raw": np.stack(df["pca_embedding_raw"].map(unpack_vector)),
        "dino": np.stack(df["brightfield"].map(unpack_vector)),
        "assay_features": df[ASSAY_FEATURES].to_numpy(dtype=np.float32),
    }
    embeddings.update(load_corrected_feature_arrays(
        df["_source_row_index"].to_numpy(dtype=int),
        ["pca_raw", "pca_normalized", "dino", "assay_features"],
    ))
    embeddings["assay_features"][~np.isfinite(embeddings["assay_features"])] = np.nan

    structure_groups = None
    if "held_out_chemical_group" in PROTOCOLS:
        compound_names = sorted(set(compound_ids))
        structures = get_structure_table(compound_names)
        structure_groups, _ = chemical_groups(structures)

    metric_rows, neighbor_rows = [], []
    for protocol in PROTOCOLS:
        available_compounds = set(compound_ids)
        if protocol == "held_out_chemical_group":
            available_compounds &= set(structure_groups or {})
            keep_rows = np.array([c in available_compounds for c in compound_ids])
            protocol_df = df.loc[keep_rows].reset_index(drop=True)
            protocol_ids = protocol_df["compound_id"].astype(str).to_numpy()
            protocol_embeddings = {
                name: matrix[keep_rows] for name, matrix in embeddings.items()
            }
            group_mapping = structure_groups
        else:
            protocol_df = df
            protocol_ids = compound_ids
            protocol_embeddings = embeddings
            group_mapping = None

        splits = build_splits(protocol_df, protocol, group_mapping)
        for task, column in [("pathway", "compound_pathway"),
                             ("target", "compound_target")]:
            labels, kept_labels = compound_label_sets(protocol_df, column)
            for fold_num, (train, test) in enumerate(splits, start=1):
                fold_id = str(protocol_df.iloc[test]["batch"].iloc[0]) if protocol == "held_out_batch" else str(fold_num)
                for embedding_name, matrix in protocol_embeddings.items():
                    gallery = mean_profiles(train, matrix, protocol_ids)
                    queries = make_queries(
                        protocol, fold_id, test, protocol_df, matrix, protocol_ids
                    )
                    if embedding_name.startswith("assay_features") and gallery and queries:
                        # Fit preprocessing on training compounds only before
                        # using cosine similarity for the assay baseline.
                        gallery_ids = list(gallery)
                        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
                        scaler = StandardScaler()
                        gallery_x = scaler.fit_transform(
                            imputer.fit_transform(np.stack([gallery[c] for c in gallery_ids]))
                        )
                        gallery = {c: x for c, x in zip(gallery_ids, gallery_x)}
                        query_x = scaler.transform(imputer.transform(
                            np.stack([q["profile"] for q in queries])
                        ))
                        for query, x in zip(queries, query_x):
                            query["profile"] = x
                    metrics, neighbors = score_queries(
                        queries, gallery, labels, protocol, fold_id,
                        embedding_name, task,
                    )
                    metric_rows.extend(metrics)
                    neighbor_rows.extend(neighbors)

    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise ValueError("No retrieval results were produced")
    summary = (
        metrics.groupby(["protocol", "embedding", "task", "k"], as_index=False)
        .agg(
            n_queries=("query_id", "nunique"),
            neighbor_precision_at_k=("neighbor_precision_at_k", "mean"),
            annotation_recall_at_k=("annotation_recall_at_k", "mean"),
            map_at_k=("map_at_k", "mean"),
        )
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "annotation_retrieval" / "embedding_retrieval_metrics.csv"
    neighbors_path = RESULTS_DIR / "annotation_retrieval" / "embedding_retrieval_neighbors.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(neighbor_rows).to_csv(neighbors_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved metrics to {summary_path}")
    print(f"Saved top-{MAX_K} neighbors per query to {neighbors_path}")


if __name__ == "__main__":
    main()
