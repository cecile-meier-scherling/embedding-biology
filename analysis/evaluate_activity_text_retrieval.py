"""Evaluate retrieval of similar biological-activity descriptions.

For a held-out compound, retrieve compounds with similar assay/PCA/DINO
profiles and score whether their biological-activity descriptions resemble
the query description. Text similarity uses TF-IDF fit on training compounds
only. The output measures similarity of descriptions, not correctness of a
mechanistic annotation.

Run from the repository root:
    uv run python analysis/evaluate_activity_text_retrieval.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, normalize

from annotation_model_utils import (
    ASSAY_FEATURES, DATA_PATH, RESULTS_DIR, load_corrected_feature_arrays,
    unpack_vector,
)
from compare_annotations_chemical_split import chemical_groups, get_structure_table


TEXT_COLUMN = "compound_biological_activity"
K_VALUES = (1, 3, 5, 10)
PROTOCOLS = ("held_out_compound", "held_out_batch", "held_out_chemical_group")
INPUTS = ("assay_features", "pca_raw", "pca_normalized", "dino",
          "pca_raw_plate_corrected", "pca_raw_batch_corrected",
          "pca_raw_plate_batch_corrected",
          "pca_normalized_plate_corrected", "pca_normalized_batch_corrected",
          "pca_normalized_plate_batch_corrected",
          "dino_plate_corrected", "dino_batch_corrected",
          "dino_plate_batch_corrected",
          "assay_features_plate_corrected", "assay_features_batch_corrected",
          "assay_features_plate_batch_corrected")


def text_by_compound(df: pd.DataFrame) -> dict[str, str]:
    return df.groupby(df["compound_id"].astype(str))[TEXT_COLUMN].apply(
        lambda values: next((" ".join(str(v).split()) for v in values
                             if pd.notna(v) and str(v).strip()), "")
    ).to_dict()


def split_indices(df: pd.DataFrame, protocol: str,
                  chemical_group_map: dict[str, int] | None = None):
    ids = df["compound_id"].astype(str).to_numpy()
    if protocol == "held_out_compound":
        return list(GroupKFold(n_splits=5).split(df, groups=ids))
    if protocol == "held_out_chemical_group":
        groups = np.array([chemical_group_map[c] for c in ids])
        return list(GroupKFold(n_splits=5).split(df, groups=groups))
    splits = []
    batches = df["batch"].to_numpy()
    for batch in sorted(pd.unique(batches)):
        test = np.flatnonzero(batches == batch)
        test_ids = set(ids[test])
        train = np.flatnonzero((batches != batch) & ~np.isin(ids, list(test_ids)))
        if len(train) and len(test):
            splits.append((train, test))
    return splits


def compound_profiles(X: np.ndarray, ids: np.ndarray,
                      row_indices: np.ndarray) -> dict[str, np.ndarray]:
    frame = pd.DataFrame(X[row_indices])
    frame["compound_id"] = ids[row_indices]
    return {str(cid): row.to_numpy(dtype=np.float32)
            for cid, row in frame.groupby("compound_id").mean(numeric_only=True).iterrows()}


def make_text_vectorizer(train_ids: list[str], texts: dict[str, str]):
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        ngram_range=(1, 2), min_df=1, max_features=100_000, sublinear_tf=True,
    )
    train_text = [texts[c] for c in train_ids]
    try:
        train_tfidf = vectorizer.fit_transform(train_text)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
        return None, None
    return vectorizer, normalize(train_tfidf)


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(DATA_PATH)
    df["_source_row_index"] = np.arange(len(df))
    df = df[df["compound_id"].notna() & df[TEXT_COLUMN].notna()].copy().reset_index(drop=True)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df[TEXT_COLUMN].ne("")].reset_index(drop=True)
    texts = text_by_compound(df)
    ids = df["compound_id"].astype(str).to_numpy()

    arrays = {
        "pca_raw": np.stack(df["pca_embedding_raw"].map(unpack_vector)),
        "pca_normalized": np.stack(df["pca_embedding_normalized"].map(unpack_vector)),
        "dino": np.stack(df["brightfield"].map(unpack_vector)),
        "assay_features": df[list(ASSAY_FEATURES)].to_numpy(dtype=np.float32),
    }
    arrays.update(load_corrected_feature_arrays(
        df["_source_row_index"].to_numpy(dtype=int),
        ["pca_raw", "pca_normalized", "dino", "assay_features"],
    ))
    arrays["assay_features"][~np.isfinite(arrays["assay_features"])] = np.nan

    chemical_map = None
    compound_list = sorted(set(ids))
    structure_table = get_structure_table(compound_list)
    chemical_map, _ = chemical_groups(structure_table)

    metric_rows, neighbor_rows = [], []
    for protocol in PROTOCOLS:
        protocol_df = df
        protocol_ids = ids
        protocol_arrays = arrays
        group_map = None
        if protocol == "held_out_chemical_group":
            allowed = set(chemical_map)
            keep = np.array([cid in allowed for cid in ids])
            protocol_df = df.loc[keep].reset_index(drop=True)
            protocol_ids = ids[keep]
            protocol_arrays = {name: matrix[keep] for name, matrix in arrays.items()}
            group_map = chemical_map

        splits = split_indices(protocol_df, protocol, group_map)
        for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
            fold = (str(protocol_df.iloc[test_idx]["batch"].iloc[0])
                    if protocol == "held_out_batch" else str(fold_num))
            gallery_text_ids = sorted(set(protocol_ids[train_idx]) & set(texts))
            gallery_text_ids = [c for c in gallery_text_ids if texts[c]]
            if len(gallery_text_ids) < 2:
                continue
            vectorizer, gallery_tfidf = make_text_vectorizer(gallery_text_ids, texts)
            if vectorizer is None:
                continue
            query_ids = sorted(set(protocol_ids[test_idx]) & set(texts))
            query_ids = [c for c in query_ids if texts[c] and c not in set(gallery_text_ids)]
            query_tfidf = normalize(vectorizer.transform([texts[c] for c in query_ids]))
            text_sim = (query_tfidf @ gallery_tfidf.T).toarray()
            random_reference = float(text_sim.mean()) if text_sim.size else np.nan

            test_batches = protocol_df.iloc[test_idx]["batch"].astype(str).to_numpy()
            for feature in INPUTS:
                train_profiles = compound_profiles(protocol_arrays[feature], protocol_ids, train_idx)
                if feature.startswith("assay_features"):
                    # Fit preprocessing on the training compound profiles only.
                    train_order = [c for c in gallery_text_ids if c in train_profiles]
                    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
                    scaler = StandardScaler()
                    train_values = scaler.fit_transform(imputer.fit_transform(
                        np.stack([train_profiles[c] for c in train_order])))
                    gallery_vec = normalize(train_values)
                    profile_map = {c: gallery_vec[i] for i, c in enumerate(train_order)}
                else:
                    profile_map = train_profiles

                # Each held-out batch is a distinct query; compound-held-out
                # protocols use one query profile per compound.
                query_units = []
                if protocol == "held_out_batch":
                    for cid in sorted(set(protocol_ids[test_idx])):
                        for batch in sorted(set(test_batches[protocol_ids[test_idx] == cid])):
                            rows = test_idx[(protocol_ids[test_idx] == cid) & (test_batches == batch)]
                            query_units.append((cid, batch, rows))
                else:
                    for cid in sorted(set(protocol_ids[test_idx])):
                        rows = test_idx[protocol_ids[test_idx] == cid]
                        query_units.append((cid, "", rows))

                gallery_order = [c for c in gallery_text_ids if c in profile_map]
                if not gallery_order:
                    continue
                gallery_matrix = np.stack([profile_map[c] for c in gallery_order]).astype(np.float32)
                gallery_norm = np.linalg.norm(gallery_matrix, axis=1, keepdims=True)
                gallery_matrix = gallery_matrix / np.maximum(gallery_norm, 1e-12)
                for cid, batch, rows in query_units:
                    if cid not in query_ids:
                        continue
                    q = protocol_arrays[feature][rows].mean(axis=0, keepdims=True)
                    if feature.startswith("assay_features"):
                        q = scaler.transform(imputer.transform(q))
                    q = q.astype(np.float32)
                    q /= max(float(np.linalg.norm(q)), 1e-12)
                    similarities = gallery_matrix @ q[0]
                    ordered = np.argsort(-similarities, kind="stable")
                    neighbors = [gallery_order[i] for i in ordered]
                    # Match the text-similarity columns to gallery compounds.
                    text_col = {c: i for i, c in enumerate(gallery_text_ids)}
                    qrow = query_ids.index(cid)
                    pair_scores = np.array([text_sim[qrow, text_col[n]] for n in neighbors])
                    for rank, ni in enumerate(ordered[:10], start=1):
                        neighbor = gallery_order[ni]
                        neighbor_rows.append({
                            "protocol": protocol, "fold": fold, "feature_set": feature,
                            "query_compound_id": cid, "query_batch": batch, "rank": rank,
                            "neighbor_compound_id": neighbor,
                            "embedding_cosine_similarity": float(similarities[ni]),
                            "activity_text_tfidf_similarity": float(text_sim[qrow, text_col[neighbor]]),
                        })
                    for k in K_VALUES:
                        top = pair_scores[:min(k, len(pair_scores))]
                        metric_rows.append({
                            "protocol": protocol, "fold": fold, "feature_set": feature,
                            "query_compound_id": cid, "query_batch": batch, "k": k,
                            "mean_activity_text_similarity_at_k": float(top.mean()),
                            "best_activity_text_similarity_at_k": float(top.max()),
                            "random_gallery_mean_activity_text_similarity": random_reference,
                        })

    per_query = pd.DataFrame(metric_rows)
    if per_query.empty:
        raise RuntimeError("No biological-activity retrieval results were produced")
    summary = per_query.groupby(["protocol", "feature_set", "k"], as_index=False).agg(
        n_queries=("query_compound_id", "nunique"),
        mean_activity_text_similarity_at_k=("mean_activity_text_similarity_at_k", "mean"),
        best_activity_text_similarity_at_k=("best_activity_text_similarity_at_k", "mean"),
        random_gallery_mean_activity_text_similarity=("random_gallery_mean_activity_text_similarity", "mean"),
    )
    summary.to_csv(RESULTS_DIR / "activity_text_retrieval" / "biological_activity_retrieval_metrics.csv", index=False)
    pd.DataFrame(neighbor_rows).to_csv(RESULTS_DIR / "activity_text_retrieval" / "biological_activity_retrieval_neighbors.csv", index=False)
    per_query.to_csv(RESULTS_DIR / "activity_text_retrieval" / "biological_activity_retrieval_per_query.csv", index=False)
    return summary, per_query


if __name__ == "__main__":
    summary, _ = evaluate()
    print(summary.to_string(index=False))
    print(f"\nSaved biological activity retrieval results under {RESULTS_DIR}")
