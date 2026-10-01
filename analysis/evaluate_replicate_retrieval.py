"""Measure whether replicate wells retrieve one another in feature spaces.

For each query well, rank other wells by cosine similarity and ask whether
the nearest wells share its compound and dose. The three candidate pools are
unrestricted, restricted to the same batch, or restricted to the same plate.
This is a descriptive diagnostic (not a held-out prediction benchmark).

Run:
    uv run python analysis/evaluate_replicate_retrieval.py

Writes replicate_retrieval_metrics.csv, replicate_retrieval_per_query.csv,
replicate_retrieval_neighbors.csv, and replicate_retrieval.png under
results/replicate_analysis/.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from annotation_model_utils import ASSAY_FEATURES, DATA_PATH, RESULTS_DIR, unpack_vector


FEATURES = ("pca_normalized", "pca_raw", "dino", "assay_features")
POOLS = ("all_wells", "same_batch", "same_plate")
K_VALUES = (1, 5, 10)
MAX_QUERIES = 1000
RANDOM_STATE = 42


def load_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    result = {
        "pca_normalized": np.stack(df.pca_embedding_normalized.map(unpack_vector)),
        "pca_raw": np.stack(df.pca_embedding_raw.map(unpack_vector)),
        "dino": np.stack(df.brightfield.map(unpack_vector)),
        "assay_features": df[ASSAY_FEATURES].to_numpy(dtype=np.float32),
    }
    for name, X in result.items():
        X = X.astype(np.float32, copy=False)
        X[~np.isfinite(X)] = np.nan
        # Impute missing dimensions across wells. Standardize the assay
        # baseline because its columns have different units; image embeddings
        # retain their existing coordinate scales for cosine retrieval.
        X = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
        if name == "assay_features":
            X = StandardScaler().fit_transform(X).astype(np.float32)
        result[name] = X
    return result


def cosine_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)


def anchor_keys(df: pd.DataFrame) -> np.ndarray:
    dose = df.compound_concentration_um.astype("string").fillna("NA")
    return (df.compound_id.astype(str) + "||dose=" + dose).to_numpy()


def analyze_pool(df: pd.DataFrame, X: np.ndarray, feature: str,
                 pool: str, max_queries: int, rng: np.random.Generator):
    n = len(df)
    anchors = anchor_keys(df)
    if pool == "all_wells":
        strata = np.zeros(n, dtype=np.int8)
    else:
        column = "batch" if pool == "same_batch" else "plate"
        strata = df[column].astype(str).to_numpy()

    # Query wells need at least one other replicate within the candidate pool.
    query_candidates = []
    for value in pd.unique(strata):
        members = np.flatnonzero(strata == value)
        counts = pd.Series(anchors[members]).value_counts()
        repeated = set(counts[counts >= 2].index)
        query_candidates.extend(i for i in members if anchors[i] in repeated)
    query_candidates = np.asarray(query_candidates, dtype=int)
    if len(query_candidates) > max_queries:
        query_candidates = np.sort(rng.choice(query_candidates, max_queries, replace=False))

    normalized = cosine_rows(X)
    records, neighbor_records = [], []
    # Small query chunks bound memory while allowing exact cosine ranking.
    chunk_size = 64
    for start in range(0, len(query_candidates), chunk_size):
        queries = query_candidates[start:start + chunk_size]
        for qidx in queries:
            if pool == "all_wells":
                gallery_idx = np.arange(n)
            else:
                gallery_idx = np.flatnonzero(strata == strata[qidx])
            gallery_idx = gallery_idx[gallery_idx != qidx]
            if len(gallery_idx) == 0:
                continue
            sims = normalized[gallery_idx] @ normalized[qidx]
            order = np.argsort(-sims, kind="stable")
            sorted_idx = gallery_idx[order]
            sorted_sims = sims[order]
            relevant = anchors[sorted_idx] == anchors[qidx]
            relevant_positions = np.flatnonzero(relevant)
            if not len(relevant_positions):
                continue
            first_rank = int(relevant_positions[0] + 1)
            anchor_count = int(relevant.sum())
            n_gallery = len(gallery_idx)
            row = {
                "feature_set": feature, "candidate_pool": pool,
                "query_well_id": str(df.iloc[qidx].well_id),
                "compound_id": str(df.iloc[qidx].compound_id),
                "dose_um": df.iloc[qidx].compound_concentration_um,
                "plate": str(df.iloc[qidx].plate), "batch": str(df.iloc[qidx].batch),
                "n_gallery": n_gallery, "n_same_anchor_gallery": anchor_count,
                "first_replicate_rank": first_rank,
                "reciprocal_rank": 1.0 / first_rank,
                "nearest_replicate_cosine": float(sorted_sims[relevant_positions[0]]),
                "nearest_nonreplicate_cosine": float(sorted_sims[~relevant][0]) if (~relevant).any() else np.nan,
                "random_replicate_rate": anchor_count / n_gallery,
                "random_precision_at_5": anchor_count / n_gallery,
                "random_hit_at_5": 1.0 - (1.0 - anchor_count / n_gallery) ** min(5, n_gallery),
            }
            for k in K_VALUES:
                k_eff = min(k, n_gallery)
                row[f"replicate_precision_at_{k}"] = float(relevant[:k_eff].sum() / k_eff)
                row[f"replicate_hit_at_{k}"] = float(relevant[:k_eff].any())
            records.append(row)
            for rank in range(min(10, n_gallery)):
                j = sorted_idx[rank]
                neighbor_records.append({
                    "feature_set": feature, "candidate_pool": pool,
                    "query_well_id": str(df.iloc[qidx].well_id),
                    "query_compound_id": str(df.iloc[qidx].compound_id),
                    "query_dose_um": df.iloc[qidx].compound_concentration_um,
                    "rank": rank + 1,
                    "neighbor_well_id": str(df.iloc[j].well_id),
                    "neighbor_compound_id": str(df.iloc[j].compound_id),
                    "neighbor_dose_um": df.iloc[j].compound_concentration_um,
                    "cosine_similarity": float(sorted_sims[rank]),
                    "same_compound_and_dose": bool(relevant[rank]),
                    "same_plate": str(df.iloc[qidx].plate) == str(df.iloc[j].plate),
                    "same_batch": str(df.iloc[qidx].batch) == str(df.iloc[j].batch),
                })
    return records, neighbor_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", choices=FEATURES, default=list(FEATURES))
    parser.add_argument("--max-queries", type=int, default=MAX_QUERIES,
                        help="Maximum query wells per feature and candidate pool; use 0 for all.")
    args = parser.parse_args()
    df = pd.read_parquet(DATA_PATH)
    # No-annotation compounds are valid replicate queries. DMSO/control wells
    # are omitted because they do not have a compound+dose anchor.
    df = df[df.compound_id.notna() & df.compound_concentration_um.notna()].copy().reset_index(drop=True)
    feature_arrays = load_features(df)
    rng = np.random.default_rng(RANDOM_STATE)
    per_query, neighbors = [], []
    for feature in args.features:
        X = feature_arrays[feature]
        for pool in POOLS:
            records, neighbor_rows = analyze_pool(
                df, X, feature, pool, args.max_queries or len(df), rng
            )
            per_query.extend(records)
            neighbors.extend(neighbor_rows)
            print(f"{feature:16s} {pool:12s} queries={len(records)}")

    query_df = pd.DataFrame(per_query)
    if query_df.empty:
        raise RuntimeError("No repeated compound-dose profiles found")
    summary = query_df.groupby(["feature_set", "candidate_pool"], as_index=False).agg(
        n_queries=("query_well_id", "size"),
        mean_replicate_precision_at_1=("replicate_precision_at_1", "mean"),
        mean_replicate_precision_at_5=("replicate_precision_at_5", "mean"),
        mean_replicate_precision_at_10=("replicate_precision_at_10", "mean"),
        replicate_hit_at_1=("replicate_hit_at_1", "mean"),
        replicate_hit_at_5=("replicate_hit_at_5", "mean"),
        replicate_hit_at_10=("replicate_hit_at_10", "mean"),
        mean_reciprocal_rank=("reciprocal_rank", "mean"),
        mean_nearest_replicate_cosine=("nearest_replicate_cosine", "mean"),
        mean_nearest_nonreplicate_cosine=("nearest_nonreplicate_cosine", "mean"),
        mean_random_replicate_rate=("random_replicate_rate", "mean"),
        random_precision_at_5=("random_precision_at_5", "mean"),
        random_hit_at_5=("random_hit_at_5", "mean"),
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "replicate_analysis" / "replicate_retrieval_metrics.csv", index=False)
    query_df.to_csv(RESULTS_DIR / "replicate_analysis" / "replicate_retrieval_per_query.csv", index=False)
    pd.DataFrame(neighbors).to_csv(RESULTS_DIR / "replicate_analysis" / "replicate_retrieval_neighbors.csv", index=False)
    plot_summary(summary)
    print(summary.to_string(index=False))
    print(f"Saved replicate retrieval files under {RESULTS_DIR}")


def plot_summary(summary: pd.DataFrame) -> None:
    features = [f for f in FEATURES if f in set(summary.feature_set)]
    pools = [p for p in POOLS if p in set(summary.candidate_pool)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    metrics = [("mean_replicate_precision_at_5", "Precision@5", "random_precision_at_5"),
               ("replicate_hit_at_5", "Hit rate@5", "random_hit_at_5")]
    colors = {"all_wells": "#4c78a8", "same_batch": "#f58518", "same_plate": "#54a24b"}
    x = np.arange(len(features))
    width = 0.24
    for ax, (metric, title, baseline) in zip(axes, metrics):
        for j, pool in enumerate(pools):
            values = summary.set_index(["feature_set", "candidate_pool"])[metric]
            ys = [values.get((feature, pool), np.nan) for feature in features]
            ax.bar(x + (j - (len(pools) - 1) / 2) * width, ys, width,
                   label=pool.replace("_", " "), color=colors[pool])
            random_values = summary.set_index(["feature_set", "candidate_pool"])[baseline]
            baseline_ys = [random_values.get((feature, pool), np.nan) for feature in features]
            ax.scatter(x + (j - (len(pools) - 1) / 2) * width, baseline_ys,
                       marker="_", s=180, color="black", linewidths=2, zorder=4)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x, [f.replace("_", " ") for f in features], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=.25)
        ax.tick_params(axis="both", labelsize=11)
    axes[0].set_ylabel("Fraction of retrieved wells", fontsize=12)
    axes[1].legend(frameon=False, fontsize=11, loc="upper left")
    fig.text(.5, .01, "Black ticks show the expected score from random retrieval at the observed replicate frequency.",
             ha="center", fontsize=10)
    fig.suptitle("Replicate Retrieval by Feature Space", fontsize=16, y=.99)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(RESULTS_DIR / "replicate_analysis" / "replicate_retrieval.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
