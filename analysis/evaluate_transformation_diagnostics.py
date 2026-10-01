"""Compare processed embeddings on replicates, annotation neighborhoods and confounds.

The transformation is fit on each outer training fold and applied unchanged to
held-out wells. We then measure held-out replicate similarity, retrieve known
pathway/target annotations from training compounds, cluster held-out compound
profiles and calculate label enrichment, and probe plate/batch/source from
within-compound residuals.

Run:
    uv run python analysis/evaluate_transformation_diagnostics.py
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from annotation_model_utils import (
    DATA_PATH, MIN_LABEL_COMPOUNDS, N_SPLITS, RESULTS_DIR, cluster_enrichment,
    parse_labels, unpack_vector,
)
from evaluate_embedding_retrieval import (
    compound_label_sets, make_queries, mean_profiles, score_queries,
)
from evaluate_processing_candidates import (
    FEATURES, TRANSFORMS, fit_transform_fold, nuisance_scores,
)


EMBEDDINGS = ("pca_raw", "pca_normalized", "dino")
PROCESSING = ("uncorrected", "matched_plate", "repeatable_top75", "pca50_whiten", "rank_gaussian")
PROTOCOLS = ("held_out_compound", "held_out_chemical_group")
NUISANCE = ("plate", "batch", "source")
K_VALUES = (1, 3, 5, 10)
MAX_RAND_PAIRS = 20000


def load_data():
    df = pd.read_parquet(DATA_PATH)
    df = df[df.compound_id.notna() & df.compound_pathway.notna() & df.compound_target.notna()].copy().reset_index(drop=True)
    ids = df.compound_id.astype(str).to_numpy()
    inputs = {
        "pca_raw": np.stack(df.pca_embedding_raw.map(unpack_vector)).astype(np.float32),
        "pca_normalized": np.stack(df.pca_embedding_normalized.map(unpack_vector)).astype(np.float32),
        "dino": np.stack(df.brightfield.map(unpack_vector)).astype(np.float32),
    }
    for matrix in inputs.values():
        matrix[~np.isfinite(matrix)] = np.nan
    meta = df[["compound_id", "compound_concentration_um", "plate", "batch", "source"]].reset_index(drop=True)
    return df, ids, inputs, meta


def make_protocol_data(df, ids, inputs, protocol, chemical_map):
    if protocol == "held_out_compound":
        current_df, current_ids = df, ids
        current_inputs = inputs
        outer_groups = current_ids
    else:
        keep = np.array([cid in chemical_map for cid in ids])
        current_df = df.loc[keep].reset_index(drop=True)
        current_ids = ids[keep]
        current_inputs = {name: X[keep] for name, X in inputs.items()}
        outer_groups = np.array([chemical_map[cid] for cid in current_ids])
    splits = list(GroupKFold(N_SPLITS).split(current_df, groups=outer_groups))
    return current_df, current_ids, current_inputs, outer_groups, splits


def label_truth(df, compound_ids, task_column):
    by_compound = {}
    for cid, vals in df.groupby(compound_ids)[task_column]:
        by_compound[str(cid)] = set(label for value in vals for label in parse_labels(value))
    counts = {}
    for labels in by_compound.values():
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    keep = {label for label, count in counts.items() if count >= MIN_LABEL_COMPOUNDS}
    return {cid: labels & keep for cid, labels in by_compound.items()}, sorted(keep)


def replicate_pair_metrics(X, meta, ids, rng):
    """Summarize within-anchor cosine similarity and matched negative pairs."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    unit = np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)
    dose = meta.compound_concentration_um.astype("string").fillna("NA").to_numpy()
    anchors = ids.astype(str) + "||" + dose
    plates = meta.plate.astype(str).to_numpy()
    same_anchor, same_anchor_same_plate, same_anchor_cross_plate = [], [], []
    for anchor in pd.unique(anchors):
        rows = np.flatnonzero(anchors == anchor)
        if len(rows) < 2:
            continue
        sims = unit[rows] @ unit[rows].T
        for i, j in combinations(range(len(rows)), 2):
            same_anchor.append(float(sims[i, j]))
            if plates[rows[i]] == plates[rows[j]]:
                same_anchor_same_plate.append(float(sims[i, j]))
            else:
                same_anchor_cross_plate.append(float(sims[i, j]))

    # Draw matched negative pairs without allocating an all-well similarity matrix.
    n = len(ids)
    a = rng.integers(0, n, size=MAX_RAND_PAIRS * 4)
    b = rng.integers(0, n, size=MAX_RAND_PAIRS * 4)
    valid = (a != b) & (anchors[a] != anchors[b])
    a, b = a[valid][:MAX_RAND_PAIRS], b[valid][:MAX_RAND_PAIRS]
    same_compound_different_dose = ids[a] == ids[b]
    different_compound = ~same_compound_different_dose
    sims = np.sum(unit[a] * unit[b], axis=1) if len(a) else np.array([])
    return {
        "n_anchor_groups": int(sum(np.sum(anchors == akey) >= 2 for akey in pd.unique(anchors))),
        "n_within_anchor_pairs": len(same_anchor),
        "within_anchor_cosine_mean": float(np.mean(same_anchor)) if same_anchor else np.nan,
        "within_anchor_cosine_sd": float(np.std(same_anchor)) if same_anchor else np.nan,
        "within_anchor_same_plate_cosine_mean": float(np.mean(same_anchor_same_plate)) if same_anchor_same_plate else np.nan,
        "within_anchor_cross_plate_cosine_mean": float(np.mean(same_anchor_cross_plate)) if same_anchor_cross_plate else np.nan,
        "same_compound_different_dose_cosine_mean": float(np.mean(sims[same_compound_different_dose])) if same_compound_different_dose.any() else np.nan,
        "different_compound_cosine_mean": float(np.mean(sims[different_compound])) if different_compound.any() else np.nan,
        "n_random_negative_pairs": len(a),
    }


def retrieval_for_fold(protocol, fold, transform, feature, X_train, X_test,
                       train_ids, test_ids, test_df, labels_by_column):
    rows = []
    for task, column in (("pathway", "compound_pathway"), ("target", "compound_target")):
        gallery = mean_profiles(np.arange(len(train_ids)), X_train, train_ids)
        queries = make_queries(protocol, str(fold), np.arange(len(test_ids)),
                               test_df, X_test, test_ids)
        scored, _ = score_queries(queries, gallery, labels_by_column[column],
                                  protocol, str(fold), feature, task)
        for row in scored:
            row["transform"] = transform
            row["feature_set"] = feature
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", choices=EMBEDDINGS, default=list(EMBEDDINGS))
    parser.add_argument("--transforms", nargs="+", choices=PROCESSING, default=list(PROCESSING))
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--skip-confounding", action="store_true",
                        help="Skip the more expensive plate/batch/source recovery probes.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df, ids, all_inputs, _ = load_data()
    audit = pd.read_csv(RESULTS_DIR / "chemical_split" / "chemical_split_structure_audit.csv")
    chemical_map = dict(zip(audit.compound_id.astype(str), audit.chemical_group))
    rng = np.random.default_rng(args.seed)
    replicate_rows, retrieval_rows, cluster_rows, confound_rows = [], [], [], []

    for protocol in args.protocols:
        current_df, current_ids, inputs, outer_groups, splits = make_protocol_data(
            df, ids, all_inputs, protocol, chemical_map
        )
        metadata = current_df[["compound_id", "compound_concentration_um", "plate", "batch", "source"]].reset_index(drop=True)
        labels_by_task = {
            task: label_truth(current_df, current_ids, column)
            for task, column in (("pathway", "compound_pathway"), ("target", "compound_target"))
        }
        retrieval_labels = {
            column: label_truth(current_df, current_ids, column)[0]
            for column in ("compound_pathway", "compound_target")
        }
        for feature in args.features:
            X = inputs[feature]
            for transform in args.transforms:
                for fold, (train, test) in enumerate(splits, start=1):
                    X_train, X_test = fit_transform_fold(X, metadata, train, test, transform)
                    train_meta = metadata.iloc[train].reset_index(drop=True)
                    test_meta = metadata.iloc[test].reset_index(drop=True)
                    rep = replicate_pair_metrics(X_test, test_meta, current_ids[test], rng)
                    replicate_rows.append({"protocol": protocol, "fold": fold,
                                           "feature_set": feature, "transform": transform, **rep})
                    retrieval_rows.extend(retrieval_for_fold(
                        protocol, fold, transform, feature, X_train, X_test,
                        current_ids[train], current_ids[test],
                        current_df.iloc[test].reset_index(drop=True), retrieval_labels))

                    # Cluster the held-out compound-dose-averaged profiles.
                    for task, column in (("pathway", "compound_pathway"), ("target", "compound_target")):
                        labels, vocab = labels_by_task[task]
                        Y_by_compound = {
                            cid: np.array([label in labels.get(cid, set()) for label in vocab], dtype=np.uint8)
                            for cid in np.unique(current_ids[test])
                        }
                        held = pd.DataFrame(X_test)
                        held["compound_id"] = current_ids[test]
                        embedding_means = held.groupby("compound_id").mean(numeric_only=True)
                        embedding_map = {str(cid): embedding_means.loc[cid].to_numpy(dtype=np.float32)
                                         for cid in embedding_means.index}
                        enrich = cluster_enrichment(embedding_map, Y_by_compound, vocab,
                                                    protocol, f"{feature}__{transform}", task)
                        for row in enrich:
                            row["transform"] = transform
                            row["feature_set"] = feature
                            row["fold"] = fold
                        cluster_rows.extend(enrich)

                    if not args.skip_confounding:
                        for nuisance in NUISANCE:
                            levels = current_df.groupby(current_ids)[nuisance].nunique()
                            multi = set(levels[levels >= 2].index.astype(str))
                            train_keep = np.isin(current_ids[train], list(multi)) & train_meta[nuisance].notna().to_numpy()
                            test_keep = np.isin(current_ids[test], list(multi)) & test_meta[nuisance].notna().to_numpy()
                            if (not train_keep.any() or not test_keep.any() or
                                    len(np.unique(test_meta.loc[test_keep, nuisance])) < 2):
                                continue
                            probe = nuisance_scores(
                                X_train[train_keep], X_test[test_keep],
                                train_meta.loc[train_keep].reset_index(drop=True),
                                test_meta.loc[test_keep].reset_index(drop=True),
                                current_ids[train][train_keep], current_ids[test][test_keep], nuisance)
                            confound_rows.append({"protocol": protocol, "fold": fold,
                                                  "feature_set": feature, "transform": transform,
                                                  "nuisance": nuisance, **probe})
                print(f"{protocol:24s} {feature:16s} {transform:18s} complete", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reps = pd.DataFrame(replicate_rows)
    retrieval = pd.DataFrame(retrieval_rows)
    clusters = pd.DataFrame(cluster_rows)
    confounds = pd.DataFrame(confound_rows)
    reps.to_csv(RESULTS_DIR / "feature_processing" / "transformation_replicate_consistency_folds.csv", index=False)
    retrieval.to_csv(RESULTS_DIR / "feature_processing" / "transformation_annotation_retrieval_per_query.csv", index=False)
    clusters.to_csv(RESULTS_DIR / "cluster_enrichment" / "transformation_cluster_enrichment.csv", index=False)
    if not confounds.empty:
        confounds.to_csv(RESULTS_DIR / "feature_processing" / "transformation_confounding_folds.csv", index=False)

    replicate_summary = reps.groupby(["protocol", "feature_set", "transform"], as_index=False).agg(
        n_folds=("fold", "nunique"), within_anchor_cosine_mean=("within_anchor_cosine_mean", "mean"),
        within_anchor_cosine_sd=("within_anchor_cosine_mean", "std"),
        within_anchor_cross_plate_cosine_mean=("within_anchor_cross_plate_cosine_mean", "mean"),
        same_compound_different_dose_cosine_mean=("same_compound_different_dose_cosine_mean", "mean"),
        different_compound_cosine_mean=("different_compound_cosine_mean", "mean"))
    replicate_summary.to_csv(RESULTS_DIR / "feature_processing" / "transformation_replicate_consistency_summary.csv", index=False)

    retrieval_summary = retrieval.groupby(["protocol", "feature_set", "transform", "task", "k"], as_index=False).agg(
        n_queries=("query_id", "nunique"), precision_at_k=("neighbor_precision_at_k", "mean"),
        annotation_recall_at_k=("annotation_recall_at_k", "mean"), map_at_k=("map_at_k", "mean"))
    retrieval_summary.to_csv(RESULTS_DIR / "feature_processing" / "transformation_annotation_retrieval_summary.csv", index=False)

    if not confounds.empty:
        confound_summary = confounds.groupby(["protocol", "feature_set", "transform", "nuisance"], as_index=False).agg(
            n_folds=("fold", "nunique"), balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"), chance_mean=("chance_balanced_accuracy", "mean"))
        confound_summary.to_csv(RESULTS_DIR / "feature_processing" / "transformation_confounding_summary.csv", index=False)

    print(f"Saved replicate, retrieval, cluster-enrichment and confounding comparisons in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
