"""Compare well-level models with compound-dose replicate consensus profiles.

The consensus options are arithmetic mean, coordinate-wise median, and MODZ-
style correlation-weighted mean. For compound and chemical holdouts, all wells
for a compound-dose may contribute to its profile; for held-out-batch tests,
profiles are aggregated separately within each batch so test profiles never
mix with wells from other batches. Model preprocessing is still fit on training
rows only.

Run the full comparison:
    uv run python analysis/evaluate_consensus_profiles.py

Example focused run:
    uv run python analysis/evaluate_consensus_profiles.py \\
        --features pca_normalized --protocols held_out_compound
"""
from __future__ import annotations

import argparse
from itertools import chain

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from annotation_model_utils import (
    ASSAY_FEATURES, DATA_PATH, MIN_LABEL_COMPOUNDS, N_SPLITS, RESULTS_DIR,
    make_label_matrix, parse_labels, train_and_score, unpack_vector,
)


FEATURE_COLUMNS = {
    "pca_raw": "pca_embedding_raw",
    "pca_normalized": "pca_embedding_normalized",
    "dino": "brightfield",
    "assay_features": None,
}
METHODS = ("well_level", "mean", "median", "modz")


def load_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    X = {
        "pca_raw": np.stack(df.pca_embedding_raw.map(unpack_vector)).astype(np.float32),
        "pca_normalized": np.stack(df.pca_embedding_normalized.map(unpack_vector)).astype(np.float32),
        "dino": np.stack(df.brightfield.map(unpack_vector)).astype(np.float32),
        "assay_features": df[ASSAY_FEATURES].to_numpy(dtype=np.float32),
    }
    for matrix in X.values():
        matrix[~np.isfinite(matrix)] = np.nan
    return X


def profile_consensus(values: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return np.nanmean(values, axis=0).astype(np.float32)
    if method == "median":
        return np.nanmedian(values, axis=0).astype(np.float32)
    if method != "modz":
        raise ValueError(f"Unknown consensus method: {method}")
    if len(values) == 1:
        return values[0].astype(np.float32)

    # Correlate each replicate with the others using Spearman correlation,
    # then weight replicates by their total similarity to the group.
    fill = np.nanmedian(values, axis=0)
    fill[~np.isfinite(fill)] = 0.0
    complete = np.where(np.isfinite(values), values, fill)
    ranked = np.apply_along_axis(rankdata, 1, complete)
    correlations = np.corrcoef(ranked)
    correlations = np.nan_to_num(correlations, nan=0.0)
    np.fill_diagonal(correlations, 0.0)
    weights = np.maximum(correlations.sum(axis=1), 0.0)
    if not np.isfinite(weights).all() or weights.sum() <= 1e-12:
        weights = np.ones(len(values), dtype=np.float64)
    weights = weights / weights.sum()
    # Ignore missing entries feature-by-feature and renormalize their weights.
    observed = np.isfinite(values)
    numerator = np.nansum(values * weights[:, None], axis=0)
    denominator = (observed * weights[:, None]).sum(axis=0)
    consensus = np.divide(numerator, denominator, out=fill.copy(), where=denominator > 0)
    return consensus.astype(np.float32)


def aggregate_profiles(df: pd.DataFrame, arrays: dict[str, np.ndarray],
                       by_batch: bool) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    keys = ["compound_id", "compound_concentration_um"]
    if by_batch:
        keys.append("batch")
    groups = list(df.groupby(keys, sort=False, dropna=False).indices.values())
    metadata_rows = []
    results = {method: {name: [] for name in arrays} for method in METHODS if method != "well_level"}
    for indices in groups:
        first = df.iloc[int(indices[0])]
        metadata_rows.append({
            "compound_id": str(first.compound_id),
            "compound_concentration_um": first.compound_concentration_um,
            "batch": str(first.batch),
        })
        for method in results:
            for feature, X in arrays.items():
                results[method][feature].append(profile_consensus(X[indices], method))
    profiles = {
        method: {feature: np.stack(vectors).astype(np.float32)
                 for feature, vectors in feature_map.items()}
        for method, feature_map in results.items()
    }
    return pd.DataFrame(metadata_rows), profiles


def compound_folds(compounds: np.ndarray, chemical_groups: np.ndarray | None = None):
    if chemical_groups is None:
        splitter_groups = compounds
    else:
        splitter_groups = chemical_groups
    splitter = GroupKFold(n_splits=N_SPLITS)
    indices = np.arange(len(compounds))
    return [(compounds[train], compounds[test])
            for train, test in splitter.split(indices, groups=splitter_groups)]


def batch_folds(df: pd.DataFrame, compounds: np.ndarray):
    folds = []
    batches = df.batch.astype(str).to_numpy()
    for batch in sorted(pd.unique(batches)):
        test_rows = np.flatnonzero(batches == batch)
        test_compounds = np.unique(compounds[test_rows])
        train_compounds = np.unique(compounds[
            (batches != batch) & ~np.isin(compounds, test_compounds)
        ])
        if len(train_compounds) and len(test_compounds):
            folds.append((train_compounds, test_compounds, batch))
    return folds


def score_fold(X, Y, train_idx, test_idx, ids, labels_by_compound, kept_labels):
    scores = train_and_score(X[train_idx], Y[train_idx], X[test_idx],
                             groups_train=ids[train_idx])
    compounds = sorted(set(ids[test_idx]))
    truth = np.stack([
        np.array([label in labels_by_compound[cid] and label in kept_labels
                  for label in sorted(kept_labels)], dtype=np.uint8)
        for cid in compounds
    ])
    compound_scores = np.stack([
        scores[ids[test_idx] == cid].mean(axis=0) for cid in compounds
    ])
    y_pred = np.zeros_like(truth)
    top = np.argsort(compound_scores, axis=1)[:, -min(3, truth.shape[1]):]
    np.put_along_axis(y_pred, top, 1, axis=1)
    return {
        "n_compounds": len(compounds),
        "macro_f1_top3": float(f1_score(truth, y_pred, average="macro", zero_division=0)),
        "micro_f1_top3": float(f1_score(truth, y_pred, average="micro", zero_division=0)),
    }


def build_protocol_data(df, arrays, method, profile_meta, profile_arrays, protocol):
    if method == "well_level":
        return df.reset_index(drop=True), arrays
    if protocol == "held_out_batch":
        return profile_meta.reset_index(drop=True), profile_arrays[method]
    return profile_meta.drop(columns="batch").reset_index(drop=True), profile_arrays[method]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", default=list(FEATURE_COLUMNS),
                        choices=list(FEATURE_COLUMNS))
    parser.add_argument("--protocols", nargs="+",
                        default=["held_out_compound", "held_out_batch", "held_out_chemical_group"],
                        choices=["held_out_compound", "held_out_batch", "held_out_chemical_group"])
    args = parser.parse_args()

    source = pd.read_parquet(DATA_PATH)
    source["_row"] = np.arange(len(source))
    # Match the established annotation benchmark: retain labeled compound wells;
    # DMSO/control wells are not model examples.
    df = source[source.compound_id.notna() & source.compound_pathway.notna()
                & source.compound_target.notna()].copy().reset_index(drop=True)
    arrays = {k: v for k, v in load_features(df).items() if k in args.features}
    ids = df.compound_id.astype(str).to_numpy()
    labels_by_task = {
        task: {
            cid: set(chain.from_iterable(values.map(parse_labels)))
            for cid, values in df.groupby(df.compound_id.astype(str))[column]
        }
        for task, column in (("pathway", "compound_pathway"), ("target", "compound_target"))
    }

    chemical_map = {}
    audit_path = RESULTS_DIR / "chemical_split" / "chemical_split_structure_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        chemical_map = dict(zip(audit.compound_id.astype(str), audit.chemical_group))
    else:
        raise FileNotFoundError(f"Missing {audit_path}; create the chemical split audit first.")

    rows = []
    for protocol in args.protocols:
        if protocol == "held_out_chemical_group":
            eligible = set(chemical_map)
            keep_rows = np.array([cid in eligible for cid in ids])
            part_df = df.loc[keep_rows].reset_index(drop=True)
            part_ids = ids[keep_rows]
            part_arrays = {name: X[keep_rows] for name, X in arrays.items()}
            compounds = np.array(sorted(set(part_ids)))
            group_by_compound = {cid: chemical_map[cid] for cid in compounds}
            comp_groups = np.array([group_by_compound[cid] for cid in compounds])
            folds = [(tr, te, str(fold)) for fold, (tr, te) in enumerate(
                compound_folds(compounds, comp_groups), start=1)]
        else:
            part_df, part_ids, part_arrays = df, ids, arrays
            if protocol == "held_out_compound":
                compounds = np.array(sorted(set(ids)))
                folds = [(tr, te, str(fold)) for fold, (tr, te) in enumerate(
                    compound_folds(compounds), start=1)]
            else:
                folds = batch_folds(df, ids)

        profile_meta, profile_arrays = aggregate_profiles(
            part_df, part_arrays, by_batch=(protocol == "held_out_batch")
        )
        mode_data = {}
        for method in METHODS:
            current_df, current_arrays = build_protocol_data(
                part_df, part_arrays, method, profile_meta, profile_arrays, protocol
            )
            current_ids = current_df.compound_id.astype(str).to_numpy()
            mode_data[method] = (current_df, current_arrays, current_ids)

        for task, column in (("pathway", "compound_pathway"),
                             ("target", "compound_target")):
            # Build the label vocabulary on compounds in this protocol's data.
            labels = current_label_rows = part_df[column].map(parse_labels).tolist()
            label_matrix, label_names = make_label_matrix(labels, part_ids)
            kept_labels = set(label_names)
            for fold_no, fold in enumerate(folds, start=1):
                if protocol == "held_out_batch":
                    train_compounds, test_compounds, fold_name = fold
                else:
                    train_compounds, test_compounds, fold_name = fold
                for method in METHODS:
                    current_df, current_arrays, current_ids = mode_data[method]
                    if protocol == "held_out_batch":
                        batch = fold_name
                        train_idx = np.flatnonzero(
                            (current_df.batch.astype(str).to_numpy() != batch) &
                            ~np.isin(current_ids, test_compounds)
                        )
                        test_idx = np.flatnonzero(current_df.batch.astype(str).to_numpy() == batch)
                    else:
                        train_idx = np.flatnonzero(np.isin(current_ids, train_compounds))
                        test_idx = np.flatnonzero(np.isin(current_ids, test_compounds))
                    Y = np.stack([
                        np.array([label in labels_by_task[task].get(cid, set()) and label in kept_labels
                                  for label in label_names], dtype=np.uint8)
                        for cid in current_ids
                    ])
                    for feature, X in current_arrays.items():
                        score = score_fold(X, Y, train_idx, test_idx, current_ids,
                                           labels_by_task[task], kept_labels)
                        rows.append({"protocol": protocol, "task": task, "feature_set": feature,
                                     "profile_method": method, "fold": fold_name,
                                     "n_profile_rows": len(current_ids), **score})
                        print(f"{protocol:24s} {task:8s} {feature:16s} {method:12s} "
                              f"Macro-F1@3={score['macro_f1_top3']:.3f} "
                              f"Micro-F1@3={score['micro_f1_top3']:.3f}")

    result = pd.DataFrame(rows)
    fold_path = RESULTS_DIR / "feature_processing" / "consensus_profile_fold_metrics.csv"
    result.to_csv(fold_path, index=False)
    rng = np.random.default_rng(42)
    summary_rows = []
    for (protocol, task, feature, method), part in result.groupby(
            ["protocol", "task", "feature_set", "profile_method"]):
        part = part.sort_values("fold")
        base = result[(result.protocol == protocol) & (result.task == task) &
                      (result.feature_set == feature) &
                      (result.profile_method == "well_level")].sort_values("fold")
        row = {"protocol": protocol, "task": task, "feature_set": feature,
               "profile_method": method, "n_folds": len(part),
               "n_profile_rows_mean": part.n_profile_rows.mean()}
        for metric in ("macro_f1_top3", "micro_f1_top3"):
            vals = part[metric].to_numpy()
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_fold_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            if method != "well_level" and len(base) == len(part):
                delta = vals - base[metric].to_numpy()
                boot = rng.choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
                row[f"delta_{metric}_vs_well_level"] = float(delta.mean())
                row[f"delta_{metric}_ci_low"] = float(np.quantile(boot, .025))
                row[f"delta_{metric}_ci_high"] = float(np.quantile(boot, .975))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "feature_processing" / "consensus_profile_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved fold metrics: {fold_path}")
    print(f"Saved summary and paired fold-bootstrap intervals: {summary_path}")


if __name__ == "__main__":
    main()
