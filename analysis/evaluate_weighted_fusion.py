"""Tune and evaluate score-level fusion of normalized PCA and assay features.

Separate logistic models are fitted to normalized PCA and assay features. Their
within-compound label rankings are combined using a weight selected by grouped
inner cross-validation on the outer training compounds only. This avoids
mixing embedding coordinates with assay units and keeps the held-out compounds
out of weight selection.

Run:
    uv run python analysis/evaluate_weighted_fusion.py

Outputs fold-level scores, selected weights, and summaries for held-out
compound, held-out batch, and held-out chemical-group evaluations.
"""
from __future__ import annotations

from itertools import chain

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from annotation_model_utils import (
    ASSAY_FEATURES, DATA_PATH, MIN_LABEL_COMPOUNDS, N_SPLITS, RESULTS_DIR,
    build_splits, make_label_matrix, parse_labels, train_and_score, unpack_vector,
)
from compare_annotations_chemical_split import chemical_groups, get_structure_table


WEIGHTS = np.linspace(0.0, 1.0, 21)  # weight on normalized PCA
INNER_SPLITS = 3


def compound_labels(df: pd.DataFrame, column: str) -> dict[str, set[str]]:
    result = {}
    for cid, values in df.groupby(df.compound_id.astype(str))[column]:
        result[str(cid)] = set(chain.from_iterable(values.map(parse_labels)))
    counts: dict[str, int] = {}
    for labels in result.values():
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    keep = {label for label, count in counts.items() if count >= MIN_LABEL_COMPOUNDS}
    return {cid: labels & keep for cid, labels in result.items()}


def rowwise_rank(scores: np.ndarray) -> np.ndarray:
    """Convert decision scores to comparable within-sample percentile ranks."""
    if scores.shape[1] < 2:
        return np.ones_like(scores, dtype=np.float32)
    return ((rankdata(scores, axis=1, method="average") - 1) /
            (scores.shape[1] - 1)).astype(np.float32)


def compound_predictions(scores: np.ndarray, labels: np.ndarray,
                         compound_ids: np.ndarray):
    ids = sorted(set(compound_ids))
    y_true, average_scores = [], []
    for cid in ids:
        rows = compound_ids == cid
        y_true.append(labels[np.flatnonzero(rows)[0]])
        average_scores.append(scores[rows].mean(axis=0))
    return ids, np.stack(y_true), np.stack(average_scores)


def top3_metrics(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    y_pred = np.zeros_like(y_true)
    top = np.argsort(scores, axis=1)[:, -min(3, scores.shape[1]):]
    np.put_along_axis(y_pred, top, 1, axis=1)
    return (float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            float(f1_score(y_true, y_pred, average="micro", zero_division=0)))


def tune_weight(X_pca, X_assay, Y, groups) -> tuple[float, dict[str, float]]:
    """Choose PCA weight using only grouped OOF predictions from outer train."""
    n_groups = len(np.unique(groups))
    n_splits = min(INNER_SPLITS, n_groups)
    if n_splits < 2:
        return 0.5, {}
    inner_splits = GroupKFold(n_splits=n_splits).split(X_pca, groups=groups)
    oof_pca = np.full(Y.shape, np.nan, dtype=np.float32)
    oof_assay = np.full(Y.shape, np.nan, dtype=np.float32)
    for train, valid in inner_splits:
        pca_scores = train_and_score(X_pca[train], Y[train], X_pca[valid],
                                     groups_train=groups[train])
        assay_scores = train_and_score(X_assay[train], Y[train], X_assay[valid],
                                       groups_train=groups[train])
        oof_pca[valid] = rowwise_rank(pca_scores)
        oof_assay[valid] = rowwise_rank(assay_scores)
    valid = np.isfinite(oof_pca).all(axis=1) & np.isfinite(oof_assay).all(axis=1)
    ids, y_true, _ = compound_predictions(oof_pca[valid], Y[valid], groups[valid])
    pca_compound = compound_predictions(oof_pca[valid], Y[valid], groups[valid])[2]
    assay_compound = compound_predictions(oof_assay[valid], Y[valid], groups[valid])[2]
    scores_by_weight = []
    for weight in WEIGHTS:
        macro, micro = top3_metrics(y_true, weight * pca_compound + (1-weight) * assay_compound)
        scores_by_weight.append((macro, micro, float(weight)))
    # Macro F1 is primary; micro F1 breaks exact macro ties; prefer balanced
    # fusion if both metrics tie to avoid a gratuitous endpoint choice.
    best = max(scores_by_weight, key=lambda x: (x[0], x[1], -abs(x[2] - 0.5)))
    return best[2], {"inner_macro_f1_top3": best[0], "inner_micro_f1_top3": best[1]}


def evaluate_fold(X_pca, X_assay, Y, groups, train, test, protocol, task, fold):
    weight, tuning = tune_weight(X_pca[train], X_assay[train], Y[train], groups[train])
    pca_scores = rowwise_rank(train_and_score(
        X_pca[train], Y[train], X_pca[test], groups_train=groups[train]))
    assay_scores = rowwise_rank(train_and_score(
        X_assay[train], Y[train], X_assay[test], groups_train=groups[train]))
    test_ids = groups[test]
    _, y_true, pca_compound = compound_predictions(pca_scores, Y[test], test_ids)
    _, _, assay_compound = compound_predictions(assay_scores, Y[test], test_ids)
    fused_compound = weight * pca_compound + (1-weight) * assay_compound
    pca_macro, pca_micro = top3_metrics(y_true, pca_compound)
    assay_macro, assay_micro = top3_metrics(y_true, assay_compound)
    fused_macro, fused_micro = top3_metrics(y_true, fused_compound)
    return {
        "protocol": protocol, "task": task, "fold": str(fold),
        "n_compounds": len(set(test_ids)), "pca_weight": weight,
        **tuning,
        "pca_macro_f1_top3": pca_macro, "pca_micro_f1_top3": pca_micro,
        "assay_macro_f1_top3": assay_macro, "assay_micro_f1_top3": assay_micro,
        "fusion_macro_f1_top3": fused_macro, "fusion_micro_f1_top3": fused_micro,
    }


def main():
    df = pd.read_parquet(DATA_PATH)
    df["_source_row_index"] = np.arange(len(df))
    df = df[df.compound_id.notna() & df.compound_pathway.notna()
            & df.compound_target.notna()].copy().reset_index(drop=True)
    groups = df.compound_id.astype(str).to_numpy()
    pca = np.stack(df.pca_embedding_normalized.map(unpack_vector)).astype(np.float32)
    assay = df[ASSAY_FEATURES].to_numpy(dtype=np.float32)
    assay[~np.isfinite(assay)] = np.nan

    splits_by_protocol = {"held_out_compound": build_splits(df)["held_out_compound"],
                          "held_out_batch": build_splits(df)["held_out_batch"]}
    compound_list = sorted(set(groups))
    structures = get_structure_table(compound_list)
    chemical_map, _ = chemical_groups(structures)
    use = np.array([cid in chemical_map for cid in groups])
    chemical_df = df.loc[use].reset_index(drop=True)
    chemical_groups_by_row = np.array([chemical_map[c] for c in groups[use]])
    splits_by_protocol["held_out_chemical_group"] = list(
        GroupKFold(N_SPLITS).split(chemical_df, groups=chemical_groups_by_row)
    )
    data_by_protocol = {
        "held_out_compound": (df, pca, assay, groups),
        "held_out_batch": (df, pca, assay, groups),
        "held_out_chemical_group": (chemical_df, pca[use], assay[use], groups[use]),
    }

    fold_rows = []
    for protocol, (current_df, Xp, Xa, current_groups) in data_by_protocol.items():
        splits = splits_by_protocol[protocol]
        for task, column in (("pathway", "compound_pathway"),
                             ("target", "compound_target")):
            labels = current_df[column].map(parse_labels).tolist()
            Y, _ = make_label_matrix(labels, current_groups)
            for fold_num, (train, test) in enumerate(splits, start=1):
                fold_name = (str(current_df.iloc[test].batch.iloc[0])
                             if protocol == "held_out_batch" else str(fold_num))
                result = evaluate_fold(Xp, Xa, Y, current_groups, train, test,
                                       protocol, task, fold_name)
                fold_rows.append(result)
                print(f"{protocol:24s} {task:8s} fold={fold_name:>4s} "
                      f"PCA weight={result['pca_weight']:.2f} "
                      f"Macro-F1@3 fusion={result['fusion_macro_f1_top3']:.3f} "
                      f"(PCA={result['pca_macro_f1_top3']:.3f}, assay={result['assay_macro_f1_top3']:.3f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_path = RESULTS_DIR / "feature_processing" / "weighted_fusion_fold_metrics.csv"
    summary_path = RESULTS_DIR / "feature_processing" / "weighted_fusion_summary.csv"
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(fold_path, index=False)
    summary = fold_df.groupby(["protocol", "task"], as_index=False).agg(
        n_folds=("fold", "count"),
        pca_weight_mean=("pca_weight", "mean"),
        pca_weight_sd=("pca_weight", "std"),
        pca_macro_f1_top3_mean=("pca_macro_f1_top3", "mean"),
        assay_macro_f1_top3_mean=("assay_macro_f1_top3", "mean"),
        fusion_macro_f1_top3_mean=("fusion_macro_f1_top3", "mean"),
        pca_micro_f1_top3_mean=("pca_micro_f1_top3", "mean"),
        assay_micro_f1_top3_mean=("assay_micro_f1_top3", "mean"),
        fusion_micro_f1_top3_mean=("fusion_micro_f1_top3", "mean"),
    )
    rng = np.random.default_rng(42)
    for metric in ("macro_f1_top3", "micro_f1_top3"):
        delta_name = f"delta_fusion_vs_pca_{metric}"
        fold_df[delta_name] = fold_df[f"fusion_{metric}"] - fold_df[f"pca_{metric}"]
        delta_summary = []
        for (protocol, task), part in fold_df.groupby(["protocol", "task"]):
            values = part[delta_name].to_numpy()
            boot = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
            delta_summary.append({
                "protocol": protocol, "task": task,
                delta_name: float(values.mean()),
                f"{delta_name}_fold_bootstrap_low": float(np.quantile(boot, .025)),
                f"{delta_name}_fold_bootstrap_high": float(np.quantile(boot, .975)),
            })
        summary = summary.merge(pd.DataFrame(delta_summary), on=["protocol", "task"])
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved fold metrics: {fold_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
