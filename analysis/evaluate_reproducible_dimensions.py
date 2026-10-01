"""Evaluate fold-fitted selection of reproducible embedding dimensions.

For each outer training fold, dimensions are ranked using replicate reliability
among training wells matched by compound and dose. The top 25%, 50%, 75%, or
100% are retained; no annotations are used by the selector. Pathway/target
prediction is then evaluated on the same held-out compound, batch, and chemical
group splits used elsewhere in the project.

Run:
    uv run python analysis/evaluate_reproducible_dimensions.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from annotation_model_utils import (
    ASSAY_FEATURES, DATA_PATH, N_SPLITS, RESULTS_DIR, build_splits,
    make_label_matrix, parse_labels, train_and_score, unpack_vector,
)


FRACTIONS = (0.25, 0.50, 0.75, 1.00)
EMBEDDINGS = ("pca_raw", "pca_normalized", "dino")


def load_inputs(df):
    X = {
        "pca_raw": np.stack(df.pca_embedding_raw.map(unpack_vector)).astype(np.float32),
        "pca_normalized": np.stack(df.pca_embedding_normalized.map(unpack_vector)).astype(np.float32),
        "dino": np.stack(df.brightfield.map(unpack_vector)).astype(np.float32),
        "assay_features": df[ASSAY_FEATURES].to_numpy(dtype=np.float32),
    }
    for values in X.values():
        values[~np.isfinite(values)] = np.nan
    return X


def replicate_reliability(X_train: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Estimate per-dimension ICC-like repeatability from training wells only.

    Anchors are compound + dose. The within-anchor variance estimates repeat
    noise; variance among anchor means estimates between-perturbation signal.
    The returned score is between/(between + within), clipped to [0, 1].
    """
    dose = meta["compound_concentration_um"].astype("string").fillna("NA")
    anchor = meta["compound_id"].astype(str) + "||" + dose
    frame = pd.DataFrame(X_train)
    frame["_anchor"] = anchor.to_numpy()
    group_sizes = frame.groupby("_anchor", sort=False).size()
    grouped_means = frame.groupby("_anchor", sort=False).mean(numeric_only=True)
    repeated = group_sizes[group_sizes >= 2].index
    if len(repeated) < 2:
        return np.zeros(X_train.shape[1], dtype=np.float64)
    row_means = grouped_means.loc[anchor].to_numpy(dtype=np.float64)
    repeated_rows = anchor.isin(repeated).to_numpy()
    residuals = X_train[repeated_rows].astype(np.float64) - row_means[repeated_rows]
    within_df = int((group_sizes.loc[repeated] - 1).sum())
    if within_df == 0:
        return np.zeros(X_train.shape[1], dtype=np.float64)
    within = np.nansum(residuals ** 2, axis=0) / within_df
    repeated_means = grouped_means.loc[repeated].to_numpy(dtype=np.float64)
    observed_between = np.nanvar(repeated_means, axis=0, ddof=1)
    mean_inverse_n = float((1.0 / group_sizes.loc[repeated]).mean())
    between = np.maximum(observed_between - within * mean_inverse_n, 0.0)
    reliability = between / np.maximum(between + within, 1e-12)
    return np.nan_to_num(reliability, nan=0.0, posinf=0.0, neginf=0.0)


def metrics_at_top3(y_true, scores):
    pred = np.zeros_like(y_true)
    top = np.argsort(scores, axis=1)[:, -min(3, scores.shape[1]):]
    np.put_along_axis(pred, top, 1, axis=1)
    return (float(f1_score(y_true, pred, average="macro", zero_division=0)),
            float(f1_score(y_true, pred, average="micro", zero_division=0)))


def score_fold(X, Y, compound_ids, train, test):
    scores = train_and_score(X[train], Y[train], X[test],
                             groups_train=compound_ids[train])
    test_compounds = sorted(set(compound_ids[test]))
    y_true = np.stack([Y[test[np.flatnonzero(compound_ids[test] == cid)[0]]]
                       for cid in test_compounds])
    compound_scores = np.stack([
        scores[compound_ids[test] == cid].mean(axis=0) for cid in test_compounds
    ])
    return metrics_at_top3(y_true, compound_scores), len(test_compounds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", nargs="+", default=list(EMBEDDINGS),
                        choices=list(EMBEDDINGS))
    parser.add_argument("--protocols", nargs="+",
                        default=["held_out_compound", "held_out_batch", "held_out_chemical_group"],
                        choices=["held_out_compound", "held_out_batch", "held_out_chemical_group"])
    args = parser.parse_args()
    df = pd.read_parquet(DATA_PATH)
    df = df[df.compound_id.notna() & df.compound_pathway.notna()
            & df.compound_target.notna()].copy().reset_index(drop=True)
    X_by_feature = load_inputs(df)
    compound_ids = df.compound_id.astype(str).to_numpy()
    dose_meta = df[["compound_id", "compound_concentration_um"]].reset_index(drop=True)

    protocols = build_splits(df)
    structure_audit = RESULTS_DIR / "chemical_split" / "chemical_split_structure_audit.csv"
    if structure_audit.exists():
        audit = pd.read_csv(structure_audit)
        chemical_map = dict(zip(audit.compound_id.astype(str), audit.chemical_group))
        keep = np.array([cid in chemical_map for cid in compound_ids])
        chem_df = df.loc[keep].reset_index(drop=True)
        chem_ids = compound_ids[keep]
        chem_groups = np.array([chemical_map[cid] for cid in chem_ids])
        protocols["held_out_chemical_group"] = list(
            GroupKFold(N_SPLITS).split(chem_df, groups=chem_groups)
        )
    else:
        raise FileNotFoundError(
            f"Missing {structure_audit}; run the chemical-split analysis first."
        )

    data = {
        "held_out_compound": (df, compound_ids, X_by_feature),
        "held_out_batch": (df, compound_ids, X_by_feature),
        "held_out_chemical_group": (
            chem_df, chem_ids, {name: matrix[keep] for name, matrix in X_by_feature.items()}
        ),
    }
    rows = []
    for protocol in args.protocols:
        current_df, current_ids, feature_arrays = data[protocol]
        splits = protocols[protocol]
        metadata = current_df[["compound_id", "compound_concentration_um"]].reset_index(drop=True)
        for task, column in (("pathway", "compound_pathway"),
                             ("target", "compound_target")):
            labels = current_df[column].map(parse_labels).tolist()
            Y, kept_labels = make_label_matrix(labels, current_ids)
            for feature in (*args.embeddings, "assay_features"):
                X = feature_arrays[feature]
                for fold, (train, test) in enumerate(splits, start=1):
                    if feature in EMBEDDINGS:
                        reliability = replicate_reliability(X[train], metadata.iloc[train])
                        dim_order = np.argsort(-reliability, kind="stable")
                    else:
                        reliability = None
                        dim_order = np.arange(X.shape[1])
                    fractions = FRACTIONS if feature in EMBEDDINGS else (1.0,)
                    for fraction in fractions:
                        n_keep = max(1, int(np.ceil(fraction * X.shape[1])))
                        selected = dim_order[:n_keep]
                        (macro, micro), n_compounds = score_fold(
                            X[:, selected], Y, current_ids, train, test
                        )
                        row = {
                            "protocol": protocol, "task": task, "embedding": feature,
                            "retained_fraction": fraction, "n_dimensions": X.shape[1],
                            "n_dimensions_retained": len(selected), "fold": fold,
                            "n_compounds": n_compounds, "n_labels": len(kept_labels),
                            "macro_f1_top3": macro, "micro_f1_top3": micro,
                            "median_selected_reliability": (
                                float(np.median(reliability[selected]))
                                if reliability is not None else np.nan
                            ),
                            "selected_dimension_indices": ";".join(map(str, selected.tolist())),
                        }
                        rows.append(row)
                    print(f"{protocol:24s} {task:8s} {feature:16s} done; "
                          f"median reliability={np.median(reliability):.3f}" if reliability is not None
                          else f"{protocol:24s} {task:8s} {feature:16s} assay baseline done")

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_path = RESULTS_DIR / "feature_processing" / "reproducible_dimension_fold_metrics.csv"
    results.to_csv(fold_path, index=False)
    summary = results.groupby(
        ["protocol", "task", "embedding", "retained_fraction"], as_index=False
    ).agg(
        n_folds=("fold", "count"),
        n_dimensions_retained=("n_dimensions_retained", "first"),
        macro_f1_top3_mean=("macro_f1_top3", "mean"),
        macro_f1_top3_fold_sd=("macro_f1_top3", "std"),
        micro_f1_top3_mean=("micro_f1_top3", "mean"),
        micro_f1_top3_fold_sd=("micro_f1_top3", "std"),
        median_selected_reliability=("median_selected_reliability", "mean"),
    )
    rng = np.random.default_rng(42)
    delta_rows = []
    for (protocol, task, feature), part in results.groupby(
            ["protocol", "task", "embedding"]):
        full = part[part.retained_fraction == 1.0].sort_values("fold")
        for fraction in sorted(part.retained_fraction.unique()):
            candidate = part[part.retained_fraction == fraction].sort_values("fold")
            row = {"protocol": protocol, "task": task, "embedding": feature,
                   "retained_fraction": fraction}
            for metric in ("macro_f1_top3", "micro_f1_top3"):
                differences = (candidate[metric].to_numpy() -
                               full[metric].to_numpy())
                boot = rng.choice(differences, size=(10000, len(differences)),
                                  replace=True).mean(axis=1)
                row[f"delta_{metric}_vs_all"] = float(differences.mean())
                row[f"delta_{metric}_vs_all_ci_low"] = float(np.quantile(boot, .025))
                row[f"delta_{metric}_vs_all_ci_high"] = float(np.quantile(boot, .975))
            delta_rows.append(row)
    summary = summary.merge(pd.DataFrame(delta_rows), on=[
        "protocol", "task", "embedding", "retained_fraction"
    ])
    summary_path = RESULTS_DIR / "feature_processing" / "reproducible_dimension_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved fold metrics: {fold_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
