"""Compare fold-fitted representation processing for biology and confounding.

Candidate transforms are fitted on each training fold only: unchanged input,
50-component PCA whitening, matched-plate offset correction, training-fold
replicate-reliability dimension selection, and a rank-Gaussian transform.
Biological prediction uses held-out compounds and chemical groups. Within-
compound residual probes test plate, batch, and source recovery on the same
folds; residualization is done separately within each compound and split.

Run:
    uv run python analysis/evaluate_processing_candidates.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from annotation_model_utils import (
    ASSAY_FEATURES, DATA_PATH, N_SPLITS, RESULTS_DIR, make_label_matrix,
    parse_labels, train_and_score, unpack_vector,
)
from evaluate_fold_safe_transforms import group_shift
from evaluate_reproducible_dimensions import replicate_reliability


FEATURES = ("pca_raw", "pca_normalized", "dino", "assay_features")
TRANSFORMS = ("uncorrected", "pca50_whiten", "matched_plate", "repeatable_top75", "rank_gaussian")
NUISANCE_COLUMNS = ("plate", "batch", "source")


def load_features(df):
    matrices = {
        "pca_raw": np.stack(df.pca_embedding_raw.map(unpack_vector)).astype(np.float32),
        "pca_normalized": np.stack(df.pca_embedding_normalized.map(unpack_vector)).astype(np.float32),
        "dino": np.stack(df.brightfield.map(unpack_vector)).astype(np.float32),
        "assay_features": df[ASSAY_FEATURES].to_numpy(dtype=np.float32),
    }
    for x in matrices.values():
        x[~np.isfinite(x)] = np.nan
    return matrices


def fit_transform_fold(X, meta, train, test, name):
    """Fit transform on training rows and apply it to both partitions."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imputer.fit_transform(X[train]).astype(np.float32)
    x_test = imputer.transform(X[test]).astype(np.float32)
    if name == "uncorrected":
        return x_train, x_test
    if name == "matched_plate":
        offsets = group_shift(x_train, meta.iloc[train], "plate")
        train_plate = meta.iloc[train].plate.astype(str).to_numpy()
        test_plate = meta.iloc[test].plate.astype(str).to_numpy()
        x_train, x_test = x_train.copy(), x_test.copy()
        for i, plate in enumerate(train_plate):
            if plate in offsets:
                x_train[i] -= offsets[plate]
        for i, plate in enumerate(test_plate):
            if plate in offsets:
                x_test[i] -= offsets[plate]
        return x_train, x_test
    if name == "pca50_whiten":
        from sklearn.decomposition import PCA
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        n = min(50, x_train.shape[1], x_train.shape[0] - 1)
        if n < 2:
            return x_train, x_test
        pca = PCA(n_components=n, whiten=True, random_state=42)
        return pca.fit_transform(x_train), pca.transform(x_test)
    if name == "repeatable_top75":
        reliability = replicate_reliability(x_train, meta.iloc[train])
        n_keep = max(1, int(np.ceil(0.75 * x_train.shape[1])))
        selected = np.argsort(reliability, kind="stable")[-n_keep:]
        return x_train[:, selected], x_test[:, selected]
    if name == "rank_gaussian":
        qt = QuantileTransformer(
            n_quantiles=min(1000, len(x_train)), output_distribution="normal",
            subsample=100_000, random_state=42,
        )
        return qt.fit_transform(x_train), qt.transform(x_test)
    raise ValueError(name)


def compound_level_metrics(y_true, scores, compound_ids):
    compounds = sorted(set(compound_ids))
    truth = np.stack([y_true[np.flatnonzero(compound_ids == c)[0]] for c in compounds])
    compound_scores = np.stack([scores[compound_ids == c].mean(axis=0) for c in compounds])
    pred = np.zeros_like(truth)
    top = np.argsort(compound_scores, axis=1)[:, -min(3, truth.shape[1]):]
    np.put_along_axis(pred, top, 1, axis=1)
    return {
        "n_compounds": len(compounds),
        "macro_f1_top3": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "micro_f1_top3": float(f1_score(truth, pred, average="micro", zero_division=0)),
    }


def within_compound_residuals(X, compound_ids):
    frame = pd.DataFrame(X)
    frame["_compound"] = compound_ids
    means = frame.groupby("_compound", sort=False).mean(numeric_only=True)
    mean_matrix = np.stack([means.loc[c].to_numpy() for c in compound_ids])
    return X - mean_matrix


def nuisance_scores(x_train, x_test, train_meta, test_meta, train_ids, test_ids, nuisance):
    """Predict technical labels from within-compound residual profiles."""
    tr_y = train_meta[nuisance].astype(str).to_numpy()
    te_y = test_meta[nuisance].astype(str).to_numpy()
    tr_x = within_compound_residuals(x_train, train_ids)
    te_x = within_compound_residuals(x_test, test_ids)
    scaler = StandardScaler()
    tr_x = scaler.fit_transform(tr_x)
    te_x = scaler.transform(te_x)
    model = RidgeClassifier(alpha=10.0, class_weight="balanced")
    model.fit(tr_x, tr_y)
    pred = model.predict(te_x)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(te_y, pred)),
        "chance_balanced_accuracy": 1.0 / len(np.unique(te_y)),
        "n_test_classes": int(len(np.unique(te_y))),
        "n_test_wells": int(len(te_y)),
    }


def make_splits(df, protocol, ids, chemical_map=None):
    if protocol == "held_out_compound":
        return list(GroupKFold(N_SPLITS).split(df, groups=ids))
    keep = np.array([cid in chemical_map for cid in ids])
    subset = df.loc[keep].reset_index(drop=True)
    subset_ids = ids[keep]
    groups = np.array([chemical_map[cid] for cid in subset_ids])
    splits = list(GroupKFold(N_SPLITS).split(subset, groups=groups))
    return subset, subset_ids, keep, splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", choices=FEATURES, default=list(FEATURES))
    parser.add_argument("--protocols", nargs="+", choices=("held_out_compound", "held_out_chemical_group"),
                        default=["held_out_compound", "held_out_chemical_group"])
    parser.add_argument("--transforms", nargs="+", choices=TRANSFORMS, default=list(TRANSFORMS))
    parser.add_argument("--include-confounding", action="store_true",
                        help="Also run the slower within-compound technical-label probes.")
    parser.add_argument("--whiten-dino-only", action="store_true",
                        help="For brightfield, evaluate the 50-component whitened input only.")
    parser.add_argument("--output-tag", default="",
                        help="Optional suffix for result files when running a focused subset.")
    args = parser.parse_args()
    suffix = f"_{args.output_tag}" if args.output_tag else ""

    df = pd.read_parquet(DATA_PATH)
    df = df[df.compound_id.notna() & df.compound_pathway.notna() & df.compound_target.notna()].copy().reset_index(drop=True)
    ids = df.compound_id.astype(str).to_numpy()
    metadata = df[["compound_id", "compound_concentration_um", "plate", "batch", "source"]].reset_index(drop=True)
    matrices = load_features(df)
    audit_path = RESULTS_DIR / "chemical_split" / "chemical_split_structure_audit.csv"
    audit = pd.read_csv(audit_path)
    chemical_map = dict(zip(audit.compound_id.astype(str), audit.chemical_group))

    biological_rows, nuisance_rows = [], []
    for protocol in args.protocols:
        if protocol == "held_out_compound":
            current_df, current_ids, keep = df, ids, np.ones(len(df), dtype=bool)
            splits = list(GroupKFold(N_SPLITS).split(df, groups=ids))
        else:
            current_df, current_ids, keep, splits = make_splits(df, protocol, ids, chemical_map)
        current_meta = current_df[["compound_id", "compound_concentration_um", "plate", "batch", "source"]].reset_index(drop=True)
        labels_by_task = {}
        for task, column in (("pathway", "compound_pathway"), ("target", "compound_target")):
            Y, label_names = make_label_matrix(current_df[column].map(parse_labels).tolist(), current_ids)
            labels_by_task[task] = Y
        for feature in args.features:
            X = matrices[feature][keep]
            feature_transforms = list(args.transforms)
            if args.whiten_dino_only and feature == "dino" and "pca50_whiten" not in feature_transforms:
                feature_transforms.append("pca50_whiten")
            for transform in feature_transforms:
                if args.whiten_dino_only and feature == "dino" and transform != "pca50_whiten":
                    continue
                for fold, (train, test) in enumerate(splits, start=1):
                    x_train, x_test = fit_transform_fold(X, current_meta, train, test, transform)
                    for task, Y in labels_by_task.items():
                        scores = train_and_score(x_train, Y[train], x_test, groups_train=current_ids[train])
                        row = {"protocol": protocol, "fold": fold, "feature_set": feature,
                               "transform": transform, "task": task,
                               **compound_level_metrics(Y[test], scores, current_ids[test])}
                        biological_rows.append(row)
                    # Nuisance probes use the same representation and held-out
                    # compounds, after per-compound centering in each split.
                    train_meta, test_meta = current_meta.iloc[train].reset_index(drop=True), current_meta.iloc[test].reset_index(drop=True)
                    for nuisance in NUISANCE_COLUMNS if args.include_confounding else ():
                        eligible = set(current_ids[pd.notna(current_df[nuisance].to_numpy())])
                        train_keep = np.array([cid in eligible for cid in current_ids[train]]) & train_meta[nuisance].notna().to_numpy()
                        test_keep = np.array([cid in eligible for cid in current_ids[test]]) & test_meta[nuisance].notna().to_numpy()
                        # Restrict to compounds known to span >=2 nuisance levels.
                        levels = current_df.groupby(current_ids)[nuisance].nunique()
                        multi = set(levels[levels >= 2].index.astype(str))
                        train_keep &= np.isin(current_ids[train], list(multi))
                        test_keep &= np.isin(current_ids[test], list(multi))
                        if not train_keep.any() or not test_keep.any() or len(np.unique(test_meta.loc[test_keep, nuisance])) < 2:
                            continue
                        try:
                            probe = nuisance_scores(
                                x_train[train_keep], x_test[test_keep],
                                train_meta.loc[train_keep].reset_index(drop=True),
                                test_meta.loc[test_keep].reset_index(drop=True),
                                current_ids[train][train_keep], current_ids[test][test_keep], nuisance,
                            )
                        except ValueError:
                            continue
                        nuisance_rows.append({"protocol": protocol, "fold": fold,
                                              "feature_set": feature, "transform": transform,
                                              "nuisance": nuisance, **probe})
                    # Save fold-level progress so long DINO runs remain useful
                    # if they are interrupted before a full feature block ends.
                    pd.DataFrame(biological_rows).to_csv(
                        RESULTS_DIR / "feature_processing" / f"processing_candidate_biology_folds{suffix}.partial.csv", index=False)
                    if nuisance_rows:
                        pd.DataFrame(nuisance_rows).to_csv(
                            RESULTS_DIR / "feature_processing" / f"processing_candidate_confounding_folds{suffix}.partial.csv", index=False)
                print(f"{protocol:24s} {feature:16s} {transform:18s} completed", flush=True)
            # Keep completed feature blocks if a long run is interrupted.
            pd.DataFrame(biological_rows).to_csv(
                RESULTS_DIR / "feature_processing" / f"processing_candidate_biology_folds{suffix}.partial.csv", index=False)
            if nuisance_rows:
                pd.DataFrame(nuisance_rows).to_csv(
                    RESULTS_DIR / "feature_processing" / f"processing_candidate_confounding_folds{suffix}.partial.csv", index=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bio = pd.DataFrame(biological_rows)
    nuisance = pd.DataFrame(nuisance_rows)
    bio.to_csv(RESULTS_DIR / "feature_processing" / f"processing_candidate_biology_folds{suffix}.csv", index=False)
    if not nuisance.empty:
        nuisance.to_csv(RESULTS_DIR / "feature_processing" / f"processing_candidate_confounding_folds{suffix}.csv", index=False)
    bio_summary = bio.groupby(["protocol", "feature_set", "transform", "task"], as_index=False).agg(
        n_folds=("fold", "nunique"), macro_f1_top3_mean=("macro_f1_top3", "mean"),
        macro_f1_top3_sd=("macro_f1_top3", "std"), micro_f1_top3_mean=("micro_f1_top3", "mean"),
        micro_f1_top3_sd=("micro_f1_top3", "std"))
    rng = np.random.default_rng(42)
    for idx, summary_row in bio_summary.iterrows():
        same = bio[(bio.protocol == summary_row.protocol) &
                   (bio.feature_set == summary_row.feature_set) &
                   (bio.task == summary_row.task)]
        base = same[same["transform"] == "uncorrected"].sort_values("fold")
        current = same[same["transform"] == summary_row["transform"]].sort_values("fold")
        for metric in ("macro_f1_top3", "micro_f1_top3"):
            if len(base) == len(current) and len(base):
                delta = current[metric].to_numpy() - base[metric].to_numpy()
                samples = rng.choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
                bio_summary.loc[idx, f"delta_{metric}_vs_uncorrected"] = delta.mean()
                bio_summary.loc[idx, f"delta_{metric}_ci_low"] = np.quantile(samples, .025)
                bio_summary.loc[idx, f"delta_{metric}_ci_high"] = np.quantile(samples, .975)
    if not nuisance.empty:
        nuisance_summary = nuisance.groupby(["protocol", "feature_set", "transform", "nuisance"], as_index=False).agg(
            n_folds=("fold", "nunique"), balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"), chance_mean=("chance_balanced_accuracy", "mean"))
    else:
        nuisance_summary = pd.DataFrame(columns=["protocol", "feature_set", "transform", "nuisance",
                                                 "n_folds", "balanced_accuracy_mean",
                                                 "balanced_accuracy_sd", "chance_mean"])
    bio_summary.to_csv(RESULTS_DIR / "feature_processing" / f"processing_candidate_biology_summary{suffix}.csv", index=False)
    if not nuisance.empty:
        nuisance_summary.to_csv(RESULTS_DIR / "feature_processing" / f"processing_candidate_confounding_summary{suffix}.csv", index=False)
    (RESULTS_DIR / "feature_processing" / f"processing_candidate_biology_folds{suffix}.partial.csv").unlink(missing_ok=True)
    (RESULTS_DIR / "feature_processing" / f"processing_candidate_confounding_folds{suffix}.partial.csv").unlink(missing_ok=True)
    print(f"Saved biological and nuisance-probe summaries under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
