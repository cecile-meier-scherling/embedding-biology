"""Evaluate Harmony fitted on training folds only, with a training-only projection.

harmonypy integrates the observations supplied to ``run_harmony`` and does not
provide a native transform method for new observations. This script fits
Harmony on each training fold only, estimates the median Harmony adjustment
for each known batch or plate-batch group from that training fold, and applies
that group adjustment to held-out profiles. The held-out profiles and labels
are never used to fit Harmony or estimate its adjustments.

This is an out-of-sample transfer of Harmony's training corrections, not a
native harmonypy transform. The held-out-batch protocol is intentionally not
included: a batch absent from the training fold has no learned batch-specific
adjustment. Such a test requires a prespecified calibration set from that
batch (for example, DMSO controls or matched compounds).

Run:
    uv run python analysis/evaluate_fold_safe_harmony.py
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from annotation_model_utils import (
    ASSAY_FEATURES, DATA_PATH, N_SPLITS, RESULTS_DIR,
    make_label_matrix, parse_labels, train_and_score, unpack_vector,
)


FEATURES = ("pca_raw", "pca_normalized", "dino", "assay_features")
GROUPINGS = {"harmony_batch": ["batch"],
             "harmony_plate_batch": ["plate", "batch"]}
PROTOCOLS = ("held_out_compound", "held_out_chemical_group")
N_BOOTSTRAP = 10_000


def load_inputs(df: pd.DataFrame) -> dict[str, np.ndarray]:
    arrays = {
        "pca_raw": np.stack(df.pca_embedding_raw.map(unpack_vector)).astype(np.float32),
        "pca_normalized": np.stack(df.pca_embedding_normalized.map(unpack_vector)).astype(np.float32),
        "dino": np.stack(df.brightfield.map(unpack_vector)).astype(np.float32),
        "assay_features": df[ASSAY_FEATURES].to_numpy(dtype=np.float32),
    }
    for values in arrays.values():
        values[~np.isfinite(values)] = np.nan
    return arrays


def fit_fold_harmony_transfer(X: np.ndarray, meta: pd.DataFrame,
                              train: np.ndarray, test: np.ndarray,
                              group_columns: list[str],
                              max_iter_harmony: int = 20) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit Harmony on training rows and transfer train group adjustments."""
    import harmonypy

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    X_train = imputer.fit_transform(X[train]).astype(np.float32)
    X_test = imputer.transform(X[test]).astype(np.float32)
    Z_train = scaler.fit_transform(X_train).astype(np.float32)
    Z_test = scaler.transform(X_test).astype(np.float32)

    train_meta = meta.iloc[train][group_columns].astype(str).reset_index(drop=True)
    test_meta = meta.iloc[test][group_columns].astype(str).reset_index(drop=True)
    harmony = harmonypy.run_harmony(
        Z_train, train_meta, group_columns,
        max_iter_harmony=max_iter_harmony, random_state=42, verbose=False,
    )
    Z_train_corrected = np.asarray(harmony.Z_corr, dtype=np.float32)
    row_adjustment = Z_train_corrected - Z_train

    # Use robust group medians, learned only from the Harmony-corrected train
    # fold, to put held-out rows from those known groups into the same space.
    correction_frame = pd.DataFrame(row_adjustment)
    group_keys = lambda frame: frame.astype(str).agg("\x1f".join, axis=1).to_numpy()
    train_keys = group_keys(train_meta)
    test_keys = group_keys(test_meta)
    correction_frame["_group"] = train_keys
    group_adjustments = correction_frame.groupby("_group", sort=False).median(numeric_only=True)
    unknown = sorted(set(test_keys) - set(group_adjustments.index))
    if unknown:
        raise ValueError(
            f"Test fold contains groups absent from Harmony training data: {unknown[:3]}"
        )
    shifts = np.stack([group_adjustments.loc[key].to_numpy(dtype=np.float32)
                       for key in test_keys])
    Z_test_corrected = Z_test + shifts
    return Z_train_corrected, Z_test_corrected, {
        "n_train_groups": int(len(group_adjustments)),
            "n_test_groups": int(len(set(test_keys))),
        "mean_test_shift_l2": float(np.linalg.norm(shifts, axis=1).mean()),
    }


def compound_top3_metrics(y_true: np.ndarray, scores: np.ndarray,
                          compound_ids: np.ndarray) -> tuple[float, float]:
    compounds = sorted(set(compound_ids))
    truth = np.stack([y_true[np.flatnonzero(compound_ids == cid)[0]] for cid in compounds])
    compound_scores = np.stack([scores[compound_ids == cid].mean(axis=0) for cid in compounds])
    prediction = np.zeros_like(truth)
    top = np.argsort(compound_scores, axis=1)[:, -min(3, truth.shape[1]):]
    np.put_along_axis(prediction, top, 1, axis=1)
    return (float(f1_score(truth, prediction, average="macro", zero_division=0)),
            float(f1_score(truth, prediction, average="micro", zero_division=0)))


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", choices=FEATURES, default=list(FEATURES))
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--max-iter-harmony", type=int, default=20)
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    parser.add_argument("--output-tag", default="",
                        help="Optional suffix to keep focused runs separate from the main output files.")
    args = parser.parse_args()
    suffix = f"_{args.output_tag}" if args.output_tag else ""

    df = pd.read_parquet(DATA_PATH)
    df = df[df.compound_id.notna() & df.compound_pathway.notna()
            & df.compound_target.notna()].copy().reset_index(drop=True)
    ids = df.compound_id.astype(str).to_numpy()
    metadata = df[["plate", "batch"]].reset_index(drop=True)
    inputs = load_inputs(df)
    audit_path = RESULTS_DIR / "chemical_split" / "chemical_split_structure_audit.csv"
    audit = pd.read_csv(audit_path)
    chemical_map = dict(zip(audit.compound_id.astype(str), audit.chemical_group))
    fold_rows: list[dict] = []
    for protocol in args.protocols:
        if protocol == "held_out_compound":
            active_df, active_ids = df, ids
            active_inputs = inputs
            groups = ids
        else:
            keep = np.array([cid in chemical_map for cid in ids])
            active_df = df.loc[keep].reset_index(drop=True)
            active_ids = ids[keep]
            active_inputs = {name: X[keep] for name, X in inputs.items()}
            groups = np.array([chemical_map[cid] for cid in active_ids])
        splits = list(GroupKFold(n_splits=args.folds).split(active_df, groups=groups))
        active_meta = active_df[["plate", "batch"]].reset_index(drop=True)
        Y_by_task = {
            task: make_label_matrix(active_df[column].map(parse_labels).tolist(), active_ids)[0]
            for task, column in (("pathway", "compound_pathway"), ("target", "compound_target"))
        }
        print(f"Starting {protocol}: {len(splits)} folds", flush=True)

        for feature in args.features:
            X = active_inputs[feature]
            for fold, (train, test) in enumerate(splits, start=1):
                # Baseline plus each training-only Harmony variant use exactly
                # the same outer split and label vocabulary.
                variants = {"uncorrected": (X[train], X[test], {})}
                for method, covariates in GROUPINGS.items():
                    group_meta = active_meta.loc[:, covariates].reset_index(drop=True)
                    try:
                        Xtr, Xte, diag = fit_fold_harmony_transfer(
                            X, group_meta, train, test, covariates,
                            max_iter_harmony=args.max_iter_harmony,
                        )
                    except ValueError as exc:
                        print(f"Skipping {protocol} fold {fold} {feature} {method}: {exc}", flush=True)
                        continue
                    variants[method] = (Xtr, Xte, diag)
                for method, (Xtr, Xte, diag) in variants.items():
                    for task, column in (("pathway", "compound_pathway"),
                                         ("target", "compound_target")):
                        Y = Y_by_task[task]
                        scores = train_and_score(
                            Xtr, Y[train], Xte, groups_train=active_ids[train]
                        )
                        macro, micro = compound_top3_metrics(Y[test], scores, active_ids[test])
                        fold_rows.append({
                            "protocol": protocol, "fold": fold, "feature_set": feature,
                            "transform": method, "task": task,
                            "macro_f1_top3": macro, "micro_f1_top3": micro,
                            "n_test_compounds": int(len(np.unique(active_ids[test]))),
                            **diag,
                        })
                # Preserve completed folds if a large embedding makes the full
                # run lengthy or the user stops it before all inputs finish.
                pd.DataFrame(fold_rows).to_csv(
                    RESULTS_DIR / "feature_processing" / f"fold_safe_harmony_folds{suffix}.csv", index=False
                )
                print(f"{protocol:24s} {feature:16s} fold {fold}/{len(splits)} complete", flush=True)

    folds = pd.DataFrame(fold_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    folds_path = RESULTS_DIR / "feature_processing" / f"fold_safe_harmony_folds{suffix}.csv"
    folds.to_csv(folds_path, index=False)

    # Paired deltas compare each transformed score with uncorrected on the same
    # fold. Bootstrap resamples the outer folds; only five folds are available
    # for compound/chemical protocols, so intervals are descriptive.
    summary_rows = []
    for (protocol, feature, task), part in folds.groupby(["protocol", "feature_set", "task"]):
        for method in ("uncorrected", *GROUPINGS.keys()):
            current = part[part["transform"] == method].set_index("fold")
            baseline = part[part["transform"] == "uncorrected"].set_index("fold")
            paired = current.join(baseline, lsuffix="_method", rsuffix="_baseline", how="inner")
            row = {"protocol": protocol, "feature_set": feature, "transform": method,
                   "task": task, "n_folds": len(paired),
                   "macro_f1_top3_mean": float(paired.macro_f1_top3_method.mean()),
                   "micro_f1_top3_mean": float(paired.micro_f1_top3_method.mean())}
            for metric in ("macro_f1_top3", "micro_f1_top3"):
                delta = paired[f"{metric}_method"] - paired[f"{metric}_baseline"]
                low, high = bootstrap_mean_ci(delta.to_numpy(), seed=42 + len(summary_rows))
                row[f"delta_{metric}_vs_uncorrected"] = float(delta.mean())
                row[f"delta_{metric}_ci_low"] = low
                row[f"delta_{metric}_ci_high"] = high
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "feature_processing" / f"fold_safe_harmony_summary{suffix}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved fold-level scores to {folds_path}", flush=True)
    print(f"Saved paired fold summaries to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
