"""Evaluate nested preprocessing selection, nonlinear models and dose effects.

All model comparisons reuse held-out-compound and held-out-chemical-group
outer folds. Nested preprocessing choices are selected only within each outer
training partition. The nonlinear probe uses a fold-fitted random Fourier
approximation to an RBF kernel. Dose-aware models add log10 concentration;
per-dose results diagnose whether averaging dose predictions hides signal.

Run the full analysis:
    uv run python analysis/evaluate_next_step_models.py

Run selected parts:
    uv run python analysis/evaluate_next_step_models.py --stages nested nonlinear dose learnability
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from annotation_model_utils import (
    ASSAY_FEATURES, DATA_PATH, N_SPLITS, RESULTS_DIR, make_label_matrix,
    parse_labels, train_and_score, unpack_vector,
)
from evaluate_processing_candidates import (
    FEATURES, TRANSFORMS, compound_level_metrics, fit_transform_fold,
)


PROTOCOLS = ("held_out_compound", "held_out_chemical_group")
INNER_SPLITS = 3
RBF_COMPONENTS = 256


def load_inputs(df):
    data = {
        "pca_raw": np.stack(df.pca_embedding_raw.map(unpack_vector)).astype(np.float32),
        "pca_normalized": np.stack(df.pca_embedding_normalized.map(unpack_vector)).astype(np.float32),
        "dino": np.stack(df.brightfield.map(unpack_vector)).astype(np.float32),
        "assay_features": df[ASSAY_FEATURES].to_numpy(dtype=np.float32),
    }
    for x in data.values():
        x[~np.isfinite(x)] = np.nan
    return data


def protocol_data(df, inputs, ids, protocol, chemical_map):
    if protocol == "held_out_compound":
        groups = ids
        X = inputs
        current_df = df
    else:
        keep = np.array([cid in chemical_map for cid in ids])
        current_df = df.loc[keep].reset_index(drop=True)
        groups = ids[keep]
        X = {key: values[keep] for key, values in inputs.items()}
    if protocol == "held_out_compound":
        splits = list(GroupKFold(N_SPLITS).split(current_df, groups=groups))
        outer_groups = groups
    else:
        outer_groups = np.array([chemical_map[cid] for cid in groups])
        splits = list(GroupKFold(N_SPLITS).split(current_df, groups=outer_groups))
    return current_df, groups, X, splits, outer_groups


def label_matrices(df, compound_ids):
    result = {}
    for task, column in (("pathway", "compound_pathway"), ("target", "compound_target")):
        y, label_names = make_label_matrix(df[column].map(parse_labels).tolist(), compound_ids)
        result[task] = (y, label_names)
    return result


def choose_transform_nested(X, meta, Y, compound_ids, inner_groups):
    """Choose a normalized-PCA transform using only outer-training rows."""
    candidates = ("uncorrected", "matched_plate", "repeatable_top75", "rank_gaussian", "pca50_whiten")
    split_count = min(INNER_SPLITS, len(np.unique(inner_groups)))
    if split_count < 2:
        return "uncorrected", {}
    scores = {name: [] for name in candidates}
    inner_splits = list(GroupKFold(split_count).split(X, groups=inner_groups))
    for name in candidates:
        for inner_train, inner_valid in inner_splits:
            xa, xb = fit_transform_fold(X, meta, inner_train, inner_valid, name)
            prediction = train_and_score(xa, Y[inner_train], xb,
                                         groups_train=compound_ids[inner_train])
            metric = compound_level_metrics(Y[inner_valid], prediction, compound_ids[inner_valid])
            scores[name].append(metric["macro_f1_top3"])
    means = {name: float(np.mean(values)) for name, values in scores.items()}
    # A tie favors the less invasive transformation.
    priority = {name: -i for i, name in enumerate(candidates)}
    selected = max(candidates, key=lambda name: (means[name], priority[name]))
    return selected, means


def run_nested(df, groups, X_by_feature, splits, inner_group_ids, protocol, task_data):
    rows = []
    for task, (Y, _) in task_data.items():
        X = X_by_feature["pca_normalized"]
        meta = df[["compound_id", "compound_concentration_um", "plate", "batch"]].reset_index(drop=True)
        for fold, (outer_train, outer_test) in enumerate(splits, start=1):
            selected, inner_scores = choose_transform_nested(
                X[outer_train], meta.iloc[outer_train].reset_index(drop=True), Y[outer_train],
                groups[outer_train], inner_group_ids[outer_train],
            )
            fitted_train, fitted_test = fit_transform_fold(
                X, meta, outer_train, outer_test, selected
            )
            selected_scores = train_and_score(
                fitted_train, Y[outer_train], fitted_test, groups_train=groups[outer_train]
            )
            baseline_scores = train_and_score(
                X[outer_train], Y[outer_train], X[outer_test], groups_train=groups[outer_train]
            )
            chosen_metric = compound_level_metrics(Y[outer_test], selected_scores, groups[outer_test])
            baseline_metric = compound_level_metrics(Y[outer_test], baseline_scores, groups[outer_test])
            rows.append({"protocol": protocol, "task": task, "fold": fold,
                         "selected_transform": selected,
                         "selected_macro_f1_top3": chosen_metric["macro_f1_top3"],
                         "selected_micro_f1_top3": chosen_metric["micro_f1_top3"],
                         "uncorrected_macro_f1_top3": baseline_metric["macro_f1_top3"],
                         "uncorrected_micro_f1_top3": baseline_metric["micro_f1_top3"],
                         **{f"inner_{key}_macro_f1": value for key, value in inner_scores.items()}})
            print(f"{protocol} nested {task} fold {fold}: selected {selected}", flush=True)
    return rows


def rbf_scores(x_train, x_test, y_train, train_groups):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    a = scaler.fit_transform(imputer.fit_transform(x_train))
    b = scaler.transform(imputer.transform(x_test))
    gamma = 1.0 / max(1, a.shape[1])
    sampler = RBFSampler(gamma=gamma, n_components=RBF_COMPONENTS, random_state=42)
    a = sampler.fit_transform(a)
    b = sampler.transform(b)
    return train_and_score(a, y_train, b, groups_train=train_groups)


def dose_covariate(meta):
    dose = pd.to_numeric(meta.compound_concentration_um, errors="coerce").to_numpy(dtype=float)
    dose = np.log10(np.maximum(dose, 1e-8))
    return dose[:, None].astype(np.float32)


def per_dose_metrics(y, scores, ids, doses, train_edges):
    rows = []
    band_ids = np.digitize(doses, train_edges[1:-1], right=True)
    names = ("low", "mid", "high")
    for bin_id in np.unique(band_ids):
        use = band_ids == bin_id
        if not use.any():
            continue
        rows.append({"dose_band": names[min(int(bin_id), 2)],
                     **compound_level_metrics(y[use], scores[use], ids[use])})
    return rows


def paired_delta_summary(frame, first, second, metric, key_columns):
    rows = []
    rng = np.random.default_rng(42)
    for key, part in frame.groupby(key_columns, dropna=False):
        a = part[part.model == first].set_index("fold")[metric]
        b = part[part.model == second].set_index("fold")[metric]
        common = a.index.intersection(b.index)
        if not len(common):
            continue
        delta = (b.loc[common] - a.loc[common]).to_numpy()
        boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(key_columns, key))
        row.update({"metric": metric, f"delta_{second}_vs_{first}": float(delta.mean()),
                    "delta_ci_low": float(np.quantile(boot, .025)),
                    "delta_ci_high": float(np.quantile(boot, .975)),
                    "n_folds": len(delta)})
        rows.append(row)
    return pd.DataFrame(rows)


def write_nested_summary(frame):
    rows = []
    rng = np.random.default_rng(42)
    keys = ["protocol", "task"]
    for key, part in frame.groupby(keys):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        row["n_folds"] = int(part.fold.nunique())
        for metric in ("macro_f1_top3", "micro_f1_top3"):
            delta = (part[f"selected_{metric}"] - part[f"uncorrected_{metric}"]).to_numpy()
            boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
            row[f"selected_{metric}_mean"] = float(part[f"selected_{metric}"].mean())
            row[f"uncorrected_{metric}_mean"] = float(part[f"uncorrected_{metric}"].mean())
            row[f"delta_{metric}"] = float(delta.mean())
            row[f"delta_{metric}_ci_low"] = float(np.quantile(boot, .025))
            row[f"delta_{metric}_ci_high"] = float(np.quantile(boot, .975))
        row["selected_transforms"] = "; ".join(part.selected_transform.astype(str))
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "annotation_prediction" / "nested_transform_selection_summary.csv", index=False)


def write_dose_summary(frame):
    overall = frame[frame.model != "baseline_by_dose_band"].copy()
    overall_summary = overall.groupby(["protocol", "task", "feature_set", "model"], as_index=False).agg(
        n_folds=("fold", "nunique"), macro_f1_top3_mean=("macro_f1_top3", "mean"),
        macro_f1_top3_sd=("macro_f1_top3", "std"), micro_f1_top3_mean=("micro_f1_top3", "mean"),
        micro_f1_top3_sd=("micro_f1_top3", "std"))
    bands = frame[frame.model == "baseline_by_dose_band"]
    band_summary = bands.groupby(["protocol", "task", "feature_set", "dose_band"], as_index=False).agg(
        model=("model", "first"), n_folds=("fold", "nunique"),
        macro_f1_top3_mean=("macro_f1_top3", "mean"), macro_f1_top3_sd=("macro_f1_top3", "std"),
        micro_f1_top3_mean=("micro_f1_top3", "mean"), micro_f1_top3_sd=("micro_f1_top3", "std"))
    overall_summary["dose_band"] = np.nan
    combined = pd.concat([overall_summary, band_summary], ignore_index=True)
    combined.to_csv(RESULTS_DIR / "annotation_prediction" / "dose_aware_prediction_summary.csv", index=False)
    delta_rows = []
    for metric in ("macro_f1_top3", "micro_f1_top3"):
        delta_rows.append(paired_delta_summary(
            overall, "no_dose_covariate", "log10_dose_covariate", metric,
            ["protocol", "task", "feature_set"]))
    pd.concat(delta_rows, ignore_index=True).to_csv(
        RESULTS_DIR / "annotation_prediction" / "dose_aware_prediction_paired_deltas.csv", index=False)


def write_nonlinear_summary(frame):
    summary = frame.groupby(["protocol", "task", "feature_set", "model"], as_index=False).agg(
        n_folds=("fold", "nunique"), macro_f1_top3_mean=("macro_f1_top3", "mean"),
        macro_f1_top3_sd=("macro_f1_top3", "std"), micro_f1_top3_mean=("micro_f1_top3", "mean"),
        micro_f1_top3_sd=("micro_f1_top3", "std"))
    summary.to_csv(RESULTS_DIR / "annotation_prediction" / "nonlinear_model_summary.csv", index=False)
    delta_rows = []
    for metric in ("macro_f1_top3", "micro_f1_top3"):
        delta_rows.append(paired_delta_summary(
            frame, "linear_logistic", "rbf_random_features", metric,
            ["protocol", "task", "feature_set"]))
    pd.concat(delta_rows, ignore_index=True).to_csv(
        RESULTS_DIR / "annotation_prediction" / "nonlinear_model_paired_deltas.csv", index=False)


def run_predictor_comparisons(df, groups, inputs, splits, protocol, task_data, stages):
    model_rows, dose_rows, label_rows = [], [], []
    meta = df[["compound_id", "compound_concentration_um", "plate", "batch"]].reset_index(drop=True)
    doses = dose_covariate(meta).ravel()
    for feature in FEATURES:
        X = inputs[feature]
        for fold, (train, test) in enumerate(splits, start=1):
            if "nonlinear" in stages:
                for task, (Y, _) in task_data.items():
                    scores = rbf_scores(X[train], X[test], Y[train], groups[train])
                    metric = compound_level_metrics(Y[test], scores, groups[test])
                    model_rows.append({"protocol": protocol, "task": task, "feature_set": feature,
                                       "model": "rbf_random_features", "fold": fold, **metric})
                    linear = train_and_score(X[train], Y[train], X[test], groups_train=groups[train])
                    linear_metric = compound_level_metrics(Y[test], linear, groups[test])
                    model_rows.append({"protocol": protocol, "task": task, "feature_set": feature,
                                       "model": "linear_logistic", "fold": fold, **linear_metric})
            if "dose" in stages:
                X_dose = np.column_stack([X, doses])
                for task, (Y, _) in task_data.items():
                    baseline = train_and_score(X[train], Y[train], X[test], groups_train=groups[train])
                    dose_scores = train_and_score(X_dose[train], Y[train], X_dose[test],
                                                  groups_train=groups[train])
                    for model_name, scores in (("no_dose_covariate", baseline),
                                               ("log10_dose_covariate", dose_scores)):
                        metric = compound_level_metrics(Y[test], scores, groups[test])
                        dose_rows.append({"protocol": protocol, "task": task,
                                          "feature_set": feature, "model": model_name,
                                          "fold": fold, **metric})
                    # Hold the model fixed and evaluate each compound-dose
                    # profile separately to reveal dose-dependent prediction.
                    train_edges = np.quantile(doses[train], [0, 1 / 3, 2 / 3, 1])
                    per_dose = per_dose_metrics(Y[test], baseline, groups[test], doses[test], train_edges)
                    for item in per_dose:
                        dose_rows.append({"protocol": protocol, "task": task,
                                          "feature_set": feature, "model": "baseline_by_dose_band",
                                          "fold": fold, **item})
            if "learnability" in stages and feature == "pca_normalized":
                for task, (Y, label_names) in task_data.items():
                    scores = train_and_score(X[train], Y[train], X[test], groups_train=groups[train])
                    compounds = sorted(set(groups[test]))
                    compound_truth = np.stack([Y[test[np.flatnonzero(groups[test] == c)[0]]]
                                               for c in compounds])
                    compound_scores = np.stack([scores[groups[test] == c].mean(axis=0)
                                                for c in compounds])
                    predicted = np.zeros_like(compound_truth)
                    top = np.argsort(compound_scores, axis=1)[:, -min(3, len(label_names)):]
                    np.put_along_axis(predicted, top, 1, axis=1)
                    for j, label in enumerate(label_names):
                        tp = int(((compound_truth[:, j] == 1) & (predicted[:, j] == 1)).sum())
                        fp = int(((compound_truth[:, j] == 0) & (predicted[:, j] == 1)).sum())
                        fn = int(((compound_truth[:, j] == 1) & (predicted[:, j] == 0)).sum())
                        support = int(compound_truth[:, j].sum())
                        label_rows.append({"protocol": protocol, "task": task, "fold": fold,
                                           "label": label, "test_compound_support": support,
                                           "true_positive": tp, "false_positive": fp, "false_negative": fn,
                                           "recall": tp / (tp + fn) if tp + fn else np.nan,
                                           "precision": tp / (tp + fp) if tp + fp else np.nan})
            print(f"{protocol} predictors {feature} fold {fold} complete", flush=True)
    return model_rows, dose_rows, label_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=("nested", "nonlinear", "dose", "learnability"),
                        default=["nested", "nonlinear", "dose", "learnability"])
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    args = parser.parse_args()

    df = pd.read_parquet(DATA_PATH)
    df = df[df.compound_id.notna() & df.compound_pathway.notna() & df.compound_target.notna()].copy().reset_index(drop=True)
    ids = df.compound_id.astype(str).to_numpy()
    inputs = load_inputs(df)
    audit = pd.read_csv(RESULTS_DIR / "chemical_split" / "chemical_split_structure_audit.csv")
    chemical_map = dict(zip(audit.compound_id.astype(str), audit.chemical_group))
    nested_rows, nonlinear_rows, dose_rows, label_rows = [], [], [], []

    for protocol in args.protocols:
        current_df, groups, current_inputs, splits, outer_groups = protocol_data(
            df, inputs, ids, protocol, chemical_map
        )
        tasks = label_matrices(current_df, groups)
        if "nested" in args.stages:
            nested_rows.extend(run_nested(current_df, groups, current_inputs, splits,
                                          outer_groups, protocol, tasks))
        model, dose, labels = run_predictor_comparisons(
            current_df, groups, current_inputs, splits, protocol, tasks, args.stages
        )
        nonlinear_rows.extend(model); dose_rows.extend(dose); label_rows.extend(labels)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if nested_rows:
        nested_frame = pd.DataFrame(nested_rows)
        nested_frame.to_csv(RESULTS_DIR / "annotation_prediction" / "nested_transform_selection_folds.csv", index=False)
        write_nested_summary(nested_frame)
    if nonlinear_rows:
        d = pd.DataFrame(nonlinear_rows)
        d.to_csv(RESULTS_DIR / "annotation_prediction" / "nonlinear_model_folds.csv", index=False)
        write_nonlinear_summary(d)
    if dose_rows:
        d = pd.DataFrame(dose_rows)
        d.to_csv(RESULTS_DIR / "annotation_prediction" / "dose_aware_prediction_folds.csv", index=False)
        write_dose_summary(d)
    if label_rows:
        labels = pd.DataFrame(label_rows)
        labels.to_csv(RESULTS_DIR / "annotation_prediction" / "label_frequency_performance.csv", index=False)
        labels["frequency_bin"] = pd.cut(labels.test_compound_support,
                                          bins=[-1, 4, 9, 24, np.inf],
                                          labels=["1-4", "5-9", "10-24", "25+"])
        labels.groupby(["protocol", "task", "frequency_bin"], observed=False, as_index=False).agg(
            n_label_fold_observations=("label", "size"), mean_support=("test_compound_support", "mean"),
            mean_recall=("recall", "mean"), mean_precision=("precision", "mean")).to_csv(
                RESULTS_DIR / "annotation_prediction" / "label_frequency_performance_summary.csv", index=False)
    print(f"Saved requested next-step analyses under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
