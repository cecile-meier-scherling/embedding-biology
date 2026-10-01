"""Evaluate retrieval, replicate consistency, and technical-label recovery for
fold-safe Harmony and DMSO-control-whitened representations.

Harmony is fitted on each training fold only, using the same transferred group
adjustment as ``evaluate_fold_safe_harmony.py``. DMSO whitening uses per-plate
control calibrations from the saved arrays; results are restricted to wells on
calibrated plates (16 plates from one batch).

Run from the repository root:
    .venv/bin/python analysis/evaluate_additional_processing_diagnostics.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from annotation_model_utils import DATA_PATH, N_SPLITS, RESULTS_DIR, unpack_vector
from evaluate_fold_safe_harmony import fit_fold_harmony_transfer
from evaluate_transformation_diagnostics import (
    label_truth,
    replicate_pair_metrics,
    retrieval_for_fold,
)
from evaluate_processing_candidates import nuisance_scores


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("pca_raw", "pca_normalized", "dino")
PROTOCOLS = ("held_out_compound", "held_out_chemical_group")
GROUPINGS = {"harmony_batch": ["batch"], "harmony_plate_batch": ["plate", "batch"]}
NUISANCE = ("plate", "batch")


def load_inputs(df: pd.DataFrame) -> dict[str, np.ndarray]:
    arrays = {
        "pca_raw": np.stack(df.pca_embedding_raw.map(unpack_vector)).astype(np.float32),
        "pca_normalized": np.stack(df.pca_embedding_normalized.map(unpack_vector)).astype(np.float32),
        "dino": np.stack(df.brightfield.map(unpack_vector)).astype(np.float32),
    }
    for x in arrays.values():
        x[~np.isfinite(x)] = np.nan
    return arrays


def standardized_train_test(X: np.ndarray, train: np.ndarray, test: np.ndarray):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    xtr = imputer.fit_transform(X[train]).astype(np.float32)
    xte = imputer.transform(X[test]).astype(np.float32)
    return scaler.fit_transform(xtr).astype(np.float32), scaler.transform(xte).astype(np.float32)


def make_splits(df: pd.DataFrame, ids: np.ndarray, protocol: str,
                chemical_map: dict[str, str]):
    if protocol == "held_out_compound":
        active_df, active_ids = df.reset_index(drop=True), ids
        groups = active_ids
    else:
        keep = np.array([cid in chemical_map for cid in ids])
        active_df = df.loc[keep].reset_index(drop=True)
        active_ids = ids[keep]
        groups = np.array([chemical_map[cid] for cid in active_ids])
    splits = list(GroupKFold(N_SPLITS).split(active_df, groups=groups))
    return active_df, active_ids, groups, splits


def run_harmony(df, ids, inputs, chemical_map, rng_seed=42):
    rows_retrieval, rows_replicate, rows_confound = [], [], []
    for protocol in PROTOCOLS:
        active_df, active_ids, groups, splits = make_splits(df, ids, protocol, chemical_map)
        meta = active_df[["compound_id", "compound_concentration_um", "plate", "batch", "source"]].reset_index(drop=True)
        labels = {
            column: label_truth(active_df, active_ids, column)[0]
            for column in ("compound_pathway", "compound_target")
        }
        for feature in FEATURES:
            X = inputs[feature]
            for fold, (train, test) in enumerate(splits, start=1):
                base_train, base_test = standardized_train_test(X, train, test)
                variants = {"harmony_uncorrected": (base_train, base_test)}
                for transform, columns in GROUPINGS.items():
                    group_meta = meta[columns].reset_index(drop=True)
                    htrain, htest, _ = fit_fold_harmony_transfer(
                        X, group_meta, train, test, columns
                    )
                    variants[transform] = (htrain, htest)

                train_meta = meta.iloc[train].reset_index(drop=True)
                test_meta = meta.iloc[test].reset_index(drop=True)
                for transform, (xtr, xte) in variants.items():
                    rows_replicate.append({
                        "protocol": protocol, "fold": fold, "feature_set": feature,
                        "transform": transform, **replicate_pair_metrics(xte, test_meta, active_ids[test], np.random.default_rng(rng_seed + fold)),
                    })
                    for task, column in (("pathway", "compound_pathway"), ("target", "compound_target")):
                        rows_retrieval.extend(retrieval_for_fold(
                            protocol, fold, transform, feature, xtr, xte,
                            active_ids[train], active_ids[test],
                            active_df.iloc[test].reset_index(drop=True), labels,
                        ))
                    for nuisance in NUISANCE:
                        levels = active_df.groupby(active_ids)[nuisance].nunique()
                        multi = set(levels[levels >= 2].index.astype(str))
                        tr_keep = np.isin(active_ids[train], list(multi)) & train_meta[nuisance].notna().to_numpy()
                        te_keep = np.isin(active_ids[test], list(multi)) & test_meta[nuisance].notna().to_numpy()
                        if not tr_keep.any() or not te_keep.any() or test_meta.loc[te_keep, nuisance].nunique() < 2:
                            continue
                        try:
                            score = nuisance_scores(
                                xtr[tr_keep], xte[te_keep],
                                train_meta.loc[tr_keep].reset_index(drop=True),
                                test_meta.loc[te_keep].reset_index(drop=True),
                                active_ids[train][tr_keep], active_ids[test][te_keep], nuisance,
                            )
                        except ValueError:
                            continue
                        rows_confound.append({"protocol": protocol, "fold": fold,
                                              "feature_set": feature, "transform": transform,
                                              "nuisance": nuisance, **score})
                for name, rows in (("retrieval", rows_retrieval), ("replicate", rows_replicate),
                                   ("confounding", rows_confound)):
                    pd.DataFrame(rows).to_csv(RESULTS_DIR / "feature_processing" / f"harmony_{name}_diagnostics.partial.csv", index=False)
                print(f"Harmony {protocol} {feature} fold {fold}/{len(splits)} complete", flush=True)
    return rows_retrieval, rows_replicate, rows_confound


def run_dmso_diagnostics(df, chemical_map, seed=42):
    arrays_path = RESULTS_DIR / "feature_processing" / "dmso_whitened_feature_arrays.npz"
    arrays = np.load(arrays_path)
    source_rows = arrays["source_row_index"].astype(int)
    active = df.iloc[source_rows].copy().reset_index(drop=True)
    active["_source_row_index"] = source_rows
    keep_labelled = active.compound_id.notna() & active.compound_pathway.notna() & active.compound_target.notna()
    active = active.loc[keep_labelled].reset_index(drop=True)
    ids = active.compound_id.astype(str).to_numpy()
    positions = active["_source_row_index"].to_numpy()
    array_positions = pd.Series(np.arange(len(source_rows)), index=source_rows)
    row_positions = array_positions.loc[positions].to_numpy()
    meta_columns = ["compound_id", "compound_concentration_um", "plate", "batch", "source"]

    all_ret, all_rep, all_conf = [], [], []
    for protocol in PROTOCOLS:
        active_df, active_ids, groups, splits = make_splits(active, ids, protocol, chemical_map)
        # Chemical split may subset rows; retain their positions in the saved arrays.
        if protocol == "held_out_compound":
            keep = np.ones(len(active), dtype=bool)
        else:
            keep = np.array([cid in chemical_map for cid in ids])
        filtered_positions = row_positions[keep]
        filtered_df = active.loc[keep].reset_index(drop=True)
        filtered_ids = ids[keep]
        meta = filtered_df[meta_columns].reset_index(drop=True)
        labels = {column: label_truth(filtered_df, filtered_ids, column)[0]
                  for column in ("compound_pathway", "compound_target")}
        for feature in FEATURES:
            raw = arrays[f"{feature}__uncorrected"][filtered_positions]
            corrected = arrays[f"{feature}__dmso_whitened"][filtered_positions]
            for fold, (train, test) in enumerate(splits, start=1):
                variants = {"uncorrected": (raw[train], raw[test]),
                            "dmso_whitened": (corrected[train], corrected[test])}
                train_meta, test_meta = meta.iloc[train].reset_index(drop=True), meta.iloc[test].reset_index(drop=True)
                for transform, (xtr, xte) in variants.items():
                    all_rep.append({"protocol": protocol, "fold": fold, "feature_set": feature,
                                    "transform": transform,
                                    **replicate_pair_metrics(xte, test_meta, filtered_ids[test], np.random.default_rng(seed + fold))})
                    for task, column in (("pathway", "compound_pathway"), ("target", "compound_target")):
                        all_ret.extend(retrieval_for_fold(
                            protocol, fold, transform, feature, xtr, xte,
                            filtered_ids[train], filtered_ids[test],
                            filtered_df.iloc[test].reset_index(drop=True), labels,
                        ))
                    for nuisance in NUISANCE:
                        levels = filtered_df.groupby(filtered_ids)[nuisance].nunique()
                        multi = set(levels[levels >= 2].index.astype(str))
                        tr_keep = np.isin(filtered_ids[train], list(multi)) & train_meta[nuisance].notna().to_numpy()
                        te_keep = np.isin(filtered_ids[test], list(multi)) & test_meta[nuisance].notna().to_numpy()
                        if not tr_keep.any() or not te_keep.any() or test_meta.loc[te_keep, nuisance].nunique() < 2:
                            continue
                        try:
                            score = nuisance_scores(
                                xtr[tr_keep], xte[te_keep],
                                train_meta.loc[tr_keep].reset_index(drop=True),
                                test_meta.loc[te_keep].reset_index(drop=True),
                                filtered_ids[train][tr_keep], filtered_ids[test][te_keep], nuisance,
                            )
                        except ValueError:
                            continue
                        all_conf.append({"protocol": protocol, "fold": fold,
                                         "feature_set": feature, "transform": transform,
                                         "nuisance": nuisance, **score})
                print(f"DMSO {protocol} {feature} fold {fold}/{len(splits)} complete", flush=True)
                for name, rows in (("retrieval", all_ret), ("replicate", all_rep), ("confounding", all_conf)):
                    pd.DataFrame(rows).to_csv(RESULTS_DIR / "feature_processing" / f"dmso_{name}_diagnostics.partial.csv", index=False)
    return all_ret, all_rep, all_conf


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    audit = pd.read_csv(RESULTS_DIR / "chemical_split" / "chemical_split_structure_audit.csv")
    chemical_map = dict(zip(audit.compound_id.astype(str), audit.chemical_group))
    labelled = df[df.compound_id.notna() & df.compound_pathway.notna() & df.compound_target.notna()].copy().reset_index(drop=True)
    harmony_inputs = load_inputs(labelled)
    harmony_ids = labelled.compound_id.astype(str).to_numpy()
    harmony = run_harmony(labelled, harmony_ids, harmony_inputs, chemical_map)
    dmso = run_dmso_diagnostics(df, chemical_map)

    for name, rows in zip(("retrieval", "replicate", "confounding"), zip(harmony, dmso)):
        merged = pd.concat([pd.DataFrame(rows[0]), pd.DataFrame(rows[1])], ignore_index=True)
        merged.to_csv(RESULTS_DIR / "feature_processing" / f"additional_{name}_diagnostics.csv", index=False)
        (RESULTS_DIR / "feature_processing" / f"harmony_{name}_diagnostics.partial.csv").unlink(missing_ok=True)
        (RESULTS_DIR / "feature_processing" / f"dmso_{name}_diagnostics.partial.csv").unlink(missing_ok=True)
    print("Saved Harmony and DMSO retrieval, replicate, and confounding diagnostics.", flush=True)


if __name__ == "__main__":
    main()
