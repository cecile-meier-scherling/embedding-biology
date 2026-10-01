"""Create matched-compound/dose plate and batch corrected feature arrays.

Matched shifts are estimated from compound-dose profiles observed in multiple
groups. Corrections use feature values and experimental metadata only; labels
are never read. This script does not run Harmony. Harmony is evaluated
separately with training-fold-only fitting in ``evaluate_fold_safe_harmony.py``.

This is a transductive, dataset-level correction: all wells contribute to the
unsupervised shift estimates. It is suitable for comparing representations on
this dataset, but the corrected cross-validation scores are not a strict
prospective estimate for a completely unseen plate/batch.
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from annotation_model_utils import ASSAY_FEATURES, DATA_PATH, RESULTS_DIR, unpack_vector


EMBEDDINGS = {
    "pca_raw": "pca_embedding_raw",
    "pca_normalized": "pca_embedding_normalized",
    "dino": "brightfield",
}
GROUPS = ("plate", "batch")
ANCHOR_COLUMNS = ("compound_id", "compound_concentration_um")
OUTPUT_PATH = RESULTS_DIR / "feature_processing" / "corrected_feature_arrays.npz"
METADATA_PATH = RESULTS_DIR / "feature_processing" / "feature_correction_metadata.csv"


def robust_group_shifts(X: np.ndarray, keys: pd.Series, group: pd.Series,
                        min_anchors: int = 3) -> tuple[np.ndarray, dict[str, int]]:
    """Estimate each group offset from matched anchor profiles.

    For each anchor represented in multiple groups, calculate each group's
    deviation from the across-group anchor mean. The group shift is the
    coordinate-wise median deviation over anchors. Centering shifts by the
    well-count-weighted mean preserves the overall location of the data.
    """
    valid = keys.notna().to_numpy() & group.notna().to_numpy()
    keys_arr = keys.astype("string").fillna("").to_numpy()
    groups_arr = group.astype("string").fillna("").to_numpy()
    frame = pd.DataFrame({"key": keys_arr, "group": groups_arr,
                          "row": np.arange(len(X))})
    frame = frame.loc[valid]
    group_values = sorted(frame["group"].unique())
    residuals: dict[str, list[np.ndarray]] = {g: [] for g in group_values}
    anchor_counts: dict[str, int] = {g: 0 for g in group_values}
    for _, anchor_rows in frame.groupby("key", sort=False):
        by_group = anchor_rows.groupby("group")["row"].apply(list)
        if len(by_group) < 2:
            continue
        means = {g: np.nanmean(X[np.asarray(rows)], axis=0)
                 for g, rows in by_group.items()}
        reference = np.nanmean(np.stack(list(means.values())), axis=0)
        for g, mean in means.items():
            residuals[g].append(mean - reference)
            anchor_counts[g] += 1

    shifts = {}
    for g in group_values:
        if anchor_counts[g] >= min_anchors:
            shifts[g] = np.nanmedian(np.stack(residuals[g]), axis=0)
        else:
            shifts[g] = np.zeros(X.shape[1], dtype=np.float32)
    counts = frame.groupby("group").size().to_dict()
    total = sum(counts.values()) or 1
    center = sum(shifts[g] * counts.get(g, 0) for g in group_values) / total
    shifts = {g: np.nan_to_num(value - center, nan=0.0).astype(np.float32)
              for g, value in shifts.items()}
    correction = np.zeros_like(X, dtype=np.float32)
    for i, g in enumerate(groups_arr):
        if g in shifts:
            correction[i] = shifts[g]
    return correction, anchor_counts


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    n = len(df)
    dose = df["compound_concentration_um"].astype("string")
    compound = df["compound_id"].astype("string")
    keys = compound.str.cat(dose, sep="||")
    metadata = df[["plate", "batch"]].copy()
    outputs: dict[str, np.ndarray] = {}
    records: list[dict] = []

    source_arrays: dict[str, np.ndarray] = {
        name: np.stack(df[column].map(unpack_vector)).astype(np.float32)
        for name, column in EMBEDDINGS.items()
    }
    assay_array = df[ASSAY_FEATURES].to_numpy(dtype=np.float32)
    assay_array[~np.isfinite(assay_array)] = np.nan
    source_arrays["assay_features"] = assay_array

    for feature, X in source_arrays.items():
        variants: dict[str, np.ndarray] = {"uncorrected": X}
        plate_delta, plate_anchors = robust_group_shifts(
            X, keys, metadata["plate"]
        )
        variants["plate_corrected"] = X - plate_delta
        batch_delta, batch_anchors = robust_group_shifts(
            X, keys, metadata["batch"]
        )
        variants["batch_corrected"] = X - batch_delta
        # Sequential correction: estimate batch shifts after first removing
        # plate offsets, then apply the resulting batch correction.
        plate_batch_delta, plate_batch_anchors = robust_group_shifts(
            variants["plate_corrected"], keys, metadata["batch"]
        )
        variants["plate_batch_corrected"] = (
            variants["plate_corrected"] - plate_batch_delta
        )

        for variant, array in variants.items():
            outputs[f"{feature}__{variant}"] = array.astype(np.float32)
        for g_name, anchor_map in (("plate", plate_anchors),
                                   ("batch", batch_anchors),
                                   ("batch_after_plate", plate_batch_anchors)):
            for group_name, count in anchor_map.items():
                records.append({"feature_set": feature, "correction": g_name,
                                "group": group_name, "n_matched_anchors": count})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, **outputs)
    pd.DataFrame(records).to_csv(METADATA_PATH, index=False)
    print(f"Saved {len(outputs)} well-level feature arrays ({n} wells) to {OUTPUT_PATH}")
    print(f"Saved matched-anchor counts to {METADATA_PATH}")


if __name__ == "__main__":
    main()
