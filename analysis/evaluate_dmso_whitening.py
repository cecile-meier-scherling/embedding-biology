"""DMSO-control-calibrated shrinkage whitening for embedding profiles.

For each plate with enough DMSO wells, estimate a control center and a
shrinkage covariance from DMSO profiles only. Apply the inverse-square-root
covariance transform to every well on that plate. Plates without controls are
left out of this corrected evaluation rather than being assigned an estimated
correction from another plate.

Run from the repository root:
    uv run python analysis/evaluate_dmso_whitening.py

Outputs include plate control coverage, corrected arrays for calibrated plates,
and held-out-compound pathway/target prediction metrics for corrected and
uncorrected inputs. Since the current data's DMSO controls occur in only one
batch, this script does not claim a held-out-batch correction result.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import GroupKFold

from annotation_model_utils import (
    ASSAY_FEATURES, DATA_PATH, N_SPLITS, RESULTS_DIR, make_label_matrix,
    parse_labels, run_prediction, unpack_vector,
)


MIN_DMSO_WELLS = 30
FEATURE_COLUMNS = {
    "pca_raw": "pca_embedding_raw",
    "pca_normalized": "pca_embedding_normalized",
    "dino": "brightfield",
    "assay_features": None,
}


def load_inputs(df: pd.DataFrame) -> dict[str, np.ndarray]:
    matrices = {
        "pca_raw": np.stack(df.pca_embedding_raw.map(unpack_vector)),
        "pca_normalized": np.stack(df.pca_embedding_normalized.map(unpack_vector)),
        "dino": np.stack(df.brightfield.map(unpack_vector)),
        "assay_features": df[ASSAY_FEATURES].to_numpy(dtype=np.float32),
    }
    for X in matrices.values():
        X[~np.isfinite(X)] = np.nan
    return matrices


def _impute_from_controls(control: np.ndarray, values: np.ndarray):
    center = np.nanmedian(control, axis=0)
    center[~np.isfinite(center)] = 0.0
    control_filled = np.where(np.isfinite(control), control, center)
    values_filled = np.where(np.isfinite(values), values, center)
    return center.astype(np.float64), control_filled.astype(np.float64), values_filled.astype(np.float64)


def dmso_whiten_by_plate(X: np.ndarray, plates: np.ndarray,
                         dmso_mask: np.ndarray,
                         min_controls: int = MIN_DMSO_WELLS):
    """Return whitened wells plus per-plate calibration audit records."""
    corrected = np.full(X.shape, np.nan, dtype=np.float32)
    audit: list[dict] = []
    for plate in sorted(pd.unique(plates.astype(str))):
        plate_rows = np.flatnonzero(plates.astype(str) == plate)
        control_rows = plate_rows[dmso_mask[plate_rows]]
        n_controls = len(control_rows)
        record = {"plate": plate, "n_dmso_wells": n_controls,
                  "n_features": int(X.shape[1]), "calibrated": False,
                  "shrinkage": np.nan, "status": "insufficient_controls"}
        if n_controls < min_controls:
            audit.append(record)
            continue

        center, controls, values = _impute_from_controls(
            X[control_rows], X[plate_rows]
        )
        centered_controls = controls - center
        # Shrinkage makes the covariance invertible even when controls are
        # fewer than embedding dimensions (common for DINO representations).
        covariance_model = LedoitWolf(assume_centered=True).fit(centered_controls)
        covariance = covariance_model.covariance_
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        floor = max(float(np.max(eigenvalues)) * 1e-6, 1e-8)
        inverse_sqrt = (eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, floor)))) @ eigenvectors.T
        corrected[plate_rows] = ((values - center) @ inverse_sqrt).astype(np.float32)
        record.update({"calibrated": True, "shrinkage": float(covariance_model.shrinkage_),
                       "status": "ok"})
        audit.append(record)
    return corrected, audit


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    df["_source_row_index"] = np.arange(len(df), dtype=np.int64)
    if df["plate"].isna().any():
        raise ValueError("Plate metadata contains missing values; cannot calibrate by plate.")

    controls = df["compound_id"].isna().to_numpy()
    plates = df["plate"].astype(str).to_numpy()
    inputs = load_inputs(df)
    corrected_inputs: dict[str, np.ndarray] = {}
    audit_by_feature: dict[str, list[dict]] = {}
    for feature, X in inputs.items():
        corrected, audit = dmso_whiten_by_plate(X, plates, controls)
        corrected_inputs[feature] = corrected
        audit_by_feature[feature] = audit
        calibrated = sum(row["calibrated"] for row in audit)
        print(f"{feature:18s}: calibrated {calibrated}/{len(audit)} plates")

    audit_frame = pd.DataFrame([
        {"feature_set": feature, **record}
        for feature, records in audit_by_feature.items() for record in records
    ])
    audit_path = RESULTS_DIR / "feature_processing" / "dmso_whitening_plate_audit.csv"
    audit_frame.to_csv(audit_path, index=False)

    calibrated_plates = set(audit_frame.loc[
        (audit_frame.feature_set == "pca_normalized") & audit_frame.calibrated, "plate"
    ].astype(str))
    calibrated_rows = np.flatnonzero(np.isin(plates, list(calibrated_plates)))
    arrays_path = RESULTS_DIR / "feature_processing" / "dmso_whitened_feature_arrays.npz"
    np.savez_compressed(
        arrays_path,
        source_row_index=df.loc[calibrated_rows, "_source_row_index"].to_numpy(),
        **{f"{name}__uncorrected": X[calibrated_rows]
           for name, X in inputs.items()},
        **{f"{name}__dmso_whitened": X[calibrated_rows]
           for name, X in corrected_inputs.items()},
    )

    treated = df.loc[calibrated_rows].copy().reset_index(drop=True)
    treated = treated[treated.compound_id.notna() & treated.compound_pathway.notna()
                      & treated.compound_target.notna()].reset_index(drop=True)
    row_ids = treated["_source_row_index"].to_numpy(dtype=int)
    compound_ids = treated.compound_id.astype(str).to_numpy()
    splits = list(GroupKFold(N_SPLITS).split(treated, groups=compound_ids))

    metric_rows: list[dict] = []
    stability_rows: list[dict] = []
    for task, label_column in (("pathway", "compound_pathway"),
                               ("target", "compound_target")):
        labels = treated[label_column].map(parse_labels).tolist()
        Y, _ = make_label_matrix(labels, compound_ids)
        for feature, raw in inputs.items():
            for version, matrix in (("uncorrected", raw),
                                    ("dmso_whitened", corrected_inputs[feature])):
                X = matrix[row_ids]
                result, _ = run_prediction(
                    X, Y, compound_ids, splits, task, f"{feature}_{version}",
                    "held_out_compound", stability_rows=stability_rows,
                    bootstrap_units=compound_ids,
                )
                result["n_calibrated_plates"] = len(calibrated_plates)
                result["n_dmso_control_wells"] = int(controls.sum())
                metric_rows.append(result)
                print(f"{task:8s} {feature:18s} {version:14s} "
                      f"Macro-F1@3={result['macro_f1_top3']:.3f} "
                      f"Micro-F1@3={result['micro_f1_top3']:.3f}")

    metrics_path = RESULTS_DIR / "feature_processing" / "dmso_whitening_prediction_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    stability_path = RESULTS_DIR / "feature_processing" / "dmso_whitening_prediction_stability.csv"
    pd.DataFrame(stability_rows).to_csv(stability_path, index=False)
    control_batches = sorted(df.loc[controls, "batch"].astype(str).unique())
    print(f"\nDMSO control wells: {int(controls.sum())}; control batches: {control_batches}")
    print(f"Evaluation compounds: {treated.compound_id.nunique()} on {len(calibrated_plates)} calibrated plates")
    print(f"Saved plate audit: {audit_path}")
    print(f"Saved corrected arrays: {arrays_path}")
    print(f"Saved prediction metrics: {metrics_path}")
    print(f"Saved fold and compound-bootstrap stability: {stability_path}")
    print("Held-out-batch correction is not evaluated because other batches have no DMSO calibration plates.")


if __name__ == "__main__":
    run()
