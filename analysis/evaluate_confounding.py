"""Probe technical information in representations and compare with biology scores.

For treated wells, profiles are centered within compound before predicting
plate, batch, or source. All wells for a compound remain in one CV fold. DMSO
wells provide an independent, treatment-free technical-control probe.

Run from the repository root:
    uv run python analysis/evaluate_confounding.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier

from annotation_model_utils import ASSAY_FEATURES, DATA_PATH, RESULTS_DIR, unpack_vector


INPUTS = ("assay_features", "pca_raw", "pca_normalized", "dino")
INPUT_LABELS = {
    "assay_features": "Assays",
    "pca_raw": "PCA raw",
    "pca_normalized": "PCA normalized",
    "dino": "DINO",
}
INPUT_COLORS = {
    "assay_features": "#777777",
    "pca_raw": "#2878B5",
    "pca_normalized": "#D9534F",
    "dino": "#2A9D65",
}
N_SPLITS = 5


def load_profiles(df: pd.DataFrame) -> dict[str, np.ndarray]:
    inputs = {
        "pca_raw": np.stack(df["pca_embedding_raw"].map(unpack_vector)),
        "pca_normalized": np.stack(df["pca_embedding_normalized"].map(unpack_vector)),
        "dino": np.stack(df["brightfield"].map(unpack_vector)),
        "assay_features": df[list(ASSAY_FEATURES)].to_numpy(dtype=np.float32),
    }
    for matrix in inputs.values():
        matrix[~np.isfinite(matrix)] = np.nan
    return inputs


def grouped_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  target: str, representation: str, analysis: str) -> dict:
    """Cross-validated nuisance classification with compound-grouped folds."""
    keep = pd.notna(y)
    X, y, groups = X[keep], y[keep].astype(str), groups[keep].astype(str)
    classes = np.unique(y)
    if len(classes) < 2:
        return {
            "analysis": analysis, "nuisance": target,
            "feature_set": representation, "n_samples": int(len(y)),
            "n_compounds": int(len(np.unique(groups))), "n_classes": int(len(classes)),
            "balanced_accuracy": np.nan, "macro_f1": np.nan, "accuracy": np.nan,
            "majority_baseline_accuracy": np.nan,
            "chance_balanced_accuracy": np.nan,
            "status": "only one observed class",
        }
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    prediction = np.empty(y.shape, dtype=object)
    baseline = np.empty(y.shape, dtype=object)
    for train, test in splitter.split(X, y, groups):
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_x = scaler.fit_transform(imputer.fit_transform(X[train]))
        test_x = scaler.transform(imputer.transform(X[test]))
        model = RidgeClassifier(alpha=10.0, class_weight="balanced")
        model.fit(train_x, y[train])
        prediction[test] = model.predict(test_x)
        majority = pd.Series(y[train]).value_counts().idxmax()
        baseline[test] = majority
    return {
        "analysis": analysis,
        "nuisance": target,
        "feature_set": representation,
        "n_samples": int(len(y)),
        "n_compounds": int(len(np.unique(groups))),
        "n_classes": int(len(classes)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=classes,
                                    average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
        "majority_baseline_accuracy": float(accuracy_score(y, baseline)),
        "chance_balanced_accuracy": 1.0 / len(classes),
        "status": "ok",
    }


def control_probe(X: np.ndarray, y: np.ndarray,
                  target: str, representation: str) -> dict:
    """Predict control-well technical labels with stratified held-out wells."""
    keep = pd.notna(y)
    X, y = X[keep], y[keep].astype(str)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < N_SPLITS:
        return {
            "analysis": "DMSO_control_wells", "nuisance": target,
            "feature_set": representation, "n_samples": int(len(y)),
            "n_compounds": 0, "n_classes": int(len(classes)),
            "balanced_accuracy": np.nan, "macro_f1": np.nan, "accuracy": np.nan,
            "majority_baseline_accuracy": np.nan,
            "chance_balanced_accuracy": 1.0 / len(classes) if len(classes) else np.nan,
            "status": "insufficient DMSO class variation or counts",
        }
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    prediction = np.empty(y.shape, dtype=object)
    baseline = np.empty(y.shape, dtype=object)
    for train, test in splitter.split(X, y):
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_x = scaler.fit_transform(imputer.fit_transform(X[train]))
        test_x = scaler.transform(imputer.transform(X[test]))
        model = RidgeClassifier(alpha=10.0, class_weight="balanced")
        model.fit(train_x, y[train])
        prediction[test] = model.predict(test_x)
        baseline[test] = pd.Series(y[train]).value_counts().idxmax()
    return {
        "analysis": "DMSO_control_wells",
        "nuisance": target,
        "feature_set": representation,
        "n_samples": int(len(y)),
        "n_compounds": 0,
        "n_classes": int(len(classes)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=classes,
                                    average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
        "majority_baseline_accuracy": float(accuracy_score(y, baseline)),
        "chance_balanced_accuracy": 1.0 / len(classes),
        "status": "ok",
    }


def run_probes() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    inputs = load_profiles(df)
    treated = df[df["compound_id"].notna()].copy().reset_index()
    treated_ids = treated["compound_id"].astype(str).to_numpy()
    rows: list[dict] = []

    for feature in INPUTS:
        X = inputs[feature][treated["index"].to_numpy()]
        # Center each well around the compound's mean profile. Restrict each
        # nuisance probe to compounds observed in at least two levels.
        frame = pd.DataFrame(X)
        frame["compound_id"] = treated_ids
        means = frame.groupby("compound_id").mean(numeric_only=True)
        residual = X - np.stack([means.loc[c].to_numpy() for c in treated_ids])
        for nuisance in ("plate", "batch", "source"):
            per_compound_levels = treated.groupby(treated_ids)[nuisance].nunique()
            eligible = set(per_compound_levels[per_compound_levels >= 2].index.astype(str))
            use = np.array([c in eligible for c in treated_ids])
            result = grouped_probe(
                residual[use], treated.loc[use, nuisance].astype(str).to_numpy(),
                treated_ids[use], nuisance, feature, "within_compound_residuals",
            )
            result["n_compounds_with_multiple_levels"] = int(len(eligible))
            rows.append(result)

        dmso_rows = df["compound_id"].isna().to_numpy()
        for nuisance in ("plate", "batch", "source"):
            result = control_probe(
                inputs[feature][dmso_rows], df.loc[dmso_rows, nuisance].to_numpy(),
                nuisance, feature,
            )
            result["n_compounds_with_multiple_levels"] = 0
            rows.append(result)

    results = pd.DataFrame(rows)
    return results


def join_biology_scores(probes: pd.DataFrame) -> pd.DataFrame:
    biological = []
    comparison_files = {
        "held_out_compound": RESULTS_DIR / "annotation_prediction" / "annotation_model_comparison_held_out_compound.csv",
        "held_out_batch": RESULTS_DIR / "annotation_prediction" / "annotation_model_comparison_held_out_batch.csv",
        "held_out_chemical_group": RESULTS_DIR / "annotation_prediction" / "annotation_model_comparison_chemical_split.csv",
    }
    for protocol, path in comparison_files.items():
        if not path.exists():
            continue
        part = pd.read_csv(path)
        part = part[part.feature_set.isin(INPUTS)].copy()
        part["protocol"] = protocol
        biological.append(part)
    if biological:
        bio = pd.concat(biological, ignore_index=True)
        bio = bio.pivot_table(
            index=["protocol", "feature_set"], columns="task",
            values=["macro_f1_top3", "micro_f1_top3"], aggfunc="first",
        )
        bio.columns = [f"{metric}_{task}" for metric, task in bio.columns]
        bio = bio.reset_index()
    else:
        bio = pd.DataFrame(columns=["protocol", "feature_set"])

    probe_wide = probes.pivot_table(
        index="feature_set", columns=["analysis", "nuisance"],
        values="balanced_accuracy", aggfunc="first",
    )
    probe_wide.columns = [f"{analysis}_{target}_balanced_accuracy"
                          for analysis, target in probe_wide.columns]
    probe_wide = probe_wide.reset_index()
    return bio.merge(probe_wide, on="feature_set", how="left")


def plot_probes(probes: pd.DataFrame) -> Path:
    panels = [
        ("within_compound_residuals", "plate", "Treated wells: plate"),
        ("within_compound_residuals", "batch", "Treated wells: batch/source"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, (analysis, nuisance, title) in zip(axes, panels):
        subset = probes[(probes.analysis == analysis) & (probes.nuisance == nuisance)]
        subset = subset.set_index("feature_set").reindex(INPUTS)
        bars = ax.bar(
            [INPUT_LABELS[x] for x in INPUTS], subset.balanced_accuracy,
            color=[INPUT_COLORS[x] for x in INPUTS],
        )
        if not subset.empty:
            chance = float(subset.chance_balanced_accuracy.dropna().iloc[0])
            ax.axhline(chance, color="#444444", linestyle="--", linewidth=1.2,
                       label=f"Chance ({chance:.2f})")
        for bar, val in zip(bars, subset.balanced_accuracy):
            if pd.notna(val):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=13)
        ax.set_ylim(0, 1.08)
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    axes[0].set_ylabel("Balanced accuracy", fontsize=11)
    fig.suptitle("Technical Information Recoverable from Each Representation",
                 fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.0)
    output = RESULTS_DIR / "confounding" / "confounding_representation_probes.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    probes = run_probes()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    probes_path = RESULTS_DIR / "confounding" / "confounding_probe_metrics.csv"
    probes.to_csv(probes_path, index=False)
    comparison = join_biology_scores(probes)
    comparison_path = RESULTS_DIR / "confounding" / "confounding_and_biology_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    data = pd.read_parquet(DATA_PATH)
    source_batch = pd.crosstab(data["source"], data["batch"])
    perfect_alias = bool((source_batch.to_numpy() > 0).sum() == len(source_batch))
    print(probes.to_string(index=False))
    print(f"\nSource and batch perfectly paired: {perfect_alias}")
    print(f"Saved nuisance probes to {probes_path}")
    print(f"Saved nuisance/biology comparison to {comparison_path}")
    print(f"Saved plot to {plot_probes(probes)}")


if __name__ == "__main__":
    main()
