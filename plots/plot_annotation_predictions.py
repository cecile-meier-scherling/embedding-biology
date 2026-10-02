"""Plot prediction performance and bootstrap confidence intervals.

Run after the held-out analysis scripts have produced their result CSVs:
    uv run python plots/plot_annotation_predictions.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from report_plot_style import enlarge_report_text
except ImportError:
    from plots.report_plot_style import enlarge_report_text


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROTOCOLS = {
    "held_out_compound": {
        "metrics": RESULTS / "annotation_prediction" / "annotation_model_comparison_held_out_compound.csv",
        "stability": RESULTS / "annotation_prediction" / "annotation_prediction_stability_held_out_compound.csv",
        "label": "Held-out compound",
        "color": "#3569a8",
    },
    "held_out_batch": {
        "metrics": RESULTS / "annotation_prediction" / "annotation_model_comparison_held_out_batch.csv",
        "stability": RESULTS / "annotation_prediction" / "annotation_prediction_stability_held_out_batch.csv",
        "label": "Held-out batch",
        "color": "#e58635",
    },
    "held_out_chemical_group": {
        "metrics": RESULTS / "annotation_prediction" / "annotation_model_comparison_chemical_split.csv",
        "stability": RESULTS / "annotation_prediction" / "annotation_prediction_stability_held_out_chemical_group.csv",
        "label": "Held-out chemical group",
        "color": "#4b8b64",
    },
}
CORRECTION_VARIANTS = [
    ("plate_corrected", "plate corrected"),
    ("batch_corrected", "batch corrected"),
    ("plate_batch_corrected", "plate + batch corrected"),
]
BASE_FEATURES = ["assay_features", "pca_raw", "pca_normalized", "dino"]
FEATURE_ORDER = ["label_frequency_baseline", "assay_features"]
FEATURE_ORDER.extend(
    f"assay_features_{variant}" for variant, _ in CORRECTION_VARIANTS
)
for _feature in ("pca_raw", "pca_normalized", "dino"):
    FEATURE_ORDER.append(_feature)
    FEATURE_ORDER.extend(
        f"{_feature}_{variant}" for variant, _ in CORRECTION_VARIANTS
    )
FEATURE_LABELS = {
    "label_frequency_baseline": "Label frequency",
    "assay_features": "Assay features",
    "pca_raw": "PCA raw (uncorrected)",
    "pca_normalized": "PCA normalized (uncorrected)",
    "dino": "DINO (uncorrected)",
}
for _variant, _variant_label in CORRECTION_VARIANTS:
    FEATURE_LABELS[f"assay_features_{_variant}"] = f"Assay features | {_variant_label}"
for _feature in ("pca_raw", "pca_normalized", "dino"):
    _base_label = {"pca_raw": "PCA raw", "pca_normalized": "PCA normalized",
                   "dino": "DINO"}[_feature]
    for _variant, _variant_label in CORRECTION_VARIANTS:
        FEATURE_LABELS[f"{_feature}_{_variant}"] = f"{_base_label} | {_variant_label}"
METRICS = [
    ("macro_f1_top3", "Macro F1@3"),
    ("micro_f1_top3", "Micro F1@3"),
]
FEATURE_MARKERS = {
    "label_frequency_baseline": "X",
    "assay_features": "s",
    "pca_raw": "o",
    "pca_normalized": "D",
    "dino": "^",
}
FEATURE_MARKER_BY_INPUT = {
    feature: FEATURE_MARKERS.get(
        feature.split("_plate_corrected")[0].split("_batch_corrected")[0]
                .split("_plate_batch_corrected")[0], "o"
    )
    for feature in FEATURE_ORDER
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [p for spec in PROTOCOLS.values() for p in (spec["metrics"], spec["stability"]) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing analysis result files:\n" + "\n".join(map(str, missing)))

    metric_frames, stability_frames = [], []
    for protocol, spec in PROTOCOLS.items():
        metric = pd.read_csv(spec["metrics"])
        stability = pd.read_csv(spec["stability"])
        metric["protocol"] = protocol
        stability["protocol"] = protocol
        metric_frames.append(metric)
        stability_frames.append(stability)

    metrics = pd.concat(metric_frames, ignore_index=True)
    stability = pd.concat(stability_frames, ignore_index=True)
    stability = stability[stability["record_type"] == "summary"].copy()
    metrics = metrics[metrics["feature_set"].isin(FEATURE_ORDER)].copy()
    stability = stability[stability["feature_set"].isin(FEATURE_ORDER)].copy()
    return metrics, stability


def plot_metrics() -> Path:
    metrics, stability = load_data()
    merged = metrics.merge(
        stability,
        on=["protocol", "task", "feature_set"],
        suffixes=("", "_stability"),
        validate="one_to_one",
    )

    # Keep the requested order while tolerating a missing input in older outputs.
    features = [f for f in FEATURE_ORDER if f in set(merged["feature_set"])]
    # Rows separate Macro and Micro, whose scales and interpretation differ.
    # Each row shares one x-axis range across pathway and target.
    fig, axes = plt.subplots(2, 2, figsize=(16, 18), sharey=True)
    panels = [
        ("macro_f1_top3", "pathway", axes[0, 0]),
        ("macro_f1_top3", "target", axes[0, 1]),
        ("micro_f1_top3", "pathway", axes[1, 0]),
        ("micro_f1_top3", "target", axes[1, 1]),
    ]
    offsets = np.linspace(-0.2, 0.2, len(PROTOCOLS))

    for metric_name, task, ax in panels:
        subset = merged[merged["task"] == task]
        for feature_idx, feature in enumerate(features):
            for protocol_idx, (protocol, spec) in enumerate(PROTOCOLS.items()):
                row = subset[(subset["protocol"] == protocol) & (subset["feature_set"] == feature)]
                if row.empty:
                    continue
                row = row.iloc[0]
                point = row[metric_name]
                lower = row[f"{metric_name}_bootstrap_ci_lower"]
                upper = row[f"{metric_name}_bootstrap_ci_upper"]
                if not np.isfinite(point) or not np.isfinite(lower) or not np.isfinite(upper):
                    continue
                ax.errorbar(
                    point, feature_idx + offsets[protocol_idx],
                    xerr=[[point - lower], [upper - point]],
                    fmt=FEATURE_MARKER_BY_INPUT[feature], markersize=6, capsize=2.5, elinewidth=1.4,
                    color=spec["color"], markeredgecolor="white", markeredgewidth=0.5,
                )

        ax.set_title(task.title(), fontsize=15)
        ax.set_xlabel("F1 score with 95% bootstrap CI", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels([FEATURE_LABELS[f] for f in features])

    # Apply the same x range to pathway and target within each metric row.
    for row_idx, metric_name in enumerate(("macro_f1_top3", "micro_f1_top3")):
        row_values = merged[metric_name].dropna()
        row_ci = pd.concat([
            merged[f"{metric_name}_bootstrap_ci_lower"],
            merged[f"{metric_name}_bootstrap_ci_upper"],
        ]).dropna()
        max_x = max(float(row_values.max()), float(row_ci.max())) if not row_values.empty and not row_ci.empty else 1.0
        for ax in axes[row_idx, :]:
            ax.set_xlim(0, max_x * 1.08)
        axes[row_idx, 0].set_ylabel(dict(METRICS)[metric_name], fontsize=13)

    from matplotlib.lines import Line2D
    feature_handles = [
        Line2D([0], [0], marker=FEATURE_MARKER_BY_INPUT[f], linestyle="none", color="#555555",
               markersize=7, label=FEATURE_LABELS[f])
        for f in ["label_frequency_baseline", "assay_features", "pca_raw",
                  "pca_normalized", "dino"] if f in features
    ]
    protocol_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=spec["color"],
               markersize=6, label=spec["label"])
        for spec in PROTOCOLS.values()
    ]
    fig.legend(handles=feature_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=len(features), frameon=False, title="Input features",
               fontsize=11, title_fontsize=12)
    fig.legend(handles=protocol_handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=len(PROTOCOLS), frameon=False, title="Validation split",
               fontsize=11, title_fontsize=12)
    fig.suptitle("Pathway and Target Prediction Across Input Features", y=1.055, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.0, w_pad=2.0)
    enlarge_report_text(fig)
    output = RESULTS / "annotation_prediction" / "annotation_prediction_metrics.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(f"Saved {plot_metrics()}")
