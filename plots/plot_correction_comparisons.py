"""Plot prediction and retrieval comparisons across corrected feature inputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METHODS = [
    ("plate_corrected", "Plate"),
    ("batch_corrected", "Batch"),
    ("plate_batch_corrected", "Plate + batch"),
]
BASE_LABELS = {
    "assay_features": "Assay features",
    "pca_raw": "PCA raw",
    "pca_normalized": "PCA normalized",
    "dino": "DINO",
}
FEATURE_ORDER = ["assay_features"]
for _feature in ("pca_raw", "pca_normalized", "dino"):
    FEATURE_ORDER.append(_feature)
    FEATURE_ORDER.extend(f"{_feature}_{name}" for name, _ in METHODS)
FEATURE_ORDER.extend(f"assay_features_{name}" for name, _ in METHODS)


def feature_label(name: str) -> str:
    for suffix, label in sorted(METHODS, key=lambda item: len(item[0]), reverse=True):
        ending = f"_{suffix}"
        if name.endswith(ending):
            return f"{BASE_LABELS[name[:-len(ending)]].replace(' (uncorrected)', '')} | {label}"
    return BASE_LABELS.get(name, name)


def plot_annotation_retrieval() -> Path:
    path = RESULTS / "annotation_retrieval" / "embedding_retrieval_metrics.csv"
    data = pd.read_csv(path)
    protocols = ["held_out_compound", "held_out_batch", "held_out_chemical_group"]
    protocol_labels = ["Held-out compound", "Held-out batch", "Held-out chemical group"]
    fig, axes = plt.subplots(2, 3, figsize=(19, 18), sharex=True, sharey=True)
    y = np.arange(len(FEATURE_ORDER))
    for row, task in enumerate(("pathway", "target")):
        for col, (protocol, protocol_label) in enumerate(zip(protocols, protocol_labels)):
            ax = axes[row, col]
            subset = data[(data.protocol == protocol) & (data.task == task) & (data.k == 5)]
            values = subset.set_index("embedding")["map_at_k"]
            plotted = np.array([values.get(feature, np.nan) for feature in FEATURE_ORDER])
            ax.scatter(plotted, y, s=35, color="#3577a8", zorder=3)
            for pos, val in enumerate(plotted):
                if np.isfinite(val):
                    ax.plot([0, val], [pos, pos], color="#b7c5cc", lw=1.2, zorder=1)
            ax.set_title(f"{protocol_label} | {task.title()}", fontsize=13)
            ax.set_xlim(0, 0.55)
            ax.grid(axis="x", alpha=0.25)
            ax.tick_params(axis="both", labelsize=9)
            if col == 0:
                ax.set_yticks(y)
                ax.set_yticklabels([feature_label(f) for f in FEATURE_ORDER])
            else:
                ax.tick_params(labelleft=False)
            if row == 1:
                ax.set_xlabel("Mean average precision@5", fontsize=11)
    fig.suptitle("Pathway and Target Retrieval Across Corrected Inputs", y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98), w_pad=1.4, h_pad=1.5)
    output = RESULTS / "feature_processing" / "corrected_annotation_retrieval_comparison.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_activity_text_retrieval() -> Path:
    path = RESULTS / "activity_text_retrieval" / "biological_activity_retrieval_metrics.csv"
    data = pd.read_csv(path)
    protocols = ["held_out_compound", "held_out_batch", "held_out_chemical_group"]
    protocol_labels = ["Held-out compound", "Held-out batch", "Held-out chemical group"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 14), sharex=True, sharey=True)
    y = np.arange(len(FEATURE_ORDER))
    for col, (protocol, protocol_label) in enumerate(zip(protocols, protocol_labels)):
        ax = axes[col]
        subset = data[(data.protocol == protocol) & (data.k == 5)]
        values = subset.set_index("feature_set")["mean_activity_text_similarity_at_k"]
        plotted = np.array([values.get(feature, np.nan) for feature in FEATURE_ORDER])
        random_value = float(subset["random_gallery_mean_activity_text_similarity"].mean())
        ax.axvline(random_value, color="#777777", linestyle="--", lw=1.5,
                   label="Random Label Frequency")
        ax.scatter(plotted, y, s=35, color="#3577a8", zorder=3)
        for pos, val in enumerate(plotted):
            if np.isfinite(val):
                ax.plot([random_value, val], [pos, pos], color="#b7c5cc", lw=1.2, zorder=1)
        ax.set_title(protocol_label, fontsize=13)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="both", labelsize=9)
        if col == 0:
            ax.set_yticks(y)
            ax.set_yticklabels([feature_label(f) for f in FEATURE_ORDER])
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel("Mean activity-text TF-IDF similarity@5", fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", frameon=False)
    fig.suptitle("Biological Activity Text Retrieval Across Corrected Inputs", y=0.98, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.4)
    output = RESULTS / "activity_text_retrieval" / "corrected_activity_text_retrieval_comparison.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(f"Saved {plot_annotation_retrieval()}")
    print(f"Saved {plot_activity_text_retrieval()}")
