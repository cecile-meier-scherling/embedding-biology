"""Create normalized-PCA figures and cross-input retrieval comparison.

Run after the annotation and retrieval analyses:
    uv run python plots/plot_normalized_pca_results.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
try:
    from report_plot_style import enlarge_report_text
except ImportError:
    from plots.report_plot_style import enlarge_report_text


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROTOCOL_FILES = {
    "Held-out compound": RESULTS / "annotation_prediction" / "annotation_model_comparison_held_out_compound.csv",
    "Held-out batch": RESULTS / "annotation_prediction" / "annotation_model_comparison_held_out_batch.csv",
    "Held-out chemical group": RESULTS / "annotation_prediction" / "annotation_model_comparison_chemical_split.csv",
}
PROTOCOL_KEYS = {
    "Held-out compound": "held_out_compound",
    "Held-out batch": "held_out_batch",
    "Held-out chemical group": "held_out_chemical_group",
}
def plot_retrieval() -> Path:
    path = RESULTS / "annotation_retrieval" / "embedding_retrieval_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run analysis/evaluate_embedding_retrieval.py first.")
    data = pd.read_csv(path)
    data = data[data.embedding == "pca_normalized"].copy()
    split_order = ["held_out_compound", "held_out_batch", "held_out_chemical_group"]
    split_labels = {
        "held_out_compound": "Held-out compound",
        "held_out_batch": "Held-out batch",
        "held_out_chemical_group": "Held-out chemical group",
    }
    metrics = [
        ("neighbor_precision_at_k", "Neighbor precision@k", "#3b6fb6", "o", "-"),
        ("annotation_recall_at_k", "Annotation recall@k", "#d47745", "s", "-"),
        ("map_at_k", "MAP@k", "#4a8a69", "^", "--"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for col, protocol in enumerate(split_order):
        for row, task in enumerate(["pathway", "target"]):
            ax = axes[row, col]
            subset = data[(data.protocol == protocol) & (data.task == task)]
            if subset.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue
            for metric, label, color, marker, linestyle in metrics:
                curve = subset.sort_values("k")
                ax.plot(curve.k, curve[metric], color=color, marker=marker,
                        linestyle=linestyle, linewidth=2, markersize=5, label=label)
            ax.set_title(f"{split_labels[protocol]}\n{task.title()}")
            ax.set_xticks(sorted(subset.k.unique()))
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.23)
            if col == 0:
                ax.set_ylabel("Score")
            if row == 1:
                ax.set_xlabel("Number of retrieved neighbors (k)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Normalized PCA Annotation Retrieval", y=1.08, fontsize=15)
    fig.tight_layout()
    enlarge_report_text(fig)
    output = RESULTS / "annotation_retrieval" / "normalized_pca_retrieval.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_annotation_retrieval_across_inputs() -> Path:
    """Compare annotation MAP retrieval across all measured feature inputs."""
    path = RESULTS / "annotation_retrieval" / "embedding_retrieval_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run analysis/evaluate_embedding_retrieval.py first.")
    data = pd.read_csv(path)
    protocols = [
        ("held_out_compound", "Held-out compound"),
        ("held_out_batch", "Held-out batch"),
        ("held_out_chemical_group", "Held-out chemical group"),
    ]
    tasks = ["pathway", "target"]
    inputs = [
        ("assay_features", "Assay features", "#777777"),
        ("pca_raw", "PCA raw", "#2878B5"),
        ("pca_normalized", "PCA normalized", "#D9534F"),
        ("dino", "DINO", "#2A9D65"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    for row, task in enumerate(tasks):
        for col, (protocol, protocol_label) in enumerate(protocols):
            ax = axes[row, col]
            subset = data[(data.protocol == protocol) & (data.task == task)]
            for feature, label, color in inputs:
                curve = subset[subset.embedding == feature].sort_values("k")
                if curve.empty:
                    continue
                ax.plot(curve.k, curve.map_at_k, marker="o", color=color,
                        linewidth=2, markersize=5, label=label)
            ax.set_title(f"{protocol_label}\n{task.title()}", fontsize=14)
            ax.set_xticks([1, 3, 5, 10])
            ax.set_ylim(0, 1.02)
            ax.tick_params(axis="both", labelsize=12)
            ax.grid(alpha=0.23)
            if col == 0:
                ax.set_ylabel("Mean average precision@k", fontsize=13)
            if row == 1:
                ax.set_xlabel("Retrieved neighbors (k)", fontsize=13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.99), fontsize=11)
    fig.suptitle("Pathway and Target Annotation Retrieval Across Input Features",
                 y=1.045, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.8)
    enlarge_report_text(fig)
    output = RESULTS / "annotation_retrieval" / "annotation_retrieval_across_features.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_activity_text_retrieval() -> Path:
    """Compare how well each input retrieves compounds with similar activity text."""
    path = RESULTS / "activity_text_retrieval" / "biological_activity_retrieval_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run analysis/evaluate_activity_text_retrieval.py first."
        )
    data = pd.read_csv(path)
    protocol_order = [
        "held_out_compound", "held_out_batch", "held_out_chemical_group"
    ]
    protocol_labels = {
        "held_out_compound": "Held-out compound",
        "held_out_batch": "Held-out batch",
        "held_out_chemical_group": "Held-out chemical group",
    }
    feature_order = ["assay_features", "pca_raw", "pca_normalized", "dino"]
    feature_labels = {
        "assay_features": "Assay features",
        "pca_raw": "PCA raw",
        "pca_normalized": "PCA normalized",
        "dino": "DINO",
    }
    colors = {
        "assay_features": "#777777",
        "pca_raw": "#2878B5",
        "pca_normalized": "#D9534F",
        "dino": "#2A9D65",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    all_values = data["mean_activity_text_similarity_at_k"].to_numpy(dtype=float)
    random_values = data[
        "random_gallery_mean_activity_text_similarity"
    ].to_numpy(dtype=float)
    ymin = max(0, min(float(np.nanmin(all_values)), float(np.nanmin(random_values))) - 0.002)
    ymax = max(float(np.nanmax(all_values)), float(np.nanmax(random_values))) + 0.002

    for ax, protocol in zip(axes, protocol_order):
        subset = data[data.protocol == protocol]
        if subset.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        baseline = float(subset["random_gallery_mean_activity_text_similarity"].mean())
        ax.axhline(baseline, color="#555555", linestyle="--", linewidth=1.3,
                   label="Random Label Frequency")
        for feature in feature_order:
            curve = subset[subset.feature_set == feature].sort_values("k")
            if curve.empty:
                continue
            ax.plot(
                curve["k"], curve["mean_activity_text_similarity_at_k"],
                color=colors[feature], marker="o", linewidth=2, markersize=5,
                label=feature_labels[feature],
            )
        ax.set_title(protocol_labels[protocol], fontsize=14)
        ax.set_xticks([1, 3, 5, 10])
        ax.set_xlabel("Retrieved neighbors (k)", fontsize=13)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean TF-IDF similarity to query activity text\n(among top-k retrieved compounds)", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=5, frameon=False, fontsize=11)
    fig.suptitle("Biological Activity Text Retrieval Across Input Features", y=1.075, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    enlarge_report_text(fig)
    output = RESULTS / "activity_text_retrieval" / "biological_activity_retrieval.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_normalized_pca_activity_retrieval() -> Path:
    """Plot activity-text retrieval for normalized PCA alone."""
    path = RESULTS / "activity_text_retrieval" / "biological_activity_retrieval_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run analysis/evaluate_activity_text_retrieval.py first."
        )
    data = pd.read_csv(path)
    data = data[data.feature_set == "pca_normalized"]
    protocols = [
        ("held_out_compound", "Held-out compound"),
        ("held_out_batch", "Held-out batch"),
        ("held_out_chemical_group", "Held-out chemical group"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    values = data["mean_activity_text_similarity_at_k"].to_numpy(dtype=float)
    baseline_values = data[
        "random_gallery_mean_activity_text_similarity"
    ].to_numpy(dtype=float)
    ymin = max(0, min(float(np.nanmin(values)), float(np.nanmin(baseline_values))) - 0.002)
    ymax = max(float(np.nanmax(values)), float(np.nanmax(baseline_values))) + 0.002
    for ax, (protocol, label) in zip(axes, protocols):
        subset = data[data.protocol == protocol].sort_values("k")
        if subset.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        ax.plot(subset.k, subset.mean_activity_text_similarity_at_k,
                color="#D9534F", marker="o", linewidth=2.2, markersize=6,
                label="PCA normalized")
        ax.axhline(subset.random_gallery_mean_activity_text_similarity.mean(),
                   color="#555555", linestyle="--", linewidth=1.4,
                   label="Random Label Frequency")
        ax.set_title(label, fontsize=14)
        ax.set_xticks([1, 3, 5, 10])
        ax.set_xlabel("Retrieved neighbors (k)", fontsize=13)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.23)
    axes[0].set_ylabel(
        "Mean TF-IDF similarity to query activity text\n(among top-k retrieved compounds)",
        fontsize=13,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.99), fontsize=11)
    fig.suptitle("Normalized PCA Biological Activity Text Retrieval",
                 y=1.055, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output = RESULTS / "activity_text_retrieval" / "normalized_pca_activity_retrieval.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_cluster_enrichment() -> Path:
    paths = [
        RESULTS / "cluster_enrichment" / "cluster_label_enrichment_held_out_compound.csv",
        RESULTS / "cluster_enrichment" / "cluster_label_enrichment_held_out_batch.csv",
    ]
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError("No held-out cluster-enrichment CSVs found in results/.")
    data = pd.concat([pd.read_csv(path) for path in existing], ignore_index=True)
    data = data[data.embedding == "pca_normalized"].copy()
    # These two protocols pool the same compounds into cluster enrichment in
    # the current outputs; collapse duplicate values rather than double-plot.
    identity = ["task", "cluster", "label", "n_cluster_compounds", "n_with_label",
                "fold_enrichment", "q_value"]
    data = data.drop_duplicates(identity)
    data = data[(data.q_value < 0.05) & (data.fold_enrichment > 1)
                & (data.n_cluster_compounds >= 20)].copy()
    if data.empty:
        raise ValueError("No enriched labels with q<0.05 and at least 20 compounds in a cluster.")

    # Keep the strongest adjusted-significance findings, ordered within task.
    data = data.sort_values("q_value")
    fig, axes = plt.subplots(1, 2, figsize=(17, 9.2))
    scatter = None
    for ax, task in zip(axes, ["pathway", "target"]):
        subset = data[data.task == task].sort_values("q_value").copy()
        subset["row_label"] = subset.apply(
            lambda r: f"{r['label']}\nC{int(r['cluster'])} · n={int(r['n_cluster_compounds'])}",
            axis=1,
        )
        subset = subset.sort_values("fold_enrichment")
        y = np.arange(len(subset))
        x = np.log2(subset.fold_enrichment.to_numpy())
        sizes = 34 + 0.14 * np.sqrt(subset.n_cluster_compounds.to_numpy()) * 34
        scatter = ax.scatter(
            x, y, s=sizes, c=-np.log10(subset.q_value.clip(lower=1e-300)),
            cmap="viridis", edgecolors="#303030", linewidths=0.55,
        )
        ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
        ax.set_yticks(y, subset.row_label, fontsize=10)
        ax.set_xlabel("log₂ fold enrichment (0 = no enrichment)", fontsize=13)
        ax.set_title(task.title(), fontsize=14)
        ax.grid(axis="x", alpha=0.23)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Enriched label and cluster (cluster size)", fontsize=13)
    colorbar_ax = fig.add_axes([0.925, 0.24, 0.018, 0.50])
    fig.colorbar(scatter, cax=colorbar_ax, label="−log₁₀ adjusted q-value")
    colorbar_ax.tick_params(labelsize=11)
    colorbar_ax.yaxis.label.set_size(12)
    fig.suptitle("Normalized PCA Cluster Enrichment", y=0.985, fontsize=15)
    fig.subplots_adjust(left=0.27, right=0.90, bottom=0.12, top=0.90, wspace=1.00)
    enlarge_report_text(fig)
    output = RESULTS / "cluster_enrichment" / "normalized_pca_cluster_enrichment.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_prediction_stability() -> Path:
    stability_path = RESULTS / "annotation_prediction" / "normalized_pca_prediction_stability.csv"
    if not stability_path.exists():
        raise FileNotFoundError(
            f"Missing {stability_path}; run analysis/evaluate_normalized_pca_stability.py first."
        )
    stability = pd.read_csv(stability_path)
    summaries = stability[
        (stability.record_type == "summary")
        & (stability.feature_set == "pca_normalized")
    ]
    folds = stability[stability.record_type == "fold"]
    metric_frames = []
    for protocol_label, path in PROTOCOL_FILES.items():
        metrics = pd.read_csv(path)
        metrics = metrics[metrics.feature_set == "pca_normalized"].copy()
        metrics["protocol"] = PROTOCOL_KEYS[protocol_label]
        metric_frames.append(metrics)
    pooled_metrics = pd.concat(metric_frames, ignore_index=True)
    protocols = [
        ("held_out_compound", "Held-out\ncompound"),
        ("held_out_batch", "Held-out\nbatch"),
        ("held_out_chemical_group", "Held-out chemical\ngroup"),
    ]
    task_colors = {
        "pathway": "#3b6fb6",
        "target": "#d47745",
    }
    metrics = [
        ("macro_f1_top3", "Macro F1@3"),
        ("micro_f1_top3", "Micro F1@3"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    task_offsets = {"pathway": -0.13, "target": 0.13}
    for ax, (metric, metric_title) in zip(axes, metrics):
        for x, (protocol, _) in enumerate(protocols):
            for task in ["pathway", "target"]:
                summary = summaries[(summaries.protocol == protocol)
                                     & (summaries.task == task)]
                if summary.empty:
                    continue
                summary = summary.iloc[0]
                pooled_row = pooled_metrics[(pooled_metrics.protocol == protocol)
                                            & (pooled_metrics.task == task)].iloc[0]
                lower = summary[f"{metric}_bootstrap_ci_lower"]
                upper = summary[f"{metric}_bootstrap_ci_upper"]
                pooled = pooled_row[metric]
                # Show each held-out fold or batch score as a faint point.
                fold_values = folds[(folds.protocol == protocol)
                                    & (folds.task == task)][metric].to_numpy()
                center = x + task_offsets[task]
                jitter = np.linspace(-0.045, 0.045, len(fold_values)) if len(fold_values) else []
                if len(fold_values):
                    ax.scatter(center + jitter, fold_values, s=28, color="#777777",
                               alpha=0.55, zorder=2)
                ax.errorbar(
                    center, pooled,
                    yerr=[[pooled - lower], [upper - pooled]],
                    fmt="o", color=task_colors[task], ecolor=task_colors[task],
                    markersize=8, capsize=4, linewidth=1.8, zorder=3,
                )
        ax.set_title(metric_title, fontsize=14)
        ax.set_xticks(np.arange(len(protocols)), [label for _, label in protocols])
        ax.set_ylabel("F1 score", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", alpha=0.23)
        ax.set_axisbelow(True)
    highest_ci = summaries[[
        "macro_f1_top3_bootstrap_ci_upper", "micro_f1_top3_bootstrap_ci_upper"
    ]].to_numpy().max()
    shared_y_max = min(1.0, np.ceil(highest_ci / 0.05) * 0.05 + 0.05)
    for ax in axes:
        ax.set_ylim(0, shared_y_max)
    legend_handles = [
        Line2D([0], [0], marker="o", color=task_colors["pathway"],
               linestyle="none", markersize=8,
               label="Pathway pooled score (95% bootstrap CI)"),
        Line2D([0], [0], marker="o", color=task_colors["target"],
               linestyle="none", markersize=8,
               label="Target pooled score (95% bootstrap CI)"),
        Line2D([0], [0], marker="o", color="#777777", alpha=0.65,
               linestyle="none", markersize=6,
               label="Fold/batch scores (gray points)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.91), fontsize=11)
    fig.suptitle("Normalized PCA Target and Pathway Prediction Performance",
                 y=0.985, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    enlarge_report_text(fig)
    output = RESULTS / "annotation_prediction" / "normalized_pca_prediction_stability.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(f"Saved {plot_prediction_stability()}")
    print(f"Saved {plot_retrieval()}")
    print(f"Saved {plot_annotation_retrieval_across_inputs()}")
    print(f"Saved {plot_activity_text_retrieval()}")
    print(f"Saved {plot_normalized_pca_activity_retrieval()}")
    print(f"Saved {plot_cluster_enrichment()}")
