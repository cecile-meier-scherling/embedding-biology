"""Plot normalized PCA, assay, and tuned weighted-fusion results.

Run after analysis/evaluate_weighted_fusion.py:
    uv run python plots/plot_weighted_fusion.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS_PATH = RESULTS / "feature_processing" / "weighted_fusion_fold_metrics.csv"
PROTOCOLS = {
    "held_out_compound": "Held-out compound",
    "held_out_batch": "Held-out batch",
    "held_out_chemical_group": "Held-out chemical group",
}
FEATURES = {
    "pca": ("Normalized PCA", "#2878B5"),
    "assay": ("Assay features", "#E68632"),
    "fusion": ("Weighted fusion", "#2A9D65"),
}
METRICS = [("macro_f1_top3", "Macro F1@3"), ("micro_f1_top3", "Micro F1@3")]


def bootstrap_ci(values: np.ndarray, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def plot() -> Path:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {METRICS_PATH}; run analysis/evaluate_weighted_fusion.py first"
        )
    df = pd.read_csv(METRICS_PATH)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="col")
    panels = [
        ("macro_f1_top3", "pathway", axes[0, 0]),
        ("macro_f1_top3", "target", axes[0, 1]),
        ("micro_f1_top3", "pathway", axes[1, 0]),
        ("micro_f1_top3", "target", axes[1, 1]),
    ]
    protocol_keys = list(PROTOCOLS)
    xbase = np.arange(len(protocol_keys))
    feature_order = ["pca", "assay", "fusion"]
    offsets = dict(zip(feature_order, (-.19, 0, .19)))
    value_columns = {
        "pca": "pca_{metric}",
        "assay": "assay_{metric}",
        "fusion": "fusion_{metric}",
    }

    for metric, task, ax in panels:
        for protocol_index, protocol in enumerate(protocol_keys):
            for feature in feature_order:
                col = value_columns[feature].format(metric=metric)
                values = df[(df.protocol == protocol) & (df.task == task)][col].dropna().to_numpy()
                if not len(values):
                    continue
                x = xbase[protocol_index] + offsets[feature]
                color = FEATURES[feature][1]
                # Individual folds show how sensitive each result is to the split.
                jitter = np.linspace(-.025, .025, len(values)) if len(values) > 1 else np.array([0.])
                ax.scatter(x + jitter, values, s=30, alpha=.45, color=color,
                           edgecolor="white", linewidth=.35, zorder=2)
                mean = float(values.mean())
                low, high = bootstrap_ci(values, seed=100 + protocol_index)
                ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]],
                            fmt="D", markersize=7, capsize=3, elinewidth=1.6,
                            color=color, markeredgecolor="white", markeredgewidth=.7,
                            zorder=4)
        ax.set_title(task.title(), fontsize=14, pad=8)
        ax.set_ylabel("F1 score at top 3", fontsize=12)
        ax.set_xticks(xbase)
        ax.set_xticklabels([PROTOCOLS[p] for p in protocol_keys], rotation=15, ha="right")
        ax.tick_params(axis="both", labelsize=10.5)
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)

    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
    for row, (metric, metric_label) in enumerate(METRICS):
        vals = np.concatenate([
            df[f"{feature}_{metric}"].dropna().to_numpy()
            for feature in ("pca", "assay", "fusion")
        ])
        axes[row, 0].set_ylim(0, min(1.0, float(vals.max()) * 1.18))
        axes[row, 0].annotate(metric_label, xy=(-.20, .5), xycoords="axes fraction",
                             rotation=90, va="center", ha="center", fontsize=12,
                             fontweight="bold")

    feature_handles = [
        Line2D([0], [0], marker="D", linestyle="none", color=color,
               markersize=7, label=label)
        for label, color in FEATURES.values()
    ]
    fig.legend(handles=feature_handles, loc="upper center", ncol=3,
               bbox_to_anchor=(.5, .995), frameon=False, fontsize=11)
    fig.suptitle("Normalized PCA, Assay, and Weighted Fusion Performance",
                 y=1.04, fontsize=16)
    fig.text(.5, .005,
             "Dots show fold scores; diamonds and bars show the mean and 95% fold-bootstrap interval.",
             ha="center", fontsize=10.5)
    fig.tight_layout(rect=(.035, .04, 1, .95), h_pad=1.5, w_pad=1.4)
    out = RESULTS / "feature_processing" / "weighted_fusion_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(f"Saved {plot()}")
