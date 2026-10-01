"""Plot pathway/target enrichment for clusters of image embeddings.

Run after compound- and/or batch-held-out analyses:
    uv run python analysis/plot_cluster_enrichment.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ENRICHMENT_FILES = {
    "held_out_compound": RESULTS / "cluster_enrichment" / "cluster_label_enrichment_held_out_compound.csv",
    "held_out_batch": RESULTS / "cluster_enrichment" / "cluster_label_enrichment_held_out_batch.csv",
}
LEGACY_FILE = RESULTS / "cluster_enrichment" / "cluster_label_enrichment.csv"


def load_enrichment() -> pd.DataFrame:
    frames = []
    for protocol, path in ENRICHMENT_FILES.items():
        if path.exists():
            frame = pd.read_csv(path)
            frame["protocol"] = protocol
            frames.append(frame)
    if not frames and LEGACY_FILE.exists():
        frames.append(pd.read_csv(LEGACY_FILE))
    if not frames:
        raise FileNotFoundError(
            "No cluster enrichment CSVs found. Run a held-out comparison script first."
        )
    data = pd.concat(frames, ignore_index=True)
    required = {
        "embedding", "task", "cluster", "label", "n_cluster_compounds",
        "fold_enrichment", "q_value",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing cluster-enrichment columns: {sorted(missing)}")

    # The held-out protocols cover the same set of compounds when all folds are
    # pooled. Collapse identical tests while preserving which sources agreed.
    values = [
        "embedding", "task", "cluster", "label", "n_cluster_compounds",
        "n_with_label", "fold_enrichment", "q_value",
    ]
    values = [column for column in values if column in data.columns]
    if "protocol" in data.columns:
        data = data.groupby(values, dropna=False, as_index=False).agg(
            protocols=("protocol", lambda x: ", ".join(sorted(set(x))))
        )
    return data


def plot_enrichment() -> Path:
    data = load_enrichment()
    significant = data[(data.q_value < 0.05) & (data.fold_enrichment > 1)].copy()
    embeddings = [name for name in ["pca_normalized", "pca_raw", "dino"]
                  if name in set(data.embedding)]
    tasks = [name for name in ["pathway", "target"] if name in set(data.task)]
    if not embeddings or not tasks:
        raise ValueError("No recognized embedding or label-task rows in enrichment data.")

    fig, axes = plt.subplots(
        len(tasks), len(embeddings), figsize=(6 * len(embeddings), 5 * len(tasks)),
        squeeze=False,
    )
    for row, task in enumerate(tasks):
        for col, embedding in enumerate(embeddings):
            ax = axes[row, col]
            subset = significant[
                (significant.task == task) & (significant.embedding == embedding)
            ].nsmallest(8, "q_value").sort_values("fold_enrichment")
            ax.set_title(f"{embedding.replace('_', ' ').title()} · {task.title()}")
            if subset.empty:
                ax.text(0.5, 0.5, "No enriched labels at q < 0.05",
                        transform=ax.transAxes, ha="center", va="center")
                ax.set_axis_off()
                continue

            log_enrichment = np.log10(subset.fold_enrichment.to_numpy())
            sizes = 35 + 18 * np.sqrt(subset.n_cluster_compounds.to_numpy())
            scatter = ax.scatter(
                log_enrichment,
                np.arange(len(subset)),
                s=sizes,
                c=-np.log10(subset.q_value.clip(lower=1e-300)),
                cmap="viridis",
                edgecolors="#333333",
                linewidths=0.5,
            )
            left = min(0.0, np.floor(log_enrichment.min() * 2) / 2)
            right = np.ceil(log_enrichment.max() * 2) / 2
            ax.set_xlim(left, right + 0.1)
            ax.set_xticks(np.arange(left, right + 0.01, 0.5))
            ax.set_yticks(np.arange(len(subset)))
            ax.set_yticklabels([
                f"{label} (C{cluster}; n={n})"
                for label, cluster, n in zip(
                    subset.label, subset.cluster, subset.n_cluster_compounds
                )
            ])
            ax.set_xlabel("log10(fold enrichment); 1 = 10×")
            ax.grid(axis="x", alpha=0.25)
            ax.set_axisbelow(True)
            fig.colorbar(scatter, ax=ax, label="−log10 adjusted q-value", pad=0.02)

    fig.suptitle("Strongest cluster–label enrichments (bubble size = cluster size)",
                 y=1.015, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output = RESULTS / "cluster_enrichment" / "cluster_label_enrichment.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(f"Saved {plot_enrichment()}")
