"""Plot effects of embedding transformations on biological and technical metrics.

Run from the repository root:
    uv run python plots/plot_processing_improvements.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TRANSFORMS = ["matched_plate", "repeatable_top75", "pca50_whiten", "rank_gaussian"]
FEATURES = ["pca_raw", "pca_normalized", "dino"]
FEATURE_LABELS = {"pca_raw": "Raw PCA", "pca_normalized": "Normalized PCA", "dino": "DINO"}
TASKS = ["pathway", "target"]
TASK_LABELS = {"pathway": "Pathway", "target": "Target"}
COLORS = "PuOr"


def paired_delta(frame: pd.DataFrame, metric: str, keys: list[str], baseline: str,
                 higher_is_better: bool = True) -> pd.DataFrame:
    """Compute fold-paired change from uncorrected for one metric."""
    wide = frame.pivot_table(index=keys, columns="transform", values=metric, aggfunc="mean")
    if baseline not in wide:
        raise ValueError(f"Baseline {baseline!r} missing for {metric}")
    out = wide.reset_index().melt(id_vars=keys, var_name="transform", value_name="value")
    out = out[out["transform"] != baseline].copy()
    base = wide[baseline].rename("baseline").reset_index()
    out = out.merge(base, on=keys, how="left")
    sign = 1 if higher_is_better else -1
    out["improvement"] = sign * (out.value - out.baseline)
    return out


def diagnostic_matrices() -> dict[str, tuple[list[str], list[str], pd.DataFrame, str]]:
    """Return fold-paired deltas, averaged equally across validation protocols."""
    # Replicate separation: same compound-dose cosine minus different-compound
    # cosine. Larger margin means replicate wells are more distinct from
    # unrelated compounds while remaining close to their own replicates.
    rep = pd.read_csv(RESULTS / "feature_processing" / "transformation_replicate_consistency_folds.csv")
    rep["margin"] = rep.within_anchor_cosine_mean - rep.different_compound_cosine_mean
    rep_delta = paired_delta(
        rep, "margin", ["protocol", "fold", "feature_set"], "uncorrected"
    )
    rep_avg = rep_delta.groupby(["feature_set", "transform"], as_index=False).improvement.mean()
    rep_rows = FEATURES
    rep_mat = rep_avg.pivot(index="feature_set", columns="transform", values="improvement").reindex(
        index=rep_rows, columns=TRANSFORMS
    )

    # Annotation retrieval: fold-paired change in mean average precision at 10.
    ret = pd.read_csv(RESULTS / "feature_processing" / "transformation_annotation_retrieval_per_query.csv")
    ret = ret[ret.k == 10].groupby(
        ["protocol", "fold", "feature_set", "task", "transform"], as_index=False
    ).map_at_k.mean()
    ret_delta = paired_delta(
        ret, "map_at_k", ["protocol", "fold", "feature_set", "task"], "uncorrected"
    )
    ret_avg = ret_delta.groupby(["feature_set", "task", "transform"], as_index=False).improvement.mean()
    ret_rows = [f"{feature}|{task}" for feature in FEATURES for task in TASKS]
    ret_avg["row"] = ret_avg.feature_set + "|" + ret_avg.task
    ret_mat = ret_avg.pivot(index="row", columns="transform", values="improvement").reindex(
        index=ret_rows, columns=TRANSFORMS
    )

    # Clustering: change in number of BH q<0.05 cluster-label enrichments per
    # fold. This is descriptive and should not be treated as a standalone
    # performance score.
    clu = pd.read_csv(RESULTS / "cluster_enrichment" / "transformation_cluster_enrichment.csv")
    clu["significant"] = clu.q_value < 0.05
    counts = clu.groupby(
        ["protocol", "fold", "feature_set", "task", "transform"], as_index=False
    ).significant.sum()
    expected = pd.MultiIndex.from_product(
        [counts.protocol.unique(), counts.fold.unique(), FEATURES, TASKS,
         ["uncorrected", *TRANSFORMS]],
        names=["protocol", "fold", "feature_set", "task", "transform"],
    ).to_frame(index=False)
    counts = expected.merge(counts, how="left").fillna({"significant": 0})
    clu_delta = paired_delta(
        counts, "significant", ["protocol", "fold", "feature_set", "task"], "uncorrected"
    )
    clu_avg = clu_delta.groupby(["feature_set", "task", "transform"], as_index=False).improvement.mean()
    clu_avg["row"] = clu_avg.feature_set + "|" + clu_avg.task
    clu_mat = clu_avg.pivot(index="row", columns="transform", values="improvement").reindex(
        index=ret_rows, columns=TRANSFORMS
    )

    # Technical label recovery: positive values mean balanced accuracy fell,
    # so plate or batch became harder to infer from within-compound residuals.
    conf = pd.read_csv(RESULTS / "feature_processing" / "transformation_confounding_folds.csv")
    conf = conf[conf.nuisance.isin(["plate", "batch"])]
    conf_delta = paired_delta(
        conf, "balanced_accuracy",
        ["protocol", "fold", "feature_set", "nuisance"], "uncorrected",
        higher_is_better=False,
    )
    conf_avg = conf_delta.groupby(["feature_set", "nuisance", "transform"], as_index=False).improvement.mean()
    conf_rows = [f"{feature}|{nuisance}" for feature in FEATURES for nuisance in ["plate", "batch"]]
    conf_avg["row"] = conf_avg.feature_set + "|" + conf_avg.nuisance
    conf_mat = conf_avg.pivot(index="row", columns="transform", values="improvement").reindex(
        index=conf_rows, columns=TRANSFORMS
    )
    return {
        "Replicate separation\n(change in cosine margin)": (rep_rows, TRANSFORMS, rep_mat, "Cosine-margin gain"),
        "Annotation retrieval\n(change in MAP@10)": (ret_rows, TRANSFORMS, ret_mat, "MAP@10 gain"),
        "Cluster enrichment\n(change in significant labels/fold)": (ret_rows, TRANSFORMS, clu_mat, "q < 0.05 count gain"),
        "Technical confound recovery\n(reduction in balanced accuracy)": (conf_rows, TRANSFORMS, conf_mat, "Balanced-accuracy reduction"),
    }


def plot_diagnostic_improvements() -> Path:
    panels = diagnostic_matrices()
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), layout="constrained")
    axes_flat = axes.ravel()
    for ax, (title, (rows, cols, matrix, _)) in zip(axes_flat, panels.items()):
        values = matrix.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        limit = max(float(np.max(np.abs(finite))) if len(finite) else 0.01, 0.01)
        mappable = ax.imshow(values, cmap=COLORS, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                             aspect="auto")
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xticks(np.arange(len(cols)), [
            "Matched\nplate", "Repeatable\ndimensions", "PCA\nwhitening", "Rank\nGaussian"
        ], fontsize=9)
        pretty_rows = []
        for row in rows:
            if "|" in row:
                feature, name = row.split("|", 1)
                pretty_rows.append(f"{FEATURE_LABELS[feature]} · {TASK_LABELS.get(name, name.title())}")
            else:
                pretty_rows.append(FEATURE_LABELS[row])
        ax.set_yticks(np.arange(len(rows)), pretty_rows, fontsize=9)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(len(rows)):
            for j in range(len(cols)):
                val = values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:+.3f}" if abs(val) < 10 else f"{val:+.1f}",
                            ha="center", va="center", fontsize=8,
                            color="white" if abs(val) > limit * 0.56 else "#252525")
        cbar = fig.colorbar(mappable, ax=ax, shrink=.82, pad=.025)
        cbar.set_label(panels[title][3], fontsize=8.5)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Processing changes relative to unprocessed embeddings", fontsize=17)
    path = RESULTS / "feature_processing" / "processing_diagnostic_improvements.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_prediction_deltas() -> Path:
    """Plot paired macro-F1 deltas for the candidate transforms with baselines."""
    regular = pd.read_csv(RESULTS / "feature_processing" / "processing_candidate_biology_folds.csv")
    geometry = pd.read_csv(RESULTS / "feature_processing" / "processing_candidate_biology_folds_geometry_candidates.csv")
    d = pd.concat([regular, geometry], ignore_index=True)
    d = d[d.feature_set.isin(["pca_raw", "pca_normalized"])]
    baseline = d[d["transform"] == "uncorrected"][
        ["protocol", "fold", "feature_set", "task", "macro_f1_top3"]
    ].rename(columns={"macro_f1_top3": "baseline"})
    d = d.merge(baseline, on=["protocol", "fold", "feature_set", "task"], how="left")
    d["delta"] = d.macro_f1_top3 - d.baseline
    d = d[d["transform"] != "uncorrected"].copy()
    # Include all tested transformations from the candidate runs that actually
    # have a matching unprocessed baseline.
    transform_names = ["matched_plate", "repeatable_top75", "pca50_whiten", "rank_gaussian"]
    d = d[d["transform"].isin(transform_names)]
    protocols = ["held_out_compound", "held_out_chemical_group"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    label_names = ["Matched plate", "Repeatable dimensions", "PCA whitening", "Rank Gaussian"]
    offsets = {name: offset for name, offset in zip(transform_names, [-.27, -.09, .09, .27])}
    colors = {"pca_raw": "#4C78A8", "pca_normalized": "#F58518"}
    rng = np.random.default_rng(42)
    for row, task in enumerate(["pathway", "target"]):
        for col, protocol in enumerate(protocols):
            ax = axes[row, col]
            subset = d[(d.task == task) & (d.protocol == protocol)]
            for feature in ["pca_raw", "pca_normalized"]:
                for ti, transform in enumerate(transform_names):
                    vals = subset[(subset.feature_set == feature) & (subset["transform"] == transform)].delta.to_numpy()
                    if not len(vals):
                        continue
                    x = ti + offsets[transform]
                    boot = rng.choice(vals, size=(10000, len(vals)), replace=True).mean(axis=1)
                    low, high = np.quantile(boot, [.025, .975])
                    mean = float(vals.mean())
                    ax.errorbar(x, mean, yerr=[[mean-low], [high-mean]], fmt="o", capsize=3,
                                color=colors[feature], markersize=7, elinewidth=1.4,
                                markeredgecolor="white", markeredgewidth=.6, zorder=3,
                                label=FEATURE_LABELS[feature] if ti == 0 else None)
            ax.axhline(0, color="#333333", linewidth=1, linestyle="--")
            ax.set_title(f"{TASK_LABELS[task]} · {protocol.replace('_', ' ').title()}", fontsize=11, pad=7)
            ax.set_xticks(range(len(transform_names)), label_names, rotation=18, ha="right", fontsize=9)
            ax.grid(axis="y", alpha=.25)
            ax.set_axisbelow(True)
            ax.set_ylabel("Change in Macro F1@3")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, .955))
    fig.suptitle("Pathway and target prediction after PCA processing", fontsize=16, y=.995)
    fig.subplots_adjust(left=.09, right=.99, top=.86, bottom=.18, wspace=.12, hspace=.14)
    path = RESULTS / "feature_processing" / "processing_prediction_deltas.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    print(plot_diagnostic_improvements())
    print(plot_prediction_deltas())


if __name__ == "__main__":
    main()
