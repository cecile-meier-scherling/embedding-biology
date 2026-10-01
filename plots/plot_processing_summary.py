"""Plot paired processing effects across prediction and embedding diagnostics.

Run from the repository root:
    .venv/bin/python plots/plot_processing_summary.py

The plot uses held-out-compound folds throughout. DMSO whitening is evaluated
on the calibrated-plate subset; fold-safe Harmony prediction is available for
raw and normalized PCA only. Harmony/DMSO diagnostics are saved separately.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FEATURES = ["pca_raw", "pca_normalized", "dino"]
FEATURE_LABEL = {"pca_raw": "Raw PCA", "pca_normalized": "Normalized PCA", "dino": "DINO"}
FEATURE_COLOR = {"pca_raw": "#4477AA", "pca_normalized": "#228833", "dino": "#CC6677"}
METHOD_LABEL = {
    "uncorrected_reference": "Unprocessed reference",
    "matched_plate": "Matched plate shift",
    "repeatable_top75": "Repeatable dimensions",
    "pca50_whiten": "PCA whitening",
    "rank_gaussian": "Rank-Gaussian",
    "harmony_batch": "Harmony: batch",
    "harmony_plate_batch": "Harmony: plate + batch",
    "dmso_whitened": "DMSO whitening (16-plate subset)",
}
METHODS_BY_FEATURE = {
    "pca_raw": list(METHOD_LABEL),
    "pca_normalized": list(METHOD_LABEL),
    # DINO fold-safe Harmony prediction did not finish; don't imply a complete
    # pathway/target comparison for those Harmony rows.
    "dino": ["uncorrected_reference", "matched_plate", "repeatable_top75", "pca50_whiten", "rank_gaussian", "dmso_whitened"],
}


def lighter(color: str, amount: float = 0.48) -> tuple[float, float, float]:
    """Blend an embedding color toward white for its unprocessed reference."""
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def paired_deltas(frame: pd.DataFrame, value: str, keys: list[str],
                  baseline_by_transform: dict[str, str],
                  higher_is_better: bool = True) -> pd.DataFrame:
    """Join each method to its matching baseline on identical fold/query keys."""
    rows = []
    sign = 1 if higher_is_better else -1
    for transform, baseline_transform in baseline_by_transform.items():
        method = frame[frame["transform"].eq(transform)][[*keys, value]].rename(columns={value: "method_value"})
        baseline = frame[frame["transform"].eq(baseline_transform)][[*keys, value]].rename(columns={value: "baseline_value"})
        joined = method.merge(baseline, on=keys, how="inner")
        if joined.empty:
            continue
        joined["transform"] = transform
        joined["delta"] = sign * (joined.method_value - joined.baseline_value)
        rows.append(joined)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[*keys, "transform", "delta"])


def bootstrap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(291)
    rows = []
    for (feature, transform), group in frame.groupby(["feature_set", "transform"]):
        values = group.delta.dropna().to_numpy(float)
        if not len(values):
            continue
        if len(values) < 2:
            low = high = float(values[0])
        else:
            means = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
            low, high = np.quantile(means, [.025, .975])
        rows.append({"feature_set": feature, "transform": transform,
                     "mean": float(values.mean()), "low": float(low), "high": float(high),
                     "n": len(values)})
    return pd.DataFrame(rows)


def plot_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize ordinary fold deltas while preserving supplied CIs verbatim.

    Harmony prediction intervals are precomputed from paired fold deltas and
    arrive as one summary row per feature/method. They must not be bootstrapped
    again as if that one summary row were a single fold observation.
    """
    frame = frame.copy()
    for column in ("low", "high"):
        if column not in frame:
            frame[column] = np.nan
    regular = frame[frame["low"].isna() | frame["high"].isna()].copy()
    summarized = bootstrap_summary(regular)
    direct = frame[frame["low"].notna() & frame["high"].notna()].copy()
    if not direct.empty:
        direct = direct.groupby(["feature_set", "transform"], as_index=False).agg(
            mean=("delta", "mean"), low=("low", "first"), high=("high", "first"), n=("delta", "size")
        )
    return pd.concat([summarized, direct], ignore_index=True)


def prediction_deltas(task: str) -> pd.DataFrame:
    paths = [RESULTS / "feature_processing" / "processing_candidate_biology_folds.csv",
             RESULTS / "feature_processing" / "processing_candidate_biology_folds_geometry_candidates.csv",
             RESULTS / "feature_processing" / "processing_candidate_biology_folds_dino_shared.csv"]
    main = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    main = main[main.protocol.eq("held_out_compound") & main.task.eq(task)]
    main = main.groupby(["protocol", "fold", "feature_set", "transform"], as_index=False).macro_f1_top3.mean()
    common = paired_deltas(
        main, "macro_f1_top3", ["protocol", "fold", "feature_set"],
        {name: "uncorrected" for name in ("matched_plate", "repeatable_top75", "pca50_whiten", "rank_gaussian")},
    )

    # Fold-safe Harmony prediction uses raw/normalized PCA and the same held-out
    # compounds as the common processing candidates.
    harmony = pd.read_csv(RESULTS / "feature_processing" / "fold_safe_harmony_summary.csv")
    harmony = harmony[harmony.protocol.eq("held_out_compound") & harmony.task.eq(task)]
    harmony_rows = []
    for _, row in harmony[harmony["transform"].isin(["harmony_batch", "harmony_plate_batch"])].iterrows():
        harmony_rows.append({"protocol": row["protocol"], "fold": "paired5", "feature_set": row["feature_set"],
                             "transform": row["transform"],
                             "delta": row["delta_macro_f1_top3_vs_uncorrected"],
                             "low": row["delta_macro_f1_top3_ci_low"],
                             "high": row["delta_macro_f1_top3_ci_high"]})
    harmony_delta = pd.DataFrame(harmony_rows)

    # DMSO prediction is an aggregate paired comparison on 295 compounds from
    # 16 calibrated plates in one batch; no fold-level interval is available.
    raw_dmso = pd.read_csv(RESULTS / "feature_processing" / "dmso_whitening_prediction_metrics.csv")
    raw_dmso = raw_dmso[raw_dmso.protocol.eq("held_out_compound") & raw_dmso.task.eq(task)].copy()
    raw_dmso["embedding"] = raw_dmso.feature_set.str.replace("_uncorrected|_dmso_whitened", "", regex=True)
    wide = raw_dmso.pivot_table(index="embedding", columns="feature_set", values="macro_f1_top3", aggfunc="mean")
    dmso_rows = []
    for feature in FEATURES:
        base_col, corrected_col = f"{feature}_uncorrected", f"{feature}_dmso_whitened"
        if feature in wide.index and base_col in wide and corrected_col in wide:
            dmso_rows.append({"protocol": "held_out_compound_calibrated_subset", "fold": "aggregate",
                              "feature_set": feature, "transform": "dmso_whitened",
                              "delta": wide.loc[feature, corrected_col] - wide.loc[feature, base_col],
                              "low": np.nan, "high": np.nan})
    dmso_delta = pd.DataFrame(dmso_rows)
    return pd.concat([common, harmony_delta, dmso_delta], ignore_index=True, sort=False)


def retrieval_deltas() -> pd.DataFrame:
    common = pd.read_csv(RESULTS / "feature_processing" / "transformation_annotation_retrieval_per_query.csv")
    common = common[common.protocol.eq("held_out_compound") & common.k.eq(10)]
    common = common.groupby(["protocol", "fold", "feature_set", "task", "transform"], as_index=False).map_at_k.mean()
    common_delta = paired_deltas(
        common, "map_at_k", ["protocol", "fold", "feature_set", "task"],
        {name: "uncorrected" for name in ("matched_plate", "repeatable_top75", "pca50_whiten", "rank_gaussian")},
    )

    extra = pd.read_csv(RESULTS / "feature_processing" / "additional_retrieval_diagnostics.csv")
    extra = extra[extra.protocol.eq("held_out_compound") & extra.k.eq(10)]
    extra = extra.groupby(["protocol", "fold", "feature_set", "task", "transform"], as_index=False).map_at_k.mean()
    extra_delta = paired_deltas(
        extra, "map_at_k", ["protocol", "fold", "feature_set", "task"],
        {"harmony_batch": "harmony_uncorrected", "harmony_plate_batch": "harmony_uncorrected",
         "dmso_whitened": "uncorrected"},
    )
    return pd.concat([common_delta, extra_delta], ignore_index=True)


def replicate_deltas() -> pd.DataFrame:
    common = pd.read_csv(RESULTS / "feature_processing" / "transformation_replicate_consistency_folds.csv")
    common = common[common.protocol.eq("held_out_compound")].copy()
    common["margin"] = common.within_anchor_cosine_mean - common.different_compound_cosine_mean
    common_delta = paired_deltas(
        common, "margin", ["protocol", "fold", "feature_set"],
        {name: "uncorrected" for name in ("matched_plate", "repeatable_top75", "pca50_whiten", "rank_gaussian")},
    )
    extra = pd.read_csv(RESULTS / "feature_processing" / "additional_replicate_diagnostics.csv")
    extra = extra[extra.protocol.eq("held_out_compound")].copy()
    extra["margin"] = extra.within_anchor_cosine_mean - extra.different_compound_cosine_mean
    extra_delta = paired_deltas(
        extra, "margin", ["protocol", "fold", "feature_set"],
        {"harmony_batch": "harmony_uncorrected", "harmony_plate_batch": "harmony_uncorrected",
         "dmso_whitened": "uncorrected"},
    )
    return pd.concat([common_delta, extra_delta], ignore_index=True)


def confound_deltas() -> pd.DataFrame:
    common = pd.read_csv(RESULTS / "feature_processing" / "transformation_confounding_folds.csv")
    common = common[common.protocol.eq("held_out_compound") & common.nuisance.isin(["plate", "batch"])]
    common_delta = paired_deltas(
        common, "balanced_accuracy", ["protocol", "fold", "feature_set", "nuisance"],
        {name: "uncorrected" for name in ("matched_plate", "repeatable_top75", "pca50_whiten", "rank_gaussian")},
        higher_is_better=False,
    )
    extra = pd.read_csv(RESULTS / "feature_processing" / "additional_confounding_diagnostics.csv")
    extra = extra[extra.protocol.eq("held_out_compound") & extra.nuisance.isin(["plate", "batch"])]
    extra_delta = paired_deltas(
        extra, "balanced_accuracy", ["protocol", "fold", "feature_set", "nuisance"],
        {"harmony_batch": "harmony_uncorrected", "harmony_plate_batch": "harmony_uncorrected",
         "dmso_whitened": "uncorrected"},
        higher_is_better=False,
    )
    return pd.concat([common_delta, extra_delta], ignore_index=True)


def main() -> None:
    panels = [
        ("Pathway prediction", prediction_deltas("pathway"), "Macro F1@3 change"),
        ("Target prediction", prediction_deltas("target"), "Macro F1@3 change"),
        ("Annotation retrieval", retrieval_deltas(), "MAP@10 change"),
        ("Replicate consistency", replicate_deltas(), "Cosine-margin change"),
        ("Technical confound recovery", confound_deltas(), "Balanced-accuracy reduction"),
    ]
    rows = [(feature, method) for feature in FEATURES for method in METHODS_BY_FEATURE[feature]]
    y_index = {key: i for i, key in enumerate(rows)}
    fig, axes = plt.subplots(1, 5, figsize=(21, 9.5), sharey=True)
    for ax, (title, values, xlabel) in zip(axes, panels):
        values = values[values.feature_set.isin(FEATURES) & values["transform"].isin(METHOD_LABEL)]
        summary = plot_summary(values)
        # Show each embedding's unprocessed baseline explicitly. All plotted
        # quantities are changes from the matching unprocessed profile, so this
        # reference is exactly zero; the DMSO comparison uses its calibrated
        # 16-plate subset baseline.
        for feature in FEATURES:
            y = y_index[(feature, "uncorrected_reference")]
            ax.plot(0, y, marker="o", linestyle="none", markersize=5,
                    color=lighter(FEATURE_COLOR[feature]), markeredgecolor="white",
                    markeredgewidth=.5, zorder=4)
        for _, point in summary.iterrows():
            key = (point["feature_set"], point["transform"])
            if key not in y_index:
                continue
            y = y_index[key]
            point_low, point_high = point.low, point.high
            ax.errorbar(point["mean"], y,
                        xerr=[[max(0, point["mean"] - point_low)], [max(0, point_high - point["mean"])]],
                        fmt="o", capsize=2.5, markersize=5,
                        color=FEATURE_COLOR[point["feature_set"]], markeredgecolor="white",
                        markeredgewidth=.5)
        ax.axvline(0, color="#333333", lw=1, ls="--")
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.grid(axis="x", alpha=.22)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([f"{FEATURE_LABEL[f]} · {METHOD_LABEL[m]}" for f, m in rows], fontsize=8)
    axes[0].invert_yaxis()
    boundary = 0
    for feature in FEATURES[:-1]:
        boundary += len(METHODS_BY_FEATURE[feature])
        for ax in axes:
            ax.axhline(boundary - .5, color="#bbbbbb", lw=.8)
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=FEATURE_COLOR[f],
                          markeredgecolor="white", markersize=7, label=FEATURE_LABEL[f]) for f in FEATURES]
    handles.append(plt.Line2D([0], [0], marker="o", linestyle="none", color="none",
                               markerfacecolor="#b8b8b8", markeredgecolor="white", markersize=7,
                               label="Unprocessed reference"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .035), bbox_transform=fig.transFigure)
    fig.suptitle("Performance of effects relative to unprocessed embeddings",
                 fontsize=16, x=.5, y=.99, ha="center")
    fig.subplots_adjust(left=.28, right=.99, top=.90, bottom=.17, wspace=.25)
    out = RESULTS / "feature_processing" / "processing_summary.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
