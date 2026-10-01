"""Compare fold-fitted matched-anchor corrections and scaling transforms.

Corrected variants are evaluated on held-out-compound, held-out-batch, and
held-out-chemical-group splits. Batch-specific corrections are omitted from
held-out-batch folds: an unseen batch needs a prespecified unlabeled
calibration set to estimate its offset. All fitted transformations use only
the training fold.

Run: uv run python analysis/evaluate_fold_safe_transforms.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import argparse
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

from annotation_model_utils import (
    ASSAY_FEATURES, DATA_PATH, N_SPLITS, RESULTS_DIR, make_label_matrix,
    parse_labels, train_and_score, unpack_vector,
)


def group_shift(train_x, train_meta, group_col):
    """Estimate group offsets from matched compound-dose profiles in training."""
    meta = train_meta.copy()
    meta["_dose"] = meta["compound_concentration_um"].astype("string").fillna("NA")
    meta["_cid"] = meta["compound_id"].astype(str)
    meta["_group"] = meta[group_col].astype(str)
    work = pd.DataFrame(train_x)
    work["_anchor"] = meta["_cid"].to_numpy() + "|" + meta["_dose"].to_numpy()
    work["_group"] = meta["_group"].to_numpy()
    # First average replicate wells within each matched anchor and group.
    grouped = work.groupby(["_anchor", "_group"], sort=False).mean(numeric_only=True)
    if grouped.empty:
        return {}
    anchor_means = grouped.groupby(level=0).mean()
    anchors_by_group = {
        str(group): set(grouped.xs(group, level=1).index.astype(str))
        for group in grouped.index.get_level_values(1).unique()
    }
    groups_by_anchor = {}
    for group, anchors in anchors_by_group.items():
        for anchor in anchors:
            groups_by_anchor.setdefault(anchor, set()).add(group)
    offsets = {}
    for group in grouped.index.get_level_values(1).unique():
        # Only anchors observed in multiple groups identify an offset.
        matched = [a for a in anchors_by_group[str(group)]
                   if len(groups_by_anchor.get(a, ())) > 1]
        if matched:
            delta = grouped.loc[[(a, group) for a in matched]].to_numpy() - anchor_means.loc[matched].to_numpy()
            offsets[str(group)] = np.nanmedian(delta, axis=0)
    return offsets


def prepare_fold(X, metadata, train, test, variant):
    """Fit imputation, correction and scaling on training data only."""
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    a = imp.fit_transform(X[train])
    b = imp.transform(X[test])
    tr_meta, te_meta = metadata.iloc[train], metadata.iloc[test]
    if variant.startswith("matched_") and "plate_batch" not in variant:
        _, group, strength = variant.split("_")
        strength = float(strength)
        shifts = group_shift(a, tr_meta, group)
        tr_groups = tr_meta[group].astype(str).to_numpy()
        te_groups = te_meta[group].astype(str).to_numpy()
        a = a.copy(); b = b.copy()
        for i, g in enumerate(tr_groups):
            if g in shifts:
                a[i] -= strength * shifts[g]
        for i, g in enumerate(te_groups):
            if g in shifts:
                b[i] -= strength * shifts[g]
    elif variant == "matched_plate_batch_0.5" or variant == "matched_plate_batch_1.0":
        strength = float(variant.rsplit("_", 1)[1])
        for group in ("plate", "batch"):
            shifts = group_shift(a, tr_meta, group)
            tr_groups = tr_meta[group].astype(str).to_numpy()
            te_groups = te_meta[group].astype(str).to_numpy()
            a = a.copy(); b = b.copy()
            for i, g in enumerate(tr_groups):
                if g in shifts: a[i] -= strength * shifts[g]
            for i, g in enumerate(te_groups):
                if g in shifts: b[i] -= strength * shifts[g]
    if variant in ("robust_scale", "pca50_whiten"):
        scaler = RobustScaler() if variant == "robust_scale" else StandardScaler()
        a = scaler.fit_transform(a)
        b = scaler.transform(b)
        if variant == "pca50_whiten":
            n = min(50, a.shape[1], max(2, a.shape[0] - 1))
            pca = PCA(n_components=n, whiten=True, random_state=42)
            a = pca.fit_transform(a)
            b = pca.transform(b)
    return a, b


def score_fold(y, score, compounds):
    compound_ids = sorted(set(compounds))
    y = np.stack([y[compounds == cid][0] for cid in compound_ids])
    score = np.stack([score[compounds == cid].mean(axis=0) for cid in compound_ids])
    pred = np.zeros_like(y)
    top = np.argsort(score, axis=1)[:, -min(3, score.shape[1]):]
    np.put_along_axis(pred, top, 1, axis=1)
    return (f1_score(y, pred, average="macro", zero_division=0),
            f1_score(y, pred, average="micro", zero_division=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", nargs="+", default=["pca_raw", "pca_normalized", "dino", "assay_features"],
                        choices=["pca_raw", "pca_normalized", "dino", "assay_features"])
    parser.add_argument("--protocols", nargs="+", default=["held_out_compound", "held_out_batch", "held_out_chemical_group"],
                        choices=["held_out_compound", "held_out_batch", "held_out_chemical_group"])
    args = parser.parse_args()
    df = pd.read_parquet(DATA_PATH)
    df = df[df.compound_id.notna() & df.compound_pathway.notna() & df.compound_target.notna()].copy().reset_index(drop=True)
    ids = df.compound_id.astype(str).to_numpy()
    metadata = df[["compound_id", "compound_concentration_um", "plate", "batch"]].reset_index(drop=True)
    base = {
        "pca_raw": np.stack(df.pca_embedding_raw.map(unpack_vector)),
        "pca_normalized": np.stack(df.pca_embedding_normalized.map(unpack_vector)),
        "dino": np.stack(df.brightfield.map(unpack_vector)),
        "assay_features": df[ASSAY_FEATURES].to_numpy(dtype=np.float32),
    }
    base["assay_features"][~np.isfinite(base["assay_features"])] = np.nan
    chemical_audit = RESULTS_DIR / "chemical_split" / "chemical_split_structure_audit.csv"
    chemical_map = {}
    if chemical_audit.exists():
        audit = pd.read_csv(chemical_audit)
        chemical_map = dict(zip(audit.compound_id.astype(str), audit.chemical_group))

    protocols = {"held_out_compound": list(GroupKFold(N_SPLITS).split(df, groups=ids))}
    batches = df.batch.to_numpy()
    batch_splits = []
    for batch in sorted(pd.unique(batches)):
        test = np.flatnonzero(batches == batch)
        held_ids = set(ids[test])
        train = np.flatnonzero((batches != batch) & ~np.isin(ids, list(held_ids)))
        if len(train) and len(test):
            batch_splits.append((train, test))
    protocols["held_out_batch"] = batch_splits
    if chemical_map:
        keep = np.array([cid in chemical_map for cid in ids])
        chem_df = df.loc[keep].reset_index(drop=True)
        chem_ids = ids[keep]
        chem_groups = np.array([chemical_map[c] for c in chem_ids])
        protocols["held_out_chemical_group"] = (
            chem_df, chem_ids,
            list(GroupKFold(N_SPLITS).split(chem_df, groups=chem_groups)), keep,
        )

    variants = ("uncorrected", "matched_plate_0.5", "matched_plate_1.0",
                "matched_batch_0.5", "matched_batch_1.0",
                "matched_plate_batch_0.5", "matched_plate_batch_1.0",
                "robust_scale", "pca50_whiten")
    rows = []
    for protocol in args.protocols:
        if protocol not in protocols:
            print(f"Skipping {protocol}: chemical structure grouping is unavailable")
            continue
        config = protocols[protocol]
        if protocol != "held_out_chemical_group":
            current_df, current_ids, splits = df, ids, config
            matrices = {k: v for k, v in base.items() if k in args.embeddings}
        else:
            current_df, current_ids, splits, mask = config
            matrices = {k: v[mask] for k, v in base.items() if k in args.embeddings}
        for task, col in (("pathway", "compound_pathway"), ("target", "compound_target")):
            y, labels = make_label_matrix(current_df[col].map(parse_labels).tolist(), current_ids)
            for name, X in matrices.items():
                for variant in variants:
                    if protocol == "held_out_batch" and (
                        variant.startswith("matched_batch_") or
                        variant.startswith("matched_plate_batch_")
                    ):
                        continue
                    fold_scores = []
                    for fold, (train, test) in enumerate(splits, 1):
                        if variant == "uncorrected":
                            xa, xb = X[train], X[test]
                        else:
                            xa, xb = prepare_fold(X, current_df[["compound_id", "compound_concentration_um", "plate", "batch"]], train, test, variant)
                        pred_scores = train_and_score(xa, y[train], xb, groups_train=current_ids[train])
                        macro, micro = score_fold(y[test], pred_scores, current_ids[test])
                        fold_scores.append((macro, micro))
                        rows.append({"protocol": protocol, "task": task, "embedding": name,
                                     "transform": variant, "fold": fold,
                                     "macro_f1_top3": macro, "micro_f1_top3": micro,
                                     "n_labels": len(labels)})
                    print(protocol, task, name, variant,
                          f"Macro={np.mean([x[0] for x in fold_scores]):.3f}",
                          f"Micro={np.mean([x[1] for x in fold_scores]):.3f}")
    detail = pd.DataFrame(rows)
    suffix = args.protocols[0] if len(args.protocols) == 1 else "all_protocols"
    out = RESULTS_DIR / "feature_processing" / f"fold_safe_transform_prediction_metrics_{suffix}.csv"
    detail.to_csv(out, index=False)
    summary_rows = []
    rng = np.random.default_rng(42)
    for (protocol, task, embedding, transform), part in detail.groupby(
            ["protocol", "task", "embedding", "transform"]):
        baseline = detail[(detail.protocol == protocol) & (detail.task == task) &
                          (detail.embedding == embedding) & (detail["transform"] == "uncorrected")]
        summary = {"protocol": protocol, "task": task, "embedding": embedding,
                   "transform": transform, "n_folds": len(part)}
        for metric in ("macro_f1_top3", "micro_f1_top3"):
            vals = part[metric].to_numpy()
            summary[f"{metric}_mean"] = float(vals.mean())
            summary[f"{metric}_fold_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            sampled = rng.choice(vals, (10000, len(vals)), replace=True).mean(axis=1)
            summary[f"{metric}_bootstrap_ci_low"] = float(np.quantile(sampled, .025))
            summary[f"{metric}_bootstrap_ci_high"] = float(np.quantile(sampled, .975))
            paired = part.sort_values("fold")[metric].to_numpy() - baseline.sort_values("fold")[metric].to_numpy()
            sampled_delta = rng.choice(paired, (10000, len(paired)), replace=True).mean(axis=1)
            summary[f"delta_{metric}_vs_uncorrected"] = float(paired.mean())
            summary[f"delta_{metric}_ci_low"] = float(np.quantile(sampled_delta, .025))
            summary[f"delta_{metric}_ci_high"] = float(np.quantile(sampled_delta, .975))
        summary_rows.append(summary)
    summary_path = RESULTS_DIR / "feature_processing" / f"fold_safe_transform_prediction_summary_{suffix}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved fold-level comparison to {out}")
    print(f"Saved means and paired fold-bootstrap intervals to {summary_path}")
    print("Batch-offset correction is omitted from held-out-batch folds because no calibration panel is configured.")


if __name__ == "__main__":
    main()
