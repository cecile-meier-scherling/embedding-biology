"""Predict pathway/target labels from biological-activity descriptions.

Uses TF-IDF text features and one-vs-rest logistic regression. Evaluation uses
the same compound-held-out and batch-held-out splits as
compare_annotation_models.py, with duplicate compounds excluded from batch
training folds. Run from any directory with:

    uv run python /path/to/axiom_takehome/analysis/predict_annotations_from_activity_text.py

Results are written to results/activity_text_retrieval/biological_activity_text_comparison.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from annotation_model_utils import (
    DATA_PATH,
    MIN_LABEL_COMPOUNDS,
    RESULTS_DIR,
    build_splits,
    parse_labels,
)


RANDOM_STATE = 42
TEXT_COLUMN = "compound_biological_activity"


def make_label_matrix(labels: list[tuple[str, ...]], groups: np.ndarray):
    compounds_by_label: dict[str, set[str]] = {}
    for compound, row_labels in zip(groups, labels):
        for label in row_labels:
            compounds_by_label.setdefault(label, set()).add(compound)
    keep = sorted(
        label for label, compounds in compounds_by_label.items()
        if len(compounds) >= MIN_LABEL_COMPOUNDS
    )
    keep_set = set(keep)
    filtered_labels = [tuple(label for label in row if label in keep_set) for row in labels]
    mlb = MultiLabelBinarizer(classes=keep)
    return mlb.fit_transform(filtered_labels), keep


def text_to_matrix(train_text: list[str], test_text: list[str]):
    # Fit vocabulary and IDF weights on training descriptions only.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )
    try:
        X_train = vectorizer.fit_transform(train_text)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
        # A small/very repetitive training fold may not meet min_df=2.
        vectorizer = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", stop_words="english",
            ngram_range=(1, 2), min_df=1, max_features=100_000,
            sublinear_tf=True,
        )
        X_train = vectorizer.fit_transform(train_text)
    return X_train, vectorizer.transform(test_text)


def fold_scores(X_train, Y_train: np.ndarray, X_test) -> np.ndarray:
    """Fit one classifier per label; score labels absent from training as negative."""
    scores = np.full((X_test.shape[0], Y_train.shape[1]), -1.0, dtype=np.float32)
    for j in range(Y_train.shape[1]):
        y = Y_train[:, j]
        if np.unique(y).size < 2:
            if y[0] == 1:
                scores[:, j] = 1.0
            continue
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y)
        scores[:, j] = model.decision_function(X_test)
    return scores


def evaluate_protocol(texts: list[str], Y: np.ndarray,
                      groups: np.ndarray, splits, protocol: str,
                      n_labels: int) -> dict[str, object]:
    score_sums: dict[str, np.ndarray] = {}
    truths: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for train, test in splits:
        X_train, X_test = text_to_matrix(
            [texts[i] for i in train], [texts[i] for i in test]
        )
        fold_predictions = fold_scores(X_train, Y[train], X_test)
        for idx, scores in zip(test, fold_predictions):
            compound = groups[idx]
            score_sums[compound] = score_sums.get(
                compound, np.zeros(Y.shape[1], dtype=float)
            ) + scores
            truths[compound] = Y[idx]
            counts[compound] = counts.get(compound, 0) + 1

    compounds = sorted(score_sums)
    y_true = np.stack([truths[c] for c in compounds])
    y_pred = np.zeros_like(y_true)
    for i, compound in enumerate(compounds):
        mean_scores = score_sums[compound] / counts[compound]
        y_pred[i, np.argsort(mean_scores)[-min(3, mean_scores.size):]] = 1

    return {
        "protocol": protocol,
        "n_compounds": len(compounds),
        "n_labels": n_labels,
        "macro_f1_top3": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "micro_f1_top3": f1_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
    }


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df = df[df["compound_id"].notna()].copy()
    df = df[df[TEXT_COLUMN].notna() & df["compound_pathway"].notna()
            & df["compound_target"].notna()].copy()
    df = df.reset_index(drop=True)
    groups = df["compound_id"].astype(str).to_numpy()
    texts = df[TEXT_COLUMN].astype(str).str.replace(r"\s+", " ", regex=True).tolist()
    splits_by_protocol = build_splits(df)
    results = []

    for task, column in [("pathway", "compound_pathway"),
                         ("target", "compound_target")]:
        label_tuples = df[column].map(parse_labels).tolist()
        Y, kept_labels = make_label_matrix(label_tuples, groups)
        for protocol, splits in splits_by_protocol.items():
            result = evaluate_protocol(
                texts, Y, groups, splits, protocol, len(kept_labels)
            )
            result["task"] = task
            result["feature_set"] = "biological_activity_text_tfidf"
            results.append(result)
            print(
                f"{protocol:18s} {task:8s} text TF-IDF "
                f"macro-F1@3={result['macro_f1_top3']:.3f} "
                f"micro-F1@3={result['micro_f1_top3']:.3f} "
                f"({result['n_compounds']} compounds, {result['n_labels']} labels)"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "activity_text_retrieval" / "biological_activity_text_comparison.csv"
    pd.DataFrame(results).to_csv(output, index=False)
    print(f"\nSaved text prediction comparison to {output}")


if __name__ == "__main__":
    main()
