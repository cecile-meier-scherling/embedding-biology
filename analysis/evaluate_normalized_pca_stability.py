"""Report fold spread and bootstrap intervals for normalized-PCA F1@3.

Uses already-generated chemical split groups from results/chemical_split/chemical_split_structure_audit.csv.
Run from any directory with:
    uv run python /path/to/axiom_takehome/analysis/evaluate_normalized_pca_stability.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from annotation_model_utils import (
    DATA_PATH,
    N_SPLITS,
    RESULTS_DIR,
    build_splits,
    make_label_matrix,
    parse_labels,
    run_prediction,
    unpack_vector,
)


EMBEDDING_COLUMN = "pca_embedding_normalized"
OUTPUT_PATH = RESULTS_DIR / "annotation_prediction" / "normalized_pca_prediction_stability.csv"
CHEMICAL_AUDIT_PATH = RESULTS_DIR / "chemical_split" / "chemical_split_structure_audit.csv"


def assess(df: pd.DataFrame, protocol: str,
           splits: list[tuple[np.ndarray, np.ndarray]],
           bootstrap_units: np.ndarray, rows: list[dict]) -> None:
    groups = df["compound_id"].astype(str).to_numpy()
    X = np.stack(df[EMBEDDING_COLUMN].map(unpack_vector))
    for task, column in [("pathway", "compound_pathway"), ("target", "compound_target")]:
        labels = df[column].map(parse_labels).tolist()
        Y, kept_labels = make_label_matrix(labels, groups)
        _, _ = run_prediction(
            X, Y, groups, splits, task, "pca_normalized", protocol,
            stability_rows=rows, bootstrap_units=bootstrap_units,
        )
        print(f"Finished {protocol} {task}: {len(kept_labels)} labels")


def main() -> None:
    base = pd.read_parquet(DATA_PATH)
    base = base[base["compound_id"].notna()]
    base = base[base["compound_pathway"].notna() & base["compound_target"].notna()]
    base = base.reset_index(drop=True)
    rows: list[dict] = []

    available = build_splits(base)
    compound_units = base["compound_id"].astype(str).to_numpy()
    assess(base, "held_out_compound", available["held_out_compound"],
           compound_units, rows)
    assess(base, "held_out_batch", available["held_out_batch"],
           compound_units, rows)

    if not CHEMICAL_AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CHEMICAL_AUDIT_PATH}; run the chemical-split comparison first."
        )
    audit = pd.read_csv(CHEMICAL_AUDIT_PATH, dtype={"compound_id": str})
    group_by_compound = audit.dropna(subset=["chemical_group"]).drop_duplicates(
        "compound_id"
    ).set_index("compound_id")["chemical_group"].to_dict()
    chemical_df = base[base["compound_id"].astype(str).isin(group_by_compound)].copy()
    chemical_df = chemical_df.reset_index(drop=True)
    chemical_compounds = chemical_df["compound_id"].astype(str).to_numpy()
    chemical_units = np.array([group_by_compound[c] for c in chemical_compounds])
    chemical_splits = list(GroupKFold(n_splits=N_SPLITS).split(
        chemical_df, groups=chemical_units
    ))
    assess(chemical_df, "held_out_chemical_group", chemical_splits,
           chemical_units, rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved fold scores and bootstrap intervals to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
