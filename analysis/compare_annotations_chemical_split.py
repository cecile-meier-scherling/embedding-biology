"""Compare annotation prediction with chemically separated cross-validation.

Compound names are resolved to PubChem SMILES (cached for review), standardized
to the largest parent fragment with RDKit, and grouped by Bemis–Murcko scaffold.
Acyclic compounds, which have no Murcko scaffold, are grouped by Butina
clusters of Morgan radius-2, 2048-bit fingerprints at Tanimoto similarity >=
0.60. Each fold holds out whole chemical groups.

Run from any directory with:
    uv run python /path/to/axiom_takehome/analysis/compare_annotations_chemical_split.py
"""
from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import GroupKFold

from annotation_model_utils import (
    ASSAY_FEATURES,
    DATA_PATH,
    N_SPLITS,
    RANDOM_STATE,
    RESULTS_DIR,
    load_corrected_feature_arrays,
    make_label_matrix,
    parse_labels,
    run_prediction,
    unpack_vector,
)


PUBCHEM_CACHE = RESULTS_DIR / "chemical_split" / "pubchem_structure_lookup.csv"
OVERRIDES = DATA_PATH.parent / "compound_structure_overrides.csv"
SIMILARITY_THRESHOLD = 0.60
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
)


def pubchem_lookup(name: str) -> dict[str, str]:
    """Resolve a name using PubChem's PUG REST name-to-SMILES endpoint."""
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quote(name, safe='')}/property/SMILES/JSON"
    )
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    properties = response.json().get("PropertyTable", {}).get("Properties", [])
    if not properties:
        return {"cid": "", "smiles": "", "status": "not_found"}
    entry = properties[0]
    smiles = entry.get("SMILES") or entry.get("ConnectivitySMILES") or ""
    return {
        "cid": str(entry.get("CID", "")),
        "smiles": smiles,
        "status": "pubchem_first_match" if smiles else "no_smiles",
    }


def get_structure_table(compound_ids: list[str]) -> pd.DataFrame:
    """Load cached structures, look up missing names, and apply optional overrides."""
    cached = {}
    if PUBCHEM_CACHE.exists():
        old = pd.read_csv(PUBCHEM_CACHE, dtype=str).fillna("")
        cached = old.drop_duplicates("compound_id", keep="last").set_index("compound_id").to_dict("index")

    rows = []
    for compound_id in compound_ids:
        record = cached.get(compound_id)
        if record is not None:
            rows.append({"compound_id": compound_id, **record})
            continue
        try:
            record = pubchem_lookup(compound_id)
        except requests.RequestException as exc:
            record = {"cid": "", "smiles": "", "status": f"lookup_error:{type(exc).__name__}"}
        rows.append({"compound_id": compound_id, **record})
        # Keep request rate conservative and reduce the chance of throttling.
        time.sleep(0.22)

    structures = pd.DataFrame(rows).drop_duplicates("compound_id", keep="last")
    if OVERRIDES.exists():
        overrides = pd.read_csv(OVERRIDES, dtype=str).fillna("")
        required = {"compound_id", "smiles"}
        if not required.issubset(overrides.columns):
            raise ValueError(f"{OVERRIDES} must contain compound_id and smiles columns")
        override_cols = [c for c in ["compound_id", "cid", "smiles"] if c in overrides.columns]
        structures = structures.merge(
            overrides[override_cols].drop_duplicates("compound_id", keep="last"),
            on="compound_id", how="left", suffixes=("", "_override"),
        )
        for col in ("cid", "smiles"):
            override_col = f"{col}_override"
            if override_col in structures:
                structures[col] = structures[override_col].where(
                    structures[override_col].astype(bool), structures[col]
                )
        overridden = structures["smiles_override"].astype(bool) if "smiles_override" in structures else False
        structures.loc[overridden, "status"] = "manual_override"
        structures = structures.drop(columns=[c for c in structures if c.endswith("_override")])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    structures.to_csv(PUBCHEM_CACHE, index=False)
    return structures


def standardize_and_fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, None
    molecule = rdMolStandardize.Cleanup(molecule)
    molecule = rdMolStandardize.FragmentParent(molecule)
    molecule = rdMolStandardize.Uncharger().uncharge(molecule)
    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    # Reparse the standardized SMILES to ensure RDKit's ring information is set.
    standardized_molecule = Chem.MolFromSmiles(canonical_smiles)
    if standardized_molecule is None:
        return None, None
    fingerprint = MORGAN_GENERATOR.GetFingerprint(standardized_molecule)
    return canonical_smiles, fingerprint


def chemical_groups(structures: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
    valid = []
    for row in structures.itertuples(index=False):
        if not row.smiles:
            continue
        canonical, fingerprint = standardize_and_fingerprint(row.smiles)
        if fingerprint is not None:
            molecule = Chem.MolFromSmiles(canonical)
            scaffold_molecule = MurckoScaffold.GetScaffoldForMol(molecule)
            scaffold = Chem.MolToSmiles(
                scaffold_molecule, canonical=True, isomericSmiles=False
            )
            valid.append({
                "compound_id": row.compound_id,
                "cid": row.cid,
                "input_smiles": row.smiles,
                "canonical_parent_smiles": canonical,
                "murcko_scaffold": scaffold,
                "status": row.status,
                "fingerprint": fingerprint,
            })

    if len(valid) < N_SPLITS:
        raise ValueError(f"Only {len(valid)} compounds have valid structures; need at least {N_SPLITS}.")
    scaffold_group_ids: dict[str, int] = {}
    group_by_compound = {}
    next_group_id = 0
    for row in valid:
        scaffold = row["murcko_scaffold"]
        if scaffold:
            key = f"scaffold:{scaffold}"
            if key not in scaffold_group_ids:
                scaffold_group_ids[key] = next_group_id
                next_group_id += 1
            group_by_compound[row["compound_id"]] = scaffold_group_ids[key]

    # Acyclic compounds have no Murcko scaffold. Group their close analogs
    # by Morgan fingerprint instead of placing every acyclic molecule together.
    acyclic_indices = [i for i, row in enumerate(valid) if not row["murcko_scaffold"]]
    fingerprints = [valid[i]["fingerprint"] for i in acyclic_indices]
    if fingerprints:
        distances = []
        for i in range(1, len(fingerprints)):
            similarities = DataStructs.BulkTanimotoSimilarity(
                fingerprints[i], fingerprints[:i]
            )
            distances.extend(1.0 - similarity for similarity in similarities)
        clusters = Butina.ClusterData(
            distances, len(fingerprints), 1.0 - SIMILARITY_THRESHOLD,
            isDistData=True, reordering=True,
        )
        for cluster in clusters:
            group_id = next_group_id
            next_group_id += 1
            for local_index in cluster:
                row = valid[acyclic_indices[local_index]]
                group_by_compound[row["compound_id"]] = group_id

    audit = pd.DataFrame([
        {k: v for k, v in row.items() if k != "fingerprint"}
        | {"chemical_group": group_by_compound[row["compound_id"]]}
        for row in valid
    ])
    return group_by_compound, audit


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df["_source_row_index"] = np.arange(len(df))
    df = df[df["compound_id"].notna()].copy()
    df = df[df["compound_pathway"].notna() & df["compound_target"].notna()].copy()
    df = df.reset_index(drop=True)
    compound_ids = df["compound_id"].astype(str).to_numpy()
    unique_compounds = sorted(set(compound_ids))

    structures = get_structure_table(unique_compounds)
    chemical_group_by_compound, audit = chemical_groups(structures)
    n_unmatched = len(set(unique_compounds) - set(chemical_group_by_compound))
    df = df[df["compound_id"].astype(str).isin(chemical_group_by_compound)].copy()
    df = df.reset_index(drop=True)
    compound_ids = df["compound_id"].astype(str).to_numpy()
    fold_groups = np.array([chemical_group_by_compound[c] for c in compound_ids])
    splits = list(GroupKFold(n_splits=N_SPLITS).split(df, groups=fold_groups))

    embeddings = {
        "pca_normalized": np.stack(df["pca_embedding_normalized"].map(unpack_vector)),
        "pca_raw": np.stack(df["pca_embedding_raw"].map(unpack_vector)),
        "dino": np.stack(df["brightfield"].map(unpack_vector)),
    }
    assays = df[ASSAY_FEATURES].to_numpy(dtype=np.float32)
    assays[~np.isfinite(assays)] = np.nan
    inputs = {**embeddings, "assay_features": assays}
    inputs.update({
        f"{name}_plus_assays": np.concatenate([X, assays], axis=1)
        for name, X in embeddings.items()
    })
    inputs.update(load_corrected_feature_arrays(
        df["_source_row_index"].to_numpy(dtype=int),
        ["pca_raw", "pca_normalized", "dino", "assay_features"],
    ))
    inputs["label_frequency_baseline"] = np.zeros((len(df), 1), dtype=np.float32)

    metric_rows, stability_rows = [], []
    for task, column in [("pathway", "compound_pathway"), ("target", "compound_target")]:
        label_tuples = df[column].map(parse_labels).tolist()
        Y, kept_labels = make_label_matrix(label_tuples, compound_ids)
        for feature_set, X in inputs.items():
            result, _ = run_prediction(
                X, Y, compound_ids, splits, task, feature_set,
                "held_out_chemical_group",
                stability_rows=stability_rows,
                bootstrap_units=fold_groups,
                label_frequency_baseline=(feature_set == "label_frequency_baseline"),
            )
            result["n_chemical_groups"] = len(set(fold_groups))
            result["n_structured_compounds"] = len(set(compound_ids))
            result["n_unmatched_structures"] = n_unmatched
            result["n_labels"] = len(kept_labels)
            metric_rows.append(result)
            print(
                f"{task:8s} {feature_set:22s} "
                f"macro-F1@3={result['macro_f1_top3']:.3f} "
                f"micro-F1@3={result['micro_f1_top3']:.3f}"
            )

    # Keep a reviewable map from source names to structures and chemical groups.
    audit_path = RESULTS_DIR / "chemical_split" / "chemical_split_structure_audit.csv"
    audit.to_csv(audit_path, index=False)
    metrics_path = RESULTS_DIR / "annotation_prediction" / "annotation_model_comparison_chemical_split.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    stability_path = RESULTS_DIR / "annotation_prediction" / "annotation_prediction_stability_held_out_chemical_group.csv"
    pd.DataFrame(stability_rows).to_csv(stability_path, index=False)
    unmatched = structures[~structures.compound_id.isin(chemical_group_by_compound)]
    print(f"\nSaved chemical split metrics to {metrics_path}")
    print(f"Saved fold and bootstrap stability results to {stability_path}")
    print(f"Saved structure/group audit to {audit_path}")
    print(f"Unmatched or invalid structures excluded: {len(unmatched)} compounds")
    if len(unmatched):
        print(f"Review unresolved names in {PUBCHEM_CACHE}; manual overrides can be added to {OVERRIDES}.")


if __name__ == "__main__":
    main()
