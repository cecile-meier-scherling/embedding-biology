"""Benchmark image embeddings against downloaded, independent public annotations.

Binary labels are measured outcomes from Tox21 and DILIrank. Broad Repurposing
Hub MOA/target records are positive annotations, so they are evaluated only with
training-neighbor retrieval, never by treating missing annotations as negatives.
The source files and checksums are documented in data/public_annotations/manifest.json.
No network calls are made by this evaluator.

Run after analysis/import_public_annotations.py:
    uv run python analysis/evaluate_public_annotations.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from annotation_model_utils import ASSAY_FEATURES, DATA_PATH, RESULTS_DIR, unpack_vector
from evaluate_drug_attributes import (
    EMBEDDINGS, SEED, annotation_metrics, equal_compound_weights,
    transform,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = PROJECT_ROOT / 'data/public_annotations'
OUT = RESULTS_DIR / 'public_annotations'
PROTOCOLS = ('held_out_compound', 'held_out_batch', 'held_out_chemical_group')
MIN_CLASS_SUPPORT = 20
MIN_RETRIEVAL_SUPPORT = 6
TOX_ENDPOINTS = [
    'tox21__NR-AR', 'tox21__NR-AR-LBD', 'tox21__NR-AhR', 'tox21__NR-Aromatase',
    'tox21__NR-ER', 'tox21__NR-ER-LBD', 'tox21__NR-PPAR-gamma', 'tox21__SR-ARE',
    'tox21__SR-ATAD5', 'tox21__SR-HSE', 'tox21__SR-MMP', 'tox21__SR-p53',
]
DILI_ENDPOINTS = ['dili__most_vs_no', 'dili__any_concern_vs_no']
DAMAGE_COLUMNS = ['mean_nuclei_count', 'ldh_ridge_norm', 'mtt_ridge_norm']


def load_profiles():
    df = pd.read_parquet(DATA_PATH)
    df = df.loc[df.compound_id.notna()].copy().reset_index(drop=True)
    df['compound_id'] = df.compound_id.astype(str)
    arrays = {
        'pca_normalized': np.stack(df.pca_embedding_normalized.map(unpack_vector)).astype(np.float32),
        'pca_raw': np.stack(df.pca_embedding_raw.map(unpack_vector)).astype(np.float32),
        'brightfield': np.stack(df.brightfield.map(unpack_vector)).astype(np.float32),
    }
    assays = df[ASSAY_FEATURES].to_numpy(dtype=np.float32)
    assays[~np.isfinite(assays)] = np.nan
    arrays['assay_features'] = assays
    for array in arrays.values():
        array[~np.isfinite(array)] = np.nan
    keys = ['compound_id', 'batch']
    grouped = df.groupby(keys, sort=True, dropna=False).indices
    rows, profiles = [], {name: [] for name in arrays}
    for key, indices in grouped.items():
        compound_id, batch = key
        row = {'compound_id': str(compound_id), 'batch': batch}
        for col in DAMAGE_COLUMNS:
            row[col] = pd.to_numeric(df.iloc[indices][col], errors='coerce').mean()
        rows.append(row)
        for name, array in arrays.items():
            with np.errstate(invalid='ignore'):
                profiles[name].append(np.nanmean(array[indices], axis=0))
    return pd.DataFrame(rows), {name: np.stack(values) for name, values in profiles.items()}


def make_chemical_map():
    path = RESULTS_DIR / "chemical_split" / "chemical_split_structure_audit.csv"
    audit = pd.read_csv(path, dtype={'compound_id': str}).dropna(subset=['chemical_group'])
    if (audit.groupby('compound_id').chemical_group.nunique() > 1).any():
        raise ValueError('Chemical split audit has conflicting compound groups')
    return audit.drop_duplicates('compound_id').set_index('compound_id').chemical_group.astype(str).to_dict()


def load_external_labels():
    parquet = PUBLIC_DIR / 'compound_annotations.parquet'
    if not parquet.exists():
        raise FileNotFoundError(f'{parquet}; first run analysis/import_public_annotations.py')
    public = pd.read_parquet(parquet)
    if public.compound_id.astype(str).duplicated().any():
        raise ValueError('Public annotations have duplicate local compound IDs')
    return public


def external_splits(meta, protocol, chemical_map):
    """Group exact chemical identities and exclude cross-batch identity leakage."""
    ids = meta.compound_id.to_numpy(str)
    identities = meta.identity_group.to_numpy(str)
    if protocol == 'held_out_compound':
        units = identities
        if len(np.unique(units)) < 5:
            raise ValueError('Fewer than five independent chemical identities')
        splits = list(GroupKFold(5).split(meta, groups=units))
    elif protocol == 'held_out_chemical_group':
        keep = np.array([cid in chemical_map for cid in ids])
        meta = meta.loc[keep].reset_index(drop=True)
        ids = meta.compound_id.to_numpy(str)
        identities = meta.identity_group.to_numpy(str)
        units = np.array([chemical_map[cid] for cid in ids])
        if len(np.unique(units)) < 5:
            raise ValueError('Fewer than five chemical groups remain')
        splits = list(GroupKFold(5).split(meta, groups=units))
    elif protocol == 'held_out_batch':
        units = identities
        batches = meta.batch.to_numpy()
        splits = []
        for batch in sorted(pd.unique(batches)):
            test = np.flatnonzero(batches == batch)
            test_identities = set(identities[test])
            train = np.flatnonzero((batches != batch) & ~np.isin(identities, list(test_identities)))
            if len(train) and len(test):
                splits.append((train, test))
    else:
        raise ValueError(f'Unknown protocol: {protocol}')
    for train, test in splits:
        if set(identities[train]) & set(identities[test]):
            raise AssertionError(f'Chemical identity leakage in {protocol}')
        if protocol == 'held_out_chemical_group' and set(units[train]) & set(units[test]):
            raise AssertionError('Chemical group leakage')
    return meta, units, splits


def consensus_identity_labels(public, column):
    """Collapse exact parent-structure aliases; discard contradictory labels."""
    table = public[['identity_group', column]].copy()
    table[column] = pd.to_numeric(table[column], errors='coerce')
    table = table.dropna(subset=['identity_group', column])
    rows, conflict_count = [], 0
    for identity, group in table.groupby('identity_group', sort=True):
        values = sorted(set(group[column].astype(int)))
        if len(values) != 1:
            conflict_count += 1
            continue
        rows.append((str(identity), int(values[0])))
    return pd.DataFrame(rows, columns=['identity_group', 'label']), conflict_count


def split_broad_labels(value):
    if not isinstance(value, str) or not value.strip():
        return ()
    parts = re.split(r'\s*[|;]\s*', value)
    unique = {}
    for part in parts:
        name = ' '.join(part.split()).strip()
        if name:
            unique.setdefault(name.casefold(), name)
    return tuple(unique[key] for key in sorted(unique))


def broad_neighbor_retrieval(meta_all, arrays_all, public, chemical_map):
    rows = []
    label_support = []
    for column, task in [('broad_moa', 'Broad Repurposing Hub MOA'),
                         ('broad_target', 'Broad Repurposing Hub target')]:
        selected = public.loc[public.broad_matched.astype(bool), ['compound_id', 'identity_group', column]].copy()
        selected['labels'] = selected[column].map(split_broad_labels)
        selected = selected[selected.labels.map(bool)]
        # Preserve aliases in metadata, but use one multilabel row per standardized identity.
        identity_rows = []
        for identity, group in selected.groupby('identity_group', sort=True):
            merged_labels = tuple(sorted(set(label for labels in group.labels for label in labels), key=str.casefold))
            identity_rows.append(dict(identity_group=str(identity), labels=merged_labels))
        id_frame = pd.DataFrame(identity_rows)
        counts = {}
        for record in identity_rows:
            for label in record['labels']:
                counts[label] = counts.get(label, 0) + 1
        eligible_labels = sorted(label for label, count in counts.items() if count >= MIN_RETRIEVAL_SUPPORT)
        label_support.extend(dict(task=task, label=label, n_positive_identities=counts[label],
                                  evaluation='positive_label_neighbor_retrieval',
                                  negative_labels='not defined; missing annotations are unlabelled')
                             for label in sorted(counts, key=str.casefold))
        if not eligible_labels:
            continue
        identity_rows = [row for row in identity_rows if row['identity_group'] in set(id_frame.identity_group)]
        label_map = {row['identity_group']: row['labels'] for row in identity_rows}
        known = meta_all.identity_group.isin(label_map).to_numpy()
        meta = meta_all.loc[known].copy().reset_index(drop=True)
        inputs_all = {name: values[known] for name, values in arrays_all.items()}
        meta['labels'] = meta.identity_group.map(label_map)
        # Remove labels below support before forming the binary relevance matrix.
        meta['labels'] = meta.labels.map(lambda vals: tuple(x for x in vals if x in set(eligible_labels)))
        keep = meta.labels.map(bool).to_numpy()
        meta = meta.loc[keep].reset_index(drop=True)
        inputs_all = {name: values[keep] for name, values in inputs_all.items()}
        ids = meta.identity_group.to_numpy(str)
        y = np.array([[label in labels for label in eligible_labels] for labels in meta.labels], dtype=int)
        chemical_units = np.array([chemical_map.get(cid, 'unresolved:' + cid) for cid in meta.compound_id])
        for protocol in PROTOCOLS:
            current, inputs = meta, inputs_all
            if protocol == 'held_out_chemical_group':
                keep = current.compound_id.isin(chemical_map).to_numpy()
                current = current.loc[keep].reset_index(drop=True)
                inputs = {name: values[keep] for name, values in inputs.items()}
            current, units, splits = external_splits(current, protocol, chemical_map)
            meta_current = current.reset_index(drop=True)
            ids_current = meta_current.identity_group.to_numpy(str)
            labels_current = np.array([[label in values for label in eligible_labels] for values in meta_current.labels], dtype=int)
            for feature, x in inputs.items():
                for fold, (train, test) in enumerate(splits):
                    transformed_train, transformed_test = transform(x[train], x[test])
                    train_profiles = pd.DataFrame(transformed_train).assign(identity_group=ids_current[train]).groupby('identity_group', sort=True).mean()
                    test_profiles = pd.DataFrame(transformed_test).assign(identity_group=ids_current[test]).groupby('identity_group', sort=True).mean()
                    truth_by_identity = pd.DataFrame(labels_current).assign(identity_group=ids_current).groupby('identity_group', sort=True).max()
                    sim = transformed_similarity(test_profiles.to_numpy(), train_profiles.to_numpy())
                    k = min(5, len(train_profiles))
                    nn = np.argsort(-sim, axis=1, kind='stable')[:, :k]
                    train_truth = truth_by_identity.loc[train_profiles.index].to_numpy()
                    test_truth = truth_by_identity.loc[test_profiles.index].to_numpy()
                    for j, label in enumerate(eligible_labels):
                        query = test_truth[:, j] == 1
                        if not query.any():
                            continue
                        retrieved = train_truth[nn[query], j].mean()
                        prevalence = train_truth[:, j].mean()
                        rows.append(dict(task=task, protocol=protocol, feature=feature, fold=fold,
                                         label=label, k=k, n_positive_queries=int(query.sum()),
                                         neighbor_precision_at_k=float(retrieved),
                                         training_label_prevalence=float(prevalence),
                                         precision_lift=float(retrieved-prevalence)))
    if not rows:
        return pd.DataFrame(), pd.DataFrame(label_support), pd.DataFrame()
    per_fold = pd.DataFrame(rows)
    summary = per_fold.groupby(['task','protocol','feature','label']).apply(
        lambda g: pd.Series(dict(
            n_positive_queries=int(g.n_positive_queries.sum()),
            neighbor_precision_at_k=float(np.average(g.neighbor_precision_at_k, weights=g.n_positive_queries)),
            training_label_prevalence=float(np.average(g.training_label_prevalence, weights=g.n_positive_queries)),
            precision_lift=float(np.average(g.precision_lift, weights=g.n_positive_queries)),
            n_folds=len(g))), include_groups=False).reset_index()
    return summary, pd.DataFrame(label_support), per_fold


def transformed_similarity(test, train):
    from sklearn.preprocessing import normalize
    return normalize(test) @ normalize(train).T


def plot_summary(binary, broad, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path as P
    path = P(path)
    if binary.empty:
        return
    subset = binary[(binary.feature == 'pca_normalized') & (binary.model == 'linear')]
    names = subset.groupby('endpoint').n_positive.min().sort_values(ascending=False).index.tolist()
    order = names[:6] + names[-6:]
    protocols = list(PROTOCOLS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for ax, protocol in zip(axes, protocols):
        part = subset[subset.protocol == protocol].set_index('endpoint').reindex(order)
        vals = part.ap_delta.to_numpy()
        low, high = part.ap_delta_ci_low.to_numpy(), part.ap_delta_ci_high.to_numpy()
        y = np.arange(len(order))
        ax.hlines(y, low, high, color='#3274a1')
        ax.scatter(vals, y, color='#3274a1', s=22)
        ax.axvline(0, color='black', linewidth=.8)
        ax.set_title(protocol.replace('held_out_', 'Unseen ').replace('_', ' '))
        ax.set_xlabel('AP above prevalence baseline')
        ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(range(len(order)), [o.replace('tox21__','Tox21: ').replace('dili__','DILIrank: ') for o in order])
    fig.suptitle('External measured bioactivity and DILI annotations\nNormalized-PCA linear probe; nominal 95% compound/chemical bootstrap intervals')
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--protocols', nargs='+', choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.bootstrap < 0 or args.workers < 1:
        parser.error('bootstrap must be nonnegative and workers positive')
    args.output.mkdir(parents=True, exist_ok=True)
    run_config = dict(
        evaluator='evaluate_public_annotations.py',
        protocols=args.protocols,
        bootstrap_draws=args.bootstrap,
        workers=args.workers,
        random_seed=SEED,
        endpoints=TOX_ENDPOINTS + DILI_ENDPOINTS,
        features=list(EMBEDDINGS) + ['assay_features', 'damage_assays'],
        minimum_class_support_per_class=MIN_CLASS_SUPPORT,
        minimum_broad_label_support=MIN_RETRIEVAL_SUPPORT,
        public_data_manifest='data/public_annotations/manifest.json',
        profile_data=str(DATA_PATH.relative_to(PROJECT_ROOT)),
    )
    (args.output / 'run_config.json').write_text(json.dumps(run_config, indent=2) + '\n')
    public = load_external_labels()
    meta, arrays = load_profiles()
    identity = public[['compound_id','identity_group']].copy()
    meta = meta.merge(identity, on='compound_id', how='left', validate='many_to_one')
    if meta.identity_group.isna().any():
        raise ValueError('Missing local identity mapping for a profiled compound')
    chemical_map = make_chemical_map()
    binary_summary, binary_folds, support, predictions = [], [], [], []
    endpoints = [(column, 'Tox21') for column in TOX_ENDPOINTS]
    endpoints += [(column, 'FDA DILIrank 2.0') for column in DILI_ENDPOINTS]
    variants = dict(arrays)
    damage = meta[DAMAGE_COLUMNS].to_numpy(dtype=float)
    variants['damage_assays'] = damage
    for name, values in arrays.items():
        variants[name + '_plus_damage_assays'] = np.column_stack([values, damage])
    for endpoint, source in endpoints:
        print(f'Evaluating {source}: {endpoint}', flush=True)
        for protocol in args.protocols:
            # Recreate endpoint-specific arrays on the same ordered profiles.
            label_table, conflicts = consensus_identity_labels(public, endpoint)
            outcome = label_table.set_index('identity_group').label
            selected_ids = set(outcome.index)
            known = meta.identity_group.isin(selected_ids).to_numpy()
            task_meta = meta.loc[known].copy().reset_index(drop=True)
            task_meta['label'] = task_meta.identity_group.map(outcome).astype(int)
            task_arrays = {name: value[known] for name, value in variants.items()}
            # Chemical holdout removes unresolved identities; filter feature rows at the same time.
            if protocol == 'held_out_chemical_group':
                keep = task_meta.compound_id.isin(chemical_map).to_numpy()
                task_meta = task_meta.loc[keep].reset_index(drop=True)
                task_arrays = {name: value[keep] for name, value in task_arrays.items()}
            task_meta, units, splits = external_splits(task_meta, protocol, chemical_map)
            y = task_meta.label.to_numpy(int)
            identity_ids = task_meta.identity_group.to_numpy(str)
            bootstrap_units = np.array([chemical_map.get(cid, 'unresolved:' + cid)
                                        for cid in task_meta.compound_id]) if protocol == 'held_out_chemical_group' else identity_ids
            npos = int(task_meta.drop_duplicates('identity_group').label.sum())
            nneg = int(task_meta.drop_duplicates('identity_group').label.eq(0).sum())
            support.append(dict(source=source, endpoint=endpoint, protocol=protocol,
                                n_independent_identities=task_meta.identity_group.nunique(),
                                n_positive=npos, n_negative=nneg, n_identity_conflicts=conflicts,
                                included=npos >= MIN_CLASS_SUPPORT and nneg >= MIN_CLASS_SUPPORT,
                                reason='eligible' if npos >= MIN_CLASS_SUPPORT and nneg >= MIN_CLASS_SUPPORT
                                else f'fewer than {MIN_CLASS_SUPPORT} positive or negative chemical identities'))
            if npos < MIN_CLASS_SUPPORT or nneg < MIN_CLASS_SUPPORT:
                continue
            task_variants = {name:value for name,value in task_arrays.items()
                             if name in arrays or name == 'damage_assays' or name.endswith('_plus_damage_assays')}
            for feature, x in task_variants.items():
                model_names = ['linear'] if feature == 'damage_assays' or feature.endswith('_plus_damage_assays') else ['linear','rbf']
                for model_name in model_names:
                    frames = []
                    print(f'  {protocol} {feature} {model_name}', flush=True)
                    for fold,(train,test) in enumerate(splits):
                        scores = predict_external(x[train], y[train], x[test], identity_ids[train], model_name, args.workers)
                        baseline = np.average(y[train], weights=equal_compound_weights(identity_ids[train]))
                        frame = pd.DataFrame(dict(source=source, endpoint=endpoint, protocol=protocol,
                            feature=feature, model=model_name, compound_id=task_meta.compound_id.iloc[test].to_numpy(),
                            identity_group=identity_ids[test], bootstrap_unit=bootstrap_units[test], fold=fold,
                            truth=y[test], prediction=scores, baseline=baseline,
                            n_train_positive_identities=len(set(identity_ids[train][y[train] == 1])),
                            n_train_negative_identities=len(set(identity_ids[train][y[train] == 0]))))
                        frames.append(frame)
                        pooled_fold = frame.groupby('identity_group',sort=True).agg(
                            truth=('truth','max'),prediction=('prediction','mean'),baseline=('baseline','mean'),
                            bootstrap_unit=('bootstrap_unit','first'))
                        if pooled_fold.truth.nunique() == 2:
                            binary_folds.append(dict(source=source,endpoint=endpoint,protocol=protocol,
                                feature=feature,model=model_name,fold=fold,
                                **annotation_metrics(pooled_fold.truth.to_numpy(),pooled_fold.prediction.to_numpy(),
                                    pooled_fold.baseline.to_numpy(),pooled_fold.bootstrap_unit.to_numpy(),0)))
                    oof = pd.concat(frames,ignore_index=True)
                    predictions.append(oof)
                    pooled = oof.groupby('identity_group',sort=True).agg(
                        truth=('truth','max'),prediction=('prediction','mean'),baseline=('baseline','mean'),
                        bootstrap_unit=('bootstrap_unit','first'))
                    binary_summary.append(dict(source=source,endpoint=endpoint,protocol=protocol,
                        feature=feature,model=model_name,
                        **annotation_metrics(pooled.truth.to_numpy(),pooled.prediction.to_numpy(),
                            pooled.baseline.to_numpy(),pooled.bootstrap_unit.to_numpy(),args.bootstrap)))
    binary = pd.DataFrame(binary_summary)
    binary.to_csv(args.output/'binary_scores.csv',index=False)
    pd.DataFrame(binary_folds).to_csv(args.output/'binary_folds.csv',index=False)
    pd.DataFrame(support).to_csv(args.output/'endpoint_support.csv',index=False)
    if predictions:
        pd.concat(predictions,ignore_index=True).to_parquet(args.output/'binary_oof_predictions.parquet',index=False)
    broad_summary,broad_support,broad_folds = broad_neighbor_retrieval(meta,arrays,public,chemical_map)
    broad_summary.to_csv(args.output/'broad_label_neighbor_summary.csv',index=False)
    broad_support.to_csv(args.output/'broad_label_support.csv',index=False)
    broad_folds.to_csv(args.output/'broad_label_neighbor_folds.csv',index=False)
    plot_summary(binary,broad_summary,args.output/'public_annotation_performance.png')
    print(f'Wrote public annotation evaluations to {args.output}',flush=True)


def predict_external(x_train,y_train,x_test,groups,model_name,workers):
    from evaluate_drug_attributes import predict_labels
    return predict_labels(x_train,y_train[:,None],x_test,equal_compound_weights(groups),model_name=='rbf',workers)[:,0]


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
