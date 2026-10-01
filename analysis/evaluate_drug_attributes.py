"""Attribute-level biological probes with compound, batch and chemical holdouts.

Fixed, prespecified probes (no outer-test model selection): ridge/logistic and
an approximate RBF probe. New preprocessing is fitted only on training data.
Run: uv run python analysis/evaluate_drug_attributes.py
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

from annotation_model_utils import DATA_PATH, RESULTS_DIR, parse_labels, unpack_vector

EMBEDDINGS = {
    'pca_normalized': 'pca_embedding_normalized',
    'pca_raw': 'pca_embedding_raw',
    'brightfield': 'brightfield',
}
TARGETS = [
    'mean_nuclei_count', 'mean_nuclei_area_mean', 'vacuole_ratio',
    'mito_puncta_ratio', 'ros_stress_granules_sum_mean_norm_ridge_norm_corr',
    'ldh_ridge_norm', 'mtt_ridge_norm',
]
DAMAGE = ['mean_nuclei_count', 'ldh_ridge_norm', 'mtt_ridge_norm']
PROTOCOLS = ['held_out_compound', 'held_out_batch', 'held_out_chemical_group']
SEED = 42


def equal_compound_weights(ids):
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse]
    return w / w.mean()


def load_data(path, features):
    raw = pd.read_parquet(path)
    df = raw.loc[raw.compound_id.notna()].copy().reset_index(drop=True)
    df['compound_id'] = df.compound_id.astype(str)
    # A compound's annotation must be consistent before any aggregation.
    for col in ['compound_pathway', 'compound_target']:
        conflicts = df.groupby('compound_id')[col].nunique(dropna=True)
        if (conflicts > 1).any():
            raise ValueError(f'Conflicting {col} annotations for {list(conflicts[conflicts > 1].index)}')
    arrays = {name: np.stack(df[col].map(unpack_vector)).astype(float)
              for name, col in EMBEDDINGS.items() if name in features}
    for values in arrays.values():
        values[~np.isfinite(values)] = np.nan
    return raw, df, arrays


def aggregate(df, arrays, keys):
    working = df.copy()
    numeric = [c for c in TARGETS + ['compound_concentration_um'] if c not in keys]
    working[numeric] = working[numeric].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    groups = working.groupby(keys, dropna=False, sort=True)
    meta = groups[numeric].mean()
    for col in ['compound_pathway', 'compound_target']:
        meta[col] = groups[col].first()
    meta['n_wells'] = groups.size()
    # Group on the identical keys/index for every matrix, retaining missing dose.
    rows = {}
    for name, values in arrays.items():
        frame = pd.DataFrame(values)
        for key in keys:
            frame[key] = working[key].to_numpy()
        rows[name] = frame.groupby(keys, dropna=False, sort=True).mean().reindex(meta.index).to_numpy()
    return meta.reset_index(), rows


def chemical_mapping(path):
    if not path.exists():
        raise FileNotFoundError(f'{path}: run compare_annotations_chemical_split.py first, or omit chemical protocol')
    d = pd.read_csv(path, dtype={'compound_id': str}).dropna(subset=['chemical_group'])
    if (d.groupby('compound_id').chemical_group.nunique() > 1).any():
        raise ValueError('Conflicting chemical groups')
    return d.drop_duplicates('compound_id').set_index('compound_id').chemical_group.astype(str).to_dict()


def split_data(meta, arrays, protocol, chemical_map, n_splits=5):
    keep = meta.compound_id.isin(chemical_map) if protocol == 'held_out_chemical_group' else np.ones(len(meta), bool)
    current = meta.loc[keep].reset_index(drop=True)
    inputs = {k: x[np.asarray(keep)] for k, x in arrays.items()}
    ids = current.compound_id.to_numpy(str)
    units = current.compound_id.map(chemical_map).to_numpy(str) if protocol == 'held_out_chemical_group' else ids
    if protocol == 'held_out_batch':
        batches = current.batch.to_numpy()
        splits = []
        for batch in sorted(pd.unique(batches)):
            test = np.flatnonzero(batches == batch)
            train = np.flatnonzero((batches != batch) & ~np.isin(ids, ids[test]))
            if len(train) and len(test):
                splits.append((train, test))
    else:
        if len(np.unique(units)) < n_splits:
            raise ValueError(f'{protocol}: fewer than {n_splits} independent groups')
        splits = list(GroupKFold(n_splits).split(current, groups=units))
    if not splits:
        raise ValueError(f'No valid folds for {protocol}')
    for train, test in splits:
        assert not set(ids[train]) & set(ids[test])
        if protocol == 'held_out_chemical_group':
            assert not set(units[train]) & set(units[test])
    return current, inputs, units, splits


def transform(train, test, nonlinear=False):
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = StandardScaler()
    a = scaler.fit_transform(imputer.fit_transform(train))
    b = scaler.transform(imputer.transform(test))
    if nonlinear:
        kernel = RBFSampler(gamma=1.0 / max(1, a.shape[1]), n_components=256, random_state=SEED)
        a, b = kernel.fit_transform(a), kernel.transform(b)
        # Give ridge's fixed alpha comparable feature scale across probes.
        scaler = StandardScaler()
        a, b = scaler.fit_transform(a), scaler.transform(b)
    return a, b


def bootstrap_counts(units, n_bootstrap):
    unique, inverse = np.unique(units, return_inverse=True)
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=n_bootstrap)
    return counts, inverse


def interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return tuple(np.quantile(finite, [.025, .975])) if len(finite) else (np.nan, np.nan)


def regression_metrics(frame, n_bootstrap):
    y, p, b = [frame[c].to_numpy(float) for c in ['truth', 'prediction', 'baseline']]
    w = equal_compound_weights(frame.compound_id.to_numpy())
    total = w.sum()
    sst = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    error, base_error = w * (y-p)**2, w * (y-b)**2
    r2 = 1-error.sum()/sst if sst > 0 else np.nan
    skill = 1-error.sum()/base_error.sum() if base_error.sum() > 0 else np.nan
    # Weighted rank correlation: each compound has equal total influence.
    ry, rp = rankdata(y), rankdata(p)
    ry -= np.average(ry, weights=w)
    rp -= np.average(rp, weights=w)
    denom = np.sqrt(np.sum(w*ry**2)*np.sum(w*rp**2))
    rho = np.sum(w*ry*rp)/denom if denom else np.nan
    counts, inverse = bootstrap_counts(frame.bootstrap_unit.to_numpy(), n_bootstrap)
    moments = np.stack([w, w*y, w*y*y, error, base_error], axis=1)
    sums = np.zeros((counts.shape[1], 5))
    np.add.at(sums, inverse, moments)
    draws = counts @ sums
    with np.errstate(divide='ignore', invalid='ignore'):
        draw_r2 = 1 - draws[:, 3] / (draws[:, 2] - draws[:, 1]**2 / draws[:, 0])
        draw_skill = 1 - draws[:, 3] / draws[:, 4]
    rlo, rhi = interval(draw_r2)
    slo, shi = interval(draw_skill)
    return dict(r2=r2, r2_ci_low=rlo, r2_ci_high=rhi, spearman=rho,
                mse=error.sum()/total, baseline_mse=base_error.sum()/total,
                mse_skill=skill, mse_skill_ci_low=slo, mse_skill_ci_high=shi,
                n_compounds=frame.compound_id.nunique(), n_profiles=len(frame))


def run_regression(meta, arrays, args, chemical_map):
    summaries, folds, coverage = [], [], []
    prediction_dir = args.output / 'regression_predictions'
    prediction_dir.mkdir(exist_ok=True)
    for protocol in args.protocols:
        d, inputs, units, splits = split_data(meta, arrays, protocol, chemical_map)
        ids = d.compound_id.to_numpy(str)
        dose = d.compound_concentration_um.to_numpy(float)
        # log1p keeps zero-dose profiles and marks invalid negative values missing.
        dose = np.log1p(np.where(dose >= 0, dose, np.nan))[:, None]
        variants = {'dose_only': dose}
        for name, x in inputs.items():
            variants[name] = x
            variants[name + '_plus_dose'] = np.column_stack([x, dose])
        if getattr(args, 'continuous_inputs', None):
            variants = {k: v for k, v in variants.items() if k in args.continuous_inputs}
            if not variants:
                raise ValueError('No requested continuous inputs are available with selected features')
        for target in TARGETS:
            valid = np.isfinite(pd.to_numeric(d[target], errors='coerce').to_numpy(float))
            coverage.append(dict(protocol=protocol, target=target, n_profiles=len(d),
                                 n_observed=int(valid.sum()), n_compounds=d.loc[valid].compound_id.nunique(),
                                 n_compounds_excluded_by_protocol=meta.compound_id.nunique()-d.compound_id.nunique()))
        for name, x in variants.items():
            for model_name in args.models:
                print(f'Regression {protocol} {name} {model_name}', flush=True)
                records = []
                for fold, (train, test) in enumerate(splits):
                    # Fit preprocessing once on the outer training features. Group
                    # targets with identical observed-training masks for multioutput
                    # ridge; this is algebraically identical to independent fits.
                    a, b = transform(x[train], x[test], model_name == 'rbf')
                    outcomes = d[TARGETS].apply(pd.to_numeric, errors='coerce').to_numpy(float)
                    mask_groups = {}
                    for j in range(len(TARGETS)):
                        mask = np.isfinite(outcomes[train, j])
                        mask_groups.setdefault(mask.tobytes(), (mask, []))[1].append(j)
                    for observed, columns in mask_groups.values():
                        tr = train[observed]
                        if len(tr) < 2:
                            continue
                        weights = equal_compound_weights(ids[tr])
                        values = outcomes[np.ix_(tr, columns)]
                        model = Ridge(alpha=100.0).fit(a[observed], values, sample_weight=weights)
                        predictions = np.asarray(model.predict(b)).reshape(len(test), len(columns))
                        baselines = np.average(values, axis=0, weights=weights)
                        for pos, j in enumerate(columns):
                            valid_test = np.isfinite(outcomes[test, j])
                            te = test[valid_test]
                            if not len(te):
                                continue
                            target = TARGETS[j]
                            frame = pd.DataFrame(dict(compound_id=ids[te], bootstrap_unit=units[te],
                                                      batch=d.batch.iloc[te].to_numpy(),
                                                      dose=d.compound_concentration_um.iloc[te].to_numpy(),
                                                      fold=fold, target=target, truth=outcomes[te, j],
                                                      prediction=predictions[valid_test, pos], baseline=baselines[pos]))
                            records.append(frame)
                            folds.append(dict(protocol=protocol, feature=name, model=model_name, target=target,
                                              fold=fold, **regression_metrics(frame, 0)))
                if not records:
                    continue
                predictions = pd.concat(records, ignore_index=True)
                predictions.to_parquet(prediction_dir / f'{protocol}__{name}__{model_name}.parquet', index=False)
                for target, frame in predictions.groupby('target'):
                    summaries.append(dict(protocol=protocol, feature=name, model=model_name, target=target,
                                          **regression_metrics(frame, args.bootstrap)))
                pd.DataFrame(summaries).to_csv(args.output / 'continuous_scores.csv', index=False)
    pd.DataFrame(folds).to_csv(args.output / 'continuous_folds.csv', index=False)
    pd.DataFrame(coverage).to_csv(args.output / 'continuous_coverage.csv', index=False)


def weighted_ap(y, score, weights):
    """AP for many bootstrap weight vectors; handles tied scores as sklearn does."""
    order = np.argsort(-score, kind='stable')
    truth = y[order]
    w = np.atleast_2d(weights).astype(float)[:, order]
    ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(y)-1]
    tp = np.cumsum(w * truth, axis=1)[:, ends]
    seen = np.cumsum(w, axis=1)[:, ends]
    increment = np.diff(np.column_stack([np.zeros(len(w)), tp]), axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
        ap = (precision * increment).sum(axis=1) / tp[:, -1]
    return ap


def annotation_metrics(y, prediction, baseline, units, n_bootstrap):
    counts, inverse = bootstrap_counts(units, n_bootstrap)
    weights = np.vstack([np.ones(len(y)), counts[:, inverse]])
    ap = weighted_ap(y, prediction, weights)
    base = weighted_ap(y, baseline, weights)
    low, high = interval(ap[1:])
    dlo, dhi = interval((ap-base)[1:])
    return dict(average_precision=ap[0], baseline_ap=base[0], ap_delta=ap[0]-base[0],
                ap_ci_low=low, ap_ci_high=high, ap_delta_ci_low=dlo, ap_delta_ci_high=dhi,
                prevalence=float(y.mean()), n_positive=int(y.sum()), n_compounds=len(y))


def predict_labels(a, y, b, weights, nonlinear, workers=4):
    a, b = transform(a, b, nonlinear)
    scores = np.zeros((len(b), y.shape[1]))
    def fit_label(j):
        if len(np.unique(y[:, j])) < 2:
            return j, np.full(len(b), y[0, j], dtype=float)
        model = LogisticRegression(C=.1, class_weight='balanced', solver='liblinear', max_iter=1000, random_state=SEED)
        model.fit(a, y[:, j], sample_weight=weights)
        return j, model.predict_proba(b)[:, 1]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for j, prediction in pool.map(fit_label, range(y.shape[1])):
            scores[:, j] = prediction
    return scores


def run_annotations(meta, arrays, args, chemical_map):
    summaries, folds, coverage, retrieval, support_rows = [], [], [], [], []
    prediction_dir = args.output / 'annotation_predictions'
    prediction_dir.mkdir(exist_ok=True)
    for task, column in [('pathway', 'compound_pathway'), ('target', 'compound_target')]:
        known = meta[column].notna() & meta[column].astype(str).str.strip().ne('')
        base = meta.loc[known].reset_index(drop=True)
        base_inputs = {k: x[known.to_numpy()] for k, x in arrays.items()}
        for protocol in args.protocols:
            d, inputs, units, splits = split_data(base, base_inputs, protocol, chemical_map)
            ids = d.compound_id.to_numpy(str)
            label_sets = d[column].map(parse_labels)
            # Prespecified evaluation support filter, not a tuned model decision.
            all_labels = sorted(set().union(*label_sets))
            labels = []
            for label in all_labels:
                positive = np.array([label in values for values in label_sets])
                count = len(np.unique(ids[positive]))
                support_rows.append(dict(task=task, protocol=protocol, label=label, n_positive=count,
                                         evaluated=count >= 6, reason='eligible' if count >= 6 else 'fewer than 6 positive compounds'))
                if count >= 6:
                    labels.append(label)
            y = np.array([[label in values for label in labels] for values in label_sets], dtype=int)
            damage = d[DAMAGE].to_numpy(float)
            variants = {'damage_assays': damage, **inputs}
            variants.update({k+'_plus_damage_assays': np.column_stack([x, damage]) for k, x in inputs.items()})
            coverage.append(dict(task=task, protocol=protocol, n_compounds=d.compound_id.nunique(),
                                 n_labels=len(labels), n_missing_annotation_compounds=meta.compound_id.nunique()-base.compound_id.nunique(),
                                 n_structure_excluded_compounds=base.compound_id.nunique()-d.compound_id.nunique()))
            for name, x in variants.items():
                for model_name in (['linear'] if name == 'damage_assays' or name.endswith('_plus_damage_assays') else args.models):
                    print(f'Annotation {task} {protocol} {name} {model_name} ({len(labels)} labels)', flush=True)
                    records = []
                    for fold, (train, test) in enumerate(splits):
                        weights = equal_compound_weights(ids[train])
                        score = predict_labels(x[train], y[train], x[test], weights, model_name == 'rbf', getattr(args, 'workers', 4))
                        baseline = np.average(y[train], axis=0, weights=weights)
                        for j, label in enumerate(labels):
                            frame = pd.DataFrame(dict(compound_id=ids[test], bootstrap_unit=units[test], fold=fold,
                                                      label=label, truth=y[test, j], prediction=score[:, j], baseline=baseline[j]))
                            records.append(frame)
                            # Macro across folds can differ from pooled OOF AP: retain both evidence levels.
                            pooled = frame.groupby('compound_id', sort=True).agg(
                                truth=('truth', 'max'), prediction=('prediction', 'mean'), baseline=('baseline', 'mean'))
                            if pooled.truth.nunique() == 2:
                                folds.append(dict(task=task, protocol=protocol, feature=name, model=model_name,
                                                  label=label, fold=fold, n_train_positive_compounds=len(np.unique(ids[train][y[train, j] == 1])),
                                                  **annotation_metrics(pooled.truth.to_numpy(), pooled.prediction.to_numpy(),
                                                                       pooled.baseline.to_numpy(), pooled.index.to_numpy(), 0)))
                        if name in inputs and model_name == 'linear':
                            retrieval.extend(retrieve_labels(x, y, ids, train, test, labels, task, protocol, name, fold))
                    predictions = pd.concat(records, ignore_index=True)
                    predictions.to_parquet(prediction_dir / f'{task}__{protocol}__{name}__{model_name}.parquet', index=False)
                    for label, frame in predictions.groupby('label'):
                        pooled = frame.groupby('compound_id', sort=True).agg(
                            truth=('truth', 'max'), prediction=('prediction', 'mean'), baseline=('baseline', 'mean'),
                            bootstrap_unit=('bootstrap_unit', 'first'))
                        if pooled.truth.nunique() < 2:
                            continue
                        summaries.append(dict(task=task, protocol=protocol, feature=name, model=model_name, label=label,
                                              **annotation_metrics(pooled.truth.to_numpy(), pooled.prediction.to_numpy(),
                                                                   pooled.baseline.to_numpy(), pooled.bootstrap_unit.to_numpy(), args.bootstrap)))
                    pd.DataFrame(summaries).to_csv(args.output / 'annotation_scores.csv', index=False)
    pd.DataFrame(folds).to_csv(args.output / 'annotation_folds.csv', index=False)
    pd.DataFrame(coverage).to_csv(args.output / 'annotation_coverage.csv', index=False)
    pd.DataFrame(retrieval).to_csv(args.output / 'annotation_retrieval_folds.csv', index=False)
    pd.DataFrame(support_rows).to_csv(args.output / 'label_support.csv', index=False)


def retrieve_labels(x, y, ids, train, test, labels, task, protocol, feature, fold):
    a, b = transform(x[train], x[test])
    train_df = pd.DataFrame(a).assign(compound_id=ids[train]).groupby('compound_id', sort=True).mean()
    test_df = pd.DataFrame(b).assign(compound_id=ids[test]).groupby('compound_id', sort=True).mean()
    truth = pd.DataFrame(y).assign(compound_id=ids).groupby('compound_id', sort=True).max()
    sims = normalize(test_df.to_numpy()) @ normalize(train_df.to_numpy()).T
    k = min(5, len(train_df))
    neighbors = np.argsort(-sims, axis=1, kind='stable')[:, :k]
    train_truth = truth.loc[train_df.index].to_numpy()
    test_truth = truth.loc[test_df.index].to_numpy()
    rows = []
    for j, label in enumerate(labels):
        positive = test_truth[:, j] == 1
        if not positive.any():
            continue
        precision = train_truth[neighbors[positive], j].mean()
        baseline = train_truth[:, j].mean()
        rows.append(dict(task=task, protocol=protocol, feature=feature, fold=fold, label=label,
                         k=k, n_positive_queries=int(positive.sum()), precision_at_5=precision,
                         random_neighbor_precision=baseline, precision_delta=precision-baseline))
    return rows


def run_replicates(df, arrays, args):
    """Descriptive raw-coordinate cosine retrieval; eligible cross-plate gallery only."""
    rows = []
    ids = df.compound_id.to_numpy(str)
    dose = df.compound_concentration_um.to_numpy(float)
    plates, batches = df.plate.to_numpy(), df.batch.to_numpy()
    for name, x in arrays.items():
        good = np.isfinite(x).all(axis=1) & np.isfinite(dose) & (np.linalg.norm(np.nan_to_num(x), axis=1) > 0)
        norm = normalize(np.nan_to_num(x))
        for pool in ['different_plate', 'different_batch']:
            # Eligibility needs at least two technical groups for a compound-dose
            # pair; group once rather than scanning every possible well pair.
            eligible = pd.DataFrame(dict(compound=ids[good], dose=dose[good],
                technical_group=(plates if pool == 'different_plate' else batches)[good]),
                index=np.flatnonzero(good))
            group_counts = eligible.groupby(['compound', 'dose']).technical_group.transform('nunique')
            candidates = eligible.index[group_counts > 1].to_numpy()
            # Match query selection across embeddings when eligibility agrees.
            rng = np.random.default_rng(SEED)
            queries = rng.choice(candidates, min(args.max_queries, len(candidates)), replace=False)
            for i in queries:
                gallery = np.flatnonzero(good & ((plates != plates[i]) if pool == 'different_plate' else (batches != batches[i])))
                relevant = (ids[gallery] == ids[i]) & (dose[gallery] == dose[i])
                ranked = np.argsort(-(norm[gallery] @ norm[i]), kind='stable')[:5]
                rows.append(dict(feature=name, pool=pool, compound_id=ids[i], dose=dose[i],
                                 query_well=str(df.well_id.iloc[i]), n_gallery=len(gallery),
                                 n_relevant=int(relevant.sum()), eligible_queries=len(candidates),
                                 hit_at_1=int(relevant[ranked[0]]), precision_at_5=relevant[ranked].mean(),
                                 random_precision=relevant.mean()))
    pd.DataFrame(rows).to_csv(args.output / 'replicate_queries.csv', index=False)
    if rows:
        per_compound = pd.DataFrame(rows).groupby(['feature', 'pool', 'compound_id']).agg(
            hit_at_1=('hit_at_1', 'mean'), precision_at_5=('precision_at_5', 'mean'), random_precision=('random_precision', 'mean')).reset_index()
        per_compound.groupby(['feature', 'pool']).agg(
            n_compounds=('compound_id', 'nunique'), hit_at_1=('hit_at_1', 'mean'),
            precision_at_5=('precision_at_5', 'mean'), random_precision=('random_precision', 'mean')).reset_index().to_csv(args.output / 'replicate_summary.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA_PATH)
    parser.add_argument('--output', type=Path, default=RESULTS_DIR / 'drug_attributes')
    parser.add_argument('--stages', nargs='+', choices=['continuous', 'annotations', 'replicates'], default=['continuous', 'annotations', 'replicates'])
    parser.add_argument('--features', nargs='+', choices=list(EMBEDDINGS), default=list(EMBEDDINGS))
    parser.add_argument('--continuous-inputs', nargs='+',
                        choices=['dose_only', *EMBEDDINGS, *[name + '_plus_dose' for name in EMBEDDINGS]],
                        help='Optional subset of continuous probes; use a separate output directory for partial runs')
    parser.add_argument('--protocols', nargs='+', choices=PROTOCOLS, default=PROTOCOLS)
    parser.add_argument('--models', nargs='+', choices=['linear', 'rbf'], default=['linear', 'rbf'])
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--max-queries', type=int, default=300)
    parser.add_argument('--workers', type=int, default=4, help='Parallel independent annotation-label fits; BLAS stays single-threaded')
    args = parser.parse_args()
    if args.bootstrap < 0 or args.max_queries < 1 or args.workers < 1:
        parser.error('bootstrap must be nonnegative; max-queries and workers must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    raw, df, arrays = load_data(args.data, args.features)
    chem = chemical_mapping(RESULTS_DIR / "chemical_split" / "chemical_split_structure_audit.csv") if 'held_out_chemical_group' in args.protocols else {}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(n_wells=len(raw), n_compound_wells=len(df), n_compounds=df.compound_id.nunique(),
                  seed=SEED, ridge_alpha=100, logistic_C=.1, logistic_solver='liblinear', rbf_components=256,
                  completed_stages=[], status='running',
                  data_size_bytes=args.data.stat().st_size, data_mtime_ns=args.data.stat().st_mtime_ns,
                  uncertainty='Percentile bootstrap of fixed OOF predictions; compounds or chemical groups; not model-refit uncertainty',
                  normalization='Supplied embeddings used as provided; upstream preprocessing provenance unknown')
    config_path = args.output / ('run_config_' + '_'.join(args.stages) + '.json')
    config_path.write_text(json.dumps(config, indent=2) + '\n')

    def completed(stage):
        config['completed_stages'].append(stage)
        config_path.write_text(json.dumps(config, indent=2) + '\n')

    with threadpool_limits(limits=1):
        if 'continuous' in args.stages:
            meta, inputs = aggregate(df, arrays, ['compound_id', 'compound_concentration_um', 'batch'])
            run_regression(meta, inputs, args, chem)
            completed('continuous')
        if 'annotations' in args.stages:
            meta, inputs = aggregate(df, arrays, ['compound_id', 'batch'])
            run_annotations(meta, inputs, args, chem)
            completed('annotations')
        if 'replicates' in args.stages:
            run_replicates(df, arrays, args)
            completed('replicates')
    config['status'] = 'complete'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    print(f'Saved attribute evaluations to {args.output}', flush=True)


if __name__ == '__main__':
    main()
