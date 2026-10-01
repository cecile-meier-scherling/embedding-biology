"""Build figures, paired incremental-value comparisons and an exploratory scorecard."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'axiom-attribute-matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'axiom-attribute-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
from evaluate_drug_attributes import EMBEDDINGS, TARGETS, annotation_metrics, regression_metrics

LABELS = {
    'mean_nuclei_count': 'Nuclei count', 'mean_nuclei_area_mean': 'Nuclear area',
    'vacuole_ratio': 'Vacuolization', 'mito_puncta_ratio': 'Mitochondrial puncta',
    'ros_stress_granules_sum_mean_norm_ridge_norm_corr': 'ROS/stress granules',
    'ldh_ridge_norm': 'LDH', 'mtt_ridge_norm': 'MTT',
}


def scorecard(continuous, annotations):
    rows = []
    for kind, data, attribute, low, high, value in [
        ('continuous', continuous, 'target', 'mse_skill_ci_low', 'mse_skill_ci_high', 'mse_skill'),
        ('annotation', annotations, 'label', 'ap_delta_ci_low', 'ap_delta_ci_high', 'ap_delta'),
    ]:
        data = data[data.feature.isin(EMBEDDINGS)]
        keys = ['feature', attribute] + (['task'] if kind == 'annotation' else [])
        for key, part in data.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            protocol_values = set(part.protocol)
            adequate = bool((part.n_compounds >= 50).all())
            if kind == 'annotation':
                adequate &= bool((part.n_positive >= 20).all() and ((part.n_compounds-part.n_positive) >= 20).all())
            robust = protocol_values == {'held_out_compound', 'held_out_batch', 'held_out_chemical_group'}
            # Require a single prespecified model to show consistent evidence across protocols.
            consistent_models = []
            for model, m in part.groupby('model'):
                if set(m.protocol) == protocol_values and (m[low] > 0).all():
                    consistent_models.append(model)
            weak = {'linear', 'rbf'}.issubset(set(part.model)) and (part[high] < .05).all()
            status = 'inconclusive'
            if adequate and robust and consistent_models:
                status = 'recoverable_across_protocols'
            elif adequate and robust and weak:
                status = 'limited_under_tested_probes'
            elif adequate and (part[low] > 0).any():
                status = 'signal_requires_robustness_review'
            rows.append(dict(kind=kind, feature=key[0], attribute=key[1],
                             task=key[2] if kind == 'annotation' else 'regression',
                             status=status, min_n_compounds=int(part.n_compounds.min()),
                             min_n_positive=int(part.n_positive.min()) if kind == 'annotation' else np.nan,
                             min_effect=float(part[value].min()), max_effect=float(part[value].max()),
                             consistent_models=';'.join(consistent_models),
                             note='Exploratory nominal intervals; no multiplicity correction or model-refit uncertainty'))
    return pd.DataFrame(rows)


def plot_continuous(data, output):
    protocols = list(data.protocol.unique())
    fig, axes = plt.subplots(1, len(protocols), figsize=(6*len(protocols), 5), squeeze=False)
    for ax, protocol in zip(axes[0], protocols):
        selected = data[(data.protocol == protocol) & data.feature.isin(EMBEDDINGS)]
        selected = selected.assign(probe=selected.feature + '\n' + selected.model)
        table = selected.pivot(index='target', columns='probe', values='mse_skill').reindex(TARGETS)
        im = ax.imshow(table.to_numpy(), cmap='RdBu', vmin=-.5, vmax=1, aspect='auto')
        ax.set_xticks(range(len(table.columns)), table.columns, rotation=50, ha='right', fontsize=8)
        ax.set_yticks(range(len(table)), [LABELS.get(t, t) for t in table.index], fontsize=9)
        ax.set_title(protocol.replace('held_out_', 'Unseen ').replace('_', ' '))
        for (i, j), value in np.ndenumerate(table.to_numpy()):
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if value > .65 or value < -.3 else 'black')
    fig.suptitle('Recovery of cellular attributes from embeddings\nMSE improvement relative to fold training-mean baseline; equal compound weights')
    fig.tight_layout(rect=(0, 0, .93, .90))
    fig.colorbar(im, cax=fig.add_axes([.95, .25, .012, .5]), label='1 − model MSE / baseline MSE')
    fig.savefig(output / 'continuous_attributes.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_annotations(data, output):
    selected = data[(data.feature == 'pca_normalized') & (data.model == 'linear')]
    base = selected[(selected.protocol == 'held_out_compound') & (selected.n_positive >= 20)].copy()
    if base.empty:
        return
    base['attribute'] = base.task + ': ' + base.label
    ordered = base.sort_values('ap_delta')
    names = list(dict.fromkeys(ordered.head(5).attribute.tolist() + ordered.tail(5).attribute.tolist()))
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, protocol in enumerate(selected.protocol.unique()):
        part = selected[selected.protocol == protocol].copy()
        part['attribute'] = part.task + ': ' + part.label
        part = part.set_index('attribute').reindex(names)
        y = np.arange(len(names)) + (i-1)*.22
        lo, hi = part.ap_delta_ci_low.to_numpy(), part.ap_delta_ci_high.to_numpy()
        ax.hlines(y, lo, hi, color=f'C{i}', alpha=.75)
        ax.scatter(part.ap_delta, y, color=f'C{i}', s=22, label=protocol.replace('held_out_', 'Unseen ').replace('_', ' '))
    ax.set_yticks(range(len(names)), names)
    ax.axvline(0, color='black', linewidth=.8)
    ax.set_xlabel('Average precision − fold-trained prevalence baseline AP (nominal 95% CI)')
    ax.set_title('Normalized PCA, linear probe: contrasting annotation examples\nSelected by compound-holdout effect; ≥20 positive compounds there; exploratory')
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(output / 'annotation_attributes.png', dpi=180)
    plt.close(fig)


def plot_scatter(data, output):
    selected = data[(data.feature == 'pca_normalized') & (data.model == 'linear') & (data.protocol == 'held_out_compound')].sort_values('mse_skill')
    if selected.empty:
        return
    targets = [selected.iloc[0].target, selected.iloc[-1].target]
    path = output / 'regression_predictions/held_out_compound__pca_normalized__linear.parquet'
    predictions = pd.read_parquet(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, target in zip(axes, targets):
        part = predictions[predictions.target == target]
        sample = part.sample(min(2500, len(part)), random_state=42)
        ax.scatter(sample.truth, sample.prediction, alpha=.15, s=5)
        bounds = [min(sample.truth.min(), sample.prediction.min()), max(sample.truth.max(), sample.prediction.max())]
        ax.plot(bounds, bounds, color='black', linestyle='--', linewidth=.7)
        ax.set(xlabel='Observed', ylabel='Held-out prediction', title=LABELS.get(target, target))
    fig.suptitle('Normalized PCA, linear probe: lowest/highest continuous skill\nDose–batch profiles; points sampled for display only')
    fig.tight_layout()
    fig.savefig(output / 'continuous_examples.png', dpi=180)
    plt.close(fig)


def incremental_comparisons(output, bootstrap):
    rows = []
    for path in sorted((output / 'regression_predictions').glob('*_plus_dose__*.parquet')):
        protocol, feature, model = path.stem.split('__')
        basepath = output / 'regression_predictions' / f'{protocol}__dose_only__{model}.parquet'
        if not basepath.exists():
            continue
        a, b = pd.read_parquet(path), pd.read_parquet(basepath)
        keys = ['compound_id', 'batch', 'dose', 'fold', 'target']
        joined = a.drop(columns='baseline').merge(b[keys+['prediction']].rename(columns={'prediction': 'baseline'}), on=keys, validate='one_to_one')
        for target, frame in joined.groupby('target'):
            m = regression_metrics(frame, bootstrap)
            rows.append(dict(kind='continuous', protocol=protocol, feature=feature, model=model,
                             attribute=target, comparator='dose_only', effect=m['mse_skill'],
                             ci_low=m['mse_skill_ci_low'], ci_high=m['mse_skill_ci_high']))
    for path in sorted((output / 'annotation_predictions').glob('*_plus_damage_assays__*.parquet')):
        task, protocol, feature, model = path.stem.split('__')
        basepath = output / 'annotation_predictions' / f'{task}__{protocol}__damage_assays__linear.parquet'
        if not basepath.exists():
            continue
        a, b = pd.read_parquet(path), pd.read_parquet(basepath)
        keys = ['compound_id', 'fold', 'label']
        # Compound holdouts can contain several batch profiles per compound.
        a = a.groupby(keys, as_index=False).agg(truth=('truth', 'max'),
            prediction=('prediction', 'mean'), bootstrap_unit=('bootstrap_unit', 'first'))
        b = b.groupby(keys, as_index=False).agg(prediction=('prediction', 'mean'))
        joined = a.merge(b[keys+['prediction']].rename(columns={'prediction':'baseline'}), on=keys, validate='one_to_one')
        for label, frame in joined.groupby('label'):
            pooled = frame.groupby('compound_id').agg(truth=('truth','max'), prediction=('prediction','mean'),
                                                     baseline=('baseline','mean'), bootstrap_unit=('bootstrap_unit','first'))
            if pooled.truth.nunique() < 2:
                continue
            m = annotation_metrics(pooled.truth.to_numpy(), pooled.prediction.to_numpy(), pooled.baseline.to_numpy(),
                                   pooled.bootstrap_unit.to_numpy(), bootstrap)
            rows.append(dict(kind=task, protocol=protocol, feature=feature, model=model, attribute=label,
                             comparator='damage_assays', effect=m['ap_delta'], ci_low=m['ap_delta_ci_low'], ci_high=m['ap_delta_ci_high']))
    pd.DataFrame(rows).to_csv(output / 'incremental_value.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'results/drug_attributes')
    parser.add_argument('--bootstrap', type=int, default=200)
    args = parser.parse_args()
    continuous = pd.read_csv(args.output / 'continuous_scores.csv')
    annotations = pd.read_csv(args.output / 'annotation_scores.csv')
    scorecard(continuous, annotations).to_csv(args.output / 'attribute_scorecard.csv', index=False)
    plot_continuous(continuous, args.output)
    plot_annotations(annotations, args.output)
    plot_scatter(continuous, args.output)
    with threadpool_limits(limits=1):
        incremental_comparisons(args.output, args.bootstrap)
    print(f'Saved scorecard, paired comparisons, and three figures to {args.output}')


if __name__ == '__main__':
    main()
