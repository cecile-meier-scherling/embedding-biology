"""Plot annotation-label AP gains for each embedding and validation protocol."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'results/drug_attributes/annotation_scores.csv'
DEFAULT_OUTPUT = ROOT / 'results/drug_attributes/annotation_embedding_comparison.png'
FEATURES = ['pca_normalized', 'pca_raw', 'brightfield']
FEATURE_LABELS = {'pca_normalized': 'Norm. PCA', 'pca_raw': 'Raw PCA', 'brightfield': 'Brightfield/DINO'}
FEATURE_SHORT_LABELS = {'pca_normalized': 'Norm. PCA', 'pca_raw': 'Raw PCA', 'brightfield': 'DINO'}
PROTOCOLS = ['held_out_compound', 'held_out_batch', 'held_out_chemical_group']
PROTOCOL_LABELS = {'held_out_compound': 'compound', 'held_out_batch': 'batch',
                   'held_out_chemical_group': 'chemical group'}
TASKS = ['pathway', 'target']


def label_embedding_split_axis(ax, columns, fontsize: float) -> None:
    """Show compact embedding ticks grouped beneath each validation split."""
    tick_labels = [FEATURE_SHORT_LABELS[feature] for _, feature in columns]
    ax.set_xticks(np.arange(len(columns)), tick_labels, fontsize=fontsize)
    ax.tick_params(axis='x', length=0, pad=5)
    for split_index, protocol in enumerate(PROTOCOLS):
        center = split_index * len(FEATURES) + (len(FEATURES) - 1) / 2
        ax.text(center, -.12, PROTOCOL_LABELS[protocol],
                transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=fontsize, fontweight='bold')


def plot_scores(scores: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    required = {'task', 'label', 'protocol', 'feature', 'model', 'ap_delta',
                'ap_delta_ci_low', 'n_positive'}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f'Annotation score file is missing columns: {sorted(missing)}')
    data = scores[(scores.model == 'linear') & scores.feature.isin(FEATURES)].copy()
    if data.empty:
        raise ValueError('No linear-probe results for all three requested embeddings')

    # Keep every endpoint in the existing score table. Order columns by split,
    # then embedding, and sort each panel by average AP gain across these cells.
    columns = pd.MultiIndex.from_product([PROTOCOLS, FEATURES], names=['protocol', 'feature'])
    matrices, supports = {}, {}
    all_values = []
    for task in TASKS:
        part = data[data.task == task]
        if part.empty:
            continue
        values = part.pivot(index='label', columns=['protocol', 'feature'], values='ap_delta').reindex(columns=columns)
        lower = part.pivot(index='label', columns=['protocol', 'feature'], values='ap_delta_ci_low').reindex(columns=columns)
        npos = part.groupby('label').n_positive.min()
        order = values.mean(axis=1).sort_values(ascending=False, na_position='last').index
        values, lower = values.reindex(order), lower.reindex(order)
        matrices[task] = (values, lower)
        supports[task] = npos.reindex(order)
        all_values.extend(values.to_numpy().ravel().tolist())

    finite = np.asarray([x for x in all_values if np.isfinite(x)])
    bound = max(float(np.quantile(np.abs(finite), .97)), .05)
    row_counts = [len(matrices[t][0]) for t in TASKS if t in matrices]
    fig_height = max(8, 3 + .22 * sum(row_counts))
    fig, axes = plt.subplots(len(row_counts), 1, figsize=(13.2, fig_height), squeeze=False,
                             gridspec_kw={'height_ratios': row_counts})
    image = None
    axis_tasks = [task for task in TASKS if task in matrices]
    for ax, task in zip(axes[:, 0], axis_tasks):
        values, lower = matrices[task]
        support = supports[task]
        masked = np.ma.masked_invalid(values.to_numpy(float))
        image = ax.imshow(masked, aspect='auto', cmap='RdBu_r',
                          norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
                          interpolation='none')
        names = [f'{label}  (n+={int(support[label])})' for label in values.index]
        ax.set_yticks(np.arange(len(names)), names, fontsize=7.5)
        label_embedding_split_axis(ax, columns, fontsize=8)
        ax.set_title(task.capitalize() + ' annotations', loc='left', fontsize=11, pad=7)
        ax.set_xticks(np.arange(-.5, len(columns), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(values), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=.7)
        ax.tick_params(which='minor', bottom=False, left=False)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values.iloc[row, col]
                if pd.isna(value):
                    continue
                star = '*' if pd.notna(lower.iloc[row, col]) and lower.iloc[row, col] > 0 else ''
                clipped = abs(value) > bound * .62
                ax.text(col, row, f'{value:+.2f}{star}', ha='center', va='center', fontsize=6.5,
                        color='white' if clipped else '#202020')
        ax.set_ylabel('Annotation label (positive compounds)')

    fig.suptitle('PHH embedding annotation performance across validation splits', y=.995, fontsize=15)
    fig.text(.5, .006,
             '* Nominal bootstrap interval lower bound > 0; no multiple-comparison correction. '
             'Color scale is centered at zero and clipped at the 97th percentile.',
             ha='center', fontsize=8)
    fig.subplots_adjust(left=.25, right=.88, top=.975, bottom=.075, hspace=.08)
    cbar = fig.colorbar(image, ax=axes[:, 0].tolist(), shrink=.82, pad=.025)
    cbar.set_label('AP gain (Δ average precision)')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)

    # Also save one compact, task-specific figure apiece so pathway and target
    # labels can be read without sharing a page with the other task's rows.
    for task, (values, lower) in matrices.items():
        fig_height = max(5.5, 3.0 + .22 * len(values))
        fig, ax = plt.subplots(figsize=(13.2, fig_height))
        image = ax.imshow(np.ma.masked_invalid(values.to_numpy(float)), aspect='auto',
                          cmap='RdBu_r',
                          norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
                          interpolation='none')
        support = supports[task]
        names = [f'{label}  (positive n={int(support[label])})' for label in values.index]
        ax.set_yticks(np.arange(len(names)), names, fontsize=8)
        label_embedding_split_axis(ax, columns, fontsize=9)
        ax.set_xticks(np.arange(-.5, len(columns), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(values), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=.7)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.set_ylabel('Annotation label')
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values.iloc[row, col]
                if pd.isna(value):
                    continue
                star = '*' if pd.notna(lower.iloc[row, col]) and lower.iloc[row, col] > 0 else ''
                ax.text(col, row, f'{value:+.2f}{star}', ha='center', va='center', fontsize=7,
                        color='white' if abs(value) > bound * .62 else '#202020')
        ax.set_title(f'{task.capitalize()} annotations across embeddings and validation splits', pad=12)
        fig.text(.5, .008,
                 '* Nominal bootstrap interval lower bound > 0; no multiple-comparison correction.',
                 ha='center', fontsize=8)
        fig.subplots_adjust(left=.28, right=.88, top=.95, bottom=.14)
        cbar = fig.colorbar(image, ax=ax, shrink=.84, pad=.025)
        cbar.set_label('AP gain over prevalence baseline')
        task_output = output.with_name(f'annotation_{task}s_across_embeddings.png')
        fig.savefig(task_output, dpi=220, bbox_inches='tight')
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_scores(pd.read_csv(args.input), args.output)
    print(f'Wrote annotation embedding comparison to {args.output}')


if __name__ == '__main__':
    main()
