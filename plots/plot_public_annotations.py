"""Plot Broad Repurposing Hub positive-label neighbor-retrieval results."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'results/public_annotations/broad_label_neighbor_summary.csv'
DEFAULT_OUTPUT = ROOT / 'results/public_annotations'
TASK_ORDER = ['Broad Repurposing Hub MOA', 'Broad Repurposing Hub target']
PROTOCOL_ORDER = ['held_out_compound', 'held_out_batch', 'held_out_chemical_group']
FEATURE_ORDER = ['pca_normalized', 'pca_raw', 'brightfield', 'assay_features']
FEATURE_LABELS = {
    'pca_normalized': 'Norm. PCA',
    'pca_raw': 'Raw PCA',
    'brightfield': 'Brightfield\n(DINO)',
    'assay_features': 'Assay\nfeatures',
}
PROTOCOL_LABELS = {
    'held_out_compound': 'Unseen compound',
    'held_out_batch': 'Unseen batch',
    'held_out_chemical_group': 'Unseen chemical group',
}


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {'task', 'protocol', 'feature', 'label', 'n_positive_queries',
                'neighbor_precision_at_k', 'training_label_prevalence', 'precision_lift'}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f'{path} is missing columns: {sorted(missing)}')
    if data.empty:
        raise ValueError(f'{path} contains no Broad retrieval results')
    return data


def plot_summary(data: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'held_out_compound': '#0072B2', 'held_out_batch': '#D55E00',
              'held_out_chemical_group': '#009E73'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    width = .24
    offsets = {'held_out_compound': -width, 'held_out_batch': 0,
               'held_out_chemical_group': width}
    for ax, task in zip(axes, TASK_ORDER):
        part = data[data.task.eq(task)]
        x = np.arange(len(FEATURE_ORDER))
        for protocol in PROTOCOL_ORDER:
            heights = []
            for feature in FEATURE_ORDER:
                rows = part[(part.protocol == protocol) & (part.feature == feature)]
                if rows.empty:
                    heights.append(np.nan)
                    continue
                heights.append(np.average(rows.precision_lift, weights=rows.n_positive_queries))
            ax.bar(x + offsets[protocol], heights, width=width, color=colors[protocol],
                   label=PROTOCOL_LABELS[protocol])
        ax.axhline(0, color='black', linewidth=.8)
        ax.set_title(task.replace('Broad Repurposing Hub ', ''))
        ax.set_xticks(range(len(FEATURE_ORDER)),
                      [FEATURE_LABELS[name] for name in FEATURE_ORDER])
        ax.grid(axis='y', alpha=.22)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Top-five neighbor precision above training-label prevalence')
    axes[1].legend(frameon=False, loc='upper right')
    fig.suptitle('Broad Drug Repurposing Hub: positive-label neighbor retrieval')
    fig.tight_layout(rect=(0, .02, 1, .92))
    fig.savefig(output, dpi=190, bbox_inches='tight')
    plt.close(fig)


def plot_label_heatmap(data: pd.DataFrame, output: Path, top_n: int) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    feature_order = ['pca_normalized', 'pca_raw', 'brightfield']
    groups = []
    for task in TASK_ORDER:
        part = data[(data.task == task) & (data.protocol == 'held_out_chemical_group')]
        support = (part[['label', 'n_positive_queries']]
                   .drop_duplicates('label').sort_values('n_positive_queries', ascending=False))
        labels = support.head(top_n).label.tolist()
        matrix = (part.pivot(index='label', columns='feature', values='precision_lift')
                  .reindex(index=labels, columns=feature_order))
        counts = support.set_index('label').n_positive_queries
        groups.append((task, labels, matrix, counts))

    values = np.concatenate([matrix.to_numpy().ravel() for _, _, matrix, _ in groups])
    bound = max(float(np.nanquantile(np.abs(values), .95)), .01)
    fig, axes = plt.subplots(1, 2, figsize=(12, max(6, top_n * .34)), sharex=True)
    for ax, (task, labels, matrix, counts) in zip(axes, groups):
        short_labels = [f'{label}  (n={int(counts[label])})' for label in labels]
        image = ax.imshow(matrix.to_numpy(), aspect='auto', cmap='RdBu_r',
                          norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
        ax.set_title(task.replace('Broad Repurposing Hub ', '') + '\nchemical-group holdout', fontsize=12)
        ax.set_yticks(np.arange(len(labels)), short_labels, fontsize=8)
        ax.set_xticks(range(len(feature_order)),
                      [FEATURE_LABELS[name] for name in feature_order])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f'{value:+.2f}', ha='center', va='center', fontsize=7,
                            color='white' if abs(value) > bound * .55 else '#202020')
    fig.subplots_adjust(left=.24, right=.88, bottom=.12, top=.88, wspace=.38)
    colorbar = fig.colorbar(image, ax=axes, shrink=.8, pad=.03)
    colorbar.set_label('Top-five precision lift')
    fig.suptitle(f'Broad labels with the most positive queries (top {top_n} per task)', y=.97, fontsize=15)
    fig.savefig(output, dpi=190, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--top-labels', type=int, default=15)
    args = parser.parse_args()
    if args.top_labels < 1:
        parser.error('--top-labels must be positive')
    data = load_data(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_summary(data, args.output_dir / 'broad_neighbor_lift.png')
    plot_label_heatmap(data, args.output_dir / 'broad_neighbor_label_examples.png', args.top_labels)
    print(f'Wrote Broad retrieval figures to {args.output_dir}')


if __name__ == '__main__':
    main()
