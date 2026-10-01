"""Plot best and weakest pathway/target labels for raw PCA and brightfield."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'results/drug_attributes/annotation_scores.csv'
DEFAULT_OUTPUT = ROOT / 'results/drug_attributes'
FEATURES = ['pca_normalized', 'pca_raw', 'brightfield']
FEATURE_TITLES = {'pca_normalized': 'Normalized PCA', 'pca_raw': 'Raw PCA',
                  'brightfield': 'DINO'}
TASKS = ['pathway', 'target']
PROTOCOLS = ['held_out_compound', 'held_out_batch', 'held_out_chemical_group']
PROTOCOL_LABELS = {
    'held_out_compound': 'Unseen compound',
    'held_out_batch': 'Unseen batch',
    'held_out_chemical_group': 'Unseen chemical group',
}


def choose_labels(scores: pd.DataFrame, feature: str, task: str, top_n: int,
                  min_positive: int, min_negative: int) -> tuple[list[str], dict[str, int]]:
    base = scores[(scores.feature == feature) & (scores.model == 'linear') &
                  (scores.task == task) & (scores.protocol == 'held_out_compound')].copy()
    base = base[(base.n_positive >= min_positive) &
                ((base.n_compounds - base.n_positive) >= min_negative)]
    base = base.drop_duplicates('label').sort_values('ap_delta')
    if base.empty:
        return [], {}
    n = min(top_n, len(base) // 2)
    chosen = list(base.head(n).label) + list(base.tail(n).sort_values('ap_delta', ascending=False).label)
    support = base.set_index('label').n_positive.astype(int).to_dict()
    return chosen, support


def plot_one_feature(scores: pd.DataFrame, feature: str, output: Path, top_n: int,
                     min_positive: int, min_negative: int) -> None:
    selected = {}
    support = {}
    for task in TASKS:
        selected[task], support[task] = choose_labels(scores, feature, task, top_n,
                                                        min_positive, min_negative)
    if not any(selected.values()):
        return

    # Use a common horizontal scale for both embeddings and both attribute types.
    extents = []
    for task, labels in selected.items():
        part = scores[(scores.feature == feature) & (scores.model == 'linear') &
                      (scores.task == task) & scores.label.isin(labels)]
        extents.extend(part.ap_delta_ci_low.dropna().tolist())
        extents.extend(part.ap_delta_ci_high.dropna().tolist())
    xmin, xmax = min(extents), max(extents)
    padding = max((xmax - xmin) * .06, .015)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)
    colors = {'held_out_compound': '#0072B2', 'held_out_batch': '#D55E00',
              'held_out_chemical_group': '#009E73'}
    task_titles = {'pathway': 'Pathway annotations', 'target': 'Target annotations'}
    for ax, task in zip(axes, TASKS):
        labels = selected[task]
        if not labels:
            ax.axis('off')
            continue
        part = scores[(scores.feature == feature) & (scores.model == 'linear') &
                      (scores.task == task) & scores.label.isin(labels)]
        # Preserve selection order: weakest first, then strongest, as in the
        # original contrast plot. Values shown for every protocol use that list.
        y_positions = np.arange(len(labels))
        for label_index, label in enumerate(labels):
            for protocol_index, protocol in enumerate(PROTOCOLS):
                row = part[(part.label == label) & (part.protocol == protocol)]
                if row.empty:
                    continue
                row = row.iloc[0]
                y = label_index + (protocol_index - 1) * .20
                ax.hlines(y, row.ap_delta_ci_low, row.ap_delta_ci_high,
                          color=colors[protocol], alpha=.75, linewidth=1.4)
                ax.scatter(row.ap_delta, y, color=colors[protocol], s=25,
                           label=PROTOCOL_LABELS[protocol] if label_index == 0 else None,
                           zorder=3)
        tick_names = [f'{label}  (pos={support[task].get(label, 0)})' for label in labels]
        ax.set_yticks(y_positions, tick_names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=.8)
        ax.axhline(len(labels) / 2 - .5, color='#777777', linestyle=':', linewidth=.8)
        ax.set_title(task_titles[task])
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
        ax.set_xlim(xmin - padding, xmax + padding)
    axes[0].set_ylabel('Pathway / target label')
    axes[0].set_xlabel('Average precision gain over prevalence baseline (nominal 95% CI)')
    axes[1].set_xlabel('Average precision gain over prevalence baseline (nominal 95% CI)')
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, .965))
    fig.suptitle(f'{FEATURE_TITLES[feature]}: strongest and weakest annotation labels',
                 y=.995, fontsize=14)
    fig.tight_layout(rect=(0, .02, 1, .92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches='tight')
    plt.close(fig)


def plot_task_across_features(scores: pd.DataFrame, task: str, output: Path, top_n: int,
                             min_positive: int, min_negative: int) -> None:
    """Compare each embedding in its own panel for one annotation type."""
    selected_by_feature = {}
    support_by_feature = {}
    extents = []
    for feature in FEATURES:
        labels, support = choose_labels(scores, feature, task, top_n,
                                        min_positive, min_negative)
        selected_by_feature[feature] = labels
        support_by_feature[feature] = support
        part = scores[(scores.feature == feature) & (scores.model == 'linear') &
                      (scores.task == task) & scores.label.isin(labels)]
        extents.extend(part.ap_delta_ci_low.dropna().tolist())
        extents.extend(part.ap_delta_ci_high.dropna().tolist())
    if not extents:
        return
    xmin, xmax = min(extents), max(extents)
    padding = max((xmax - xmin) * .06, .015)

    fig, axes = plt.subplots(1, len(FEATURES), figsize=(19, 7), sharex=True)
    colors = {'held_out_compound': '#0072B2', 'held_out_batch': '#D55E00',
              'held_out_chemical_group': '#009E73'}
    task_title = 'Pathway' if task == 'pathway' else 'Target'
    for ax, feature in zip(axes, FEATURES):
        labels = selected_by_feature[feature]
        if not labels:
            ax.axis('off')
            continue
        part = scores[(scores.feature == feature) & (scores.model == 'linear') &
                      (scores.task == task) & scores.label.isin(labels)]
        for label_index, label in enumerate(labels):
            for protocol_index, protocol in enumerate(PROTOCOLS):
                row = part[(part.label == label) & (part.protocol == protocol)]
                if row.empty:
                    continue
                row = row.iloc[0]
                y = label_index + (protocol_index - 1) * .20
                ax.hlines(y, row.ap_delta_ci_low, row.ap_delta_ci_high,
                          color=colors[protocol], alpha=.75, linewidth=1.4)
                ax.scatter(row.ap_delta, y, color=colors[protocol], s=24,
                           label=PROTOCOL_LABELS[protocol] if label_index == 0 else None,
                           zorder=3)
        tick_names = [f'{label}  (pos={support_by_feature[feature].get(label, 0)})'
                      for label in labels]
        ax.set_yticks(np.arange(len(labels)), tick_names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=.8)
        ax.axhline(len(labels) / 2 - .5, color='#777777', linestyle=':', linewidth=.8)
        ax.set_title(FEATURE_TITLES[feature])
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
        ax.set_xlim(xmin - padding, xmax + padding)
        ax.set_xlabel('AP gain over prevalence baseline\n(nominal 95% CI)')
    axes[0].set_ylabel(f'{task_title} annotation (positive compounds)')
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, .96))
    fig.suptitle(f'{task_title} labels: strongest and weakest for each embedding',
                 y=.995, fontsize=14)
    fig.tight_layout(rect=(0, .02, 1, .90))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--top-n', type=int, default=5,
                        help='number of weakest and strongest labels per task (default: 5)')
    parser.add_argument('--min-positive', type=int, default=20)
    parser.add_argument('--min-negative', type=int, default=20)
    args = parser.parse_args()
    if min(args.top_n, args.min_positive, args.min_negative) < 1:
        parser.error('--top-n, --min-positive, and --min-negative must be positive')
    scores = pd.read_csv(args.input)
    for feature in FEATURES:
        path = args.output_dir / f'annotation_attributes_{feature}_top_worst.png'
        plot_one_feature(scores, feature, path, args.top_n, args.min_positive, args.min_negative)
        print(f'Wrote {path}')
    for task in TASKS:
        path = args.output_dir / f'annotation_{task}s_top_worst_across_embeddings.png'
        plot_task_across_features(scores, task, path, args.top_n,
                                  args.min_positive, args.min_negative)
        print(f'Wrote {path}')


if __name__ == '__main__':
    main()
