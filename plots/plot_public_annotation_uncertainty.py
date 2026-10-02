"""Plot endpoint and label variation for the public annotation benchmarks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/public_annotations'
OUTPUT = DATA / 'public_annotation_uncertainty_variation.png'
FEATURES = ['pca_normalized', 'pca_raw', 'brightfield', 'assay_features']
FEATURE_LABELS = {
    'pca_normalized': 'Normalized PCA', 'pca_raw': 'Raw PCA',
    'brightfield': 'DINO', 'assay_features': 'Assay features',
}
PROTOCOLS = ['held_out_compound', 'held_out_batch', 'held_out_chemical_group']
PROTOCOL_LABELS = {
    'held_out_compound': 'Unseen compound', 'held_out_batch': 'Unseen batch',
    'held_out_chemical_group': 'Unseen chemical group',
}
COLORS = {
    'held_out_compound': '#0072B2', 'held_out_batch': '#D55E00',
    'held_out_chemical_group': '#009E73',
}
MARKERS = {'held_out_compound': 'o', 'held_out_batch': 's', 'held_out_chemical_group': '^'}


def bootstrap_weighted_label_mean(values: np.ndarray, weights: np.ndarray,
                                  rng: np.random.Generator, draws: int = 3000) -> tuple[float, float, float]:
    """Percentile interval for a positive-query-weighted mean across labels."""
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[valid], weights[valid]
    if not len(values):
        return np.nan, np.nan
    mean = float(np.average(values, weights=weights))
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    sample_values, sample_weights = values[indices], weights[indices]
    estimates = (sample_values * sample_weights).sum(axis=1) / sample_weights.sum(axis=1)
    return mean, float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))


def endpoint_label(endpoint: str) -> str:
    return endpoint.replace('tox21__', '').replace('dili__', '').replace('_', ' ')


def plot(binary_path: Path, broad_path: Path, output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    binary = pd.read_csv(binary_path)
    binary = binary[(binary.model == 'linear') & binary.feature.isin(FEATURES)].copy()
    broad = pd.read_csv(broad_path)
    broad = broad[broad.feature.isin(FEATURES)].copy()
    if binary.empty or broad.empty:
        raise ValueError('Expected linear binary scores and Broad per-label retrieval summaries')

    fig, axes = plt.subplots(4, 4, figsize=(20, 15), constrained_layout=False)
    dataset_rows = [
        ('Tox21', [f'tox21__{x}' for x in [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']]),
        ('FDA DILIrank 2.0', ['dili__most_vs_no', 'dili__any_concern_vs_no']),
    ]

    # Binary benchmark: endpoint-specific AP gain with the saved compound/group
    # bootstrap interval for each split. All feature columns share a scale by row.
    for row_idx, (source, endpoints) in enumerate(dataset_rows):
        subset = binary[binary.source == source]
        bound_values = subset[['ap_delta_ci_low', 'ap_delta_ci_high']].to_numpy(float).ravel()
        bound_values = bound_values[np.isfinite(bound_values)]
        extent = max(float(np.max(np.abs(bound_values))) if len(bound_values) else .1, .05)
        xlim = (-extent * 1.08, extent * 1.08)
        for col_idx, feature in enumerate(FEATURES):
            ax = axes[row_idx, col_idx]
            part = subset[subset.feature == feature]
            for task_idx, endpoint in enumerate(endpoints):
                for split_idx, protocol in enumerate(PROTOCOLS):
                    match = part[(part.endpoint == endpoint) & (part.protocol == protocol)]
                    if match.empty:
                        continue
                    result = match.iloc[0]
                    x = float(result.ap_delta)
                    lo, hi = float(result.ap_delta_ci_low), float(result.ap_delta_ci_high)
                    y = task_idx + (split_idx - 1) * .22
                    xerr = np.array([[max(0.0, x - lo)], [max(0.0, hi - x)]])
                    ax.errorbar(x, y, xerr=xerr, fmt=MARKERS[protocol], color=COLORS[protocol],
                                markersize=4, capsize=1.8, linewidth=1.0, alpha=.9)
            ax.axvline(0, color='#444444', linewidth=.8, linestyle='--')
            ax.set_xlim(xlim)
            ax.set_yticks(np.arange(len(endpoints)), [endpoint_label(e) for e in endpoints], fontsize=6.5)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=.18)
            ax.set_axisbelow(True)
            if row_idx == 0:
                ax.set_title(FEATURE_LABELS[feature], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f'{source}\nAssay / task', fontsize=9)
            else:
                ax.tick_params(axis='y', labelleft=False)
            if row_idx == 1:
                ax.set_xlabel('AP gain over prevalence baseline', fontsize=8)

    # Broad labels are positive-only annotations. Plot the across-label spread,
    # plus a query-count-weighted average with a label-bootstrap interval.
    rng = np.random.default_rng(20261001)
    broad_tasks = [
        ('Broad Repurposing Hub MOA', 'Broad MOA'),
        ('Broad Repurposing Hub target', 'Broad target'),
    ]
    for task_idx, (task, task_title) in enumerate(broad_tasks):
        row_idx = task_idx + 2
        for col_idx, feature in enumerate(FEATURES):
            ax = axes[row_idx, col_idx]
            part = broad[(broad.task == task) & (broad.feature == feature)]
            for split_idx, protocol in enumerate(PROTOCOLS):
                points = part[part.protocol == protocol]
                values = points.precision_lift.to_numpy(float)
                weights = points.n_positive_queries.to_numpy(float)
                mean, low, high = bootstrap_weighted_label_mean(values, weights, rng)
                y = split_idx
                jitter = rng.uniform(-.16, .16, size=len(values))
                ax.scatter(values, y + jitter, color=COLORS[protocol], s=8, alpha=.23,
                           linewidths=0, rasterized=True)
                if np.isfinite(mean):
                    ax.errorbar(mean, y, xerr=np.array([[mean-low], [high-mean]]),
                                fmt=MARKERS[protocol], color=COLORS[protocol], markersize=5,
                                capsize=2.5, linewidth=1.6, zorder=4)
            ax.axvline(0, color='#444444', linewidth=.8, linestyle='--')
            ax.grid(axis='x', alpha=.18)
            ax.set_axisbelow(True)
            ax.set_yticks(np.arange(3), [PROTOCOL_LABELS[p] for p in PROTOCOLS], fontsize=7)
            ax.invert_yaxis()
            if col_idx == 0:
                ax.set_ylabel(f'{task_title}\nValidation split', fontsize=9)
            else:
                ax.tick_params(axis='y', labelleft=False)
            if task_idx == 1:
                ax.set_xlabel('Top-five neighbor precision lift', fontsize=8)
            if task_idx == 0:
                ax.set_title(FEATURE_LABELS[feature], fontsize=10)

    protocol_handles = [Line2D([0], [0], color=COLORS[p], marker=MARKERS[p], linewidth=1.2,
                               markersize=5, label=PROTOCOL_LABELS[p]) for p in PROTOCOLS]
    fig.legend(handles=protocol_handles, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, .99))
    fig.suptitle('Public annotation performance: endpoint and label variation', y=1.015,
                 fontsize=15)
    fig.text(.5, .006,
             'Tox21/DILIrank: whiskers are 95% bootstrap intervals for fixed out-of-fold predictions. '
             'Broad: faint dots are per-label lifts; dark points/whiskers are query-weighted means and '
             '95% bootstrap intervals across labels (not compound-level intervals).',
             ha='center', fontsize=8)
    fig.subplots_adjust(left=.12, right=.99, top=.94, bottom=.055, wspace=.25, hspace=.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=DATA / 'binary_scores.csv')
    parser.add_argument('--broad', type=Path, default=DATA / 'broad_label_neighbor_summary.csv')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    plot(args.binary, args.broad, args.output)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
