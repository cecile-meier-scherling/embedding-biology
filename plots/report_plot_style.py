"""Consistent, more readable typography for figures included in the report."""
from __future__ import annotations

import matplotlib.pyplot as plt


def enlarge_report_text(fig: plt.Figure) -> None:
    """Increase titles, axis labels, tick labels, and legends in-place."""
    for ax in fig.axes:
        if ax.get_title():
            ax.title.set_fontsize(max(ax.title.get_fontsize() + 2, 16))
        if ax.get_xlabel():
            ax.xaxis.label.set_fontsize(max(ax.xaxis.label.get_fontsize() + 2, 15))
        if ax.get_ylabel():
            ax.yaxis.label.set_fontsize(max(ax.yaxis.label.get_fontsize() + 2, 15))
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontsize(max(label.get_fontsize() + 1, 12))
        legend = ax.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_fontsize(max(label.get_fontsize() + 2, 13))
            if legend.get_title().get_text():
                legend.get_title().set_fontsize(max(legend.get_title().get_fontsize() + 2, 14))
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(max(fig._suptitle.get_fontsize() + 2, 18))
    for legend in fig.legends:
        for label in legend.get_texts():
            label.set_fontsize(max(label.get_fontsize() + 2, 13))
        if legend.get_title().get_text():
            legend.get_title().set_fontsize(max(legend.get_title().get_fontsize() + 2, 14))
