"""plotting functions for pipeline comparison visualization."""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.decomposition_strategy_selection.config import (
    PIPELINE_LABELS,
    PIPELINE_COLORS,
)


METRIC_LABELS = {
    'det_deviation': 'DET Deviation',
    'lam_deviation': 'LAM Deviation',
    'tt_deviation': 'TT Deviation',
    'entr_deviation': 'ENTR Deviation',
    'hamming_distance': 'Hamming Distance',
    'rp_pattern_corr': 'RP Pattern Correlation'
}


def plot_boxplot_comparisons(agg_df, metrics, output_dir):
    """
    generate box plots for pipeline comparison with consistent style.

    args:
        agg_df: DataFrame with columns ['trial', 'pipeline', metric1, metric2, ...]
        metrics: list of metric names to plot
        output_dir: directory to save plots
    """
    df = agg_df.copy()
    df['pipeline'] = df['pipeline'].map(PIPELINE_LABELS)

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # prepare data for boxplot
        pipelines = ['SWT-R', 'SWT-D', 'DWT-R']
        data = [
            df[df['pipeline'] == p][metric].dropna().values
            for p in pipelines
        ]
        colors = [PIPELINE_COLORS[p] for p in pipelines]

        # create boxplot
        box = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.55,
            showfliers=False
        )

        # apply colors
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)

        # formatting
        ax.set_xticks(range(1, len(pipelines) + 1))
        ax.set_xticklabels(pipelines, fontsize=11)

        ax.set_xlabel('Pipeline', fontsize=12, fontweight='bold')
        ax.set_ylabel(
            METRIC_LABELS.get(metric, metric),
            fontsize=12,
            fontweight='bold'
        )

        ax.set_title(
            METRIC_LABELS.get(metric, metric),
            fontsize=13,
            fontweight='bold'
        )

        ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    # save
    plot_path = os.path.join(output_dir, 'pipeline_comparison_boxplots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    plot_path_pdf = os.path.join(output_dir, 'pipeline_comparison_boxplots.pdf')
    plt.savefig(plot_path_pdf, bbox_inches='tight')

    plt.close()


def plot_violinplot_comparisons(agg_df, metrics, output_dir):
    """
    generate violin plots for pipeline comparison with consistent style.

    args:
        agg_df: DataFrame with columns ['trial', 'pipeline', metric1, metric2, ...]
        metrics: list of metric names to plot
        output_dir: directory to save plots
    """
    # rename pipelines for display
    df = agg_df.copy()
    df['pipeline'] = df['pipeline'].map(PIPELINE_LABELS)

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # prepare data for violin plot
        pipelines = ['SWT-R', 'SWT-D', 'DWT-R']
        data = [
            df[df['pipeline'] == p][metric].dropna().values
            for p in pipelines
        ]
        colors = [PIPELINE_COLORS[p] for p in pipelines]

        # create violin plot
        parts = ax.violinplot(
            data,
            positions=range(1, len(pipelines) + 1),
            widths=0.55,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )

        # apply colors
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.35)

        # add quartiles and medians manually
        for i, vals in enumerate(data, start=1):
            quartile1, median, quartile3 = np.percentile(vals, [25, 50, 75])
            ax.vlines(i, quartile1, quartile3, color='k', linestyle='-', lw=5)
            ax.vlines(i, np.min(vals), np.max(vals), color='k', linestyle='-', lw=1)
            ax.scatter(i, median, color='white', s=30, zorder=3, edgecolors='k')

        # formatting
        ax.set_xticks(range(1, len(pipelines) + 1))
        ax.set_xticklabels(pipelines, fontsize=11)

        ax.set_xlabel('Pipeline', fontsize=12, fontweight='bold')
        ax.set_ylabel(
            METRIC_LABELS.get(metric, metric),
            fontsize=12,
            fontweight='bold'
        )

        ax.set_title(
            METRIC_LABELS.get(metric, metric),
            fontsize=13,
            fontweight='bold'
        )

        ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    # save
    plot_path = os.path.join(output_dir, 'pipeline_comparison_violinplots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    plot_path_pdf = os.path.join(output_dir, 'pipeline_comparison_violinplots.pdf')
    plt.savefig(plot_path_pdf, bbox_inches='tight')

    plt.close()