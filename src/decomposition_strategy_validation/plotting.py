"""plotting functions for pipeline comparison visualization."""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.decomposition_strategy_validation.config import (
    PIPELINE_LABELS,
    PIPELINE_COLORS,
)


METRIC_LABELS = {
    'det_deviation': 'DET Deviation',
    'lam_deviation': 'LAM Deviation',
    'tt_deviation': 'TT Deviation',
    'entr_deviation': 'ENTR Deviation',
    'hamming_distance': 'Hamming Distance',
    'rp_pattern_corr': 'RP Person Correlation'
}


def plot_boxplot_comparisons(agg_df, metrics, output_dir):
    """
    generate box plots comparing pipeline performance across multiple metrics with consistent styling

    Args:
        agg_df (pandas.DataFrame): aggregated results dataframe with columns ['trial', 'pipeline', metric1, metric2, ...]
        metrics (list): list of metric column names to visualize
        output_dir (str): directory path where plots will be saved as PNG and PDF files
    """
    df = agg_df.copy()
    df['pipeline'] = df['pipeline'].map(PIPELINE_LABELS)

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        pipelines = ['SWT-R', 'SWT-D', 'DWT-R']
        data = [
            df[df['pipeline'] == p][metric].dropna().values
            for p in pipelines
        ]
        colors = [PIPELINE_COLORS[p] for p in pipelines]

        box = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.55,
            showfliers=False
        )

        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)

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

    plot_path = os.path.join(output_dir, 'pipeline_comparison_boxplots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    plot_path_pdf = os.path.join(output_dir, 'pipeline_comparison_boxplots.pdf')
    plt.savefig(plot_path_pdf, bbox_inches='tight')

    plt.close()


def plot_violinplot_comparisons(agg_df, metrics, output_dir):
    """
    generate violin plots showing distribution of pipeline performance across multiple metrics with enhanced visualization

    Args:
        agg_df (pandas.DataFrame): aggregated results dataframe with columns ['trial', 'pipeline', metric1, metric2, ...]
        metrics (list): list of metric column names to visualize
        output_dir (str): directory path where plots will be saved as PNG and PDF files
    """

    df = agg_df.copy()
    df['pipeline'] = df['pipeline'].map(PIPELINE_LABELS)

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 5.5))

    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        pipelines = ['SWT-R', 'SWT-D', 'DWT-R']
        data = [
            df[df['pipeline'] == p][metric].dropna().values
            for p in pipelines
        ]
        colors = [PIPELINE_COLORS[p] for p in pipelines]

        parts = ax.violinplot(
            data,
            positions=range(1, len(pipelines) + 1),
            widths=0.6,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_linewidth(1.2)
            pc.set_alpha(0.4)

        for i, vals in enumerate(data, start=1):
            quartile1, median, quartile3 = np.percentile(vals, [25, 50, 75])
            ax.vlines(i, np.min(vals), np.max(vals), color='#2C3E50', linestyle='-', lw=1, zorder=2)
            ax.vlines(i, quartile1, quartile3, color='#2C3E50', linestyle='-', lw=4.5,  zorder=2)
            ax.scatter(i, median, color='white', s=25, zorder=3, edgecolors='#2C3E50', linewidths=1.5)

        ax.set_xticks(range(1, len(pipelines) + 1))
        ax.set_xticklabels(pipelines, fontsize=10.5, color='#333333')

        ax.set_xlabel('Pipeline', fontsize=11, fontweight='bold', labelpad=10, color='#222222')
        ax.tick_params(axis='both', which='major', labelsize=10, labelcolor='#333333')

        ax.set_title(
            METRIC_LABELS.get(metric, metric),
            fontsize=12,
            fontweight='bold',
            pad=14,
            color='#111111'
        )

        ax.grid(True, alpha=0.25, linestyle='--', color='#999999', zorder=0)
        ax.set_axisbelow(True)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'pipeline_comparison_violinplots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    plot_path_pdf = os.path.join(output_dir, 'pipeline_comparison_violinplots.pdf')
    plt.savefig(plot_path_pdf, bbox_inches='tight')

    plt.close()