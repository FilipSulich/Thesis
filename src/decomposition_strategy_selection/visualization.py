"""Visualization for decomposition strategy comparison."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


class DecompositionVisualizer:
    """Handles plotting for decomposition analysis results."""

    PIPELINE_LABELS = {
        'swt_recon': 'SWT-R',
        'swt_direct': 'SWT-D',
        'dwt_recon': 'DWT-R',
    }

    PIPELINE_COLORS = {
        'SWT-R': '#1f77b4',
        'SWT-D': '#d62728',
        'DWT-R': '#2ca02c',
    }

    METRIC_LABELS = {
        'det_deviation': 'DET Deviation',
        'lam_deviation': 'LAM Deviation',
        'tt_deviation': 'TT Deviation',
        'entr_deviation': 'ENTR Deviation',
        'hamming_distance': 'Hamming Distance',
        'rp_pattern_corr': 'RP Pattern Correlation'
    }

    def __init__(self, output_dir):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_boxplot_comparisons(self, agg_df, metrics):
        """
        Generate box plots for pipeline comparison with consistent style.
        Matches create_auc_boxplot style from compare_roc_curves.

        Args:
            agg_df: DataFrame with columns ['trial', 'pipeline', metric1, metric2, ...]
            metrics: List of metric names to plot
        """
        print("\nGenerating box plots...")

        # Rename pipelines for display
        df = agg_df.copy()
        df['pipeline'] = df['pipeline'].map(self.PIPELINE_LABELS)

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # Prepare data for boxplot
            pipelines = ['SWT-R', 'SWT-D', 'DWT-R']
            data = [
                df[df['pipeline'] == p][metric].dropna().values
                for p in pipelines
            ]
            colors = [self.PIPELINE_COLORS[p] for p in pipelines]

            # Create boxplot
            box = ax.boxplot(
                data,
                patch_artist=True,
                widths=0.55,
                showfliers=False
            )

            # Apply colors
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.35)

            # Formatting
            ax.set_xticks(range(1, len(pipelines) + 1))
            ax.set_xticklabels(pipelines, fontsize=11)

            ax.set_xlabel('Pipeline', fontsize=12, fontweight='bold')
            ax.set_ylabel(
                self.METRIC_LABELS.get(metric, metric),
                fontsize=12,
                fontweight='bold'
            )

            ax.set_title(
                self.METRIC_LABELS.get(metric, metric),
                fontsize=13,
                fontweight='bold'
            )

            ax.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()

        # Save
        plot_path = os.path.join(self.output_dir, 'pipeline_comparison_boxplots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')

        plot_path_pdf = os.path.join(self.output_dir, 'pipeline_comparison_boxplots.pdf')
        plt.savefig(plot_path_pdf, bbox_inches='tight')

        plt.close()
        print(f"  Box plot saved to {plot_path}")