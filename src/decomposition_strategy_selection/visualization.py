"""Visualization for decomposition strategy comparison."""

import os
import matplotlib.pyplot as plt
import seaborn as sns


class DecompositionVisualizer:
    """Handles plotting for decomposition analysis results."""

    PIPELINE_LABELS = {
        'swt_recon': 'SWT-R',
        'swt_direct': 'SWT-D',
        'dwt_recon': 'DWT-R',
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
        sns.set_theme(style="whitegrid", font_scale=1.1)

    def plot_violin_comparisons(self, agg_df, metrics):
        """
        Generate violin plots for pipeline comparison.

        Args:
            agg_df: DataFrame with columns ['trial', 'pipeline', metric1, metric2, ...]
            metrics: List of metric names to plot
        """
        print("\nGenerating violin plots...")

        # Rename pipelines for display
        df = agg_df.copy()
        df['pipeline'] = df['pipeline'].map(self.PIPELINE_LABELS)

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            sns.violinplot(
                data=df,
                x='pipeline',
                y=metric,
                ax=ax,
                color='steelblue'  # Single default color
            )

            ax.set_title(self.METRIC_LABELS.get(metric, metric),
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Pipeline', fontsize=12)
            ax.set_ylabel(self.METRIC_LABELS.get(metric, metric), fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'pipeline_comparison_violins.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Violin plot saved to {plot_path}")

    def plot_boxplot_comparisons(self, agg_df, metrics):
        """
        Generate box plots for pipeline comparison.

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

            sns.boxplot(
                data=df,
                x='pipeline',
                y=metric,
                ax=ax,
                color='steelblue',  # Single default color
                showfliers=True
            )

            ax.set_title(self.METRIC_LABELS.get(metric, metric),
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Pipeline', fontsize=12)
            ax.set_ylabel(self.METRIC_LABELS.get(metric, metric), fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'pipeline_comparison_boxplots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Box plot saved to {plot_path}")