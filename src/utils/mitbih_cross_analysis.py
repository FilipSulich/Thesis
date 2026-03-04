"""
MIT-BIH Cross-Analysis: Compare RQA metrics across lead times and scales.

Provides comprehensive comparison tools for:
1. Lead time progression (how metrics change as AF approaches)
2. Single-scale vs Multi-scale comparison
3. Subband-specific temporal patterns
"""

import csv
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr

from mitbih_rqa_statistics import MITBIHRQAStatistics


class MITBIHCrossAnalysis:
    """
    Cross-analysis for MIT-BIH RQA results.

    Analyzes temporal trends and scale comparisons.
    """

    RQA_METRICS = [
        "recurrence_rate",
        "determinism",
        "average_diagonal_line",
        "longest_diagonal_line",
        "divergence",
        "entropy_diagonal_lines",
        "laminarity",
        "trapping_time",
        "longest_vertical_line",
        "entropy_vertical_lines",
    ]

    def __init__(self, results_dir: str):
        """
        Initialize cross-analysis.

        Args:
            results_dir: Directory containing RQA results and summaries
        """
        self.results_dir = Path(results_dir)
        self.lead_times = [5, 10, 15, 30, 45]

    # =========================================================================
    # 1. Load Summary Data
    # =========================================================================

    def load_summary_data(self, analysis_type: str = 'multi_scale') -> Dict[int, Dict[str, pd.DataFrame]]:
        """
        Load all summary CSV files.

        Args:
            analysis_type: 'single_scale' or 'multi_scale'

        Returns:
            Dictionary: {lead_time: {prefix: DataFrame}}
        """
        data = {}

        if analysis_type == 'single_scale':
            prefixes = ['signal']
        else:
            prefixes = ['D1', 'D2', 'D3', 'D4', 'A4']

        for lt in self.lead_times:
            data[lt] = {}
            for prefix in prefixes:
                csv_path = self.results_dir / f'summary_{prefix}_lt{lt}.csv'
                if csv_path.exists():
                    data[lt][prefix] = pd.read_csv(csv_path)
                else:
                    print(f"Warning: {csv_path} not found")

        return data

    # =========================================================================
    # 2. Temporal Progression Analysis
    # =========================================================================

    def plot_temporal_progression(
        self,
        metric: str,
        prefix: str = 'D4',
        save_path: Optional[Path] = None
    ):
        """
        Plot how a metric changes across lead times.

        Shows mean and error bars for pre-AF and normal groups.
        """
        data = self.load_summary_data('multi_scale')

        lead_times = []
        preaf_means = []
        preaf_stds = []
        normal_means = []
        normal_stds = []
        p_values = []

        for lt in self.lead_times:
            if lt not in data or prefix not in data[lt]:
                continue

            df = data[lt][prefix]
            row = df[df['metric'] == metric]

            if row.empty:
                continue

            lead_times.append(lt)
            preaf_means.append(row['preaf_mean'].values[0])
            preaf_stds.append(row['preaf_std'].values[0])
            normal_means.append(row['normal_mean'].values[0])
            normal_stds.append(row['normal_std'].values[0])
            p_values.append(row['p_value'].values[0])

        if not lead_times:
            print(f"No data for {metric} in {prefix}")
            return

        # Plot
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

        # Main plot
        ax1.errorbar(lead_times, preaf_means, yerr=preaf_stds,
                    marker='o', linewidth=2, capsize=5, label='Pre-AF', color='red')
        ax1.errorbar(lead_times, normal_means, yerr=normal_stds,
                    marker='s', linewidth=2, capsize=5, label='Normal', color='blue')

        ax1.set_ylabel(f'{metric}', fontsize=12)
        ax1.set_title(f'{prefix} - {metric} Progression Across Lead Times', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_all_metrics_progression(
        self,
        prefix: str = 'D4',
        save_dir: Optional[Path] = None,
        significant_only: bool = False
    ):
        """
        Plot temporal progression for all metrics in a grid.
        """
        data = self.load_summary_data('multi_scale')

        # Determine which metrics to plot
        metrics_to_plot = []
        for metric in self.RQA_METRICS:
            # Check if metric has any significant results
            has_significant = False
            for lt in self.lead_times:
                if lt in data and prefix in data[lt]:
                    df = data[lt][prefix]
                    row = df[df['metric'] == metric]
                    if not row.empty and row['p_value'].values[0] < 0.05:
                        has_significant = True
                        break

            if not significant_only or has_significant:
                metrics_to_plot.append(metric)

        if not metrics_to_plot:
            print("No metrics to plot")
            return

        n_metrics = len(metrics_to_plot)
        ncols = 3
        nrows = (n_metrics + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
        axes = np.atleast_1d(axes).flatten()

        for idx, metric in enumerate(metrics_to_plot):
            lead_times = []
            preaf_means = []
            normal_means = []

            for lt in self.lead_times:
                if lt not in data or prefix not in data[lt]:
                    continue
                df = data[lt][prefix]
                row = df[df['metric'] == metric]
                if row.empty:
                    continue

                lead_times.append(lt)
                preaf_means.append(row['preaf_mean'].values[0])
                normal_means.append(row['normal_mean'].values[0])

            if lead_times:
                axes[idx].plot(lead_times, preaf_means, 'ro-', linewidth=2, label='Pre-AF', markersize=6)
                axes[idx].plot(lead_times, normal_means, 'bs-', linewidth=2, label='Normal', markersize=6)
                axes[idx].set_xlabel('Lead Time (min)', fontsize=9)
                axes[idx].set_ylabel(metric, fontsize=9)
                axes[idx].set_title(metric.replace('_', ' ').title(), fontsize=10, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].set_xticks(lead_times)
                if idx == 0:
                    axes[idx].legend(fontsize=9)

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'{prefix} - Temporal Progression of RQA Metrics',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f'{prefix}_temporal_progression.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
            plt.close()
        else:
            plt.show()

    # =========================================================================
    # 3. Single-Scale vs Multi-Scale Comparison
    # =========================================================================

    def compare_scales(
        self,
        metric: str,
        lead_time: int = 5,
        save_path: Optional[Path] = None
    ):
        """
        Compare single-scale vs multi-scale for a specific metric and lead time.

        Shows which subbands differ from full signal analysis.
        """
        # Load single-scale data
        single_data = self.load_summary_data('single_scale')
        multi_data = self.load_summary_data('multi_scale')

        if lead_time not in single_data or lead_time not in multi_data:
            print(f"Data not available for lead time {lead_time}")
            return

        # Get single-scale value
        single_df = single_data[lead_time].get('signal')
        if single_df is None:
            print("Single-scale data not found")
            return

        single_row = single_df[single_df['metric'] == metric]
        if single_row.empty:
            print(f"Metric {metric} not found in single-scale data")
            return

        single_preaf = single_row['preaf_mean'].values[0]
        single_normal = single_row['normal_mean'].values[0]
        single_p = single_row['p_value'].values[0]

        # Get multi-scale values
        subbands = ['D1', 'D2', 'D3', 'D4', 'A4']
        multi_preaf = []
        multi_normal = []
        multi_p = []
        available_subbands = []

        for subband in subbands:
            if subband not in multi_data[lead_time]:
                continue
            df = multi_data[lead_time][subband]
            row = df[df['metric'] == metric]
            if row.empty:
                continue

            available_subbands.append(subband)
            multi_preaf.append(row['preaf_mean'].values[0])
            multi_normal.append(row['normal_mean'].values[0])
            multi_p.append(row['p_value'].values[0])

        if not available_subbands:
            print("No multi-scale data found")
            return

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pre-AF comparison
        x_pos = np.arange(len(available_subbands) + 1)
        all_preaf = [single_preaf] + multi_preaf
        colors_preaf = ['darkred'] + ['red']*len(available_subbands)

        bars1 = ax1.bar(x_pos, all_preaf, color=colors_preaf, alpha=0.7, edgecolor='black')
        ax1.set_ylabel(f'{metric}', fontsize=12)
        ax1.set_title(f'Pre-AF: {metric} (Lead Time {lead_time} min)', fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(['Single-Scale'] + available_subbands, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')

        # Normal comparison
        all_normal = [single_normal] + multi_normal
        colors_normal = ['darkblue'] + ['blue']*len(available_subbands)

        bars2 = ax2.bar(x_pos, all_normal, color=colors_normal, alpha=0.7, edgecolor='black')
        ax2.set_ylabel(f'{metric}', fontsize=12)
        ax2.set_title(f'Normal: {metric} (Lead Time {lead_time} min)', fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['Single-Scale'] + available_subbands, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add p-value annotations
        all_p = [single_p] + multi_p
        for i, (bar, p_val) in enumerate(zip(bars1, all_p)):
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = ''

            if sig:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        sig, ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
            plt.close()
        else:
            plt.show()

    # =========================================================================
    # 4. Heatmap Visualizations
    # =========================================================================

    def plot_heatmap_temporal(
        self,
        group: str = 'preaf',
        analysis_type: str = 'multi_scale',
        save_path: Optional[Path] = None
    ):
        """
        Heatmap showing metric values across lead times and subbands/scales.

        Args:
            group: 'preaf' or 'normal'
            analysis_type: 'single_scale' or 'multi_scale'
        """
        data = self.load_summary_data(analysis_type)

        if analysis_type == 'single_scale':
            prefixes = ['signal']
        else:
            prefixes = ['D1', 'D2', 'D3', 'D4', 'A4']

        # Build matrix: rows = metrics, columns = lead_times x prefixes
        column_names = []
        for lt in self.lead_times:
            for prefix in prefixes:
                column_names.append(f'{prefix}_lt{lt}')

        matrix = np.zeros((len(self.RQA_METRICS), len(column_names)))

        for i, metric in enumerate(self.RQA_METRICS):
            col_idx = 0
            for lt in self.lead_times:
                for prefix in prefixes:
                    if lt in data and prefix in data[lt]:
                        df = data[lt][prefix]
                        row = df[df['metric'] == metric]
                        if not row.empty:
                            if group == 'preaf':
                                matrix[i, col_idx] = row['preaf_mean'].values[0]
                            else:
                                matrix[i, col_idx] = row['normal_mean'].values[0]
                    col_idx += 1

        # Plot
        fig, ax = plt.subplots(figsize=(max(12, len(column_names)*0.5), 10))

        sns.heatmap(matrix, annot=False, cmap='RdYlBu_r', center=0,
                   xticklabels=column_names, yticklabels=self.RQA_METRICS,
                   cbar_kws={'label': f'{group.capitalize()} Mean Value'},
                   linewidths=0.5, ax=ax)

        ax.set_title(f'{group.capitalize()} - RQA Metrics Heatmap ({analysis_type})',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Subband & Lead Time', fontsize=12)
        ax.set_ylabel('RQA Metric', fontsize=12)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_significance_heatmap(
        self,
        analysis_type: str = 'multi_scale',
        save_path: Optional[Path] = None
    ):
        """
        Heatmap showing p-values across lead times and subbands.

        Color indicates significance level.
        """
        data = self.load_summary_data(analysis_type)

        if analysis_type == 'single_scale':
            prefixes = ['signal']
        else:
            prefixes = ['D1', 'D2', 'D3', 'D4', 'A4']

        # Build matrix
        column_names = []
        for lt in self.lead_times:
            for prefix in prefixes:
                column_names.append(f'{prefix}_lt{lt}')

        matrix = np.ones((len(self.RQA_METRICS), len(column_names)))  # Default to 1 (not significant)

        for i, metric in enumerate(self.RQA_METRICS):
            col_idx = 0
            for lt in self.lead_times:
                for prefix in prefixes:
                    if lt in data and prefix in data[lt]:
                        df = data[lt][prefix]
                        row = df[df['metric'] == metric]
                        if not row.empty:
                            matrix[i, col_idx] = row['p_value'].values[0]
                    col_idx += 1

        # Convert to -log10 for better visualization
        matrix_log = -np.log10(matrix + 1e-10)  # Add small value to avoid log(0)

        # Plot
        fig, ax = plt.subplots(figsize=(max(12, len(column_names)*0.5), 10))

        sns.heatmap(matrix_log, annot=False, cmap='RdYlGn',
                   xticklabels=column_names, yticklabels=self.RQA_METRICS,
                   cbar_kws={'label': '-log10(p-value)'},
                   linewidths=0.5, ax=ax, vmin=0, vmax=3)

        # Add reference lines for significance thresholds
        ax.axhline(0, color='white', linewidth=2)

        ax.set_title(f'Statistical Significance Heatmap ({analysis_type})',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Subband & Lead Time', fontsize=12)
        ax.set_ylabel('RQA Metric', fontsize=12)

        # Add text annotation for significance levels
        ax.text(1.02, 0.95, 'p < 0.001: > 3.0', transform=ax.transAxes, fontsize=10)
        ax.text(1.02, 0.90, 'p < 0.01: > 2.0', transform=ax.transAxes, fontsize=10)
        ax.text(1.02, 0.85, 'p < 0.05: > 1.3', transform=ax.transAxes, fontsize=10)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
            plt.close()
        else:
            plt.show()

    # =========================================================================
    # 5. Summary Report
    # =========================================================================

    def generate_summary_report(self, output_path: Optional[Path] = None):
        """
        Generate comprehensive text report of findings.
        """
        if output_path is None:
            output_path = self.results_dir / 'cross_analysis_summary.txt'

        multi_data = self.load_summary_data('multi_scale')
        single_data = self.load_summary_data('single_scale')

        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MIT-BIH CROSS-ANALYSIS SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")

            # 1. Significant findings per lead time
            f.write("1. SIGNIFICANT FINDINGS BY LEAD TIME (p < 0.05)\n")
            f.write("-"*80 + "\n\n")

            for lt in self.lead_times:
                f.write(f"Lead Time: {lt} minutes\n")
                f.write("-"*40 + "\n")

                if lt not in multi_data:
                    f.write("  No data available\n\n")
                    continue

                for prefix in ['D1', 'D2', 'D3', 'D4', 'A4']:
                    if prefix not in multi_data[lt]:
                        continue

                    df = multi_data[lt][prefix]
                    sig_metrics = df[df['p_value'] < 0.05]

                    if not sig_metrics.empty:
                        f.write(f"\n  {prefix}:\n")
                        for _, row in sig_metrics.iterrows():
                            f.write(f"    - {row['metric']}: p={row['p_value']:.4f}, ")
                            f.write(f"Cohen's d={row['cohens_d']:.3f}\n")

                f.write("\n")

            # 2. Temporal trends
            f.write("\n2. TEMPORAL TRENDS (Spearman correlation with lead time)\n")
            f.write("-"*80 + "\n\n")

            for prefix in ['D1', 'D2', 'D3', 'D4', 'A4']:
                f.write(f"{prefix}:\n")

                for metric in self.RQA_METRICS:
                    lead_times_available = []
                    preaf_values = []

                    for lt in self.lead_times:
                        if lt in multi_data and prefix in multi_data[lt]:
                            df = multi_data[lt][prefix]
                            row = df[df['metric'] == metric]
                            if not row.empty:
                                lead_times_available.append(lt)
                                preaf_values.append(row['preaf_mean'].values[0])

                    if len(lead_times_available) >= 3:
                        corr, p_val = spearmanr(lead_times_available, preaf_values)
                        if abs(corr) > 0.5 and p_val < 0.10:
                            f.write(f"  - {metric}: r={corr:.3f}, p={p_val:.4f}\n")

                f.write("\n")

            f.write("\n" + "="*80 + "\n")
            f.write("Report generation complete\n")
            f.write("="*80 + "\n")

        print(f"Summary report saved to {output_path}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    RESULTS_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/mitbih/results')
    OUTPUT_DIR = RESULTS_DIR / 'cross_analysis'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize
    analysis = MITBIHCrossAnalysis(str(RESULTS_DIR))

    print("="*80)
    print("MIT-BIH CROSS-ANALYSIS")
    print("="*80)

    prefix = 'A4'
    # 1. Generate summary report
    print("\n1. Generating summary report...")
    analysis.generate_summary_report(OUTPUT_DIR / 'summary_report.txt')

    # 2. Plot temporal progression for D4 (most significant subband)
    print("\n2. Plotting temporal progression for D4...")
    analysis.plot_all_metrics_progression(
        prefix=prefix,
        save_dir=OUTPUT_DIR,
        significant_only=False
    )

    # 3. Plot temporal progression for specific metrics
    print("\n3. Plotting specific metric progressions...")

    for metric in ['longest_vertical_line', 'recurrence_rate', 'determinism']:
        analysis.plot_temporal_progression(
            metric=metric,
            prefix=prefix,
            save_path=OUTPUT_DIR / f'temporal_{metric}_{prefix}.png'
        )

    # 4. Compare single-scale vs multi-scale
    print("\n4. Comparing single-scale vs multi-scale...")
    for metric in ['recurrence_rate', 'longest_vertical_line']:
        analysis.compare_scales(
            metric=metric,
            lead_time=5,
            save_path=OUTPUT_DIR / f'scale_comparison_{metric}_lt5.png'
        )

    # 5. Generate heatmaps
    print("\n5. Generating heatmaps...")
    analysis.plot_heatmap_temporal(
        group='preaf',
        analysis_type='multi_scale',
        save_path=OUTPUT_DIR / 'heatmap_preaf_multi.png'
    )

    analysis.plot_significance_heatmap(
        analysis_type='multi_scale',
        save_path=OUTPUT_DIR / 'heatmap_significance_multi.png'
    )

    print("\n" + "="*80)
    print(f"Cross-analysis complete! Results saved to: {OUTPUT_DIR}")
    print("="*80)
