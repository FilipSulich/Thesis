"""
MIT-BIH RQA Statistics

Patient-level statistical analysis for MIT-BIH RQA results.
Uses paired statistical tests since we compare pre-AF vs normal within same patients.
"""

import csv
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MITBIHRQAStatistics:
    """
    Statistical analysis for MIT-BIH RQA results.

    Performs patient-level paired comparisons (pre-AF vs normal within patients).
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

    def __init__(self, csv_path: str):
        """
        Initialize statistics analyzer.

        Args:
            csv_path: Path to RQA results CSV file
        """
        self.csv_path = csv_path
        self.data = self._load_data()
        self.lead_time = self._get_lead_time()

    def _load_data(self) -> List[Dict]:
        """Load data from CSV."""
        with open(self.csv_path, "r") as f:
            return list(csv.DictReader(f))

    def _get_lead_time(self) -> int:
        """Get lead time from data."""
        if self.data:
            return int(self.data[0].get('lead_time', 0))
        return 0

    def _aggregate_per_patient(self, metric: str, prefix: str = "signal") -> Dict:
        """
        Aggregate RQA metric values per patient.

        For each patient, computes mean across all segments (pre-AF and normal separately).

        Args:
            metric: RQA metric name
            prefix: Metric prefix ('signal', 'D1', 'D2', etc.)

        Returns:
            dict: patient_id -> {"preaf": [values], "normal": [values]}
        """
        col_name = f"{prefix}_{metric}"
        patient_dict = {}

        for row in self.data:
            if col_name not in row or not row[col_name]:
                continue
            if row[col_name].strip().lower() == "nan":
                continue

            patient_id = row["patient_id"]
            label = int(row["label"])
            value = float(row[col_name])

            if patient_id not in patient_dict:
                patient_dict[patient_id] = {"preaf": [], "normal": []}

            if label == 1:
                patient_dict[patient_id]["preaf"].append(value)
            else:
                patient_dict[patient_id]["normal"].append(value)

        # Average within each patient's group
        result = {}
        for patient_id, data in patient_dict.items():
            if data["preaf"] and data["normal"]:  # Only keep patients with both
                result[patient_id] = {
                    "preaf_mean": float(np.mean(data["preaf"])),
                    "normal_mean": float(np.mean(data["normal"])),
                }

        return result

    def compare_groups_paired(self, metric: str, prefix: str = "signal") -> Dict:
        """
        Compare pre-AF vs normal using paired Wilcoxon test.

        Since we have matched pre-AF and normal segments from the same patients,
        we use a paired test.

        Args:
            metric: RQA metric name
            prefix: Metric prefix ('signal', 'D1', etc.)

        Returns:
            Dictionary with statistics
        """
        patient_data = self._aggregate_per_patient(metric, prefix)

        if not patient_data:
            return {}

        preaf_means = []
        normal_means = []

        for patient_id, data in patient_data.items():
            preaf_means.append(data["preaf_mean"])
            normal_means.append(data["normal_mean"])

        preaf = np.array(preaf_means)
        normal = np.array(normal_means)

        result = {}
        result["n_patients"] = len(preaf)
        result["preaf_mean"] = float(np.mean(preaf))
        result["preaf_std"] = float(np.std(preaf, ddof=1))
        result["preaf_median"] = float(np.median(preaf))
        result["normal_mean"] = float(np.mean(normal))
        result["normal_std"] = float(np.std(normal, ddof=1))
        result["normal_median"] = float(np.median(normal))

        # Wilcoxon signed-rank test (paired)
        try:
            stat, p_value = wilcoxon(preaf, normal, alternative='two-sided')
            result["p_value"] = float(p_value)
            result["test"] = "wilcoxon"
        except Exception as e:
            # If Wilcoxon fails (e.g., all differences zero), try Mann-Whitney
            try:
                stat, p_value = mannwhitneyu(preaf, normal, alternative='two-sided')
                result["p_value"] = float(p_value)
                result["test"] = "mannwhitneyu (fallback)"
            except:
                result["p_value"] = float('nan')
                result["test"] = "failed"

        # Paired Cohen's d (using difference scores)
        diff = preaf - normal
        result["cohens_d"] = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-8))

        # Store patient-level values for plotting
        result["preaf_values"] = preaf.tolist()
        result["normal_values"] = normal.tolist()
        result["patient_ids"] = list(patient_data.keys())

        return result

    def print_summary(self, prefix: str = "signal", significant_only: bool = False, alpha: float = 0.05):
        """
        Print summary statistics table.

        Args:
            prefix: Metric prefix ('signal', 'D1', etc.)
            significant_only: If True, only show metrics with p < alpha
            alpha: Significance level
        """
        print(f"\n{'='*100}")
        print(f"RQA Statistics Summary - Lead Time: {self.lead_time} min - Prefix: {prefix}")
        print(f"{'='*100}")
        print(f"{'Metric':<30} {'N':>5} {'Pre-AF (mean±std)':>22} {'Normal (mean±std)':>22} "
              f"{'Pre-AF (med)':>14} {'Normal (med)':>14} {'Cohen d':>9} {'P-value':>10} {'Test':>12}")
        print("-" * 100)

        for metric in self.RQA_METRICS:
            data = self.compare_groups_paired(metric, prefix)
            if not data:
                continue

            # Skip if not significant and significant_only is True
            if significant_only and data.get('p_value', 1.0) >= alpha:
                continue

            preaf_str = f"{data['preaf_mean']:.4f}±{data['preaf_std']:.4f}"
            normal_str = f"{data['normal_mean']:.4f}±{data['normal_std']:.4f}"
            preaf_median_str = f"{data['preaf_median']:.4f}"
            normal_median_str = f"{data['normal_median']:.4f}"

            # Mark significant results
            p_val = data.get('p_value', float('nan'))
            p_str = f"{p_val:.4f}"
            if p_val < 0.001:
                p_str += " ***"
            elif p_val < 0.01:
                p_str += " **"
            elif p_val < 0.05:
                p_str += " *"

            print(
                f"{metric:<30} "
                f"{data['n_patients']:>5} "
                f"{preaf_str:>22} "
                f"{normal_str:>22} "
                f"{preaf_median_str:>14} "
                f"{normal_median_str:>14} "
                f"{data['cohens_d']:>9.3f} "
                f"{p_str:>10} "
                f"{data.get('test', 'N/A'):>12}"
            )

        print("-" * 100)
        print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
        print()

    def plot_metric_comparison(
        self,
        metric: str,
        prefix: str = "signal",
        save_path: Optional[Path] = None
    ):
        """
        Plot paired comparison for a single metric.

        Shows:
        - Box plots for pre-AF vs normal
        - Individual patient lines connecting paired values
        """
        data = self.compare_groups_paired(metric, prefix)
        if not data:
            print(f"No data for {prefix}_{metric}")
            return

        preaf = np.array(data['preaf_values'])
        normal = np.array(data['normal_values'])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Box plot
        box_data = [normal, preaf]
        bp = axes[0].boxplot(box_data, labels=['Normal', 'Pre-AF'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')

        axes[0].set_ylabel(f'{prefix}_{metric}')
        axes[0].set_title(f'Distribution (Lead Time: {self.lead_time} min)')
        axes[0].grid(True, alpha=0.3)

        # Add significance annotation
        p_val = data.get('p_value', float('nan'))
        sig_text = f"p = {p_val:.4f}"
        if p_val < 0.001:
            sig_text += " ***"
        elif p_val < 0.01:
            sig_text += " **"
        elif p_val < 0.05:
            sig_text += " *"

        axes[0].text(0.5, 0.95, sig_text, transform=axes[0].transAxes,
                    ha='center', va='top', fontsize=11, fontweight='bold')

        # Paired plot
        for i in range(len(preaf)):
            axes[1].plot([0, 1], [normal[i], preaf[i]], 'o-', alpha=0.5, color='gray')

        axes[1].plot([-0.2, 1.2], [np.mean(normal), np.mean(preaf)], 'ro-', linewidth=2, markersize=8, label='Mean')
        axes[1].set_xlim(-0.3, 1.3)
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(['Normal', 'Pre-AF'])
        axes[1].set_ylabel(f'{prefix}_{metric}')
        axes[1].set_title(f'Paired Comparison (n={len(preaf)} patients)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.suptitle(f'{metric} - Lead Time {self.lead_time} min', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    def plot_all_metrics(
        self,
        prefix: str = "signal",
        save_dir: Optional[Path] = None,
        significant_only: bool = False,
        alpha: float = 0.05
    ):
        """
        Plot all metrics in a grid.

        Args:
            prefix: Metric prefix
            save_dir: Directory to save plots
            significant_only: Only plot significant metrics
            alpha: Significance level
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        metrics_to_plot = []
        for metric in self.RQA_METRICS:
            data = self.compare_groups_paired(metric, prefix)
            if not data:
                continue
            if significant_only and data.get('p_value', 1.0) >= alpha:
                continue
            metrics_to_plot.append(metric)

        if not metrics_to_plot:
            print("No metrics to plot.")
            return

        n_metrics = len(metrics_to_plot)
        ncols = 3
        nrows = (n_metrics + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        axes = np.atleast_1d(axes).flatten()

        for i, metric in enumerate(metrics_to_plot):
            data = self.compare_groups_paired(metric, prefix)
            preaf = np.array(data['preaf_values'])
            normal = np.array(data['normal_values'])

            # Paired plot
            for j in range(len(preaf)):
                axes[i].plot([0, 1], [normal[j], preaf[j]], 'o-', alpha=0.3, color='gray', markersize=4)

            axes[i].plot([0, 1], [np.mean(normal), np.mean(preaf)], 'ro-', linewidth=2, markersize=8)
            axes[i].set_xlim(-0.3, 1.3)
            axes[i].set_xticks([0, 1])
            axes[i].set_xticklabels(['Normal', 'Pre-AF'], fontsize=9)
            axes[i].set_ylabel(metric, fontsize=9)
            axes[i].grid(True, alpha=0.3)

            # Add p-value
            p_val = data.get('p_value', float('nan'))
            sig_text = f"p={p_val:.3f}"
            if p_val < 0.001:
                sig_text += "***"
            elif p_val < 0.01:
                sig_text += "**"
            elif p_val < 0.05:
                sig_text += "*"

            axes[i].set_title(sig_text, fontsize=10, fontweight='bold')

        # Hide unused axes
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'{prefix} - Lead Time {self.lead_time} min (n={len(preaf)} patients)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_dir:
            save_path = save_dir / f'{prefix}_all_metrics_lt{self.lead_time}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    def export_summary_csv(self, prefix: str = "signal", output_path: Optional[Path] = None):
        """
        Export summary statistics to CSV.

        Args:
            prefix: Metric prefix
            output_path: Output CSV path
        """
        if output_path is None:
            output_path = Path(self.csv_path).parent / f'summary_{prefix}_lt{self.lead_time}.csv'

        rows = []
        for metric in self.RQA_METRICS:
            data = self.compare_groups_paired(metric, prefix)
            if not data:
                continue

            row = {
                'metric': metric,
                'prefix': prefix,
                'lead_time': self.lead_time,
                'n_patients': data['n_patients'],
                'preaf_mean': data['preaf_mean'],
                'preaf_std': data['preaf_std'],
                'preaf_median': data['preaf_median'],
                'normal_mean': data['normal_mean'],
                'normal_std': data['normal_std'],
                'normal_median': data['normal_median'],
                'cohens_d': data['cohens_d'],
                'p_value': data.get('p_value', float('nan')),
                'test': data.get('test', 'N/A'),
            }
            rows.append(row)

        if not rows:
            print("No data to export.")
            return

        with open(output_path, 'w', newline='') as f:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Exported summary to {output_path}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    import sys

    # Configuration
    LEAD_TIME = 45
    DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/mitbih/results')

    # Analysis type
    ANALYSIS_TYPE = 'multi_scale'  # 'single_scale' or 'multi_scale'
    embedding_dim = 7
    time_delay = 21
    path  = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/mitbih/results/rqa_multi_scale_lt5_tau_21_embed_7.csv"

    # Input file
    if ANALYSIS_TYPE == 'single_scale':
        CSV_PATH = DATA_DIR / f'rqa_single_scale_lt{LEAD_TIME}_tau_{time_delay}_embed_{embedding_dim}.csv'
        PREFIXES = ['signal']
    else:
        CSV_PATH = DATA_DIR / f'rqa_multi_scale_lt{LEAD_TIME}_tau_{time_delay}_embed_{embedding_dim}.csv'
        PREFIXES = ['D1', 'D2', 'D3', 'D4', 'A4']

    print(f"\n{'='*100}")
    print(f"MIT-BIH RQA Statistical Analysis")
    print(f"Analysis: {ANALYSIS_TYPE}")
    print(f"Lead Time: {LEAD_TIME} minutes")
    print(f"Input: {CSV_PATH}")
    print(f"{'='*100}\n")

    # Initialize statistics
    stats = MITBIHRQAStatistics(str(CSV_PATH))

    # Print summary for each prefix
    for prefix in PREFIXES:
        stats.print_summary(prefix=prefix, significant_only=False)

        # Export summary
        stats.export_summary_csv(prefix=prefix)

        # Plot all metrics
        # stats.plot_all_metrics(prefix=prefix, save_dir=DATA_DIR / 'plots')

    print("\nStatistical analysis complete!")
