"""
Simple ablation study boxplot generator with consistent styling.

Creates side-by-side boxplots for AUC and AUPRC across 4 model configurations:
1. Low freq (D2, D3, D4)
2. High freq (D5, D6, D7, D8)
3. Attention-based (D2, D4, D7)
4. Full model (all 7 subbands)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class AblationBoxplotGenerator:
    """
    Generate consistent-style boxplots for ablation study.

    Matches the styling used in wavelet selection, sine test, etc.
    """

    # Configuration labels and colors
    # NOTE: JSON keys are backwards in cross_validation.py!
    # - 'lower_freq' in JSON → D2-D4 → Actually HIGH frequency
    # - 'higher_freq' in JSON → D5-D8 → Actually LOW frequency
    CONFIGS = {
        'high_freq': {
            'label': 'High Frequency\n(D2, D3, D4)',
            'color': '#1f77b4',  # Blue
            'key': 'lower_freq'  # ← Maps to D2-D4 (high freq)
        },
        'low_freq': {
            'label': 'Low Frequency\n(D5, D6, D7, D8)',
            'color': '#ff7f0e',  # Orange
            'key': 'higher_freq'  # ← Maps to D5-D8 (low freq)
        },
        'attention': {
            'label': 'Attention-Based\n(D2, D4, D7)',
            'color': '#2ca02c',  # Green
            'key': 'selected_freq'
        },
        'full': {
            'label': 'Full Model\n(D2-D8)',
            'color': '#d62728',  # Red
            'key': None  # Loaded from separate file
        }
    }

    def __init__(self, ablation_dir, full_model_dir, output_dir=None):
        """
        Initialize generator.

        Args:
            ablation_dir: Path to ablation study results
            full_model_dir: Path to full model results
            output_dir: Output directory (defaults to ablation_dir)
        """
        self.ablation_dir = Path(ablation_dir)
        self.full_model_dir = Path(full_model_dir)
        self.output_dir = Path(output_dir) if output_dir else self.ablation_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.ablation_data = self._load_ablation_data()
        self.full_model_data = self._load_full_model_data()

    def _load_ablation_data(self):
        """Load ablation study results."""
        ablation_file = self.ablation_dir / 'ablation_study_combined.json'

        if not ablation_file.exists():
            raise FileNotFoundError(f"Ablation results not found: {ablation_file}")

        with open(ablation_file, 'r') as f:
            return json.load(f)

    def _load_full_model_data(self):
        """Load full model results."""
        full_model_file = self.full_model_dir / 'kfold_results.json'

        if not full_model_file.exists():
            raise FileNotFoundError(f"Full model results not found: {full_model_file}")

        with open(full_model_file, 'r') as f:
            return json.load(f)

    def _extract_metric_values(self, metric):
        """
        Extract per-fold values for each configuration.

        Args:
            metric: 'auc_roc' or 'auprc'

        Returns:
            List of (label, values, color) tuples
        """
        data = []

        for config_name, config_info in self.CONFIGS.items():
            label = config_info['label']
            color = config_info['color']
            key = config_info['key']

            if config_name == 'full':
                # Full model from separate file
                values = self.full_model_data['trained_attention']['aggregated'][metric]['values']
            else:
                # Ablation configurations
                values = self.ablation_data[key]['trained_attention']['aggregated'][metric]['values']

            data.append((label, values, color))

        return data

    def plot_metric_boxplot(self, metric, metric_label):
        """
        Create boxplot for a single metric with consistent styling.

        Args:
            metric: Metric key ('auc_roc' or 'auprc')
            metric_label: Display label for metric
        """
        print(f"\nGenerating {metric_label} boxplot...")

        # Extract data
        config_data = self._extract_metric_values(metric)

        # Prepare for boxplot
        labels = [d[0] for d in config_data]
        data = [d[1] for d in config_data]
        colors = [d[2] for d in config_data]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create boxplot with consistent styling
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

        # Individual fold dots with jitter
        rng = np.random.default_rng(42)

        for i, (vals, color) in enumerate(zip(data, colors), start=1):
            x_jitter = rng.normal(
                loc=i,
                scale=0.04,
                size=len(vals)
            )

            ax.scatter(
                x_jitter,
                vals,
                color=color,
                s=55,
                alpha=0.9,
                zorder=3
            )

        # Mean markers
        means = [np.mean(vals) for vals in data]

        ax.plot(
            range(1, len(data) + 1),
            means,
            linestyle='None',
            marker='D',
            markersize=6,
            color='black',
            label='Mean',
            zorder=4
        )

        # Formatting
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=11)

        ax.set_ylabel(
            metric_label,
            fontsize=12,
            fontweight='bold'
        )

        ax.set_title(
            f'Ablation Study: {metric_label} Distribution\n10-Fold Cross-Validation',
            fontsize=13,
            fontweight='bold'
        )

        ax.grid(alpha=0.3, linestyle='--')

        ax.legend(loc='lower right', fontsize=10)

        plt.tight_layout()

        # Save
        output_png = self.output_dir / f'ablation_{metric}_boxplot.png'
        plt.savefig(output_png, dpi=300, bbox_inches='tight')

        output_pdf = self.output_dir / f'ablation_{metric}_boxplot.pdf'
        plt.savefig(output_pdf, bbox_inches='tight')

        plt.close()

        print(f"  ✓ Saved to {output_png}")

    def plot_all(self):
        """Generate boxplots for AUC and AUPRC."""
        print(f"\n{'='*70}")
        print("ABLATION STUDY BOXPLOTS")
        print(f"{'='*70}")

        # AUC boxplot
        self.plot_metric_boxplot('auc_roc', 'AUC-ROC')

        # AUPRC boxplot
        self.plot_metric_boxplot('auprc', 'AUPRC')

        print(f"\n{'='*70}")
        print("COMPLETE")
        print(f"{'='*70}\n")

    def print_statistics(self):
        """Print summary statistics for both metrics."""
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")

        for metric, metric_label in [('auc_roc', 'AUC-ROC'), ('auprc', 'AUPRC')]:
            print(f"\n{metric_label}:")
            print(f"{'-'*70}")
            print(f"{'Configuration':<25} {'Mean':<10} {'Median':<10} {'Std':<10}")
            print(f"{'-'*70}")

            config_data = self._extract_metric_values(metric)

            for label, values, _ in config_data:
                values_arr = np.array(values)
                mean_val = np.mean(values_arr)
                median_val = np.median(values_arr)
                std_val = np.std(values_arr, ddof=1)

                # Clean label for printing
                clean_label = label.replace('\n', ' ')
                print(f"{clean_label:<25} {mean_val:<10.4f} {median_val:<10.4f} {std_val:<10.4f}")

        print(f"\n{'='*70}\n")


if __name__ == '__main__':
    # Configuration
    ABLATION_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_final/results_kfold2/ablation')
    FULL_MODEL_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_final/results_kfold2/multi')
    results_dir = ABLATION_DIR / 'results_final'
    # Generate plots
    generator = AblationBoxplotGenerator(
        ablation_dir=ABLATION_DIR,
        full_model_dir=FULL_MODEL_DIR,
        output_dir=results_dir,
    )

    generator.print_statistics()
    generator.plot_all()
