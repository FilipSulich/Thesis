"""Pipeline comparison experiment runner with statistical analysis."""

import os
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

from src.decomposition_strategy_selection.decomposition_analyzer import DecompositionAnalyzer
from src.decomposition_strategy_selection.visualization import DecompositionVisualizer


class ExperimentRunner:
    """
    Manages pipeline comparison experiments and statistical analysis.

    Runs multiple trials, performs statistical testing, and generates visualizations.
    """

    METRICS = [
        'det_deviation',
        'lam_deviation',
        'tt_deviation',
        # 'entr_deviation',
        'hamming_distance',
        'rp_pattern_corr'
    ]

    def __init__(self, subband_freqs, fs=128, n=1024, level=4, wavelet='db4', output_dir=None):
        """
        Initialize experiment runner.

        Args:
            subband_freqs: Dict mapping band names to frequencies (e.g., {'D4': 6, 'D3': 12, ...})
            fs: Sampling frequency (Hz)
            n: Signal length (samples)
            level: Wavelet decomposition level
            wavelet: Wavelet type (e.g., 'db4')
            output_dir: Directory for saving results
        """
        self.analyzer = DecompositionAnalyzer(
            subband_freqs=subband_freqs,
            fs=fs,
            n=n,
            level=level,
            wavelet=wavelet,
            output_dir=output_dir
        )
        self.output_dir = self.analyzer.OUTPUT_DIR
        self.visualizer = DecompositionVisualizer(output_dir=self.output_dir)

    def run_experiment(self, n_trials=100):
        """
        Run complete pipeline comparison experiment.

        Args:
            n_trials: Number of trials with random phases

        Returns:
            agg_df: Aggregated results DataFrame
            stats_df: Statistical analysis results
        """
        print(f"\n{'='*80}")
        print(f"RUNNING PIPELINE COMPARISON: {n_trials} TRIALS")
        print(f"{'='*80}")
        print(f"Output directory: {self.output_dir}\n")

        # Load or generate ground truth
        gt_path = os.path.join(self.output_dir, 'ground_truth.json')
        if os.path.exists(gt_path):
            print("Loading ground truth...")
            ground_truth, ground_truth_data = self.analyzer.load_ground_truth(gt_path)
        else:
            print("Generating ground truth...")
            ground_truth, ground_truth_data = self.analyzer.generate_ground_truth(gt_path)

        # Run trials
        print(f"\nRunning trials...")
        trial_data = []

        for trial in range(n_trials):
            if (trial + 1) % 10 == 0:
                print(f'  Trial {trial + 1}/{n_trials}')

            # Generate signal with random phase
            sig = self.analyzer.generate_composite_sine(random_phase=True)

            # Analyze each pipeline
            for pipeline in self.analyzer.PIPELINES:
                df = self.analyzer.analyze_pipeline(sig, pipeline, ground_truth, ground_truth_data)

                # Aggregate metrics across bands
                row = {'trial': trial, 'pipeline': pipeline}
                for metric in self.METRICS:
                    if metric in df.columns:
                        row[metric] = df[metric].mean()

                trial_data.append(row)

        # Save results
        agg_df = pd.DataFrame(trial_data)
        results_path = os.path.join(self.output_dir, 'pipeline_comparison_results.csv')
        agg_df.to_csv(results_path, index=False)
        print(f"\n✓ Results saved to {results_path}")

        # Statistical analysis
        stats_df = self._run_statistical_analysis(agg_df)

        # Generate plots
        print("\nGenerating visualizations...")
        self.visualizer.plot_violin_comparisons(agg_df, self.METRICS)
        self.visualizer.plot_boxplot_comparisons(agg_df, self.METRICS)

        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*80}\n")

        return agg_df, stats_df

    def _run_statistical_analysis(self, agg_df):
        """
        Perform statistical tests on pipeline comparison results.

        Tests:
        - Kruskal-Wallis: Overall differences across pipelines
        - Dunn: Pairwise comparisons with Bonferroni correction
        - Cohen's D: Effect sizes

        Args:
            agg_df: Aggregated results DataFrame

        Returns:
            DataFrame with statistical test results
        """
        print("\nRunning statistical analysis...")

        stats_results = []

        for metric in self.METRICS:
            # Prepare data for each pipeline
            groups = [
                agg_df[agg_df['pipeline'] == p][metric].dropna().values
                for p in self.analyzer.PIPELINES
            ]

            # Kruskal-Wallis test
            h_stat, kw_pvalue = kruskal(*groups)

            stats_results.append({
                'metric': metric,
                'test': 'Kruskal-Wallis',
                'pipeline1': 'all',
                'pipeline2': 'all',
                'statistic': h_stat,
                'p_value': kw_pvalue,
                'cohens_d': np.nan,
                'significance': self._get_significance(kw_pvalue)
            })

            # Dunn pairwise comparisons
            dunn_df = posthoc_dunn(agg_df, val_col=metric, group_col='pipeline', p_adjust='bonferroni')

            pipelines = self.analyzer.PIPELINES
            for i, pipe1 in enumerate(pipelines):
                for pipe2 in pipelines[i + 1:]:
                    x = agg_df[agg_df['pipeline'] == pipe1][metric].dropna()
                    y = agg_df[agg_df['pipeline'] == pipe2][metric].dropna()

                    if len(x) < 3 or len(y) < 3:
                        continue

                    # Cohen's D
                    pooled_std = np.sqrt(
                        ((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) /
                        (len(x) + len(y) - 2)
                    )
                    cohens_d = (x.mean() - y.mean()) / pooled_std if pooled_std > 1e-9 else np.nan

                    dunn_pvalue = dunn_df.loc[pipe1, pipe2]

                    stats_results.append({
                        'metric': metric,
                        'test': 'Dunn',
                        'pipeline1': pipe1,
                        'pipeline2': pipe2,
                        'statistic': np.nan,
                        'p_value': dunn_pvalue,
                        'cohens_d': cohens_d,
                        'significance': self._get_significance(dunn_pvalue)
                    })

        # Save results
        stats_df = pd.DataFrame(stats_results)
        stats_path = os.path.join(self.output_dir, 'statistical_analysis.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"✓ Statistical analysis saved to {stats_path}")

        return stats_df

    @staticmethod
    def _get_significance(p_value):
        """Convert p-value to significance notation."""
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'


def main():
    """Run pipeline comparison experiment with default parameters."""

    # Harmonic frequencies (all multiples of fundamental)
    SUBBAND_FREQS_HARM = {
        'A4': 2,
        'D4': 6,    # 4-8 Hz
        'D3': 12,   # 8-16 Hz
        'D2': 24,   # 16-32 Hz
        'D1': 48    # 32-64 Hz
    }

    # Non-harmonic frequencies
    SUBBAND_FREQS_NON_HARM = {
        'A4': 3,
        'D4': 7,    # 4-8 Hz
        'D3': 13,   # 8-16 Hz
        'D2': 31,   # 16-32 Hz
        'D1': 61    # 32-64 Hz
    }

    output_dir = '/Users/filipsulich/Developer/uni/Thesis/Thesis/src/decomposition_strategy_selection/results_test_harm2'

    runner = ExperimentRunner(
        subband_freqs=SUBBAND_FREQS_HARM,
        output_dir=output_dir
    )

    # Run experiment
    agg_df, stats_df = runner.run_experiment(n_trials=100)


if __name__ == '__main__':
    main()