"""Inspect outliers in pipeline comparison results."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def identify_outliers(data, method='iqr', threshold=1.5):
    """
    Identify outliers in data.

    Args:
        data: 1D array of values
        method: 'iqr' (interquartile range) or 'zscore'
        threshold: For IQR method, multiplier (default 1.5)
                  For zscore, number of standard deviations (default 3)

    Returns:
        Boolean mask where True indicates outlier
    """
    if method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (data < lower) | (data > upper)
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    else:
        raise ValueError("method must be 'iqr' or 'zscore'")


def inspect_pipeline_outliers(results_path, pipeline, metric, output_dir=None):
    """
    Inspect outliers for a specific pipeline and metric.

    Args:
        results_path: Path to pipeline_comparison_results.csv
        pipeline: Pipeline name (e.g., 'swt_recon', 'swt_direct', 'dwt_recon')
        metric: Metric name (e.g., 'tt_deviation')
        output_dir: Directory to save inspection results
    """
    # Load results
    df = pd.read_csv(results_path)

    # Filter for pipeline
    pipeline_df = df[df['pipeline'] == pipeline].copy()

    if metric not in pipeline_df.columns:
        print(f"Error: Metric '{metric}' not found in results")
        return

    # Get metric values
    values = pipeline_df[metric].dropna().values
    trials = pipeline_df[pipeline_df[metric].notna()]['trial'].values

    # Identify outliers
    outlier_mask = identify_outliers(values, method='iqr', threshold=1.5)
    n_outliers = outlier_mask.sum()

    print(f"\n{'='*80}")
    print(f"OUTLIER INSPECTION: {pipeline} - {metric}")
    print(f"{'='*80}\n")

    print(f"Total samples: {len(values)}")
    print(f"Outliers detected (IQR method, threshold=1.5): {n_outliers}")
    print(f"Outlier percentage: {100 * n_outliers / len(values):.2f}%\n")

    # Statistics
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1

    print("Distribution statistics:")
    print(f"  Min:    {values.min():.6f}")
    print(f"  Q1:     {q1:.6f}")
    print(f"  Median: {median:.6f}")
    print(f"  Q3:     {q3:.6f}")
    print(f"  Max:    {values.max():.6f}")
    print(f"  IQR:    {iqr:.6f}")
    print(f"  Mean:   {values.mean():.6f}")
    print(f"  Std:    {values.std():.6f}\n")

    # Outlier details
    if n_outliers > 0:
        outlier_trials = trials[outlier_mask]
        outlier_values = values[outlier_mask]

        print(f"Outlier details (sorted by value):")
        print(f"{'Trial':<10} {'Value':<15} {'Distance from median':<20}")
        print(f"{'-'*50}")

        # Sort by value
        sorted_idx = np.argsort(outlier_values)
        for idx in sorted_idx:
            trial = outlier_trials[idx]
            value = outlier_values[idx]
            distance = abs(value - median)
            print(f"{trial:<10} {value:<15.6f} {distance:<20.6f}")

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Box plot with outliers highlighted
        ax1.boxplot(values, vert=True, showfliers=True)
        ax1.scatter([1] * n_outliers, outlier_values, color='red',
                   s=100, alpha=0.6, zorder=5, label='Outliers')
        ax1.set_ylabel(metric)
        ax1.set_title(f'{pipeline} - {metric}\nBox plot with outliers')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Histogram with outliers marked
        ax2.hist(values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        for val in outlier_values:
            ax2.axvline(val, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax2.axvline(median, color='green', linestyle='-', linewidth=2, label='Median')
        ax2.set_xlabel(metric)
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{pipeline} - {metric}\nDistribution (outliers marked in red)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_dir:
            output_dir = Path(output_dir)
            plot_path = output_dir / f'outliers_{pipeline}_{metric}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualization saved to {plot_path}")
        else:
            plt.show()

        plt.close()

        # Save outlier trial numbers to file
        if output_dir:
            output_dir = Path(output_dir)
            outlier_file = output_dir / f'outlier_trials_{pipeline}_{metric}.txt'
            with open(outlier_file, 'w') as f:
                f.write(f"Outlier trials for {pipeline} - {metric}\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"{'Trial':<10} {'Value':<15}\n")
                f.write(f"{'-'*30}\n")
                for idx in sorted_idx:
                    f.write(f"{outlier_trials[idx]:<10} {outlier_values[idx]:<15.6f}\n")
            print(f"✓ Outlier trials saved to {outlier_file}")

    else:
        print("No outliers detected!")

    print(f"\n{'='*80}\n")

    return trials[outlier_mask] if n_outliers > 0 else np.array([])


def compare_outlier_vs_normal_trials(results_path, pipeline, metric, outlier_trials):
    """
    Compare outlier trials vs normal trials across all metrics.

    Args:
        results_path: Path to pipeline_comparison_results.csv
        pipeline: Pipeline name
        metric: Metric that was used to identify outliers
        outlier_trials: Array of trial numbers that are outliers
    """
    if len(outlier_trials) == 0:
        print("No outlier trials to compare")
        return

    df = pd.read_csv(results_path)
    pipeline_df = df[df['pipeline'] == pipeline].copy()

    metrics = [col for col in pipeline_df.columns
              if col not in ['trial', 'pipeline']]

    print(f"\n{'='*80}")
    print(f"COMPARISON: OUTLIER vs NORMAL TRIALS FOR {pipeline}")
    print(f"{'='*80}\n")
    print(f"Outliers identified based on: {metric}")
    print(f"Number of outlier trials: {len(outlier_trials)}\n")

    # Separate outlier and normal trials
    outlier_mask = pipeline_df['trial'].isin(outlier_trials)
    outlier_data = pipeline_df[outlier_mask]
    normal_data = pipeline_df[~outlier_mask]

    print(f"{'Metric':<25} {'Normal Mean':<15} {'Outlier Mean':<15} {'Difference':<15}")
    print(f"{'-'*70}")

    for m in metrics:
        normal_mean = normal_data[m].mean()
        outlier_mean = outlier_data[m].mean()
        diff = outlier_mean - normal_mean

        print(f"{m:<25} {normal_mean:<15.6f} {outlier_mean:<15.6f} {diff:<15.6f}")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    # Example usage - inspect SWT-R tt_deviation outliers in harmonic test

    RESULTS_PATH = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/'
                       'src/decomposition_strategy_selection/results_test/pipeline_comparison_results.csv')

    OUTPUT_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/'
                     'src/decomposition_strategy_selection/results_test')

    if RESULTS_PATH.exists():
        # Inspect outliers
        outlier_trials = inspect_pipeline_outliers(
            results_path=RESULTS_PATH,
            pipeline='swt_recon',
            metric='tt_deviation',
            output_dir=OUTPUT_DIR
        )

        # Compare outlier vs normal trials
        if len(outlier_trials) > 0:
            compare_outlier_vs_normal_trials(
                results_path=RESULTS_PATH,
                pipeline='swt_recon',
                metric='tt_deviation',
                outlier_trials=outlier_trials
            )
    else:
        print(f"Error: Results file not found at {RESULTS_PATH}")
        print("Please run experiment_runner.py first to generate results.")