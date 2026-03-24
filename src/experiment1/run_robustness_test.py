"""
Sensitivity Analysis: Feature Stability vs Perturbation Intensity.
Tests synthetic signals (Sine, Lorenz, Noise) to prove ground truth decay.

Usage:
    python3 -m src.experiment1.run_robustness_test
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment1.parallelization import ParallelExperiment, RobustnessTest


def plot_decay_curves(battery_results, perturbation_type='awgn'):
    """
    Plot MAPE decay curves for all pipelines (Stability vs Perturbation Intensity).

    Args:
        battery_results: Results from run_full_sensitivity_battery
        perturbation_type: 'awgn', 'baseline_wander', or 'phase_jitter'
    """
    signal_types = ['Sine', 'Lorenz', 'Noise']
    pipelines = ['single_scale', 'dwt_recon', 'dwt_direct', 'swt_recon', 'swt_direct']
    pipeline_labels = {
        'single_scale': 'Single-Scale',
        'dwt_recon': 'DWT Recon',
        'dwt_direct': 'DWT Direct',
        'swt_recon': 'SWT Recon',
        'swt_direct': 'SWT Direct'
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Feature Stability: {perturbation_type.replace("_", " ").title()}',
                 fontsize=16, fontweight='bold')

    for idx, signal_type in enumerate(signal_types):
        ax = axes[idx]

        for pipeline in pipelines:
            key = f"{signal_type}_{pipeline}"
            if key in battery_results and perturbation_type in battery_results[key]:
                df = battery_results[key][perturbation_type]

                if perturbation_type == 'awgn':
                    x_col = 'snr_db'
                    x_label = 'SNR (dB)'
                elif perturbation_type == 'baseline_wander':
                    x_col = 'amplitude'
                    x_label = 'Drift Amplitude'
                else:
                    x_col = 'jitter_pct'
                    x_label = 'Jitter (%)'

                ax.plot(df[x_col], df['mape'], marker='o', label=pipeline_labels[pipeline], linewidth=2)

        ax.set_title(f'{signal_type} Signal', fontweight='bold', fontsize=14)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('MAPE (%)', fontsize=12)
        ax.legend(fontsize='small')
        ax.grid(alpha=0.3)

        # Invert x-axis for SNR (higher SNR = cleaner)
        if perturbation_type == 'awgn':
            ax.invert_xaxis()

    plt.tight_layout()
    plt.show()


def summarize_stability_metrics(battery_results):
    """
    Compute average MAPE for each pipeline across all perturbations.
    The pipeline with the lowest average MAPE is the most stable.
    """
    summary = []

    for key, tests in battery_results.items():
        signal_name, pipeline = key.rsplit('_', 1)

        for perturbation_type, df in tests.items():
            avg_mape = df['mape'].mean()
            max_mape = df['mape'].max()

            summary.append({
                'Signal': signal_name,
                'Pipeline': pipeline,
                'Perturbation': perturbation_type,
                'Avg_MAPE': avg_mape,
                'Max_MAPE': max_mape
            })

    return pd.DataFrame(summary)


def main():
    """Run sensitivity analysis."""

    # Initialize experiment
    print("Initializing experiment...")
    exp = ParallelExperiment(n_jobs=-1)

    # Generate dataset
    print("Generating synthetic dataset...")
    X, y = exp.synthetic_data_generator.generate_dataset()

    # Initialize robustness tester
    robustness = RobustnessTest(exp)

    # Run sensitivity battery
    print("\nRunning sensitivity battery...")
    print("Testing 3 representative signals (one per class)")
    print("="*60)

    battery_results = robustness.run_full_sensitivity_battery(X, y)

    # Summarize stability
    print("\n" + "="*60)
    print("STABILITY SUMMARY (Average MAPE)")
    print("="*60)
    summary = summarize_stability_metrics(battery_results)

    # Pivot table for each perturbation
    for perturbation in ['awgn', 'baseline_wander', 'phase_jitter']:
        print(f"\n{perturbation.upper()}:")
        subset = summary[summary['Perturbation'] == perturbation]
        pivot = subset.pivot_table(
            index='Pipeline',
            columns='Signal',
            values='Avg_MAPE',
            aggfunc='mean'
        )
        print(pivot.round(2))

    # Plot decay curves
    print("\n" + "="*60)
    print("Generating decay curve plots...")
    print("="*60)

    plot_decay_curves(battery_results, 'awgn')
    plot_decay_curves(battery_results, 'baseline_wander')
    plot_decay_curves(battery_results, 'phase_jitter')

    # Identify winner
    print("\n" + "="*60)
    print("WINNER ANALYSIS")
    print("="*60)

    overall_avg = summary.groupby('Pipeline')['Avg_MAPE'].mean().sort_values()
    print("\nOverall Ranking (Lower MAPE = More Stable):")
    for pipeline, mape in overall_avg.items():
        print(f"  {pipeline:<15}: {mape:.2f}%")

    winner = overall_avg.idxmin()
    print(f"\n Winner: {winner} (Most Stable Pipeline)")

    print("\nSensitivity analysis complete!")


if __name__ == '__main__':
    main()