"""
Visualization and detailed reporting for robustness analysis results.
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import os

RESULTS_PATH = '/Users/filipsulich/Developer/uni/Thesis/Thesis/outputs/experiment_1_robustness/robustness_results.csv'
OUTPUT_DIR = '/Users/filipsulich/Developer/uni/Thesis/Thesis/outputs/experiment_1_robustness'

RQA_METRICS = ['RR', 'DET', 'L_mean', 'ENTR', 'LAM', 'TT']
PIPELINES = ['swt_recon', 'swt_direct', 'dwt_recon']
PIPELINE_LABELS = {
    'swt_recon': 'SWT Recon',
    'swt_direct': 'SWT Direct',
    'dwt_recon': 'DWT Recon'
}


def load_results():
    """Load robustness results from CSV."""
    results = []
    with open(RESULTS_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row.keys():
                if key not in ['perturbation', 'pipeline']:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
            results.append(row)
    return results


def create_summary_table(results):
    """Create formatted summary table as shown in requirements."""
    print("\n" + "="*120)
    print("DETAILED ROBUSTNESS RESULTS")
    print("="*120)

    # Header
    header = f"{'Perturbation':<15} {'Pipeline':<12}"
    for metric in RQA_METRICS:
        header += f" {metric:>12}"
    header += f" {'Mean':>12}"
    print(header)
    print("-"*120)

    # Group by perturbation
    perturbations = ['noise_30dB', 'noise_20dB', 'noise_10dB', 'shift_-50', 'shift_+50']

    for pert in perturbations:
        pert_results = [r for r in results if r['perturbation'] == pert]

        for row in pert_results:
            line = f"{pert:<15} {PIPELINE_LABELS[row['pipeline']]:<12}"

            for metric in RQA_METRICS:
                median = row[f'{metric}_median']
                q25 = row[f'{metric}_q25']
                q75 = row[f'{metric}_q75']
                # Format as median (IQR)
                line += f" {median:>5.3f}({q75-q25:.3f})"

            line += f" {row['mean_delta']:>12.3f}"
            print(line)

        if pert != perturbations[-1]:
            print("-"*120)


def plot_robustness_heatmap(results):
    """Create heatmap of ΔRQA values."""
    perturbations = ['noise_30dB', 'noise_20dB', 'noise_10dB', 'shift_-50', 'shift_+50']
    pert_labels = ['Noise 30dB', 'Noise 20dB', 'Noise 10dB', 'Shift -50', 'Shift +50']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for pipeline_idx, pipeline in enumerate(PIPELINES):
        ax = axes[pipeline_idx]

        # Create matrix: rows = perturbations, cols = metrics
        matrix = np.zeros((len(perturbations), len(RQA_METRICS)))

        for i, pert in enumerate(perturbations):
            pert_results = [r for r in results if r['perturbation'] == pert and r['pipeline'] == pipeline]
            if pert_results:
                row_data = pert_results[0]
                for j, metric in enumerate(RQA_METRICS):
                    matrix[i, j] = row_data[f'{metric}_median']

        # Plot heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)

        # Set ticks
        ax.set_xticks(np.arange(len(RQA_METRICS)))
        ax.set_yticks(np.arange(len(perturbations)))
        ax.set_xticklabels(RQA_METRICS)
        ax.set_yticklabels(pert_labels)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add values as text
        for i in range(len(perturbations)):
            for j in range(len(RQA_METRICS)):
                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9)

        ax.set_title(PIPELINE_LABELS[pipeline], fontsize=14, fontweight='bold')
        ax.set_xlabel('RQA Metric', fontsize=11)
        if pipeline_idx == 0:
            ax.set_ylabel('Perturbation', fontsize=11)

    # Colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Median ΔRQA', rotation=270, labelpad=20)

    plt.suptitle('Robustness Analysis: Median ΔRQA Under Perturbations', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.93, 0.96])

    output_path = os.path.join(OUTPUT_DIR, 'robustness_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved heatmap: {output_path}")
    plt.close()


def plot_pipeline_comparison(results):
    """Bar chart comparing pipelines across perturbations."""
    perturbations = ['noise_30dB', 'noise_20dB', 'noise_10dB', 'shift_-50', 'shift_+50']
    pert_labels = ['Noise\n30dB', 'Noise\n20dB', 'Noise\n10dB', 'Shift\n-50', 'Shift\n+50']

    x = np.arange(len(perturbations))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, pipeline in enumerate(PIPELINES):
        means = []
        errors = []

        for pert in perturbations:
            pert_results = [r for r in results if r['perturbation'] == pert and r['pipeline'] == pipeline]
            if pert_results:
                means.append(pert_results[0]['mean_delta'])
                # Use max metric IQR as error bar
                max_iqr = max([pert_results[0][f'{m}_q75'] - pert_results[0][f'{m}_q25']
                              for m in RQA_METRICS])
                errors.append(max_iqr)
            else:
                means.append(0)
                errors.append(0)

        offset = (i - 1) * width
        ax.bar(x + offset, means, width, label=PIPELINE_LABELS[pipeline],
               color=colors[i], alpha=0.8, yerr=errors, capsize=5)

    ax.set_xlabel('Perturbation Type', fontsize=13)
    ax.set_ylabel('Mean ΔRQA (across all metrics)', fontsize=13)
    ax.set_title('Pipeline Robustness Comparison\n(Lower is better)', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pert_labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'pipeline_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path}")
    plt.close()


def plot_metric_sensitivity(results):
    """Plot which metrics are most sensitive to perturbations."""
    perturbations = ['noise_30dB', 'noise_20dB', 'noise_10dB', 'shift_-50', 'shift_+50']

    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()

    for pert_idx, pert in enumerate(perturbations):
        ax = axes[pert_idx]

        # Collect data for this perturbation
        metric_data = {metric: [] for metric in RQA_METRICS}

        for pipeline in PIPELINES:
            pert_results = [r for r in results if r['perturbation'] == pert and r['pipeline'] == pipeline]
            if pert_results:
                row = pert_results[0]
                for metric in RQA_METRICS:
                    metric_data[metric].append(row[f'{metric}_median'])

        # Create grouped bar chart
        x = np.arange(len(RQA_METRICS))
        width = 0.25

        for i, pipeline in enumerate(PIPELINES):
            values = [metric_data[m][i] if len(metric_data[m]) > i else 0 for m in RQA_METRICS]
            offset = (i - 1) * width
            ax.bar(x + offset, values, width, label=PIPELINE_LABELS[pipeline], alpha=0.8)

        ax.set_xlabel('RQA Metric', fontsize=11)
        ax.set_ylabel('Median ΔRQA', fontsize=11)
        ax.set_title(pert.replace('_', ' ').replace('dB', ' dB'), fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(RQA_METRICS)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Metric-Level Sensitivity Analysis', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'metric_sensitivity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved metric sensitivity plot: {output_path}")
    plt.close()


def compute_overall_ranking(results):
    """Compute and display overall robustness ranking."""
    print("\n" + "="*80)
    print("OVERALL ROBUSTNESS RANKING")
    print("="*80)

    pipeline_scores = {}

    for pipeline in PIPELINES:
        pipeline_results = [r for r in results if r['pipeline'] == pipeline]

        # Collect all mean deltas
        all_means = [r['mean_delta'] for r in pipeline_results]

        # Compute statistics
        overall_mean = np.mean(all_means)
        overall_std = np.std(all_means)
        overall_min = np.min(all_means)
        overall_max = np.max(all_means)

        pipeline_scores[pipeline] = {
            'mean': overall_mean,
            'std': overall_std,
            'min': overall_min,
            'max': overall_max
        }

    # Rank by mean (lower is better)
    ranked = sorted(pipeline_scores.items(), key=lambda x: x[1]['mean'])

    print(f"\n{'Rank':<6} {'Pipeline':<15} {'Mean ΔRQA':<15} {'Std':<10} {'Range':<20}")
    print("-"*80)

    for rank, (pipeline, scores) in enumerate(ranked, 1):
        star = '★' if rank == 1 else ''
        print(f"{rank:<6} {PIPELINE_LABELS[pipeline]:<15} "
              f"{scores['mean']:.6f}      {scores['std']:.6f}   "
              f"[{scores['min']:.6f}, {scores['max']:.6f}] {star}")

    print("\n★ = Most robust pipeline")

    # Interpret results
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    winner = ranked[0][0]

    if winner == 'swt_recon':
        print("✓ SWT Reconstruction shows the best overall robustness")
        print("  - Expected due to redundancy providing noise averaging (Nason & Silverman, 1995)")
    elif winner == 'swt_direct':
        print("✓ SWT Direct Coefficients shows the best overall robustness")
        print("  - Combines shift-invariance with efficient coefficient-space processing")
    elif winner == 'dwt_recon':
        print("✓ DWT Reconstruction shows the best overall robustness")
        print("  - Orthogonal decomposition provides compact representation")

    # Check shift invariance
    print("\nShift Invariance Analysis:")
    for pipeline in PIPELINES:
        shift_results = [r for r in results if 'shift' in r['perturbation'] and r['pipeline'] == pipeline]
        if shift_results:
            shift_deltas = [r['mean_delta'] for r in shift_results]
            avg_shift_delta = np.mean(shift_deltas)

            if avg_shift_delta < 0.05:
                verdict = "✓ Excellent shift invariance (ΔRQA < 0.05)"
            elif avg_shift_delta < 0.10:
                verdict = "~ Good shift invariance (ΔRQA < 0.10)"
            else:
                verdict = "✗ Poor shift invariance (ΔRQA > 0.10)"

            print(f"  {PIPELINE_LABELS[pipeline]:<15}: {avg_shift_delta:.6f}  {verdict}")


def main():
    """Generate all visualizations and reports."""
    print("="*80)
    print("ROBUSTNESS ANALYSIS VISUALIZATION")
    print("="*80)

    # Load results
    results = load_results()
    print(f"\nLoaded {len(results)} result rows")

    # Create summary table
    create_summary_table(results)

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    plot_robustness_heatmap(results)
    plot_pipeline_comparison(results)
    plot_metric_sensitivity(results)

    # Compute overall ranking
    compute_overall_ranking(results)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()