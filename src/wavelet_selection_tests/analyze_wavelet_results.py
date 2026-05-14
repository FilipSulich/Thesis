"""
Statistical analysis and visualization of wavelet selection results.

Reads wavelet_selection_results.json and performs:
1. Friedman test for significance across wavelets
2. Box plot visualization
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare


def load_results(json_path: Path) -> Dict:
    """Load wavelet selection results from JSON."""
    with open(json_path) as f:
        return json.load(f)


def extract_auc_matrix(results: Dict) -> Tuple[List[str], np.ndarray]:
    """
    Extract AUC values organized as (wavelets × subbands) matrix.

    Returns:
        wavelets: List of wavelet names
        auc_matrix: (n_wavelets, n_subbands) array of mean AUC values
    """
    wavelets = sorted(results.keys())

    # Get subband names from first wavelet
    first_wavelet = results[wavelets[0]]
    subbands = sorted([k for k in first_wavelet['per_subband'].keys()
                      if first_wavelet['per_subband'][k] is not None])

    n_wavelets = len(wavelets)
    n_subbands = len(subbands)

    auc_matrix = np.zeros((n_wavelets, n_subbands))

    for i, wavelet in enumerate(wavelets):
        for j, subband in enumerate(subbands):
            subband_data = results[wavelet]['per_subband'].get(subband)
            if subband_data is not None:
                auc_matrix[i, j] = subband_data['mean']
            else:
                auc_matrix[i, j] = np.nan

    return wavelets, subbands, auc_matrix


def friedman_test(auc_matrix: np.ndarray, wavelets: List[str]) -> Dict:
    """
    Apply Friedman test to check if wavelets differ significantly.

    Friedman test is a non-parametric alternative to repeated-measures ANOVA.
    Null hypothesis: All wavelets have the same median AUC across subbands.

    Args:
        auc_matrix: (n_wavelets, n_subbands) array
        wavelets: List of wavelet names

    Returns:
        Dictionary with test results
    """
    # Friedman test requires data as groups (one per wavelet)
    # Each group contains AUC values across subbands
    groups = [auc_matrix[i, :] for i in range(len(wavelets))]

    # Remove NaN values if any
    groups_clean = []
    for group in groups:
        if not np.any(np.isnan(group)):
            groups_clean.append(group)

    if len(groups_clean) < 3:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'message': 'Insufficient data for Friedman test (need ≥3 groups)'
        }

    statistic, p_value = friedmanchisquare(*groups_clean)

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'alpha': 0.05,
        'n_wavelets': len(groups_clean),
        'n_subbands': int(auc_matrix.shape[1])
    }


def plot_boxplot_comparison(
    results: Dict,
    wavelets: List[str],
    auc_matrix: np.ndarray,
    save_path: Path
) -> None:
    """
    Create side-by-side box plot comparing AUC distributions across wavelets.

    Args:
        results: Full results dictionary
        wavelets: List of wavelet names
        auc_matrix: (n_wavelets, n_subbands) array
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Prepare data for box plot
    positions = np.arange(len(wavelets))
    box_data = [auc_matrix[i, ~np.isnan(auc_matrix[i, :])] for i in range(len(wavelets))]

    # Create box plot
    bplot = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.7,
        patch_artist=True,  # Allows filling the boxes with color
    )

    # Color boxes by wavelet family
    for patch, wavelet in zip(bplot['boxes'], wavelets):
        if wavelet.startswith('db'):
            color = '#1f77b4'  # Blue for Daubechies
        elif wavelet.startswith('sym'):
            color = '#d62728'  # Red for Symlets
        elif wavelet.startswith('coif'):
            color = '#2ca02c'  # Green for Coiflets
        else:
            color = '#7f7f7f'  # Gray for others

        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Styling median lines to ensure visibility
    for median in bplot['medians']:
        median.set(color='black', linewidth=1.5)

    # Formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(wavelets, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('AUC (across D2-D8 subbands)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Wavelet', fontsize=12, fontweight='bold')
    ax.set_title('Wavelet AUC Distribution Comparison',
                 fontsize=13, fontweight='bold')

    # Add horizontal line at chance level
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5,
               label='Chance level')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.6, label='Daubechies'),
        Patch(facecolor='#d62728', alpha=0.6, label='Symlets'),
        Patch(facecolor='#2ca02c', alpha=0.6, label='Coiflets'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Box plot saved to {save_path}")


def analyze_wavelet_results(results_json: Path, output_dir: Path) -> None:
    """
    Main analysis function.

    Args:
        results_json: Path to wavelet_selection_results.json
        output_dir: Directory to save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS OF WAVELET SELECTION RESULTS")
    print(f"{'='*70}\n")

    # Load results
    print(f"Loading results from {results_json}...")
    results = load_results(results_json)

    # Extract AUC matrix
    wavelets, subbands, auc_matrix = extract_auc_matrix(results)
    print(f"Loaded {len(wavelets)} wavelets × {len(subbands)} subbands")
    print(f"Subbands: {', '.join(subbands)}")

    # Friedman test
    print(f"\n{'─'*70}")
    print("FRIEDMAN TEST FOR WAVELET SIGNIFICANCE")
    print(f"{'─'*70}")
    print("Null hypothesis: All wavelets have equal median AUC across subbands")
    print("Alternative: At least one wavelet differs significantly\n")

    friedman_results = friedman_test(auc_matrix, wavelets)

    print(f"Test statistic (χ²): {friedman_results['statistic']:.4f}")
    print(f"P-value: {friedman_results['p_value']:.4e}")
    print(f"Significance level (α): {friedman_results.get('alpha', 0.05)}")
    print(f"\nResult: {'SIGNIFICANT' if friedman_results['significant'] else 'NOT SIGNIFICANT'}")

    if friedman_results['significant']:
        print("→ There are statistically significant differences between wavelets")
    else:
        print("→ No significant differences detected between wavelets")

    # Save Friedman test results
    friedman_path = output_dir / 'friedman_test_results.json'
    with open(friedman_path, 'w') as f:
        json.dump(friedman_results, f, indent=2)
    print(f"\n✓ Friedman test results saved to {friedman_path}")

    # Create box plot
    print(f"\n{'─'*70}")
    print("GENERATING BOX PLOT")
    print(f"{'─'*70}\n")

    boxplot_path = output_dir / 'wavelet_auc_boxplot_comparison.png'
    plot_boxplot_comparison(results, wavelets, auc_matrix, boxplot_path)

    # Summary statistics
    print(f"\n{'─'*70}")
    print("SUMMARY STATISTICS")
    print(f"{'─'*70}\n")

    summary = []
    for i, wavelet in enumerate(wavelets):
        auc_vals = auc_matrix[i, ~np.isnan(auc_matrix[i, :])]
        summary.append({
            'wavelet': wavelet,
            'median_auc': results[wavelet]['median_auc'],
            'mean_auc': results[wavelet]['mean_auc'],
            'std_auc': float(np.std(auc_vals)),
            'min_auc': results[wavelet]['min_auc'],
            'max_auc': results[wavelet]['max_auc'],
        })

    # Sort by median AUC
    summary.sort(key=lambda x: x['median_auc'], reverse=True)

    print(f"{'Wavelet':<8} {'Median':<8} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print(f"{'-'*60}")
    for s in summary:
        print(f"{s['wavelet']:<8} {s['median_auc']:<8.4f} {s['mean_auc']:<8.4f} "
              f"{s['std_auc']:<8.4f} {s['min_auc']:<8.4f} {s['max_auc']:<8.4f}")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    # Example usage
    RESULTS_JSON = Path('/results_final/results/wavelet_selection_utrp/wavelet_selection_results.json')
    OUTPUT_DIR = Path('/results_final/results/results/wavelet_selection_utrp')

    if RESULTS_JSON.exists():
        analyze_wavelet_results(RESULTS_JSON, OUTPUT_DIR)
    else:
        print(f"Error: Results file not found at {RESULTS_JSON}")
        print("Please run ecg_wavelet_evaluator.py first to generate results.")