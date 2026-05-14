"""Check which subbands are producing TT deviation outliers."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.decomposition_strategy_selection.decomposition_analyzer import DecompositionAnalyzer


def load_outlier_trials(outlier_file):
    """Load outlier trial numbers from file."""
    trials = []
    with open(outlier_file, 'r') as f:
        for line in f:
            if line.strip() and line[0].isdigit():
                trial_num = int(line.split()[0])
                trials.append(trial_num)
    return np.array(trials)


def analyze_band_specific_tt_deviation(outlier_trials, normal_trials, subband_freqs, output_dir):
    """
    Compute TT deviation for each band separately for outlier vs normal trials.

    This will reveal if outliers are caused by specific subbands or all of them.
    """
    print(f"\n{'='*80}")
    print("BAND-SPECIFIC TT DEVIATION ANALYSIS")
    print(f"{'='*80}\n")

    # Initialize analyzer
    analyzer = DecompositionAnalyzer(
        subband_freqs=subband_freqs,
        fs=128,
        n=1024,
        level=4,
        wavelet='db4',
        output_dir=str(output_dir)
    )

    # Load ground truth
    gt_path = output_dir / 'ground_truth.json'
    ground_truth, ground_truth_data = analyzer.load_ground_truth(str(gt_path))

    bands = ['D1', 'D2', 'D3', 'D4', 'A4']

    # Store results per band
    results = {band: {'outlier': [], 'normal': []} for band in bands}

    # Sample trials
    n_test = 10

    print("Processing outlier trials...")
    for trial_idx, trial in enumerate(outlier_trials[:n_test]):
        print(f"  Trial {trial} ({trial_idx+1}/{n_test})")
        np.random.seed(trial)
        sig = analyzer.generate_composite_sine(random_phase=True)

        # Decompose using SWT-R
        subbands_raw = analyzer.decompose_signal(sig, 'swt_recon')
        subbands = {band: analyzer.min_max_normalize(subbands_raw[band])
                   for band in subbands_raw}

        # Compute TT for each band
        for band in bands:
            if band not in ground_truth:
                continue

            gt = ground_truth[band]
            tau = gt['tau']
            m = gt['m']
            epsilon = gt['epsilon']

            try:
                rp_gen = analyzer.compute_rp(subbands[band], tau, m, epsilon)
                rqa_result = analyzer.compute_rqa(rp_gen)

                tt_true = gt.get('tt', 0)
                tt_pred = rqa_result.get('trapping_time', 0)
                tt_dev = abs(tt_true - tt_pred)

                results[band]['outlier'].append(tt_dev)
                print(f"    {band}: TT_true={tt_true:.4f}, TT_pred={tt_pred:.4f}, dev={tt_dev:.4f}")
            except Exception as e:
                print(f"    {band}: ERROR - {e}")
                continue

    print("\nProcessing normal trials...")
    for trial_idx, trial in enumerate(normal_trials[:n_test]):
        print(f"  Trial {trial} ({trial_idx+1}/{n_test})")
        np.random.seed(trial)
        sig = analyzer.generate_composite_sine(random_phase=True)

        subbands_raw = analyzer.decompose_signal(sig, 'swt_recon')
        subbands = {band: analyzer.min_max_normalize(subbands_raw[band])
                   for band in subbands_raw}

        for band in bands:
            if band not in ground_truth:
                continue

            gt = ground_truth[band]
            tau = gt['tau']
            m = gt['m']
            epsilon = gt['epsilon']

            try:
                rp_gen = analyzer.compute_rp(subbands[band], tau, m, epsilon)
                rqa_result = analyzer.compute_rqa(rp_gen)

                tt_true = gt.get('tt', 0)
                tt_pred = rqa_result.get('trapping_time', 0)
                tt_dev = abs(tt_true - tt_pred)

                results[band]['normal'].append(tt_dev)
                print(f"    {band}: TT_true={tt_true:.4f}, TT_pred={tt_pred:.4f}, dev={tt_dev:.4f}")
            except Exception as e:
                print(f"    {band}: ERROR - {e}")
                continue

    # Analysis
    print(f"\n{'='*80}")
    print("RESULTS BY BAND")
    print(f"{'='*80}\n")

    print(f"{'Band':<10} {'Outlier Mean':<15} {'Normal Mean':<15} {'Ratio':<10} {'Status':<20}")
    print(f"{'-'*75}")

    summary = []
    for band in bands:
        outlier_vals = results[band]['outlier']
        normal_vals = results[band]['normal']

        if not outlier_vals or not normal_vals:
            print(f"{band:<10} {'N/A':<15} {'N/A':<15} {'N/A':<10} {'Insufficient data':<20}")
            continue

        outlier_mean = np.mean(outlier_vals)
        normal_mean = np.mean(normal_vals)
        ratio = outlier_mean / normal_mean if normal_mean > 1e-9 else np.inf

        status = "🔴 PROBLEM!" if ratio > 5 else "✓ OK"

        print(f"{band:<10} {outlier_mean:<15.6f} {normal_mean:<15.6f} {ratio:<10.2f}x {status:<20}")

        summary.append({
            'band': band,
            'frequency': subband_freqs.get(band, 'N/A'),
            'outlier_mean': outlier_mean,
            'normal_mean': normal_mean,
            'outlier_std': np.std(outlier_vals),
            'normal_std': np.std(normal_vals),
            'ratio': ratio
        })

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot - mean TT deviation by band
    ax = axes[0]
    x = np.arange(len(bands))
    width = 0.35

    outlier_means = [np.mean(results[b]['outlier']) if results[b]['outlier'] else 0 for b in bands]
    normal_means = [np.mean(results[b]['normal']) if results[b]['normal'] else 0 for b in bands]

    ax.bar(x - width/2, normal_means, width, label='Normal Trials', color='lightblue', edgecolor='black')
    ax.bar(x + width/2, outlier_means, width, label='Outlier Trials', color='salmon', edgecolor='black')

    ax.set_xlabel('Subband', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean TT Deviation', fontsize=12, fontweight='bold')
    ax.set_title('TT Deviation by Subband', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Ratio plot
    ax = axes[1]
    ratios = [outlier_means[i] / normal_means[i] if normal_means[i] > 1e-9 else 0
              for i in range(len(bands))]
    colors = ['red' if r > 5 else 'orange' if r > 2 else 'green' for r in ratios]

    ax.bar(x, ratios, color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Parity')
    ax.axhline(5.0, color='red', linestyle=':', linewidth=2, alpha=0.5, label='5x threshold')

    ax.set_xlabel('Subband', fontsize=12, fontweight='bold')
    ax.set_ylabel('Outlier/Normal Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Outlier Amplification by Subband', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = output_dir / 'band_specific_tt_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Visualization saved to {plot_path}")

    # Save detailed results
    df = pd.DataFrame(summary)
    csv_path = output_dir / 'band_specific_tt_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to {csv_path}")

    # Conclusion
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}\n")

    problem_bands = [s['band'] for s in summary if s['ratio'] > 5]
    if problem_bands:
        print(f"Problem bands (ratio > 5x): {', '.join(problem_bands)}")
        print(f"\nThe outliers are caused by specific subbands, not all of them!")
        print(f"This suggests a band-specific reconstruction issue.")
    else:
        print("No single band shows extreme outliers.")
        print("The problem appears to be aggregate across all bands.")

    return summary


if __name__ == '__main__':
    # Paths
    RESULTS_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/'
                      'src/decomposition_strategy_selection/results_test')

    OUTLIER_FILE = RESULTS_DIR / 'outlier_trials_swt_recon_tt_deviation.txt'
    RESULTS_PATH = RESULTS_DIR / 'pipeline_comparison_results.csv'

    SUBBAND_FREQS_HARM = {
        'A4': 2, 'D4': 6, 'D3': 12, 'D2': 24, 'D1': 48
    }

    if not OUTLIER_FILE.exists():
        print(f"Error: Run inspect_outliers.py first to generate outlier list")
        exit(1)

    # Load trials
    outlier_trials = load_outlier_trials(OUTLIER_FILE)
    df = pd.read_csv(RESULTS_PATH)
    all_trials = df['trial'].unique()
    normal_trials = np.array([t for t in all_trials if t not in outlier_trials])

    print(f"Loaded {len(outlier_trials)} outlier trials and {len(normal_trials)} normal trials")

    # Run analysis
    summary = analyze_band_specific_tt_deviation(
        outlier_trials,
        normal_trials,
        SUBBAND_FREQS_HARM,
        RESULTS_DIR
    )