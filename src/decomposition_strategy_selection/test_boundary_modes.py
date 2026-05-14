"""Test different SWT boundary modes to fix TT deviation outliers."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.decomposition_strategy_selection.decomposition_analyzer import DecompositionAnalyzer
from src.pipelines.pipeline_wavelets_modes import Pipeline_Modes


def load_outlier_trials(outlier_file):
    """Load outlier trial numbers from file."""
    trials = []
    with open(outlier_file, 'r') as f:
        for line in f:
            if line.strip() and line[0].isdigit():
                trial_num = int(line.split()[0])
                trials.append(trial_num)
    return np.array(trials)


def test_boundary_mode(analyzer, signal, ground_truth, ground_truth_data, mode='symmetric'):
    """
    Test SWT-R pipeline with a specific boundary mode.

    Args:
        analyzer: DecompositionAnalyzer instance
        signal: Input signal
        ground_truth: Ground truth RQA metrics
        ground_truth_data: Ground truth RP data
        mode: Boundary mode to test

    Returns:
        Mean TT deviation across all bands
    """
    # Create pipeline with specified mode
    pipeline_obj = Pipeline_Modes(
        pipeline_name='all',
        level=analyzer.LEVEL,
        wavelet=analyzer.WAVELET,
        mode=mode
    )

    # Decompose signal
    subbands_raw = pipeline_obj.pipeline_swt_recon(signal)

    # Normalize subbands
    subbands = {
        band: analyzer.min_max_normalize(subbands_raw[band])
        for band in subbands_raw
    }

    # Compute RQA for each band
    tt_deviations = []

    for band in ['D1', 'D2', 'D3', 'D4', 'A4']:
        if band not in ground_truth:
            continue

        gt = ground_truth[band]
        tau = gt['tau']
        m = gt['m']
        epsilon = gt['epsilon']

        # Compute RP
        try:
            rp_gen = analyzer.compute_rp(subbands[band], tau, m, epsilon)
            rqa_result = analyzer.compute_rqa(rp_gen)

            # Compute TT deviation
            tt_true = gt.get('tt', 0)
            tt_pred = rqa_result.get('trapping_time', 0)
            tt_dev = abs(tt_true - tt_pred)

            tt_deviations.append(tt_dev)
        except:
            continue

    return np.mean(tt_deviations) if tt_deviations else np.nan


def compare_boundary_modes(outlier_trials, normal_trials, subband_freqs, output_dir):
    """
    Compare different boundary modes on outlier vs normal trials.

    Args:
        outlier_trials: Array of outlier trial numbers
        normal_trials: Array of normal trial numbers
        subband_freqs: Dict of subband frequencies
        output_dir: Directory to save results
    """
    print(f"\n{'='*80}")
    print("TESTING BOUNDARY MODES FOR SWT-R")
    print(f"{'='*80}\n")

    # Boundary modes to test
    modes = ['symmetric', 'zero', 'periodic', 'smooth']

    # Initialize analyzer (using same output_dir as experiment for ground truth)
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

    # Test each mode on sample of outlier and normal trials
    results = []
    n_test_trials = 10  # Test on 10 trials each

    for mode in modes:
        print(f"\nTesting mode: {mode}")

        outlier_devs = []
        normal_devs = []

        # Test on outlier trials
        print(f"  Testing on {n_test_trials} outlier trials...")
        for trial in outlier_trials[:n_test_trials]:
            np.random.seed(trial)
            sig = analyzer.generate_composite_sine(random_phase=True)

            tt_dev = test_boundary_mode(analyzer, sig, ground_truth, ground_truth_data, mode)
            if not np.isnan(tt_dev):
                outlier_devs.append(tt_dev)

        # Test on normal trials
        print(f"  Testing on {n_test_trials} normal trials...")
        for trial in normal_trials[:n_test_trials]:
            np.random.seed(trial)
            sig = analyzer.generate_composite_sine(random_phase=True)

            tt_dev = test_boundary_mode(analyzer, sig, ground_truth, ground_truth_data, mode)
            if not np.isnan(tt_dev):
                normal_devs.append(tt_dev)

        # Store results
        outlier_mean = np.mean(outlier_devs) if outlier_devs else np.nan
        normal_mean = np.mean(normal_devs) if normal_devs else np.nan

        results.append({
            'mode': mode,
            'outlier_mean_tt_dev': outlier_mean,
            'normal_mean_tt_dev': normal_mean,
            'outlier_std': np.std(outlier_devs) if outlier_devs else np.nan,
            'normal_std': np.std(normal_devs) if normal_devs else np.nan,
            'improvement': ((outlier_mean / normal_mean) if normal_mean > 1e-9 else np.nan)
        })

        print(f"    Outlier trials: TT dev = {outlier_mean:.6f} ± {np.std(outlier_devs) if outlier_devs else 0:.6f}")
        print(f"    Normal trials:  TT dev = {normal_mean:.6f} ± {np.std(normal_devs) if normal_devs else 0:.6f}")
        print(f"    Ratio: {outlier_mean / normal_mean if normal_mean > 1e-9 else np.inf:.2f}x")

    # Save results
    df = pd.DataFrame(results)
    results_path = output_dir / 'boundary_mode_comparison.csv'
    df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot - Mean TT deviation
    ax = axes[0]
    x = np.arange(len(modes))
    width = 0.35

    outlier_means = df['outlier_mean_tt_dev'].values
    normal_means = df['normal_mean_tt_dev'].values

    ax.bar(x - width/2, normal_means, width, label='Normal Trials',
           color='lightblue', edgecolor='black')
    ax.bar(x + width/2, outlier_means, width, label='Outlier Trials',
           color='salmon', edgecolor='black')

    ax.set_xlabel('Boundary Mode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean TT Deviation', fontsize=12, fontweight='bold')
    ax.set_title('TT Deviation by Boundary Mode', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Ratio plot - Outlier/Normal ratio
    ax = axes[1]
    ratios = outlier_means / normal_means
    colors = ['red' if r > 10 else 'orange' if r > 5 else 'green' for r in ratios]

    ax.bar(x, ratios, color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=2,
               label='Parity (1.0 = no difference)')
    ax.axhline(23.0, color='red', linestyle=':', linewidth=2, alpha=0.5,
               label='Original SWT-R ratio (23x)')

    ax.set_xlabel('Boundary Mode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Outlier/Normal Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Outlier Amplification Factor', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = output_dir / 'boundary_mode_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualization saved to {plot_path}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Mode':<15} {'Outlier TT Dev':<20} {'Normal TT Dev':<20} {'Ratio':<10}")
    print(f"{'-'*70}")
    for _, row in df.iterrows():
        print(f"{row['mode']:<15} {row['outlier_mean_tt_dev']:<20.6f} "
              f"{row['normal_mean_tt_dev']:<20.6f} {row['improvement']:<10.2f}x")

    # Find best mode
    best_mode = df.loc[df['improvement'].idxmin(), 'mode']
    best_ratio = df.loc[df['improvement'].idxmin(), 'improvement']

    print(f"\n{'='*80}")
    print(f"BEST MODE: {best_mode.upper()}")
    print(f"Outlier/Normal ratio: {best_ratio:.2f}x (vs 23.0x for default symmetric)")
    print(f"{'='*80}\n")

    return df


if __name__ == '__main__':
    # Paths
    RESULTS_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/'
                      'src/decomposition_strategy_selection/results_test')

    OUTLIER_FILE = RESULTS_DIR / 'outlier_trials_swt_recon_tt_deviation.txt'
    RESULTS_PATH = RESULTS_DIR / 'pipeline_comparison_results.csv'

    # Configuration
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

    # Run comparison
    results_df = compare_boundary_modes(
        outlier_trials,
        normal_trials,
        SUBBAND_FREQS_HARM,
        RESULTS_DIR
    )