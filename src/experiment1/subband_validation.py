import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.pipeline_wavelets import Pipeline
from recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

class SubbandValidator:
    def __init__(self):
        self.FS = 128  # Sampling frequency (Hz)
        self.N = 512  # Signal length (samples)
        self.LEVEL = 4  # Wavelet decomposition level
        self.WAVELET = 'db4'

        self.SUBBAND_FREQS = {
            'A4': 2,   # 0-4 Hz
            'D4': 6,   # 4-8 Hz
            'D3': 12,  # 8-16 Hz
            'D2': 24,  # 16-32 Hz
            'D1': 48   # 32-64 Hz
        }

        self.BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']
        self.PIPELINES = ['swt_recon', 'swt_direct', 'dwt_recon']
        self.OUTPUT_DIR = '/Users/filipsulich/Developer/uni/Thesis/Thesis/outputs/experiment1_synthetic'
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def generate_composite_sine(self, random_phase=False):
        """
        Generate composite sine signal with one component per subband.
        All components have equal amplitude (unit amplitude).

        Args:
            random_phase: If True, add random phase to each component

        Returns:
            signal: Composite sine signal (N samples)
        """
        t = np.arange(self.N) / self.FS
        signal = np.zeros(self.N)

        # All components have amplitude = 1 (unit amplitude)
        for band, freq in self.SUBBAND_FREQS.items():
            if random_phase:
                phase = np.random.uniform(0, 2*np.pi)
            else:
                phase = 0
            signal += np.sin(2 * np.pi * freq * t + phase)

        # signal += np.sin(2 * np.pi  * t)

        return signal

    def get_analytic_embedding_params(self, freq):
        """Get embedding parameters using RecurrencePlotGenerator utilities."""
        tau = RecurrencePlotGenerator.estimate_time_delay_analytic(freq, self.FS)
        m = RecurrencePlotGenerator.estimate_embedding_dim_analytic()
        return tau, m

    def decompose_signal(self, signal, pipeline_name):
        pipeline_obj = Pipeline(pipeline_name='all', level=self.LEVEL, wavelet=self.WAVELET)
        func = getattr(pipeline_obj, f"pipeline_{pipeline_name}")
        return func(signal)

    def compute_rp_metrics(self, signal, tau, m, epsilon):
        try:
            os.environ['PYOPENCL_CTX'] = '0'
            rp_gen = RecurrencePlotGenerator(
                signal=signal,
                time_delay=tau,
                embedding_dim=m,
                path=self.OUTPUT_DIR,
                epsilon=epsilon,
                unthresholded=False
            )
            rqa_result, rp_matrix = rp_gen.compute_rqa_binary()

            l_mean_diag = rqa_result.average_diagonal_line
            if np.isnan(l_mean_diag): l_mean_diag = 0.0

            longest_diagonal = rqa_result.longest_diagonal_line
            if np.isnan(longest_diagonal): longest_diagonal = 0.0

            tt = rqa_result.trapping_time
            if np.isnan(tt): tt = 0.0

            longest_vertical_line  = rqa_result.longest_vertical_line
            if np.isnan(longest_vertical_line): longest_vertical_line = 0

            det = rqa_result.determinism
            if np.isnan(det): det = 0.0

            lam = rqa_result.laminarity
            if np.isnan(lam): lam = 0.0

            return {
                'DET': det,
                'ENTR': rqa_result.entropy_diagonal_lines,
                'LAM': lam,
                'L_mean_diagonal': l_mean_diag,
                'L_longest_diagonal': longest_diagonal,
                'L_longest_vertical': longest_vertical_line,
                'TT': tt,
                'RR': rqa_result.recurrence_rate,
                'rp_matrix': rp_matrix
            }
        except Exception as e:
            print(f"Warning: RQA computation failed: {e}")
            return {
                'DET': 0.0,
                'ENTR': 0.0,
                'LAM': 0.0,
                'L_mean_diagonal': 0.0,
                'L_longest_diagonal': 0.0,
                'L_max_vertical': 0.0,
                'TT': 0.0,
                'RR': 0.0,
                'rp_matrix': None
            }

    def find_epsilon_for_target_rr(self, signal, tau, m, target_rr=0.1):
        """
        Find epsilon that achieves target recurrence rate using bisection.

        Args:
            signal: Input signal
            tau: Embedding delay
            m: Embedding dimension
            target_rr: Target recurrence rate

        Returns:
            epsilon: Threshold value
        """
        from scipy.spatial.distance import pdist, squareform

        # Embed signal
        n = len(signal) - (m - 1) * tau
        if n < 10:
            return 0.1

        embedded = np.array([signal[i:i + m * tau:tau] for i in range(n)])
        distances = squareform(pdist(embedded, metric='euclidean'))

        # Find epsilon range
        all_dists = distances[np.triu_indices_from(distances, k=1)]
        eps_min, eps_max = np.percentile(all_dists, 1), np.percentile(all_dists, 20)

        # Bisection search
        for _ in range(20):
            eps_mid = (eps_min + eps_max) / 2
            rp_test = (distances <= eps_mid).astype(int)
            np.fill_diagonal(rp_test, 0)
            rr = np.sum(rp_test) / (rp_test.shape[0] * (rp_test.shape[0] - 1))

            if abs(rr - target_rr) < 0.01:
                break

            if rr < target_rr:
                eps_min = eps_mid
            else:
                eps_max = eps_mid

        return eps_mid

    def compute_dominant_period(self, rp_result):
        """
        Compute dominant period from RP diagonal structure using autocorrelation.

        Args:
            rp_result: RPResult object from pyrqa

        Returns:
            dominant_lag: Dominant period in samples
        """
        if rp_result is None:
            return 0

        try:
            # Extract recurrence matrix from RPResult
            rp_matrix = rp_result.recurrence_matrix_reverse_normalized

            if rp_matrix is None or rp_matrix.size == 0:
                return 0

            # Sum along diagonals to get recurrence profile
            n = rp_matrix.shape[0]
            recurrence_profile = np.array([np.sum(np.diag(rp_matrix, k=i)) for i in range(n)])

            # Autocorrelation of recurrence profile
            autocorr = np.correlate(recurrence_profile, recurrence_profile, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags only

            # Find dominant lag (ignore lag=0)
            if len(autocorr) > 1:
                dominant_lag = np.argmax(autocorr[1:]) + 1
            else:
                dominant_lag = 0

            return dominant_lag
        except:
            return 0

    def validate_pipeline(self, signal, pipeline_name, reference_epsilons=None):

        subbands = self.decompose_signal(signal, pipeline_name)

        results = []
        epsilons_used = {}


        for band in self.BAND_NAMES:
            freq = self.SUBBAND_FREQS[band]
            tau, m = self.get_analytic_embedding_params(freq)

            expected_period = self.FS / freq

            if len(subbands) == 1:
                subband_signal = subbands[0]
            else:
             subband_signal = subbands[band]

            if subband_signal.std() > 0:
                subband_signal = (subband_signal - subband_signal.mean()) / subband_signal.std()

            if reference_epsilons is not None and band in reference_epsilons:
                epsilon = reference_epsilons[band]
            else:
                epsilon = RecurrencePlotGenerator.find_epsilon_for_target_rr(subband_signal, tau, m, target_rr=0.1)
                epsilons_used[band] = epsilon

            metrics = self.compute_rp_metrics(subband_signal, tau, m, epsilon)

            dominant_period = self.compute_dominant_period(metrics['rp_matrix'])

            det_deviation = abs(1.0 - metrics['DET'])  # DET should be 1.0 for pure sine
            entr_deviation = metrics['ENTR']            # ENTR should be 0.0
            lam_deviation = metrics['LAM']              # LAM should be 0.0
            tt_deviation = metrics['TT'] # should be 0
            longest_diagonal_line_deviation = abs(1 -  metrics['L_longest_diagonal'] / len(subband_signal)) #should approx the length of the signal
            longest_vertical_line_deviation = metrics['L_longest_vertical'] / len(subband_signal)#should approx the length of the signal
            period_error_abs = abs(dominant_period - expected_period)
            period_error_relative = period_error_abs / expected_period

            results.append({
                'pipeline': pipeline_name,
                'band': band,
                'freq_hz': freq,
                'tau': tau,
                'm': m,
                'epsilon': epsilon,

                'DET': metrics['DET'],
                'ENTR': metrics['ENTR'],
                'LAM': metrics['LAM'],
                'RR': metrics['RR'],

                'L_mean_diagonal': metrics['L_mean_diagonal'],
                'L_longest_diagonal': metrics['L_longest_diagonal'],
                'L_longest_vertical': metrics['L_longest_diagonal'],
                'TT': metrics['TT'],

                'dominant_period': dominant_period,
                'expected_period': expected_period,
                'det_deviation': det_deviation,
                'entr_deviation': entr_deviation,
                'lam_deviation': lam_deviation,
                'tt_deviation' : tt_deviation,
                'longest_diagonal_deviation': longest_diagonal_line_deviation,
                'longest_vertical_deviation': longest_vertical_line_deviation,
                'period_error_abs': period_error_abs,
                'period_error_relative': period_error_relative
            })

        if reference_epsilons is None:
            return pd.DataFrame(results), epsilons_used
        else:
            return pd.DataFrame(results), None

    def validate_with_noise(self, signal, pipeline_name, reference_epsilons, snr_db, n_realizations=50):
        """
        Validate pipeline with additive white Gaussian noise.

        Args:
            signal: Clean composite sine signal
            pipeline_name: Pipeline to validate
            reference_epsilons: Dictionary of epsilon values from clean signal
            snr_db: Signal-to-noise ratio in dB
            n_realizations: Number of noise realizations

        Returns:
            results_mean: Mean metrics across realizations
            results_std: Std metrics across realizations
        """
        all_results = []

        for _ in range(n_realizations):
            # Add AWGN
            signal_power = np.mean(signal ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.sqrt(noise_power) * np.random.randn(len(signal))
            noisy_signal = signal + noise

            # Validate
            df, _ = self.validate_pipeline(noisy_signal, pipeline_name, reference_epsilons)
            all_results.append(df)

        # Concatenate all realizations
        combined = pd.concat(all_results, ignore_index=True)

        # Compute mean and std for each band
        metric_cols = ['det_deviation', 'entr_deviation', 'lam_deviation', 'tt_deviation',
                      'longest_diagonal_deviation', 'longest_vertical_deviation', 'period_error_relative']

        results_mean = combined.groupby('band')[metric_cols].mean().reset_index()
        results_std = combined.groupby('band')[metric_cols].std().reset_index()

        results_mean['pipeline'] = pipeline_name
        results_mean['snr_db'] = snr_db
        results_std['pipeline'] = pipeline_name
        results_std['snr_db'] = snr_db

        return results_mean, results_std

    def run_full_validation(self):
        """
        Run complete validation on clean signal and with noise.

        Returns:
            clean_results: Dictionary with clean signal results per pipeline
            noise_results: Dictionary with noise results per pipeline
        """
        print("="*60)
        print("SUBBAND ISOLATION VALIDATION")
        print("="*60)

        # Generate composite sine signal
        print("\nGenerating composite sine signal...")
        signal = self.generate_composite_sine()
        print(f"  Signal length: {self.N} samples")
        print(f"  Sampling rate: {self.FS} Hz")
        print(f"  Component frequencies: {list(self.SUBBAND_FREQS.values())} Hz")

        clean_results = {}
        reference_epsilons = None

        # Validate clean signal for all pipelines
        print("\n" + "-"*60)
        print("CLEAN SIGNAL VALIDATION")
        print("-"*60)

        for pipeline in self.PIPELINES:
            print(f"\nValidating {pipeline}...")

            if pipeline == 'swt_recon':
                # Use SWT Recon as reference for epsilon values
                df, epsilons = self.validate_pipeline(signal, pipeline)
                reference_epsilons = epsilons
                print(f"  Reference epsilons: {epsilons}")
            else:
                # Use reference epsilons from SWT Recon
                df, _ = self.validate_pipeline(signal, pipeline, reference_epsilons)

            clean_results[pipeline] = df

            # Print summary
            print(f"\n  {pipeline} Results:")
            print(f"    Mean DET deviation: {df['det_deviation'].mean():.4f}")
            print(f"    Mean ENTR deviation: {df['entr_deviation'].mean():.4f}")
            print(f"    Mean LAM deviation: {df['lam_deviation'].mean():.4f}")
            print(f"    Mean TT deviation: {df['tt_deviation'].mean():.4f}")
            print(f"    Mean Longest vertical deviation: {df['longest_vertical_deviation'].mean():.4f}")
            print(f"    Mean Longest diagonal deviation: {df['longest_diagonal_deviation'].mean():.4f}")
            print(f"    Mean period error (relative): {df['period_error_relative'].mean():.3f} ({df['period_error_relative'].mean()*100:.1f}%)")

        # Validate with noise
        print("\n" + "-"*60)
        print("NOISE ROBUSTNESS VALIDATION")
        print("-"*60)

        noise_results = {}

        for snr_db in [20, 10, 5, 1]:
            print(f"\nSNR = {snr_db} dB (50 realizations)...")

            for pipeline in self.PIPELINES:
                print(f"  Testing {pipeline}...")

                mean_df, std_df = self.validate_with_noise(
                    signal, pipeline, reference_epsilons, snr_db, n_realizations=50
                )

                noise_results[f"{pipeline}_snr{snr_db}_mean"] = mean_df
                noise_results[f"{pipeline}_snr{snr_db}_std"] = std_df

        return clean_results, noise_results, reference_epsilons

    def compute_composite_scores(self, clean_results):
        """
        Compute normalized composite scores for each pipeline.
        Excludes A4 LAM (physical artifact) and uses relative period error.

        Args:
            clean_results: Dictionary with clean signal results

        Returns:
            DataFrame with composite scores
        """
        # Combine all results
        all_dfs = []
        for pipeline, df in clean_results.items():
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)

        # Exclude A4 LAM from scoring (physical artifact, not pipeline issue)
        combined_filtered = combined.copy()
        combined_filtered.loc[combined_filtered['band'] == 'A4', 'lam_deviation'] = np.nan

        # Use relative period error (normalized by expected period)
        metric_cols = ['det_deviation', 'entr_deviation', 'lam_deviation', 'period_error_relative']

        normalized = combined_filtered.copy()
        for col in metric_cols:
            # Skip NaN values when normalizing
            valid_vals = combined_filtered[col].dropna()
            if len(valid_vals) > 0:
                min_val = valid_vals.min()
                max_val = valid_vals.max()
                if max_val > min_val:
                    normalized[f'{col}_norm'] = (combined_filtered[col] - min_val) / (max_val - min_val)
                else:
                    normalized[f'{col}_norm'] = 0.0
            else:
                normalized[f'{col}_norm'] = 0.0

        # Compute composite score (mean of normalized metrics, excluding NaNs)
        norm_cols = [f'{col}_norm' for col in metric_cols]
        normalized['composite_score'] = normalized[norm_cols].mean(axis=1, skipna=True)

        # Aggregate by pipeline
        scores = normalized.groupby('pipeline')['composite_score'].mean().sort_values()

        return scores


def main():
    validator = SubbandValidator()

    clean_results, noise_results, ref_eps = validator.run_full_validation()

    print("COMPOSITE SCORES (Clean Signal)")

    scores = validator.compute_composite_scores(clean_results)
    print("\nRanking (Lower is Better):")
    for pipeline, score in scores.items():
        print(f"  {pipeline:<15}: {score:.4f}")

    winner = scores.idxmin()
    print(f"\n🏆 Winner: {winner}")

    print("\n" + "="*60)
    print("D3 SUBBAND RESULTS (12 Hz, Most Relevant for Pre-AF)")
    print("="*60)

    for pipeline, df in clean_results.items():
        d3_row = df[df['band'] == 'D3'].iloc[0]
        print(f"\n{pipeline}:")

        print(f"  DET deviation:      {d3_row['det_deviation']:.4f}")
        print(f"  ENTR deviation:     {d3_row['entr_deviation']:.4f}")
        print(f"  LAM deviation:      {d3_row['lam_deviation']:.4f}")
        print(f"  TT deviation:       {d3_row['tt_deviation'].mean():.4f}")
        print(f"  Longest vertical deviation:       {d3_row['longest_vertical_deviation'].mean():.4f}")
        print(f"  Mean Longest diagonal deviation:   {d3_row['longest_diagonal_deviation'].mean():.4f}")
        print(f"  Period error (rel): {d3_row['period_error_relative']:.3f} ({d3_row['period_error_relative']*100:.1f}%)")

    print("DETAILED RESULTS (All Subbands)")

    for pipeline, df in clean_results.items():
        print(f"\n{pipeline}:")
        print(df[['band', 'det_deviation', 'entr_deviation', 'lam_deviation', 'tt_deviation', 'longest_vertical_deviation', 'longest_diagonal_deviation', 'period_error_relative']].to_string(index=False))

    print("NOISE ROBUSTNESS SUMMARY")
    for snr_db in [20, 10, 5, 1]:
        print(f"\nSNR = {snr_db} dB:")
        for pipeline in validator.PIPELINES:
            mean_df = noise_results[f"{pipeline}_snr{snr_db}_mean"]
            std_df = noise_results[f"{pipeline}_snr{snr_db}_std"]

            print(f"\n  {pipeline}:")
            print(f"    DET deviation: {mean_df['det_deviation'].mean():.4f} ± {std_df['det_deviation'].mean():.4f}")
            print(f"    ENTR deviation: {mean_df['entr_deviation'].mean():.4f} ± {std_df['entr_deviation'].mean():.4f}")
            print(f"    LAM deviation: {mean_df['lam_deviation'].mean():.4f} ± {std_df['lam_deviation'].mean():.4f}")
            print(f"    TT deviation: {mean_df['tt_deviation'].mean():.4f}")
            print(f"    Longest vertical deviation: {mean_df['longest_vertical_deviation'].mean():.4f}")
            print(f"    Mean Longest diagonal deviation: {mean_df['longest_diagonal_deviation'].mean():.4f}")
            print(f"    Period error (relative): {mean_df['period_error_relative'].mean():.3f} ± {std_df['period_error_relative'].mean():.3f}")

    results_file = os.path.join(validator.OUTPUT_DIR, 'subband_validation_results.csv')
    all_results = pd.concat(clean_results.values(), ignore_index=True)
    all_results.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")


def run_statistical_comparison(n_trials=100):
    from scipy.stats import ttest_ind

    validator = SubbandValidator()

    results = {
        'swt_recon': {'DET': [], 'LAM': [], 'ENTR': [], "TT" : [], "L_longest_diagonal": [], "L_longest_vertical": []},
        'swt_direct':  {'DET': [], 'LAM': [], 'ENTR': [], "TT" : [], "L_longest_diagonal": [], "L_longest_vertical": []},
        'dwt_recon':  {'DET': [], 'LAM': [], 'ENTR': [], "TT" : [], "L_longest_diagonal": [], "L_longest_vertical": []},
    }

    pipelines = ['swt_recon', 'swt_direct', 'dwt_recon']

    # Run trials
    for trial in range(n_trials):
        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{n_trials}...")

        # Generate composite sine with random phase
        signal = validator.generate_composite_sine(random_phase=True)

        # Decompose and compute RQA for each pipeline
        for pipeline in pipelines:
            df, _ = validator.validate_pipeline(signal, pipeline)

            # Average metrics across all subbands
            avg_det = df['det_deviation'].mean()
            avg_lam = df['lam_deviation'].mean()
            avg_entr = df['entr_deviation'].mean()
            avg_tt = df['tt_deviation'].mean()
            avg_l_vert_max = df['longest_vertical_deviation'].mean()
            avg_l_diag_max = df['longest_diagonal_deviation'].mean()

            results[pipeline]['DET'].append(avg_det)
            results[pipeline]['LAM'].append(avg_lam)
            results[pipeline]['ENTR'].append(avg_entr)
            # results[pipeline]['TT'].append(avg_tt)
            # results[pipeline]['L_longest_diagonal'].append(avg_l_diag_max)
            # results[pipeline]['L_longest_vertical'].append(avg_l_vert_max)

    # Convert to numpy arrays
    for pipeline in pipelines:
        for metric in ['DET', 'LAM', 'ENTR']:
            results[pipeline][metric] = np.array(results[pipeline][metric])

    # Perform Welch's t-tests
    print("\n" + "="*60)
    print("WELCH'S T-TEST RESULTS")
    print("="*60)

    comparisons = [
        ('swt_recon', 'swt_direct', 'SWT Recon vs SWT Direct'),
        ('swt_recon', 'dwt_recon', 'SWT Recon vs DWT Recon'),
        ('swt_direct', 'dwt_recon', 'SWT Direct vs DWT Recon')
    ]

    stats_results = []

    for pipe1, pipe2, label in comparisons:
        print(f"\n{label}:")
        print("-" * 60)

        for metric in ['DET', 'LAM', 'ENTR']:
            data1 = results[pipe1][metric]
            data2 = results[pipe2][metric]

            # Check if data has variance
            if data1.std() < 1e-10 and data2.std() < 1e-10:
                # Both groups have no variance - skip test
                print(f"  {metric}: No variance in either group - test skipped")
                continue

            # Welch's t-test (unequal variances)
            t_stat, p_value = ttest_ind(data1, data2, equal_var=False)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
            if pooled_std < 1e-10:
                cohens_d = 0.0
            else:
                cohens_d = (data1.mean() - data2.mean()) / pooled_std

            # Determine significance
            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = 'ns'

            print(f"  {metric}:")
            print(f"    {pipe1}: {data1.mean():.6f} ± {data1.std():.6f}")
            print(f"    {pipe2}: {data2.mean():.6f} ± {data2.std():.6f}")
            print(f"    t = {t_stat:.4f}, p = {p_value:.6f} {sig}")
            print(f"    Cohen's d = {cohens_d:.4f}")

            stats_results.append({
                'comparison': label,
                'metric': metric,
                'pipeline1': pipe1,
                'pipeline2': pipe2,
                'mean1': data1.mean(),
                'std1': data1.std(),
                'mean2': data2.mean(),
                'std2': data2.std(),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significance': sig
            })

    # Save results
    stats_df = pd.DataFrame(stats_results)
    stats_file = os.path.join(validator.OUTPUT_DIR, 'welch_ttest_results.csv')
    stats_df.to_csv(stats_file, index=False)

    print("\n" + "="*60)
    print(f"Statistical results saved to: {stats_file}")
    print("="*60)

    # Visualize distributions
    print("\nGenerating visualizations...")
    visualize_statistical_distributions(results, validator.OUTPUT_DIR)
    visualize_bar_comparison(results, validator.OUTPUT_DIR)
    visualize_snr_degradation(validator, validator.OUTPUT_DIR, n_realizations=30)

    return results, stats_df


def visualize_bar_comparison(results, output_dir):
    """Create bar charts comparing pipelines across metrics with error bars."""
    import matplotlib.pyplot as plt

    pipelines = ['swt_recon', 'swt_direct', 'dwt_recon']
    metrics = ['DET', 'LAM', 'ENTR']

    colors = {
        'swt_recon': '#2ecc71',
        'swt_direct': '#3498db',
        'dwt_recon': '#e74c3c'
    }

    labels = {
        'swt_recon': 'SWT Recon',
        'swt_direct': 'SWT Direct',
        'dwt_recon': 'DWT Recon'
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        means = [results[p][metric].mean() for p in pipelines]
        stds = [results[p][metric].std() for p in pipelines]

        x_pos = np.arange(len(pipelines))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=[colors[p] for p in pipelines],
                     edgecolor='black', linewidth=1.5, alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([labels[p] for p in pipelines], fontsize=11)
        ax.set_ylabel(f'{metric} Deviation', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.005, f'{mean:.4f}±{std:.4f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    filename = os.path.join(output_dir, 'bar_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved bar comparison: {filename}")
    plt.show()


def visualize_statistical_distributions(results, output_dir):
    """Create histogram distributions for each metric."""
    import matplotlib.pyplot as plt

    pipelines = ['swt_recon', 'swt_direct', 'dwt_recon']
    metrics = ['DET', 'LAM', 'ENTR']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {
        'swt_recon': '#2ecc71',
        'swt_direct': '#3498db',
        'dwt_recon': '#e74c3c'
    }

    labels = {
        'swt_recon': 'SWT Recon',
        'swt_direct': 'SWT Direct',
        'dwt_recon': 'DWT Recon'
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for pipeline in pipelines:
            data = results[pipeline][metric]
            ax.hist(data, bins=15, alpha=0.5, color=colors[pipeline],
                   label=labels[pipeline], edgecolor='black')

        ax.set_xlabel(f'{metric} Deviation', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{metric} Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    filename = os.path.join(output_dir, 'statistical_distributions.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved distribution plot: {filename}")
    plt.show()


def visualize_snr_degradation(validator, output_dir, n_realizations=50):
    """Create SNR degradation curves for each pipeline."""
    import matplotlib.pyplot as plt

    print("\nGenerating SNR degradation curves...")

    signal_clean = validator.generate_composite_sine()
    snr_levels = [30, 20, 15, 10, 5, 1, 0]
    metrics = ['det_deviation', 'lam_deviation', 'entr_deviation']

    pipelines = ['swt_recon', 'swt_direct', 'dwt_recon']

    # Store results
    snr_results = {p: {m: {'mean': [], 'std': []} for m in metrics} for p in pipelines}

    # Compute reference epsilons from clean signal
    ref_epsilons = {}
    for pipeline in pipelines:
        subbands_clean = validator.decompose_signal(signal_clean, pipeline)
        ref_epsilons[pipeline] = {}

        for band in validator.BAND_NAMES:
            freq = validator.SUBBAND_FREQS[band]
            tau, m = validator.get_analytic_embedding_params(freq)
            subband_signal = subbands_clean[band]

            if subband_signal.std() > 0:
                subband_signal = (subband_signal - subband_signal.mean()) / subband_signal.std()

            ref_epsilons[pipeline][band] = RecurrencePlotGenerator.find_epsilon_for_target_rr(
                subband_signal, tau, m, target_rr=0.1)

    # Test each SNR level
    for snr_db in snr_levels:
        print(f"  Testing SNR = {snr_db} dB...")

        for pipeline in pipelines:
            mean_df, std_df = validator.validate_with_noise(
                signal_clean, pipeline, ref_epsilons[pipeline], snr_db, n_realizations)

            for metric in metrics:
                snr_results[pipeline][metric]['mean'].append(mean_df[metric].mean())
                snr_results[pipeline][metric]['std'].append(std_df[metric].mean())

    # Plot
    colors = {
        'swt_recon': '#2ecc71',
        'swt_direct': '#3498db',
        'dwt_recon': '#e74c3c'
    }

    labels = {
        'swt_recon': 'SWT Recon',
        'swt_direct': 'SWT Direct',
        'dwt_recon': 'DWT Recon'
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for pipeline in pipelines:
            means = np.array(snr_results[pipeline][metric]['mean'])
            stds = np.array(snr_results[pipeline][metric]['std'])

            ax.plot(snr_levels, means, marker='o', linewidth=2, markersize=8,
                   color=colors[pipeline], label=labels[pipeline])
            ax.fill_between(snr_levels, means - stds, means + stds,
                           color=colors[pipeline], alpha=0.2)

        ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} vs SNR', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.invert_xaxis()  # Higher SNR on left

    plt.tight_layout()
    filename = os.path.join(output_dir, 'snr_degradation_curves.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved SNR degradation curves: {filename}")
    plt.show()


if __name__ == '__main__':
    n_trials = 200
    # run_statistical_comparison(n_trials=n_trials)
    main()
