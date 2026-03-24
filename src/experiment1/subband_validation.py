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
    """Validates subband isolation quality using synthetic multi-sine signal."""

    def __init__(self):
        self.FS = 128  # Sampling frequency (Hz)
        self.N = 2048  # Signal length (samples)
        self.LEVEL = 4  # Wavelet decomposition level
        self.WAVELET = 'db4'

        # Subband center frequencies (Hz)
        self.SUBBAND_FREQS = {
            'A4': 2,   # 0-4 Hz
            'D4': 6,   # 4-8 Hz
            'D3': 12,  # 8-16 Hz
            'D2': 24,  # 16-32 Hz
            'D1': 48   # 32-64 Hz
        }

        self.BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']
        self.PIPELINES = ['swt_recon', 'swt_direct', 'dwt_recon', 'dwt_direct']

        self.OUTPUT_DIR = '/Users/filipsulich/Developer/uni/Thesis/Thesis/outputs/experiment1_synthetic'
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def generate_composite_sine(self):
        """
        Generate composite sine signal with one component per subband.
        All components have equal amplitude.

        Returns:
            signal: Composite sine signal (N samples)
        """
        t = np.arange(self.N) / self.FS
        signal = np.zeros(self.N)

        for band, freq in self.SUBBAND_FREQS.items():
            signal += np.sin(2 * np.pi * freq * t)

        return signal

    def get_analytic_embedding_params(self, freq):
        """
        Compute embedding parameters analytically from known sine frequency.

        Args:
            freq: Sine frequency in Hz

        Returns:
            tau: Embedding delay (samples)
            m: Embedding dimension
        """
        T = self.FS / freq  # Period in samples
        tau = max(1, int(T / 4))  # τ = T/4, minimum 1
        m = 2               # Fixed for all subbands
        return tau, m

    def decompose_signal(self, signal, pipeline_name):
        """
        Decompose signal using specified pipeline.

        Args:
            signal: Input signal
            pipeline_name: Pipeline to use

        Returns:
            subbands: Dictionary of subband signals
        """
        pipeline_obj = Pipeline(pipeline_name='all', level=self.LEVEL, wavelet=self.WAVELET)
        func = getattr(pipeline_obj, f"pipeline_{pipeline_name}")
        return func(signal)

    def compute_rp_metrics(self, signal, tau, m, epsilon):
        """
        Compute RQA metrics for a signal with fixed parameters.

        Args:
            signal: Input signal
            tau: Embedding delay
            m: Embedding dimension
            epsilon: Recurrence threshold

        Returns:
            Dictionary with RQA metrics
        """
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

            return {
                'DET': rqa_result.determinism,
                'ENTR': rqa_result.entropy_diagonal_lines,
                'LAM': rqa_result.laminarity,
                'RR': rqa_result.recurrence_rate,
                'rp_matrix': rp_matrix
            }
        except Exception as e:
            print(f"Warning: RQA computation failed: {e}")
            return {
                'DET': 0.0,
                'ENTR': 0.0,
                'LAM': 0.0,
                'RR': 0.0,
                'rp_matrix': None
            }

    def find_epsilon_for_target_rr(self, signal, tau, m, target_rr=0.05):
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
        """
        Validate a single pipeline on the composite sine signal.

        Args:
            signal: Composite sine signal
            pipeline_name: Pipeline to validate
            reference_epsilons: Dictionary of epsilon values (optional)

        Returns:
            results: DataFrame with metrics per subband
            epsilons: Dictionary of epsilon values used
        """
        # Decompose signal
        subbands = self.decompose_signal(signal, pipeline_name)

        results = []
        epsilons_used = {}

        for band in self.BAND_NAMES:
            freq = self.SUBBAND_FREQS[band]
            tau, m = self.get_analytic_embedding_params(freq)
            expected_period = self.FS / freq  # Expected period in samples

            subband_signal = subbands[band]

            # Normalize
            if subband_signal.std() > 0:
                subband_signal = (subband_signal - subband_signal.mean()) / subband_signal.std()

            # Find or use epsilon
            if reference_epsilons is not None and band in reference_epsilons:
                epsilon = reference_epsilons[band]
            else:
                epsilon = self.find_epsilon_for_target_rr(subband_signal, tau, m, target_rr=0.05)
                epsilons_used[band] = epsilon

            # Compute RQA
            metrics = self.compute_rp_metrics(subband_signal, tau, m, epsilon)

            # Compute dominant period
            dominant_period = self.compute_dominant_period(metrics['rp_matrix'])

            # Compute deviations from theoretical values
            det_deviation = abs(1.0 - metrics['DET'])  # DET should be 1.0 for pure sine
            entr_deviation = metrics['ENTR']            # ENTR should be 0.0
            lam_deviation = metrics['LAM']              # LAM should be 0.0
            period_error = abs(dominant_period - expected_period)  # Period should match

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
                'dominant_period': dominant_period,
                'expected_period': expected_period,
                'det_deviation': det_deviation,
                'entr_deviation': entr_deviation,
                'lam_deviation': lam_deviation,
                'period_error': period_error
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
        metric_cols = ['det_deviation', 'entr_deviation', 'lam_deviation', 'period_error']

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
            print(f"    Mean period error: {df['period_error'].mean():.2f} samples")

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

        # Normalize each metric to [0,1] across pipelines
        metric_cols = ['det_deviation', 'entr_deviation', 'lam_deviation', 'period_error']

        normalized = combined.copy()
        for col in metric_cols:
            min_val = combined[col].min()
            max_val = combined[col].max()
            if max_val > min_val:
                normalized[f'{col}_norm'] = (combined[col] - min_val) / (max_val - min_val)
            else:
                normalized[f'{col}_norm'] = 0.0

        # Compute composite score (mean of normalized metrics)
        norm_cols = [f'{col}_norm' for col in metric_cols]
        normalized['composite_score'] = normalized[norm_cols].mean(axis=1)

        # Aggregate by pipeline
        scores = normalized.groupby('pipeline')['composite_score'].mean().sort_values()

        return scores

    def visualize_decomposition_and_rps(self, signal, pipeline_name, reference_epsilons):
        """
        Visualize signal decomposition and corresponding RPs for a single pipeline.

        Args:
            signal: Composite sine signal
            pipeline_name: Pipeline to visualize
            reference_epsilons: Dictionary of epsilon values per band

        Returns:
            Figure object
        """
        # Decompose signal
        subbands = self.decompose_signal(signal, pipeline_name)

        # Create figure with grid: Original signal + RP on top, then 5 rows for subbands
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(6, 3, height_ratios=[1.2, 1, 1, 1, 1, 1], hspace=0.3, wspace=0.3)

        # Row 0: Original composite signal + its RP
        ax_orig = fig.add_subplot(gs[0, :2])
        t = np.arange(len(signal)) / self.FS
        ax_orig.plot(t, signal, 'k-', linewidth=1.5)
        ax_orig.set_title(f'Composite Sine Signal (5 components: 2, 6, 12, 24, 48 Hz)', fontweight='bold')
        ax_orig.set_xlabel('Time (s)')
        ax_orig.set_ylabel('Amplitude')
        ax_orig.grid(alpha=0.3)

        # Compute RP for original signal
        ax_rp_orig = fig.add_subplot(gs[0, 2])
        sig_norm = (signal - signal.mean()) / (signal.std() + 1e-10)
        tau_orig, m_orig = 16, 2  # Fixed params for composite
        epsilon_orig = self.find_epsilon_for_target_rr(sig_norm, tau_orig, m_orig, target_rr=0.05)

        try:
            os.environ['PYOPENCL_CTX'] = '0'
            rp_gen = RecurrencePlotGenerator(
                signal=sig_norm, time_delay=tau_orig, embedding_dim=m_orig,
                path=self.OUTPUT_DIR, epsilon=epsilon_orig, unthresholded=False
            )
            _, rp_result = rp_gen.compute_rqa_binary()
            rp_matrix = rp_result.recurrence_matrix_reverse_normalized
            ax_rp_orig.imshow(rp_matrix, cmap='binary', origin='lower')
            ax_rp_orig.set_title(f'RP (Original)\nτ={tau_orig}, m={m_orig}', fontsize=10)
        except:
            ax_rp_orig.text(0.5, 0.5, 'RP Failed', ha='center', va='center')
        ax_rp_orig.set_xticks([])
        ax_rp_orig.set_yticks([])

        # Rows 1-5: Each subband + its RP
        for idx, band in enumerate(self.BAND_NAMES):
            row = idx + 1
            freq = self.SUBBAND_FREQS[band]
            tau, m = self.get_analytic_embedding_params(freq)
            epsilon = reference_epsilons[band]

            subband_signal = subbands[band]

            # Normalize
            if subband_signal.std() > 0:
                subband_signal_norm = (subband_signal - subband_signal.mean()) / subband_signal.std()
            else:
                subband_signal_norm = subband_signal

            # Plot decomposed signal
            ax_sig = fig.add_subplot(gs[row, :2])
            t_sub = np.arange(len(subband_signal)) / self.FS
            ax_sig.plot(t_sub, subband_signal, linewidth=1.2)
            ax_sig.set_title(f'{band} ({freq} Hz) - τ={tau}, m={m}', fontweight='bold', fontsize=10)
            ax_sig.set_xlabel('Time (s)', fontsize=9)
            ax_sig.set_ylabel('Amplitude', fontsize=9)
            ax_sig.grid(alpha=0.3)

            # Compute and plot RP
            ax_rp = fig.add_subplot(gs[row, 2])
            try:
                rp_gen = RecurrencePlotGenerator(
                    signal=subband_signal_norm, time_delay=tau, embedding_dim=m,
                    path=self.OUTPUT_DIR, epsilon=epsilon, unthresholded=False
                )
                rqa_result, rp_result = rp_gen.compute_rqa_binary()
                rp_matrix = rp_result.recurrence_matrix_reverse_normalized

                ax_rp.imshow(rp_matrix, cmap='binary', origin='lower')
                ax_rp.set_title(f'RP: DET={rqa_result.determinism:.3f}\nENTR={rqa_result.entropy_diagonal_lines:.3f}',
                               fontsize=9)
            except:
                ax_rp.text(0.5, 0.5, 'RP Failed', ha='center', va='center', fontsize=8)

            ax_rp.set_xticks([])
            ax_rp.set_yticks([])

        fig.suptitle(f'Pipeline: {pipeline_name.upper()}', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save figure
        filename = f'decomposition_rp_{pipeline_name}.png'
        plt.savefig(os.path.join(self.OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")

        return fig


def main():
    """Run subband validation."""

    validator = SubbandValidator()

    # Run full validation
    clean_results, noise_results, ref_eps = validator.run_full_validation()

    # Compute composite scores
    print("\n" + "="*60)
    print("COMPOSITE SCORES (Clean Signal)")
    print("="*60)

    scores = validator.compute_composite_scores(clean_results)
    print("\nRanking (Lower is Better):")
    for pipeline, score in scores.items():
        print(f"  {pipeline:<15}: {score:.4f}")

    winner = scores.idxmin()
    print(f"\n🏆 Winner: {winner}")

    # Print detailed results
    print("\n" + "="*60)
    print("DETAILED RESULTS (Clean Signal)")
    print("="*60)

    for pipeline, df in clean_results.items():
        print(f"\n{pipeline}:")
        print(df[['band', 'det_deviation', 'entr_deviation', 'lam_deviation', 'period_error']].to_string(index=False))

    # Print noise results summary
    print("\n" + "="*60)
    print("NOISE ROBUSTNESS SUMMARY")
    print("="*60)

    for snr_db in [20, 10]:
        print(f"\nSNR = {snr_db} dB:")
        for pipeline in validator.PIPELINES:
            mean_df = noise_results[f"{pipeline}_snr{snr_db}_mean"]
            std_df = noise_results[f"{pipeline}_snr{snr_db}_std"]

            print(f"\n  {pipeline}:")
            print(f"    DET deviation: {mean_df['det_deviation'].mean():.4f} ± {std_df['det_deviation'].mean():.4f}")
            print(f"    ENTR deviation: {mean_df['entr_deviation'].mean():.4f} ± {std_df['entr_deviation'].mean():.4f}")
            print(f"    LAM deviation: {mean_df['lam_deviation'].mean():.4f} ± {std_df['lam_deviation'].mean():.4f}")
            print(f"    Period error: {mean_df['period_error'].mean():.2f} ± {std_df['period_error'].mean():.2f} samples")

    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING DECOMPOSITION & RP VISUALIZATIONS")
    print("="*60)

    signal = validator.generate_composite_sine()

    for pipeline in validator.PIPELINES:
        print(f"\nGenerating visualization for {pipeline}...")
        validator.visualize_decomposition_and_rps(signal, pipeline, ref_eps)

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print(f"\nVisualizations saved to: {validator.OUTPUT_DIR}")


if __name__ == '__main__':
    main()
