import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.pipeline_wavelets import Pipeline
from src.experiment1.visualize_decomposition import DecompositionVisualizer


class ChaoticReconstructionTest:
    """
    Test reconstruction error for chaotic systems (Lorenz and Rossler).

    For each pipeline (SWT recon, SWT direct, DWT recon, DWT direct):
    1. Take a snippet of the x-axis from Lorenz or Rossler system
    2. Decompose it using db4 wavelet
    3. Sum up all decomposed subbands
    4. Compute reconstruction error (RMS) comparing summed signal with original input

    This tests whether the wavelet decomposition is perfectly invertible and whether
    the sum of subbands equals the original signal.
    """

    BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']
    PIPELINES = ['swt_recon', 'dwt_recon']
    SYSTEM_TYPES = ['rossler', 'lorenz']

    def __init__(
        self,
        wavelet: str = 'db4',
        level: int = 4,
        n_samples: int = 1024,
        n_trials: int = 100,
        output_dir: str = 'chaotic_reconstruction_output'
    ):
        """
        Initialize chaotic reconstruction test.

        Args:
            wavelet: Wavelet name (default: 'db4')
            level: Decomposition level (default: 4)
            n_samples: Number of samples to use from the chaotic system
            n_trials: Number of trials with different initial conditions (default: 100)
            output_dir: Directory to save outputs
        """
        self.wavelet = wavelet
        self.level = level
        self.n_samples = n_samples
        self.n_trials = n_trials
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._results = []

    def _generate_signal(self, system_type: str, seed: int = 42) -> np.ndarray:
        """
        Generate a snippet from the specified chaotic system.

        Args:
            system_type: 'rossler' or 'lorenz'
            seed: Random seed for reproducibility

        Returns:
            Normalized signal snippet
        """
        # Create visualizer with specific seed for this trial
        visualizer = DecompositionVisualizer(seed=seed)

        if system_type == 'rossler':
            signal = visualizer._generate_rossler(seed=seed)
        elif system_type == 'lorenz':
            signal = visualizer.generate_lorenz_system(seed=seed)
        else:
            raise ValueError(f"Unknown system type: {system_type}")

        # Take only n_samples
        return signal[:self.n_samples]

    def _compute_reconstruction_error(
        self,
        original: np.ndarray,
        subbands: dict,
        pipeline_name: str
    ) -> float:
        """
        Compute RMS reconstruction error between original signal and sum of subbands.

        Args:
            original: Original input signal
            subbands: Dictionary of decomposed subbands
            pipeline_name: Name of the pipeline used

        Returns:
            RMS reconstruction error
        """
        # Sum all subbands
        reconstructed = np.zeros_like(original)

        for band in self.BAND_NAMES:
            subband = subbands[band]

            # Handle different lengths for direct pipelines (DWT/SWT direct)
            if len(subband) != len(original):
                # For direct pipelines, we need to pad or truncate
                # Typically, DWT direct coefficients are shorter
                # We'll use zero-padding or truncation
                if len(subband) < len(original):
                    # Pad with zeros
                    padded = np.zeros(len(original))
                    padded[:len(subband)] = subband
                    reconstructed += padded
                else:
                    # Truncate
                    reconstructed += subband[:len(original)]
            else:
                reconstructed += subband

        # Compute RMS error
        rms_error = np.sqrt(np.mean((original - reconstructed) ** 2))

        # Also compute relative error (normalized by signal magnitude)
        signal_magnitude = np.sqrt(np.mean(original ** 2))
        relative_error = rms_error / signal_magnitude if signal_magnitude > 1e-9 else np.inf

        return rms_error, relative_error

    def run_test(self):
        """Run reconstruction test for all systems and pipelines with multiple trials."""
        print(f"\n{'='*70}")
        print(f"CHAOTIC SYSTEM RECONSTRUCTION TEST")
        print(f"Wavelet: {self.wavelet}, Level: {self.level}, Samples: {self.n_samples}")
        print(f"Trials: {self.n_trials} (different initial conditions)")
        print(f"{'='*70}\n")

        for system_type in self.SYSTEM_TYPES:
            print(f"\nTesting {system_type.upper()} system...")

            for pipeline_name in self.PIPELINES:
                print(f"  Pipeline: {pipeline_name}")

                # Create pipeline
                pipeline = Pipeline(
                    pipeline_name=pipeline_name,
                    level=self.level,
                    wavelet=self.wavelet
                )
                func = getattr(pipeline, f"pipeline_{pipeline_name}")

                # Run multiple trials with different initial conditions
                for trial_idx in range(self.n_trials):
                    # Generate signal with unique seed for each trial
                    seed = trial_idx + 1000  # Start from 1000 to avoid conflicts
                    signal = self._generate_signal(system_type, seed=seed)

                    # Decompose signal
                    subbands = func(signal)

                    # Compute reconstruction error
                    rms_error, relative_error = self._compute_reconstruction_error(
                        signal, subbands, pipeline_name
                    )

                    # Store results
                    result = {
                        'system': system_type,
                        'pipeline': pipeline_name,
                        'trial': trial_idx,
                        'seed': seed,
                        'wavelet': self.wavelet,
                        'level': self.level,
                        'n_samples': self.n_samples,
                        'rms_error': rms_error,
                        'relative_error': relative_error,
                    }

                    # Also store individual subband energies for analysis
                    for band in self.BAND_NAMES:
                        energy = np.sum(subbands[band] ** 2)
                        result[f'{band}_energy'] = energy

                    # Store signal and reconstruction for first trial only (to save memory)
                    if trial_idx == 0:
                        result['_signal'] = signal
                        result['_subbands'] = subbands

                    self._results.append(result)

                if (trial_idx + 1) % 20 == 0 or trial_idx == 0:
                    print(f"    Trial {trial_idx + 1}/{self.n_trials} done")

        # Compute statistics
        print("\n" + "="*70)
        print("STATISTICAL ANALYSIS")
        print("="*70)
        self._compute_statistics()

        # Save results
        self._save_results()
        self._plot_results()

        print(f"\n{'='*70}")
        print(f"Results saved to {self.output_dir}")
        print(f"{'='*70}\n")

    def _compute_statistics(self):
        """Compute statistical comparisons between SWT recon and DWT recon."""
        stats_results = []

        for system_type in self.SYSTEM_TYPES:
            print(f"\n{system_type.upper()} System:")

            # Extract RMS errors for each pipeline
            swt_errors = [r['rms_error'] for r in self._results
                         if r['system'] == system_type and r['pipeline'] == 'swt_recon']
            dwt_errors = [r['rms_error'] for r in self._results
                         if r['system'] == system_type and r['pipeline'] == 'dwt_recon']

            swt_errors = np.array(swt_errors)
            dwt_errors = np.array(dwt_errors)

            # Compute summary statistics
            swt_mean = np.mean(swt_errors)
            swt_median = np.median(swt_errors)
            swt_std = np.std(swt_errors)

            dwt_mean = np.mean(dwt_errors)
            dwt_median = np.median(dwt_errors)
            dwt_std = np.std(dwt_errors)

            print(f"  SWT Recon: Mean={swt_mean:.6e}, Median={swt_median:.6e}, Std={swt_std:.6e}")
            print(f"  DWT Recon: Mean={dwt_mean:.6e}, Median={dwt_median:.6e}, Std={dwt_std:.6e}")

            # Mann-Whitney U test
            u_stat, p_value = mannwhitneyu(swt_errors, dwt_errors, alternative='two-sided')
            print(f"  Mann-Whitney U: U={u_stat:.2f}, p={p_value:.6e}")

            # Cohen's d (effect size)
            pooled_std = np.sqrt((swt_std**2 + dwt_std**2) / 2)
            cohens_d = (swt_mean - dwt_mean) / pooled_std if pooled_std > 1e-20 else 0.0
            print(f"  Cohen's d: {cohens_d:.6f}")

            # Interpretation
            if abs(cohens_d) < 0.2:
                effect_size = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_size = "small"
            elif abs(cohens_d) < 0.8:
                effect_size = "medium"
            else:
                effect_size = "large"

            print(f"  Effect size: {effect_size}")

            # Store statistics
            stats_results.append({
                'system': system_type,
                'swt_mean': swt_mean,
                'swt_median': swt_median,
                'swt_std': swt_std,
                'dwt_mean': dwt_mean,
                'dwt_median': dwt_median,
                'dwt_std': dwt_std,
                'mann_whitney_u': u_stat,
                'mann_whitney_p': p_value,
                'cohens_d': cohens_d,
                'effect_size': effect_size,
                '_swt_errors': swt_errors,
                '_dwt_errors': dwt_errors,
            })

        # Save statistics
        self._stats_results = stats_results
        stats_df = pd.DataFrame([{k: v for k, v in s.items() if not k.startswith('_')}
                                for s in stats_results])
        stats_path = os.path.join(self.output_dir, 'statistics.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"\nSaved statistics to {stats_path}")

    def _save_results(self):
        """Save results to CSV."""
        # Prepare summary table (exclude internal fields)
        summary_rows = []
        for result in self._results:
            row = {k: v for k, v in result.items() if not k.startswith('_')}
            summary_rows.append(row)

        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(self.output_dir, 'reconstruction_errors_all_trials.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSaved all trials to {csv_path}")

    def _plot_results(self):
        """Generate plots for reconstruction test results."""
        self._plot_violin_plots()
        self._plot_error_summary()
        self._plot_reconstruction_examples()

    def _plot_violin_plots(self):
        """Plot violin plots of RMS errors for all systems and pipelines."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, system_type in enumerate(self.SYSTEM_TYPES):
            ax = axes[idx]

            # Prepare data for violin plot
            data_rows = []
            for result in self._results:
                if result['system'] == system_type:
                    data_rows.append({
                        'pipeline': result['pipeline'],
                        'rms_error': result['rms_error']
                    })

            df = pd.DataFrame(data_rows)

            # Create violin plot
            sns.violinplot(
                data=df,
                x='pipeline',
                y='rms_error',
                ax=ax,
                palette='Set2',
                inner='box'
            )

            # Add statistical annotation
            if hasattr(self, '_stats_results'):
                stats = next((s for s in self._stats_results if s['system'] == system_type), None)
                if stats:
                    # Add text box with statistics
                    textstr = f"Mann-Whitney U: p={stats['mann_whitney_p']:.3e}\n"
                    textstr += f"Cohen's d: {stats['cohens_d']:.4f} ({stats['effect_size']})"
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                           fontsize=9, verticalalignment='top', bbox=props)

            ax.set_xlabel('Pipeline', fontweight='bold')
            ax.set_ylabel('RMS Error', fontweight='bold')
            ax.set_title(f'{system_type.upper()} System - RMS Error Distribution\n'
                        f'({self.n_trials} trials)', fontweight='bold')
            ax.set_yscale('log')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, 'violin_plots.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved violin plots to {out_path}")

    def _plot_error_summary(self):
        """Plot summary statistics (mean and median) for RMS errors."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, system_type in enumerate(self.SYSTEM_TYPES):
            ax = axes[idx]

            # Extract errors
            swt_errors = [r['rms_error'] for r in self._results
                         if r['system'] == system_type and r['pipeline'] == 'swt_recon']
            dwt_errors = [r['rms_error'] for r in self._results
                         if r['system'] == system_type and r['pipeline'] == 'dwt_recon']

            swt_errors = np.array(swt_errors)
            dwt_errors = np.array(dwt_errors)

            # Compute statistics
            pipelines = ['SWT Recon', 'DWT Recon']
            means = [np.mean(swt_errors), np.mean(dwt_errors)]
            medians = [np.median(swt_errors), np.median(dwt_errors)]
            stds = [np.std(swt_errors), np.std(dwt_errors)]

            x = np.arange(len(pipelines))
            width = 0.35

            # Plot bars
            bars1 = ax.bar(x - width/2, means, width, label='Mean',
                          yerr=stds, capsize=5, alpha=0.8, color='steelblue')
            bars2 = ax.bar(x + width/2, medians, width, label='Median',
                          alpha=0.8, color='coral')

            ax.set_xlabel('Pipeline', fontweight='bold')
            ax.set_ylabel('RMS Error', fontweight='bold')
            ax.set_title(f'{system_type.upper()} System - Error Summary', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(pipelines)
            ax.set_yscale('log')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, 'error_summary.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved error summary to {out_path}")


    def _plot_reconstruction_examples(self):
        """Plot example reconstructions for visual inspection (first trial only)."""
        for system_type in self.SYSTEM_TYPES:
            # Filter results for this system (trial 0 only, which has stored signals)
            system_results = [r for r in self._results
                            if r['system'] == system_type and r.get('trial', -1) == 0
                            and '_signal' in r]

            if not system_results:
                continue

            fig, axes = plt.subplots(len(self.PIPELINES), 1,
                                    figsize=(14, 3 * len(self.PIPELINES)))

            if len(self.PIPELINES) == 1:
                axes = [axes]

            fig.suptitle(f'{system_type.upper()} System - Reconstruction Example (Trial 0)',
                        fontsize=14, fontweight='bold')

            for idx, result in enumerate(system_results):
                ax = axes[idx]
                pipeline_name = result['pipeline']
                signal = result['_signal']
                subbands = result['_subbands']

                # Compute reconstruction (sum of subbands)
                reconstructed = np.zeros_like(signal)
                for band in self.BAND_NAMES:
                    subband = subbands[band]
                    if len(subband) < len(signal):
                        padded = np.zeros(len(signal))
                        padded[:len(subband)] = subband
                        reconstructed += padded
                    else:
                        reconstructed += subband[:len(signal)]

                # Plot
                time_axis = np.arange(len(signal))
                ax.plot(time_axis, signal, label='Original', linewidth=1.5, alpha=0.7)
                ax.plot(time_axis, reconstructed, label='Reconstructed (sum of subbands)',
                       linewidth=1.5, alpha=0.7, linestyle='--')

                # Plot difference (scaled for visibility)
                diff = signal - reconstructed
                max_diff = np.max(np.abs(diff))
                if max_diff > 1e-10:
                    scale = 100
                    ax.plot(time_axis, diff * scale, label=f'Difference × {scale}',
                           linewidth=1, alpha=0.5, color='red')

                ax.set_title(f'{pipeline_name} - RMS Error: {result["rms_error"]:.6e}',
                           fontweight='bold')
                ax.set_xlabel('Sample')
                ax.set_ylabel('Amplitude')
                ax.legend(loc='upper right')
                ax.grid(alpha=0.3)

            plt.tight_layout()
            out_path = os.path.join(self.output_dir,
                                   f'reconstruction_{system_type}.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved reconstruction example to {out_path}")


if __name__ == '__main__':
    # Run test with db4 wavelet and 100 trials
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'outputs',
        'chaotic_reconstruction_test'
    )
    test = ChaoticReconstructionTest(
        wavelet='db4',
        level=4,
        n_samples=1024,
        n_trials=100,
        output_dir=output_dir
    )
    test.run_test()