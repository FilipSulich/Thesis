import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.pipeline_wavelets import Pipeline


class LeakageTest:
    """
    Test for subband energy leakage in SWT reconstruction across different wavelets.

    Measures:
    1. Subband energy confusion matrix (5x5): where energy ends up vs where it should
    2. Diagonal Energy Ratio (DER): mean of diagonal entries (correct routing)
    3. Mean Off-Diagonal Leakage (MDL): mean of off-diagonal entries (contamination)
    4. Reconstruction error: RMS difference between reconstructed and true sine
    """

    BAND_MAP = {
        'D1': {'f0': 48, 'f_low': 32, 'f_high': 64},
        'D2': {'f0': 24, 'f_low': 16, 'f_high': 32},
        'D3': {'f0': 12, 'f_low':  8, 'f_high': 16},
        'D4': {'f0':  6, 'f_low':  4, 'f_high':  8},
        'A4': {'f0':  2, 'f_low':  0, 'f_high': 4},
    }
    BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']

    def __init__(
        self,
        wavelets: list[str] = None,
        fs: int = 128,
        n_samples: int = 1280,
        level: int = 4,
        output_dir: str = '../../outputs/leakage_tests'
    ):
        """
        Initialize leakage test.

        Args:
            wavelets: List of wavelet names to test (e.g., ['db4', 'sym4', 'coif2'])
            fs: Sampling frequency
            n_samples: Number of samples
            level: Decomposition level
            output_dir: Directory to save outputs
        """
        self.wavelets = wavelets or ['db4', 'sym4', 'coif2', 'bior3.5']
        self.FS = fs
        self.N = n_samples
        self.level = level
        self.OUTPUT_DIR = output_dir
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        self._results = []

    def _generate_single_sine(self, f0: float) -> np.ndarray:
        """Generate a single frequency sine wave at f0 Hz."""
        t = np.arange(self.N) / self.FS
        sig = np.sin(2 * np.pi * f0 * t)
        # Normalize
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-9)
        return sig

    def _compute_confusion_matrix(self, wavelet: str, n_trials: int = 20) -> np.ndarray:
        """
        Compute 5x5 confusion matrix for a given wavelet, averaged over multiple trials.

        Matrix[i, j] = fraction of energy from band i that appears in band j
        Diagonal should be ~1.0, off-diagonal should be ~0.0

        Args:
            wavelet: Wavelet name
            n_trials: Number of trials with random phases to average over

        Returns:
            5x5 numpy array (confusion matrix)
        """
        n_bands = len(self.BAND_NAMES)
        confusion_sum = np.zeros((n_bands, n_bands))

        pipeline = Pipeline(pipeline_name='swt_recon', level=self.level, wavelet=wavelet)

        # Average over multiple trials with random phases
        for trial in range(n_trials):
            # For each source band (row)
            for i, source_band in enumerate(self.BAND_NAMES):
                source_f0 = self.BAND_MAP[source_band]['f0']

                # Generate single sine at source band center frequency with random phase
                t = np.arange(self.N) / self.FS
                phase = np.random.uniform(0, 2 * np.pi)
                pure_sine = np.sin(2 * np.pi * source_f0 * t + phase)

                # Decompose with SWT reconstruction
                outputs = pipeline.pipeline_swt_recon(pure_sine)

                # Measure energy in EVERY output band (sum of squared samples)
                total_output_energy = sum(np.sum(outputs[b] ** 2) for b in self.BAND_NAMES)

                # Measure energy fraction in each target band (column)
                for j, output_band in enumerate(self.BAND_NAMES):
                    band_energy = np.sum(outputs[output_band] ** 2)
                    energy_fraction = band_energy / total_output_energy if total_output_energy > 0 else 0
                    confusion_sum[i, j] += energy_fraction

        # Average over trials
        confusion = confusion_sum / n_trials

        return confusion

    def _compute_reconstruction_error(self, wavelet: str) -> dict:
        """
        Compute reconstruction error for each band.

        Measures RMS difference between:
        - True isolated sine at band center frequency
        - Reconstructed subband from SWT when fed the composite signal

        Args:
            wavelet: Wavelet name

        Returns:
            Dictionary mapping band -> RMS error
        """
        pipeline = Pipeline(pipeline_name='swt_recon', level=self.level, wavelet=wavelet)

        # Generate composite signal (sum of all sines)
        t = np.arange(self.N) / self.FS
        composite = sum(
            np.sin(2 * np.pi * self.BAND_MAP[band]['f0'] * t)
            for band in self.BAND_NAMES
        )

        # Decompose composite
        subbands = pipeline.pipeline_swt_recon(composite)

        errors = {}
        for band in self.BAND_NAMES:
            # True isolated sine
            f0 = self.BAND_MAP[band]['f0']
            true_sine = np.sin(2 * np.pi * f0 * t)

            # Reconstructed subband
            reconstructed = subbands[band]

            # RMS error
            rms_error = np.sqrt(np.mean((reconstructed - true_sine) ** 2))
            errors[band] = rms_error

        return errors



    def run_test(self):
        """Run leakage test for all wavelets."""
        print(f"Running leakage test for {len(self.wavelets)} wavelets...")

        for wavelet in self.wavelets:
            print(f"\nTesting wavelet: {wavelet}")

            # Compute confusion matrix
            confusion = self._compute_confusion_matrix(wavelet)

            # Compute summary metrics
            diagonal = np.diag(confusion)
            der = np.mean(diagonal)  # Diagonal Energy Ratio

            # Off-diagonal leakage
            mask = ~np.eye(len(self.BAND_NAMES), dtype=bool)
            mdl = np.mean(confusion[mask])  # Mean off-Diagonal Leakage

            # Reconstruction errors
            recon_errors = self._compute_reconstruction_error(wavelet)
            mean_recon_error = np.mean(list(recon_errors.values()))

            # Store results
            result = {
                'wavelet': wavelet,
                'der': der,
                'mdl': mdl,
                'mean_recon_error': mean_recon_error,
                'confusion_matrix': confusion,
            }

            # Add per-band metrics
            for i, band in enumerate(self.BAND_NAMES):
                result[f'{band}_correct'] = diagonal[i]
                result[f'{band}_recon_error'] = recon_errors[band]

            self._results.append(result)

            print(f"  DER: {der:.4f}")
            print(f"  MDL: {mdl:.4f}")
            print(f"  Mean reconstruction error: {mean_recon_error:.4f}")

        # Save results
        self._save_results()
        self._plot_results()

        print(f"\nResults saved to {self.OUTPUT_DIR}")

    def _save_results(self):
        """Save results to CSV and numpy files."""
        # Summary table
        summary_rows = []
        for result in self._results:
            row = {
                'wavelet': result['wavelet'],
                'DER': result['der'],
                'MDL': result['mdl'],
                'mean_recon_error': result['mean_recon_error'],
            }
            # Add per-band correct routing
            for band in self.BAND_NAMES:
                row[f'{band}_correct'] = result[f'{band}_correct']
                row[f'{band}_recon_error'] = result[f'{band}_recon_error']
            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        csv_path = os.path.join(self.OUTPUT_DIR, 'leakage_summary.csv')
        df_summary.to_csv(csv_path, index=False)
        print(f"Saved summary to {csv_path}")

        # Save confusion matrices
        for result in self._results:
            wavelet = result['wavelet']
            confusion = result['confusion_matrix']
            np_path = os.path.join(self.OUTPUT_DIR, f'confusion_{wavelet}.npy')
            np.save(np_path, confusion)

    def _plot_results(self):
        """Generate plots for leakage test results."""
        self._plot_confusion_matrices()
        self._plot_summary_comparison()
        self._plot_reconstruction_errors()
        self._plot_average_reconstruction_error()

    def _plot_confusion_matrices(self):
        """Plot confusion matrices as heatmaps."""
        n_wavelets = len(self.wavelets)
        cols = min(4, n_wavelets)
        rows = (n_wavelets + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
        if n_wavelets == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for idx, result in enumerate(self._results):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            confusion = result['confusion_matrix']
            wavelet = result['wavelet']

            sns.heatmap(
                confusion,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Energy fraction'},
                xticklabels=self.BAND_NAMES,
                yticklabels=self.BAND_NAMES,
                ax=ax
            )

            ax.set_title(
                f"{wavelet}\nDER={result['der']:.3f}",
                fontweight='bold'
            )
            ax.set_xlabel('Target subband')
            ax.set_ylabel('Source subband')


        # Hide unused subplots
        for idx in range(n_wavelets, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')


        plt.tight_layout()
        out_path = os.path.join(self.OUTPUT_DIR, 'confusion_matrices.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrices to {out_path}")

    def _plot_summary_comparison(self):
        """Plot bar chart comparing DER and MDL across wavelets."""
        wavelets = [r['wavelet'] for r in self._results]
        ders = [r['der'] for r in self._results]
        mdls = [r['mdl'] for r in self._results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # DER plot
        ax1.bar(wavelets, ders, color='steelblue', alpha=0.8)
        ax1.set_ylabel('Diagonal Energy Ratio (DER)', fontweight='bold')
        ax1.set_xlabel('Wavelet')
        ax1.set_title('Mean Correct Routing\n(higher is better)', fontweight='bold')
        ax1.set_ylim([0, 1.0])
        ax1.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Perfect (1.0)')
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend()

        # MDL plot
        ax2.bar(wavelets, mdls, color='tomato', alpha=0.8)
        ax2.set_ylabel('Mean off-Diagonal Leakage (MDL)', fontweight='bold')
        ax2.set_xlabel('Wavelet')
        ax2.set_title('Mean Contamination\n(lower is better)', fontweight='bold')
        ax2.axhline(0.0, color='green', linestyle='--', alpha=0.5, label='Perfect (0.0)')
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        out_path = os.path.join(self.OUTPUT_DIR, 'summary_comparison.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved summary comparison to {out_path}")

    def _plot_reconstruction_errors(self):
        """Plot reconstruction errors per band and wavelet."""
        # Prepare data_old
        data_rows = []
        for result in self._results:
            wavelet = result['wavelet']
            for band in self.BAND_NAMES:
                data_rows.append({
                    'wavelet': wavelet,
                    'band': band,
                    'recon_error': result[f'{band}_recon_error']
                })

        df = pd.DataFrame(data_rows)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(
            data=df,
            x='band',
            y='recon_error',
            hue='wavelet',
            ax=ax,
            palette='colorblind'
        )

        ax.set_ylabel('RMS Reconstruction Error', fontweight='bold')
        ax.set_xlabel('Subband', fontweight='bold')
        ax.set_title(
            'Reconstruction Error per Subband\n'
            '(RMS difference between true sine and SWT reconstructed subband)',
            fontweight='bold'
        )
        ax.legend(title='Wavelet', frameon=True)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(self.OUTPUT_DIR, 'reconstruction_errors.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved reconstruction errors to {out_path}")

    def _plot_average_reconstruction_error(self):
        """
        Plot the average reconstruction error across all subbands per wavelet type.
        Provides a single aggregated performance metric for each wavelet.
        """
        # Extract wavelet names and their corresponding mean reconstruction errors
        wavelets = [r['wavelet'] for r in self._results]
        avg_errors = [r['mean_recon_error'] for r in self._results]

        # Create the figure
        plt.figure(figsize=(10, 6))

        # Plotting with seaborn
        sns.barplot(
            x=wavelets,
            y=avg_errors,
            palette='viridis',
            hue=wavelets,
            legend=False
        )

        # Formatting to match your existing style
        plt.ylabel('Mean RMS Reconstruction Error', fontweight='bold')
        plt.xlabel('Wavelet Type', fontweight='bold')
        plt.title(
            'Global Average Reconstruction Error\n'
            '(Average of all subbands per wavelet)',
            fontweight='bold'
        )

        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3, linestyle='--')

        # Draw a horizontal line at 0 for reference
        plt.axhline(0, color='black', linewidth=0.8)

        plt.tight_layout()

        # Save output
        out_path = os.path.join(self.OUTPUT_DIR, 'average_reconstruction_summary.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved average reconstruction summary to {out_path}")


if __name__ == '__main__':
    # Test multiple wavelets
    wavelets_to_test = [
        'db2', 'db4', 'db6', 'db8',
        'sym2', 'sym4', 'sym6', 'sym8',
        'coif1', 'coif2', 'coif4','coif8'
    ]

    test = LeakageTest(
        wavelets=wavelets_to_test,
        fs=128,
        n_samples=1280,
        level=4
    )
    test.run_test()