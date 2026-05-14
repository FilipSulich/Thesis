"""
Spectral leakage test for 17 wavelets using energy-based confusion matrix.

Tests 17 wavelets on two sine conditions (harmonic and non-harmonic) using 4-level SWT-Recon.

For each wavelet, condition, and trial:
1. For each source band:
   a. Generate pure sine at source frequency
   b. Decompose using 4-level SWT reconstruction (swt_recon pipeline)
   c. Measure energy in each reconstructed subband
   d. Compute energy fraction: E_target / E_total
2. Build confusion matrix: energy[source_band, target_band]
3. Compute DER (Diagonal Energy Ratio) = mean(diagonal)
4. Run N trials and report mean, median, std per wavelet

Output: Box plots of DER values for all wavelets with both conditions.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipelines.pipeline_wavelets import Pipeline


class SpectralLeakageTestV2:
    """
    Spectral leakage test comparing 17 wavelets on 4-level SWT-Recon.
    
    Tests using two sine frequency conditions (harmonic and non-harmonic).
    For each wavelet and condition, runs multiple trials differing only by phase shift.
    Each trial computes DER via confusion matrix.
    Final analysis: Kruskal-Wallis test across all wavelets with Dunn post-hoc.
    """
    # Harmonic frequencies (all related by factors of 2)
    SUBBAND_FREQS_HARM = {
        'A4': 2,
        'D4': 6,    # 4-8 Hz
        'D3': 12,   # 8-16 Hz
        'D2': 24,   # 16-32 Hz
        'D1': 48    # 32-64 Hz
    }

    # Non-harmonic frequencies (not perfectly aligned with dyadic bands)
    SUBBAND_FREQS_NON_HARM = {
        'A4': 3,
        'D4': 7,    # 4-8 Hz
        'D3': 13,   # 8-16 Hz
        'D2': 31,   # 16-32 Hz
        'D1': 61    # 32-64 Hz
    }

    BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']

    # 17 wavelets: 7 Daubechies + 7 Symlets + 3 Coiflets
    WAVELETS = (
        [f'db{i}' for i in range(2, 9)] +     # db2-db8
        [f'sym{i}' for i in range(2, 9)] +    # sym2-sym8
        [f'coif{i}' for i in range(1, 4)]     # coif1-coif3
    )

    def __init__(
        self,
        fs: int = 128,
        n_samples: int = 1280,
        level: int = 4,
        output_dir: str = None,
    ):
        """
        Initialize leakage test.

        Args:
            fs: Sampling frequency
            n_samples: Number of samples
            level: Decomposition level (fixed to 4 for SWT-Recon)
            output_dir: Directory to save results
        """
        self.FS = fs
        self.N = n_samples
        self.LEVEL = level
        self.output_dir = Path(output_dir) if output_dir else Path('leakage_test_results_v2')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._results = []

    # ────────────────────────────────────────────────────────────────────

    def generate_composite_sine(self, freq_map, random_phase=False) -> np.ndarray:
        """Generate composite sine from all subband frequencies."""
        t = np.arange(self.N) / self.FS
        signal = np.zeros(self.N)
        for band, freq in freq_map.items():
            phase = np.random.uniform(0, 2 * np.pi) if random_phase else 0.0
            signal += np.sin(2 * np.pi * freq * t + phase)
        return signal.astype(np.float32)

    def generate_nonharmonic_sines(self) -> np.ndarray:
        """Generate five pure sines (all bands including A4)."""
        t = np.arange(self.N) / self.FS
        signal = np.zeros(self.N)
        for band in self.BAND_NAMES:
            freq = self.SUBBAND_FREQS_HARM[band]
            signal += np.sin(2 * np.pi * freq * t)
        return signal.astype(np.float32)

    # def generate_pure_sine(self, freq: float, seed: int = None) -> np.ndarray:
    #     """Generate single sine at given frequency with optional seed for randomness."""
    #     if seed is not None:
    #         np.random.seed(seed)
    #
    #     t = np.arange(self.N) / self.FS
    #     phase = np.random.uniform(0, 2 * np.pi)
    #     return np.sin(2 * np.pi * freq * t + phase).astype(np.float32)

    def generate_pure_sine(self, freq: float, jitter_amount: float = 0.0) -> np.ndarray:
        """
        Generate single sine with optional frequency jitter.

        Args:
            freq: Target frequency
            jitter_amount: Max relative deviation (e.g., 0.05 = 5% jitter)
        """
        # Apply jitter: freq * (1 +/- jitter)
        if jitter_amount > 0:
            freq = freq + np.random.uniform(-jitter_amount, jitter_amount) * freq

        t = np.arange(self.N) / self.FS
        # Phase still randomized to ensure generalizability
        phase = np.random.uniform(0, 2 * np.pi)
        return np.sin(2 * np.pi * freq * t + phase).astype(np.float32)

    def _compute_confusion_matrix(
        self,
        wavelet: str,
        freq_map: Dict,
        n_trials: int = 1,
        trial_seed: int = None
    ) -> np.ndarray:
        """
        Compute confusion matrix for a given wavelet and frequency map.

        Args:
            wavelet: Wavelet name
            freq_map: Dictionary mapping band names to frequencies
            n_trials: Number of trials to average over (only 1 needed since phase varies)

        Returns:
            5x5 confusion matrix
        """
        n_bands = len(self.BAND_NAMES)
        confusion_sum = np.zeros((n_bands, n_bands))

        pipeline = Pipeline(
            pipeline_name='swt_recon',
            level=self.LEVEL,
            wavelet=wavelet
        )

        # Average over multiple trials (each with different random phase)
        for trial in range(n_trials):
            # For each source band (row)
            for i, source_band in enumerate(self.BAND_NAMES):
                source_freq = freq_map[source_band]

                # Generate single sine at source frequency
                # seed_val = trial_seed + i if trial_seed is not None else None
                pure_sine = self.generate_pure_sine(source_freq, jitter_amount=0.05)

                # Decompose using pipeline_swt_recon
                try:
                    outputs = pipeline.pipeline_swt_recon(pure_sine)
                except Exception as e:
                    continue

                # Measure energy in EVERY output band
                total_energy = sum(
                    np.sum(outputs[band] ** 2) for band in self.BAND_NAMES
                    if band in outputs
                )

                # Compute energy fraction for each target band (column)
                for j, target_band in enumerate(self.BAND_NAMES):
                    if target_band in outputs:
                        band_energy = np.sum(outputs[target_band] ** 2)
                        energy_fraction = (
                            band_energy / total_energy
                            if total_energy > 1e-9 else 0.0
                        )
                        confusion_sum[i, j] += energy_fraction

        # Average over trials
        confusion = confusion_sum / n_trials

        return confusion

    def compute_der(self, confusion: np.ndarray) -> float:
        """
        Compute Diagonal Energy Ratio.

        Args:
            confusion: Confusion matrix

        Returns:
            DER score (higher is better, max ~1.0 with 5 bands)
        """
        diagonal = np.diag(confusion)
        return float(np.mean(diagonal))

    def test_wavelet(
        self,
        wavelet: str,
        freq_map: Dict,
        trial_num: int
    ) -> Dict:
        """
        Test a single wavelet on a frequency map (single trial with specific phase).

        Args:
            wavelet: Wavelet name
            freq_map: Frequency map (harmonic or non-harmonic)
            trial_num: Trial number (used only for tracking)

        Returns:
            Dict with DER and other metrics
        """
        confusion = self._compute_confusion_matrix(wavelet, freq_map, n_trials=1)
        der = self.compute_der(confusion)

        # Determine condition name
        condition = 'Harmonic' if freq_map == self.SUBBAND_FREQS_HARM else 'Non-harmonic'

        return {
            'wavelet': wavelet,
            'condition': condition,
            'trial': trial_num,
            'DER': der,
            'confusion_matrix': confusion.tolist(),
        }

    def run_all_wavelets(self, n_trials: int = 20) -> pd.DataFrame:
        """
        Run leakage test for all wavelets on both conditions.
        Each trial differs only by phase shift (like in decomposition analyzer).

        Args:
            n_trials: Number of independent trials (phase shifts)

        Returns:
            DataFrame with DER values per trial
        """
        print(f"\n{'='*80}")
        print(f"SPECTRAL LEAKAGE TEST — 4-LEVEL SWT-RECON (Wavelet Comparison)")
        print(f"{'='*80}")
        print(f"Wavelets: {len(self.WAVELETS)}")
        print(f"Conditions: harmonic, non-harmonic")
        print(f"Trials per wavelet-condition: {n_trials}")
        print(f"Bands: {len(self.BAND_NAMES)}")
        print(f"{'='*80}\n")

        all_results = []

        # Run multiple independent trials (differing by phase shift)
        for trial_num in range(n_trials):
            if (trial_num + 1) % 5 == 0:
                print(f"Trial {trial_num + 1}/{n_trials}")

            # Test each wavelet on both frequency maps
            for wavelet in tqdm(self.WAVELETS, desc=f"Trial {trial_num+1}", leave=False):
                # Test on harmonic frequencies
                result_harm = self.test_wavelet(wavelet, self.SUBBAND_FREQS_HARM, trial_num)
                all_results.append(result_harm)

                # Test on non-harmonic frequencies
                result_nonharm = self.test_wavelet(wavelet, self.SUBBAND_FREQS_NON_HARM, trial_num)
                all_results.append(result_nonharm)

        # Create DataFrame with all trials
        df = pd.DataFrame([
            {
                'wavelet': r['wavelet'],
                'condition': r['condition'],
                'trial': r['trial'],
                'DER': r['DER'],
            }
            for r in all_results
        ])

        # Save results
        results_path = self.output_dir / 'der_results.json'
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")

        # Save CSV
        csv_path = self.output_dir / 'der_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ CSV saved to {csv_path}")

        return df

    def plot_der_comparison(self, df: pd.DataFrame) -> None:
        """
        Refined publication-level DER comparison plot.
        """
        # 1. Style Setup
        sns.set_theme(style="whitegrid")
        sns.set_context("paper", font_scale=1.5)

        fig, ax = plt.subplots(figsize=(14, 7))

        # 2. Logical Sorting: Group by family, then order
        def wavelet_key(w):
            family = "".join([c for c in w if not c.isdigit()])
            order = int("".join([c for c in w if c.isdigit()]))
            family_map = {'db': 0, 'sym': 1, 'coif': 2}
            return (family_map.get(family, 3), order)

        sorted_wavelets = sorted(df['wavelet'].unique(), key=wavelet_key)

        # 3. Professional Color Map (Viridis-style contrast)
        # Deep Blue for Harmonic, Vibrant Green/Teal for Non-Harmonic
        palette = {"Harmonic": "#2c3e50", "Non-harmonic": "#16a085"}

        sns.boxplot(
            data=df,
            x='wavelet',
            y='DER',
            hue='condition',
            order=sorted_wavelets,
            palette=palette,
            width=0.7,
            linewidth=1.5,
            fliersize=4,
            ax=ax
        )

        # 4. Axis and Limits (Ensuring no cut-offs)
        # ax.set_ylim(0, 1.05)  # Shows the full scale from 0 to 1

        ax.set_xlabel('Wavelet Function', fontweight='bold')
        ax.set_ylabel('Energy Ratio', fontweight='bold')
        ax.set_title('Energy Leakage Analysis \n Harmonic vs. Non-harmonic Frequency Placement Conditions \n (Higher is better)', pad=20,
                     fontweight='bold')

        # 5. Final Formatting
        plt.xticks(rotation=45)

        # --- LEGEND REDUCTION HERE ---
        # Overriding the global font_scale=1.5 with absolute font sizes (11 and 12)
        # handlelength reduces the width of the color boxes inside the legend
        ax.legend(
            title='Condition',
            frameon=True,
            loc='lower right',
            fontsize=11,  # Shrinks the labels
            title_fontsize=12,  # Shrinks the title
            handlelength=1.2,  # Makes the color box shorter
            borderpad=0.5  # Reduces padding inside the legend box
        )

        # Despine for a cleaner "Modern Scientific" look
        sns.despine()

        plt.tight_layout()
        plot_path = self.output_dir / 'der_clean_publication.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

    def print_summary(self, df: pd.DataFrame) -> None:
        """
        Print summary statistics per wavelet and condition.
        Perform Kruskal-Wallis test with Dunn post-hoc and Bonferroni correction.
        """
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS BY WAVELET")
        print(f"{'='*80}\n")

        for condition in ['Harmonic', 'Non-harmonic']:
            df_cond = df[df['condition'] == condition]
            
            print(f"\n{condition.upper()} CONDITION:")
            print("-" * 110)
            print(f"{'Wavelet':<12} {'Mean':<12} {'Median':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'N Trials':<10}")
            print("-" * 110)
            
            summary_stats = []
            for wavelet in sorted(df_cond['wavelet'].unique()):
                ders = df_cond[df_cond['wavelet'] == wavelet]['DER'].values
                summary_stats.append({
                    'wavelet': wavelet,
                    'mean': float(np.mean(ders)),
                    'median': float(np.median(ders)),
                    'std': float(np.std(ders)),
                    'min': float(np.min(ders)),
                    'max': float(np.max(ders)),
                    'n_trials': len(ders),
                })
                print(f"{wavelet:<12} {np.mean(ders):<12.6f} {np.median(ders):<12.6f} "
                      f"{np.std(ders):<12.6f} {np.min(ders):<12.6f} {np.max(ders):<12.6f} {len(ders):<10}")
            
            summary_df = pd.DataFrame(summary_stats)
            summary_df = summary_df.sort_values('mean', ascending=False)
            
            # Save condition-specific statistics
            summary_path = self.output_dir / f'summary_statistics_{condition}.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"\n✓ Summary saved to {summary_path}")

            # Perform statistical tests
            print(f"\n{'='*80}")
            print(f"STATISTICAL ANALYSIS — {condition.upper()}")
            print(f"{'='*80}\n")

            self._run_statistical_tests(df_cond, condition)

    def _run_statistical_tests(self, df_cond: pd.DataFrame, condition: str) -> None:
        """
        Perform Kruskal-Wallis test across wavelets with Dunn post-hoc.
        
        Args:
            df_cond: DataFrame filtered to one condition
            condition: Condition name (for output filename)
        """
        # Prepare groups for Kruskal-Wallis
        wavelets = sorted(df_cond['wavelet'].unique())
        groups = [df_cond[df_cond['wavelet'] == w]['DER'].values for w in wavelets]
        
        # Kruskal-Wallis test
        h_stat, p_value_kw = kruskal(*groups)
        
        print(f"Kruskal-Wallis Test:")
        print(f"  H-statistic: {h_stat:.6f}")
        print(f"  p-value: {p_value_kw:.6e}")
        print(f"  Significance: {'***' if p_value_kw < 0.001 else '**' if p_value_kw < 0.01 else '*' if p_value_kw < 0.05 else 'ns'}")
        
        # Dunn post-hoc test with Bonferroni correction
        print(f"\nDunn Post-Hoc Test (Bonferroni correction):")
        print("-" * 90)
        print(f"{'Comparison':<25} {'p-value':<12} {'Significance':<15} {'Mean Diff':<12}")
        print("-" * 90)
        
        try:
            dunn_results = posthoc_dunn(df_cond, val_col='DER', group_col='wavelet', p_adjust='fdr_bh')
            
            # Save Dunn results
            dunn_path = self.output_dir / f'dunn_posthoc_{condition}.csv'
            dunn_results.to_csv(dunn_path)
            print(f"✓ Full Dunn post-hoc results saved to {dunn_path}\n")
            
            # Print ALL comparisons (not just significant ones)
            significant_count = 0
            for i, wav1 in enumerate(wavelets):
                for j, wav2 in enumerate(wavelets):
                    if i < j:
                        p_val = dunn_results.loc[wav1, wav2]
                        mean1 = df_cond[df_cond['wavelet'] == wav1]['DER'].mean()
                        mean2 = df_cond[df_cond['wavelet'] == wav2]['DER'].mean()
                        mean_diff = abs(mean1 - mean2)
                        
                        if p_val < 0.001:
                            sig = '***'
                            significant_count += 1
                        elif p_val < 0.01:
                            sig = '**'
                            significant_count += 1
                        elif p_val < 0.05:
                            sig = '*'
                            significant_count += 1
                        else:
                            sig = 'ns'
                        
                        print(f"{wav1:>5} vs {wav2:<5} {p_val:<12.6f} {sig:<15} {mean_diff:<12.6f}")
            
            print("-" * 90)
            print(f"Significant comparisons (p < 0.05): {significant_count} / {len(wavelets) * (len(wavelets)-1) // 2}")
        
        except Exception as e:
            print(f"  Warning: Dunn test failed: {e}")


if __name__ == '__main__':
    import json

    output_dir = Path('/src/wavelet_selection_tests/leakage_test_results_v2')

    tester = SpectralLeakageTestV2(
        fs=128,
        n_samples=1280,
        level=4,
        output_dir=output_dir
    )

    # Run test with 20 trials (differing by phase shift)
    df = tester.run_all_wavelets(n_trials=100)

    # Plot results
    print("\nGenerating plots...")
    tester.plot_der_comparison(df)

    # Print summary and perform statistical tests
    tester.print_summary(df)

