import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .signal_generator import SignalGenerator

HARM_FREQS = {
    'A4': 2,
    'D4': 6,
    'D3': 12,
    'D2': 24,
    'D1': 48
}

NON_HARM_FREQS = {
    'A4': 3,
    'D4': 7,
    'D3': 13,
    'D2': 31,
    'D1': 61
}

BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']

class LeakageTester:
    """tests wavelets for frequency leakage across subbands using synthetic signals"""

    def __init__(self, fs=128, n_samples=1280, level=4, output_dir=None):
        """
        initializes leakage tester with signal and decomposition parameters

        Args:
            fs (int): sampling frequency in Hz
            n_samples (int): number of samples per test signal
            level (int): decomposition level for wavelet transform
            output_dir (str): directory to save results (JSON + CSV)
        """
        self.fs = fs
        self.level = level
        self.output_dir = Path(output_dir) if output_dir else Path('leakage_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sig_gen = SignalGenerator(fs, n_samples)

    def compute_confusion_matrix(self, wavelet, freq_map):
        """
        builds confusion matrix showing how energy leaks across frequency subbands

        Args:
            wavelet (str): wavelet function name (e.g., 'db8')
            freq_map (dict): mapping of band names to test frequencies

        Returns:
            np.array: confusion matrix where rows=source band, cols=target band, diagonal=correct isolation
        """
        from src.pipelines.pipeline_wavelets import Pipeline

        n_bands = len(BAND_NAMES)
        confusion = np.zeros((n_bands, n_bands))

        pipeline = Pipeline(
            pipeline_name='swt_recon',
            level=self.level,
            wavelet=wavelet
        )

        for i, source_band in enumerate(BAND_NAMES):
            source_freq = freq_map[source_band]

            # Generate pure sine
            signal = self.sig_gen.pure_sine(source_freq, jitter=0.05)

            outputs = pipeline.pipeline_swt_recon(signal)

            # Measure energy in each band
            total_energy = sum(
                np.sum(outputs[band] ** 2)
                for band in BAND_NAMES
                if band in outputs
            )

            for j, target_band in enumerate(BAND_NAMES):
                if target_band in outputs:
                    band_energy = np.sum(outputs[target_band] ** 2)
                    confusion[i, j] = (
                        band_energy / total_energy
                        if total_energy > 1e-9 else 0.0
                    )

        return confusion

    def compute_energy_ratio(self, confusion):
        """
        computes diagonal energy ratio showing frequency isolation quality

        Args:
            confusion (np.array): confusion matrix from compute_confusion_matrix

        Returns:
            float: mean diagonal energy ratio (higher = better isolation)
        """
        diagonal = np.diag(confusion)
        return float(np.mean(diagonal))

    def test_wavelet(self, wavelet, freq_map, condition_name, trial_num):
        """
        runs single test trial for one wavelet with specified frequency mapping

        Args:
            wavelet (str): wavelet function name
            freq_map (dict): mapping of band names to test frequencies
            condition_name (str): condition identifier (e.g., 'Harmonic' or 'Non-harmonic')
            trial_num (int): trial number for tracking

        Returns:
            dict: test results containing wavelet, condition, trial, DER, and confusion matrix
        """

        confusion = self.compute_confusion_matrix(wavelet, freq_map)
        der = self.compute_energy_ratio(confusion)

        return {
            'wavelet': wavelet,
            'condition': condition_name,
            'trial': trial_num,
            'DER': der,
            'confusion_matrix': confusion.tolist(),
        }

    def run_all(self, wavelets, n_trials=20):
        """
        runs complete leakage analysis across all wavelets and conditions

        Args:
            wavelets (list): list of wavelet function names to test
            n_trials (int): number of trials per wavelet-condition pair

        Returns:
            pd.DataFrame: results dataframe with columns [wavelet, condition, trial, DER]
        """
        all_results = []

        for trial_num in range(n_trials):
            if (trial_num + 1) % 5 == 0:
                print(f"Trial {trial_num + 1}/{n_trials}")

            for wavelet in tqdm(wavelets, desc=f"Trial {trial_num+1}", leave=False):
                # Test harmonic
                result_harm = self.test_wavelet(
                    wavelet, HARM_FREQS, 'Harmonic', trial_num
                )
                all_results.append(result_harm)

                # Test non-harmonic
                result_nonharm = self.test_wavelet(
                    wavelet, NON_HARM_FREQS, 'Non-harmonic', trial_num
                )
                all_results.append(result_nonharm)

        df = pd.DataFrame([
            {
                'wavelet': r['wavelet'],
                'condition': r['condition'],
                'trial': r['trial'],
                'DER': r['DER'],
            }
            for r in all_results
        ])

        json_path = self.output_dir / 'der_results.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        csv_path = self.output_dir / 'der_results.csv'
        df.to_csv(csv_path, index=False)
        return df
