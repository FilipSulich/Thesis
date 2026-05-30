import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .signal_generator import SignalGenerator

# Test frequencies
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
    def __init__(self, fs=128, n_samples=1280, level=4, output_dir=None):
        """
        Args:
            fs: Sampling rate
            n_samples: Signal length
            level: Decomposition level
            output_dir: Where to save results
        """
        self.fs = fs
        self.level = level
        self.output_dir = Path(output_dir) if output_dir else Path('leakage_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sig_gen = SignalGenerator(fs, n_samples)

    def compute_confusion_matrix(self, wavelet, freq_map):
        """
        Build confusion matrix showing energy distribution.

        Rows = source band (where freq should go)
        Cols = target band (where energy actually went)
        Diagonal = good (energy in right place)
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

            # Decompose
            try:
                outputs = pipeline.pipeline_swt_recon(signal)
            except Exception:
                continue

            # Measure energy in each band
            total_energy = sum(
                np.sum(outputs[band] ** 2)
                for band in BAND_NAMES
                if band in outputs
            )

            # Fill confusion row
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
        Energy Ratio - how much energy stays in right band.
        Higher = better (max ~1.0 with 5 bands).
        """
        diagonal = np.diag(confusion)
        return float(np.mean(diagonal))

    def test_wavelet(self, wavelet, freq_map, condition_name, trial_num):
        """Run one test trial for a wavelet."""
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

        # Create results DataFrame
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
