import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
import csv
from scipy.spatial.distance import pdist

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavelets import stationary_wavelet_transform as stationary_wavelet
from wavelets import discrete_wavelet_transform as discrete_wavelet
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RPComputation


DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/mitbih')
OUTPUT_DIR = DATA_DIR / 'results'

SAMPLING_FREQ = 250
SUBBANDS = ['D1', 'D2', 'D3', 'D4', 'A4']


class MITBIHVisualizer:
    def __init__(
        self,
        lead_time: int = 5,
        epsilon_method: str = 'std',
        epsilon_factor: float = 0.2,
        target_rr: float = 0.05,
        tau: int = 21,
        m: int = 7,
        wavelet: str = 'db4',
        level: int = 4,
        target_density: float = 0.01,
        subband_params: Optional[Dict[str, Dict]] = None,
    ):

        self.lead_time = lead_time
        self.epsilon_method = epsilon_method
        self.epsilon_factor = epsilon_factor
        self.target_rr = target_rr
        self.tau = tau
        self.m = m
        self.wavelet = wavelet
        self.level = level
        self.target_density = target_density

        # Subband-specific parameters
        self.subband_params = subband_params or {}

        # Load data
        csv_path = OUTPUT_DIR / f'dataset_lt{lead_time}.csv'
        self.data = self._load_csv(csv_path)

        print(f"Loaded {len(self.data)} signals from lead time {lead_time} min")
        print(f"  Pre-AF: {sum(1 for d in self.data if d['label'] == 1)}")
        print(f"  Normal: {sum(1 for d in self.data if d['label'] == 0)}")
        if self.subband_params:
            print(f"  Using custom subband parameters")
        print(f"  Epsilon method: {epsilon_method}")

    def _load_csv(self, csv_path: Path):
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                signal_cols = [k for k in row.keys() if k.startswith('s') and k[1:].isdigit()]
                signal = np.array([float(row[col]) for col in sorted(signal_cols, key=lambda x: int(x[1:]))])

                data.append({
                    'signal': signal,
                    'label': int(row['label']),
                    'patient_id': row['patient_id'],
                    'segment_id': row['segment_id']
                })
        return data

    def _decompose_subbands(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        swt = stationary_wavelet.SWT(wavelet=self.wavelet, level=self.level)
        coeffs = swt.decompose(signal, level=self.level)

        subbands = {}

        for i in range(len(coeffs)):
            zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
            subband = swt.reconstruct(zeroed)
            subband = (subband - np.mean(subband)) / (np.std(subband) + 1e-8)
            subbands[f"D{i+1}"] = subband

        zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        subband = swt.reconstruct(zeroed)
        subband = (subband - np.mean(subband)) / (np.std(subband) + 1e-8)
        subbands["A4"] = subband

        return subbands

    from scipy.spatial.distance import pdist

    def epsilon_from_percentile(embedded, percentile=10):
        distances = pdist(embedded, metric='euclidean')
        return np.percentile(distances, percentile)

    def _compute_epsilon(self, signal: np.ndarray, tau: int, m: int) -> float:
        if self.epsilon_method == 'std':
            return self.epsilon_factor * np.std(signal)

        elif self.epsilon_method == 'max_distance':
            N = len(signal) - (m - 1) * tau
            if N <= 1:
                return self.epsilon_factor * np.std(signal)
            embedded = np.array([signal[i:i + m * tau:tau] for i in range(N)])
            distances = pdist(embedded, metric='euclidean')
            return self.epsilon_factor * np.max(distances)

        elif self.epsilon_method == 'target_rr':
            return self._find_epsilon_for_target_rr(signal, tau, m)
        elif self.epsilon_method == 'density':
            return self.epsilon_from_recurrence_density(signal, tau, m, target_density=self.target_density)
        else:
            return self.epsilon_factor * np.std(signal)


    def embed_signal(self, signal, m, tau):
        N = len(signal)
        M = N - (m - 1) * tau
        return np.array([signal[i:i + (m - 1) * tau + 1:tau] for i in range(M)])

    def epsilon_from_recurrence_density(self, signal, m, tau, target_density=0.01):
        embedded = self.embed_signal(signal, m, tau)
        distances = pdist(embedded, metric='euclidean')

        # Step 3: Find the percentile of the distance distribution
        # that corresponds to the target recurrence density
        # Since the recurrence matrix is symmetric with ones on diagonal,
        # the off-diagonal recurrence rate equals the fraction of distances below epsilon.
        # pdist gives us the upper triangle, which is an unbiased sample of all off-diagonal distances.
        epsilon = np.percentile(distances, target_density * 100)

        return epsilon
    def _find_epsilon_for_target_rr(
        self,
        signal: np.ndarray,
        tau: int,
        m: int,
        max_iterations: int = 10
    ) -> float:
        """Binary search for epsilon that achieves target recurrence rate."""
        eps_low = 0.01 * np.std(signal)
        eps_high = 2.0 * np.std(signal)

        for _ in range(max_iterations):
            eps_mid = (eps_low + eps_high) / 2

            # Compute RR with this epsilon
            rp = self._compute_rp_with_epsilon(signal, tau, m, eps_mid)
            current_rr = np.sum(rp) / (rp.shape[0] * rp.shape[1])

            if abs(current_rr - self.target_rr) < 0.005:
                return eps_mid

            if current_rr < self.target_rr:
                eps_high = eps_mid
            else:
                eps_low = eps_mid

        return (eps_low + eps_high) / 2

    def _compute_rp_with_epsilon(
        self,
        signal: np.ndarray,
        tau: int,
        m: int,
        epsilon: float
    ) -> np.ndarray:
        """Compute RP with specific epsilon."""
        ts = TimeSeries(signal, embedding_dimension=m, time_delay=tau)
        settings = Settings(
            time_series=ts,
            analysis_type=Classic,
            neighbourhood=FixedRadius(epsilon),
            similarity_measure=EuclideanMetric,
            theiler_corrector=1
        )
        rp_computation = RPComputation.create(settings, verbose=False)
        result = rp_computation.run()
        return result.recurrence_matrix

    def _compute_rp(self, signal: np.ndarray, tau: int = None, m: int = None, subband_name: str = None) -> np.ndarray:
        if subband_name and subband_name in self.subband_params:
            if tau is None:
                tau = self.subband_params[subband_name].get('tau', self.tau)
            if m is None:
                m = self.subband_params[subband_name].get('m', self.m)
        else:
            if tau is None:
                tau = self.tau
            if m is None:
                m = self.m

        epsilon = self._compute_epsilon(signal, tau, m)
        return self._compute_rp_with_epsilon(signal, tau, m, epsilon)

    def plot_patient_comparison(self, save_path: Optional[Path] = None):
        """
        Plot pre-AF vs normal from same patient with single-scale and multi-scale RPs.

        Layout:
        - Row 0: Pre-AF signal + single-scale RP | Normal signal + single-scale RP
        - Rows 1-5: Multi-scale subbands with Pre-AF RP | Normal RP
        """
        # Find patients with both pre-AF and normal
        patients_with_both = {}
        for d in self.data:
            pid = d['patient_id']
            if pid not in patients_with_both:
                patients_with_both[pid] = {'preaf': [], 'normal': []}

            if d['label'] == 1:
                patients_with_both[pid]['preaf'].append(d)
            else:
                patients_with_both[pid]['normal'].append(d)

        # Filter to only patients with both
        valid_patients = {k: v for k, v in patients_with_both.items()
                         if len(v['preaf']) > 0 and len(v['normal']) > 0}

        if not valid_patients:
            print("No patients found with both pre-AF and normal signals!")
            return

        # Pick random patient
        patient_id = np.random.choice(list(valid_patients.keys()))
        preaf_data = np.random.choice(valid_patients[patient_id]['preaf'])
        normal_data = np.random.choice(valid_patients[patient_id]['normal'])

        preaf_signal = preaf_data['signal']
        normal_signal = normal_data['signal']

        print(f"\nPatient: {patient_id}")
        print(f"  Pre-AF: {preaf_data['segment_id']}")
        print(f"  Normal: {normal_data['segment_id']}")

        # Decompose both signals
        preaf_subbands = self._decompose_subbands(preaf_signal)
        normal_subbands = self._decompose_subbands(normal_signal)

        # Create figure: 6 rows, 4 columns
        fig = plt.figure(figsize=(16, 18))
        gs = fig.add_gridspec(6, 4, hspace=0.4, wspace=0.3)

        # =====================================================================
        # Row 0: Single-scale signals and RPs
        # =====================================================================

        # Pre-AF signal
        ax_preaf_sig = fig.add_subplot(gs[0, 0])
        t = np.arange(len(preaf_signal)) / SAMPLING_FREQ
        ax_preaf_sig.plot(t, preaf_signal, color='red', linewidth=0.7)
        ax_preaf_sig.set_ylabel('Pre-AF\nSignal')
        ax_preaf_sig.set_xlabel('Time (s)')
        ax_preaf_sig.set_title(f'Patient: {patient_id}\nPre-AF Signal',
                              fontsize=10, color='red', fontweight='bold')
        ax_preaf_sig.grid(True, alpha=0.3)

        # Pre-AF RP
        ax_preaf_rp = fig.add_subplot(gs[0, 1])
        rp_preaf = self._compute_rp(preaf_signal, subband_name='original')
        rr_preaf = np.sum(rp_preaf) / (rp_preaf.shape[0] * rp_preaf.shape[1])
        ax_preaf_rp.imshow(rp_preaf, cmap='binary', origin='lower', aspect='equal')

        # Get params for title
        tau_orig = self.subband_params.get('original', {}).get('tau', self.tau)
        m_orig = self.subband_params.get('original', {}).get('m', self.m)
        ax_preaf_rp.set_title(f'Pre-AF RP\nτ={tau_orig}, m={m_orig}\nRR={rr_preaf:.3f}', fontsize=9)
        ax_preaf_rp.set_xlabel('i')
        ax_preaf_rp.set_ylabel('j')

        # Normal signal
        ax_normal_sig = fig.add_subplot(gs[0, 2])
        ax_normal_sig.plot(t, normal_signal, color='blue', linewidth=0.7)
        ax_normal_sig.set_ylabel('Normal\nSignal')
        ax_normal_sig.set_xlabel('Time (s)')
        ax_normal_sig.set_title(f'Normal Signal',
                               fontsize=10, color='blue', fontweight='bold')
        ax_normal_sig.grid(True, alpha=0.3)

        # Normal RP
        ax_normal_rp = fig.add_subplot(gs[0, 3])
        rp_normal = self._compute_rp(normal_signal, subband_name='original')
        rr_normal = np.sum(rp_normal) / (rp_normal.shape[0] * rp_normal.shape[1])
        ax_normal_rp.imshow(rp_normal, cmap='binary', origin='lower', aspect='equal')
        ax_normal_rp.set_title(f'Normal RP\nτ={tau_orig}, m={m_orig}\nRR={rr_normal:.3f}', fontsize=9)
        ax_normal_rp.set_xlabel('i')
        ax_normal_rp.set_ylabel('j')

        # =====================================================================
        # Rows 1-5: Multi-scale decomposition
        # =====================================================================

        for i, subband_name in enumerate(SUBBANDS):
            row_idx = i + 1

            # Pre-AF subband signal
            ax_preaf_sub_sig = fig.add_subplot(gs[row_idx, 0])
            preaf_sub = preaf_subbands[subband_name]
            t_sub = np.arange(len(preaf_sub)) / SAMPLING_FREQ
            ax_preaf_sub_sig.plot(t_sub, preaf_sub, color='red', linewidth=0.7)
            ax_preaf_sub_sig.set_ylabel(f'{subband_name}\nPre-AF')
            ax_preaf_sub_sig.grid(True, alpha=0.3)
            if i == len(SUBBANDS) - 1:
                ax_preaf_sub_sig.set_xlabel('Time (s)')

            # Pre-AF subband RP
            ax_preaf_sub_rp = fig.add_subplot(gs[row_idx, 1])
            rp_preaf_sub = self._compute_rp(preaf_sub, subband_name=subband_name)
            rr_preaf_sub = np.sum(rp_preaf_sub) / (rp_preaf_sub.shape[0] * rp_preaf_sub.shape[1])
            ax_preaf_sub_rp.imshow(rp_preaf_sub, cmap='binary', origin='lower', aspect='equal')

            # Get params for title
            tau_sub = self.subband_params.get(subband_name, {}).get('tau', self.tau)
            m_sub = self.subband_params.get(subband_name, {}).get('m', self.m)
            ax_preaf_sub_rp.set_title(f'{subband_name} Pre-AF RP\nτ={tau_sub}, m={m_sub}\nRR={rr_preaf_sub:.3f}', fontsize=8)
            ax_preaf_sub_rp.set_xlabel('i')
            ax_preaf_sub_rp.set_ylabel('j')

            # Normal subband signal
            ax_normal_sub_sig = fig.add_subplot(gs[row_idx, 2])
            normal_sub = normal_subbands[subband_name]
            ax_normal_sub_sig.plot(t_sub, normal_sub, color='blue', linewidth=0.7)
            ax_normal_sub_sig.set_ylabel(f'{subband_name}\nNormal')
            ax_normal_sub_sig.grid(True, alpha=0.3)
            if i == len(SUBBANDS) - 1:
                ax_normal_sub_sig.set_xlabel('Time (s)')

            # Normal subband RP
            ax_normal_sub_rp = fig.add_subplot(gs[row_idx, 3])
            rp_normal_sub = self._compute_rp(normal_sub, subband_name=subband_name)
            rr_normal_sub = np.sum(rp_normal_sub) / (rp_normal_sub.shape[0] * rp_normal_sub.shape[1])
            ax_normal_sub_rp.imshow(rp_normal_sub, cmap='binary', origin='lower', aspect='equal')
            ax_normal_sub_rp.set_title(f'{subband_name} Normal RP\nτ={tau_sub}, m={m_sub}\nRR={rr_normal_sub:.3f}', fontsize=8)
            ax_normal_sub_rp.set_xlabel('i')
            ax_normal_sub_rp.set_ylabel('j')

        plt.suptitle(f'Pre-AF vs Normal - Patient {patient_id} (Lead Time: {self.lead_time} min, Epsilon: {self.epsilon_method})',
                     fontsize=14, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved to {save_path}")
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    LEAD_TIME = 5
    EPSILON_METHOD = 'density'

    subband_params = {
        'original': {'tau': 21, 'm': 7},
        'D1': {'tau': 6, 'm': 4},
        'D2': {'tau': 3, 'm': 6},
        'D3': {'tau': 2, 'm': 5},
        'D4': {'tau': 1, 'm': 7},
        'A4': {'tau': 17, 'm': 9},
    }

    viz = MITBIHVisualizer(
        lead_time=LEAD_TIME,
        epsilon_method=EPSILON_METHOD,
        epsilon_factor=0.2,
        target_rr=0.1,
        tau=21,
        m=7,
        wavelet='db4',
        level=4,
        target_density=0.1,
        subband_params=subband_params
    )

    output_dir = OUTPUT_DIR / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        print(f"\n{'='*80}")
        print(f"Generating comparison {i+1}/3...")
        print(f"{'='*80}")

        viz.plot_patient_comparison(
            save_path=output_dir / f'patient_comparison_{i+1}_lt{LEAD_TIME}_{EPSILON_METHOD}.png'
        )

    print(f"\n{'='*80}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*80}\n")
