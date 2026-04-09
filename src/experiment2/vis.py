import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import h5py
import matplotlib.pyplot as plt
import sys
import os

from scipy.fftpack import hilbert
from scipy.signal import find_peaks

from src.recurrence_plot.AMI import AMI
from src.recurrence_plot.FNN import FNN
from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.wavelets.stationary_wavelet_transform as stationary_wavelet
from src.data_cleaning.data_cleaner import DataCleaner

DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data_new/exp2')
HDF5_PATH = DATA_DIR / 'paf_with_quality.h5'

SAMPLING_FREQ = 128
SUBBAND_NAMES = ["D1", "D2", "D3", "D4", "A4"]
WAVELET = 'coif2'


class PAFVisualizer:
    def __init__(
            self,
            hdf5_path: Optional[Path] = None,
            random_seed: int = 42,
    ):
        self.hdf5_path = hdf5_path or HDF5_PATH
        self.rng = np.random.default_rng(random_seed)
        self.cleaner = DataCleaner(fs=SAMPLING_FREQ)

        self.signals = None
        self.labels = None
        self.segment_ids = None
        self.patient_ids = None
        self._load_data()

    def _load_data(self):
        with h5py.File(self.hdf5_path, 'r') as f:
            grp = f['all']
            self.signals = grp['signals'][:]
            self.labels = grp['labels'][:]
            self.segment_ids = [s.decode() if isinstance(s, bytes) else s for s in grp['segment_ids'][:]]
            self.patient_ids = (
                [s.decode() if isinstance(s, bytes) else s for s in grp['patient_ids'][:]]
                if 'patient_ids' in grp else [''] * len(self.signals)
            )
        print(f"Loaded {len(self.signals)} signals from {self.hdf5_path}")
        print(f"  Normal: {sum(self.labels == 0)}, Pre-AF: {sum(self.labels == 1)}")

    def get_signal(self, idx: int) -> Tuple[np.ndarray, int, str, str]:
        return self.signals[idx], self.labels[idx], self.segment_ids[idx], self.patient_ids[idx]

    def get_random_indices(self, n_normal: int = 3, n_preaf: int = 3) -> Dict[str, List[int]]:
        normal_idx = np.where(self.labels == 0)[0]
        normal_idx_filtered = [idx for idx in normal_idx if '_n' in self.patient_ids[idx].lower()]

        preaf_idx = np.where(self.labels == 1)[0]
        return {
            'normal': self.rng.choice(normal_idx_filtered, min(n_normal, len(normal_idx_filtered)),
                                      replace=False).tolist(),
            'preaf': self.rng.choice(preaf_idx, min(n_preaf, len(preaf_idx)), replace=False).tolist(),
        }

    def decompose_signal(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        swt = stationary_wavelet.SWT(wavelet=WAVELET)
        coeffs = swt.decompose(signal, level=4)

        reconstructions = {}
        SWT_LEVEL = len(coeffs)

        for i in range(SWT_LEVEL):
            label = f"D{SWT_LEVEL - i}"
            zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
            reconstructions[label] = self.cleaner.preprocess_subband(swt.reconstruct(zeroed))

        zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        reconstructions["A4"] = self.cleaner.preprocess_subband(swt.reconstruct(zeroed))

        return reconstructions

    def get_epsilon(self, signal, m, tau):
        return RecurrencePlotGenerator.find_epsilon_for_target_rr(signal, m, tau, target_rr=0.05)

    def compute_rp(
            self,
            signal: np.ndarray,
            time_delay: int,
            embedding_dim: int,
            epsilon_method: str = 'percentile',
            epsilon_factor: float = 0.05,
    ) -> Tuple[np.ndarray, float]:
        from pyrqa.time_series import TimeSeries
        from pyrqa.settings import Settings
        from pyrqa.analysis_type import Classic
        from pyrqa.neighbourhood import FixedRadius
        from pyrqa.metric import EuclideanMetric
        from pyrqa.computation import RPComputation
        from scipy.spatial.distance import pdist

        n_points = len(signal) - (embedding_dim - 1) * time_delay
        embedded = np.array([
            signal[i: i + embedding_dim * time_delay: time_delay]
            for i in range(n_points)
        ])

        if len(embedded) > 2000:
            idx = np.linspace(0, len(embedded) - 1, 2000, dtype=int)
            embedded_sub = embedded[idx]
        else:
            embedded_sub = embedded

        distances = pdist(embedded_sub, metric='euclidean')

        if epsilon_method == 'percentile':
            epsilon = np.percentile(distances, epsilon_factor * 100)
        elif epsilon_method == 'max_distance':
            epsilon = epsilon_factor * np.max(distances)
        elif epsilon_method == 'std':
            epsilon = epsilon_factor * np.std(signal)
        else:
            epsilon = self.get_epsilon(signal, m=embedding_dim, tau=time_delay)

        ts = TimeSeries(signal, embedding_dimension=embedding_dim, time_delay=time_delay)
        settings = Settings(
            time_series=ts,
            analysis_type=Classic,
            neighbourhood=FixedRadius(epsilon),
            similarity_measure=EuclideanMetric,
            theiler_corrector=1,
        )
        result = RPComputation.create(settings, verbose=False).run()
        return result.recurrence_matrix, epsilon

    def compute_rp_unthresholded(
            self,
            signal: np.ndarray,
            time_delay: int,
            embedding_dim: int,
    ) -> np.ndarray:
        from pyrqa.time_series import TimeSeries
        from pyrqa.settings import Settings
        from pyrqa.analysis_type import Classic
        from pyrqa.neighbourhood import Unthresholded
        from pyrqa.metric import EuclideanMetric
        from pyrqa.computation import RPComputation

        ts = TimeSeries(signal, embedding_dimension=embedding_dim, time_delay=time_delay)
        settings = Settings(
            time_series=ts,
            analysis_type=Classic,
            neighbourhood=Unthresholded(),
            similarity_measure=EuclideanMetric,
            theiler_corrector=1,
        )
        result = RPComputation.create(settings, verbose=False).run()
        return result.recurrence_matrix_reverse_normalized

    @staticmethod
    def _rr(rp_matrix: np.ndarray) -> float:
        return np.sum(rp_matrix) / rp_matrix.size

    def _label_meta(self, label: int) -> Tuple[str, str]:
        return ('Pre-AF', 'red') if label == 1 else ('Normal', 'blue')

    def plot_signal(self, idx: int, ax: Optional[plt.Axes] = None, show: bool = True):
        signal, label, seg_id, pat_id = self.get_signal(idx)
        label_str, color = self._label_meta(label)
        t = np.arange(len(signal)) / SAMPLING_FREQ

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))

        ax.plot(t, signal, color=color, linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{label_str} | {seg_id} | Patient: {pat_id}')
        ax.grid(True, alpha=0.3)

        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_subband_rps(
            self,
            idx: int,
            epsilon_method: str = 'percentile',
            epsilon_factor: float = 0.05,
            save_path: Optional[Path] = None,
            show: bool = True,
    ):
        """
        Three-column layout: [Subband Signal] | [Thresholded RP] | [Unthresholded RP (UTRP)]
        """
        signal, label, seg_id, pat_id = self.get_signal(idx)
        subbands = self.decompose_signal(signal)
        label_str, color = self._label_meta(label)

        ncols = 3
        fig, axes = plt.subplots(len(SUBBAND_NAMES), ncols, figsize=(6 * ncols, 3 * len(SUBBAND_NAMES)))

        for i, name in enumerate(SUBBAND_NAMES):
            sub = subbands[name]
            tau, m = self.get_optimal_params(sub)

            if sub is None:
                axes[i, 0].text(0.5, 0.5, f'{name}: insufficient energy', ha='center', va='center')
                axes[i, 1].axis('off')
                axes[i, 2].axis('off')
                continue

            # 1. Signal Plot
            t_sub = np.arange(len(sub)) / SAMPLING_FREQ
            axes[i, 0].plot(t_sub, sub, color=color, linewidth=0.5)
            axes[i, 0].set_ylabel(f'{name}\nstd={np.std(sub):.4f}')
            axes[i, 0].grid(True, alpha=0.3)
            if i == 0: axes[i, 0].set_title(f'{label_str} | {seg_id}')

            # 2. Thresholded RP
            try:
                rp, epsilon = self.compute_rp(sub, tau, m, epsilon_method, epsilon_factor)
                axes[i, 1].imshow(rp, cmap='binary', origin='lower', aspect='equal')
                axes[i, 1].set_title(f'RP | RR={self._rr(rp):.4f}, eps={epsilon:.4f}, {WAVELET}')
            except Exception as e:
                axes[i, 1].text(0.5, 0.5, f'Error: {str(e)[:20]}', ha='center')

            # 3. Unthresholded RP (UTRP)
            try:
                rp_u = self.compute_rp_unthresholded(sub, tau, m)
                im = axes[i, 2].imshow(rp_u, cmap='viridis', origin='lower', aspect='equal')
                axes[i, 2].set_title(f'UTRP | tau={tau}, m={m}')
                plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
            except Exception as e:
                axes[i, 2].text(0.5, 0.5, f'Error: {str(e)[:20]}', ha='center')

        axes[-1, 0].set_xlabel('Time (s)')
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=150)
        if show: plt.show()

    def plot_rp_comparison(
            self,
            idx1: int,
            idx2: int,
            epsilon_method: str = 'percentile',
            epsilon_factor: float = 0.10,
            save_path: Optional[Path] = None,
            show: bool = True,
    ):
        """Standardized 3-column comparison for two different signal indices."""
        indices = [idx1, idx2]
        ncols = 3
        fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 10))

        for row, idx in enumerate(indices):
            sig, t, l_str, color, sid, pid = self._get_signal_context(idx)
            tau, m = self.get_optimal_params(sig)

            # Col 0: Signal
            axes[row, 0].plot(t, sig, color=color, linewidth=0.5)
            axes[row, 0].set_title(f'{l_str} | {sid}')
            axes[row, 0].set_ylabel('Amplitude')

            # Col 1: RP
            rp, eps = self.compute_rp(sig, tau, m, epsilon_method, epsilon_factor)
            axes[row, 1].imshow(rp, cmap='binary', origin='lower', aspect='equal')
            axes[row, 1].set_title(f'RP | RR={self._rr(rp):.4f}\neps={eps:.4f}')

            # Col 2: UTRP
            rp_u = self.compute_rp_unthresholded(sig, tau, m)
            im = axes[row, 2].imshow(rp_u, cmap='viridis', origin='lower', aspect='equal')
            axes[row, 2].set_title('UTRP')
            plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=150)
        if show: plt.show()

    def get_optimal_params(self, signal):
        ami = AMI(signal)
        time_delay = ami.get_first_minimum()
        fnn = FNN(signal=signal, tau=time_delay)
        embedding_dim = fnn.get_optimal_dimension()
        return time_delay, embedding_dim

    def _get_signal_context(self, idx: int):
        sig, label, sid, pid = self.get_signal(idx)
        label_str, color = self._label_meta(label)
        t = np.arange(len(sig)) / SAMPLING_FREQ
        return sig, t, label_str, color, sid, pid

    def investigate(self, idx: int, epsilon_method: str = 'percentile', epsilon_factor: float = 0.05):
        self.plot_signal(idx)
        self.plot_subband_rps(idx, epsilon_method=epsilon_method, epsilon_factor=epsilon_factor)


if __name__ == '__main__':
    viz = PAFVisualizer(random_seed=43)
    indices = viz.get_random_indices(n_normal=1, n_preaf=1)

    idx_p = indices['preaf'][0]
    idx_n = indices['normal'][0]
    viz.investigate(idx_p)
    viz.investigate(idx_n)