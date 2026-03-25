import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import h5py
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wavelets.stationary_wavelet_transform as stationary_wavelet
from data_cleaning.data_cleaner import DataCleaner

DATA_DIR   = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted_clean/paf1')
HDF5_PATH  = DATA_DIR / 'paf_segments.h5'
# HDF5_PATH  = DATA_DIR / 'paf_segments_with_quality_check.h5'

SAMPLING_FREQ  = 128
SUBBAND_NAMES  = ["D1", "D2", "D3", "D4", "A4"]
WAVELET        = 'db4'


class PAFVisualizer:
    def __init__(
        self,
        subband_params: Dict[str, Dict[str, Any]],
        hdf5_path: Optional[Path] = None,
        random_seed: int = 42,
    ):
        self.hdf5_path     = hdf5_path or HDF5_PATH
        self.subband_params = subband_params
        self.rng           = np.random.default_rng(random_seed)
        self.cleaner       = DataCleaner(fs=SAMPLING_FREQ)

        self.signals     = None
        self.labels      = None
        self.segment_ids = None
        self.patient_ids = None
        self._load_data()

    def _load_data(self):
        with h5py.File(self.hdf5_path, 'r') as f:
            grp = f['all']
            self.signals     = grp['signals'][:]
            self.labels      = grp['labels'][:]
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

        preaf_idx  = np.where(self.labels == 1)[0]
        return {
            'normal': self.rng.choice(normal_idx_filtered, min(n_normal, len(normal_idx_filtered)), replace=False).tolist(),
            'preaf':  self.rng.choice(preaf_idx,  min(n_preaf,  len(preaf_idx)),  replace=False).tolist(),
        }
    def decompose_signal(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """SWT decomposition with reconstruction for each subband (zeroing out other coefficients)."""
        swt    = stationary_wavelet.SWT(wavelet=WAVELET)
        coeffs = swt.decompose(signal, level=4)

        reconstructions = {}
        SWT_LEVEL = len(coeffs)

        # Reconstruct each detail level by zeroing out all other coefficients
        for i in range(SWT_LEVEL):
            label = f"D{SWT_LEVEL - i}"
            # Zero out all coefficients except the one we want
            zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            # Keep only the detail coefficients for this level
            zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
            reconstructions[label] = self.cleaner.preprocess_subband(swt.reconstruct(zeroed))

        # Reconstruct A4 (approximation) by zeroing out all detail coefficients
        zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        reconstructions["A4"] = self.cleaner.preprocess_subband(swt.reconstruct(zeroed))

        return reconstructions

    def decompose_signal_coeffs(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """SWT decomposition returning raw coefficients (no reconstruction)."""
        swt    = stationary_wavelet.SWT(wavelet=WAVELET)
        coeffs = swt.decompose(signal, level=4)

        coefficients = {}
        SWT_LEVEL = len(coeffs)

        # Detail coefficients D1-D4
        for i in range(SWT_LEVEL):
            label = f"D{SWT_LEVEL - i}"
            # coeffs[i] is a tuple of (cA, cD), we want the detail coefficients (cD)
            coefficients[label] = self.cleaner.preprocess_subband(coeffs[i][1])

        # Approximation coefficients A4
        # coeffs[0] is the highest level, use approximation part (cA)
        coefficients["A4"] = self.cleaner.preprocess_subband(coeffs[0][0])

        return coefficients

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
            idx      = np.linspace(0, len(embedded) - 1, 2000, dtype=int)
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
            raise ValueError(f"Unknown epsilon_method: {epsilon_method}")

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

    def compute_rp_from_coefficients(
        self,
        coefficients: np.ndarray,
        time_delay: int,
        embedding_dim: int,
        epsilon_method: str = 'percentile',
        epsilon_factor: float = 0.05,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute RP directly from wavelet coefficients without reconstruction.

        Args:
            coefficients: Wavelet coefficients (1D array)
            time_delay: Time delay for phase space reconstruction
            embedding_dim: Embedding dimension
            epsilon_method: Method for epsilon computation ('percentile', 'max_distance', 'std')
            epsilon_factor: Factor for epsilon computation

        Returns:
            Tuple of (recurrence_matrix, epsilon_used)
        """
        from pyrqa.time_series import TimeSeries
        from pyrqa.settings import Settings
        from pyrqa.analysis_type import Classic
        from pyrqa.neighbourhood import FixedRadius
        from pyrqa.metric import EuclideanMetric
        from pyrqa.computation import RPComputation
        from scipy.spatial.distance import pdist

        n_points = len(coefficients) - (embedding_dim - 1) * time_delay

        embedded = np.array([
            coefficients[i: i + embedding_dim * time_delay: time_delay]
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
            epsilon = epsilon_factor * np.std(coefficients)
        else:
            raise ValueError(f"Unknown epsilon_method: {epsilon_method}")

        ts = TimeSeries(coefficients, embedding_dimension=embedding_dim, time_delay=time_delay)
        settings = Settings(
            time_series=ts,
            analysis_type=Classic,
            neighbourhood=FixedRadius(epsilon),
            similarity_measure=EuclideanMetric,
            theiler_corrector=1,
        )
        result = RPComputation.create(settings, verbose=False).run()
        return result.recurrence_matrix, epsilon

    def decompose_signal_dwt(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        DWT decomposition with reconstruction for each subband.

        Args:
            signal: Input signal

        Returns:
            Dictionary mapping subband names (D1, D2, D3, D4, A4) to reconstructed signals
        """
        import pywt

        # Perform 4-level DWT decomposition
        coeffs = pywt.wavedec(signal, WAVELET, level=4)

        # coeffs format: [cA4, cD4, cD3, cD2, cD1]
        reconstructions = {}

        # Reconstruct A4 (approximation)
        coeffs_a4 = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        rec_a4 = pywt.waverec(coeffs_a4, WAVELET)[:len(signal)]
        reconstructions['A4'] = self.cleaner.preprocess_subband(rec_a4)

        # Reconstruct detail coefficients (D4, D3, D2, D1)
        for i in range(1, len(coeffs)):
            level = 5 - i  # Convert index to level number (D4, D3, D2, D1)
            label = f"D{level}"

            # Zero out all coefficients except the one we want
            coeffs_single = [np.zeros_like(coeffs[0])] + [np.zeros_like(c) for c in coeffs[1:]]
            coeffs_single[i] = coeffs[i]

            # Reconstruct
            rec = pywt.waverec(coeffs_single, WAVELET)[:len(signal)]
            reconstructions[label] = self.cleaner.preprocess_subband(rec)

        return reconstructions

    def decompose_signal_dwt_coeffs(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        DWT decomposition returning raw coefficients (no reconstruction).

        Args:
            signal: Input signal

        Returns:
            Dictionary mapping subband names (D1, D2, D3, D4, A4) to coefficient arrays
        """
        import pywt

        # Perform 4-level DWT decomposition
        coeffs = pywt.wavedec(signal, WAVELET, level=4)

        # coeffs format: [cA4, cD4, cD3, cD2, cD1]
        coefficients = {}

        # Approximation coefficients (A4)
        coefficients['A4'] = self.cleaner.preprocess_subband(coeffs[0])

        # Detail coefficients (D4, D3, D2, D1)
        for i in range(1, len(coeffs)):
            level = 5 - i  # Convert index to level number (D4, D3, D2, D1)
            label = f"D{level}"
            coefficients[label] = self.cleaner.preprocess_subband(coeffs[i])

        return coefficients

    def compute_rp_from_dwt(
        self,
        signal: np.ndarray,
        subband_name: str,
        epsilon_method: str = 'percentile',
        epsilon_factor: float = 0.05,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute RP from DWT-based decomposition for a specific subband.

        Args:
            signal: Input signal
            subband_name: Name of subband (D1, D2, D3, D4, A4)
            epsilon_method: Method for epsilon computation
            epsilon_factor: Factor for epsilon computation

        Returns:
            Tuple of (recurrence_matrix, epsilon_used)
        """
        # Get DWT coefficients
        dwt_subbands = self.decompose_signal_dwt(signal)
        coeffs = dwt_subbands[subband_name]

        # Get parameters for this subband
        params = self.subband_params[subband_name]
        tau, m = params['tau'], params['m']

        # Compute RP directly from coefficients
        return self.compute_rp_from_coefficients(coeffs, tau, m, epsilon_method, epsilon_factor)


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

    def plot_subbands(self, idx: int, save_path: Optional[Path] = None):
        signal, label, seg_id, pat_id = self.get_signal(idx)
        subbands = self.decompose_signal(signal)
        label_str, color = self._label_meta(label)
        t_orig = np.arange(len(signal)) / SAMPLING_FREQ

        fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=False)

        axes[0].plot(t_orig, signal, color=color, linewidth=0.5)
        axes[0].set_ylabel('Original')
        axes[0].set_title(f'{label_str} | {seg_id} | Patient: {pat_id}')
        axes[0].grid(True, alpha=0.3)

        for i, name in enumerate(SUBBAND_NAMES):
            sub = subbands[name]
            if sub is None:
                axes[i+1].text(0.5, 0.5, f'{name}: insufficient energy',
                               ha='center', va='center', transform=axes[i+1].transAxes)
                axes[i+1].set_ylabel(name)
                continue
            t_sub = np.arange(len(sub)) / SAMPLING_FREQ
            axes[i+1].plot(t_sub, sub, color=color, linewidth=0.5)
            axes[i+1].set_ylabel(name)
            axes[i+1].grid(True, alpha=0.3)
            axes[i+1].text(0.98, 0.95, f'std={np.std(sub):.4f}',
                           transform=axes[i+1].transAxes, ha='right', va='top',
                           fontsize=9, family='monospace')

        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_subbands_comparison(self, idx1: int, idx2: int, save_path: Optional[Path] = None):
        sig1, label1, seg_id1, _ = self.get_signal(idx1)
        sig2, label2, seg_id2, _ = self.get_signal(idx2)
        sub1 = self.decompose_signal(sig1)
        sub2 = self.decompose_signal(sig2)
        label_str1, color1 = self._label_meta(label1)
        label_str2, color2 = self._label_meta(label2)
        t_orig = np.arange(len(sig1)) / SAMPLING_FREQ

        fig, axes = plt.subplots(6, 2, figsize=(16, 12), sharex=False)

        for col, (sig, subbands, color, label_str, seg_id) in enumerate([
            (sig1, sub1, color1, label_str1, seg_id1),
            (sig2, sub2, color2, label_str2, seg_id2),
        ]):
            axes[0, col].plot(t_orig, sig, color=color, linewidth=0.5)
            axes[0, col].set_title(f'{label_str} | {seg_id}')
            axes[0, col].set_ylabel('Original')
            axes[0, col].grid(True, alpha=0.3)

            for i, name in enumerate(SUBBAND_NAMES):
                s = subbands[name]
                if s is None:
                    axes[i+1, col].text(0.5, 0.5, f'{name}: insufficient energy',
                                        ha='center', va='center',
                                        transform=axes[i+1, col].transAxes)
                    axes[i+1, col].set_ylabel(name)
                    continue
                t_sub = np.arange(len(s)) / SAMPLING_FREQ
                axes[i+1, col].plot(t_sub, s, color=color, linewidth=0.5)
                axes[i+1, col].set_ylabel(name)
                axes[i+1, col].grid(True, alpha=0.3)
                axes[i+1, col].text(0.98, 0.95, f'std={np.std(s):.4f}',
                                    transform=axes[i+1, col].transAxes,
                                    ha='right', va='top', fontsize=9)

        axes[-1, 0].set_xlabel('Time (s)')
        axes[-1, 1].set_xlabel('Time (s)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_rp_comparison(
        self,
        idx1: int,
        idx2: int,
        epsilon_method: str = 'percentile',
        epsilon_factor: float = 0.10,
        show_unthresholded: bool = False,
        save_path: Optional[Path] = None,
        show: bool = True,
    ):
        sig1, label1, seg_id1, _ = self.get_signal(idx1)
        sig2, label2, seg_id2, _ = self.get_signal(idx2)

        params     = self.subband_params['original']
        time_delay = params['tau']
        emb_dim    = params['m']

        rp1, eps1 = self.compute_rp(sig1, time_delay, emb_dim, epsilon_method, epsilon_factor)
        rp2, eps2 = self.compute_rp(sig2, time_delay, emb_dim, epsilon_method, epsilon_factor)

        label_str1, color1 = self._label_meta(label1)
        label_str2, color2 = self._label_meta(label2)
        t = np.arange(len(sig1)) / SAMPLING_FREQ

        ncols = 3 if show_unthresholded else 2
        fig, axes = plt.subplots(2, ncols, figsize=(6*ncols, 10))

        for row, (sig, rp, eps, color, label_str, seg_id) in enumerate([
            (sig1, rp1, eps1, color1, label_str1, seg_id1),
            (sig2, rp2, eps2, color2, label_str2, seg_id2),
        ]):
            axes[row, 0].plot(t, sig, color=color, linewidth=0.5)
            axes[row, 0].set_title(f'{label_str} | {seg_id}')
            axes[row, 0].set_xlabel('Time (s)')
            axes[row, 0].set_ylabel('Amplitude')
            axes[row, 0].grid(True, alpha=0.3)

            if show_unthresholded:
                rp_u = self.compute_rp_unthresholded(sig, time_delay, emb_dim)
                im = axes[row, 1].imshow(rp_u, cmap='viridis', origin='lower', aspect='equal')
                axes[row, 1].set_title(f'{label_str} Unthresholded RP')
                plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)
                rp_col = 2
            else:
                rp_col = 1

            axes[row, rp_col].imshow(rp, cmap='binary', origin='lower', aspect='equal')
            axes[row, rp_col].set_title(
                f'{label_str} RP | RR={self._rr(rp):.4f}\n'
                f'tau={time_delay}, m={emb_dim}, eps={eps:.4f} ({epsilon_method})'
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()


    def plot_single_scale_ecg(
                self,
                idx: int,
                normalize: bool = False,
                detrend: bool = False,
                save_path: Optional[Path] = None,
                show: bool = True,
        ):
            from scipy.signal import detrend as scipy_detrend

            signal, label, seg_id, pat_id = self.get_signal(idx)
            label_str, color = self._label_meta(label)

            x = signal.copy()

            # Optional preprocessing (controlled, minimal)
            if detrend:
                x = scipy_detrend(x)

            if normalize:
                x = (x - np.mean(x)) / (np.std(x) + 1e-8)

            t = np.arange(len(x)) / SAMPLING_FREQ

            fig, ax = plt.subplots(figsize=(14, 4))

            ax.plot(t, x, color=color, linewidth=0.7)

            ax.set_title(
                f'Single-Scale ECG | {label_str} | {seg_id} | Patient: {pat_id}'
            )
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude' + (' (normalized)' if normalize else ''))

            ax.grid(True, alpha=0.3)

            # Optional statistics overlay (useful for sanity checks)
            ax.text(
                0.99, 0.95,
                f"mean={np.mean(x):.4f}\nstd={np.std(x):.4f}",
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=9,
                family='monospace'
            )

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150)

            if show:
                plt.show()

            return ax

    def plot_subband_rps(
        self,
        idx: int,
        epsilon_method: str = 'percentile',
        epsilon_factor: float = 0.05,
        show_unthresholded: bool = False,
        save_path: Optional[Path] = None,
        show: bool = True,
    ):

        signal, label, seg_id, pat_id = self.get_signal(idx)
        subbands = self.decompose_signal(signal)
        label_str, color = self._label_meta(label)

        ncols = 3 if show_unthresholded else 2
        fig, axes = plt.subplots(len(SUBBAND_NAMES), ncols, figsize=(6*ncols, 3*len(SUBBAND_NAMES)))

        for i, name in enumerate(SUBBAND_NAMES):
            sub = subbands[name]
            params = self.subband_params[name]
            tau, m = params['tau'], params['m']

            if sub is None:
                axes[i, 0].text(0.5, 0.5, f'{name}: insufficient energy',
                                ha='center', va='center', transform=axes[i, 0].transAxes)
                axes[i, 0].set_ylabel(name)
                for c in range(1, ncols):
                    axes[i, c].axis('off')
                if i == 0:
                    axes[i, 0].set_title(f'{label_str} | {seg_id} | {epsilon_method}')
                continue

            t_sub = np.arange(len(sub)) / SAMPLING_FREQ
            axes[i, 0].plot(t_sub, sub, color=color, linewidth=0.5)
            axes[i, 0].set_ylabel(f'{name}\nstd={np.std(sub):.4f}')
            axes[i, 0].grid(True, alpha=0.3)
            if i == 0:
                axes[i, 0].set_title(f'{label_str} | {seg_id} | {epsilon_method}')

            if show_unthresholded:
                try:
                    rp_u = self.compute_rp_unthresholded(sub, tau, m)
                    if rp_u.size > 0:
                        im = axes[i, 1].imshow(rp_u, cmap='viridis', origin='lower', aspect='equal')
                        axes[i, 1].set_title(f'{name} Unthresholded RP | tau={tau}, m={m}')
                        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)
                    else:
                        axes[i, 1].text(0.5, 0.5, f'{name}: empty RP',
                                        ha='center', va='center', transform=axes[i, 1].transAxes)
                except (ValueError, Exception) as e:
                    axes[i, 1].text(0.5, 0.5, f'{name}: RP error\n{str(e)[:30]}',
                                    ha='center', va='center', transform=axes[i, 1].transAxes)
                rp_col = 2
            else:
                rp_col = 1

            try:
                rp, epsilon = self.compute_rp(sub, tau, m, epsilon_method, epsilon_factor)
                rr = self._rr(rp)
                axes[i, rp_col].imshow(rp, cmap='binary', origin='lower', aspect='equal')
                axes[i, rp_col].set_title(
                    f'{name} RP | RR={rr:.4f}, eps={epsilon:.4f}, tau={tau}, m={m}'
                )
            except (ValueError, Exception) as e:
                axes[i, rp_col].text(0.5, 0.5, f'{name}: RP error\n{str(e)[:30]}',
                                     ha='center', va='center', transform=axes[i, rp_col].transAxes)

        axes[-1, 0].set_xlabel('Time (s)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()

    def investigate(
        self,
        idx: int,
        epsilon_method: str = 'percentile',
        epsilon_factor: float = 0.05,
        show_unthresholded: bool = False,
        save_path: Optional[Path] = None,
    ):
        # self.plot_signal(idx)
        # self.plot_subbands(idx)
        self.plot_subband_rps(idx, epsilon_method=epsilon_method,
                              epsilon_factor=epsilon_factor,
                              show_unthresholded=show_unthresholded,
                              save_path=save_path)

    def quick_compare(
        self,
        idx_n: int,
        idx_p: int,
        epsilon_method: str = 'percentile',
        epsilon_factor: float = 0.05,
        show_unthresholded: bool = False,
        save_path: Optional[Path] = None,
    ):
        # self.plot_subbands_comparison(idx_n, idx_p)
        self.plot_rp_comparison(idx_n, idx_p,epsilon_method=epsilon_method,epsilon_factor=epsilon_factor,show_unthresholded=show_unthresholded,save_path=save_path)

    def plot_three_methods_comparison(
        self,
        idx: int,
        epsilon_method: str = 'percentile',
        epsilon_factor: float = 0.05,
        save_prefix: Optional[Path] = None,
    ):
        """
        Generate three separate plots for the same signal:
        1. SWT decomposition + reconstruction -> RPs
        2. SWT decomposition -> RPs from coefficients directly
        3. DWT decomposition + reconstruction -> RPs

        Args:
            idx: Signal index
            epsilon_method: Epsilon computation method
            epsilon_factor: Epsilon factor
            save_prefix: Path prefix for saving (will append _method1.png, _method2.png, _method3.png)
        """
        signal, label, seg_id, pat_id = self.get_signal(idx)
        label_str, color = self._label_meta(label)

        # Method 1: SWT + Reconstruction -> RP
        print(f"Generating Method 1: SWT + Reconstruction -> RP...")
        self._plot_method_swt_reconstruction(
            signal, label_str, seg_id, pat_id, color,
            epsilon_method, epsilon_factor,
            save_prefix.parent / f"{save_prefix.name}_method1_swt_recon.png" if save_prefix else None
        )

        # Method 2: SWT -> Direct RP from coefficients
        print(f"Generating Method 2: SWT -> Direct RP from coefficients...")
        self._plot_method_swt_direct(
            signal, label_str, seg_id, pat_id, color,
            epsilon_method, epsilon_factor,
            save_prefix.parent / f"{save_prefix.name}_method2_swt_direct.png" if save_prefix else None
        )

        # Method 3: DWT + Reconstruction -> RP
        print(f"Generating Method 3: DWT + Reconstruction -> RP...")
        self._plot_method_dwt_reconstruction(
            signal, label_str, seg_id, pat_id, color,
            epsilon_method, epsilon_factor,
            save_prefix.parent / f"{save_prefix.name}_method3_dwt_recon.png" if save_prefix else None
        )

    def _plot_method_swt_reconstruction(
        self, signal, label_str, seg_id, pat_id, color,
        epsilon_method, epsilon_factor, save_path
    ):
        """Method 1: SWT decomposition + reconstruction -> RPs"""
        subbands = self.decompose_signal(signal)

        fig, axes = plt.subplots(len(SUBBAND_NAMES), 2, figsize=(12, 3*len(SUBBAND_NAMES)))
        fig.suptitle(f'Method 1: SWT + Reconstruction -> RP | {label_str} | {seg_id}', fontsize=14)

        for i, name in enumerate(SUBBAND_NAMES):
            sub = subbands[name]
            params = self.subband_params[name]
            tau, m = params['tau'], params['m']

            if sub is None:
                axes[i, 0].text(0.5, 0.5, f'{name}: insufficient energy',
                                ha='center', va='center', transform=axes[i, 0].transAxes)
                axes[i, 0].set_ylabel(name)
                axes[i, 1].axis('off')
                continue

            # Plot reconstructed subband
            t_sub = np.arange(len(sub)) / SAMPLING_FREQ
            axes[i, 0].plot(t_sub, sub, color=color, linewidth=0.5)
            axes[i, 0].set_ylabel(f'{name} (reconstructed)')
            axes[i, 0].grid(True, alpha=0.3)

            # Compute and plot RP
            try:
                rp, epsilon = self.compute_rp(sub, tau, m, epsilon_method, epsilon_factor)
                rr = self._rr(rp)
                axes[i, 1].imshow(rp, cmap='binary', origin='lower', aspect='equal')
                axes[i, 1].set_title(f'RR={rr:.4f}, eps={epsilon:.4f}')
            except Exception as e:
                axes[i, 1].text(0.5, 0.5, f'RP error\n{str(e)[:30]}',
                               ha='center', va='center', transform=axes[i, 1].transAxes)

        axes[-1, 0].set_xlabel('Time (s)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved: {save_path.name}")
        plt.close('all')

    def _plot_method_swt_direct(
        self, signal, label_str, seg_id, pat_id, color,
        epsilon_method, epsilon_factor, save_path
    ):
        """Method 2: SWT decomposition -> Direct RP from coefficients"""
        coeffs = self.decompose_signal_coeffs(signal)

        fig, axes = plt.subplots(len(SUBBAND_NAMES), 2, figsize=(12, 3*len(SUBBAND_NAMES)))
        fig.suptitle(f'Method 2: SWT -> Direct RP from Coefficients | {label_str} | {seg_id}', fontsize=14)

        for i, name in enumerate(SUBBAND_NAMES):
            coeff = coeffs[name]
            params = self.subband_params[name]
            tau, m = params['tau'], params['m']

            if coeff is None:
                axes[i, 0].text(0.5, 0.5, f'{name}: insufficient energy',
                                ha='center', va='center', transform=axes[i, 0].transAxes)
                axes[i, 0].set_ylabel(name)
                axes[i, 1].axis('off')
                continue

            # Plot coefficients
            axes[i, 0].plot(coeff, color=color, linewidth=0.5)
            axes[i, 0].set_ylabel(f'{name} (coefficients)')
            axes[i, 0].grid(True, alpha=0.3)

            # Compute RP directly from coefficients
            try:
                rp, epsilon = self.compute_rp_from_coefficients(coeff, tau, m, epsilon_method, epsilon_factor)
                rr = self._rr(rp)
                axes[i, 1].imshow(rp, cmap='binary', origin='lower', aspect='equal')
                axes[i, 1].set_title(f'RR={rr:.4f}, eps={epsilon:.4f}')
            except Exception as e:
                axes[i, 1].text(0.5, 0.5, f'RP error\n{str(e)[:30]}',
                               ha='center', va='center', transform=axes[i, 1].transAxes)

        axes[-1, 0].set_xlabel('Sample index')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved: {save_path.name}")
        plt.close('all')

    def _plot_method_dwt_reconstruction(
        self, signal, label_str, seg_id, pat_id, color,
        epsilon_method, epsilon_factor, save_path
    ):
        """Method 3: DWT decomposition + reconstruction -> RPs"""
        subbands = self.decompose_signal_dwt(signal)

        fig, axes = plt.subplots(len(SUBBAND_NAMES), 2, figsize=(12, 3*len(SUBBAND_NAMES)))
        fig.suptitle(f'Method 3: DWT + Reconstruction -> RP | {label_str} | {seg_id}', fontsize=14)

        for i, name in enumerate(SUBBAND_NAMES):
            sub = subbands[name]
            params = self.subband_params[name]
            tau, m = params['tau'], params['m']

            if sub is None:
                axes[i, 0].text(0.5, 0.5, f'{name}: insufficient energy',
                                ha='center', va='center', transform=axes[i, 0].transAxes)
                axes[i, 0].set_ylabel(name)
                axes[i, 1].axis('off')
                continue

            # Plot reconstructed subband
            t_sub = np.arange(len(sub)) / SAMPLING_FREQ
            axes[i, 0].plot(t_sub, sub, color=color, linewidth=0.5)
            axes[i, 0].set_ylabel(f'{name} (reconstructed)')
            axes[i, 0].grid(True, alpha=0.3)

            # Compute and plot RP
            try:
                rp, epsilon = self.compute_rp(sub, tau, m, epsilon_method, epsilon_factor)
                rr = self._rr(rp)
                axes[i, 1].imshow(rp, cmap='binary', origin='lower', aspect='equal')
                axes[i, 1].set_title(f'RR={rr:.4f}, eps={epsilon:.4f}')
            except Exception as e:
                axes[i, 1].text(0.5, 0.5, f'RP error\n{str(e)[:30]}',
                               ha='center', va='center', transform=axes[i, 1].transAxes)

        axes[-1, 0].set_xlabel('Time (s)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved: {save_path.name}")
        plt.close('all')
if __name__ == '__main__':
    subband_params = {
        "original": {"tau": 12,  "m": 7},
        'D1': {'tau': 1, 'm': 7},
        'D2': {'tau': 2, 'm': 7},
        'D3': {'tau': 3, 'm': 7},
        'D4': {'tau': 6, 'm': 7},
        'A4': {'tau': 20, 'm': 7},
    }


    viz = PAFVisualizer(subband_params=subband_params, random_seed=42)
    #42,43 seeds are good

    epsilon_method = 'percentile'
    epsilon_factor = 0.05

    indices = viz.get_random_indices(n_normal=1, n_preaf=1)

    idx_n   = indices['normal'][0]
    idx_p   = indices['preaf'][0]

    output_dir = Path(__file__).parent.parent.parent / 'outputs' / 'paf_visualizations_3methods'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to: {output_dir}")

    indices = viz.get_random_indices(n_normal=2, n_preaf=2)

    # viz.plot_single_scale_ecg(idx_n)
    # viz.plot_single_scale_ecg(idx_p)

    # print("\n=== Processing Normal segments ===")
    for i, idx in enumerate(indices['normal']):
        sig, label, seg_id, pat_id = viz.get_signal(idx)
        print(f"\nNormal sample {i+1}: {seg_id} (Patient: {pat_id})")
        save_prefix = output_dir / f'normal_{i}_{seg_id}'
        viz.plot_three_methods_comparison(idx, epsilon_method, epsilon_factor, save_prefix)

    # print("\n=== Processing Pre-AF segments ===")
    # for i, idx in enumerate(indices['preaf']):
    #     sig, label, seg_id, pat_id = viz.get_signal(idx)
    #     print(f"\nPre-AF sample {i+1}: {seg_id} (Patient: {pat_id})")
    #     save_prefix = output_dir / f'preaf_{i}_{seg_id}'
    #     viz.plot_three_methods_comparison(idx, epsilon_method, epsilon_factor, save_prefix)
