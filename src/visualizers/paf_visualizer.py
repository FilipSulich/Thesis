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

DATA_DIR   = Path('/data/data_new/exp2')
HDF5_PATH  = DATA_DIR / 'paf_new.h5'

SAMPLING_FREQ  = 128
SUBBAND_NAMES  = ["D1", "D2", "D3", "D4", "A4"]
WAVELET        = 'coif2'


class PAFVisualizer:
    def __init__(
        self,
        hdf5_path: Optional[Path] = None,
        random_seed: int = 42,
    ):
        self.hdf5_path     = hdf5_path or HDF5_PATH
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

    def get_epsilon(selfs, signal, m ,tau):
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
        # epsilon = self.get_epsilon(signal, m=embedding_dim, tau=time_delay)
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

        time_delay1, emb_dim1 = self.get_optimal_params(sig1)
        time_delay2, emb_dim2 = self.get_optimal_params(sig2)

        rp1, eps1 = self.compute_rp(sig1, time_delay1, emb_dim1, epsilon_method, epsilon_factor)
        rp2, eps2 = self.compute_rp(sig2, time_delay2, emb_dim2, epsilon_method, epsilon_factor)

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
                rp_u = self.compute_rp_unthresholded(sig, time_delay1, emb_dim1)
                im = axes[row, 1].imshow(rp_u, cmap='viridis', origin='lower', aspect='equal')
                axes[row, 1].set_title(f'{label_str} Unthresholded RP')
                plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)
                rp_col = 2
            else:
                rp_col = 1

            axes[row, rp_col].imshow(rp, cmap='binary', origin='lower', aspect='equal')
            axes[row, rp_col].set_title(
                f'{label_str} RP | RR={self._rr(rp):.4f}\n'
                f'tau={time_delay1}, m={emb_dim1}, eps={eps:.4f} ({epsilon_method})'
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

            tau, m = self.get_optimal_params(sub)

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
                    f'{name}| eps={epsilon:.4f}, tau={tau}, m={m}, wavelet={WAVELET}'
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
        self.plot_signal(idx)
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
        epsilon_factor: float = 0.1,
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
        signal, label, seg_id, pat_id = self.get_signal(idx)
        label_str, color = self._label_meta(label)

        self._plot_method_swt_reconstruction(
            signal, label_str, seg_id, pat_id, color,
            epsilon_method, epsilon_factor,
            save_prefix.parent / f"{save_prefix.name}_method1_swt_recon.png" if save_prefix else None
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
            tau, m = self.get_optimal_params(sub)

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

    def get_optimal_params(self, signal):
        ami = AMI(signal)
        time_delay = ami.get_first_minimum()
        fnn = FNN(signal=signal, tau=time_delay)
        embedding_dim = fnn.get_optimal_dimension()

        # time_delay = 3
        # embedding_dim = 3

        return time_delay, embedding_dim

    def _get_signal_context(self, idx: int):
        """Helper to get signal, metadata, and time axis in one call."""
        sig, label, sid, pid = self.get_signal(idx)
        label_str, color = ('Pre-AF', 'red') if label == 1 else ('Normal', 'blue')
        t = np.arange(len(sig)) / SAMPLING_FREQ
        return sig, t, label_str, color, sid, pid

    def plot_subband_envelopes(self, idx: int, save_path: Optional[Path] = None, show: bool = True):
        """
        Calculates and plots the Hilbert envelope for each wavelet subband.
        The envelope represents the instantaneous amplitude of the signal.
        """
        sig, t_orig, l_str, color, sid, pid = self._get_signal_context(idx)
        subbands = self.decompose_signal(sig)

        fig, axes = plt.subplots(len(SUBBAND_NAMES), 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f'Hilbert Envelopes | {l_str} | {sid} | Patient: {pid}', fontsize=14)

        for i, name in enumerate(SUBBAND_NAMES):
            sub = subbands[name]
            ax = axes[i]

            if sub is not None:
                # 1. Compute the analytic signal using the Hilbert transform
                analytic_signal = hilbert(sub)
                # 2. The envelope is the magnitude (absolute value) of the analytic signal
                amplitude_envelope = np.abs(analytic_signal)

                t_sub = np.arange(len(sub)) / SAMPLING_FREQ

                # Plot the original subband signal in a lighter shade
                ax.plot(t_sub, sub, color=color, alpha=0.3, linewidth=0.5, label='Subband')
                # Plot the Hilbert envelope on top
                ax.plot(t_sub, amplitude_envelope, color='black', linewidth=1.2, label='Envelope')

                ax.set_ylabel(f'{name}')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.2)
            else:
                ax.text(0.5, 0.5, f'{name}: Insufficient Energy',
                        ha='center', va='center', transform=ax.transAxes)

        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle

        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()

    def _detect_r_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Basic R-peak detection using height and distance constraints."""
        # Normalize signal for peak detection consistency
        norm_sig = (signal - np.mean(signal)) / np.std(signal)
        # Assuming min heart rate of 40bpm at 128Hz -> ~192 samples distance
        # We use a lower bound (0.4s) to catch faster rates
        min_dist = int(0.4 * SAMPLING_FREQ)
        peaks, _ = find_peaks(norm_sig, distance=min_dist, height=1.5)
        return peaks

    def plot_single_rr_analysis(
            self,
            idx: int,
            rr_index: int = 1,  # Which heartbeat to pick (0 is first, 1 is usually more stable)
            epsilon_method: str = 'percentile',
            epsilon_factor: float = 0.05,
            save_path: Optional[Path] = None
    ):
        """
        Detects R-peaks, extracts a single RR interval, decomposes it,
        and visualizes the subbands alongside their Recurrence Plots.
        """
        sig, t_orig, l_str, color, sid, pid = self._get_signal_context(idx)
        peaks = self._detect_r_peaks(sig)

        if len(peaks) < rr_index + 2:
            print(f"Not enough heartbeats found in segment {sid}")
            return

        # Extract the specific RR interval
        start_idx = peaks[rr_index]
        end_idx = peaks[rr_index + 1]
        rr_signal = sig[start_idx:end_idx]

        # Decompose only this interval
        subbands = self.decompose_signal(rr_signal)

        fig, axes = plt.subplots(len(SUBBAND_NAMES), 2, figsize=(12, 18))
        fig.suptitle(f'Single RR Interval Analysis | {l_str} | {sid}\n'
                     f'Interval: {start_idx / SAMPLING_FREQ:.2f}s to {end_idx / SAMPLING_FREQ:.2f}s', fontsize=14)

        for i, name in enumerate(SUBBAND_NAMES):
            sub = subbands[name]
            if sub is None: continue

            # 1. Plot Subband Signal
            t_rr = np.arange(len(sub)) / SAMPLING_FREQ
            axes[i, 0].plot(t_rr, sub, color=color, linewidth=1)
            axes[i, 0].set_ylabel(f'{name}')
            axes[i, 0].grid(True, alpha=0.2)
            if i == 0: axes[i, 0].set_title("Subband Signal")

            # 2. Compute and Plot RP
            try:
                tau, m = self.get_optimal_params(sub)
                rp, eps = self.compute_rp(sub, tau, m, epsilon_method, epsilon_factor)
                rr_val = self._rr(rp)

                axes[i, 1].imshow(rp, cmap='binary', origin='lower', aspect='equal')
                axes[i, 1].set_title(f'RP: RR={rr_val:.4f}, eps={eps:.4f}, m={m}')
            except Exception as e:
                axes[i, 1].text(0.5, 0.5, f"RP Error: {str(e)[:20]}",
                                ha='center', transform=axes[i, 1].transAxes)

        axes[-1, 0].set_xlabel('Time (s) within RR')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_utrp_feature_visibility(
            self,
            idx: int,
            subband_to_test: str = 'D4',
            zoom_samples: int = 512,  # Focus on a few beats
            save_path: Optional[Path] = None
    ):
        """
        Comparison of Raw vs Subband UTRP to prove multi-scale unmasking.
        """
        sig, _, l_str, color, sid, pid = self._get_signal_context(idx)
        subbands = self.decompose_signal(sig)
        sub = subbands[subband_to_test]

        # Focus on a specific window to see texture clearly
        start = len(sig) // 2
        raw_window = sig[start: start + zoom_samples]
        sub_window = sub[start: start + zoom_samples]

        # Z-normalize both for fair contrast comparison
        raw_window = (raw_window - np.mean(raw_window)) / np.std(raw_window)
        sub_window = (sub_window - np.mean(sub_window)) / np.std(sub_window)

        # Use a fixed tau/m for comparison proof
        tau, m = 4, 3

        # Compute Unthresholded Matrices (UTRP)
        rp_raw = self.compute_rp_unthresholded(raw_window, tau, m)
        rp_sub = self.compute_rp_unthresholded(sub_window, tau, m)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Raw vs {subband_to_test} | {l_str}', fontsize=16)

        # Row 1: Signals
        axes[0, 0].plot(raw_window, color='gray', alpha=0.8)
        axes[0, 0].set_title("Raw Signal (Z-Normalized)")

        axes[0, 1].plot(sub_window, color=color)
        axes[0, 1].set_title(f"Subband {subband_to_test} (Z-Normalized)")

        # Row 2: UTRPs
        im1 = axes[1, 0].imshow(rp_raw, cmap='magma', origin='lower')
        axes[1, 0].set_title("Raw UTRP ")
        plt.colorbar(im1, ax=axes[1, 0])

        im2 = axes[1, 1].imshow(rp_sub, cmap='magma', origin='lower')
        axes[1, 1].set_title(f"{subband_to_test} UTRP")
        plt.colorbar(im2, ax=axes[1, 1])

        plt.tight_layout()
        if save_path: plt.savefig(save_path)
        plt.show()

if __name__ == '__main__':
    path = Path("/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_new/exp2/paf_new.h5")
    viz = PAFVisualizer(path, random_seed=43)
    #42,43 seeds are good

    epsilon_method = 'percentile'
    epsilon_factor = 0.05

    indices = viz.get_random_indices(n_normal=1, n_preaf=1)

    idx_n   = indices['normal'][0]
    idx_p   = indices['preaf'][0]

    output_dir = Path(__file__).parent.parent.parent / 'outputs' / 'paf_visualizations_3methods'
    output_dir.mkdir(parents=True, exist_ok=True)

    viz.investigate(idx_p)
    viz.investigate(idx_n)
    subbands = ['D1', 'D2', 'D3', 'D4', 'A4']

    # for subband in subbands:
    #     viz.plot_utrp_feature_visibility(idx_p, subband_to_test=subband)
        # viz.plot_utrp_feature_visibility(idx_n, subband_to_test=subband)
    # viz.plot_subband_envelopes(idx_p, save_path=output_dir)