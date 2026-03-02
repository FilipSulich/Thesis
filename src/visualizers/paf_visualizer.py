import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import h5py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wavelets.stationary_wavelet_transform as stationary_wavelet

DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf')
# HDF5_PATH = DATA_DIR / 'paf_segments_all_clean.h5'
HDF5_PATH = DATA_DIR / 'paf_segments_all_clean_new_cleaning.h5'
# HDF5_PATH = DATA_DIR / 'paf_segments_all_clean_new_cleaning_no_upper_filter.h5'
# HDF5_PATH = DATA_DIR / 'paf_segments_all_clean_new_cleaning_min_max_with_upper.h5'

SAMPLING_FREQ = 128
SUBBAND_NAMES = ["D1", "D2", "D3", "D4", "A4"]


class PAFVisualizer:
    def __init__(self, subband_params: Optional[Dict[str, Dict[str, Any]]], hdf5_path: Optional[Path] = None, ):
        self.hdf5_path = hdf5_path or HDF5_PATH
        self.signals = None
        self.labels = None
        self.segment_ids = None
        self.patient_ids = None
        self.embedding_dimension = None
        self.time_delay = None
        self.target_rr = None
        self.subband_params = subband_params
        self._load_data()

    def _load_data(self):
        """Load data from HDF5."""
        with h5py.File(self.hdf5_path, 'r') as f:
            grp = f['all']
            self.signals = grp['signals'][:]
            self.labels = grp['labels'][:]
            self.segment_ids = [s.decode() if isinstance(s, bytes) else s for s in grp['segment_ids'][:]]
            if 'patient_ids' in grp:
                self.patient_ids = [s.decode() if isinstance(s, bytes) else s for s in grp['patient_ids'][:]]
            else:
                self.patient_ids = [''] * len(self.signals)

        print(f"Loaded {len(self.signals)} signals from {self.hdf5_path}")
        print(f"  Normal: {sum(self.labels == 0)}, Pre-AF: {sum(self.labels == 1)}")

    def get_signal(self, idx: int) -> Tuple[np.ndarray, int, str, str]:
        """Get signal by index. Returns (signal, label, segment_id, patient_id)."""
        return self.signals[idx], self.labels[idx], self.segment_ids[idx], self.patient_ids[idx]

    def get_random_indices(self, n_normal: int = 3, n_preaf: int = 3) -> Dict[str, List[int]]:
        """Get random indices for normal and pre-AF signals."""
        normal_idx = np.where(self.labels == 0)[0]
        preaf_idx = np.where(self.labels == 1)[0]

        return {
            'normal': np.random.choice(normal_idx, min(n_normal, len(normal_idx)), replace=False).tolist(),
            'preaf': np.random.choice(preaf_idx, min(n_preaf, len(preaf_idx)), replace=False).tolist()
        }

    # =========================================================================
    # 1. Signal Plotting
    # =========================================================================

    def plot_signal(self, idx: int, ax: Optional[plt.Axes] = None, show: bool = True):
        """Plot a single signal."""
        signal, label, seg_id, pat_id = self.get_signal(idx)
        t = np.arange(len(signal)) / SAMPLING_FREQ

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))

        color = 'red' if label == 1 else 'blue'
        label_str = 'Pre-AF' if label == 1 else 'Normal'

        ax.plot(t, signal, color=color, linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{label_str} | {seg_id} | Patient: {pat_id}')
        ax.grid(True, alpha=0.3)

        if show and ax is None:
            plt.tight_layout()
            plt.show()

        return ax

    def plot_signals_grid(self, indices: List[int], ncols: int = 2, save_path: Optional[Path] = None):
        """Plot multiple signals in a grid."""
        n = len(indices)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
        axes = np.atleast_2d(axes).flatten()

        for i, idx in enumerate(indices):
            self.plot_signal(idx, ax=axes[i], show=False)

        # Hide unused axes
        for i in range(n, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        plt.show()

    def plot_comparison(self, n_each: int = 3, save_path: Optional[Path] = None):
        """Plot normal vs pre-AF comparison."""
        indices = self.get_random_indices(n_each, n_each)
        all_indices = indices['normal'] + indices['preaf']
        self.plot_signals_grid(all_indices, ncols=2, save_path=save_path)

    # =========================================================================
    # 2. Subband Plotting
    # =========================================================================

    def decompose_signal(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Decompose signal into SWT subbands."""
        swt = stationary_wavelet.SWT()
        coeffs = swt.decompose(signal, level=4)

        reconstructions = {}

        # Detail coefficients D1-D4
        for i in range(len(coeffs)):
            zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
            reconstructions[f"D{i+1}"] = swt.reconstruct(zeroed)

        # Approximation A4
        zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        reconstructions["A4"] = swt.reconstruct(zeroed)

        return reconstructions

    def plot_subbands(self, idx: int, save_path: Optional[Path] = None):
        """Plot signal and all its subbands."""
        signal, label, seg_id, pat_id = self.get_signal(idx)
        subbands = self.decompose_signal(signal)

        t = np.arange(len(signal)) / SAMPLING_FREQ
        color = 'red' if label == 1 else 'blue'
        label_str = 'Pre-AF' if label == 1 else 'Normal'

        fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)

        # Original signal
        axes[0].plot(t, signal, color=color, linewidth=0.5)
        axes[0].set_ylabel('Original')
        axes[0].set_title(f'{label_str} | {seg_id} | Patient: {pat_id}')
        axes[0].grid(True, alpha=0.3)

        # Subbands
        for i, name in enumerate(SUBBAND_NAMES):
            sub = subbands[name]
            axes[i+1].plot(t[:len(sub)], sub, color=color, linewidth=0.5)
            axes[i+1].set_ylabel(name)
            axes[i+1].grid(True, alpha=0.3)

            # Show stats
            std = np.std(sub)
            axes[i+1].text(0.98, 0.95, f'std={std:.4f}', transform=axes[i+1].transAxes,
                          ha='right', va='top', fontsize=9, family='monospace')

        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        plt.show()

    def plot_subbands_comparison(self, idx1: int, idx2: int, save_path: Optional[Path] = None):
        """Compare subbands between two signals (uses actual labels)."""
        sig1, label1, seg_id1, _ = self.get_signal(idx1)
        sig2, label2, seg_id2, _ = self.get_signal(idx2)

        subbands1 = self.decompose_signal(sig1)
        subbands2 = self.decompose_signal(sig2)

        t = np.arange(len(sig1)) / SAMPLING_FREQ

        fig, axes = plt.subplots(6, 2, figsize=(16, 12), sharex=True)

        # Determine colors and labels from actual data
        color1 = 'red' if label1 == 1 else 'blue'
        color2 = 'red' if label2 == 1 else 'blue'
        label_str1 = 'PRE-AF' if label1 == 1 else 'NORMAL'
        label_str2 = 'PRE-AF' if label2 == 1 else 'NORMAL'

        # Original signals
        axes[0, 0].plot(t, sig1, color=color1, linewidth=0.5)
        axes[0, 0].set_ylabel('Original')
        axes[0, 0].set_title(f'{label_str1} | {seg_id1}')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(t, sig2, color=color2, linewidth=0.5)
        axes[0, 1].set_ylabel('Original')
        axes[0, 1].set_title(f'{label_str2} | {seg_id2}')
        axes[0, 1].grid(True, alpha=0.3)

        # Subbands
        for i, name in enumerate(SUBBAND_NAMES):
            sub1 = subbands1[name]
            sub2 = subbands2[name]

            axes[i+1, 0].plot(t[:len(sub1)], sub1, color=color1, linewidth=0.5)
            axes[i+1, 0].set_ylabel(name)
            axes[i+1, 0].grid(True, alpha=0.3)
            axes[i+1, 0].text(0.98, 0.95, f'std={np.std(sub1):.4f}',
                             transform=axes[i+1, 0].transAxes, ha='right', va='top', fontsize=9)

            axes[i+1, 1].plot(t[:len(sub2)], sub2, color=color2, linewidth=0.5)
            axes[i+1, 1].set_ylabel(name)
            axes[i+1, 1].grid(True, alpha=0.3)
            axes[i+1, 1].text(0.98, 0.95, f'std={np.std(sub2):.4f}',
                             transform=axes[i+1, 1].transAxes, ha='right', va='top', fontsize=9)

        axes[-1, 0].set_xlabel('Time (s)')
        axes[-1, 1].set_xlabel('Time (s)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        plt.show()

    def compute_rp(
            self,
            signal: np.ndarray,
            time_delay: int = 12,
            embedding_dim: int = 7,
            epsilon_method: str = 'std',  # 'std' or 'max_distance'
            epsilon_factor: float = 0.2
    ) -> np.ndarray:
        """Compute recurrence plot. Epsilon computed from raw unnormalized signal."""
        from pyrqa.time_series import TimeSeries
        from pyrqa.settings import Settings
        from pyrqa.analysis_type import Classic
        from pyrqa.neighbourhood import FixedRadius
        from pyrqa.metric import EuclideanMetric
        from pyrqa.computation import RPComputation

        # Compute epsilon from raw signal BEFORE any normalization
        if epsilon_method == 'std':
            epsilon = epsilon_factor * np.std(signal)
        elif epsilon_method == 'max_distance':
            n = len(signal) - (embedding_dim - 1) * time_delay
            embedded = np.array([signal[i:i + embedding_dim * time_delay:time_delay] for i in range(n)])
            from scipy.spatial.distance import pdist
            distances = pdist(embedded, metric='euclidean')
            epsilon = epsilon_factor * np.max(distances)

        ts = TimeSeries(signal, embedding_dimension=embedding_dim, time_delay=time_delay)
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

    def compute_rp_unthresholded(
            self,
            signal: np.ndarray,
            time_delay: int = 12,
            embedding_dim: int = 7
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
            theiler_corrector=1
        )
        rp_computation = RPComputation.create(settings, verbose=False)
        result = rp_computation.run()
        return result.recurrence_matrix_reverse_normalized

    def plot_rp(
        self,
        idx: int,
        time_delay: int = 12,
        embedding_dim: int = 7,
        epsilon_percent: float = 0.05,
        show_unthresholded: bool = False,
        save_path: Optional[Path] = None
    ):
        signal, label, seg_id, pat_id = self.get_signal(idx)
        rp_thresh = self.compute_rp(signal, time_delay, embedding_dim, epsilon_percent=epsilon_percent)

        label_str = 'Pre-AF' if label == 1 else 'Normal'
        rr = np.sum(rp_thresh) / (rp_thresh.shape[0] * rp_thresh.shape[1])

        if show_unthresholded:
            rp_unthresh = self.compute_rp_unthresholded(signal, time_delay, embedding_dim)
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Signal
        t = np.arange(len(signal)) / SAMPLING_FREQ
        color = 'red' if label == 1 else 'blue'
        axes[0].plot(t, signal, color=color, linewidth=0.5)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'{label_str} Signal | {seg_id}')
        axes[0].grid(True, alpha=0.3)

        if show_unthresholded:
            # Unthresholded RP
            im1 = axes[1].imshow(rp_unthresh, cmap='viridis', origin='lower', aspect='equal')
            axes[1].set_xlabel('i')
            axes[1].set_ylabel('j')
            axes[1].set_title(f'Unthresholded RP (Distance Matrix)')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            # Thresholded RP
            axes[2].imshow(rp_thresh, cmap='binary', origin='lower', aspect='equal')
            axes[2].set_xlabel('i')
            axes[2].set_ylabel('j')
            axes[2].set_title(f'Thresholded RP | tau={time_delay}, m={embedding_dim}, RR={rr:.4f}')
        else:
            # Thresholded RP only
            axes[1].imshow(rp_thresh, cmap='binary', origin='lower', aspect='equal')
            axes[1].set_xlabel('i')
            axes[1].set_ylabel('j')
            axes[1].set_title(f'RP | tau={time_delay}, m={embedding_dim}, RR={rr:.4f}')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        plt.show()

    def plot_rp_comparison(
        self,
        idx1: int,
        idx2: int,
        time_delay: int = 12,
        embedding_dim: int = 7,
        epsilon_percent: float = 0.10,
        show_unthresholded: bool = False,
        save_path: Optional[Path] = None
    ):
        sig1, label1, seg_id1, _ = self.get_signal(idx1)
        sig2, label2, seg_id2, _ = self.get_signal(idx2)

        rp1_thresh = self.compute_rp(sig1, time_delay, embedding_dim, epsilon_percent=epsilon_percent)
        rp2_thresh = self.compute_rp(sig2, time_delay, embedding_dim, epsilon_percent=epsilon_percent)

        rr1 = np.sum(rp1_thresh) / (rp1_thresh.shape[0] * rp1_thresh.shape[1])
        rr2 = np.sum(rp2_thresh) / (rp2_thresh.shape[0] * rp2_thresh.shape[1])

        # Determine colors and labels from actual data
        color1 = 'red' if label1 == 1 else 'blue'
        color2 = 'red' if label2 == 1 else 'blue'
        label_str1 = 'PRE-AF' if label1 == 1 else 'NORMAL'
        label_str2 = 'PRE-AF' if label2 == 1 else 'NORMAL'

        if show_unthresholded:
            rp1_unthresh = self.compute_rp_unthresholded(sig1, time_delay, embedding_dim)
            rp2_unthresh = self.compute_rp_unthresholded(sig2, time_delay, embedding_dim)
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        t = np.arange(len(sig1)) / SAMPLING_FREQ

        # Signal 1
        axes[0, 0].plot(t, sig1, color=color1, linewidth=0.5)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title(f'{label_str1} | {seg_id1}')
        axes[0, 0].grid(True, alpha=0.3)

        if show_unthresholded:
            # Unthresholded RP 1
            im1 = axes[0, 1].imshow(rp1_unthresh, cmap='viridis', origin='lower', aspect='equal')
            axes[0, 1].set_title(f'{label_str1} Unthresholded RP')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

            # Thresholded RP 1
            axes[0, 2].imshow(rp1_thresh, cmap='binary', origin='lower', aspect='equal')
            axes[0, 2].set_title(f'{label_str1} Thresholded RP | RR={rr1:.4f}')
        else:
            # Thresholded RP 1
            axes[0, 1].imshow(rp1_thresh, cmap='binary', origin='lower', aspect='equal')
            axes[0, 1].set_title(f'{label_str1} RP | RR={rr1:.4f}')

        # Signal 2
        axes[1, 0].plot(t, sig2, color=color2, linewidth=0.5)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].set_title(f'{label_str2} | {seg_id2}')
        axes[1, 0].grid(True, alpha=0.3)

        if show_unthresholded:
            # Unthresholded RP 2
            im2 = axes[1, 1].imshow(rp2_unthresh, cmap='viridis', origin='lower', aspect='equal')
            axes[1, 1].set_title(f'{label_str2} Unthresholded RP')
            plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

            # Thresholded RP 2
            axes[1, 2].imshow(rp2_thresh, cmap='binary', origin='lower', aspect='equal')
            axes[1, 2].set_title(f'{label_str2} Thresholded RP | RR={rr2:.4f}')
        else:
            # Thresholded RP 2
            axes[1, 1].imshow(rp2_thresh, cmap='binary', origin='lower', aspect='equal')
            axes[1, 1].set_title(f'{label_str2} RP | RR={rr2:.4f}')

        plt.suptitle(f'tau={time_delay}, m={embedding_dim}, eps%={epsilon_percent}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        plt.show()

    def plot_subband_rps(
            self,
            idx: int,
            epsilon_method: str = 'std',
            epsilon_factor: float = 0.2,
            show_unthresholded: bool = False,
            save_path: Optional[Path] = None
    ):
        signal, label, seg_id, pat_id = self.get_signal(idx)
        subbands = self.decompose_signal(signal)

        label_str = 'Pre-AF' if label == 1 else 'Normal'
        color = 'red' if label == 1 else 'blue'

        if show_unthresholded:
            fig, axes = plt.subplots(5, 3, figsize=(16, 15))
        else:
            fig, axes = plt.subplots(5, 2, figsize=(12, 15))

        for i, name in enumerate(SUBBAND_NAMES):
            sub = subbands[name]
            params = self.subband_params[name]
            tau = params.get('tau', 3)
            m = params.get('m', 3)

            t = np.arange(len(sub)) / SAMPLING_FREQ

            axes[i, 0].plot(t, sub, color=color, linewidth=0.5)
            axes[i, 0].set_ylabel(f'{name}\nstd={np.std(sub):.4f}')
            axes[i, 0].grid(True, alpha=0.3)
            if i == 0:
                axes[i, 0].set_title(f'{label_str} | {seg_id} | id: {idx}')

            eps = epsilon_factor * np.std(sub) if epsilon_method == 'std' else epsilon_factor * np.max(sub)

            if show_unthresholded:
                rp_unthresh = self.compute_rp_unthresholded(sub, tau, m)
                im = axes[i, 1].imshow(rp_unthresh, cmap='viridis', origin='lower', aspect='equal')
                axes[i, 1].set_title(f'{name} Unthresholded RP')
                plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

                rp_thresh = self.compute_rp(sub, tau, m,
                                            epsilon_method=epsilon_method,
                                            epsilon_factor=epsilon_factor)
                rr = np.sum(rp_thresh) / (rp_thresh.shape[0] * rp_thresh.shape[1])
                axes[i, 2].imshow(rp_thresh, cmap='binary', origin='lower', aspect='equal')
                axes[i, 2].set_title(f'{name} Thresholded RP | RR={rr:.4f}, eps={eps:.4f}')
            else:
                rp = self.compute_rp(sub, tau, m,
                                     epsilon_method=epsilon_method,
                                     epsilon_factor=epsilon_factor)
                rr = np.sum(rp) / (rp.shape[0] * rp.shape[1])
                axes[i, 1].imshow(rp, cmap='binary', origin='lower', aspect='equal')
                axes[i, 1].set_title(f'{name} RP | RR={rr:.4f}, eps={eps:.4f}')

        axes[-1, 0].set_xlabel('Time (s)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


    def investigate(self, idx: int, time_delay: int = 12, embedding_dim: int = 7,
                    epsilon_method: str = 'std', epsilon_factor: float = 0.2,
                    show_unthresholded: bool = False):
        self.plot_signal(idx)

        # self.plot_subbands(idx)

        # self.plot
        self.plot_subband_rps(idx, epsilon_method=epsilon_method, epsilon_factor=epsilon_factor, show_unthresholded=show_unthresholded)
    def quick_compare(self, idx_n, idx_p, time_delay: int = 12, embedding_dim: int = 7, embedding_percent: float = 0.05, show_unthresholded: bool = False):
        self.plot_subbands_comparison(idx_n, idx_p)
        self.plot_rp_comparison(idx_n, idx_p, time_delay=time_delay, embedding_dim=embedding_dim, epsilon_percent=embedding_percent, show_unthresholded=show_unthresholded)


if __name__ == '__main__':
    subband_params = {
        "original": {"tau": 3, "m": 3, "target_rr": 0.05},
        "D1": {"tau": 6, "m": 7, "target_rr": 0.05},
        "D2": {"tau": 4, "m": 7, "target_rr": 0.05},
        "D3": {"tau": 2, "m": 7, "target_rr": 0.05},
        "D4": {"tau": 2, "m": 7, "target_rr": 0.05},
        "A4": {"tau": 16, "m": 8, "target_rr": 0.05},
    }
    # subband_params = {
    #     "D1": {"tau": 3, "m": 3, "target_rr": 0.05},
    #     "D2": {"tau": 3, "m": 3, "target_rr": 0.05},
    #     "D3": {"tau":3 , "m": 3, "target_rr": 0.05},
    #     "D4": {"tau":3, "m": 3, "target_rr": 0.05},
    #     "A4": {"tau": 3, "m": 3, "target_rr": 0.05},
    # }
    viz = PAFVisualizer(subband_params=subband_params)
    # indices = viz.get_random_indices(n_normal=1, n_preaf=1)
    # idx_n = indices['normal'][0]
    # idx_p = indices['preaf'][0]

    #optimal tau = 12, m = 7
    time_delay = 3
    embedding_dim = 3
    epsilon_percent = 0.1
    indices = viz.get_random_indices(n_normal=1, n_preaf=1)
    idx_n = indices['normal'][0]
    idx_p = indices['preaf'][0]
    # idx_n = 293
    # idx_p = 9693

    # viz.quick_compare(idx_n, idx_p, time_delay=time_delay, embedding_dim=embedding_dim, embedding_percent=epsilon_percent)

    # id=6486 - pure noise


    #TODO observation : normal patients in the p* group have a more recurrent D4 subband compared to normal patients from the non-af n group

    #TODO observation 2: there seems to be more recurrent states in the D4 subband (4-8Hz) for pre_AF signals - but it is similar for chaotic normal signals

    for i in range(0, 2):
        indices = viz.get_random_indices(n_normal=1, n_preaf=1)
        idx_n = indices['normal'][0]
        idx_p = indices['preaf'][0]
        viz.investigate(idx_p, epsilon_method="max_distance", epsilon_factor=0.05, show_unthresholded=True)