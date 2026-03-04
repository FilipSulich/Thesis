"""
Temporal Window Visualizer

Visualizes chronological progression of recurrence plots for PAF temporal windows.
Supports both single-scale and multi-scale (subband) analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_cleaning.data_cleaner import DataCleaner
import wavelets.stationary_wavelet_transform as stationary_wavelet


class TemporalWindowVisualizer:
    """
    Visualize temporal progression of recurrence plots for PAF windows.

    Supports:
    - Chronological RP progression (thresholded)
    - Unthresholded distance matrices
    - Single-scale and multi-scale (subband) analysis
    """

    def __init__(
        self,
        hdf5_path: Path,
        sampling_freq: int = 128,
        wavelet: str = 'db4',
        swt_level: int = 4
    ):
        self.hdf5_path = hdf5_path
        self.sampling_freq = sampling_freq
        self.wavelet = wavelet
        self.swt_level = swt_level
        self.cleaner = DataCleaner(fs=sampling_freq)

        # Data containers
        self.signals = None
        self.labels = None
        self.window_ids = None
        self.patient_ids = None
        self.window_indices = None
        self.subbands = {}

        self._load_data()

    def _load_data(self):
        """Load temporal window data from HDF5."""
        with h5py.File(self.hdf5_path, 'r') as f:
            grp = f['temporal_windows']
            self.signals = grp['signals'][:]
            self.labels = grp['labels'][:]
            self.window_ids = [s.decode() if isinstance(s, bytes) else s for s in grp['window_ids'][:]]
            self.patient_ids = [s.decode() if isinstance(s, bytes) else s for s in grp['patient_ids'][:]]
            self.window_indices = grp['window_indices'][:]

            # Load subbands if available
            subband_names = ['D1', 'D2', 'D3', 'D4', 'A4']
            for sb_name in subband_names:
                key = f'subband_{sb_name}'
                if key in grp:
                    self.subbands[sb_name] = grp[key][:]

        print(f"Loaded {len(self.signals)} temporal windows from {self.hdf5_path}")
        print(f"  Normal: {sum(self.labels == 0)}, Pre-AF: {sum(self.labels == 1)}")
        if self.subbands:
            print(f"  Subbands available: {list(self.subbands.keys())}")

    def get_patient_windows(self, patient_id: str, label: int) -> List[int]:
        """
        Get indices of all windows for a specific patient and label,
        sorted chronologically by window_index.
        """
        indices = []
        for i, (pid, lbl, widx) in enumerate(zip(self.patient_ids, self.labels, self.window_indices)):
            if pid == patient_id and lbl == label:
                indices.append((i, widx))

        # Sort by window_index to maintain chronological order
        indices = sorted(indices, key=lambda x: x[1])
        return [idx for idx, _ in indices]

    def compute_rp(
        self,
        signal: np.ndarray,
        time_delay: int,
        embedding_dim: int,
        epsilon_method: str = 'percentile',
        epsilon_factor: float = 0.05,
    ) -> Tuple[np.ndarray, float]:
        """Compute thresholded recurrence plot."""
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

        # Subsample for epsilon calculation if too large
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
        """Compute unthresholded distance matrix."""
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
        return result.recurrence_matrix_reverse

    def plot_temporal_progression_single_scale(
        self,
        patient_id: str,
        time_delay: int = 1,
        embedding_dim: int = 10,
        epsilon_method: str = 'percentile',
        epsilon_factor: float = 0.05,
        max_windows: Optional[int] = None,
        save_path: Optional[Path] = None,
        show_unthresholded: bool = False
    ):
        """
        Plot chronological progression of RPs for single-scale signal.

        Parameters
        ----------
        patient_id : str
            Patient ID (e.g., 'paf_p01')
        time_delay : int
            Time delay for phase space reconstruction
        embedding_dim : int
            Embedding dimension
        epsilon_method : str
            Method for epsilon calculation ('percentile', 'max_distance', 'std')
        epsilon_factor : float
            Factor for epsilon calculation
        max_windows : int, optional
            Maximum number of windows to plot (useful for limiting subplot count)
        save_path : Path, optional
            Path to save figure
        show_unthresholded : bool
            If True, show unthresholded distance matrices below RPs
        """
        # Get pre-AF windows in chronological order
        preaf_indices = self.get_patient_windows(patient_id, label=1)

        if not preaf_indices:
            print(f"No pre-AF windows found for patient {patient_id}")
            return

        if max_windows:
            preaf_indices = preaf_indices[:max_windows]

        n_windows = len(preaf_indices)
        n_rows = 2 if show_unthresholded else 1

        fig, axes = plt.subplots(n_rows, n_windows, figsize=(3 * n_windows, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_windows == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f'Temporal Progression - {patient_id} (Single Scale)\n'
                     f'τ={time_delay}, m={embedding_dim}, ε={epsilon_method}({epsilon_factor})',
                     fontsize=14, y=0.98)

        for col_idx, window_idx in enumerate(preaf_indices):
            signal = self.signals[window_idx]
            window_pos = self.window_indices[window_idx]

            # Compute thresholded RP
            rp, epsilon = self.compute_rp(signal, time_delay, embedding_dim,
                                         epsilon_method, epsilon_factor)

            # Plot thresholded RP
            ax = axes[0, col_idx]
            ax.imshow(rp, cmap='binary', origin='lower', interpolation='nearest')
            ax.set_title(f'Window {window_pos}\nε={epsilon:.3f}', fontsize=10)
            ax.set_xlabel('Time')
            ax.set_ylabel('Time')
            ax.set_xticks([])
            ax.set_yticks([])

            # Plot unthresholded distance matrix if requested
            if show_unthresholded:
                dist_matrix = self.compute_rp_unthresholded(signal, time_delay, embedding_dim)
                ax_unthresh = axes[1, col_idx]
                im = ax_unthresh.imshow(dist_matrix, cmap='viridis', origin='lower',
                                       interpolation='nearest')
                ax_unthresh.set_title('Unthresholded', fontsize=10)
                ax_unthresh.set_xlabel('Time')
                ax_unthresh.set_ylabel('Time')
                ax_unthresh.set_xticks([])
                ax_unthresh.set_yticks([])
                plt.colorbar(im, ax=ax_unthresh, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_temporal_progression_multi_scale(
        self,
        patient_id: str,
        subband_name: str,
        time_delay: int = 1,
        embedding_dim: int = 10,
        epsilon_method: str = 'percentile',
        epsilon_factor: float = 0.05,
        max_windows: Optional[int] = None,
        save_path: Optional[Path] = None,
        show_unthresholded: bool = False
    ):
        """
        Plot chronological progression of RPs for a specific subband.

        Parameters
        ----------
        patient_id : str
            Patient ID (e.g., 'paf_p01')
        subband_name : str
            Subband name (e.g., 'D1', 'D2', 'D3', 'D4', 'A4')
        time_delay : int
            Time delay for phase space reconstruction
        embedding_dim : int
            Embedding dimension
        epsilon_method : str
            Method for epsilon calculation
        epsilon_factor : float
            Factor for epsilon calculation
        max_windows : int, optional
            Maximum number of windows to plot
        save_path : Path, optional
            Path to save figure
        show_unthresholded : bool
            If True, show unthresholded distance matrices below RPs
        """
        if subband_name not in self.subbands:
            print(f"Subband {subband_name} not available. Available: {list(self.subbands.keys())}")
            return

        # Get pre-AF windows in chronological order
        preaf_indices = self.get_patient_windows(patient_id, label=1)

        if not preaf_indices:
            print(f"No pre-AF windows found for patient {patient_id}")
            return

        if max_windows:
            preaf_indices = preaf_indices[:max_windows]

        n_windows = len(preaf_indices)
        n_rows = 2 if show_unthresholded else 1

        fig, axes = plt.subplots(n_rows, n_windows, figsize=(3 * n_windows, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_windows == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f'Temporal Progression - {patient_id} ({subband_name})\n'
                     f'τ={time_delay}, m={embedding_dim}, ε={epsilon_method}({epsilon_factor})',
                     fontsize=14, y=0.98)

        subband_signals = self.subbands[subband_name]

        for col_idx, window_idx in enumerate(preaf_indices):
            signal = subband_signals[window_idx]
            window_pos = self.window_indices[window_idx]

            # Compute thresholded RP
            rp, epsilon = self.compute_rp(signal, time_delay, embedding_dim,
                                         epsilon_method, epsilon_factor)

            # Plot thresholded RP
            ax = axes[0, col_idx]
            ax.imshow(rp, cmap='binary', origin='lower', interpolation='nearest')
            ax.set_title(f'Window {window_pos}\nε={epsilon:.3f}', fontsize=10)
            ax.set_xlabel('Time')
            ax.set_ylabel('Time')
            ax.set_xticks([])
            ax.set_yticks([])

            # Plot unthresholded distance matrix if requested
            if show_unthresholded:
                dist_matrix = self.compute_rp_unthresholded(signal, time_delay, embedding_dim)
                ax_unthresh = axes[1, col_idx]
                im = ax_unthresh.imshow(dist_matrix, cmap='viridis', origin='lower',
                                       interpolation='nearest')
                ax_unthresh.set_title('Unthresholded', fontsize=10)
                ax_unthresh.set_xlabel('Time')
                ax_unthresh.set_ylabel('Time')
                ax_unthresh.set_xticks([])
                ax_unthresh.set_yticks([])
                plt.colorbar(im, ax=ax_unthresh, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_all_subbands_progression(
        self,
        patient_id: str,
        time_delay: int = 1,
        embedding_dim: int = 10,
        epsilon_method: str = 'percentile',
        epsilon_factor: float = 0.05,
        max_windows: Optional[int] = None,
        save_path: Optional[Path] = None
    ):
        """
        Plot progression across all subbands in a single figure.
        Rows = subbands, Columns = time windows.

        Parameters
        ----------
        patient_id : str
            Patient ID (e.g., 'paf_p01')
        time_delay : int
            Time delay for phase space reconstruction
        embedding_dim : int
            Embedding dimension
        epsilon_method : str
            Method for epsilon calculation
        epsilon_factor : float
            Factor for epsilon calculation
        max_windows : int, optional
            Maximum number of windows to plot
        save_path : Path, optional
            Path to save figure
        """
        if not self.subbands:
            print("No subbands available in the dataset")
            return

        # Get pre-AF windows in chronological order
        preaf_indices = self.get_patient_windows(patient_id, label=1)

        if not preaf_indices:
            print(f"No pre-AF windows found for patient {patient_id}")
            return

        if max_windows:
            preaf_indices = preaf_indices[:max_windows]

        n_windows = len(preaf_indices)
        subband_names = ['D1', 'D2', 'D3', 'D4', 'A4']
        available_subbands = [sb for sb in subband_names if sb in self.subbands]
        n_subbands = len(available_subbands)

        fig, axes = plt.subplots(n_subbands, n_windows,
                                figsize=(2.5 * n_windows, 2.5 * n_subbands))

        if n_subbands == 1:
            axes = axes.reshape(1, -1)
        elif n_windows == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f'Multi-Scale Temporal Progression - {patient_id}\n'
                     f'τ={time_delay}, m={embedding_dim}, ε={epsilon_method}({epsilon_factor})',
                     fontsize=14, y=0.99)

        for row_idx, subband_name in enumerate(available_subbands):
            subband_signals = self.subbands[subband_name]

            for col_idx, window_idx in enumerate(preaf_indices):
                signal = subband_signals[window_idx]
                window_pos = self.window_indices[window_idx]

                # Compute thresholded RP
                rp, epsilon = self.compute_rp(signal, time_delay, embedding_dim,
                                             epsilon_method, epsilon_factor)

                ax = axes[row_idx, col_idx]
                ax.imshow(rp, cmap='binary', origin='lower', interpolation='nearest')

                # Title for first row (window index)
                if row_idx == 0:
                    ax.set_title(f'W{window_pos}', fontsize=10)

                # Y-label for first column (subband name)
                if col_idx == 0:
                    ax.set_ylabel(subband_name, fontsize=10, fontweight='bold')

                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_12_window_progression(
        self,
        patient_id: str,
        time_delay: int = 1,
        embedding_dim: int = 10,
        epsilon_method: str = 'percentile',
        epsilon_factor: float = 0.05,
        thresholded: bool = True,
        scale: str = 'single',
        subband_name: Optional[str] = None,
        save_path: Optional[Path] = None
    ):
        """
        Plot 12 windows in chronological order: 6 pre-AF followed by 6 normal.

        Parameters
        ----------
        patient_id : str
            Patient ID
        time_delay : int
            Time delay for phase space reconstruction
        embedding_dim : int
            Embedding dimension
        epsilon_method : str
            Epsilon calculation method
        epsilon_factor : float
            Epsilon factor
        thresholded : bool
            If True, plot thresholded RPs; if False, plot unthresholded distance matrices
        scale : str
            'single' for single-scale or 'multi' for multi-scale
        subband_name : str, optional
            Subband name if scale='multi' (e.g., 'D1', 'D2', 'D3', 'D4', 'A4')
        save_path : Path, optional
            Path to save figure
        """
        # Get windows
        preaf_indices = self.get_patient_windows(patient_id, label=1)
        normal_indices = self.get_patient_windows(patient_id, label=0)

        if len(preaf_indices) < 6 or len(normal_indices) < 6:
            print(f"Not enough windows for patient {patient_id}")
            print(f"  Pre-AF: {len(preaf_indices)}, Normal: {len(normal_indices)}")
            return

        # Take first 6 of each
        preaf_indices = preaf_indices[:6]
        normal_indices = normal_indices[:6]
        all_indices = preaf_indices + normal_indices
        all_labels = ['Pre-AF'] * 6 + ['Normal'] * 6

        fig, axes = plt.subplots(2, 6, figsize=(18, 6))

        plot_type = 'Thresholded RP' if thresholded else 'Unthresholded Distance Matrix'
        scale_str = f' ({subband_name})' if scale == 'multi' and subband_name else ''
        fig.suptitle(f'{patient_id} - Temporal Progression{scale_str}\n{plot_type}, τ={time_delay}, m={embedding_dim}',
                     fontsize=14, y=0.98)

        for idx, (window_idx, label_str) in enumerate(zip(all_indices, all_labels)):
            row = idx // 6
            col = idx % 6

            # Get signal
            if scale == 'multi' and subband_name:
                if subband_name not in self.subbands:
                    print(f"Subband {subband_name} not available")
                    return
                signal = self.subbands[subband_name][window_idx]
            else:
                signal = self.signals[window_idx]

            # Compute RP or distance matrix
            if thresholded:
                matrix, epsilon = self.compute_rp(signal, time_delay, embedding_dim,
                                                 epsilon_method, epsilon_factor)
                cmap = 'binary'
                title_suffix = f'ε={epsilon:.3f}'
            else:
                matrix = self.compute_rp_unthresholded(signal, time_delay, embedding_dim)
                cmap = 'viridis'
                title_suffix = ''

            ax = axes[row, col]
            im = ax.imshow(matrix, cmap=cmap, origin='lower', interpolation='nearest')

            window_pos = self.window_indices[window_idx]
            if col == 0:
                ax.set_ylabel(label_str, fontsize=12, fontweight='bold')
            ax.set_title(f'W{window_pos}\n{title_suffix}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

            if not thresholded:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_single_comparison_multi_scale(
        self,
        patient_id: str,
        time_delay: int = 1,
        embedding_dim: int = 10,
        epsilon_method: str = 'percentile',
        epsilon_factor: float = 0.05,
        save_path: Optional[Path] = None
    ):
        """
        Multi-scale comparison: one pre-AF window vs one normal window across all subbands.

        Layout: N_subbands rows × 2 cols (pre-AF, normal)
        Shows only thresholded RPs.
        """
        if not self.subbands:
            print("No subbands available")
            return

        # Get one pre-AF and one normal window
        preaf_indices = self.get_patient_windows(patient_id, label=1)
        normal_indices = self.get_patient_windows(patient_id, label=0)

        if not preaf_indices or not normal_indices:
            print(f"Not enough windows for patient {patient_id}")
            return

        preaf_idx = preaf_indices[0]
        normal_idx = normal_indices[0]

        subband_names = ['D1', 'D2', 'D3', 'D4', 'A4']
        available_subbands = [sb for sb in subband_names if sb in self.subbands]
        n_subbands = len(available_subbands)

        fig, axes = plt.subplots(n_subbands, 2, figsize=(8, 3 * n_subbands))
        if n_subbands == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'{patient_id} - Multi-Scale Comparison\nτ={time_delay}, m={embedding_dim}',
                     fontsize=14, y=0.99)

        for row_idx, subband_name in enumerate(available_subbands):
            subband_signals = self.subbands[subband_name]

            for col_idx, (window_idx, label_str) in enumerate([(preaf_idx, 'Pre-AF'), (normal_idx, 'Normal')]):
                signal = subband_signals[window_idx]

                # Compute and plot RP
                rp, epsilon = self.compute_rp(signal, time_delay, embedding_dim,
                                             epsilon_method, epsilon_factor)

                ax = axes[row_idx, col_idx]
                ax.imshow(rp, cmap='binary', origin='lower', interpolation='nearest')

                # Title for first row
                if row_idx == 0:
                    ax.set_title(f'{label_str}', fontsize=12)

                # Y-label for first column (subband name)
                if col_idx == 0:
                    ax.set_ylabel(f'{subband_name}\nε={epsilon:.3f}', fontsize=10, fontweight='bold')
                else:
                    ax.text(1.02, 0.5, f'ε={epsilon:.3f}', transform=ax.transAxes,
                           fontsize=9, va='center')

                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':
    # Example usage
    DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf_updated')
    HDF5_PATH = DATA_DIR / 'paf_temporal_windows.h5'
    OUTPUT_DIR = DATA_DIR / 'visualizations'
    OUTPUT_DIR.mkdir(exist_ok=True)

    viz = TemporalWindowVisualizer(HDF5_PATH)

    unique_patients = sorted(set(viz.patient_ids))
    if unique_patients:
        patient_id = unique_patients[0]
        print(f"\nVisualizing patient: {patient_id}")

        print("Generating single-scale temporal progression (thresholded)...")
        viz.plot_12_window_progression(
            patient_id=patient_id,
            time_delay=12,
            embedding_dim=7,
            epsilon_method='percentile',
            epsilon_factor=0.05,
            thresholded=False,
            scale='single',
            save_path=OUTPUT_DIR / f'{patient_id}_12win_single_thresholded.png'
        )

        # Single-scale: 12 windows unthresholded
        # print("Generating single-scale temporal progression (unthresholded)...")
        # viz.plot_12_window_progression(
        #     patient_id=patient_id,
        #     time_delay=1,
        #     embedding_dim=10,
        #     thresholded=False,
        #     scale='single',
        #     save_path=OUTPUT_DIR / f'{patient_id}_12win_single_unthresholded.png'
        # )
        #
        # # Multi-scale: 12 windows for each subband
        # if viz.subbands:
        #     for subband in ['D1', 'D2', 'D3', 'D4', 'A4']:
        #         if subband in viz.subbands:
        #             print(f"Generating multi-scale progression for {subband}...")
        #             viz.plot_12_window_progression(
        #                 patient_id=patient_id,
        #                 time_delay=12,
        #                 embedding_dim=7,
        #                 epsilon_method='percentile',
        #                 epsilon_factor=0.05,
        #                 thresholded=True,
        #                 scale='multi',
        #                 subband_name=subband,
        #                 save_path=OUTPUT_DIR / f'{patient_id}_12win_{subband}_thresholded.png'
        #             )

        print(f"\nAll visualizations saved to {OUTPUT_DIR}")
