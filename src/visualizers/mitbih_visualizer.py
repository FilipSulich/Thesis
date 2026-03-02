"""
MIT-BIH Visualizer

Visualization tools for MIT-BIH dataset:
1. Pre/post cleaning comparison
2. Multi-scale decomposition (SWT subbands)
3. Recurrence plots (single-scale and multi-scale)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import h5py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wavelets.stationary_wavelet_transform as stationary_wavelet
from data_cleaning.data_cleaner import DataCleaner

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/mitbih')
RAW_DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/MIT')
OUTPUT_DIR = DATA_DIR / 'results'

SAMPLING_FREQ = 250
SUBBAND_NAMES = ["D1", "D2", "D3", "D4", "A4"]


# =============================================================================
# Visualizer Class
# =============================================================================

class MITBIHVisualizer:
    def __init__(
        self,
        subband_params: Optional[Dict[str, Dict[str, Any]]] = None,
        csv_path: Optional[Path] = None,
        lead_time: int = 5,
    ):
        """
        Initialize MIT-BIH visualizer.

        Args:
            subband_params: Parameters for each subband (tau, m, target_rr)
            csv_path: Path to CSV file with cleaned segments (contains both pre-AF and normal)
            lead_time: Lead time to load (default: 5)
        """
        self.lead_time = lead_time
        if csv_path is None:
            csv_path = OUTPUT_DIR / f'dataset_lt{lead_time}.csv'
        self.csv_path = csv_path
        self.subband_params = subband_params
        self.signals = None
        self.labels = None
        self.segment_ids = None
        self.patient_ids = None
        self.lead_times = None
        self._load_data()

        # Initialize data cleaner for visualization (but signals are already cleaned)
        self.cleaner = DataCleaner(
            fs=SAMPLING_FREQ,
            highpass_cutoff=0.5,
            swt_denoise=True,
            swt_level=4,
        )

    def _load_data(self):
        """Load data from CSV file."""
        import pandas as pd

        print(f"Loading data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)

        # Extract metadata
        self.segment_ids = df['segment_id'].values
        self.patient_ids = df['patient_id'].values
        self.labels = df['label'].values
        self.lead_times = df['lead_time'].values

        # Extract signals
        signal_cols = [c for c in df.columns if c.startswith('s') and c[1:].isdigit()]
        signal_cols = sorted(signal_cols, key=lambda x: int(x[1:]))
        self.signals = df[signal_cols].values.astype(np.float32)

        print(f"Loaded {len(self.signals)} signals from CSV")
        print(f"  Pre-AF: {sum(self.labels == 1)}, Normal: {sum(self.labels == 0)}")
        print(f"  Lead times: {sorted(np.unique(self.lead_times))}")
        print(f"  Patients: {len(np.unique(self.patient_ids))}")

    def get_signal(self, idx: int) -> Tuple[np.ndarray, int, str, str, int]:
        """Get signal by index. Returns (signal, label, segment_id, patient_id, lead_time)."""
        return (
            self.signals[idx],
            self.labels[idx],
            self.segment_ids[idx],
            self.patient_ids[idx],
            self.lead_times[idx]
        )

    def get_random_indices(self, lead_time: Optional[int] = None, label: Optional[int] = None, n: int = 3) -> List[int]:
        """
        Get random indices for signals.

        Args:
            lead_time: Filter by lead time (None = any)
            label: Filter by label (0=normal, 1=pre-AF, None=any)
            n: Number of indices to return
        """
        mask = np.ones(len(self.signals), dtype=bool)

        if lead_time is not None:
            mask = mask & (self.lead_times == lead_time)

        if label is not None:
            mask = mask & (self.labels == label)

        indices = np.where(mask)[0]
        if len(indices) == 0:
            print(f"No signals found with lead_time={lead_time}, label={label}")
            return []

        return np.random.choice(indices, min(n, len(indices)), replace=False).tolist()

    # =========================================================================
    # 1. Pre/Post Cleaning Visualization
    # =========================================================================

    def _apply_cleaning(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply full cleaning pipeline.

        NOTE: Signals loaded from CSV are already cleaned!
        This method is kept for compatibility but should not be needed.
        """
        # Signals from CSV are already cleaned, just return them
        return signal

    def plot_cleaning_comparison(
        self,
        idx: int,
        wavelet: str = 'db4',
        save_path: Optional[Path] = None
    ):
        """
        Plot pre-cleaning vs post-cleaning signal comparison.

        Shows:
        - Raw signal
        - After basic filtering
        - After SWT denoising
        - After normalization (final)
        """
        signal, label, seg_id, pat_id, lead_time = self.get_signal(idx)

        label_str = f'Pre-AF ({lead_time} min)' if label == 1 else 'Normal'
        color = 'red' if label == 1 else 'blue'

        # Processing steps
        raw = signal.copy()

        # Step 1: Basic filtering
        filtered = raw - np.mean(raw)
        filtered = self.cleaner.highpass_filter(filtered)
        filtered = self.cleaner.lowpass_filter(filtered)

        # Step 2: SWT denoising (may change length slightly)
        denoised = self.cleaner.swt_adaptive_denoise(filtered, wavelet=wavelet)

        # Step 3: Normalization
        normalized = self.cleaner.normalize(denoised)

        # Create time arrays for each signal (they may have different lengths)
        t_raw = np.arange(len(raw)) / SAMPLING_FREQ
        t_filtered = np.arange(len(filtered)) / SAMPLING_FREQ
        t_denoised = np.arange(len(denoised)) / SAMPLING_FREQ
        t_normalized = np.arange(len(normalized)) / SAMPLING_FREQ

        # Plot
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))

        # Raw signal
        axes[0].plot(t_raw, raw, color='gray', linewidth=0.7, alpha=0.8)
        axes[0].set_ylabel('Raw Signal')
        axes[0].set_title(f'{label_str} | {seg_id} | Patient: {pat_id}')
        axes[0].grid(True, alpha=0.3)
        axes[0].text(0.98, 0.95, f'std={np.std(raw):.2f}',
                     transform=axes[0].transAxes, ha='right', va='top', fontsize=9)

        # After basic filtering
        axes[1].plot(t_filtered, filtered, color='orange', linewidth=0.7)
        axes[1].set_ylabel('After Filtering\n(HP + LP)')
        axes[1].grid(True, alpha=0.3)
        axes[1].text(0.98, 0.95, f'std={np.std(filtered):.2f}',
                     transform=axes[1].transAxes, ha='right', va='top', fontsize=9)

        # After SWT denoising
        axes[2].plot(t_denoised, denoised, color='green', linewidth=0.7)
        axes[2].set_ylabel('After SWT\nDenoising')
        axes[2].grid(True, alpha=0.3)
        axes[2].text(0.98, 0.95, f'std={np.std(denoised):.2f}',
                     transform=axes[2].transAxes, ha='right', va='top', fontsize=9)

        # After normalization
        axes[3].plot(t_normalized, normalized, color=color, linewidth=0.7)
        axes[3].set_ylabel('Final\n(Normalized)')
        axes[3].set_xlabel('Time (s)')
        axes[3].grid(True, alpha=0.3)
        axes[3].text(0.98, 0.95, f'mean={np.mean(normalized):.4f}, std={np.std(normalized):.4f}',
                     transform=axes[3].transAxes, ha='right', va='top', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
            plt.close()
        else:
            plt.show()

    # =========================================================================
    # 2. Multi-Scale Decomposition
    # =========================================================================

    def decompose_signal(self, signal: np.ndarray, normalize_per_subband: bool = True) -> Dict[str, np.ndarray]:
        """
        Decompose signal into SWT subbands.

        Args:
            signal: Input signal (already cleaned and normalized from dataset)
            normalize_per_subband: If True, z-score normalize each subband (for multi-scale RQA)

        Returns:
            Dictionary of subbands
        """
        swt = stationary_wavelet.SWT(wavelet='db4', level=4)
        coeffs = swt.decompose(signal, level=4)

        reconstructions = {}

        # Detail coefficients D1-D4
        for i in range(len(coeffs)):
            zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
            subband = swt.reconstruct(zeroed)

            if normalize_per_subband:
                subband = (subband - np.mean(subband)) / (np.std(subband) + 1e-8)

            reconstructions[f"D{i+1}"] = subband

        # Approximation A4
        zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        subband = swt.reconstruct(zeroed)

        if normalize_per_subband:
            subband = (subband - np.mean(subband)) / (np.std(subband) + 1e-8)

        reconstructions["A4"] = subband

        return reconstructions

    def plot_subbands(
        self,
        idx: int,
        apply_cleaning: bool = True,
        normalize_per_subband: bool = False,
        save_path: Optional[Path] = None
    ):
        """
        Plot signal and all its subbands.

        Args:
            idx: Signal index
            apply_cleaning: Whether to apply cleaning before decomposition
            normalize_per_subband: Whether to z-score normalize each subband (for multi-scale)
            save_path: Optional save path
        """
        signal, label, seg_id, pat_id, lead_time = self.get_signal(idx)

        # Apply cleaning if requested (gives signal ready for single-scale)
        if apply_cleaning:
            signal = self._apply_cleaning(signal)

        # Decompose (optionally with per-subband normalization for multi-scale)
        subbands = self.decompose_signal(signal, normalize_per_subband=normalize_per_subband)

        t = np.arange(len(signal)) / SAMPLING_FREQ
        label_str = f'Pre-AF ({lead_time} min)' if label == 1 else 'Normal'
        color = 'red' if label == 1 else 'blue'

        fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)

        # Original signal
        axes[0].plot(t, signal, color=color, linewidth=0.7)
        axes[0].set_ylabel('Original')
        axes[0].set_title(f'{label_str} | {seg_id} | Patient: {pat_id}')
        axes[0].grid(True, alpha=0.3)

        # Subbands
        for i, name in enumerate(SUBBAND_NAMES):
            sub = subbands[name]
            axes[i+1].plot(t[:len(sub)], sub, color=color, linewidth=0.7)
            axes[i+1].set_ylabel(name)
            axes[i+1].grid(True, alpha=0.3)

            # Show stats
            std = np.std(sub)
            axes[i+1].text(0.98, 0.95, f'std={std:.4f}',
                          transform=axes[i+1].transAxes,
                          ha='right', va='top', fontsize=9, family='monospace')

        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
            plt.close()
        else:
            plt.show()

    # =========================================================================
    # 3. Recurrence Plot Visualization
    # =========================================================================

    def compute_rp(
        self,
        signal: np.ndarray,
        time_delay: int = 12,
        embedding_dim: int = 7,
        epsilon_method: str = 'std',
        epsilon_factor: float = 0.2
    ) -> np.ndarray:
        """Compute recurrence plot."""
        from pyrqa.time_series import TimeSeries
        from pyrqa.settings import Settings
        from pyrqa.analysis_type import Classic
        from pyrqa.neighbourhood import FixedRadius
        from pyrqa.metric import EuclideanMetric
        from pyrqa.computation import RPComputation

        # Compute epsilon
        if epsilon_method == 'std':
            epsilon = epsilon_factor * np.std(signal)
        elif epsilon_method == 'max_distance':
            n = len(signal) - (embedding_dim - 1) * time_delay
            embedded = np.array([signal[i:i + embedding_dim * time_delay:time_delay] for i in range(n)])
            from scipy.spatial.distance import pdist
            distances = pdist(embedded, metric='euclidean')
            epsilon = epsilon_factor * np.max(distances)
        else:
            epsilon = epsilon_factor * np.std(signal)

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
        """Compute unthresholded recurrence plot (distance matrix)."""
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


    def plot_subband_rps(
        self,
        idx: Optional[int] = None,
        label_filter: Optional[int] = None,
        save_path: Optional[Path] = None
    ):
        """
        Plot signal and its RP, plus all subbands with their RPs.

        Simplified version with:
        - Original signal + RP in first row
        - Each subband + RP in subsequent rows
        - Option to filter by label (0=normal, 1=pre-AF)

        Args:
            idx: Specific signal index (if None, random from label_filter)
            label_filter: Filter by label: 0=normal, 1=pre-AF, None=any
            save_path: Optional save path
        """
        # Select signal
        if idx is None:
            # Filter by label
            mask = np.ones(len(self.signals), dtype=bool)
            if label_filter is not None:
                mask = mask & (self.labels == label_filter)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                print(f"No signals found with label={label_filter}")
                return
            idx = np.random.choice(indices)

        signal, label, seg_id, pat_id, lead_time = self.get_signal(idx)

        # Signals from CSV are already cleaned, so just decompose
        # Decompose with per-subband normalization (for multi-scale RQA)
        subbands = self.decompose_signal(signal, normalize_per_subband=True)

        label_str = f'Pre-AF ({lead_time} min)' if label == 1 else 'Normal'
        color = 'red' if label == 1 else 'blue'

        # Create figure: 6 rows (original + 5 subbands), 2 columns (signal + RP)
        fig, axes = plt.subplots(6, 2, figsize=(12, 16))

        # Get default parameters
        default_params = {'tau': 12, 'm': 7}
        if self.subband_params:
            default_params = self.subband_params.get('original', default_params)

        # Row 0: Original signal + RP
        t_orig = np.arange(len(signal)) / SAMPLING_FREQ
        axes[0, 0].plot(t_orig, signal, color=color, linewidth=0.7)
        axes[0, 0].set_ylabel('Original Signal')
        axes[0, 0].set_title(f'{label_str} | {seg_id} | Patient: {pat_id}')
        axes[0, 0].grid(True, alpha=0.3)

        # Compute RP for original
        tau_orig = default_params.get('tau', 12)
        m_orig = default_params.get('m', 7)
        rp_orig = self.compute_rp(signal, tau_orig, m_orig, 'std', 0.2)
        rr_orig = np.sum(rp_orig) / (rp_orig.shape[0] * rp_orig.shape[1])

        axes[0, 1].imshow(rp_orig, cmap='binary', origin='lower', aspect='equal')
        axes[0, 1].set_title(f'Original RP | tau={tau_orig}, m={m_orig}, RR={rr_orig:.3f}')
        axes[0, 1].set_xlabel('i')
        axes[0, 1].set_ylabel('j')

        # Rows 1-5: Subbands + RPs
        for i, name in enumerate(SUBBAND_NAMES):
            row_idx = i + 1
            sub = subbands[name]

            # Get parameters for this subband
            if self.subband_params and name in self.subband_params:
                params = self.subband_params[name]
                tau = params.get('tau', 3)
                m = params.get('m', 3)
            else:
                tau = 3
                m = 3

            t = np.arange(len(sub)) / SAMPLING_FREQ

            # Signal
            axes[row_idx, 0].plot(t, sub, color=color, linewidth=0.7)
            axes[row_idx, 0].set_ylabel(f'{name}')
            axes[row_idx, 0].grid(True, alpha=0.3)

            # RP
            rp = self.compute_rp(sub, tau, m, 'std', 0.2)
            rr = np.sum(rp) / (rp.shape[0] * rp.shape[1])
            axes[row_idx, 1].imshow(rp, cmap='binary', origin='lower', aspect='equal')
            axes[row_idx, 1].set_title(f'{name} RP | tau={tau}, m={m}, RR={rr:.3f}')
            axes[row_idx, 1].set_xlabel('i')
            axes[row_idx, 1].set_ylabel('j')

        axes[-1, 0].set_xlabel('Time (s)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def investigate(self, idx: int):
        """Complete investigation of a signal: cleaning, decomposition, and RPs."""
        print("\n1. Cleaning Comparison:")
        self.plot_cleaning_comparison(idx)

        print("\n2. Subband Decomposition:")
        self.plot_subbands(idx, apply_cleaning=True)

        print("\n3. Multi-Scale RPs:")
        self.plot_subband_rps(idx, apply_cleaning=True)


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    # Subband parameters for RQA
    subband_params = {
        "original": {"tau": 21, "m": 7, "target_rr": 0.05},
        "D1": {"tau": 6, "m": 7, "target_rr": 0.05},
        "D2": {"tau": 4, "m": 7, "target_rr": 0.05},
        "D3": {"tau": 2, "m": 7, "target_rr": 0.05},
        "D4": {"tau": 2, "m": 7, "target_rr": 0.05},
        "A4": {"tau": 16, "m": 8, "target_rr": 0.05},
    }
    wavelet = "db4"

    # Initialize visualizer
    viz = MITBIHVisualizer(subband_params=subband_params)

    # Output directory
    output_dir = DATA_DIR / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("Example 1: Plot random pre-AF signal with subbands and RPs")
    print("="*80)
    viz.plot_subband_rps(
        label_filter=1,  # Pre-AF only
        save_path=output_dir / 'preaf_subbands_rps.png'
    )

    viz.plot_subband_rps(
        label_filter=0,  # Normal only
        save_path=output_dir / 'normal_subbands_rps.png'
    )

    indices = viz.get_random_indices(lead_time=5, n=1)
    idx = indices[0]
    viz.plot_cleaning_comparison(idx, save_path=output_dir / f'cleaning_comparison_idx{idx}.png')

    print(f"\nAll visualizations saved to: {output_dir}")
