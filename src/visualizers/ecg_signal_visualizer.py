from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt
import wfdb
from scipy import signal as sp_signal

# Local dependency - ensure this is in your python path
from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

# Constants matching dataset_generator.py
FS_ORIG = 128
FS_TARGET = 500
SEGMENT_SAMPLES = FS_TARGET * 10  # 5000 samples = 10s
UTRP_SIZE = 256
PAD_SAMPLES = 8192

WAVELET = 'db8'
SUBBAND_LEVELS = [2, 3, 4, 5, 6, 7, 8]
N_LEVELS_SWT = 10


def _minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1]."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


def _pad(x: np.ndarray, pad_samples: int = PAD_SAMPLES) -> np.ndarray:
    """Pad signal to nearest power of 2 for SWT compatibility."""
    pad = pad_samples - len(x)
    if pad <= 0:
        return x[:pad_samples]
    return np.pad(x, (0, pad), mode='reflect')


class ECGSignalVisualizer:
    """
    Visualizer for ECG signals and Unthresholded Recurrence Plots (UTRP).
    """

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            # Default path preserved from original logic
            data_dir = Path('/data_old/paf')
        self.data_dir = Path(data_dir)

    def load_signal(self, record_id: str) -> Optional[np.ndarray]:
        """Load and resample ECG signal to 500 Hz."""
        try:
            rec = wfdb.rdrecord(str(self.data_dir / record_id))
            raw = rec.p_signal[:, 0].astype(np.float64)
            # Resampling ratio logic preserved
            n = int(len(raw) * FS_TARGET / FS_ORIG)
            return sp_signal.resample(raw, n).astype(np.float32)
        except Exception as e:
            print(f"Error loading {record_id}: {e}")
            return None

    def extract_segment(self, signal: np.ndarray, segment_idx: int = 0) -> np.ndarray:
        """Extract a 10-second segment from signal using 1/14th step increments."""
        max_start = len(signal) - SEGMENT_SAMPLES
        if max_start <= 0:
            return signal[:SEGMENT_SAMPLES]
        # Logic branch preserved for segment indexing
        start = int(segment_idx * max_start / max(1, 14))
        return signal[start: start + SEGMENT_SAMPLES]

    def extract_subbands(self, signal_padded: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract detail subbands using Stationary Wavelet Transform (SWT)."""
        coeffs = pywt.swt(signal_padded, WAVELET, level=N_LEVELS_SWT)
        n_lvls = len(coeffs)
        subbands: Dict[str, np.ndarray] = {}

        for level in SUBBAND_LEVELS:
            target_idx = n_lvls - level
            # Reconstruct only the specific detail level (cD)
            zeroed = [
                ((np.zeros_like(cA), cD) if j == target_idx
                 else (np.zeros_like(cA), np.zeros_like(cD)))
                for j, (cA, cD) in enumerate(coeffs)
            ]
            sub = pywt.iswt(zeroed, WAVELET)[:SEGMENT_SAMPLES].astype(np.float32)
            subbands[f'D{level}'] = _minmax(sub)

        return subbands

    def compute_utrp(
            self,
            signal: np.ndarray,
            tau: int = 3,
            m: int = 3,
            size: int = UTRP_SIZE,
    ) -> Optional[np.ndarray]:
        """Generate a colorized Unthresholded Recurrence Plot."""
        try:
            sig = signal[:SEGMENT_SAMPLES].astype(np.float32)
            # Phase space check: N - (m-1)*tau
            if len(sig) - (m - 1) * tau <= 0:
                return np.zeros((3, size, size), dtype=np.float32)

            gen = RecurrencePlotGenerator(
                signal=sig, time_delay=tau, embedding_dim=m, epsilon=0
            )
            rp = gen.compute_rp_unthresholded()

            # Normalization and diagonal suppression
            d_max = rp.max()
            if d_max > 1e-9:
                rp /= d_max

            rp_resized = cv2.resize(rp.astype(np.float32), (size, size),
                                    interpolation=cv2.INTER_LINEAR)
            np.fill_diagonal(rp_resized, 0.0)

            # Apply JET colormap
            u8 = (rp_resized * 255).astype(np.uint8)
            bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return rgb.transpose(2, 0, 1)  # CHW format

        except Exception as e:
            print(f"Error computing UTRP: {e}")
            return None

    # Plotting methods (Implementation remains identical to your original structure)
    def plot_original_signal_and_utrp(self, signal, utrp, record_id, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        time = np.arange(len(signal)) / FS_TARGET
        ax1.plot(time, signal, 'b-', linewidth=1)
        ax1.set_title(f'ECG signal \n record: {record_id}', fontweight='bold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 10)

        utrp_img = utrp.transpose(1, 2, 0)
        ax2.imshow(utrp_img)
        ax2.set_title(f'UTRP \n (τ={3}, m={3})', fontweight='bold')
        # ax2.set_xticks([]);
        # ax2.set_yticks([])
        ax2.set_xlabel('Time Index (i)', fontsize=9)
        ax2.set_ylabel('Time Index (j)', fontsize=9)

        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show();
        plt.close()

    def plot_original_signal(self, signal, utrp, record_id, save_path=None):
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 4))
        time = np.arange(len(signal)) / FS_TARGET
        ax1.plot(time, signal, 'b-', linewidth=1)
        ax1.set_title(f'ECG signal \n record: {record_id}', fontweight='bold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 10)

        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show();
        plt.close()

    def plot_subband_signals(self, subbands, record_id, save_path=None):
        n_sub = len(SUBBAND_LEVELS)
        fig, axes = plt.subplots(n_sub, 1, figsize=(12, 10), sharex=True)
        time = np.arange(SEGMENT_SAMPLES) / FS_TARGET
        for idx, level in enumerate(SUBBAND_LEVELS):
            axes[idx].plot(time, subbands[f'D{level}'], color='darkgreen', linewidth=0.8)
            axes[idx].set_ylabel(f'D{level}', fontweight='bold')
            axes[idx].set_ylim(-0.1, 1.1)
            axes[idx].grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'Subband ECG Signals (D2-D8) - {record_id}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show();
        plt.close()

    def plot_subband_utrps(self, utrps, record_id, save_path=None):
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        axes = axes.flatten()
        for idx, level in enumerate(SUBBAND_LEVELS):
            ax = axes[idx]
            if f'D{level}' in utrps:
                ax.imshow(utrps[f'D{level}'].transpose(1, 2, 0))
            ax.set_title(f'D{level}', fontweight='bold')
            # ax.set_xticks([]);
            # ax.set_yticks([])
            ax.set_xlabel('Time Index (i)', fontsize=9)
            ax.set_ylabel('Time Index (j)', fontsize=9)
        axes[-1].set_visible(False)
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show();
        plt.close()

    def visualize_complete(self, record_id, segment_idx=0, tau=3, m=3, save_dir=None):
        print(f"\nProcessing: {record_id} (segment {segment_idx})")
        signal_full = self.load_signal(record_id)
        if signal_full is None: return

        segment = self.extract_segment(signal_full, segment_idx)
        subbands = self.extract_subbands(_pad(segment))

        utrp_original = self.compute_utrp(segment, tau, m)
        utrps = {name: self.compute_utrp(sig, tau, m) for name, sig in subbands.items()}

        paths = [None] * 3
        if save_dir:
            sd = Path(save_dir);
            sd.mkdir(parents=True, exist_ok=True)
            paths = [sd / f'{record_id}_seg{segment_idx}_{i}.png' for i in ['orig', 'sub', 'utrp']]

        self.plot_original_signal_and_utrp(segment, utrp_original, record_id, paths[0])
        self.plot_original_signal(segment, utrp_original, record_id, paths[0])
        self.plot_subband_signals(subbands, record_id, paths[1])
        self.plot_subband_utrps(utrps, record_id, paths[2])


if __name__ == '__main__':
    # Usage preserved for n31 and p01
    vis = ECGSignalVisualizer()
    for rid in ['n31', 'p34']:
        vis.visualize_complete(record_id=rid, save_dir='./results')