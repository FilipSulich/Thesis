"""
Visualizer class for comparing uncleaned and cleaned signals side by side.
Shows the complete cleaning pipeline step-by-step.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_cleaning.data_cleaner import DataCleaner


class CleaningVisualizer:
    """
    Visualizer for comparing uncleaned and cleaned ECG signals.
    Shows step-by-step cleaning pipeline transformations.
    """

    def __init__(self, fs=250):
        """
        Initialize the cleaning visualizer.

        Args:
            fs: Sampling frequency (Hz)
        """
        self.fs = fs
        self.cleaner = DataCleaner(fs=fs)

    def plot_cleaning_pipeline(self, signal, title_prefix="", save_path=None):
        """
        Plot the complete cleaning pipeline step-by-step.

        Args:
            signal: Raw input signal (numpy array)
            title_prefix: Prefix for plot titles (e.g., patient ID, label)
            save_path: Optional path to save the figure
        """
        time = np.arange(len(signal)) / self.fs

        # Step-by-step cleaning
        signal_original = signal.copy()
        signal_demeaned = signal - np.mean(signal)
        signal_highpass = self.cleaner.highpass_filter(signal_demeaned)
        signal_lowpass = self.cleaner.lowpass_filter(signal_highpass)
        signal_trimmed = self.cleaner.trim_to_center(signal_lowpass, target_seconds=10)

        # Quality check
        is_good, qc_info = self.cleaner.quality_check(signal_trimmed)

        signal_normalized = self.cleaner.normalize(signal_trimmed)

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25)

        # Original signal
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time, signal_original, color='black', linewidth=0.8, alpha=0.7)
        ax1.set_title(f'{title_prefix}Step 1: Original Signal', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Amplitude (mV)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.text(0.02, 0.95, f'Mean: {np.mean(signal_original):.3f}\nStd: {np.std(signal_original):.3f}',
                transform=ax1.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

        # Demeaned
        ax2 = fig.add_subplot(gs[1, 0])
        time_demean = np.arange(len(signal_demeaned)) / self.fs
        ax2.plot(time_demean, signal_demeaned, color='blue', linewidth=0.8, alpha=0.7)
        ax2.set_title('Step 2: Demeaned (Zero DC)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Amplitude', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2.text(0.02, 0.95, f'Mean: {np.mean(signal_demeaned):.3f}',
                transform=ax2.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

        # Highpass filtered
        ax3 = fig.add_subplot(gs[1, 1])
        time_hp = np.arange(len(signal_highpass)) / self.fs
        ax3.plot(time_hp, signal_highpass, color='green', linewidth=0.8, alpha=0.7)
        ax3.set_title(f'Step 3: Highpass Filter ({self.cleaner.highpass_cutoff} Hz)',
                     fontsize=11, fontweight='bold')
        ax3.set_ylabel('Amplitude', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.text(0.02, 0.95, 'Removes baseline\nwander',
                transform=ax3.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

        # Lowpass filtered
        ax4 = fig.add_subplot(gs[2, 0])
        time_lp = np.arange(len(signal_lowpass)) / self.fs
        ax4.plot(time_lp, signal_lowpass, color='purple', linewidth=0.8, alpha=0.7)
        ax4.set_title('Step 4: Lowpass Filter (40 Hz)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Amplitude', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.text(0.02, 0.95, 'Removes high-freq\nnoise',
                transform=ax4.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='plum', alpha=0.6))

        # Trimmed
        ax5 = fig.add_subplot(gs[2, 1])
        time_trim = np.arange(len(signal_trimmed)) / self.fs
        ax5.plot(time_trim, signal_trimmed, color='orange', linewidth=0.8, alpha=0.7)
        ax5.set_title('Step 5: Trimmed to Center 10s', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Amplitude', fontsize=10)
        ax5.set_xlabel('Time (s)', fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5.text(0.02, 0.95, f'Length: {len(signal_trimmed)} samples\n({len(signal_trimmed)/self.fs:.1f}s)',
                transform=ax5.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.6))

        # Quality check visualization
        ax6 = fig.add_subplot(gs[3, 0])
        ax6.plot(time_trim, signal_trimmed, color='gray', linewidth=0.8, alpha=0.5, label='Signal')

        # Show detected peaks if quality check passed
        if is_good:
            from scipy.signal import find_peaks
            signal_range = np.percentile(signal_trimmed, 95) - np.percentile(signal_trimmed, 5)
            height_threshold = signal_range * 0.4
            min_distance = int(0.3 * self.fs)
            peaks, _ = find_peaks(signal_trimmed, height=height_threshold, distance=min_distance)
            ax6.plot(time_trim[peaks], signal_trimmed[peaks], 'ro', markersize=6, label='Detected beats')
            qc_color = 'green'
            qc_status = 'PASSED'
        else:
            qc_color = 'red'
            qc_status = 'FAILED'

        ax6.set_title('Step 6: Quality Check', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Amplitude', fontsize=10)
        ax6.set_xlabel('Time (s)', fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc='upper right', fontsize=8)

        qc_text = f"Status: {qc_status}\n"
        if 'num_beats' in qc_info:
            qc_text += f"Beats: {qc_info['num_beats']}\n"
        if 'snr' in qc_info and qc_info['snr'] is not None:
            qc_text += f"SNR: {qc_info['snr']:.2f}\n"
        qc_text += f"Reason: {qc_info['reason']}"

        ax6.text(0.02, 0.95, qc_text,
                transform=ax6.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor=qc_color, alpha=0.3))

        # Final normalized signal
        ax7 = fig.add_subplot(gs[3, 1])
        time_norm = np.arange(len(signal_normalized)) / self.fs
        ax7.plot(time_norm, signal_normalized, color='darkblue', linewidth=0.8, alpha=0.8)
        ax7.set_title('Step 7: Z-Score Normalized', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Amplitude (normalized)', fontsize=10)
        ax7.set_xlabel('Time (s)', fontsize=10)
        ax7.grid(True, alpha=0.3)
        ax7.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax7.text(0.02, 0.95, f'Mean: {np.mean(signal_normalized):.3f}\nStd: {np.std(signal_normalized):.3f}',
                transform=ax7.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

        plt.suptitle(f'{title_prefix}ECG Cleaning Pipeline Visualization',
                    fontsize=14, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        else:
            plt.show()

        plt.close()

        return signal_normalized, is_good, qc_info

    def plot_before_after_comparison(self, signal_raw, signal_clean, title_prefix="",
                                     qc_info=None, save_path=None):
        """
        Simple side-by-side comparison of raw and cleaned signals.

        Args:
            signal_raw: Raw input signal
            signal_clean: Cleaned signal
            title_prefix: Prefix for titles
            qc_info: Quality check info dictionary
            save_path: Optional path to save
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

        # Raw signal
        time_raw = np.arange(len(signal_raw)) / self.fs
        axes[0].plot(time_raw, signal_raw, color='red', linewidth=0.8, alpha=0.7)
        axes[0].set_title(f'{title_prefix}BEFORE Cleaning - Raw Signal',
                         fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Amplitude (mV)', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].text(0.02, 0.95,
                    f'Mean: {np.mean(signal_raw):.3f}\nStd: {np.std(signal_raw):.3f}\nLength: {len(signal_raw)} samples',
                    transform=axes[0].transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.7))

        # Clean signal
        time_clean = np.arange(len(signal_clean)) / self.fs
        axes[1].plot(time_clean, signal_clean, color='green', linewidth=0.8, alpha=0.8)
        axes[1].set_title(f'{title_prefix}AFTER Cleaning - Processed Signal',
                         fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Amplitude (normalized)', fontsize=11)
        axes[1].set_xlabel('Time (s)', fontsize=11)
        axes[1].grid(True, alpha=0.3)

        info_text = f'Mean: {np.mean(signal_clean):.3f}\nStd: {np.std(signal_clean):.3f}\nLength: {len(signal_clean)} samples'
        if qc_info:
            if 'num_beats' in qc_info:
                info_text += f'\nBeats: {qc_info["num_beats"]}'
            if 'snr' in qc_info and qc_info['snr'] is not None:
                info_text += f'\nSNR: {qc_info["snr"]:.2f}'

        axes[1].text(0.02, 0.95, info_text,
                    transform=axes[1].transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def min_max(self, signal):
        max = np.int32(np.max(signal))
        min = np.int32(np.min(signal))
        if min == 0:
            return signal
        return (signal - min) / (max - min)

    def visualize_from_hdf5(self, file_path, split='learning', index=0,
                           show_pipeline=True, save_dir=None):
        """
        Load signal from HDF5 file and visualize cleaning process.

        Args:
            file_path: Path to HDF5 file
            split: Dataset split name
            index: Signal index
            show_pipeline: If True, show step-by-step pipeline; if False, show before/after only
            save_dir: Directory to save plots
        """
        with h5py.File(file_path, 'r') as f:
            signals = f[f"{split}/signals"]
            labels = f[f"{split}/labels"]

            if index >= len(signals):
                print(f"Index {index} out of range (max: {len(signals)-1})")
                return

            signal_raw = signals[index]
            label = labels[index]

            # Get patient and segment IDs if available
            try:
                patient_ids = f[f"{split}/patient_ids"]
                segment_ids = f[f"{split}/segment_ids"]
                patient_id = patient_ids[index]
                segment_id = segment_ids[index]

                # Decode if bytes
                if isinstance(patient_id, bytes):
                    patient_id = patient_id.decode()
                if isinstance(segment_id, bytes):
                    segment_id = segment_id.decode()

                title_prefix = f"[{label}] {patient_id} - {segment_id}\n"
            except:
                title_prefix = f"[Label {label}] Index {index}\n"

        # Run cleaning pipeline
        signal_clean, qc_info = self.cleaner.preprocess(signal_raw, check_quality=True)

        if signal_clean is None:
            print(f"Signal failed quality check: {qc_info}")
            signal_clean = self.cleaner.normalize(
                self.cleaner.trim_to_center(
                    self.cleaner.lowpass_filter(
                        self.cleaner.highpass_filter(signal_raw - np.mean(signal_raw))
                    ),
                    target_seconds=10
                )
            )

        # Generate plots
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        if show_pipeline:
            save_path = save_dir / f"pipeline_{index}.png" if save_dir else None
            self.plot_cleaning_pipeline(signal_raw, title_prefix=title_prefix, save_path=save_path)
        else:
            save_path = save_dir / f"comparison_{index}.png" if save_dir else None
            self.plot_before_after_comparison(signal_raw, signal_clean,
                                             title_prefix=title_prefix,
                                             qc_info=qc_info,
                                             save_path=save_path)

        return signal_raw, signal_clean, qc_info

    def batch_visualize(self, file_path, split='learning', indices=None, n_random=5,
                       save_dir=None, show_pipeline=True):
        """
        Visualize multiple signals in batch.

        Args:
            file_path: Path to HDF5 file
            split: Dataset split
            indices: List of specific indices to visualize (if None, use random)
            n_random: Number of random signals if indices not provided
            save_dir: Directory to save plots
            show_pipeline: Show full pipeline or just before/after
        """
        if indices is None:
            with h5py.File(file_path, 'r') as f:
                n_signals = len(f[f"{split}/signals"])
            indices = np.random.choice(n_signals, size=min(n_random, n_signals), replace=False)

        print(f"Visualizing {len(indices)} signals...")

        for i, idx in enumerate(indices):
            print(f"\n[{i+1}/{len(indices)}] Processing index {idx}...")
            self.visualize_from_hdf5(file_path, split, idx, show_pipeline, save_dir)


def main():
    """Example usage"""

    # Configuration
    FILE_PATH = Path("/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf/paf_segments_all_no_clean.h5")
    SAVE_DIR = Path("/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf/cleaning_visualization")

    # Create visualizer
    visualizer = CleaningVisualizer(fs=128)  # Adjust fs to match your data

    # # Example 1: Visualize a single signal with full pipeline
    # print("="*80)
    # print("SINGLE SIGNAL VISUALIZATION - FULL PIPELINE")
    # print("="*80)
    # visualizer.visualize_from_hdf5(
    #     file_path=FILE_PATH,
    #     split='all',
    #     index=0,
    #     show_pipeline=True,
    #     save_dir=SAVE_DIR
    # )
    #
    # # Example 2: Visualize a single signal with before/after comparison
    # print("\n" + "="*80)
    # print("SINGLE SIGNAL VISUALIZATION - BEFORE/AFTER")
    # print("="*80)
    # visualizer.visualize_from_hdf5(
    #     file_path=FILE_PATH,
    #     split='all',
    #     index=1,
    #     show_pipeline=False,
    #     save_dir=SAVE_DIR
    # )

    # Example 3: Batch visualization
    print("\n" + "="*80)
    print("BATCH VISUALIZATION")
    print("="*80)
    visualizer.batch_visualize(
        file_path=FILE_PATH,
        split='all',
        n_random=5,
        show_pipeline=True,
        save_dir=SAVE_DIR
    )

    print("\n" + "="*80)
    print(f"All visualizations saved to: {SAVE_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()