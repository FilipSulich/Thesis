"""
ECG Subband Analyzer: Decompose, visualize, and analyze wavelet subbands with SNR measurement.

Features:
- SWT decomposition into N subbands
- Vertical subband visualization
- Min-max normalization per subband
- UTRP generation from normalized subbands
- SNR measurement for each subband
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import wfdb
from pathlib import Path
from typing import Tuple, Dict, List
from scipy import signal as sp_signal

from src.CNN.dataset.dataset_generator import (
    extract_subbands_swt, _minmax_norm, SEGMENT_SAMPLES, FS_ORIG, FS_TARGET
)
from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/paf')


class SubbandAnalyzer:
    """Analyze ECG signal through wavelet subbands with visualization and UTRP generation."""
    
    def __init__(self, record_id: str, segment_idx: int = 0, wavelet: str = 'db4', 
                 n_levels: int = 7, utrp_size: int = 256):
        """
        Initialize analyzer.
        
        Args:
            record_id: PAF dataset record ID (e.g., 'p01', 'n01')
            segment_idx: Which segment to analyze (0-19)
            wavelet: Wavelet name (db4, db5, etc.)
            n_levels: Number of decomposition levels
            utrp_size: Size of UTRP output (256×256 typical)
        """
        self.record_id = record_id
        self.segment_idx = segment_idx
        self.wavelet = wavelet
        self.n_levels = n_levels
        self.utrp_size = utrp_size
        
        # Will be populated by load_segment()
        self.raw_signal = None
        self.signal = None
        self.subbands = {}
        self.normalized_subbands = {}
        self.utrps = {}
        self.snr_values = {}
    
    def load_segment(self) -> np.ndarray:
        """Load and preprocess ECG segment from dataset."""
        try:
            rec = wfdb.rdrecord(str(DATA_DIR / self.record_id))
            raw = rec.p_signal[:, 0].astype(np.float64)
            
            # Resample to target FS
            n_tgt = int(len(raw) * FS_TARGET / FS_ORIG)
            resampled = sp_signal.resample(raw, n_tgt)
            
            # Extract segment
            max_start = len(resampled) - SEGMENT_SAMPLES
            if max_start <= 0:
                raise ValueError(f"Record too short: {len(resampled)} samples")
            
            starts = np.linspace(0, max_start, 20, dtype=int)
            segment = resampled[starts[self.segment_idx]:starts[self.segment_idx] + SEGMENT_SAMPLES]
            
            if len(segment) < SEGMENT_SAMPLES:
                raise ValueError(f"Segment too short: {len(segment)} < {SEGMENT_SAMPLES}")
            
            self.raw_signal = segment
            self.signal = _minmax_norm(segment.astype(np.float32))
            
            print(f"✓ Loaded {self.record_id} segment {self.segment_idx}")
            print(f"  Raw signal: shape={self.raw_signal.shape}, range=[{self.raw_signal.min():.4f}, {self.raw_signal.max():.4f}]")
            print(f"  Normalized: shape={self.signal.shape}, range=[{self.signal.min():.4f}, {self.signal.max():.4f}]")
            
            return self.signal
        
        except Exception as e:
            raise RuntimeError(f"Failed to load segment: {e}")
    
    def decompose_subbands(self) -> Dict[str, np.ndarray]:
        """Decompose signal into subbands using SWT."""
        if self.signal is None:
            raise ValueError("Call load_segment() first")
        
        self.subbands = extract_subbands_swt(
            self.signal, 
            wavelet=self.wavelet, 
            level=self.n_levels,
            target_length=None  # Use dyadic length
        )
        
        print(f"\n✓ Decomposed into {len(self.subbands)} subbands ({self.wavelet}, level {self.n_levels})")
        for name, subband in self.subbands.items():
            print(f"  {name}: shape={subband.shape}, range=[{subband.min():.4f}, {subband.max():.4f}]")
        
        return self.subbands
    
    def normalize_subbands(self) -> Dict[str, np.ndarray]:
        """Min-max normalize each subband separately."""
        if not self.subbands:
            raise ValueError("Call decompose_subbands() first")
        
        self.normalized_subbands = {}
        print(f"\n✓ Normalizing {len(self.subbands)} subbands independently:")
        
        for name, subband in self.subbands.items():
            normalized = _minmax_norm(subband)
            self.normalized_subbands[name] = normalized
            print(f"  {name}: normalized range=[{normalized.min():.4f}, {normalized.max():.4f}]")
        
        return self.normalized_subbands
    
    def calculate_snr(self) -> Dict[str, float]:
        """
        Calculate SNR for each subband.
        
        SNR = 10 * log10(P_signal / P_noise)
        
        Uses high-pass filtered residual as noise estimate.
        """
        if not self.subbands:
            raise ValueError("Call decompose_subbands() first")
        
        self.snr_values = {}
        print(f"\n✓ Computing SNR for {len(self.subbands)} subbands:")
        
        for name, subband in self.subbands.items():
            # Signal power: variance of the subband
            P_signal = np.var(subband)
            
            # Noise estimate: high-pass filtered residual
            # Use 3rd-order Butterworth high-pass filter
            sos = sp_signal.butter(3, 0.1, 'hp', output='sos')
            highpassed = sp_signal.sosfilt(sos, subband)
            P_noise = np.var(highpassed)
            
            # Avoid log(0)
            if P_noise < 1e-12:
                P_noise = 1e-12
            
            snr_db = 10 * np.log10(P_signal / P_noise) if P_signal > 0 else -np.inf
            self.snr_values[name] = float(snr_db)
            
            print(f"  {name}: P_signal={P_signal:.6f}, P_noise={P_noise:.6f}, SNR={snr_db:.2f} dB")
        
        return self.snr_values
    
    def generate_utrps(self, tau: int = 12, m: int = 3) -> Dict[str, np.ndarray]:
        """
        Generate UTRP from each normalized subband.
        
        Args:
            tau: Time delay for embedding
            m: Embedding dimension
        
        Returns:
            Dict mapping subband names to (3, utrp_size, utrp_size) RGB images
        """
        if not self.normalized_subbands:
            raise ValueError("Call normalize_subbands() first")
        
        self.utrps = {}
        print(f"\n✓ Generating UTRPs from {len(self.normalized_subbands)} subbands:")
        
        for name, subband in self.normalized_subbands.items():
            # Create recurrence plot using embedding
            sig = subband[:SEGMENT_SAMPLES]
            n = len(sig) - (m - 1) * tau
            
            if n <= 0:
                print(f"  {name}: SKIP (signal too short for embedding)")
                continue
            
            # Embed signal
            embedded = np.array([sig[i:i + m * tau:tau] for i in range(n)])
            diff = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff ** 2, axis=2)).astype(np.float32)
            
            # Normalize by max
            d_max = distances.max()
            if d_max > 1e-9:
                distances = distances / d_max
            
            # Resize
            dist_r = cv2.resize(distances, (self.utrp_size, self.utrp_size), 
                              interpolation=cv2.INTER_LINEAR).astype(np.float32)
            np.fill_diagonal(dist_r, 0.0)
            
            # Convert to RGB using JET colormap
            u8 = (dist_r * 255).astype(np.uint8)
            bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            utrp = rgb.transpose(2, 0, 1)  # (3, H, W)
            self.utrps[name] = utrp
            
            print(f"  {name}: UTRP shape={utrp.shape}")
        
        return self.utrps
    
    def plot_subbands_vertical(self, figsize: Tuple[int, int] = (16, 20), 
                              save_path: Path = None) -> plt.Figure:
        """
        Plot all subbands vertically with their SNR values.
        
        Args:
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        if not self.subbands:
            raise ValueError("Call decompose_subbands() first")
        
        n_subbands = len(self.subbands)
        fig, axes = plt.subplots(n_subbands + 2, 1, figsize=figsize)
        
        if n_subbands == 1:
            axes = [axes]
        
        # Plot original signal
        ax = axes[0]
        ax.plot(self.signal, linewidth=0.8, color='black')
        ax.set_title(f'Original Signal ({self.record_id} seg {self.segment_idx})', fontsize=12, fontweight='bold')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        # Plot each subband
        colors = plt.cm.viridis(np.linspace(0, 1, n_subbands))
        
        for i, (name, subband) in enumerate(self.subbands.items()):
            ax = axes[i + 1]
            ax.plot(subband, linewidth=0.8, color=colors[i])
            
            # Add SNR to title if available
            snr_text = f" (SNR: {self.snr_values[name]:.2f} dB)" if name in self.snr_values else ""
            ax.set_title(f'{name}: {self.wavelet} Level {n_subbands - i}{snr_text}', 
                        fontsize=11, fontweight='bold')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
        
        # Plot normalized subbands comparison
        ax = axes[-1]
        for i, (name, subband) in enumerate(self.normalized_subbands.items()):
            ax.plot(subband, linewidth=0.8, label=name, alpha=0.7)
        
        ax.set_title('All Normalized Subbands', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Amplitude')
        ax.set_xlabel('Sample')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved plot to {save_path}")
        
        return fig
    
    def plot_utrps(self, figsize: Tuple[int, int] = None, 
                   save_path: Path = None) -> plt.Figure:
        """
        Plot all UTRPs vertically.
        
        Args:
            figsize: Figure size (auto if None)
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        if not self.utrps:
            raise ValueError("Call generate_utrps() first")
        
        n_utrps = len(self.utrps)
        if figsize is None:
            figsize = (14, 4 * n_utrps)
        
        fig, axes = plt.subplots(n_utrps, 1, figsize=figsize)
        if n_utrps == 1:
            axes = [axes]
        
        for i, (name, utrp) in enumerate(self.utrps.items()):
            ax = axes[i]
            # UTRP is (3, H, W), convert to (H, W, 3) for display
            utrp_display = utrp.transpose(1, 2, 0)
            ax.imshow(utrp_display, cmap='jet')
            snr_text = f" SNR: {self.snr_values[name]:.2f} dB" if name in self.snr_values else ""
            ax.set_title(f'{name} UTRP{snr_text}', fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved UTRP plot to {save_path}")
        
        return fig
    
    def print_summary(self) -> None:
        """Print analysis summary."""
        print("\n" + "="*70)
        print(f"SUBBAND ANALYSIS SUMMARY: {self.record_id} segment {self.segment_idx}")
        print("="*70)
        
        print(f"\nDecomposition:")
        print(f"  Wavelet: {self.wavelet}")
        print(f"  Levels: {self.n_levels}")
        print(f"  Signal length: {len(self.signal)} samples")
        
        if self.snr_values:
            print(f"\nSNR per subband (dB):")
            for name in sorted(self.snr_values.keys()):
                snr = self.snr_values[name]
                print(f"  {name}: {snr:7.2f} dB")
            
            avg_snr = np.mean(list(self.snr_values.values()))
            print(f"  Average: {avg_snr:7.2f} dB")
        
        if self.utrps:
            print(f"\nUTRPs generated: {len(self.utrps)}")
            for name, utrp in self.utrps.items():
                print(f"  {name}: {utrp.shape}")
        
        print("="*70 + "\n")
    
    def run_full_analysis(self, save_dir: Path = None) -> Dict:
        """
        Run complete analysis pipeline.
        
        Args:
            save_dir: Directory to save plots
        
        Returns:
            Results dictionary
        """
        save_dir = Path(save_dir) if save_dir else Path('/tmp')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Run pipeline
        self.load_segment()
        self.decompose_subbands()
        self.normalize_subbands()
        self.calculate_snr()
        self.generate_utrps()
        
        # Plot and save
        fig1 = self.plot_subbands_vertical(save_path=save_dir / f"{self.record_id}_seg{self.segment_idx}_subbands.png")
        fig2 = self.plot_utrps(save_path=save_dir / f"{self.record_id}_seg{self.segment_idx}_utrps.png")
        
        self.print_summary()
        
        return {
            'record_id': self.record_id,
            'segment_idx': self.segment_idx,
            'wavelet': self.wavelet,
            'n_levels': self.n_levels,
            'subbands': self.subbands,
            'normalized_subbands': self.normalized_subbands,
            'snr_values': self.snr_values,
            'utrps': self.utrps,
        }


if __name__ == '__main__':
    # Example usage
    analyzer = SubbandAnalyzer(record_id='p34', segment_idx=0, wavelet='sym2', n_levels=10)
    results = analyzer.run_full_analysis(save_dir='/Users/filipsulich/Developer/uni/Thesis/Thesis/results')
    
    plt.show()

