"""
Subband Recurrence Plot Visualizer

Generates and visualizes recurrence plots for each wavelet subband using the
same preprocessing pipeline as dataset_generator.py.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt
import wfdb
from scipy import signal as sp_signal
from scipy.spatial.distance import pdist

from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator
from src.recurrence_plot.AMI import AMI
from src.recurrence_plot.FNN import FNN


# ── Pipeline Constants (matching dataset_generator.py) ────────────────────────
FS_ORIG = 128
FS_TARGET = 500
SEGMENT_SAMPLES = FS_TARGET * 10  # 10 seconds
PAD_SAMPLES = 8192

WAVELET = 'db8'
SUBBAND_LEVELS = [2, 3, 4, 5, 6, 7, 8]
N_LEVELS_SWT = 10


# ── Helper Functions (matching dataset_generator.py) ──────────────────────────
def _minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalization."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


def _pad(x: np.ndarray, pad_samples: int = PAD_SAMPLES) -> np.ndarray:
    """Pad or truncate signal to fixed length."""
    pad = pad_samples - len(x)
    if pad <= 0:
        return x[:pad_samples]
    return np.pad(x, (0, pad), mode='reflect')


def extract_subbands_no_denoise(
    signal_padded: np.ndarray,
    wavelet: str = WAVELET,
    n_levels: int = N_LEVELS_SWT,
    subband_levels: List[int] = SUBBAND_LEVELS,
) -> Dict[str, np.ndarray]:
    """Extract subbands without denoising (matching dataset_generator.py)."""
    coeffs = pywt.swt(signal_padded, wavelet, level=n_levels)
    n_lvls = len(coeffs)
    subbands: Dict[str, np.ndarray] = {}

    for level in subband_levels:
        target_idx = n_lvls - level
        zeroed = [
            ((np.zeros_like(cA), cD) if j == target_idx
             else (np.zeros_like(cA), np.zeros_like(cD)))
            for j, (cA, cD) in enumerate(coeffs)
        ]
        sub = pywt.iswt(zeroed, wavelet)[:SEGMENT_SAMPLES].astype(np.float32)
        subbands[f'D{level}'] = _minmax(sub)

    return subbands


def denoise_and_extract_subbands(
    signal_padded: np.ndarray,
    wavelet: str = WAVELET,
    n_levels: int = N_LEVELS_SWT,
    subband_levels: List[int] = SUBBAND_LEVELS,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Denoise using soft-thresholding and extract subbands.
    Matches dataset_generator.py implementation.
    """
    import math
    coeffs = pywt.swt(signal_padded, wavelet, level=n_levels)

    N_log = math.log2(FS_TARGET / 0.5)
    denoised_coeffs = []
    for cA, cD in coeffs:
        sigma = np.median(np.abs(cD)) / 0.6745
        T = sigma * math.sqrt(2.0 * math.log(N_log))
        cD_d = np.sign(cD) * np.maximum(np.abs(cD) - T, 0.0)
        denoised_coeffs.append((cA, cD_d))

    denoised = pywt.iswt(denoised_coeffs, wavelet)[:SEGMENT_SAMPLES].astype(np.float32)

    n_lvls = len(denoised_coeffs)
    subbands: Dict[str, np.ndarray] = {}

    for level in subband_levels:
        target_idx = n_lvls - level
        zeroed = [
            ((np.zeros_like(cA), cD) if j == target_idx
             else (np.zeros_like(cA), np.zeros_like(cD)))
            for j, (cA, cD) in enumerate(denoised_coeffs)
        ]
        sub = pywt.iswt(zeroed, wavelet)[:SEGMENT_SAMPLES].astype(np.float32)
        subbands[f'D{level}'] = _minmax(sub)

    return denoised, subbands


def reconstruct_bandpass(
    signal_padded: np.ndarray,
    wavelet: str = WAVELET,
) -> np.ndarray:
    """
    Reconstruct signal keeping only D2-D8 (structural bandpass).
    Matches dataset_generator.py implementation.
    """
    coeffs = pywt.swt(signal_padded, wavelet, level=N_LEVELS_SWT)
    n_lvls = len(coeffs)
    keep = set(SUBBAND_LEVELS)

    filtered = []
    for j, (cA, cD) in enumerate(coeffs):
        actual_level = n_lvls - j
        if actual_level in keep:
            filtered.append((np.zeros_like(cA), cD))
        else:
            filtered.append((np.zeros_like(cA), np.zeros_like(cD)))

    reconstructed = pywt.iswt(filtered, wavelet)[:SEGMENT_SAMPLES]
    return _minmax(reconstructed.astype(np.float32))


# ── Main Visualizer Class ──────────────────────────────────────────────────────
class SubbandRPVisualizer:
    """
    Visualizes recurrence plots for wavelet subbands using the same
    preprocessing pipeline as dataset_generator.py.
    """

    def __init__(
        self,
        wavelet: str = WAVELET,
        subband_levels: List[int] = None,
        use_denoise: bool = False,
        fs_target: int = FS_TARGET,
        segment_samples: int = SEGMENT_SAMPLES,
    ):
        """
        Args:
            wavelet: Wavelet family (default: 'db8')
            subband_levels: List of decomposition levels (default: [2-8])
            use_denoise: Whether to apply soft-threshold denoising
            fs_target: Target sampling frequency
            segment_samples: Segment length in samples
        """
        self.wavelet = wavelet
        self.subband_levels = subband_levels or SUBBAND_LEVELS
        self.use_denoise = use_denoise
        self.fs_target = fs_target
        self.segment_samples = segment_samples

    def load_signal_from_record(
        self,
        record_path: Path,
        resample: bool = True,
    ) -> np.ndarray:
        """
        Load ECG signal from WFDB record.

        Args:
            record_path: Path to WFDB record (without extension)
            resample: Whether to resample from FS_ORIG to FS_TARGET

        Returns:
            Signal array (float32)
        """
        try:
            rec = wfdb.rdrecord(str(record_path))
            raw = rec.p_signal[:, 0].astype(np.float64)

            if resample:
                n = int(len(raw) * self.fs_target / FS_ORIG)
                signal = sp_signal.resample(raw, n).astype(np.float32)
            else:
                signal = raw.astype(np.float32)

            return signal

        except Exception as e:
            raise ValueError(f"Failed to load record {record_path}: {e}")

    def extract_segment(
        self,
        signal: np.ndarray,
        start_sec: float = 0.0,
    ) -> np.ndarray:
        """
        Extract a segment from the signal.

        Args:
            signal: Full signal array
            start_sec: Start time in seconds

        Returns:
            Segment of length segment_samples, min-max normalized
        """
        start_idx = int(start_sec * self.fs_target)
        end_idx = start_idx + self.segment_samples

        if end_idx > len(signal):
            raise ValueError(
                f"Segment [{start_sec}s - {start_sec + self.segment_samples/self.fs_target}s] "
                f"exceeds signal length {len(signal)/self.fs_target:.1f}s"
            )

        segment = signal[start_idx:end_idx]
        return _minmax(segment.astype(np.float32))

    def decompose_signal(
        self,
        signal: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Decompose signal into subbands using the dataset_generator pipeline.

        Args:
            signal: Input signal (segment_samples length)

        Returns:
            denoised_signal: Denoised signal if use_denoise=True, else None
            subbands: Dict mapping 'D2', 'D3', ... to subband arrays
        """
        signal_pad = _pad(signal, PAD_SAMPLES)

        if self.use_denoise:
            denoised, subbands = denoise_and_extract_subbands(
                signal_pad, self.wavelet, N_LEVELS_SWT, self.subband_levels
            )
            return denoised, subbands
        else:
            subbands = extract_subbands_no_denoise(
                signal_pad, self.wavelet, N_LEVELS_SWT, self.subband_levels
            )
            return None, subbands

    def get_optimal_params(
        self,
        signal: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Compute optimal tau and m using AMI and FNN.

        Args:
            signal: Input signal

        Returns:
            tau: Time delay
            m: Embedding dimension
        """
        ami = AMI(signal)
        tau = ami.get_first_minimum()
        fnn = FNN(signal=signal, tau=tau)
        m = fnn.get_optimal_dimension()
        return tau, m

    def compute_rp_binary(
        self,
        signal: np.ndarray,
        tau: int,
        m: int,
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Compute binary (thresholded) recurrence plot and RQA metrics.

        Epsilon is automatically computed as 10% of the maximum pairwise distance
        in the embedded space.

        Args:
            signal: Input signal
            tau: Time delay
            m: Embedding dimension

        Returns:
            rp_matrix: Binary recurrence plot
            epsilon: Computed threshold distance (10% of max distance)
            rqa_metrics: Dict of RQA values (DET, LAM, REC, ENTR, L, Lmax, TT)
        """
        from pyrqa.time_series import TimeSeries
        from pyrqa.settings import Settings
        from pyrqa.analysis_type import Classic
        from pyrqa.neighbourhood import FixedRadius
        from pyrqa.metric import EuclideanMetric
        from pyrqa.computation import RPComputation, RQAComputation

        n_points = len(signal) - (m - 1) * tau
        embedded = np.array([
            signal[i: i + m* tau: tau]
            for i in range(n_points)
        ])

        if len(embedded) > 2000:
            idx = np.linspace(0, len(embedded) - 1, 2000, dtype=int)
            embedded_sub = embedded[idx]
        else:
            embedded_sub = embedded

        distances = pdist(embedded_sub, metric='euclidean')

        # Epsilon = 10% of maximum pairwise distance
        epsilon = 0.1 * np.max(distances)

        ts = TimeSeries(signal, embedding_dimension=m, time_delay=tau)
        settings = Settings(
            time_series=ts,
            analysis_type=Classic,
            neighbourhood=FixedRadius(epsilon),
            similarity_measure=EuclideanMetric,
            theiler_corrector=1,
        )

        rp_comp = RPComputation.create(settings, verbose=False)
        rqa_comp = RQAComputation.create(settings, verbose=False)

        rp = rp_comp.run()
        rqa = rqa_comp.run()

        metrics = {
            'DET': rqa.determinism,
            'LAM': rqa.laminarity,
            'REC': rqa.recurrence_rate,
            'ENTR': rqa.entropy_diagonal_lines,
            'Lmax': rqa.longest_diagonal_line,
            'TT': rqa.trapping_time,
        }

        return rp.recurrence_matrix_reverse, epsilon, metrics

    def compute_rp_unthresholded(
        self,
        signal: np.ndarray,
        tau: int,
        m: int,
        size: int = 256,
    ) -> np.ndarray:
        """
        Compute unthresholded RP (UTRP) as RGB colormap.
        Matches dataset_generator.py implementation.

        Args:
            signal: Input signal
            tau: Time delay
            m: Embedding dimension
            size: Output image size

        Returns:
            RGB UTRP image (3, size, size) float32
        """
        try:
            sig = signal[:self.segment_samples].astype(np.float32)
            n = len(sig) - (m - 1) * tau
            if n <= 0:
                return np.zeros((3, size, size), dtype=np.float32)

            gen = RecurrencePlotGenerator(
                signal=sig, time_delay=tau, embedding_dim=m, epsilon=0
            )
            rp = gen.compute_rp_unthresholded()

            # Normalize
            d_max = rp.max()
            if d_max > 1e-9:
                rp = rp / d_max

            # Resize
            rp_r = cv2.resize(
                rp.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR
            )
            np.fill_diagonal(rp_r, 0.0)

            # Apply JET colormap
            u8 = (rp_r * 255).astype(np.uint8)
            bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            return rgb.transpose(2, 0, 1)  # (3, H, W)

        except Exception as e:
            print(f"    [ERROR] compute_rp_unthresholded: {e}")
            return np.zeros((3, size, size), dtype=np.float32)

    def find_epsilon_for_target_rr(
        self,
        signal: np.ndarray,
        tau: int,
        m: int,
        target_rr: float = 0.05,
    ) -> float:
        """Find epsilon that achieves target recurrence rate."""
        return RecurrencePlotGenerator.find_epsilon_for_target_rr(
            signal, tau, m, target_rr
        )

    def visualize_subband_rps(
        self,
        signal: np.ndarray,
        params_dict: Optional[Dict[str, Tuple[int, int]]] = None,
        utrp_size: int = 256,
        save_path: Optional[Path] = None,
        title: str = "Subband Recurrence Plots",
        show: bool = True,
    ) -> plt.Figure:
        """
        Visualize RPs for all subbands in a grid.

        Layout: [Subband Signal] | [Binary RP] | [UTRP (colormap)]

        Binary RP epsilon is automatically computed as 10% of max pairwise distance.

        Args:
            signal: Input signal (segment_samples length)
            params_dict: Optional dict mapping subband names to (tau, m).
                        If None, computes optimal params for each subband.
            utrp_size: Size of UTRP images
            save_path: Path to save figure
            title: Figure title
            show: Whether to display the figure

        Returns:
            matplotlib Figure
        """
        # Decompose signal
        denoised, subbands = self.decompose_signal(signal)

        n_subbands = len(self.subband_levels)
        fig, axes = plt.subplots(
            n_subbands, 3, figsize=(18, 3.5 * n_subbands)
        )

        if n_subbands == 1:
            axes = axes.reshape(1, -1)

        t = np.arange(self.segment_samples) / self.fs_target

        for i, level in enumerate(self.subband_levels):
            name = f'D{level}'
            sub = subbands[name]

            # Get tau, m
            if params_dict and name in params_dict:
                tau, m = params_dict[name]
            else:
                tau, m = self.get_optimal_params(sub)

            # Column 0: Signal
            axes[i, 0].plot(t, sub, color='steelblue', linewidth=0.6)
            axes[i, 0].set_ylabel(f'{name}\nstd={np.std(sub):.4f}', fontweight='bold')
            axes[i, 0].grid(True, alpha=0.3)
            if i == 0:
                axes[i, 0].set_title('Subband Signal', fontweight='bold', fontsize=12)
            if i == n_subbands - 1:
                axes[i, 0].set_xlabel('Time (s)', fontweight='bold')

            # Column 1: Binary RP
            try:
                rp_bin, epsilon, rqa = self.compute_rp_binary(sub, tau, m)
                axes[i, 1].imshow(rp_bin, cmap='binary', origin='lower', aspect='equal')
                axes[i, 1].set_title(
                    f'Binary RP\nτ={tau}, m={m}, ε={epsilon:.4f}\n',
                    fontsize=9
                )
                axes[i, 1].axis('off')
            except Exception as e:
                axes[i, 1].text(
                    0.5, 0.5, f'Error:\n{str(e)[:40]}',
                    ha='center', va='center', fontsize=8
                )
                axes[i, 1].axis('off')

            # Column 2: UTRP
            try:
                utrp = self.compute_rp_unthresholded(sub, tau, m, utrp_size)
                # Convert (3, H, W) -> (H, W, 3)
                utrp_img = utrp.transpose(1, 2, 0)
                axes[i, 2].imshow(utrp_img, origin='lower', aspect='equal')
                axes[i, 2].set_title(f'UTRP (JET colormap)', fontsize=9)
                axes[i, 2].axis('off')
            except Exception as e:
                axes[i, 2].text(
                    0.5, 0.5, f'Error:\n{str(e)[:40]}',
                    ha='center', va='center', fontsize=8
                )
                axes[i, 2].axis('off')

        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved figure → {save_path}")

        if show:
            plt.show()

        return fig

    def visualize_single_bandpass(
        self,
        signal: np.ndarray,
        tau: int,
        m: int,
        utrp_size: int = 256,
        save_path: Optional[Path] = None,
        title: str = "Bandpass Signal (D2-D8) Recurrence Plots",
        show: bool = True,
    ) -> plt.Figure:
        """
        Visualize RP for the bandpass-reconstructed signal (D2-D8 only).

        Binary RP epsilon is automatically computed as 10% of max pairwise distance.

        Args:
            signal: Input signal (segment_samples length)
            tau: Time delay
            m: Embedding dimension
            utrp_size: UTRP image size
            save_path: Save path
            title: Figure title
            show: Whether to display

        Returns:
            matplotlib Figure
        """
        signal_pad = _pad(signal, PAD_SAMPLES)
        bandpass = reconstruct_bandpass(signal_pad, self.wavelet)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        t = np.arange(len(bandpass)) / self.fs_target

        # Signal
        axes[0].plot(t, bandpass, color='darkgreen', linewidth=0.6)
        axes[0].set_xlabel('Time (s)', fontweight='bold')
        axes[0].set_ylabel('Amplitude', fontweight='bold')
        axes[0].set_title('Bandpass Signal (D2-D8)', fontweight='bold', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Binary RP
        try:
            rp_bin, epsilon, rqa = self.compute_rp_binary(bandpass, tau, m)
            axes[1].imshow(rp_bin, cmap='binary', origin='lower', aspect='equal')
            axes[1].set_title(
                f'Binary RP\nτ={tau}, m={m}, ε={epsilon:.4f}\n',
                fontweight='bold', fontsize=11
            )
            axes[1].axis('off')
        except Exception as e:
            axes[1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
            axes[1].axis('off')

        # UTRP
        try:
            utrp = self.compute_rp_unthresholded(bandpass, tau, m, utrp_size)
            utrp_img = utrp.transpose(1, 2, 0)
            axes[2].imshow(utrp_img, origin='lower', aspect='equal')
            axes[2].set_title('UTRP (JET colormap)', fontweight='bold', fontsize=11)
            axes[2].axis('off')
        except Exception as e:
            axes[2].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
            axes[2].axis('off')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved figure → {save_path}")

        if show:
            plt.show()

        return fig

    def load_params_from_json(self, params_file: Path) -> Dict[str, Tuple[int, int]]:
        """
        Load embedding parameters from JSON file.

        Expected format:
        {
            "tau_single": int,
            "m_single": int,
            "subband_tau": {"D2": int, "D3": int, ...},
            "subband_m": {"D2": int, "D3": int, ...}
        }

        Returns:
            Dict mapping subband names and 'single' to (tau, m) tuples
        """
        with open(params_file) as f:
            params = json.load(f)

        result = {}
        result['single'] = (params['tau_single'], params['m_single'])

        for name in [f'D{level}' for level in self.subband_levels]:
            tau = params['subband_tau'].get(name, 3)
            m = params['subband_m'].get(name, 3)
            result[name] = (tau, m)

        return result


# ── Example Usage ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from pathlib import Path

    # Configuration
    DATA_DIR = Path('/data_old/paf')
    OUTPUT_DIR = Path('/data/outputs/subband_rp_vis')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    PARAMS_FILE = Path('/data_old/data_new/attention/tau_m_params.json')

    # Initialize visualizer
    viz = SubbandRPVisualizer(
        wavelet='db8',
        subband_levels=[2, 3, 4, 5, 6, 7, 8],
        use_denoise=False,  # Set to True to match denoising pipeline
    )

    # Load parameters
    params = viz.load_params_from_json(PARAMS_FILE)

    # Example 1: Load from WFDB record
    record_id = 'n34'  # Pre-AF record
    print(f"Loading record: {record_id}")
    signal = viz.load_signal_from_record(DATA_DIR / record_id, resample=True)

    # Extract 10-second segment
    segment = viz.extract_segment(signal, start_sec=0.0)
    print(f"Segment shape: {segment.shape} ({len(segment)/FS_TARGET:.1f} seconds)")

    # Visualize subband RPs
    viz.visualize_subband_rps(
        segment,
        params_dict=params,
        save_path=OUTPUT_DIR / f'{record_id}_subband_rps.png',
        title=f'Subband RPs - {record_id} (db8, no denoise)',
        show=False,
    )

    # Visualize bandpass single signal
    tau_single, m_single = params['single']
    viz.visualize_single_bandpass(
        segment,
        tau=tau_single,
        m=m_single,
        save_path=OUTPUT_DIR / f'{record_id}_bandpass_rp.png',
        title=f'Bandpass RP - {record_id} (D2-D8)',
        show=False,
    )

    print(f"\n✓ All visualizations saved to {OUTPUT_DIR}")