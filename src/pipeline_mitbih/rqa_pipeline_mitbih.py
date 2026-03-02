"""
MIT-BIH RQA Pipeline: Single-scale and Multi-scale RQA analysis.

Pipeline 1 (single-scale): clean ECG (10s) -> RP -> RQA
Pipeline 2 (multi-scale): clean ECG (10s) -> SWT -> inverse SWT per sub-band -> RP per sub-band -> RQA per sub-band

Processing order:
1. Raw ECG
2. Basic filtering (highpass + lowpass)
3. SWT-based adaptive denoising
4. Z-score normalization
5. Embedding & RP & RQA
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import csv
from pathlib import Path
from scipy.spatial.distance import pdist
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wavelets.stationary_wavelet_transform as stationary_wavelet
import recurrence_plot.recurrence_plot_generation as recurrence_plot
from data_cleaning.data_cleaner import DataCleaner

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/mitbih')
OUTPUT_DIR = DATA_DIR / 'results'

SAMPLING_FREQ = 250
SUBBAND_NAMES = ["D1", "D2", "D3", "D4", "A4"]


# =============================================================================
# Helper Functions
# =============================================================================

def extract_rqa_metrics(rqa_result) -> Dict[str, float]:
    """Extract RQA metrics from pyrqa result object."""
    return {
        "recurrence_rate": rqa_result.recurrence_rate,
        "determinism": rqa_result.determinism,
        "average_diagonal_line": rqa_result.average_diagonal_line,
        "longest_diagonal_line": rqa_result.longest_diagonal_line,
        "divergence": rqa_result.divergence,
        "entropy_diagonal_lines": rqa_result.entropy_diagonal_lines,
        "laminarity": rqa_result.laminarity,
        "trapping_time": rqa_result.trapping_time,
        "longest_vertical_line": rqa_result.longest_vertical_line,
        "entropy_vertical_lines": rqa_result.entropy_vertical_lines,
    }


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MITBIHResult:
    """Result container for MIT-BIH RQA analysis."""
    segment_id: str
    patient_id: str
    label: int  # 0 = normal, 1 = pre-AF
    lead_time: int  # minutes before AF onset
    single_scale: bool
    rqa_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        result = {
            "segment_id": self.segment_id,
            "patient_id": self.patient_id,
            "label": self.label,
            "lead_time": self.lead_time,
            "single_scale": self.single_scale,
        }
        for key, rqa in self.rqa_results.items():
            metrics = extract_rqa_metrics(rqa)
            for metric_name, value in metrics.items():
                result[f"{key}_{metric_name}"] = value
        return result


# =============================================================================
# Pipeline Class
# =============================================================================

class MITBIHRQAPipeline:
    """
    MIT-BIH RQA Pipeline for single-scale and multi-scale analysis.

    Processes cleaned signals from CSV files and computes RQA metrics.
    """

    def __init__(
        self,
        time_delay: int = 12,
        embedding_dim: int = 7,
        epsilon_method: str = 'std',
        epsilon_factor: float = 0.2,
        subband_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize MIT-BIH RQA pipeline.

        Args:
            time_delay: Time delay for embedding (tau)
            embedding_dim: Embedding dimension (m)
            epsilon_method: Method for computing epsilon ('std' or 'max_distance')
            epsilon_factor: Factor for epsilon computation
            subband_params: Parameters for each subband (tau, m)
        """
        self.time_delay = time_delay
        self.embedding_dim = embedding_dim
        self.epsilon_method = epsilon_method
        self.epsilon_factor = epsilon_factor
        self.subband_params = subband_params

    def load_data_from_csv(self, csv_path: Path) -> Tuple[List[np.ndarray], List[int], List[str], List[str], List[int]]:
        """
        Load cleaned signals from CSV file.

        Returns:
            Tuple of (signals, labels, segment_ids, patient_ids, lead_times)
        """
        print(f"\nLoading data from {csv_path}...")

        signals = []
        labels = []
        segment_ids = []
        patient_ids = []
        lead_times = []

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract signal (only columns that start with 's' followed by a digit)
                signal_cols = [k for k in row.keys() if k.startswith('s') and k[1:].isdigit()]
                signal = np.array([float(row[col]) for col in sorted(signal_cols, key=lambda x: int(x[1:]))])

                signals.append(signal)
                labels.append(int(row['label']))
                segment_ids.append(row['segment_id'])
                patient_ids.append(row['patient_id'])
                lead_times.append(int(row['lead_time']))

        signals = np.array(signals)
        labels = np.array(labels)
        lead_times = np.array(lead_times)

        print(f"Loaded {len(signals)} signals")
        print(f"  Pre-AF: {sum(labels == 1)}, Normal: {sum(labels == 0)}")
        print(f"  Lead times: {sorted(np.unique(lead_times))}")

        return signals, labels, segment_ids, patient_ids, lead_times

    def _reconstruct_subbands(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decompose signal into SWT subbands with proper per-subband processing.

        For multi-scale:
        1. Takes CLEANED signal (already filtered + denoised from dataset)
        2. Decomposes into subbands
        3. Reconstructs each subband independently
        4. Z-score normalizes EACH subband separately
        5. Each normalized subband then goes to RP → RQA

        Returns:
            Dictionary of normalized subbands ready for RP computation
        """
        swt = stationary_wavelet.SWT(wavelet='rbio3.9', level=4)
        coeffs = swt.decompose(signal, level=4)

        reconstructions = {}

        # Detail coefficients D1-D4
        for i in range(len(coeffs)):
            zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
            subband = swt.reconstruct(zeroed)

            # Z-score normalize THIS subband
            subband = (subband - np.mean(subband)) / (np.std(subband) + 1e-8)
            reconstructions[f"D{i+1}"] = subband

        # Approximation A4
        zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        subband = swt.reconstruct(zeroed)

        # Z-score normalize THIS subband
        subband = (subband - np.mean(subband)) / (np.std(subband) + 1e-8)
        reconstructions["A4"] = subband

        return reconstructions

    def _get_epsilon(
        self,
        signal: np.ndarray,
        embedding_dim: int,
        time_delay: int
    ) -> float:
        """
        Compute epsilon based on configured method.

        Args:
            signal: Input signal
            embedding_dim: Embedding dimension
            time_delay: Time delay

        Returns:
            Epsilon value
        """
        if self.epsilon_method == 'std':
            return self.epsilon_factor * np.std(signal)
        elif self.epsilon_method == 'max_distance':
            N = len(signal) - (embedding_dim - 1) * time_delay
            if N <= 1:
                return self.epsilon_factor * np.std(signal)  # Fallback

            embedded = np.array([
                signal[i:i + embedding_dim * time_delay:time_delay]
                for i in range(N)
            ])
            distances = pdist(embedded, metric='euclidean')
            return self.epsilon_factor * np.max(distances)
        else:
            return self.epsilon_factor * np.std(signal)

    def _compute_rqa(
        self,
        signal: np.ndarray,
        subband_name: Optional[str] = None,
    ):
        """
        Compute RQA for a signal or subband.

        Args:
            signal: Input signal (already cleaned and normalized)
            subband_name: Name of subband (for parameter lookup)

        Returns:
            RQA result object
        """
        # Get parameters
        if subband_name and self.subband_params and subband_name in self.subband_params:
            params = self.subband_params[subband_name]
            tau = params.get("tau", self.time_delay)
            m = params.get("m", self.embedding_dim)
        else:
            tau = self.time_delay
            m = self.embedding_dim

        # Compute epsilon
        epsilon = self._get_epsilon(signal, m, tau)

        # Compute RQA
        rp_gen = recurrence_plot.RecurrencePlotGenerator(
            signal=signal,
            time_delay=tau,
            embedding_dim=m,
            epsilon=epsilon,
            path=None,
            unthresholded=True,
        )
        rqa, _ = rp_gen.compute_rqa_binary()
        return rqa

    def run_single(
        self,
        signal: np.ndarray,
        segment_id: str,
        patient_id: str,
        label: int,
        lead_time: int,
        single_scale: bool = True,
    ) -> MITBIHResult:
        """
        Run RQA analysis on a single signal.

        Args:
            signal: Cleaned and normalized signal
            segment_id: Segment identifier
            patient_id: Patient identifier
            label: 0 (normal) or 1 (pre-AF)
            lead_time: Lead time in minutes
            single_scale: If True, compute single-scale RQA; else multi-scale

        Returns:
            MITBIHResult object
        """
        result = MITBIHResult(
            segment_id=segment_id,
            patient_id=patient_id,
            label=label,
            lead_time=lead_time,
            single_scale=single_scale,
        )

        if single_scale:
            # Single-scale: RP -> RQA on full signal
            result.rqa_results["signal"] = self._compute_rqa(signal)
        else:
            # Multi-scale: SWT -> RP+RQA per subband
            subbands = self._reconstruct_subbands(signal)
            for name, subband_signal in subbands.items():
                result.rqa_results[name] = self._compute_rqa(subband_signal, subband_name=name)

        return result

    def run_batch(
        self,
        csv_path: Path,
        single_scale: bool = True,
        max_samples: Optional[int] = None,
    ) -> List[MITBIHResult]:
        """
        Run RQA analysis on all signals in a CSV file.

        Args:
            csv_path: Path to CSV file with cleaned signals
            single_scale: If True, single-scale; else multi-scale
            max_samples: Optional limit on number of samples to process

        Returns:
            List of MITBIHResult objects
        """
        signals, labels, segment_ids, patient_ids, lead_times = self.load_data_from_csv(csv_path)

        if max_samples:
            signals = signals[:max_samples]
            labels = labels[:max_samples]
            segment_ids = segment_ids[:max_samples]
            patient_ids = patient_ids[:max_samples]
            lead_times = lead_times[:max_samples]

        results = []
        mode = "Single-scale" if single_scale else "Multi-scale"

        for i in tqdm(range(len(signals)), desc=f"Processing {mode}"):
            try:
                result = self.run_single(
                    signal=signals[i],
                    segment_id=segment_ids[i],
                    patient_id=patient_ids[i],
                    label=int(labels[i]),
                    lead_time=int(lead_times[i]),
                    single_scale=single_scale,
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing {segment_ids[i]}: {e}")

        return results

    def results_to_csv(self, results: List[MITBIHResult], output_path: Path):
        """Save results to CSV."""
        if not results:
            print("No results to save.")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = [r.to_dict() for r in results]
        fieldnames = list(rows[0].keys())

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved {len(results)} results to {output_path}")

    def run_multiscale_batch(
        self,
        csv_path: Path,
        max_samples: Optional[int] = None,
    ) -> List[MITBIHResult]:
        """Run multi-scale RQA analysis."""
        return self.run_batch(csv_path=csv_path, single_scale=False, max_samples=max_samples)


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    # Configuration
    LEAD_TIME = 45  # Process data for this lead time

    # Subband parameters

    subband_params = {
        "D1": {"tau": 7, "m":4},
        "D2": {"tau": 4, "m": 6},
        "D3": {"tau": 2, "m": 5},
        "D4": {"tau": 1, "m": 7},
        "A4": {"tau": 20, "m": 9},
    }

    TIME_DELAY = 21
    EMBEDDING_DIM = 7
    EPSILON_METHOD = 'std'
    EPSILON_FACTOR = 0.2

    INPUT_CSV = OUTPUT_DIR / f'dataset_lt{LEAD_TIME}.csv'
    OUTPUT_SINGLE_SCALE = OUTPUT_DIR / f'rqa_single_scale_lt{LEAD_TIME}_tau_{TIME_DELAY}_embed_{EMBEDDING_DIM}.csv'
    OUTPUT_MULTI_SCALE = OUTPUT_DIR / f'rqa_multi_scale_lt{LEAD_TIME}_tau_{TIME_DELAY}_embed_{EMBEDDING_DIM}.csv'

    # Initialize pipeline
    pipeline = MITBIHRQAPipeline(
        time_delay=TIME_DELAY,
        embedding_dim=EMBEDDING_DIM,
        epsilon_method=EPSILON_METHOD,
        epsilon_factor=EPSILON_FACTOR,
        subband_params=subband_params,
    )

    # Run single-scale analysis
    print("\n" + "="*80)
    print("SINGLE-SCALE RQA ANALYSIS")
    print("="*80)
    results_single = pipeline.run_batch(csv_path=INPUT_CSV, single_scale=True)
    pipeline.results_to_csv(results_single, output_path=OUTPUT_SINGLE_SCALE)

    # Run multi-scale analysis
    print("\n" + "="*80)
    print("MULTI-SCALE RQA ANALYSIS")
    print("="*80)
    results_multi = pipeline.run_multiscale_batch(csv_path=INPUT_CSV)
    pipeline.results_to_csv(results_multi, output_path=OUTPUT_MULTI_SCALE)

    print("\n" + "="*80)
    print("RQA ANALYSIS COMPLETE")
    print("="*80)
