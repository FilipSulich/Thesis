"""
MIT-BIH RQA Pipeline

Simplified pipeline for RQA analysis:
- Single-scale: Clean ECG → RP → RQA
- Multi-scale: Clean ECG → SWT subbands → RP per subband → RQA per subband

Usage:
    pipeline = MITBIHRQAPipeline(epsilon_method='target_rr', target_rr=0.05)
    pipeline.process(lead_times=[5, 10], mode='both')
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
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

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/mitbih')
OUTPUT_DIR = DATA_DIR / 'results_min_max'

SAMPLING_FREQ = 250


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
# Pipeline Class
# =============================================================================

class MITBIHRQAPipeline:
    """
    Simplified MIT-BIH RQA Pipeline.

    Features:
    - Single-scale and multi-scale analysis
    - Multiple epsilon methods: 'std', 'max_distance', 'target_rr'
    - Process single lead time or multiple lead times
    """

    def __init__(
        self,
        time_delay: int = 21,
        embedding_dim: int = 7,
        epsilon_method: str = 'std',
        epsilon_factor: float = 0.2,
        target_rr: float = 0.05,
        subband_params: Optional[Dict[str, Dict]] = None,
        wavelet: str = 'db4',
        level: int = 4,
        target_density: float = 0.01
    ):
        """
        Initialize RQA pipeline.

        Args:
            time_delay: Time delay (tau)
            embedding_dim: Embedding dimension (m)
            epsilon_method: 'std', 'max_distance', or 'target_rr'
            epsilon_factor: Factor for std/max_distance methods
            target_rr: Target recurrence rate for 'target_rr' method
            subband_params: Parameters per subband {subband_name: {tau, m}}
            wavelet: Wavelet type for SWT
            level: SWT decomposition level
        """
        self.time_delay = time_delay
        self.embedding_dim = embedding_dim
        self.epsilon_method = epsilon_method
        self.epsilon_factor = epsilon_factor
        self.target_rr = target_rr
        self.subband_params = subband_params or {}
        self.wavelet = wavelet
        self.level = level
        self.target_density = target_density

        print(f"\n{'='*80}")
        print(f"MIT-BIH RQA Pipeline")
        print(f"  Epsilon method: {epsilon_method}")
        if epsilon_method == 'target_rr':
            print(f"  Target RR: {target_rr}")
        else:
            print(f"  Epsilon factor: {epsilon_factor}")
        print(f"  Default tau: {time_delay}, m: {embedding_dim}")
        print(f"  Wavelet: {wavelet}, Level: {level}")
        print(f"{'='*80}\n")

    def _load_csv(self, csv_path: Path) -> Tuple[List, List, List, List, List]:
        """Load data from CSV."""
        signals, labels, segment_ids, patient_ids, lead_times = [], [], [], [], []

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                signal_cols = [k for k in row.keys() if k.startswith('s') and k[1:].isdigit()]
                signal = np.array([float(row[col]) for col in sorted(signal_cols, key=lambda x: int(x[1:]))])

                signals.append(signal)
                labels.append(int(row['label']))
                segment_ids.append(row['segment_id'])
                patient_ids.append(row['patient_id'])
                lead_times.append(int(row['lead_time']))

        return signals, labels, segment_ids, patient_ids, lead_times

    def _decompose_subbands(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Decompose signal into normalized subbands."""
        swt = stationary_wavelet.SWT(wavelet=self.wavelet, level=self.level)
        coeffs = swt.decompose(signal, level=self.level)

        subbands = {}

        # Details D1-D4
        for i in range(len(coeffs)):
            zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
            subband = swt.reconstruct(zeroed)
            subband = (subband - np.mean(subband)) / (np.std(subband) + 1e-8)
            subbands[f"D{i+1}"] = subband

        # Approximation A4
        zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        subband = swt.reconstruct(zeroed)
        subband = (subband - np.mean(subband)) / (np.std(subband) + 1e-8)
        subbands["A4"] = subband

        return subbands

    def _compute_epsilon(
        self,
        signal: np.ndarray,
        tau: int,
        m: int,
    ) -> float:
        """
        Compute epsilon based on method.

        Methods:
        - 'std': epsilon = factor * std(signal)
        - 'max_distance': epsilon = factor * max_distance(embedded)
        - 'target_rr': binary search to achieve target recurrence rate
        """
        if self.epsilon_method == 'std':
            print(self.epsilon_factor * np.std(signal))
            return self.epsilon_factor * np.std(signal)

        elif self.epsilon_method == 'max_distance':
            N = len(signal) - (m - 1) * tau
            if N <= 1:
                return self.epsilon_factor * np.std(signal)
            embedded = np.array([signal[i:i + m * tau:tau] for i in range(N)])
            distances = pdist(embedded, metric='euclidean')
            return self.epsilon_factor * np.max(distances)

        elif self.epsilon_method == 'target_rr':
            # Use binary search to find epsilon that gives target RR
            # Start with std-based estimate
            epsilon = self.epsilon_from_recurrence_density(signal, m=m, tau=tau, target_density=self.target_density)
            return epsilon

        else:
            return self.epsilon_factor * np.std(signal)

    def _find_epsilon_for_target_rr(
        self,
        signal: np.ndarray,
        tau: int,
        m: int,
        max_iterations: int = 10
    ) -> float:
        """Binary search for epsilon that achieves target recurrence rate."""
        # Initial bounds
        eps_low = 0.01 * np.std(signal)
        eps_high = 2.0 * np.std(signal)

        for _ in range(max_iterations):
            eps_mid = (eps_low + eps_high) / 2

            # Compute RR with this epsilon
            rp_gen = recurrence_plot.RecurrencePlotGenerator(
                signal=signal,
                time_delay=tau,
                embedding_dim=m,
                epsilon=eps_mid,
                path=None,
                unthresholded=True,
            )
            rqa, _ = rp_gen.compute_rqa_binary()
            current_rr = rqa.recurrence_rate

            if abs(current_rr - self.target_rr) < 0.005:  # Within 0.5%
                return eps_mid

            if current_rr < self.target_rr:
                eps_low = eps_mid
            else:
                eps_high = eps_mid

        return (eps_low + eps_high) / 2

    import numpy as np
    from scipy.spatial.distance import pdist

    def embed_signal(self, signal, m, tau):
        """
        Reconstruct phase space using time-delay embedding.
        Returns matrix of shape (N - (m-1)*tau, m)
        """
        N = len(signal)
        M = N - (m - 1) * tau
        return np.array([signal[i:i + (m - 1) * tau + 1:tau] for i in range(M)])

    def epsilon_from_recurrence_density(self, signal, m, tau, target_density=0.01):
        """
        Estimate epsilon threshold such that the recurrence point density
        is approximately equal to target_density (default 1%).

        Parameters
        ----------
        signal : np.ndarray
            1D input signal (already preprocessed and normalized)
        m : int
            Embedding dimension
        tau : int
            Time delay
        target_density : float
            Target recurrence point density (default 0.01 = 1%)

        Returns
        -------
        epsilon : float
            Distance threshold corresponding to the target recurrence density
        """
        # Step 1: Phase space reconstruction
        embedded = self.embed_signal(signal, m, tau)

        # Step 2: Compute all pairwise Euclidean distances (upper triangle only)
        distances = pdist(embedded, metric='euclidean')

        # Step 3: Find the percentile of the distance distribution
        # that corresponds to the target recurrence density
        # Since the recurrence matrix is symmetric with ones on diagonal,
        # the off-diagonal recurrence rate equals the fraction of distances below epsilon.
        # pdist gives us the upper triangle, which is an unbiased sample of all off-diagonal distances.
        epsilon = np.percentile(distances, target_density * 100)

        return epsilon

    def _compute_rqa(
        self,
        signal: np.ndarray,
        subband_name: Optional[str] = None
    ):
        """Compute RQA for signal."""
        # Get parameters
        if subband_name and subband_name in self.subband_params:
            tau = self.subband_params[subband_name].get("tau", self.time_delay)
            m = self.subband_params[subband_name].get("m", self.embedding_dim)
        else:
            tau = self.time_delay
            m = self.embedding_dim

        # Compute epsilon
        epsilon = self._compute_epsilon(signal, tau, m)

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

    def _process_single(
        self,
        signal: np.ndarray,
        segment_id: str,
        patient_id: str,
        label: int,
        lead_time: int,
        single_scale: bool
    ) -> Dict:
        """Process single signal and return metrics."""
        result = {
            "segment_id": segment_id,
            "patient_id": patient_id,
            "label": label,
            "lead_time": lead_time,
        }

        if single_scale:
            # Single-scale RQA
            rqa = self._compute_rqa(signal)
            metrics = extract_rqa_metrics(rqa)
            for k, v in metrics.items():
                result[f"signal_{k}"] = v
        else:
            # Multi-scale RQA
            subbands = self._decompose_subbands(signal)
            for name, subband_signal in subbands.items():
                rqa = self._compute_rqa(subband_signal, subband_name=name)
                metrics = extract_rqa_metrics(rqa)
                for k, v in metrics.items():
                    result[f"{name}_{k}"] = v

        return result

    def process(
        self,
        lead_times: List[int],
            epsilon_method: str,
        mode: str = 'both'
    ):
        for lead_time in lead_times:
            csv_path = OUTPUT_DIR / f'dataset_lt{lead_time}.csv'

            if not csv_path.exists():
                print(f"Warning: {csv_path} not found, skipping...")
                continue

            signals, labels, segment_ids, patient_ids, lead_times_data = self._load_csv(csv_path)

            # Single-scale
            if mode in ['single', 'both']:
                print(f"\nSingle-scale RQA...")
                results_single = []

                for i in tqdm(range(len(signals)), desc="Processing"):
                    try:
                        result = self._process_single(
                            signal=signals[i],
                            segment_id=segment_ids[i],
                            patient_id=patient_ids[i],
                            label=labels[i],
                            lead_time=lead_time,
                            single_scale=True
                        )
                        results_single.append(result)
                    except Exception as e:
                        print(f"Error: {segment_ids[i]}: {e}")

                output_path = OUTPUT_DIR / f'rqa_single_scale_lt{lead_time}_{epsilon_method}.csv'
                self._save_csv(results_single, output_path)

            if mode in ['multi', 'both']:
                print(f"\nMulti-scale RQA...")
                results_multi = []

                for i in tqdm(range(len(signals)), desc="Processing"):
                    try:
                        result = self._process_single(
                            signal=signals[i],
                            segment_id=segment_ids[i],
                            patient_id=patient_ids[i],
                            label=labels[i],
                            lead_time=lead_time,
                            single_scale=False
                        )
                        results_multi.append(result)
                    except Exception as e:
                        print(f"Error: {segment_ids[i]}: {e}")

                output_path = OUTPUT_DIR / f'rqa_multi_scale_lt{lead_time}_{epsilon_method}.csv'
                self._save_csv(results_multi, output_path)

    def _save_csv(self, results: List[Dict], output_path: Path):
        """Save results to CSV."""
        if not results:
            print("No results to save.")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(results[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"  Saved: {output_path}")


if __name__ == '__main__':
    LEAD_TIMES = [5,10,15,30,45]

    subband_params = {
        "D1": {"tau": 7, "m": 4},
        "D2": {"tau": 4, "m": 6},
        "D3": {"tau": 2, "m": 5},
        "D4": {"tau": 1, "m": 7},
        "A4": {"tau": 20, "m": 9},
    }
    epsilon_method = "max_distance"
    pipeline = MITBIHRQAPipeline(
        time_delay=7,
        embedding_dim=3,
        epsilon_method=epsilon_method,
        epsilon_factor=0.05,
        target_rr=0.05,
        subband_params=subband_params,
        wavelet='db4',
        level=4,
        target_density= 0.05
    )

    pipeline.process(
        lead_times=LEAD_TIMES,
        mode='both',
        epsilon_method=epsilon_method,
    )
