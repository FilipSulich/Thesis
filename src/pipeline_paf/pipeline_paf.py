"""
PAF Pipeline: Single-scale and Multi-scale RQA analysis for PAF dataset.

Pipeline 1 (single-scale): clean ECG (10s) -> RP -> RQA
Pipeline 2 (multi-scale): clean ECG (10s) -> SWT -> inverse SWT per sub-band -> RP per sub-band -> RQA per sub-band
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import h5py
from pathlib import Path
from scipy.spatial.distance import pdist
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wavelets.stationary_wavelet_transform as stationary_wavelet
import recurrence_plot.recurrence_plot_generation as recurrence_plot

DATA_PATH = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf/paf_segments_all_clean.h5')
OUTPUT_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf')

SAMPLING_FREQ = 128
SUBBAND_NAMES = ["D1", "D2", "D3", "D4", "A4"]


def extract_rqa_metrics(rqa_result) -> Dict[str, float]:
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


@dataclass
class PAFResult:
    segment_id: str
    patient_id: str
    label: int  # 0 = normal, 1 = pre-AF
    single_scale: bool
    rqa_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "segment_id": self.segment_id,
            "patient_id": self.patient_id,
            "label": self.label,
            "single_scale": self.single_scale,
        }
        for key, rqa in self.rqa_results.items():
            metrics = extract_rqa_metrics(rqa)
            for metric_name, value in metrics.items():
                result[f"{key}_{metric_name}"] = value
        return result



class PAFPipeline:
    def __init__(
        self,
        time_delay: int = 12,
        embedding_dim: int = 7,
        target_rr: float = 0.05,
        percentage=0.1,
        subband_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.time_delay = time_delay
        self.embedding_dim = embedding_dim
        self.target_rr = target_rr
        self.subband_params = subband_params
        self.percentage = percentage

    def load_data(self, split: str = 'all', use_filtered: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        if split == 'all':
            if use_filtered:
                path = OUTPUT_DIR / 'paf_segments_all_clean_filtered.h5'
            else:
                path = OUTPUT_DIR / 'paf_segments_all_clean.h5'
        else:
            path = OUTPUT_DIR / 'paf_segments.h5'

        with h5py.File(path, 'r') as f:
            grp = f[split]
            signals = grp['signals'][:]
            labels = grp['labels'][:]
            segment_ids = [s.decode() if isinstance(s, bytes) else s for s in grp['segment_ids'][:]]

            if 'patient_ids' in grp:
                patient_ids = [s.decode() if isinstance(s, bytes) else s for s in grp['patient_ids'][:]]
            else:
                patient_ids = [''] * len(signals)
        print("data loaded from " + str(path))
        return signals, labels, segment_ids, patient_ids

    def _reconstruct_subbands(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Decompose signal into SWT subbands."""
        swt = stationary_wavelet.SWT()
        coeffs = swt.decompose(signal, level=4)

        reconstructions = {}

        for i in range(len(coeffs)):
            zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
            reconstructions[f"D{i+1}"] = swt.reconstruct(zeroed)

        zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        reconstructions["A4"] = swt.reconstruct(zeroed)

        return reconstructions

    def get_epsilon_percentage_of_max(self, signal: np.ndarray, embedding_dim: int, time_delay: int) -> float:
        N = len(signal) - (embedding_dim - 1) * time_delay
        embedded = np.array([
            signal[i:i + embedding_dim * time_delay:time_delay]
            for i in range(N)
        ])
        distances = pdist(embedded, metric='euclidean')
        max_distance = float(np.max(distances))
        standard_deviation = np.std(distances)
        # const = 0.2 * standard_deviation
        return self.percentage * max_distance
        return const

    def _get_epsilon_for_target_rr(
        self,
        signal: np.ndarray,
        embedding_dim: int,
        time_delay: int,
        target_rr: float = 0.05,
    ) -> float:
        """Compute epsilon for target recurrence rate."""
        N = len(signal) - (embedding_dim - 1) * time_delay
        if N <= 1:
            raise ValueError("Signal too short for embedding.")

        embedded = np.array([
            signal[i:i + embedding_dim * time_delay:time_delay]
            for i in range(N)
        ])
        distances = pdist(embedded, metric="euclidean")
        return float(np.percentile(distances, target_rr * 100.0))

    def _compute_rqa(
        self,
        signal: np.ndarray,
        subband_name: Optional[str] = None,
        percentage: float = 0.10,
    ):
        if subband_name and self.subband_params and subband_name in self.subband_params:
            params = self.subband_params[subband_name]
            tau = params.get("tau", self.time_delay)
            m = params.get("m", self.embedding_dim)
            target_rr = params.get("target_rr", self.target_rr)
        else:
            tau = self.time_delay
            m = self.embedding_dim
            target_rr = self.target_rr

        # Use target RR percentile approach for z-score normalized signals
        # epsilon = self._get_epsilon_for_target_rr(signal, m, tau, target_rr)
        # epsilon = self._get_epsilon_for_target_rr(signal, m, tau, target_rr)

        epsilon = percentage * np.std(signal)
        # epsilon = self.get_epsilon_percentage_of_max(signal, m, tau)

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
        single_scale: bool = True,
            percentage: float = 0.10,
    ) -> PAFResult:
        result = PAFResult(
            segment_id=segment_id,
            patient_id=patient_id,
            label=label,
            single_scale=single_scale,
        )

        if single_scale:
            result.rqa_results["signal"] = self._compute_rqa(signal, percentage=percentage)
        else:
            subbands = self._reconstruct_subbands(signal)
            for name, subband_signal in subbands.items():
                result.rqa_results[name] = self._compute_rqa(subband_signal, subband_name=name, percentage=percentage)

        return result

    def run_batch(
        self,
        split: str = 'train',
        single_scale: bool = True,
        max_samples: Optional[int] = None,
        use_filtered: bool = True,
            percentage: float = 0.10,
    ) -> List[PAFResult]:
        signals, labels, segment_ids, patient_ids = self.load_data(split, use_filtered=use_filtered)
        if max_samples:
            signals = signals[:max_samples]
            labels = labels[:max_samples]
            segment_ids = segment_ids[:max_samples]
            patient_ids = patient_ids[:max_samples]

        results = []
        for i in tqdm(range(len(signals)), desc=f"Processing {split}"):
            try:
                result = self.run_single(
                    signal=signals[i],
                    segment_id=segment_ids[i],
                    patient_id=patient_ids[i],
                    label=int(labels[i]),
                    single_scale=single_scale,
                    percentage=percentage,

                )
                results.append(result)
            except Exception as e:
                print(f"Error processing {segment_ids[i]}: {e}")

        return results

    def results_to_csv(self, results: List[PAFResult], output_path: Path):
        """Save results to CSV."""
        import csv

        if not results:
            return
        if not output_path:
            output_path = Path(__file__).parent / "output.csv"

        rows = [r.to_dict() for r in results]
        fieldnames = list(rows[0].keys())

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved {len(results)} results to {output_path}")

    def decompose_to_csv(self, split: str = 'all', output_dir: Optional[Path] = None):

        import csv

        signals, labels, segment_ids, patient_ids = self.load_data(split)

        if output_dir is None:
            output_dir = OUTPUT_DIR / 'subbands'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_samples = signals.shape[1]
        subband_data = {name: [] for name in SUBBAND_NAMES}

        print(f"Decomposing {len(signals)} signals into subbands...")

        for idx in tqdm(range(len(signals)), desc="Decomposing"):
            signal = signals[idx]
            subbands = self._reconstruct_subbands(signal)

            for name in SUBBAND_NAMES:
                sub_signal = subbands[name]
                # Normalize subband
                sub_signal = (sub_signal - np.mean(sub_signal)) / (np.std(sub_signal, ddof=1) + 1e-8)

                row = {
                    'segment_id': segment_ids[idx],
                    'patient_id': patient_ids[idx],
                    'label': int(labels[idx]),
                }
                for j, val in enumerate(sub_signal[:n_samples]):
                    row[f's{j}'] = val

                subband_data[name].append(row)

        header = ['segment_id', 'patient_id', 'label'] + [f's{i}' for i in range(n_samples)]

        for name in SUBBAND_NAMES:
            output_path = output_dir / f'{name}.csv'
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(subband_data[name])
            print(f"Saved {name} to {output_path}")

        print(f"\nDone! Subband CSVs saved to {output_dir}")

    def run_multiscale_batch(
        self,
        split: str = 'all',
        max_samples: Optional[int] = None,
            percentage: float = 0.10,
    ) -> List[PAFResult]:
        return self.run_batch(split=split, single_scale=False, max_samples=max_samples, percentage=percentage)



if __name__ == '__main__':
    subband_params = {
        "D1": {"tau": 6, "m": 7, "target_rr": 0.1},
        "D2": {"tau": 4, "m": 7, "target_rr": 0.1},
        "D3": {"tau": 2, "m": 7, "target_rr": 0.1},
        "D4": {"tau": 2, "m": 7, "target_rr": 0.1},
        "A4": {"tau": 16, "m": 8, "target_rr": 0.1},
    }

    #for single scale, optimal tau=12,m=7
    pipeline = PAFPipeline(
        time_delay=3,
        embedding_dim=3,
        target_rr=0.05,
        percentage= 0.2,
        subband_params=subband_params,
    )

    # pipeline.decompose_to_csv(split='all')

    # results = pipeline.run_batch(split='all', single_scale=True)
    results = pipeline.run_multiscale_batch(split='all')

    output_path = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/output_new/multi_scale_20_std.csv"
    pipeline.results_to_csv(results, output_path=output_path)