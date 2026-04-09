from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import h5py
import csv
from pathlib import Path
from scipy.spatial.distance import pdist
from tqdm import tqdm

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from src.recurrence_plot.AMI import AMI
from src.recurrence_plot.FNN import FNN
from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.wavelets.stationary_wavelet_transform as stationary_wavelet
import src.recurrence_plot.recurrence_plot_generation as recurrence_plot
from src.data_cleaning.data_cleaner import DataCleaner

OUTPUT_DIR = Path('/data/data_new/exp2')
HDF5_PATH = OUTPUT_DIR / 'paf_with_quality.h5'

SAMPLING_FREQ = 128
WAVELET = 'db4'
SWT_LEVEL = 4

ANALYSIS_SUBBANDS = ['D1', 'D2', 'D3', 'D4', 'A4']

DEFAULT_EPSILON_PERCENTILE = 5.0


@dataclass
class RQAWithEpsilon:
    rqa: Any
    epsilon: float


@dataclass
class PAFResult:
    segment_id: str
    patient_id: str
    label: int
    single_scale: bool
    method: str
    rqa_results: Dict[str, RQAWithEpsilon] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        row = {
            "segment_id": self.segment_id,
            "patient_id": self.patient_id,
            "label": self.label,
            "single_scale": self.single_scale,
            "method": self.method,
        }
        for key, result in self.rqa_results.items():
            row[f"{key}_epsilon"] = result.epsilon
            for metric_name, value in _extract_rqa_metrics(result.rqa).items():
                row[f"{key}_{metric_name}"] = value
        return row


def _extract_rqa_metrics(rqa_result) -> Dict[str, float]:
    return {
        "recurrence_rate": rqa_result.recurrence_rate,
        "determinism": rqa_result.determinism,
        "longest_diagonal_line": rqa_result.longest_diagonal_line,
        "divergence": rqa_result.divergence,
        "laminarity": rqa_result.laminarity,
        "trapping_time": rqa_result.trapping_time,
    }


class PAFPipeline:
    def __init__(self, epsilon_percentile: float = DEFAULT_EPSILON_PERCENTILE):
        self.epsilon_percentile = epsilon_percentile
        self.cleaner = DataCleaner(fs=SAMPLING_FREQ)

    def load_data(self, category: str = 'all') -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Modified loader using your specific record_id logic.
        Categories: 'pre_af' (p-even), 'nsr' (n01-n25), 'af' (p*c), or 'all'.
        """
        with h5py.File(HDF5_PATH, 'r') as f:
            grp = f['all']
            signals = grp['signals'][:]
            labels = grp['labels'][:]
            segment_ids = [s.decode() if isinstance(s, bytes) else s for s in grp['segment_ids'][:]]
            patient_ids = [s.decode() if isinstance(s, bytes) else s for s in grp['patient_ids'][:]]
            record_ids = [s.decode() if isinstance(s, bytes) else s for s in grp['record_ids'][:]]

        keep_indices = []

        for i, rec_id in enumerate(record_ids):
            rec_id_lower = rec_id.lower()
            is_match = False

            # 1. NSR Logic (n01-n25)
            if rec_id_lower.startswith('n') and not rec_id_lower.endswith('c'):
                try:
                    num_str = ''.join(filter(str.isdigit, rec_id_lower[1:]))
                    if num_str and 1 <= int(num_str) <= 25:
                        if category in ['nsr', 'all']:
                            is_match = True
                except ValueError:
                    pass

            # 2. AF Logic (p*c)
            elif rec_id_lower.startswith('p') and rec_id_lower.endswith('c'):
                if category in ['af', 'all']:
                    is_match = True

            # 3. Pre-AF Logic (p-even)
            elif rec_id_lower.startswith('p') and not rec_id_lower.endswith('c'):
                try:
                    num_str = ''.join(filter(str.isdigit, rec_id_lower[1:]))
                    if num_str and int(num_str) % 2 == 0:
                        if category in ['pre_af', 'all']:
                            is_match = True
                except ValueError:
                    pass

            if is_match:
                keep_indices.append(i)

        # Filtering the arrays
        signals = signals[keep_indices]
        labels = labels[keep_indices]
        segment_ids = [segment_ids[i] for i in keep_indices]
        patient_ids = [patient_ids[i] for i in keep_indices]

        print(f"Loaded {len(signals)} segments for category: {category}")
        return signals, labels, segment_ids, patient_ids

    def _decompose(self, signal: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        swt = stationary_wavelet.SWT(wavelet=WAVELET)
        coeffs = swt.decompose(signal, level=SWT_LEVEL)
        subbands = {}
        for i in range(SWT_LEVEL):
            zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
            label = f"D{SWT_LEVEL - i}"
            subbands[label] = self.cleaner.preprocess_subband(swt.reconstruct(zeroed))
        zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        subbands['A4'] = self.cleaner.preprocess_subband(swt.reconstruct(zeroed))
        return subbands

    def get_epsilon(self, signal, tau, m, target_rr=0.01):
        return RecurrencePlotGenerator.find_epsilon_for_target_rr(signal, tau, m, target_rr)

    def get_optimal_params(self, signal):
        ami = AMI(signal)
        time_delay = ami.get_first_minimum()
        fnn = FNN(signal=signal, tau=time_delay)
        embedding_dim = fnn.get_optimal_dimension()
        return time_delay, embedding_dim

    def _compute_rqa(self, signal: np.ndarray, tau: int, m: int) -> RQAWithEpsilon:
        epsilon = self.get_epsilon(signal, tau, m)
        rp_gen = recurrence_plot.RecurrencePlotGenerator(
            signal=signal, time_delay=tau, embedding_dim=m,
            epsilon=epsilon, path=None, unthresholded=True,
        )
        rqa, _ = rp_gen.compute_rqa_binary()
        return RQAWithEpsilon(rqa=rqa, epsilon=epsilon)

    def _run_single_scale(self, signal: np.ndarray) -> Dict[str, RQAWithEpsilon]:
        tau, m = self.get_optimal_params(signal)
        return {'signal': self._compute_rqa(signal, tau, m)}

    def _run_multi_scale(self, signal: np.ndarray) -> Optional[Dict[str, RQAWithEpsilon]]:
        subbands = self._decompose(signal)
        results = {}
        for name in ANALYSIS_SUBBANDS:
            sb = subbands.get(name)
            if sb is None: continue
            tau, m = self.get_optimal_params(sb)

            signal_norm = self.normalize_signal(sb)

            results[name] = self._compute_rqa(signal_norm, tau, m)
        return results

    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        return (signal - np.mean(signal)) / np.std(signal)

    def _process_segment(self, data: Tuple, method: str, scale_fn, is_single_scale) -> Optional[PAFResult]:
        signal, label, segment_id, patient_id = data
        try:
            rqa_results = scale_fn(signal)
            if rqa_results is None: return None
            return PAFResult(
                segment_id=segment_id, patient_id=patient_id,
                label=int(label), single_scale=is_single_scale,
                method=method, rqa_results=rqa_results,
            )
        except Exception as e:
            print(f"  Error on {segment_id}: {e}")
            return None

    def run_batch(
            self,
            method: str = 'single_scale',
            category: str = 'all',
            max_samples: Optional[int] = None,
            n_jobs: int = -1
    ) -> List[PAFResult]:
        signals, labels, segment_ids, patient_ids = self.load_data(category=category)

        if max_samples:
            signals, labels = signals[:max_samples], labels[:max_samples]
            segment_ids, patient_ids = segment_ids[:max_samples], patient_ids[:max_samples]

        method_map = {
            'single_scale': (self._run_single_scale, "Single-scale RQA", True),
            'swt_recon': (self._run_multi_scale, "Multi-scale RQA (SWT+Recon)", False),
        }

        scale_fn, desc, is_single_scale = method_map[method]
        data_list = list(zip(signals, labels, segment_ids, patient_ids))
        workers = os.cpu_count() if n_jobs == -1 else n_jobs

        worker_fn = partial(self._process_segment, method=method, scale_fn=scale_fn, is_single_scale=is_single_scale)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            raw_results = list(
                tqdm(executor.map(worker_fn, data_list), total=len(data_list), desc=f"{desc} [{category}]"))

        results = [r for r in raw_results if r is not None]
        print(f"Done — {len(results)} processed, {len(data_list) - len(results)} skipped")
        return results

    def results_to_csv(self, results: List[PAFResult], output_path: Path):
        if not results: return
        rows = [r.to_dict() for r in results]
        fieldnames = list(rows[0].keys())
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    pipeline = PAFPipeline()
    results_dir = OUTPUT_DIR / 'results_parallel'

    preaf_results = pipeline.run_batch(method='single_scale', category='pre_af')
    nsr_results_single = pipeline.run_batch(method='single_scale', category='nsr')
    #
    pipeline.results_to_csv(preaf_results, results_dir / 'preaf_single_rqa.csv')
    pipeline.results_to_csv(nsr_results_single, results_dir / 'nsr_single_rqa.csv')

    # Example: Run specifically for NSR (Healthy) segments
    # nsr_results = pipeline.run_batch(method='swt_recon', category='nsr')
    # preaf_results_multi = pipeline.run_batch(method='swt_recon', category='pre_af')

    # pipeline.results_to_csv(nsr_results, results_dir / 'nsr_swt_rqa.csv')
    # pipeline.results_to_csv(preaf_results_multi, results_dir / 'preaf_swt_rqa.csv')