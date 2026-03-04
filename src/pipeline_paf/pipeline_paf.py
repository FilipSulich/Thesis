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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wavelets.stationary_wavelet_transform as stationary_wavelet
import recurrence_plot.recurrence_plot_generation as recurrence_plot
from data_cleaning.data_cleaner import DataCleaner

OUTPUT_DIR    = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf_updated')
HDF5_PATH     = OUTPUT_DIR / 'paf_segments_all_clean_new_cleaning.h5'


SAMPLING_FREQ = 128
WAVELET       = 'db4'
SWT_LEVEL     = 4

ALL_SUBBANDS     = ['D1', 'D2', 'D3', 'D4', 'A4']
ANALYSIS_SUBBANDS = ['D1', 'D2', 'D3', 'D4', 'A4']

DEFAULT_SINGLE_SCALE_PARAMS = {'tau': 12, 'm': 7}

DEFAULT_SUBBAND_PARAMS = {
    'D1': {'tau': 1,  'm': 4},
    'D2': {'tau': 2,  'm': 4},
    'D3': {'tau': 3,  'm': 4},
    'D4': {'tau': 6,  'm': 5},
    'A4': {'tau': 32, 'm': 8},
}
DEFAULT_EPSILON_PERCENTILE = 5.0
epsilon_distance = 0.05

def _extract_rqa_metrics(rqa_result) -> Dict[str, float]:
    return {
        "recurrence_rate":        rqa_result.recurrence_rate,
        "determinism":            rqa_result.determinism,
        "average_diagonal_line":  rqa_result.average_diagonal_line,
        "longest_diagonal_line":  rqa_result.longest_diagonal_line,
        "divergence":             rqa_result.divergence,
        "entropy_diagonal_lines": rqa_result.entropy_diagonal_lines,
        "laminarity":             rqa_result.laminarity,
        "trapping_time":          rqa_result.trapping_time,
        "longest_vertical_line":  rqa_result.longest_vertical_line,
        "entropy_vertical_lines": rqa_result.entropy_vertical_lines,
    }


@dataclass
class RQAWithEpsilon:
    rqa:     Any
    epsilon: float


@dataclass
class PAFResult:
    segment_id:   str
    patient_id:   str
    label:        int
    single_scale: bool
    rqa_results:  Dict[str, RQAWithEpsilon] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        row = {
            "segment_id":   self.segment_id,
            "patient_id":   self.patient_id,
            "label":        self.label,
            "single_scale": self.single_scale,
        }
        for key, result in self.rqa_results.items():
            row[f"{key}_epsilon"] = result.epsilon
            for metric_name, value in _extract_rqa_metrics(result.rqa).items():
                row[f"{key}_{metric_name}"] = value
        return row

class PAFPipeline:
    def __init__(
        self,
        subband_params: Dict[str, Dict[str, Any]] = None,
        single_scale_params: Dict[str, Any] = None,
        epsilon_percentile: float = DEFAULT_EPSILON_PERCENTILE,
    ):
        self.subband_params      = subband_params or DEFAULT_SUBBAND_PARAMS
        self.single_scale_params = single_scale_params or DEFAULT_SINGLE_SCALE_PARAMS
        self.epsilon_percentile  = epsilon_percentile
        self.cleaner             = DataCleaner(fs=SAMPLING_FREQ)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        with h5py.File(HDF5_PATH, 'r') as f:
            grp         = f['all']
            signals     = grp['signals'][:]
            labels      = grp['labels'][:]
            print(f['all'].attrs['n_preaf'], f['all'].attrs['n_normal'])
            segment_ids = [s.decode() if isinstance(s, bytes) else s
                           for s in grp['segment_ids'][:]]
            patient_ids = (
                [s.decode() if isinstance(s, bytes) else s
                 for s in grp['patient_ids'][:]]
                if 'patient_ids' in grp else [''] * len(signals)
            )
        print(f"Loaded {len(signals)} — "
              f"Normal: {sum(labels==0)}, Pre-AF: {sum(labels==1)}")
        print(f"HDF5: {HDF5_PATH}")
        return signals, labels, segment_ids, patient_ids

    def _decompose(self, signal: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        swt    = stationary_wavelet.SWT(wavelet=WAVELET)
        coeffs = swt.decompose(signal, level=SWT_LEVEL)

        subbands = {}
        for i in range(SWT_LEVEL):
            zeroed    = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
            label = f"D{SWT_LEVEL - i}"
            subbands[label] = self.cleaner.preprocess_subband(swt.reconstruct(zeroed))

        zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        subbands['A4'] = self.cleaner.preprocess_subband(swt.reconstruct(zeroed))

        return subbands

    def _compute_epsilon(self, signal: np.ndarray, tau: int, m: int) -> float:
        n_points = len(signal) - (m - 1) * tau
        if n_points <= 0:
            raise ValueError(
                f"Signal too short for tau={tau}, m={m}: "
                f"need >{(m-1)*tau} samples, got {len(signal)}"
            )
        embedded = np.array([signal[i: i + m * tau: tau] for i in range(n_points)])
        if len(embedded) > 2000:
            idx      = np.linspace(0, len(embedded) - 1, 2000, dtype=int)
            embedded = embedded[idx]
        # distance = np.max(signal) * epsilon_distance
        percentile = float(np.percentile(pdist(embedded, metric='euclidean'),self.epsilon_percentile))
        return percentile

    def _compute_rqa(self, signal: np.ndarray, tau: int, m: int) -> RQAWithEpsilon:
        epsilon = self._compute_epsilon(signal, tau, m)
        rp_gen  = recurrence_plot.RecurrencePlotGenerator(
            signal=signal,
            time_delay=tau,
            embedding_dim=m,
            epsilon=epsilon,
            path=None,
            unthresholded=True,
        )
        rqa, _ = rp_gen.compute_rqa_binary()
        return RQAWithEpsilon(rqa=rqa, epsilon=epsilon)

    def _run_single_scale(self, signal: np.ndarray) -> Dict[str, RQAWithEpsilon]:
        tau = self.single_scale_params['tau']
        m   = self.single_scale_params['m']
        return {'signal': self._compute_rqa(signal, tau, m)}

    def _run_multi_scale(
        self, signal: np.ndarray
    ) -> Optional[Dict[str, RQAWithEpsilon]]:
        subbands = self._decompose(signal)

        for name in ['D3', 'D4', 'A4']:
            if subbands.get(name) is None:
                return None

        results = {}
        for name in ANALYSIS_SUBBANDS:
            sb = subbands.get(name)
            if sb is None:
                continue
            params      = self.subband_params[name]
            results[name] = self._compute_rqa(sb, params['tau'], params['m'])

        return results

    def run_batch(
        self,
        single_scale: bool = True,
        max_samples: Optional[int] = None,
    ) -> List[PAFResult]:

        signals, labels, segment_ids, patient_ids = self.load_data()

        if max_samples:
            signals     = signals[:max_samples]
            labels      = labels[:max_samples]
            segment_ids = segment_ids[:max_samples]
            patient_ids = patient_ids[:max_samples]

        scale_fn = self._run_single_scale if single_scale else self._run_multi_scale
        desc     = "Single-scale RQA" if single_scale else "Multi-scale RQA"
        results, skipped = [], 0

        for i in tqdm(range(len(signals)), desc=desc):
            try:
                rqa_results = scale_fn(signals[i])
                if rqa_results is None:
                    skipped += 1
                    continue
                results.append(PAFResult(
                    segment_id=segment_ids[i],
                    patient_id=patient_ids[i],
                    label=int(labels[i]),
                    single_scale=single_scale,
                    rqa_results=rqa_results,
                ))
            except Exception as e:
                print(f"  Error on {segment_ids[i]}: {e}")
                skipped += 1

        print(f"Done — {len(results)} processed, {skipped} skipped")
        return results

    def results_to_csv(self, results: List[PAFResult], output_path: Path):
        if not results:
            print("No results to save.")
            return
        rows       = [r.to_dict() for r in results]
        fieldnames = list(rows[0].keys())
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved results to {output_path}")


if __name__ == '__main__':
    pipeline = PAFPipeline()

    results_ss = pipeline.run_batch(single_scale=True)
    pipeline.results_to_csv(results_ss, OUTPUT_DIR / 'results' / 'single_scale_rqa_5_dist.csv')

    results_ms = pipeline.run_batch(single_scale=False)
    pipeline.results_to_csv(results_ms, OUTPUT_DIR / 'results' / 'multi_scale_rqa_5_dist.csv')
