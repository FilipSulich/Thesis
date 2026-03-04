"""
PAF Pipeline: Single-scale and Multi-scale RQA analysis.

Records epsilon per subband alongside RQA metrics.
All subbands D1-D4 and A4 are computed; D1/D2 are included
in output so their null result can be reported explicitly.

SWT coefficient ordering (verified by PSD):
  coeffs[0] -> D4 (4-8 Hz)
  coeffs[1] -> D3 (8-16 Hz)
  coeffs[2] -> D2 (16-32 Hz)
  coeffs[3] -> D1 (32-64 Hz)
  A4        -> 0-4 Hz

Verified embedding parameters:
  Raw: tau=12, m=7
  D4:  tau=6,  m=5
  D3:  tau=3,  m=4
  D2:  tau=2,  m=4   (expected null result — above 40Hz lowpass)
  D1:  tau=1,  m=4   (expected null result — above 40Hz lowpass)
  A4:  tau=32, m=8

Epsilon: 10th percentile of pairwise distances in phase space.
"""

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

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR    = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf_updated')
HDF5_PATH     = OUTPUT_DIR / 'paf_segments_all_clean_new_cleaning.h5'


SAMPLING_FREQ = 128
WAVELET       = 'db4'
SWT_LEVEL     = 4

# All subbands — D1/D2 included to document null results explicitly
ALL_SUBBANDS     = ['D1', 'D2', 'D3', 'D4', 'A4']
ANALYSIS_SUBBANDS = ['D1', 'D2', 'D3', 'D4', 'A4']  # all included in output

# Verified embedding parameters
DEFAULT_SINGLE_SCALE_PARAMS = {'tau': 12, 'm': 7}
DEFAULT_SUBBAND_PARAMS = {
    'D1': {'tau': 1,  'm': 4},   # 32-64 Hz — expected null, above lowpass
    'D2': {'tau': 2,  'm': 4},   # 16-32 Hz — expected null, above lowpass
    'D3': {'tau': 3,  'm': 4},   # 8-16 Hz
    'D4': {'tau': 6,  'm': 5},   # 4-8 Hz
    'A4': {'tau': 32, 'm': 8},   # 0-4 Hz
}
DEFAULT_SUBBAND_PARAMS2 = {
    'D1': {'tau': 1,  'm': 4},   # 32-64 Hz — expected null, above lowpass
    'D2': {'tau': 2,  'm': 4},   # 16-32 Hz — expected null, above lowpass
    'D3': {'tau': 3,  'm': 4},   # 8-16 Hz
    'D4': {'tau': 6,  'm': 5},   # 4-8 Hz
    'A4': {'tau': 32, 'm': 8},   # 0-4 Hz
}
DEFAULT_EPSILON_PERCENTILE = 5.0
epsilon_distance = 0.05


# =============================================================================
# RQA helpers
# =============================================================================

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
    """Wraps an RQA result with the epsilon value used to compute it."""
    rqa:     Any
    epsilon: float


@dataclass
class PAFResult:
    segment_id:   str
    patient_id:   str
    label:        int
    single_scale: bool
    # values are RQAWithEpsilon instances
    rqa_results:  Dict[str, RQAWithEpsilon] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        row = {
            "segment_id":   self.segment_id,
            "patient_id":   self.patient_id,
            "label":        self.label,
            "single_scale": self.single_scale,
        }
        for key, result in self.rqa_results.items():
            # Record epsilon first so columns are grouped naturally
            row[f"{key}_epsilon"] = result.epsilon
            for metric_name, value in _extract_rqa_metrics(result.rqa).items():
                row[f"{key}_{metric_name}"] = value
        return row


# =============================================================================
# Pipeline
# =============================================================================

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

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Subband decomposition
    # ------------------------------------------------------------------

    def _decompose(self, signal: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """
        SWT decomposition. Verified coefficient ordering:
          coeffs[0]=D4, coeffs[1]=D3, coeffs[2]=D2, coeffs[3]=D1
        """
        swt    = stationary_wavelet.SWT(wavelet=WAVELET)
        coeffs = swt.decompose(signal, level=SWT_LEVEL)

        subbands = {}
        for i in range(SWT_LEVEL):
            zeroed    = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
            label = f"D{SWT_LEVEL - i}"   # i=0->D4, i=1->D3, i=2->D2, i=3->D1
            subbands[label] = self.cleaner.preprocess_subband(swt.reconstruct(zeroed))

        zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        subbands['A4'] = self.cleaner.preprocess_subband(swt.reconstruct(zeroed))

        return subbands

    # ------------------------------------------------------------------
    # Epsilon and RQA
    # ------------------------------------------------------------------

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
        distance = np.max(signal) * epsilon_distance
        # percentile = float(np.percentile(pdist(embedded, metric='euclidean'),self.epsilon_percentile))
        return distance

    def _compute_rqa(self, signal: np.ndarray, tau: int, m: int) -> RQAWithEpsilon:
        """Compute RQA and return result together with the epsilon used."""
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

    # ------------------------------------------------------------------
    # Single segment processing
    # ------------------------------------------------------------------

    def _run_single_scale(self, signal: np.ndarray) -> Dict[str, RQAWithEpsilon]:
        tau = self.single_scale_params['tau']
        m   = self.single_scale_params['m']
        return {'signal': self._compute_rqa(signal, tau, m)}

    def _run_multi_scale(
        self, signal: np.ndarray
    ) -> Optional[Dict[str, RQAWithEpsilon]]:
        """
        Decompose and compute RQA for all subbands.

        D1/D2 are included even though they are expected nulls — their
        epsilon and RQA values document the frequency localization finding
        explicitly: subbands above the 40 Hz lowpass show no structure.

        Returns None only if a primary analysis subband (D3, D4, A4)
        has insufficient energy after preprocessing.
        """
        subbands = self._decompose(signal)

        # Fail segment if any primary subband is unusable
        for name in ['D3', 'D4', 'A4']:
            if subbands.get(name) is None:
                return None

        results = {}
        for name in ANALYSIS_SUBBANDS:
            sb = subbands.get(name)
            if sb is None:
                # D1/D2 may be None (below energy threshold after lowpass)
                # Skip silently — their absence is itself the null result
                continue
            params      = self.subband_params[name]
            results[name] = self._compute_rqa(sb, params['tau'], params['m'])

        return results

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

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
        print(f"Saved {len(results)} results -> {output_path}")


if __name__ == '__main__':
    pipeline = PAFPipeline()

    results_ss = pipeline.run_batch(single_scale=True)
    pipeline.results_to_csv(results_ss, OUTPUT_DIR / 'results' / 'single_scale_rqa_5_dist.csv')

    results_ms = pipeline.run_batch(single_scale=False)
    pipeline.results_to_csv(results_ms, OUTPUT_DIR / 'results' / 'multi_scale_rqa_5_dist.csv')
