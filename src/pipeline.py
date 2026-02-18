# pipeline 1 (single-scale) : clean ECG (10 second snippet) -> RP -> RQA
# pipeline 2 (multi-scale) : clean ECG (10 second snippet) -> SWT -> inverse SWT per sub-band -> RP per sub-band -> RQA per sub-band

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wavelets.stationary_wavelet_transform as stationary_wavelet
import recurrence_plot.recurrence_plot_generation as recurrence_plot

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
class PipelineResult:
    patient_id: str
    segment_id: str
    is_normal: bool
    lead_time: Optional[int]  # None for normal segments
    single_scale: bool
    rqa_results: Dict[str, Any] = field(default_factory=dict)
    # For single-scale: {"signal": rqa}
    # For multi-scale: {"D1": rqa, "D2": rqa, "D3": rqa, "D4": rqa, "A4": rqa}

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "patient_id": self.patient_id,
            "segment_id": self.segment_id,
            "is_normal": self.is_normal,
            "lead_time": self.lead_time,
            "single_scale": self.single_scale,
        }
        for key, rqa in self.rqa_results.items():
            metrics = extract_rqa_metrics(rqa)
            for metric_name, value in metrics.items():
                result[f"{key}_{metric_name}"] = value
        return result


class Pipeline:
    SUBBAND_NAMES = ["D1", "D2", "D3", "D4", "A4"]

    def __init__(
        self,
        patient_id: str,
        segment_id: str,
        is_normal: bool,
        lead_time: Optional[int],
        single_scale: bool,
        time_delay: int = 1,
        embedding_dim: int = 10,
    ):
        self.patient_id = patient_id
        self.segment_id = segment_id
        self.is_normal = is_normal
        self.lead_time = lead_time
        self.single_scale = single_scale
        self.time_delay = time_delay
        self.embedding_dim = embedding_dim

    def _reconstruct_subbands(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Decompose signal and reconstruct each sub-band independently."""
        swt = stationary_wavelet.SWT()
        coeffs = swt.decompose(signal, level=4)
        # coeffs is [(cD1, cA1), (cD2, cA2), (cD3, cA3), (cD4, cA4)] from level 1 to 4

        reconstructions = {}
        num_levels = len(coeffs)

        # Reconstruct each detail coefficient (D1-D4)
        for i in range(num_levels):
            zeroed_coeffs = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            zeroed_coeffs[i] = (coeffs[i][0], np.zeros_like(coeffs[i][1]))  # Keep only detail
            reconstructions[f"D{i+1}"] = swt.reconstruct(zeroed_coeffs)

        # Reconstruct final approximation (A4)
        zeroed_coeffs = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
        zeroed_coeffs[-1] = (np.zeros_like(coeffs[-1][0]), coeffs[-1][1])  # Keep only approx at last level
        reconstructions["A4"] = swt.reconstruct(zeroed_coeffs)

        return reconstructions

    def _compute_rqa(self, signal: np.ndarray, id: str, index: int, plot: bool):
        rp_gen = recurrence_plot.RecurrencePlotGenerator(
            signal=signal,
            time_delay=self.time_delay,
            embedding_dim=self.embedding_dim,
            path="/Users/filipsulich/Library/CloudStorage/GoogleDrive-sulich.f@gmail.com/My Drive/Thesis/Recurrence_Plots_single_scale_unthresholded",
            unthresholded=True,
        )
        rqa, _ = rp_gen.compute_rqa_binary()
        if plot:
            rp_gen.plot_recurrence(id, index)
        return rqa

    def run(self, clean_signal: np.ndarray, index) -> PipelineResult:
        result = PipelineResult(
            patient_id=self.patient_id,
            segment_id=self.segment_id,
            is_normal=self.is_normal,
            lead_time=self.lead_time,
            single_scale=self.single_scale,
        )

        if self.single_scale:
            result.rqa_results["signal"] = self._compute_rqa(clean_signal, self.segment_id, index, plot=True)
        else:
            subbands = self._reconstruct_subbands(clean_signal)
            for name, subband_signal in subbands.items():
                result.rqa_results[name] = self._compute_rqa(subband_signal, self.segment_id, index, plot=False)

        return result


def run_pipeline_from_csv(
    csv_path: str,
    single_scale: bool = True,
    time_delay: int = 1,
    embedding_dim: int = 10,
) -> List[PipelineResult]:
    results = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row["patient_id"]
            segment_id = row["segment_id"]
            lead_time = int(row["lead_time"])
            is_normal = row["is_normal"] == "True"

            signal = np.array([float(row[f"s{i}"]) for i in range(2499)])

            pipeline = Pipeline(
                patient_id=patient_id,
                segment_id=segment_id,
                is_normal=is_normal,
                lead_time=lead_time,
                single_scale=single_scale,
                time_delay=time_delay,
                embedding_dim=embedding_dim,
            )

            result = pipeline.run(signal, int(row["index"]))
            results.append(result)
            print(f"Processed: {segment_id} (normal={is_normal})")

    return results


def results_to_csv(results: List[PipelineResult], output_path: str):
    if not results:
        return

    rows = [r.to_dict() for r in results]
    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(results)} results to {output_path}")

def analyze_result(results):
    import pandas as pd
    data = pd.read_csv(results)
    return

if __name__ == '__main__':
    csv_path = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/matched_5min_50.csv"
    output_path = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/rqa_results.csv"

    results = run_pipeline_from_csv(
        csv_path=csv_path,
        single_scale=True,
        time_delay=19,
        embedding_dim=3,
    )

    results_to_csv(results, output_path)