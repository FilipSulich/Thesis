
# # pipeline 1 (single-scale) : clean ECG (10 second snippet) -> RP -> RQA
# # pipeline 2 (multi-scale) : clean ECG (10 second snippet) -> SWT -> inverse SWT per sub-band -> RP per sub-band -> RQA per sub-band

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
import csv
import os
import sys

from scipy.spatial.distance import pdist

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
        time_delay: int = 19,
        embedding_dim: int = 7,
        epsilon: Optional[float] = None,
        target_rr: Optional[float] = None,
        subband_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.patient_id = patient_id
        self.segment_id = segment_id
        self.is_normal = is_normal
        self.lead_time = lead_time
        self.single_scale = single_scale
        self.time_delay = time_delay
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.target_rr = target_rr
        # Per-subband parameters: {"D1": {"tau": 5, "m": 7}, "D2": {...}, ...}
        self.subband_params = subband_params

    def get_epsilon_percentage_of_max(self, signal: np.ndarray, embedding_dim: int, time_delay: int,
                                          percentage: float = 0.10) -> float:
            """
            Compute epsilon as a percentage of the maximum phase space diameter.

            Args:
                signal: 1D time series
                embedding_dim: embedding dimension (m)
                time_delay: time delay (tau)
                percentage: fraction of max distance (default 0.10 = 10%)

            Returns:
                epsilon value
            """
            N = len(signal) - (embedding_dim - 1) * time_delay
            embedded = np.array([
                signal[i:i + embedding_dim * time_delay:time_delay]
                for i in range(N)
            ])
            distances = pdist(embedded, metric='euclidean')
            max_distance = float(np.max(distances))
            const = 0.2 * np.std(distances)
            return const

    def _reconstruct_subbands(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        swt = stationary_wavelet.SWT()
        coeffs = swt.decompose(signal, level=4)

        reconstructions = {}
        num_levels = len(coeffs)

        # Reconstruct each detail subband (D1-D4)
        for i in range(num_levels):
            zeroed_coeffs = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            # Keep only detail: (zeros, cD)
            zeroed_coeffs[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
            reconstructions[f"D{i+1}"] = swt.reconstruct(zeroed_coeffs)

        # Reconstruct final approximation (A4)
        # Keep ALL approximations, zero ALL details
        zeroed_coeffs = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        reconstructions["A4"] = swt.reconstruct(zeroed_coeffs)

        return reconstructions


    def _compute_rqa(
            self,
            signal: np.ndarray,
            seg_id: str,
            index: int,
            plot: bool,
            subband_name: Optional[str] = None,
    ):
        if subband_name and self.subband_params and subband_name in self.subband_params:
            params = self.subband_params[subband_name]
            tau = params.get("tau", self.time_delay)
            m = params.get("m", self.embedding_dim)
            subband_rr = params.get("target_rr", None)
        else:
            tau = self.time_delay
            m = self.embedding_dim
            subband_rr = None

        # Choose epsilon
        if subband_rr is not None:
            epsilon = self.get_epsilon_for_target_rr(signal, m, tau, target_rr=subband_rr)
        elif self.target_rr is not None:
            epsilon = self.get_epsilon_for_target_rr(signal, m, tau, target_rr=self.target_rr)
        else:
            epsilon = self.get_epsilon_for_target_rr(signal, m, tau)

        rp_gen = recurrence_plot.RecurrencePlotGenerator(
            signal=signal,
            time_delay=tau,
            embedding_dim=m,
            epsilon=epsilon,
            path=None,
            unthresholded=True,
        )
        rqa, _ = rp_gen.compute_rqa_binary()
        if plot:
            rp_gen.plot_recurrence(seg_id, index)
        return rqa

    def run(self, clean_signal: np.ndarray, index) -> PipelineResult:
        result = PipelineResult(
            patient_id=self.patient_id,
            segment_id=self.segment_id,
            is_normal=self.is_normal,
            lead_time=self.lead_time,
            single_scale=self.single_scale,
        )

        # self.diagnose_multiscale_rqa(clean_signal,self.subband_params)

        if self.single_scale:
            signal_to_use = clean_signal
            result.rqa_results["signal"] = self._compute_rqa(signal_to_use, self.segment_id, index, plot=False)
        else:
            subbands = self._reconstruct_subbands(clean_signal)
            for name, subband_signal in subbands.items():
                result.rqa_results[name] = self._compute_rqa(subband_signal, self.segment_id, index, plot=False, subband_name=name)

        return result


    def get_epsilon_for_target_rr(
            self,
            signal: np.ndarray,
            embedding_dim: int,
            time_delay: int,
            target_rr: float = 0.03,
    ) -> float:
        """
        Compute epsilon such that recurrence rate ≈ target_rr.

        Procedure:
        1. Embed signal
        2. Compute all pairwise distances
        3. Choose epsilon as the percentile corresponding to target_rr

        Args:
            signal: 1D time series
            embedding_dim: m
            time_delay: tau
            target_rr: desired recurrence rate (e.g. 0.03 = 3%)

        Returns:
            epsilon threshold
        """

        N = len(signal) - (embedding_dim - 1) * time_delay
        if N <= 1:
            raise ValueError("Signal too short for embedding.")

        embedded = np.array([
            signal[i:i + embedding_dim * time_delay:time_delay]
            for i in range(N)
        ])

        distances = pdist(embedded, metric="euclidean")

        # Percentile corresponding to target recurrence rate
        epsilon = np.percentile(distances, target_rr * 100.0)

        return float(epsilon)


def decompose_signals_to_csv(input_csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    swt = stationary_wavelet.SWT()
    subband_names = ["D1", "D2", "D3", "D4", "A4"]

    with open(input_csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Decomposing {len(rows)} signals into subbands...")

    subband_data = {name: [] for name in subband_names}

    for idx, row in enumerate(rows):
        # Extract signal
        signal = np.array([float(row[f"s{i}"]) for i in range(2500)])

        # Decompose - pywt.swt returns [(cA1, cD1), (cA2, cD2), (cA3, cD3), (cA4, cD4)]
        coeffs = swt.decompose(signal, level=4)

        # Reconstruct each detail subband (D1-D4)
        for i in range(4):
            zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            # Keep only detail: (zeros, cD)
            zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
            subband_signal = swt.reconstruct(zeroed)

            subband_signal = (subband_signal - np.mean(subband_signal)) / np.std(subband_signal, ddof=1)

            # Create row with metadata + subband signal
            subband_row = {
                "index": row["index"],
                "patient_id": row["patient_id"],
                "segment_id": row["segment_id"],
                "lead_time": row["lead_time"],
                "is_normal": row["is_normal"],
            }
            for j, val in enumerate(subband_signal[:2500]):  # Trim to original length
                subband_row[f"s{j}"] = val

            subband_data[f"D{i+1}"].append(subband_row)

        # A4 (approximation) - keep ALL approximations, zero ALL details
        zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        subband_signal = swt.reconstruct(zeroed)

        subband_row = {
            "index": row["index"],
            "patient_id": row["patient_id"],
            "segment_id": row["segment_id"],
            "lead_time": row["lead_time"],
            "is_normal": row["is_normal"],
        }
        for j, val in enumerate(subband_signal[:2500]):
            subband_row[f"s{j}"] = val

        subband_data["A4"].append(subband_row)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(rows)} signals")

    # Write each subband to its own CSV
    header = ["index", "patient_id", "segment_id", "lead_time", "is_normal"] + [f"s{i}" for i in range(2500)]

    for name in subband_names:
        output_path = os.path.join(output_dir, f"{name}.csv")
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(subband_data[name])
        print(f"Saved {name} to {output_path}")

    print(f"\nDone! Subband CSVs saved to {output_dir}")


def run_pipeline_from_csv(
    csv_path: str,
    time_delay: Optional[int] = None,
    embedding_dim: Optional[int] = None,
    single_scale: bool = True,
    epsilon: Optional[float] = None,
    target_rr: Optional[float] = None,
    subband_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[PipelineResult]:
    results = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row["patient_id"]
            segment_id = row["segment_id"]
            lead_time = int(row["lead_time"])
            is_normal = row["is_normal"] == "True"

            signal = np.array([float(row[f"s{i}"]) for i in range(2500)])

            pipeline = Pipeline(
                patient_id=patient_id,
                segment_id=segment_id,
                is_normal=is_normal,
                lead_time=lead_time,
                single_scale=single_scale,
                time_delay=time_delay,
                embedding_dim=embedding_dim,
                epsilon=0.3,
                target_rr=target_rr,
                subband_params=subband_params,
            )

            # visualize_decomposition(signal)
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




def visualize_decomposition(signal: np.ndarray, output_path: str = None, fs: int = 250):
    """
    Visualize original signal and its SWT subbands.

    Args:
        signal: 1D signal array
        output_path: If provided, saves figure to this path
        fs: Sampling frequency (for time axis)
    """
    import matplotlib.pyplot as plt

    swt = stationary_wavelet.SWT()
    coeffs = swt.decompose(signal, level=4)

    # Reconstruct subbands
    subbands = {}

    # D1-D4
    for i in range(4):
        zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
        zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
        subbands[f"D{i+1}"] = swt.reconstruct(zeroed)[:len(signal)]

    # A4
    zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
    subbands["A4"] = swt.reconstruct(zeroed)[:len(signal)]

    # Time axis
    t = np.arange(len(signal)) / fs

    # Plot
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, signal, 'b', linewidth=0.5)
    axes[0].set_ylabel("Original")
    axes[0].set_title("SWT Decomposition (db4, 4 levels)")

    for i, name in enumerate(["D1", "D2", "D3", "D4", "A4"]):
        axes[i+1].plot(t, subbands[name], 'b', linewidth=0.5)
        axes[i+1].set_ylabel(name)

    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()

if __name__ == '__main__':
    input_csv = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/rqa_dataset_all_leadtimes.csv"
    output_path_single = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/old/rqa_results_single_scale.csv"
    output_path_multi = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/old/rqa_results_multiscale.csv"

    subband_params = {
        "D1": {"tau": 7, "m": 7, "target_rr": 0.03},
        "D2": {"tau": 3, "m": 7, "target_rr": 0.03},
        "D3": {"tau": 1, "m": 7, "target_rr": 0.03},
        "D4": {"tau": 1, "m": 7, "target_rr": 0.03},
        "A4": {"tau": 17, "m": 7, "target_rr": 0.03},
    }

    single_scale = False

    results = run_pipeline_from_csv(
        csv_path=input_csv,
        single_scale=single_scale,
        subband_params=subband_params,
        embedding_dim=3,
        time_delay=3,
        target_rr=0.05
    )

    if single_scale:
        results_to_csv(results, output_path_single)
    else :
        results_to_csv(results, output_path_multi)


