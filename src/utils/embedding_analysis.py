"""
False Nearest Neighbors (Kennel et al. 1992) - Correct Implementation
Reads signals from CSV (columns s0, s1, ..., sN)
"""

import numpy as np
import csv
from scipy.spatial import KDTree
from typing import Tuple, List


class FalseNearestNeighbors:
    def __init__(
        self,
        tau: int,
        max_dim: int = 15,
        rtol: float = 15.0,
        atol: float = 2.0,
        theiler_window: int = None,
    ):
        self.tau = tau
        self.max_dim = max_dim
        self.rtol = rtol
        self.atol = atol
        self.theiler_window = theiler_window if theiler_window is not None else tau

    # -------------------------------------------------------
    # Time-delay embedding
    # -------------------------------------------------------
    @staticmethod
    def embed(signal: np.ndarray, m: int, tau: int) -> np.ndarray:
        N = len(signal) - (m - 1) * tau
        if N <= 0:
            raise ValueError("Signal too short for embedding.")
        return np.array([signal[i:i + m * tau:tau] for i in range(N)])

    # -------------------------------------------------------
    # Compute FNN
    # -------------------------------------------------------
    def compute(self, signal: np.ndarray):

        signal = np.asarray(signal, dtype=np.float64)
        std_signal = np.std(signal)

        dims = np.arange(1, self.max_dim + 1)
        fnn_percent = np.zeros(self.max_dim)

        N_total = len(signal)

        for m in dims:

            N_m = N_total - m * self.tau
            if N_m <= 10:
                fnn_percent[m - 1] = np.nan
                continue

            # Build embedding explicitly limited to N_m
            embedded_m = np.array([
                signal[i:i + m * self.tau:self.tau]
                for i in range(N_m)
            ])

            tree = KDTree(embedded_m)

            false_neighbors = 0
            total = 0

            for i in range(N_m):

                distances, indices = tree.query(embedded_m[i], k=10)

                for dist, j in zip(distances[1:], indices[1:]):

                    # Theiler window exclusion
                    if abs(i - j) < self.theiler_window:
                        continue

                    if dist == 0:
                        continue

                    # Ensure j also valid for delta
                    if j >= N_m:
                        continue

                    total += 1

                    # Proper delta (added coordinate only)
                    delta = abs(
                        signal[i + m * self.tau] -
                        signal[j + m * self.tau]
                    )

                    criterion1 = (delta / dist) > self.rtol
                    criterion2 = (delta / std_signal) > self.atol

                    if criterion1 or criterion2:
                        false_neighbors += 1

                    break

            if total > 0:
                fnn_percent[m - 1] = 100.0 * false_neighbors / total
            else:
                fnn_percent[m - 1] = np.nan

        optimal_dim = self.max_dim
        for dim, fnn in zip(dims, fnn_percent):
            if not np.isnan(fnn) and fnn < 1.0:
                optimal_dim = dim
                break

        return dims, fnn_percent, optimal_dim


# -------------------------------------------------------
# Multi-signal analysis
# -------------------------------------------------------
def analyze_signals(signals: List[np.ndarray], tau: int, max_dim: int = 20):

    fnn_analyzer = FalseNearestNeighbors(
        tau=tau,
        max_dim=max_dim,
        rtol=15.0,
        atol=2.0,
        theiler_window=tau
    )

    optimal_dims = []
    all_fnn_curves = []

    for idx, sig in enumerate(signals):
        try:
            dims, fnn, opt_dim = fnn_analyzer.compute(sig)
            optimal_dims.append(opt_dim)
            all_fnn_curves.append(fnn)
        except Exception as e:
            print(f"Signal {idx} failed: {e}")

    optimal_dims = np.array(optimal_dims)
    all_fnn_curves = np.array(all_fnn_curves)

    mean_fnn = np.nanmean(all_fnn_curves, axis=0)
    std_fnn = np.nanstd(all_fnn_curves, axis=0)

    return dims, optimal_dims, mean_fnn, std_fnn


# -------------------------------------------------------
# Load signals from CSV
# -------------------------------------------------------
def load_signals_from_csv(csv_path: str, signal_length: int = 1280):

    signals = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            signal = np.array(
                [float(row[f"s{i}"]) for i in range(signal_length)]
            )
            signals.append(signal)

    return signals


# -------------------------------------------------------
# Main execution
# -------------------------------------------------------
if __name__ == "__main__":

    csv_path = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf/subbands/A4.csv"
    csv_path2 = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/matched_segments_full.csv"
    csv_path3 = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/mitbih/results/subbands/mitbih_lt5_D4.csv"
    csv_path4 = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf_updated/subbands/single_scale.csv"

    tau = 12
    max_dim = 20

    #optimal parameters
    #single scale: time delay = 12, m =
    #A4: time delay = 16, m = 10
    #D3: time delay = 2, m = 4
    #D4: time delay = 1, m = 3

    subband_params = {
        "original": {"tau": 20, "m": 8, "target_rr": 0.05},
        "D1": {"tau": 7, "m": 4, "target_rr": 0.05},
        "D2": {"tau": 4, "m": 6, "target_rr": 0.05},
        "D3": {"tau": 2, "m": 5, "target_rr": 0.05},
        "D4": {"tau": 1, "m": 7, "target_rr": 0.05},
        "A4": {"tau": 20, "m": 9, "target_rr": 0.05},
    }

    signals = load_signals_from_csv(csv_path4)

    print(f"Loaded {len(signals)} signals")
    print(f"Using tau = {tau}")
    print()

    dims, optimal_dims, mean_fnn, std_fnn = analyze_signals(
        signals,
        tau=tau,
        max_dim=max_dim
    )

    print("=" * 60)
    print("FALSE NEAREST NEIGHBORS RESULTS")
    print("=" * 60)
    print()
    print(f"Optimal embedding dimension statistics:")
    print(f"  Mean   : {np.mean(optimal_dims):.2f}")
    print(f"  Median : {np.median(optimal_dims):.2f}")
    print(f"  Std    : {np.std(optimal_dims):.2f}")
    print(f"  Range  : [{np.min(optimal_dims)}, {np.max(optimal_dims)}]")
    print()
    print("Mean FNN curve:")
    print("-" * 40)
    for d, m_fnn, s_fnn in zip(dims, mean_fnn, std_fnn):
        if not np.isnan(m_fnn):
            print(f"m={d:2d}: {m_fnn:6.2f}% ± {s_fnn:5.2f}%")