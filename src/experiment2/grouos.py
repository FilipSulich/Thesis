import numpy as np
import pandas as pd
import pywt
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
import multiprocessing
from itertools import combinations


class GroupSeparabilityAnalyzer:
    """
    Analyzes wavelet discriminability using the Group Separability Ratio (GSR)
    based on the Frobenius distance between UTRP distance matrices.
    """

    def __init__(self, wavelets, fs=128, level=4, signal_subbands=['D3', 'D4']):
        self.wavelets = wavelets
        self.fs = fs
        self.level = level
        self.signal_subbands = signal_subbands

        # Mapping subband names to pywt coefficient indices for level 4
        # [cA4, cD4, cD3, cD2, cD1]
        self.subband_map = {
            'A4': 0, 'D4': 1, 'D3': 2, 'D2': 3, 'D1': 4
        }

    # ------------------------------------------------------------------ #
    # Core Mathematical Methods                                          #
    # ------------------------------------------------------------------ #

    def _get_subband_data(self, signal, wavelet, subband):
        """Extracts specific subband coefficients using DWT."""
        coeffs = pywt.wavedec(signal, wavelet, level=self.level)
        idx = self.subband_map.get(subband)
        return coeffs[idx]

    def _embed_signal(self, data, tau, dim):
        """
        Performs time-delay embedding to reconstruct the attractor.
        $X(n) = [x(n), x(n+\tau), \dots, x(n+(m-1)\tau)]$
        """
        n = len(data)
        if n < (dim - 1) * tau + 1:
            return data.reshape(-1, 1)  # Fallback if signal too short

        indices = np.arange(dim) * tau
        attractor = np.array([data[i:i + n - (dim - 1) * tau] for i in indices])
        return attractor.T

    def _compute_utrp(self, signal_data, tau=1, dim=3):
        """Generates the Unthresholded Recurrence Plot (Distance Matrix)."""
        embedded_space = self._embed_signal(signal_data, tau, dim)
        # Compute pairwise Euclidean distances
        dist_vec = pdist(embedded_space, metric='euclidean')
        return squareform(dist_vec)

    def frobenius_distance(self, M1, M2):
        """
        Computes the Frobenius distance between two matrices.
        $$d_F(M_1, M_2) = \sqrt{\sum_{i,j} |(M_1)_{ij} - (M_2)_{ij}|^2}$$
        """
        # Ensure identical dimensions for comparison
        n = min(M1.shape[0], M2.shape[0])
        return np.linalg.norm(M1[:n, :n] - M2[:n, :n], ord='fro')

    # ------------------------------------------------------------------ #
    # GSR Calculation Logic                                              #
    # ------------------------------------------------------------------ #

    def calculate_gsr(self, utrp_preaf, utrp_nsr):
        """
        Calculates GSR = between_group_dist / within_group_dist.
        """
        # 1. Within-group distances (Pre-AF)
        within_preaf_list = [
            self.frobenius_distance(utrp_preaf[i], utrp_preaf[j])
            for i, j in combinations(range(len(utrp_preaf)), 2)
        ]
        w_preaf = np.mean(within_preaf_list) if within_preaf_list else 0

        # 2. Within-group distances (NSR)
        within_nsr_list = [
            self.frobenius_distance(utrp_nsr[i], utrp_nsr[j])
            for i, j in combinations(range(len(utrp_nsr)), 2)
        ]
        w_nsr = np.mean(within_nsr_list) if within_nsr_list else 0

        within_mean = (w_preaf + w_nsr) / 2

        # 3. Between-group distances
        between_list = [
            self.frobenius_distance(p, n)
            for p in utrp_preaf
            for n in utrp_nsr
        ]
        between_mean = np.mean(between_list) if between_list else 0

        gsr = between_mean / (within_mean + 1e-12)
        return gsr, between_mean, within_mean

    # ------------------------------------------------------------------ #
    # Parallel Processing Pipeline                                       #
    # ------------------------------------------------------------------ #

    def process_wavelet(self, wavelet, segments_pre_af, segments_nsr, embedding_params):
        """Worker function for a single wavelet across all subbands."""
        results = []

        for subband in self.signal_subbands:
            params = embedding_params.get(subband, {'tau': 1, 'dim': 3})

            # Helper to get median UTRP per patient
            def get_patient_medians(data_dict):
                patient_matrices = []
                for pid, segments in data_dict.items():
                    segs_matrices = []
                    for s in segments:
                        coeffs = self._get_subband_data(s, wavelet, subband)
                        segs_matrices.append(self._compute_utrp(coeffs, **params))

                    if segs_matrices:
                        # Compute element-wise median across patient segments
                        patient_matrices.append(np.median(np.array(segs_matrices), axis=0))
                return patient_matrices

            # Generate representations
            preaf_mats = get_patient_medians(segments_pre_af)
            nsr_mats = get_patient_medians(segments_nsr)

            # Compute Metric
            gsr, b_dist, w_dist = self.calculate_gsr(preaf_mats, nsr_mats)

            results.append({
                'wavelet': wavelet,
                'subband': subband,
                'gsr': gsr,
                'between_dist': b_dist,
                'within_dist': w_dist
            })

        return results

    def run_analysis(self, segments_pre_af, segments_nsr, embedding_params):
        """Main entry point for parallel GSR analysis."""
        n_cores = min(len(self.wavelets), multiprocessing.cpu_count())

        print(f"Starting GSR Analysis for {len(self.wavelets)} wavelets...")

        # Distribute wavelet tasks across cores
        raw_results = Parallel(n_jobs=n_cores)(
            delayed(self.process_wavelet)(
                w, segments_pre_af, segments_nsr, embedding_params
            ) for w in self.wavelets
        )

        # Flatten list of lists
        flat_results = [item for sublist in raw_results for item in sublist]
        df = pd.DataFrame(flat_results)

        # Aggregate combined score (Sum of GSR for D3 and D4)
        summary = df.groupby('wavelet')['gsr'].sum().reset_index()
        summary.columns = ['wavelet', 'combined_gsr']
        summary = summary.sort_values('combined_gsr', ascending=False)

        return df, summary


# ------------------------------------------------------------------ #
# Example Usage logic                                                #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # Define Embedding parameters per subband (from previous estimation steps)
    # Example values:
    emb_params = {
        'D3': {'tau': 2, 'dim': 3},
        'D4': {'tau': 4, 'dim': 3}
    }

    # Configuration from your original script
    WAVELETS = ['db2', 'db4', 'db6', 'sym4', 'sym6', 'sym8', 'coif1', 'coif2', 'coif3']

    # Initialize and Run
    analyzer = GroupSeparabilityAnalyzer(wavelets=WAVELETS, signal_subbands=['D3', 'D4'])

    # Note: segments_pre_af and segments_nsr should be loaded here as:
    # { 'patient_1': [np.array, np.array, ...], 'patient_2': [...] }

    # detailed_df, summary_df = analyzer.run_analysis(segments_pre_af, segments_nsr, emb_params)
    # print(summary_df)