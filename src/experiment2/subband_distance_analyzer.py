"""
Subband Distance Analyzer for Pre-AF Detection via Unthresholded Recurrence Plot Geometry.

This module implements distance matrix feature extraction from wavelet subbands to
distinguish pre-AF from NSR based on attractor dynamics and flickering signatures.

Theoretical basis:
  - Lan et al. (2020): Pre-AF flickering produces transient bimodal attractor structure
  - Ngamga et al. (2012, PRE 85, 026217): Recurrence time entropy distinguishes regimes
  - Unthresholded distance matrices avoid epsilon sensitivity of binary RQA

Key metrics:
  1. Mean pairwise distance — attractor size (larger in pre-AF due to flickering)
  2. Distance variance — attractor dispersion (higher in pre-AF, bimodal structure)
  3. Distance skewness — attractor bimodality (heavier tail from between-basin distances)
  4. Recurrence time entropy — irregularity of recurrence timing (pre-AF alternates between basins)
  5. Diagonal profile variance — multi-scale temporal structure (complex RR(τ) in pre-AF)

Wavelet selection criterion:
  - Signal subbands: D3 (8-16 Hz), D4 (4-8 Hz), A4 (0-4 Hz) — atrial frequency bands
  - Control subbands: D1 (32-64 Hz), D2 (16-32 Hz) — negative controls for broadband effects
  - Selectivity ratio: (D3+D4+A4 composite score) / (D1+D2 composite score)
  - Best wavelet = highest selectivity ratio
"""

import numpy as np
import pandas as pd
import pywt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.stats import skew, mannwhitneyu
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recurrence_plot.AMI import AMI
from recurrence_plot.FNN import FNN


class SubbandDistanceAnalyzer:
    """
    Analyzer for ECG subband distance matrix geometry.

    Decomposes ECG segments into wavelet subbands, computes unthresholded
    distance matrices from embedded trajectories, and extracts geometric
    features that distinguish pre-AF from NSR based on attractor dynamics.
    """

    def __init__(
        self,
        wavelets: List[str],
        level: int = 4,
        fs: int = 128,
        signal_subbands: List[str] = None,
        control_subbands: List[str] = None,
        random_seed: int = 42
    ):
        """
        Initialize the analyzer.

        Parameters
        ----------
        wavelets : List[str]
            List of wavelet names to test (e.g., ['db2', 'db4', 'sym4'])
        level : int
            SWT decomposition level (default: 4)
        fs : int
            Sampling frequency in Hz (default: 128)
        signal_subbands : List[str]
            Subbands expected to show pre-AF differences (default: ['D3', 'D4', 'A4'])
        control_subbands : List[str]
            Negative control subbands (default: ['D1', 'D2'])
        random_seed : int
            Random seed for reproducibility
        """
        self.wavelets = wavelets
        self.level = level
        self.fs = fs
        self.signal_subbands = signal_subbands or ['D3', 'D4', 'A4']
        self.control_subbands = control_subbands or ['D1', 'D2']
        self.all_subbands = ['D1', 'D2', 'D3', 'D4', 'A4']
        self.rng = np.random.default_rng(random_seed)

        # Cache for embedding parameters {subband: {'tau': int, 'm': int}}
        self.embedding_params: Dict[str, Dict[str, int]] = {}

        # Storage for results
        self.patient_features: Optional[pd.DataFrame] = None
        self.statistical_results: Optional[pd.DataFrame] = None
        self.selection_scores: Optional[pd.DataFrame] = None

    def decompose_to_subbands(self, signal: np.ndarray, wavelet: str) -> Dict[str, np.ndarray]:
        """
        Decompose signal into subbands via SWT reconstruction.

        Parameters
        ----------
        signal : np.ndarray
            1D signal to decompose
        wavelet : str
            Wavelet name (e.g., 'db4')

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping subband names to reconstructed signals
        """
        coeffs = pywt.swt(signal, wavelet, level=self.level)
        subbands = {}
        n = len(signal)

        # Reconstruct detail subbands D1-D4
        for i in range(self.level):
            band = f'D{i+1}'
            target = [(np.zeros_like(cA), np.zeros_like(cD)) for cA, cD in coeffs]
            # pywt.swt returns coeffs from coarsest to finest
            target[self.level - 1 - i] = (
                np.zeros_like(coeffs[self.level - 1 - i][0]),
                coeffs[self.level - 1 - i][1]
            )
            reconstructed = pywt.iswt(target, wavelet)[:n]
            # Z-score normalization
            subbands[band] = (reconstructed - np.mean(reconstructed)) / (np.std(reconstructed) + 1e-9)

        # Reconstruct approximation A4
        target = [(cA, np.zeros_like(cD)) for cA, cD in coeffs]
        for j in range(1, self.level):
            target[j] = (np.zeros_like(target[j][0]), np.zeros_like(target[j][1]))
        reconstructed = pywt.iswt(target, wavelet)[:n]
        subbands['A4'] = (reconstructed - np.mean(reconstructed)) / (np.std(reconstructed) + 1e-9)

        return subbands

    def estimate_embedding_params(
        self,
        sample_segments_pre_af: Dict[str, List[np.ndarray]],
        sample_segments_nsr: Dict[str, List[np.ndarray]],
        reference_wavelet: str = 'db4',
        n_samples: int = 5
    ):
        """
        Estimate optimal embedding parameters (tau, m) for each subband.

        Uses AMI for time delay and FNN for embedding dimension.
        Parameters are estimated once using sample segments and cached.

        Parameters
        ----------
        sample_segments_pre_af : Dict[str, List[np.ndarray]]
            Sample segments from pre-AF patients
        sample_segments_nsr : Dict[str, List[np.ndarray]]
            Sample segments from NSR patients
        reference_wavelet : str
            Wavelet to use for parameter estimation (default: 'db4')
        n_samples : int
            Number of segments to sample from each cohort per subband
        """
        print(f"\nEstimating embedding parameters using {reference_wavelet}...")

        # Collect sample segments from both cohorts
        all_sample_segments = []
        for cohort_dict in [sample_segments_pre_af, sample_segments_nsr]:
            for patient_id, segments in list(cohort_dict.items())[:n_samples]:
                if segments:
                    all_sample_segments.append(segments[0])

        for subband in self.all_subbands:
            print(f"  {subband}: ", end='', flush=True)

            # Decompose samples and extract subband
            subband_signals = []
            for seg in all_sample_segments:
                subbands = self.decompose_to_subbands(seg, reference_wavelet)
                subband_signals.append(subbands[subband])

            # Use first sample to estimate tau and m
            sample_signal = subband_signals[0]

            try:
                # Estimate tau using AMI (with reduced max_lag for speed)
                ami = AMI(sample_signal, max_lag=50)
                tau = ami.get_first_minimum()

                # Fallback if no minimum found
                if tau is None or tau < 1:
                    tau = 1
                    print(f"(AMI fallback) ", end='', flush=True)

                # Cap tau to reasonable value
                tau = min(tau, 20)

                # Estimate m using FNN (with reduced max_dim for speed)
                fnn = FNN(signal=sample_signal, tau=tau, max_dim=10)
                m = fnn.get_optimal_dimension()

                # Ensure m is reasonable
                if m is None or m < 2:
                    m = 3
                    print(f"(FNN fallback) ", end='', flush=True)

                m = min(m, 10)  # Cap to prevent memory issues

            except Exception as e:
                # Ultimate fallback
                print(f"(error: {str(e)[:20]}, using defaults) ", end='', flush=True)
                tau = 1
                m = 3

            # Cache parameters
            self.embedding_params[subband] = {'tau': tau, 'm': m}
            print(f"tau={tau}, m={m}")

        print("Embedding parameters estimated and cached.\n")

    def embed_signal(self, signal: np.ndarray, tau: int, m: int) -> np.ndarray:
        """
        Create time-delay embedding of signal.

        Parameters
        ----------
        signal : np.ndarray
            1D signal to embed
        tau : int
            Time delay
        m : int
            Embedding dimension

        Returns
        -------
        np.ndarray
            Embedded trajectory of shape (n_points, m)
        """
        n_points = len(signal) - (m - 1) * tau
        if n_points <= 0:
            raise ValueError(f"Signal too short for embedding: len={len(signal)}, tau={tau}, m={m}")

        embedded = np.zeros((n_points, m))
        for i in range(m):
            embedded[:, i] = signal[i * tau : i * tau + n_points]

        return embedded

    def compute_distance_matrix(self, embedded_trajectory: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distance matrix (unthresholded RP).

        Parameters
        ----------
        embedded_trajectory : np.ndarray
            Embedded trajectory of shape (n_points, m)

        Returns
        -------
        np.ndarray
            Symmetric distance matrix of shape (n_points, n_points)
        """
        distances = pdist(embedded_trajectory, metric='euclidean')
        distance_matrix = squareform(distances)
        return distance_matrix

    def compute_recurrence_time_entropy(self, distance_matrix: np.ndarray) -> float:
        """
        Compute Shannon entropy of recurrence time distribution.

        Uses adaptive local threshold (median per row) to define recurrences.
        Measures irregularity of recurrence timing — pre-AF flickering produces
        more irregular recurrence as trajectory alternates between basins.

        Reference: Ngamga et al. (2012, PRE 85, 026217)

        Parameters
        ----------
        distance_matrix : np.ndarray
            Unthresholded distance matrix

        Returns
        -------
        float
            Shannon entropy of recurrence time distribution
        """
        N = len(distance_matrix)
        all_recurrence_times = []

        for i in range(N):
            # Adaptive local threshold: median distance for row i
            row_distances = distance_matrix[i, :]
            threshold = np.median(row_distances)

            # Find recurrent points (columns j where d[i,j] < threshold)
            recurrent_indices = np.where(row_distances < threshold)[0]

            # Compute gaps between recurrent points (recurrence times)
            if len(recurrent_indices) > 1:
                recurrence_times = np.diff(recurrent_indices)
                all_recurrence_times.extend(recurrence_times)

        # Compute Shannon entropy of recurrence time distribution
        if len(all_recurrence_times) == 0:
            return 0.0

        # Bin the recurrence times
        max_rt = max(all_recurrence_times)
        if max_rt == 0:
            return 0.0

        bins = np.arange(1, max_rt + 2)
        hist, _ = np.histogram(all_recurrence_times, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins

        if len(hist) == 0:
            return 0.0

        entropy = -np.sum(hist * np.log2(hist + 1e-12))
        return entropy

    def compute_diagonal_profile_variance(self, distance_matrix: np.ndarray) -> float:
        """
        Compute variance of diagonal profile RR(τ).

        For each diagonal offset τ, compute mean distance along that diagonal.
        Variance of this profile captures multi-scale temporal structure.
        Pre-AF shows higher variance due to multiple characteristic timescales
        from two-attractor dynamics.

        Parameters
        ----------
        distance_matrix : np.ndarray
            Unthresholded distance matrix

        Returns
        -------
        float
            Variance of diagonal profile
        """
        N = len(distance_matrix)
        max_offset = N // 4  # τ from 1 to N/4

        if max_offset < 2:
            return 0.0

        diagonal_means = []
        for tau in range(1, max_offset + 1):
            # Extract diagonal at offset τ
            diag = np.diagonal(distance_matrix, offset=tau)
            if len(diag) > 0:
                diagonal_means.append(np.mean(diag))

        if len(diagonal_means) < 2:
            return 0.0

        # Variance of RR(τ) profile
        profile_variance = np.var(diagonal_means)
        return profile_variance

    def extract_distance_features(self, distance_matrix: np.ndarray) -> Dict[str, float]:
        """
        Extract all five geometric features from distance matrix.

        Parameters
        ----------
        distance_matrix : np.ndarray
            Unthresholded distance matrix

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - mean_dist: mean pairwise distance (attractor size)
            - var_dist: distance variance (attractor dispersion)
            - skew_dist: distance skewness (attractor bimodality)
            - recurrence_time_entropy: recurrence timing irregularity
            - diagonal_profile_var: multi-scale temporal structure
        """
        # Extract upper triangle (exclude diagonal)
        upper_tri_indices = np.triu_indices_from(distance_matrix, k=1)
        upper_tri = distance_matrix[upper_tri_indices]

        if len(upper_tri) == 0:
            return {
                'mean_dist': 0.0,
                'var_dist': 0.0,
                'skew_dist': 0.0,
                'recurrence_time_entropy': 0.0,
                'diagonal_profile_var': 0.0
            }

        # Metric 1: Mean pairwise distance
        mean_dist = float(np.mean(upper_tri))

        # Metric 2: Distance variance
        var_dist = float(np.var(upper_tri))

        # Metric 3: Distance skewness
        skew_dist = float(skew(upper_tri))

        # Metric 4: Recurrence time entropy
        recurrence_time_entropy = self.compute_recurrence_time_entropy(distance_matrix)

        # Metric 5: Diagonal profile variance
        diagonal_profile_var = self.compute_diagonal_profile_variance(distance_matrix)

        return {
            'mean_dist': mean_dist,
            'var_dist': var_dist,
            'skew_dist': skew_dist,
            'recurrence_time_entropy': recurrence_time_entropy,
            'diagonal_profile_var': diagonal_profile_var
        }

    def analyze_segment(
        self,
        signal: np.ndarray,
        wavelet: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze a single segment: decompose, embed, compute features.

        Parameters
        ----------
        signal : np.ndarray
            Raw ECG segment
        wavelet : str
            Wavelet name

        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dict: {subband: {feature: value}}
        """
        # Decompose into subbands
        subbands = self.decompose_to_subbands(signal, wavelet)

        segment_features = {}

        for subband in self.all_subbands:
            subband_signal = subbands[subband]

            # Get cached embedding parameters
            if subband not in self.embedding_params:
                raise ValueError(f"Embedding parameters not estimated for {subband}")

            tau = self.embedding_params[subband]['tau']
            m = self.embedding_params[subband]['m']

            try:
                # Embed signal
                embedded = self.embed_signal(subband_signal, tau, m)

                # Compute distance matrix
                distance_matrix = self.compute_distance_matrix(embedded)

                # Extract features
                features = self.extract_distance_features(distance_matrix)

                segment_features[subband] = features

            except (ValueError, Exception) as e:
                # Handle edge cases (signal too short, etc.)
                segment_features[subband] = {
                    'mean_dist': np.nan,
                    'var_dist': np.nan,
                    'skew_dist': np.nan,
                    'recurrence_time_entropy': np.nan,
                    'diagonal_profile_var': np.nan
                }

        return segment_features

    def analyze_patient_segments(
        self,
        segments_list: List[np.ndarray],
        wavelet: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze all segments from one patient and aggregate by median.

        Parameters
        ----------
        segments_list : List[np.ndarray]
            List of segments for one patient
        wavelet : str
            Wavelet name

        Returns
        -------
        Dict[str, Dict[str, float]]
            Patient-level features: {subband: {feature: median_value}}
        """
        # Collect features from all segments
        all_segment_features = []
        for seg in segments_list:
            try:
                seg_features = self.analyze_segment(seg, wavelet)
                all_segment_features.append(seg_features)
            except Exception as e:
                # Skip problematic segments
                continue

        if len(all_segment_features) == 0:
            return {}

        # Aggregate by median across segments
        patient_features = {}

        for subband in self.all_subbands:
            feature_values = {
                'mean_dist': [],
                'var_dist': [],
                'skew_dist': [],
                'recurrence_time_entropy': [],
                'diagonal_profile_var': []
            }

            for seg_feat in all_segment_features:
                if subband in seg_feat:
                    for feat_name, feat_val in seg_feat[subband].items():
                        if not np.isnan(feat_val):
                            feature_values[feat_name].append(feat_val)

            # Compute median
            patient_features[subband] = {
                feat_name: np.median(vals) if len(vals) > 0 else np.nan
                for feat_name, vals in feature_values.items()
            }

        return patient_features

    def analyze_cohort(
        self,
        segments_dict: Dict[str, List[np.ndarray]],
        wavelet: str,
        cohort_label: str
    ) -> pd.DataFrame:
        """
        Analyze all patients in a cohort.

        Parameters
        ----------
        segments_dict : Dict[str, List[np.ndarray]]
            Patient ID -> list of segments
        wavelet : str
            Wavelet name
        cohort_label : str
            'pre_af' or 'nsr'

        Returns
        -------
        pd.DataFrame
            Rows = patients, columns = [patient_id, wavelet, cohort, subband, feature, value]
        """
        rows = []

        for patient_id, segments in tqdm(segments_dict.items(),
                                         desc=f"  {cohort_label} ({wavelet})"):
            patient_features = self.analyze_patient_segments(segments, wavelet)

            for subband, features in patient_features.items():
                for feature_name, feature_value in features.items():
                    rows.append({
                        'patient_id': patient_id,
                        'wavelet': wavelet,
                        'cohort': cohort_label,
                        'subband': subband,
                        'feature': feature_name,
                        'value': feature_value
                    })

        return pd.DataFrame(rows)

    def compute_composite_scores(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute composite scores and selectivity ratios for wavelet selection.

        For each wavelet:
        - Compute absolute group differences per metric per subband
        - Sum differences in signal subbands (D3+D4+A4)
        - Sum differences in control subbands (D1+D2)
        - Compute selectivity ratio

        Parameters
        ----------
        df : pd.DataFrame
            Combined patient features from both cohorts

        Returns
        -------
        pd.DataFrame
            Wavelet selection scores
        """
        scores = []

        for wavelet in self.wavelets:
            wavelet_df = df[df['wavelet'] == wavelet]

            signal_score = 0.0
            control_score = 0.0

            for subband in self.all_subbands:
                for feature in ['mean_dist', 'var_dist', 'skew_dist',
                              'recurrence_time_entropy', 'diagonal_profile_var']:

                    # Get patient-level medians for each cohort
                    pre_af_vals = wavelet_df[
                        (wavelet_df['cohort'] == 'pre_af') &
                        (wavelet_df['subband'] == subband) &
                        (wavelet_df['feature'] == feature)
                    ]['value'].values

                    nsr_vals = wavelet_df[
                        (wavelet_df['cohort'] == 'nsr') &
                        (wavelet_df['subband'] == subband) &
                        (wavelet_df['feature'] == feature)
                    ]['value'].values

                    # Remove NaNs
                    pre_af_vals = pre_af_vals[~np.isnan(pre_af_vals)]
                    nsr_vals = nsr_vals[~np.isnan(nsr_vals)]

                    if len(pre_af_vals) == 0 or len(nsr_vals) == 0:
                        continue

                    # Cohort medians
                    pre_af_median = np.median(pre_af_vals)
                    nsr_median = np.median(nsr_vals)

                    # Absolute difference
                    diff = abs(pre_af_median - nsr_median)

                    # Add to appropriate score
                    if subband in self.signal_subbands:
                        signal_score += diff
                    elif subband in self.control_subbands:
                        control_score += diff

            # Selectivity ratio
            selectivity_ratio = signal_score / (control_score + 1e-9)

            scores.append({
                'wavelet': wavelet,
                'signal_score': signal_score,
                'control_score': control_score,
                'selectivity_ratio': selectivity_ratio
            })

        scores_df = pd.DataFrame(scores).sort_values('selectivity_ratio', ascending=False)
        return scores_df

    def perform_statistical_tests(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Perform Mann-Whitney U tests and compute Cohen's d for all combinations.

        Parameters
        ----------
        df : pd.DataFrame
            Combined patient features from both cohorts

        Returns
        -------
        pd.DataFrame
            Statistical test results
        """
        results = []

        for wavelet in tqdm(self.wavelets, desc="Statistical tests"):
            wavelet_df = df[df['wavelet'] == wavelet]

            for subband in self.all_subbands:
                for feature in ['mean_dist', 'var_dist', 'skew_dist',
                              'recurrence_time_entropy', 'diagonal_profile_var']:

                    # Get patient-level values
                    pre_af_vals = wavelet_df[
                        (wavelet_df['cohort'] == 'pre_af') &
                        (wavelet_df['subband'] == subband) &
                        (wavelet_df['feature'] == feature)
                    ]['value'].values

                    nsr_vals = wavelet_df[
                        (wavelet_df['cohort'] == 'nsr') &
                        (wavelet_df['subband'] == subband) &
                        (wavelet_df['feature'] == feature)
                    ]['value'].values

                    # Remove NaNs
                    pre_af_vals = pre_af_vals[~np.isnan(pre_af_vals)]
                    nsr_vals = nsr_vals[~np.isnan(nsr_vals)]

                    if len(pre_af_vals) < 2 or len(nsr_vals) < 2:
                        continue

                    # Mann-Whitney U test
                    u_stat, p_value = mannwhitneyu(pre_af_vals, nsr_vals,
                                                   alternative='two-sided')

                    # Cohen's d
                    n1, n2 = len(pre_af_vals), len(nsr_vals)
                    pooled_std = np.sqrt(
                        ((n1 - 1) * np.var(pre_af_vals, ddof=1) +
                         (n2 - 1) * np.var(nsr_vals, ddof=1)) / (n1 + n2 - 2)
                    )
                    cohen_d = (np.mean(pre_af_vals) - np.mean(nsr_vals)) / (pooled_std + 1e-9)

                    # Summary statistics
                    results.append({
                        'wavelet': wavelet,
                        'subband': subband,
                        'feature': feature,
                        'pre_af_median': np.median(pre_af_vals),
                        'pre_af_mean': np.mean(pre_af_vals),
                        'pre_af_std': np.std(pre_af_vals),
                        'nsr_median': np.median(nsr_vals),
                        'nsr_mean': np.mean(nsr_vals),
                        'nsr_std': np.std(nsr_vals),
                        'u_statistic': u_stat,
                        'p_value': p_value,
                        'cohen_d': cohen_d
                    })

        return pd.DataFrame(results)

    def run_full_analysis(
        self,
        segments_pre_af: Dict[str, List[np.ndarray]],
        segments_nsr: Dict[str, List[np.ndarray]],
        output_dir: Optional[Path] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run complete analysis pipeline for all wavelets.

        Parameters
        ----------
        segments_pre_af : Dict[str, List[np.ndarray]]
            Pre-AF patient segments
        segments_nsr : Dict[str, List[np.ndarray]]
            NSR patient segments
        output_dir : Optional[Path]
            Directory to save results (if None, don't save)

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing:
            - 'patient_features': raw patient-level features
            - 'statistical_tests': Mann-Whitney U and Cohen's d results
            - 'selection_scores': wavelet composite scores and selectivity ratios
        """
        print("\n" + "="*80)
        print("SUBBAND DISTANCE ANALYZER — FULL ANALYSIS")
        print("="*80)

        # Analyze all wavelets for both cohorts
        all_patient_features = []

        for wavelet in self.wavelets:
            print(f"\nAnalyzing wavelet: {wavelet}")

            # Pre-AF cohort
            df_pre_af = self.analyze_cohort(segments_pre_af, wavelet, 'pre_af')
            all_patient_features.append(df_pre_af)

            # NSR cohort
            df_nsr = self.analyze_cohort(segments_nsr, wavelet, 'nsr')
            all_patient_features.append(df_nsr)

        # Combine all results
        self.patient_features = pd.concat(all_patient_features, ignore_index=True)

        print("\n" + "-"*80)
        print("Computing composite scores and selectivity ratios...")
        self.selection_scores = self.compute_composite_scores(self.patient_features)

        print("\n" + "-"*80)
        print("Performing statistical tests...")
        self.statistical_results = self.perform_statistical_tests(self.patient_features)

        # Save results if output directory specified
        if output_dir is not None:
            self.export_results(output_dir)

        # Print summary
        self._print_summary()

        return {
            'patient_features': self.patient_features,
            'statistical_tests': self.statistical_results,
            'selection_scores': self.selection_scores
        }

    def export_results(self, output_dir: Path):
        """
        Export all results to CSV files.

        Parameters
        ----------
        output_dir : Path
            Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting results to {output_dir}...")

        # File 1: Patient-level features
        path1 = output_dir / 'distance_features_per_patient.csv'
        self.patient_features.to_csv(path1, index=False)
        print(f"  Saved: {path1.name}")

        # File 2: Statistical tests
        path2 = output_dir / 'statistical_comparison.csv'
        self.statistical_results.to_csv(path2, index=False)
        print(f"  Saved: {path2.name}")

        # File 3: Wavelet selection scores
        path3 = output_dir / 'wavelet_selection_scores.csv'
        self.selection_scores.to_csv(path3, index=False)
        print(f"  Saved: {path3.name}")

        # File 4: Best wavelet summary
        best_wavelet = self.selection_scores.iloc[0]['wavelet']
        best_ratio = self.selection_scores.iloc[0]['selectivity_ratio']

        path4 = output_dir / 'best_wavelet_summary.txt'
        with open(path4, 'w') as f:
            f.write("="*80 + "\n")
            f.write("WAVELET SELECTION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Selected wavelet: {best_wavelet}\n")
            f.write(f"Selectivity ratio: {best_ratio:.4f}\n\n")
            f.write("Selectivity ratio = (D3+D4+A4 composite) / (D1+D2 composite)\n\n")
            f.write("-"*80 + "\n")
            f.write("Top 5 wavelets:\n")
            f.write("-"*80 + "\n")
            f.write(self.selection_scores.head(5).to_string(index=False))
            f.write("\n")
        print(f"  Saved: {path4.name}")

    def _print_summary(self):
        """Print analysis summary to console."""
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)

        print(f"\nTotal patients analyzed:")
        print(f"  Pre-AF: {len(self.patient_features[self.patient_features['cohort'] == 'pre_af']['patient_id'].unique())}")
        print(f"  NSR:    {len(self.patient_features[self.patient_features['cohort'] == 'nsr']['patient_id'].unique())}")

        print(f"\nWavelets tested: {len(self.wavelets)}")
        print(f"Subbands analyzed: {len(self.all_subbands)}")
        print(f"  Signal subbands: {', '.join(self.signal_subbands)}")
        print(f"  Control subbands: {', '.join(self.control_subbands)}")
        print(f"Features per subband: 5")

        print("\n" + "-"*80)
        print("TOP 5 WAVELETS BY SELECTIVITY RATIO")
        print("-"*80)
        print(self.selection_scores.head(5).to_string(index=False))

        best_wavelet = self.selection_scores.iloc[0]['wavelet']
        print(f"\n>>> SELECTED WAVELET: {best_wavelet} <<<\n")