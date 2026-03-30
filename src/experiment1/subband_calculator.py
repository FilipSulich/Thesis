import os
import sys
import numpy as np
import pandas as pd
import json

from src.lorenz_solver.lorenz import LorenzSolver
from src.recurrence_plot.AMI import AMI
from src.recurrence_plot.FNN import FNN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.pipeline_wavelets import Pipeline
from recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


class SubbandCalculator:
    """
    Calculator class containing all logic and computation methods for subband validation.
    No visualization methods - pure computation only.
    """

    def __init__(self, is_sine, is_lorenz):
        self.FS = 128
        self.N = 512
        self.LEVEL = 4
        self.WAVELET = 'db4'
        self.is_sine = is_sine
        self.is_lorenz = is_lorenz

        self.SUBBAND_FREQS = {
            'A4': 2,   # 0-4 Hz
            'D4': 6,   # 4-8 Hz
            'D3': 12,  # 8-16 Hz
            'D2': 24,  # 16-32 Hz
            'D1': 48   # 32-64 Hz
        }

        self.BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']
        self.PIPELINES = ['swt_recon', 'swt_direct', 'dwt_recon', 'dwt_direct']

        if self.is_sine:
            self.OUTPUT_DIR = '/Users/filipsulich/Developer/uni/Thesis/Thesis/outputs/experiment1_sine_new'
        elif self.is_lorenz:
            self.OUTPUT_DIR = '/Users/filipsulich/Developer/uni/Thesis/Thesis/outputs/experiment1_lorenz'

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        self._cleanup_pyopencl_lock()

    @staticmethod
    def _cleanup_pyopencl_lock():
        try:
            import glob
            home = os.path.expanduser("~")
            lock_pattern = os.path.join(home, "Library/Caches/pyopencl/*/lock")
            lock_files = glob.glob(lock_pattern)
            for lock_file in lock_files:
                try:
                    os.remove(lock_file)
                except (OSError, PermissionError):
                    pass
        except Exception:
            pass

    def generate_composite_sine(self, random_phase=False):
        t = np.arange(self.N) / self.FS
        signal = np.zeros(self.N)

        for band, freq in self.SUBBAND_FREQS.items():
            if random_phase:
                phase = np.random.uniform(0, 2*np.pi)
            else:
                phase = 0
            signal += np.sin(2 * np.pi * freq * t + phase)

        return signal

    def generate_lorenz_system(self, sigma, rho, beta, n_points):
        lorenz = LorenzSolver(sigma, rho, beta)
        signal = lorenz.generate_lorenz_signal(n_points, target_fs=self.FS)
        return signal

    def get_optimal_params(self, signal):
        ami = AMI(signal)
        tau = ami.get_first_minimum()
        fnn = FNN(signal=signal, tau=tau)
        m = fnn.get_optimal_dimension()

        return tau, m

    def generate_reference_rqa_values(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.OUTPUT_DIR, 'reference_rqa_values.json')

        reference_values = {}
        reference_data = {}
        t = np.arange(self.N) / self.FS

        for band, freq in self.SUBBAND_FREQS.items():
            pure_sine = np.sin(2 * np.pi * freq * t)

            if pure_sine.std() > 0:
                pure_sine = (pure_sine - pure_sine.mean()) / pure_sine.std()

            tau, m = self.get_optimal_params(pure_sine)
            epsilon = RecurrencePlotGenerator.find_epsilon_for_target_rr(signal=pure_sine, tau=tau, m=m, target_rr=0.1)

            metrics = self.compute_rp_metrics(pure_sine, tau, m, epsilon, compute_distance_matrix=True)

            reference_values[band] = {
                'freq_hz': freq,
                'tau': int(tau),
                'm': int(m),
                'epsilon': float(epsilon),
                'DET': float(metrics['DET']),
                'ENTR': float(metrics['ENTR']),
                'LAM': float(metrics['LAM']),
                'L_mean_diagonal': float(metrics['L_mean_diagonal']),
                'L_longest_diagonal': float(metrics['L_longest_diagonal']),
                'L_longest_vertical': float(metrics['L_longest_vertical']),
                'TT': float(metrics['TT']),
            }

            reference_data[band] = {
                'rp_matrix': metrics['rp_matrix'],
                'distance_matrix': metrics['distance_matrix'],
                'pure_sine': pure_sine
            }

            print(f"    DET: {metrics['DET']:.4f}, ENTR: {metrics['ENTR']:.4f}, "
                  f"LAM: {metrics['LAM']:.4f}, TT: {metrics['TT']:.4f}")

        with open(save_path, 'w') as f:
            json.dump(reference_values, f, indent=2)

        ref_data_path = save_path.replace('.json', '_data.npz')
        rp_matrices = {}
        distance_matrices = {}
        for band in self.BAND_NAMES:
            rp_obj = reference_data[band]['rp_matrix']
            if hasattr(rp_obj, 'recurrence_matrix_reverse_normalized'):
                rp_matrices[f'{band}_rp'] = rp_obj.recurrence_matrix_reverse_normalized
            else:
                rp_matrices[f'{band}_rp'] = rp_obj

            dist_matrix = reference_data[band]['distance_matrix']
            if dist_matrix is not None:
                distance_matrices[f'{band}_dist'] = dist_matrix

        signals = {f'{band}_signal': reference_data[band]['pure_sine'] for band in self.BAND_NAMES}
        np.savez(ref_data_path, **rp_matrices, **distance_matrices, **signals)

        return reference_values

    def load_reference_rqa_values(self, load_path=None):
        if load_path is None:
            load_path = os.path.join(self.OUTPUT_DIR, 'reference_rqa_values.json')

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Reference values file not found: {load_path}")

        with open(load_path, 'r') as f:
            print("Loading reference rqa values...")
            reference_values = json.load(f)

        return reference_values

    def load_reference_data(self, load_path=None):
        if load_path is None:
            load_path = os.path.join(self.OUTPUT_DIR, 'reference_rqa_values_data.npz')

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Reference data file not found: {load_path}")

        data = np.load(load_path, allow_pickle=True)
        reference_data = {}

        for band in self.BAND_NAMES:
            reference_data[band] = {
                'rp_matrix': data[f'{band}_rp'],
                'pure_sine': data[f'{band}_signal'],
                'distance_matrix': data[f'{band}_dist'] if f'{band}_dist' in data else None
            }

        return reference_data

    def get_analytic_embedding_params(self, freq, signal):
        tau, m = self.get_optimal_params(signal)
        return tau, m

    def compute_frobenius_distance(self, dist_matrix_ref, dist_matrix_pipe):
        """
        Compute Frobenius norm (Euclidean distance) between two distance matrices.
        """
        if dist_matrix_ref is None or dist_matrix_pipe is None:
            return np.nan

        if dist_matrix_ref.shape != dist_matrix_pipe.shape:
            min_size = min(dist_matrix_ref.shape[0], dist_matrix_pipe.shape[0])
            dist_matrix_ref = dist_matrix_ref[:min_size, :min_size]
            dist_matrix_pipe = dist_matrix_pipe[:min_size, :min_size]

        # Frobenius norm: sqrt(sum of squared differences)
        frobenius_dist = np.linalg.norm(dist_matrix_ref - dist_matrix_pipe, ord='fro')
        return frobenius_dist

    def compute_hamming_distance(self, rp_gt, rp_pipe):
        if rp_gt.shape != rp_pipe.shape:
            min_size = min(rp_gt.shape[0], rp_pipe.shape[0])
            rp_gt = rp_gt[:min_size, :min_size]
            rp_pipe = rp_pipe[:min_size, :min_size]

        N = rp_gt.shape[0]
        rp_gt_bin = (rp_gt > 0).astype(int)
        rp_pipe_bin = (rp_pipe > 0).astype(int)

        hamming = np.sum(np.abs(rp_gt_bin - rp_pipe_bin))
        d_H = hamming / (N * N)

        return d_H

    def embed_signal(self, signal, tau, m):
        n = len(signal) - (m - 1) * tau
        if n < 1:
            return None
        embedded = np.array([signal[i:i + m * tau:tau] for i in range(n)])
        return embedded

    def compute_hausdorff_distance(self, sig_gt, sig_pipe, tau, m):
        """Compute Hausdorff distance between phase space trajectories."""
        from scipy.spatial.distance import directed_hausdorff

        A = self.embed_signal(sig_gt, tau, m)
        B = self.embed_signal(sig_pipe, tau, m)

        if A is None or B is None:
            return np.nan

        h_AB = directed_hausdorff(A, B)[0]
        h_BA = directed_hausdorff(B, A)[0]

        return max(h_AB, h_BA)

    def compute_inband_power_fraction(self, sig_pipe, freq, fs, band_name, nperseg=256):
        from scipy import signal
        freqs, psd = signal.welch(sig_pipe, fs=fs, nperseg=min(nperseg, len(sig_pipe)))

        if band_name == 'A4':
            f_low = 0
            f_high = 4.5
        else:
            f_low = freq * 0.8
            f_high = freq * 1.2

        in_band_mask = (freqs >= f_low) & (freqs <= f_high)
        total_power = np.trapezoid(psd, freqs)
        in_band_power = np.trapezoid(psd[in_band_mask], freqs[in_band_mask])

        return in_band_power / total_power if total_power > 0 else 0.0

    def compute_rp_pattern_correlation(self, rp_gt, rp_pipe):
        """Assess spatial organization preservation using Pearson correlation."""
        if rp_gt.shape != rp_pipe.shape:
            min_size = min(rp_gt.shape[0], rp_pipe.shape[0])
            rp_gt = rp_gt[:min_size, :min_size]
            rp_pipe = rp_pipe[:min_size, :min_size]

        V_gt = rp_gt.ravel()
        V_pipe = rp_pipe.ravel()

        r = np.corrcoef(V_gt, V_pipe)[0, 1]
        return r

    def compute_crp_det(self, sig_gt, sig_pipe, tau, m, epsilon, min_line_length=2):
        embed_gt = self.embed_signal(sig_gt, tau, m)
        embed_pipe = self.embed_signal(sig_pipe, tau, m)

        if embed_gt is None or embed_pipe is None:
            return 0.0

        N_gt = embed_gt.shape[0]
        N_pipe = embed_pipe.shape[0]

        crp = np.zeros((N_gt, N_pipe))
        for i in range(N_gt):
            dists = np.linalg.norm(embed_pipe - embed_gt[i], axis=1)
            crp[i, :] = (dists <= epsilon).astype(int)

        total_recurrence = np.sum(crp)
        if total_recurrence == 0:
            return 0.0

        diagonal_points = 0
        for offset in range(-N_gt + 1, N_pipe):
            diagonal = np.diagonal(crp, offset=offset)

            current_length = 0
            for val in diagonal:
                if val == 1:
                    current_length += 1
                else:
                    if current_length >= min_line_length:
                        diagonal_points += current_length
                    current_length = 0

            if current_length >= min_line_length:
                diagonal_points += current_length

        det = diagonal_points / total_recurrence
        return det

    def decompose_signal(self, signal, pipeline_name):
        pipeline_obj = Pipeline(pipeline_name='all', level=self.LEVEL, wavelet=self.WAVELET)
        func = getattr(pipeline_obj, f"pipeline_{pipeline_name}")
        return func(signal)

    def compute_rp_metrics(self, signal, tau, m, epsilon, compute_distance_matrix=False):
        try:
            os.environ['PYOPENCL_CTX'] = '0'
            rp_gen = RecurrencePlotGenerator(
                signal=signal,
                time_delay=tau,
                embedding_dim=m,
                path=self.OUTPUT_DIR,
                epsilon=epsilon,
                unthresholded=False
            )
            rqa_result, rp_matrix = rp_gen.compute_rqa_binary()

            l_mean_diag = rqa_result.average_diagonal_line
            if np.isnan(l_mean_diag): l_mean_diag = 0.0

            longest_diagonal = rqa_result.longest_diagonal_line
            if np.isnan(longest_diagonal): longest_diagonal = 0.0

            tt = rqa_result.trapping_time
            if np.isnan(tt): tt = 0.0

            longest_vertical_line = rqa_result.longest_vertical_line
            if np.isnan(longest_vertical_line): longest_vertical_line = 0

            det = rqa_result.determinism
            if np.isnan(det): det = 0.0

            lam = rqa_result.laminarity
            if np.isnan(lam): lam = 0.0

            distance_matrix = None
            if compute_distance_matrix:
                # Compute unthresholded distance matrix
                rp_gen_unthresh = RecurrencePlotGenerator(
                    signal=signal,
                    time_delay=tau,
                    embedding_dim=m,
                    path=self.OUTPUT_DIR,
                    epsilon=epsilon,
                    unthresholded=True
                )
                from pyrqa.settings import Settings
                from pyrqa.analysis_type import Cross
                from pyrqa.neighbourhood import Unthresholded
                from pyrqa.metric import EuclideanMetric
                from pyrqa.computation import RPComputation
                from pyrqa.time_series import TimeSeries

                time_series = TimeSeries(signal, embedding_dimension=m, time_delay=tau)
                settings_unthresh = Settings(
                    time_series=time_series,
                    analysis_type=Cross,
                    neighbourhood=Unthresholded(),
                    similarity_measure=EuclideanMetric,
                    theiler_corrector=1
                )
                rp_comp_unthresh = RPComputation.create(settings_unthresh, verbose=False)
                rp_result_unthresh = rp_comp_unthresh.run()
                distance_matrix = rp_result_unthresh.recurrence_matrix_reverse_normalized

            return {
                'DET': det,
                'ENTR': rqa_result.entropy_diagonal_lines,
                'LAM': lam,
                'L_mean_diagonal': l_mean_diag,
                'L_longest_diagonal': longest_diagonal,
                'L_longest_vertical': longest_vertical_line,
                'TT': tt,
                'rp_matrix': rp_matrix,
                'distance_matrix': distance_matrix
            }
        except Exception as e:
            return {
                'DET': 0.0,
                'ENTR': 0.0,
                'LAM': 0.0,
                'L_mean_diagonal': 0.0,
                'L_longest_diagonal': 0.0,
                'L_max_vertical': 0.0,
                'TT': 0.0,
                'rp_matrix': None,
                'distance_matrix': None
            }

    def get_unthresholded_matrix(self, signal, tau, m):
        """Explicitly computes the UTRP for Frobenius comparison."""
        from pyrqa.settings import Settings
        from pyrqa.analysis_type import Cross
        from pyrqa.neighbourhood import Unthresholded
        from pyrqa.metric import EuclideanMetric
        from pyrqa.computation import RPComputation
        from pyrqa.time_series import TimeSeries

        ts = TimeSeries(signal, embedding_dimension=m, time_delay=tau)
        settings = Settings(ts, analysis_type=Cross,
                            neighbourhood=Unthresholded(),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=1)

        computation = RPComputation.create(settings, verbose=False)
        result = computation.run()
        return result.recurrence_matrix_reverse_normalized

    def validate_pipeline(self, signal, pipeline_name, reference_rqa=None, reference_data=None):
        from scipy.signal import resample

        # 1. Decompose the signal into subbands
        subbands = self.decompose_signal(signal, pipeline_name)
        results = []

        # 2. Iterate through each band (D1, D2, D3, D4, A4)
        for band in self.BAND_NAMES:
            freq = self.SUBBAND_FREQS[band]

            # Extract the specific subband signal
            if isinstance(subbands, (list, np.ndarray)) and len(subbands) == 1:
                subband_signal = subbands[0]
            else:
                subband_signal = subbands[band]

            # --- FIX: Handle Decimation Mismatch (DWT Direct) ---
            target_length = self.N
            if len(subband_signal) != target_length:
                subband_signal = resample(subband_signal, target_length)

            # 3. Standardize the signal
            if subband_signal.std() > 0:
                subband_signal = (subband_signal - subband_signal.mean()) / subband_signal.std()

            # 4. Parameter Injection: Use Reference Values to ensure 1:1 comparison
            if reference_rqa is not None and band in reference_rqa:
                tau = reference_rqa[band]['tau']
                m = reference_rqa[band]['m']
                epsilon = reference_rqa[band]['epsilon']
            else:
                # Fallback to AMI/FNN if no reference is provided
                tau, m = self.get_optimal_params(subband_signal)
                epsilon = RecurrencePlotGenerator.find_epsilon_for_target_rr(
                    signal=subband_signal, tau=tau, m=m, target_rr=0.1
                )

            # 5. Compute Pipeline RP and Distance Matrix (UTRP)
            compute_dist_matrix = reference_data is not None
            metrics = self.compute_rp_metrics(
                subband_signal, tau, m, epsilon,
                compute_distance_matrix=compute_dist_matrix
            )

            # Initialize absolute metrics
            hamming_distance = np.nan
            rp_pattern_corr = np.nan
            frobenius_distance = np.nan
            ibf = self.compute_inband_power_fraction(subband_signal, freq, fs=self.FS, band_name=band)

            # 6. Perform Direct Matrix-Level Comparisons
            if reference_data is not None and band in reference_data:
                ref_data = reference_data[band]

                # Compare Binary Recurrence Plots (Hamming & Correlation)
                if metrics['rp_matrix'] is not None and ref_data['rp_matrix'] is not None:
                    try:
                        # Extract raw matrices (handles potential wrapper objects)
                        ref_matrix = ref_data['rp_matrix'].recurrence_matrix_reverse_normalized \
                            if hasattr(ref_data['rp_matrix'], 'recurrence_matrix_reverse_normalized') \
                            else ref_data['rp_matrix']

                        pipe_matrix = metrics['rp_matrix'].recurrence_matrix_reverse_normalized \
                            if hasattr(metrics['rp_matrix'], 'recurrence_matrix_reverse_normalized') \
                            else metrics['rp_matrix']

                        hamming_distance = self.compute_hamming_distance(ref_matrix, pipe_matrix)
                        rp_pattern_corr = self.compute_rp_pattern_correlation(ref_matrix, pipe_matrix)
                    except Exception as e:
                        print(f"Error in structural comparison for {band}: {e}")

                # Compare Unthresholded Distance Matrices (Frobenius)
                if ref_data.get('distance_matrix') is not None and metrics['distance_matrix'] is not None:
                    frobenius_distance = self.compute_frobenius_distance(
                        ref_data['distance_matrix'],
                        metrics['distance_matrix']
                    )

            # 7. Compute RQA Deviations (Relative to Reference)
            if reference_rqa is not None and band in reference_rqa:
                ref = reference_rqa[band]
                det_deviation = abs(ref['DET'] - metrics['DET'])
                lam_deviation = abs(ref['LAM'] - metrics['LAM'])
                tt_deviation = abs(ref['TT'] - metrics['TT'])
                # Normalized length deviations
                long_diag_dev = abs(ref['L_longest_diagonal'] - metrics['L_longest_diagonal']) / target_length
                long_vert_dev = abs(ref['L_longest_vertical'] - metrics['L_longest_vertical']) / target_length
            else:
                # Default fallback values
                det_deviation, lam_deviation, tt_deviation = 0.0, 0.0, 0.0
                long_diag_dev, long_vert_dev = 0.0, 0.0

            # 8. Store Results
            results.append({
                'pipeline': pipeline_name,
                'band': band,
                'freq_hz': freq,
                'tau': tau,
                'm': m,
                'epsilon': epsilon,
                'DET': metrics['DET'],
                'LAM': metrics['LAM'],
                'TT': metrics['TT'],
                'ENTR': metrics['ENTR'],
                'L_mean_diagonal': metrics['L_mean_diagonal'],
                'L_longest_diagonal': metrics['L_longest_diagonal'],
                'L_longest_vertical': metrics['L_longest_vertical'],
                'det_deviation': det_deviation,
                'lam_deviation': lam_deviation,
                'tt_deviation': tt_deviation,
                'longest_diagonal_deviation': long_diag_dev,
                'longest_vertical_deviation': long_vert_dev,
                'hamming_distance': hamming_distance,
                'rp_pattern_corr': rp_pattern_corr,
                'frobenius_distance': frobenius_distance,
                'ibf': ibf
            })

        # Compute cross-band correlation for the entire set of subbands
        corr_matrix = self.compute_cross_band_correlation(subbands)
        avg_offdiag_corr = np.mean([
            corr_matrix[i, j]
            for i in range(len(self.BAND_NAMES))
            for j in range(len(self.BAND_NAMES))
            if i != j
        ])

        # Add avg_offdiag_corr to all results
        df = pd.DataFrame(results)
        df['avg_offdiag_corr'] = avg_offdiag_corr

        return df, None

    def run_leakage_analysis(self, n_trials=20):
        pipelines = ['swt_recon', 'swt_direct', 'dwt_recon', 'dwt_direct']
        leakage_results = []

        for trial in range(n_trials):
            for target_band, target_freq in self.SUBBAND_FREQS.items():
                # 1. Generate ONLY the pure sine for the target band
                t = np.arange(self.N) / self.FS
                phase = np.random.uniform(0, 2 * np.pi)
                pure_sine = np.sin(2 * np.pi * target_freq * t + phase)

                for pipe_name in pipelines:
                    # 2. Decompose the pure sine
                    outputs = self.decompose_signal(pure_sine, pipe_name)

                    # 3. Measure energy in EVERY output band
                    total_output_energy = sum(np.sum(outputs[b] ** 2) for b in self.BAND_NAMES)

                    for output_band in self.BAND_NAMES:
                        band_energy = np.sum(outputs[output_band] ** 2)
                        energy_fraction = band_energy / total_output_energy if total_output_energy > 0 else 0

                        leakage_results.append({
                            'trial': trial,
                            'pipeline': pipe_name,
                            'input_sine': target_band,  # The source frequency
                            'output_subband': output_band,  # Where energy appeared
                            'energy_fraction': energy_fraction
                        })

        return pd.DataFrame(leakage_results)

    def validate_with_noise(self, signal, pipeline_name, reference_epsilons, snr_db, n_realizations=50, reference_rqa=None, reference_data=None):
        all_results = []

        for _ in range(n_realizations):
            signal_power = np.mean(signal ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.sqrt(noise_power) * np.random.randn(len(signal))
            noisy_signal = signal + noise

            df, _ = self.validate_pipeline(noisy_signal, pipeline_name, reference_epsilons, reference_rqa, reference_data)
            all_results.append(df)

        combined = pd.concat(all_results, ignore_index=True)

        metric_cols = ['det_deviation', 'lam_deviation', 'tt_deviation','longest_diagonal_deviation', 'longest_vertical_deviation']

        results_mean = combined.groupby('band')[metric_cols].mean().reset_index()
        results_std = combined.groupby('band')[metric_cols].std().reset_index()

        results_mean['pipeline'] = pipeline_name
        results_mean['snr_db'] = snr_db
        results_std['pipeline'] = pipeline_name
        results_std['snr_db'] = snr_db

        return results_mean, results_std

    def compute_cross_band_correlation(self, subbands):
        """Compute correlation matrix between all subband pairs."""
        from scipy.signal import resample

        corr_matrix = np.zeros((len(self.BAND_NAMES), len(self.BAND_NAMES)))

        for i, b1 in enumerate(self.BAND_NAMES):
            for j, b2 in enumerate(self.BAND_NAMES):
                s1 = subbands[b1]
                s2 = subbands[b2]

                if len(s1) != self.N:
                    s1 = resample(s1, self.N)
                if len(s2) != self.N:
                    s2 = resample(s2, self.N)

                corr_matrix[i, j] = np.corrcoef(s1, s2)[0, 1]

        return corr_matrix

    def perform_shift_invariance_tests(self, signal, shift=5):
        """Test shift invariance by comparing RPs of original vs shifted signals."""
        from scipy.signal import resample

        results = []

        original = signal
        shifted = np.roll(signal, shift)

        for pipe in self.PIPELINES:
            subbands_orig = self.decompose_signal(original, pipe)
            subbands_shift = self.decompose_signal(shifted, pipe)

            for band in self.BAND_NAMES:
                s1 = subbands_orig[band]
                s2 = subbands_shift[band]

                if len(s1) != self.N:
                    s1 = resample(s1, self.N)
                if len(s2) != self.N:
                    s2 = resample(s2, self.N)

                if s1.std() > 0:
                    s1 = (s1 - s1.mean()) / s1.std()
                if s2.std() > 0:
                    s2 = (s2 - s2.mean()) / s2.std()

                tau, m = self.get_optimal_params(original)
                eps = RecurrencePlotGenerator.find_epsilon_for_target_rr(s1, tau, m, 0.1)

                m1 = self.compute_rp_metrics(s1, tau, m, eps, compute_distance_matrix=True)
                m2 = self.compute_rp_metrics(s2, tau, m, eps, compute_distance_matrix=True)

                rp1 = m1['rp_matrix']
                rp2 = m2['rp_matrix']

                if hasattr(rp1, 'recurrence_matrix_reverse_normalized'):
                    rp1 = rp1.recurrence_matrix_reverse_normalized
                if hasattr(rp2, 'recurrence_matrix_reverse_normalized'):
                    rp2 = rp2.recurrence_matrix_reverse_normalized

                hamming_shift = self.compute_hamming_distance(rp1, rp2)
                frobenius_shift = self.compute_frobenius_distance(
                    m1['distance_matrix'],
                    m2['distance_matrix']
                ) if m1['distance_matrix'] is not None and m2['distance_matrix'] is not None else np.nan

                results.append({
                    'pipeline': pipe,
                    'band': band,
                    'shift': shift,
                    'hamming_shift_diff': hamming_shift,
                    'frobenius_shift_diff': frobenius_shift
                })

        return pd.DataFrame(results)

    def run_shift_invariance_analysis(self, n_trials=20, shift=5):
        """Run shift invariance tests over multiple trials with random phases."""
        results = []

        for trial in range(n_trials):
            signal = self.generate_composite_sine(random_phase=True)
            df_shift = self.perform_shift_invariance_tests(signal, shift=shift)
            df_shift['trial'] = trial
            results.append(df_shift)

        return pd.concat(results, ignore_index=True)

    def analyze_aliasing_and_leakage(self, df_alias, df_leakage):
        """Compute statistical summaries for aliasing and leakage tests."""
        from scipy.stats import mannwhitneyu

        df_der = []

        for (pipe, trial), sub in df_leakage.groupby(['pipeline', 'trial']):
            diag = sub[sub['input_sine'] == sub['output_subband']]['energy_fraction'].sum()
            total = sub['energy_fraction'].sum()

            df_der.append({
                'pipeline': pipe,
                'trial': trial,
                'DER': diag / total
            })

        df_der = pd.DataFrame(df_der)

        df_alias_summary = df_alias.groupby(['pipeline', 'trial'])[['hamming_shift_diff', 'frobenius_shift_diff']].mean().reset_index()

        def cohens_d(x, y):
            return (x.mean() - y.mean()) / np.sqrt((x.var() + y.var()) / 2)

        stats_results = []

        pipelines = df_alias_summary['pipeline'].unique()

        for i in range(len(pipelines)):
            for j in range(i + 1, len(pipelines)):
                p1 = pipelines[i]
                p2 = pipelines[j]

                x1 = df_alias_summary[df_alias_summary['pipeline'] == p1]['hamming_shift_diff']
                x2 = df_alias_summary[df_alias_summary['pipeline'] == p2]['hamming_shift_diff']

                stat_a, p_a = mannwhitneyu(x1, x2)
                d_a = cohens_d(x1, x2)

                y1 = df_der[df_der['pipeline'] == p1]['DER']
                y2 = df_der[df_der['pipeline'] == p2]['DER']

                stat_l, p_l = mannwhitneyu(y1, y2)
                d_l = cohens_d(y1, y2)

                stats_results.append({
                    'comparison': f'{p1} vs {p2}',
                    'alias_p': p_a,
                    'alias_cohens_d': d_a,
                    'leakage_p': p_l,
                    'leakage_cohens_d': d_l
                })

        df_stats = pd.DataFrame(stats_results)
        df_stats.to_csv(os.path.join(self.OUTPUT_DIR, "stats_summary.csv"), index=False)

        print("\n=== DER (Leakage) Summary ===")
        print(df_der.groupby('pipeline')['DER'].agg(['mean', 'std']))

        print("\n=== Aliasing (Hamming) Summary ===")
        print(df_alias_summary.groupby('pipeline')['hamming_shift_diff'].agg(['mean', 'std']))

        print("\n=== Statistical Comparison ===")
        print(df_stats)

        return df_der, df_alias_summary, df_stats

    def run_full_validation(self, use_reference_rqa=True):
        reference_rqa = None
        reference_data = None

        if use_reference_rqa:
            ref_path = os.path.join(self.OUTPUT_DIR, 'reference_rqa_values.json')
            ref_data_path = os.path.join(self.OUTPUT_DIR, 'reference_rqa_values_data.npz')

            if os.path.exists(ref_path):
                reference_rqa = self.load_reference_rqa_values(ref_path)
                if os.path.exists(ref_data_path):
                    reference_data = self.load_reference_data(ref_data_path)
            else:
                reference_rqa = self.generate_reference_rqa_values(ref_path)
                reference_data = self.load_reference_data(ref_data_path)

        signal = self.generate_composite_sine()
        clean_results = {}
        reference_epsilons = None

        for pipeline in self.PIPELINES:
            if pipeline == 'swt_recon':
                df, epsilons = self.validate_pipeline(signal, pipeline, reference_rqa=reference_rqa, reference_data=reference_data)
                reference_epsilons = epsilons
            else:
                df, _ = self.validate_pipeline(signal, pipeline, reference_epsilons, reference_rqa, reference_data)

            clean_results[pipeline] = df

        noise_results = {}

        leakage = self.run_leakage_analysis(100)

        for snr_db in [20, 10, 5, 1]:
            for pipeline in self.PIPELINES:
                mean_df, std_df = self.validate_with_noise(
                    signal, pipeline, reference_epsilons, snr_db, n_realizations=50, reference_rqa=reference_rqa, reference_data=reference_data
                )

                noise_results[f"{pipeline}_snr{snr_db}_mean"] = mean_df
                noise_results[f"{pipeline}_snr{snr_db}_std"] = std_df

        return clean_results, noise_results, reference_epsilons, leakage