import os
import sys
import numpy as np
import pandas as pd
import json
from scipy.stats import mannwhitneyu
from scipy.signal import resample

from src.lorenz_solver.lorenz import LorenzSolver
from src.recurrence_plot.AMI import AMI
from src.recurrence_plot.FNN import FNN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.pipeline_wavelets import Pipeline
from recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator


class SubbandValidator:
    def __init__(self, is_sine, is_lorenz):
        self.FS = 128
        self.N = 512
        self.LEVEL = 4
        self.WAVELET = 'db4'
        self.is_sine = is_sine
        self.is_lorenz = is_lorenz

        self.SUBBAND_FREQS = {
            'A4': 2, 'D4': 6, 'D3': 12, 'D2': 24, 'D1': 48
        }
        self.BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']
        self.PIPELINES = ['swt_recon', 'swt_direct', 'dwt_recon', 'dwt_direct']

        if self.is_sine:
            self.OUTPUT_DIR = '/Users/filipsulich/Developer/uni/Thesis/Thesis/outputs/experiment1_sine2'
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
            for lock_file in glob.glob(lock_pattern):
                try:
                    os.remove(lock_file)
                except:
                    pass
        except:
            pass

    def generate_lorenz_wave(self, rho, beta, sigma):
        lorenz = LorenzSolver(rho=rho, beta=beta, sigma=sigma)
        signal = lorenz.generate_lorenz_signal(n_points=10000, target_fs=self.FS, dt=0.01)
        #TODO finish
        return signal

    def generate_composite_sine(self, random_phase=False):
        t = np.arange(self.N) / self.FS
        signal = np.zeros(self.N)
        for band, freq in self.SUBBAND_FREQS.items():
            phase = np.random.uniform(0, 2 * np.pi) if random_phase else 0
            signal += np.sin(2 * np.pi * freq * t + phase)
        return signal

    def analyze_aliasing_and_leakage(self, df_alias, df_leakage, output_dir):
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
        df_stats.to_csv(os.path.join(output_dir, "stats_summary.csv"), index=False)

        print("\n=== DER (Leakage) Summary ===")
        print(df_der.groupby('pipeline')['DER'].agg(['mean', 'std']))

        print("\n=== Aliasing (Hamming) Summary ===")
        print(df_alias_summary.groupby('pipeline')['hamming_shift_diff'].agg(['mean', 'std']))

        print("\n=== Statistical Comparison ===")
        print(df_stats)

        return df_der, df_alias_summary, df_stats

    def get_optimal_params(self, signal):
        ami = AMI(signal)
        tau = ami.get_first_minimum()
        fnn = FNN(signal=signal, tau=tau)
        m = fnn.get_optimal_dimension()
        return int(tau), int(m)

    def get_unthresholded_matrix(self, signal, tau, m):
        from pyrqa.settings import Settings
        from pyrqa.analysis_type import Cross
        from pyrqa.neighbourhood import Unthresholded
        from pyrqa.metric import EuclideanMetric
        from pyrqa.computation import RPComputation
        from pyrqa.time_series import TimeSeries
        ts = TimeSeries(signal, embedding_dimension=m, time_delay=tau)
        settings = Settings(ts, analysis_type=Cross, neighbourhood=Unthresholded(),
                            similarity_measure=EuclideanMetric, theiler_corrector=1)
        return RPComputation.create(settings, verbose=False).run().recurrence_matrix_reverse_normalized

    def compute_rp_metrics(self, signal, tau, m, epsilon, compute_distance_matrix=False):
        try:
            os.environ['PYOPENCL_CTX'] = '0'
            rp_gen = RecurrencePlotGenerator(signal=signal, time_delay=tau, embedding_dim=m,
                                             path=self.OUTPUT_DIR, epsilon=epsilon, unthresholded=False)
            rqa_result, rp_matrix = rp_gen.compute_rqa_binary()

            # Handle NaN values in RQA metrics
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

            entr = rqa_result.entropy_diagonal_lines
            if np.isnan(entr): entr = 0.0

            # Compute distance matrix if requested
            distance_matrix = None
            if compute_distance_matrix:
                distance_matrix = self.get_unthresholded_matrix(signal, tau, m)

            return {
                'DET': det,
                'ENTR': entr,
                'LAM': lam,
                'L_mean_diagonal': l_mean_diag,
                'L_longest_diagonal': longest_diagonal,
                'L_longest_vertical': longest_vertical_line,
                'TT': tt,
                'rp_matrix': rp_matrix,
                'distance_matrix': distance_matrix
            }
        except Exception as e:
            print(f"Error computing RP metrics: {e}")
            return {
                'DET': 0.0,
                'ENTR': 0.0,
                'LAM': 0.0,
                'L_mean_diagonal': 0.0,
                'L_longest_diagonal': 0.0,
                'L_longest_vertical': 0.0,
                'TT': 0.0,
                'rp_matrix': None,
                'distance_matrix': None
            }

    def validate_pipeline(self, signal, pipeline_name, reference_rqa=None, reference_data=None):
        subbands = self.decompose_signal(signal, pipeline_name)
        results = []

        for band in self.BAND_NAMES:
            freq = self.SUBBAND_FREQS[band]

            # Extract the specific subband signal
            if isinstance(subbands, (list, np.ndarray)) and len(subbands) == 1:
                sub_sig = subbands[0]
            else:
                sub_sig = subbands[band]

            # Handle decimation mismatch (resample to target length if needed)
            if len(sub_sig) != self.N:
                sub_sig = resample(sub_sig, self.N)

            # Standardize the signal
            if sub_sig.std() > 0:
                sub_sig = (sub_sig - sub_sig.mean()) / sub_sig.std()

            # Parameter Injection: Use Reference Values for 1:1 comparison
            if reference_rqa and band in reference_rqa:
                tau = reference_rqa[band]['tau']
                m = reference_rqa[band]['m']
                epsilon = reference_rqa[band]['epsilon']
            else:
                # Fallback to AMI/FNN if no reference is provided
                tau, m = self.get_optimal_params(sub_sig)
                epsilon = RecurrencePlotGenerator.find_epsilon_for_target_rr(sub_sig, tau, m, 0.1)

            # Compute Pipeline RP and Distance Matrix
            compute_dist_matrix = reference_data is not None
            metrics = self.compute_rp_metrics(sub_sig, tau, m, epsilon, compute_distance_matrix=compute_dist_matrix)

            # Initialize structural comparison metrics
            hamming_distance = np.nan
            rp_pattern_corr = np.nan
            frobenius_distance = np.nan

            # Perform Direct Matrix-Level Comparisons
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

            # Compute RQA Deviations (Relative to Reference)
            if reference_rqa is not None and band in reference_rqa:
                ref = reference_rqa[band]
                det_deviation = abs(ref['DET'] - metrics['DET'])
                lam_deviation = abs(ref['LAM'] - metrics['LAM'])
                tt_deviation = abs(ref['TT'] - metrics['TT'])
                # Normalized length deviations
                long_diag_dev = abs(ref['L_longest_diagonal'] - metrics['L_longest_diagonal']) / self.N
                long_vert_dev = abs(ref['L_longest_vertical'] - metrics['L_longest_vertical']) / self.N
            else:
                # Default fallback values
                det_deviation, lam_deviation, tt_deviation = 0.0, 0.0, 0.0
                long_diag_dev, long_vert_dev = 0.0, 0.0

            # Compute in-band power fraction
            ibf = self.compute_inband_power_fraction(sub_sig, freq, self.FS, band)

            # Compute cross-band correlation
            corr_matrix = self.compute_cross_band_correlation(subbands)
            avg_offdiag_corr = np.mean([
                corr_matrix[i, j]
                for i in range(len(self.BAND_NAMES))
                for j in range(len(self.BAND_NAMES))
                if i != j
            ])

            # Store Results
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
                'ibf': ibf,
                'avg_offdiag_corr': avg_offdiag_corr,
            })

        return pd.DataFrame(results), None

    def run_statistical_comparison(self, n_trials=100, use_reference_rqa=True):
        from scipy.stats import ttest_ind

        # Ensure reference data is loaded
        reference_rqa = None
        reference_data = None
        if use_reference_rqa:
            ref_path = os.path.join(self.OUTPUT_DIR, 'reference_rqa_values.json')
            ref_data_path = os.path.join(self.OUTPUT_DIR, 'reference_rqa_values_data.npz')
            if os.path.exists(ref_path):
                reference_rqa = self.load_reference_rqa_values(ref_path)
                if os.path.exists(ref_data_path):
                    reference_data = self.load_reference_data(ref_data_path)

        # Dictionary to store the AVERAGE deviation per trial
        # This is what makes the plot look like Image 1
        trial_averages = {
            pipe: {metric: [] for metric in ['DET', 'LAM', 'TT', 'HAMMING', 'RP_CORR']}
            for pipe in self.PIPELINES
        }

        all_raw_data = []

        for trial in range(n_trials):
            if (trial + 1) % 10 == 0:
                print(f"Processing Trial {trial + 1}/{n_trials}...")

            # Generate signal with random phase to ensure robustness
            signal = self.generate_composite_sine(random_phase=True)

            for pipeline in self.PIPELINES:
                # validate_pipeline returns a DataFrame containing results for ALL bands
                df_bands, _ = self.validate_pipeline(
                    signal,
                    pipeline,
                    reference_rqa=reference_rqa,
                    reference_data=reference_data
                )

                # Save raw data for deep-dive CSVs
                df_bands['trial'] = trial
                all_raw_data.append(df_bands)

                # --- THE FIX: TRIAL AVERAGING ---
                # Instead of plotting every row in df_bands, we take the mean of the deviations.
                # This collapses the 5 subbands into 1 average score for this specific trial.
                trial_averages[pipeline]['DET'].append(df_bands['det_deviation'].mean())
                trial_averages[pipeline]['LAM'].append(df_bands['lam_deviation'].mean())
                trial_averages[pipeline]['TT'].append(df_bands['tt_deviation'].mean())
                trial_averages[pipeline]['HAMMING'].append(df_bands['hamming_distance'].mean())
                trial_averages[pipeline]['RP_CORR'].append(df_bands['rp_pattern_corr'].mean())

        # Convert lists to numpy arrays for statistical testing
        for pipe in self.PIPELINES:
            for metric in trial_averages[pipe]:
                trial_averages[pipe][metric] = np.array(trial_averages[pipe][metric])

        # Save master raw data
        master_df = pd.concat(all_raw_data, ignore_index=True)
        master_df.to_csv(os.path.join(self.OUTPUT_DIR, 'raw_validation_results.csv'), index=False)

        return trial_averages, master_df

    def compute_hamming_distance(self, rp_gt, rp_pipe):
        """Compute normalized Hamming distance between two binary recurrence plots."""
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

    def compute_frobenius_distance(self, dist_matrix_ref, dist_matrix_pipe):
        """Compute Frobenius norm (Euclidean distance) between two distance matrices."""
        if dist_matrix_ref is None or dist_matrix_pipe is None:
            return np.nan

        if dist_matrix_ref.shape != dist_matrix_pipe.shape:
            min_size = min(dist_matrix_ref.shape[0], dist_matrix_pipe.shape[0])
            dist_matrix_ref = dist_matrix_ref[:min_size, :min_size]
            dist_matrix_pipe = dist_matrix_pipe[:min_size, :min_size]

        # Frobenius norm: sqrt(sum of squared differences)
        frobenius_dist = np.linalg.norm(dist_matrix_ref - dist_matrix_pipe, ord='fro')
        return frobenius_dist

    def compute_between_band_corr(self, b1, b2):
        if b1.shape != b2.shape: return np.nan
        return np.corrcoef(b1.ravel(), b2.ravel())[0, 1]

    def compute_diagonal_energy_ratio(self, df):
        results = []

        for pipe in df['pipeline'].unique():
            sub = df[df['pipeline'] == pipe]

            correct_energy = 0
            total_energy = 0

            for _, row in sub.iterrows():
                total_energy += row['energy_fraction']
                if row['input_sine'] == row['output_subband']:
                    correct_energy += row['energy_fraction']

            results.append({
                'pipeline': pipe,
                'diagonal_energy_ratio': correct_energy / total_energy
            })

        return pd.DataFrame(results)

    def perform_shift_invariance_tests(self, signal, shift=5):
        results = []

        original = signal
        shifted = np.roll(signal, shift)

        for pipe in self.PIPELINES:
            subbands_orig = self.decompose_signal(original, pipe)
            subbands_shift = self.decompose_signal(shifted, pipe)

            for band in self.BAND_NAMES:
                s1 = subbands_orig[band]
                s2 = subbands_shift[band]

                if len(s1) != self.N: s1 = resample(s1, self.N)
                if len(s2) != self.N: s2 = resample(s2, self.N)

                tau, m = self.get_optimal_params(original)
                eps = RecurrencePlotGenerator.find_epsilon_for_target_rr(s1, tau, m, 0.1)

                m1 = self.compute_rp_metrics(s1, tau, m, eps)
                m2 = self.compute_rp_metrics(s2, tau, m, eps)

                rp1 = m1['rp_matrix']
                rp2 = m2['rp_matrix']

                if hasattr(rp1, 'recurrence_matrix_reverse_normalized'):
                    rp1 = rp1.recurrence_matrix_reverse_normalized
                if hasattr(rp2, 'recurrence_matrix_reverse_normalized'):
                    rp2 = rp2.recurrence_matrix_reverse_normalized

                results.append({
                    'pipeline': pipe,
                    'band': band,
                    'shift': shift,
                    'hamming_shift_diff': self.compute_hamming_distance(rp1, rp2),
                    'frobenius_shift_diff': self.compute_frobenius_distance(
                        self.get_unthresholded_matrix(s1, tau, m),
                        self.get_unthresholded_matrix(s2, tau, m)
                    )
                })

        return pd.DataFrame(results)

    def run_shift_invariance_analysis(self, n_trials=20, shift=5):
        results = []

        for trial in range(n_trials):
            signal = self.generate_composite_sine(random_phase=True)

            df_shift = self.perform_shift_invariance_tests(signal, shift=shift)

            df_shift['trial'] = trial
            results.append(df_shift)

        return pd.concat(results, ignore_index=True)

    def compute_cross_band_correlation(self, subbands):
        corr_matrix = np.zeros((len(self.BAND_NAMES), len(self.BAND_NAMES)))

        for i, b1 in enumerate(self.BAND_NAMES):
            for j, b2 in enumerate(self.BAND_NAMES):
                s1 = subbands[b1]
                s2 = subbands[b2]

                if len(s1) != self.N: s1 = resample(s1, self.N)
                if len(s2) != self.N: s2 = resample(s2, self.N)

                corr_matrix[i, j] = np.corrcoef(s1, s2)[0, 1]

        return corr_matrix

    def compute_inband_power_fraction(self, sig, freq, fs, band_name):
        from scipy.signal import welch
        f, psd = welch(sig, fs=fs, nperseg=len(sig) // 2)
        f_low, f_high = (0, 4.5) if band_name == 'A4' else (freq * 0.8, freq * 1.2)
        idx = (f >= f_low) & (f <= f_high)
        return np.trapezoid(psd[idx], f[idx]) / np.trapezoid(psd, f) if np.trapezoid(psd, f) > 0 else 0

    def decompose_signal(self, signal, pipeline_name):
        p = Pipeline(pipeline_name='all', level=self.LEVEL, wavelet=self.WAVELET)
        return getattr(p, f"pipeline_{pipeline_name}")(signal)

    def run_leakage_analysis(self, n_trials=20):
        res = []
        for trial in range(n_trials):
            for b_in, f_in in self.SUBBAND_FREQS.items():
                sig = np.sin(2 * np.pi * f_in * np.arange(self.N) / self.FS + np.random.uniform(0, 2 * np.pi))
                for pipe in self.PIPELINES:
                    out = self.decompose_signal(sig, pipe)
                    total_e = sum(np.sum(out[b] ** 2) for b in self.BAND_NAMES)
                    for b_out in self.BAND_NAMES:
                        res.append({'pipeline': pipe, 'trial': trial, 'input_sine': b_in, 'output_subband': b_out,
                                    'energy_fraction': np.sum(out[b_out] ** 2) / total_e})
        return pd.DataFrame(res)

    def generate_reference_rqa_values(self, save_path):
        reference_values, reference_data = {}, {}
        for band, freq in self.SUBBAND_FREQS.items():
            sig = np.sin(2 * np.pi * freq * (np.arange(self.N) / self.FS))
            sig = (sig - sig.mean()) / sig.std()
            tau, m = self.get_optimal_params(sig)
            eps = RecurrencePlotGenerator.find_epsilon_for_target_rr(sig, tau, m, 0.1)
            met = self.compute_rp_metrics(sig, tau, m, eps, compute_distance_matrix=True)

            # Handle NaN: when LAM=0, TT is undefined, set to 0.0
            tt_value = 0.0 if np.isnan(met['TT']) else float(met['TT'])

            reference_values[band] = {'tau': int(tau), 'm': int(m), 'epsilon': float(eps), 'DET': float(met['DET']),
                                      'LAM': float(met['LAM']), 'TT': tt_value, 'ENTR': float(met['ENTR']),
                                      'L_mean_diagonal': float(met['L_mean_diagonal']),
                                      'L_longest_diagonal': float(met['L_longest_diagonal']),
                                      'L_longest_vertical': float(met['L_longest_vertical'])}
            reference_data[band] = {'rp_matrix': met['rp_matrix'], 'distance_matrix': met['distance_matrix'],
                                    'pure_sine': sig}

        with open(save_path, 'w') as f: json.dump(reference_values, f, indent=2)

        # Extract numpy arrays from RPResult objects before saving to npz
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

        np.savez(save_path.replace('.json', '_data.npz'), **rp_matrices, **distance_matrices, **signals)
        return reference_values

    def load_reference_rqa_values(self, p):
        with open(p, 'r') as f: return json.load(f)

    def load_reference_data(self, p):
        d = np.load(p, allow_pickle=True)
        return {b: {'rp_matrix': d[f'{b}_rp'], 'pure_sine': d[f'{b}_signal'], 'distance_matrix': d.get(f'{b}_dist')} for
                b in self.BAND_NAMES}
