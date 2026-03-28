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
            self.OUTPUT_DIR = '/Users/filipsulich/Developer/uni/Thesis/Thesis/outputs/experiment1_sine'
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
        return signal

    def generate_composite_sine(self, random_phase=False):
        t = np.arange(self.N) / self.FS
        signal = np.zeros(self.N)
        for band, freq in self.SUBBAND_FREQS.items():
            phase = np.random.uniform(0, 2 * np.pi) if random_phase else 0
            signal += np.sin(2 * np.pi * freq * t + phase)
        return signal

    def add_awgn(self, signal, snr_db):
        if np.isinf(snr_db): return signal
        sig_avg_watts = np.mean(signal ** 2)
        sig_avg_db = 10 * np.log10(sig_avg_watts)
        noise_avg_db = sig_avg_db - snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        noise_volts = np.random.normal(0, np.sqrt(noise_avg_watts), len(signal))
        return signal + noise_volts

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
            distance_matrix = self.get_unthresholded_matrix(signal, tau, m) if compute_distance_matrix else None

            return {
                'DET': getattr(rqa_result, 'determinism', 0.0),
                'ENTR': getattr(rqa_result, 'entropy_diagonal_lines', 0.0),
                'LAM': getattr(rqa_result, 'laminarity', 0.0),
                'L_mean_diagonal': getattr(rqa_result, 'average_diagonal_line', 0.0),
                'L_longest_diagonal': getattr(rqa_result, 'longest_diagonal_line', 0.0),
                'L_longest_vertical': getattr(rqa_result, 'longest_vertical_line', 0.0),
                'TT': getattr(rqa_result, 'trapping_time', 0.0),
                'rp_matrix': rp_matrix, 'distance_matrix': distance_matrix
            }
        except:
            return {m: 0.0 for m in
                    ['DET', 'ENTR', 'LAM', 'L_mean_diagonal', 'L_longest_diagonal', 'L_longest_vertical', 'TT']} | {
                'rp_matrix': None, 'distance_matrix': None}

    def validate_pipeline(self, signal, pipeline_name, reference_rqa=None, reference_data=None):
        subbands = self.decompose_signal(signal, pipeline_name)
        results = []
        for band in self.BAND_NAMES:
            freq = self.SUBBAND_FREQS[band]
            sub_sig = subbands[0] if (isinstance(subbands, (list, np.ndarray)) and len(subbands) == 1) else subbands[
                band]

            if len(sub_sig) != self.N: sub_sig = resample(sub_sig, self.N)
            if sub_sig.std() > 0: sub_sig = (sub_sig - sub_sig.mean()) / sub_sig.std()

            if reference_rqa and band in reference_rqa:
                tau, m, epsilon = reference_rqa[band]['tau'], reference_rqa[band]['m'], reference_rqa[band]['epsilon']
            else:
                tau, m = self.get_optimal_params(sub_sig)
                epsilon = RecurrencePlotGenerator.find_epsilon_for_target_rr(sub_sig, tau, m, 0.1)

            metrics = self.compute_rp_metrics(sub_sig, tau, m, epsilon,
                                              compute_distance_matrix=(reference_data is not None))

            h_dist, p_corr, f_dist = np.nan, np.nan, np.nan
            if reference_data and band in reference_data:
                ref_m = reference_data[band]['rp_matrix']
                ref_m = ref_m.recurrence_matrix_reverse_normalized if hasattr(ref_m,
                                                                              'recurrence_matrix_reverse_normalized') else ref_m
                pipe_m = metrics['rp_matrix']
                pipe_m = pipe_m.recurrence_matrix_reverse_normalized if hasattr(pipe_m,
                                                                                'recurrence_matrix_reverse_normalized') else pipe_m

                h_dist = self.compute_hamming_distance(ref_m, pipe_m)
                p_corr = self.compute_rp_pattern_correlation(ref_m, pipe_m)
                if reference_data[band]['distance_matrix'] is not None and metrics['distance_matrix'] is not None:
                    f_dist = self.compute_frobenius_distance(reference_data[band]['distance_matrix'],
                                                             metrics['distance_matrix'])

            results.append({
                'pipeline': pipeline_name, 'band': band, 'freq_hz': freq, 'tau': tau, 'm': m, 'epsilon': epsilon,
                'DET': metrics['DET'], 'LAM': metrics['LAM'], 'TT': metrics['TT'], 'ENTR': metrics['ENTR'],
                'L_mean_diagonal': metrics['L_mean_diagonal'], 'L_longest_diagonal': metrics['L_longest_diagonal'],
                'L_longest_vertical': metrics['L_longest_vertical'],
                'det_deviation': abs(metrics['DET'] - reference_rqa[band]['DET']) if reference_rqa else 0,
                'lam_deviation': abs(metrics['LAM'] - reference_rqa[band]['LAM']) if reference_rqa else 0,
                'tt_deviation': abs(metrics['TT'] - reference_rqa[band]['TT']) if reference_rqa else 0,
                'hamming_distance': h_dist, 'rp_pattern_corr': p_corr, 'frobenius_distance': f_dist,
                'ibf': self.compute_inband_power_fraction(sub_sig, freq, self.FS, band)
            })
        return pd.DataFrame(results), None

    def compute_hamming_distance(self, r1, r2):
        if r1.shape != r2.shape: return np.nan
        return np.sum(np.abs((r1 > 0).astype(int) - (r2 > 0).astype(int))) / (r1.shape[0] ** 2)

    def compute_rp_pattern_correlation(self, r1, r2):
        if r1.shape != r2.shape: return np.nan
        return np.corrcoef(r1.ravel(), r2.ravel())[0, 1]

    def compute_frobenius_distance(self, d1, d2):
        if d1.shape != d2.shape: return np.nan
        return np.linalg.norm(d1 - d2, ord='fro')

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
                        res.append({'pipeline': pipe, 'input_sine': b_in, 'output_subband': b_out,
                                    'energy_fraction': np.sum(out[b_out] ** 2) / total_e})
        return pd.DataFrame(res)

    def validate_with_noise(self, signal, pipeline_name, ref_eps, snr_db, n_real=50, ref_rqa=None, ref_data=None):
        all_res = []
        for _ in range(n_real):
            noisy = self.add_awgn(signal, snr_db)
            df, _ = self.validate_pipeline(noisy, pipeline_name, ref_rqa, ref_data)
            all_res.append(df)
        combined = pd.concat(all_res, ignore_index=True)
        m_cols = ['det_deviation', 'lam_deviation', 'tt_deviation']
        return combined.groupby('band')[m_cols].mean().reset_index(), combined.groupby('band')[
            m_cols].std().reset_index()

    def generate_reference_rqa_values(self, save_path):
        reference_values, reference_data = {}, {}
        for band, freq in self.SUBBAND_FREQS.items():
            sig = np.sin(2 * np.pi * freq * (np.arange(self.N) / self.FS))
            sig = (sig - sig.mean()) / sig.std()
            tau, m = self.get_optimal_params(sig)
            eps = RecurrencePlotGenerator.find_epsilon_for_target_rr(sig, tau, m, 0.1)
            met = self.compute_rp_metrics(sig, tau, m, eps, compute_distance_matrix=True)
            reference_values[band] = {'tau': int(tau), 'm': int(m), 'epsilon': float(eps), 'DET': float(met['DET']),
                                      'LAM': float(met['LAM']), 'TT': float(met['TT']), 'ENTR': float(met['ENTR']),
                                      'L_mean_diagonal': float(met['L_mean_diagonal']),
                                      'L_longest_diagonal': float(met['L_longest_diagonal']),
                                      'L_longest_vertical': float(met['L_longest_vertical'])}
            reference_data[band] = {'rp_matrix': met['rp_matrix'], 'distance_matrix': met['distance_matrix'],
                                    'pure_sine': sig}
        with open(save_path, 'w') as f: json.dump(reference_values, f, indent=2)
        np.savez(save_path.replace('.json', '_data.npz'),
                 **{f'{b}_rp': reference_data[b]['rp_matrix'] for b in self.BAND_NAMES},
                 **{f'{b}_dist': reference_data[b]['distance_matrix'] for b in self.BAND_NAMES},
                 **{f'{b}_signal': reference_data[b]['pure_sine'] for b in self.BAND_NAMES})
        return reference_values

    def load_reference_rqa_values(self, p):
        with open(p, 'r') as f: return json.load(f)

    def load_reference_data(self, p):
        d = np.load(p, allow_pickle=True)
        return {b: {'rp_matrix': d[f'{b}_rp'], 'pure_sine': d[f'{b}_signal'], 'distance_matrix': d.get(f'{b}_dist')} for
                b in self.BAND_NAMES}