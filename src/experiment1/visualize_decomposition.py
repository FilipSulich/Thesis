import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from scipy.signal import resample, firwin, filtfilt

# Path adjustment for your local src/pipelines structure
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recurrence_plot.AMI import AMI
from src.recurrence_plot.FNN import FNN
from pipelines.pipeline_wavelets import Pipeline
from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

os.environ['PYOPENCL_NO_CACHE'] = '1'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'


class DecompositionVisualizer:
    def __init__(self):
        self.FS = 128
        self.N = 1024  # Original length
        self.TRIM = 0  # Samples to discard from each edge
        self.LEVEL = 4
        self.WAVELET = 'db4'
        self.transient = 2000

        self.SUBBAND_FREQS = {
            'A4': 2, 'D4': 6, 'D3': 12, 'D2': 24, 'D1': 48
        }
        self.BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']
        self.OUTPUT_DIR = '/Users/filipsulich/Developer/uni/Thesis/Thesis/outputs/experiment1_synthetic'
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        self.reference_params = {}

    def _trim_signal(self, signal):
        """Discards the first and last self.TRIM samples."""
        if self.TRIM == 0:
            return signal
        if len(signal) <= 2 * self.TRIM:
            print(f"Warning: Signal length ({len(signal)}) is too short to trim {self.TRIM} from both sides.")
            return signal
        return signal[self.TRIM: -self.TRIM]

    def _get_selected_signal(self, system_type):
        if system_type == 'lorenz':
            return self.generate_lorenz_system()[:self.N]
        elif system_type == 'rossler':
            return self._generate_rossler()
        else:
            return self.generate_composite_sine()

    def decompose_fir(self, signal):
        """Zero-phase FIR filter bank decomposition."""
        subbands = {}
        nyq = 0.5 * self.FS
        numtaps = 101
        ranges = {
            'A4': [0.1, 4], 'D4': [4, 8], 'D3': [8, 16],
            'D2': [16, 32], 'D1': [32, 63.5]
        }
        for band, (low, high) in ranges.items():
            taps = firwin(numtaps, [low / nyq, high / nyq], pass_zero=False)
            # Using filtfilt for zero-phase filtering
            subbands[band] = filtfilt(taps, 1.0, signal)
        return subbands

    def prepare_reference_parameters(self, system_type):
        """
        Executes FIR decomposition and trims outputs to calculate
        baseline embedding parameters.
        """
        print(f"--- Calculating reference parameters ($\\tau, m$) via Trimmed FIR for {system_type.upper()} ---")
        signal_clean = self._get_selected_signal(system_type)
        fir_subbands = self.decompose_fir(signal_clean)

        self.reference_params = {}
        for band in self.BAND_NAMES:
            # Apply trimming to eliminate edge effects
            sub_sig = self._trim_signal(fir_subbands[band])
            sub_norm = (sub_sig - sub_sig.mean()) / (sub_sig.std() + 1e-9)

            tau, m = self.get_optimal_parameters(sub_norm)
            self.reference_params[band] = {'tau': tau, 'm': m}
            print(f"  Band {band} (N={len(sub_sig)}): $\\tau$={tau}, $m$={m}")

    def generate_composite_sine(self):
        t = np.arange(self.N) / self.FS
        signal = np.zeros(self.N)
        for band, freq in self.SUBBAND_FREQS.items():
            signal += np.sin(2 * np.pi * freq * t)
        return signal

    def _generate_rossler(self, seed: int = None) -> np.ndarray:
        if seed is not None: np.random.seed(seed)

        def rossler(t, xyz, a=0.2, b=0.2, c=5.7):
            x, y, z = xyz
            return [-(y + z), x + a * y, b + z * (x - c)]

        total_points = self.N + self.transient
        t_span = (0, 200);
        t_eval = np.linspace(t_span[0], t_span[1], total_points)
        ic = np.random.uniform(-1, 1, 3)
        sol = solve_ivp(rossler, t_span, ic, t_eval=t_eval, method='RK45')
        x = sol.y[0][self.transient:]
        x_resampled = resample(x, self.N)
        return (x_resampled - np.mean(x_resampled)) / (np.std(x_resampled) + 1e-9)

    def generate_lorenz_system(self, sigma=10, rho=28, beta=8 / 3, n_points=25000, discard=5000):
        def lorenz_rk4(t, state):
            x, y, z = state
            return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

        dt = 0.01;
        t_span = (0, n_points * dt);
        t_eval = np.arange(0, n_points * dt, dt)
        sol = solve_ivp(lorenz_rk4, t_span, [1.0, 1.0, 1.0], t_eval=t_eval, method='RK45')
        x = sol.y[0][discard:]
        n_samples_target = int(len(x) * self.FS / 100)
        x_resampled = resample(x, n_samples_target)
        return (x_resampled - x_resampled.mean()) / x_resampled.std()

    def add_awgn(self, signal, snr_db):
        if snr_db is None or np.isinf(snr_db): return signal
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        return signal + np.sqrt(noise_power) * np.random.randn(len(signal))

    def decompose_signal(self, signal, pipeline_name):
        if pipeline_name == 'fir':
            return self.decompose_fir(signal)
        pipeline_obj = Pipeline(pipeline_name='all', level=self.LEVEL, wavelet=self.WAVELET)
        func = getattr(pipeline_obj, f"pipeline_{pipeline_name}")
        return func(signal)

    def compute_rp(self, signal, tau, m, epsilon):
        try:
            os.environ['PYOPENCL_CTX'] = '0'
            rp_gen = RecurrencePlotGenerator(signal=signal, time_delay=tau, embedding_dim=m,
                                             path=self.OUTPUT_DIR, epsilon=epsilon, unthresholded=False)
            rqa_result, rp_result = rp_gen.compute_rqa_binary()
            return rp_result.recurrence_matrix_reverse_normalized, rqa_result
        except Exception as e:
            print(f"Warning: RP computation failed: {e}");
            return None, None

    def get_optimal_parameters(self, signal):
        """
        Compute optimal tau and m for time delay embedding.

        Uses AMI for tau and FNN for m. If AMI fails to find a minimum,
        falls back to tau = max(1, len(signal) // 100).
        """
        tau_candidate = AMI(signal=signal).get_first_minimum()

        if tau_candidate is None:
            # Fallback: use 1% of signal length or minimum of 1
            tau = max(1, len(signal) // 100)
            print(f"  Warning: AMI found no minimum, using fallback tau={tau}")
        else:
            tau = int(tau_candidate)

        m = FNN(tau=tau, signal=signal).get_optimal_dimension()
        return tau, m

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

    def visualize_decomposition_with_rps(self, system_type, pipeline_name, snr_levels=[np.inf, 20, 10, 5]):
        if not self.reference_params:
            self.prepare_reference_parameters(system_type)

        signal_clean = self._get_selected_signal(system_type)

        for snr in snr_levels:
            n_bands = len(self.BAND_NAMES)
            fig = plt.figure(figsize=(22, 4 * n_bands), constrained_layout=True)
            gs = GridSpec(n_bands, 3, figure=fig, width_ratios=[1.2, 1, 1])

            fig.suptitle(f'{system_type.upper()} | {pipeline_name.upper()} (Trimmed) | Noise: {snr}dB',
                         fontsize=20, fontweight='bold')

            signal_noisy = self.add_awgn(signal_clean, snr)
            subbands = self.decompose_signal(signal_noisy, pipeline_name)

            for idx, band in enumerate(self.BAND_NAMES):
                p = self.reference_params[band]

                # TRIM the output before computing RP
                sub_data = self._trim_signal(subbands[band])
                sub_norm = (sub_data - sub_data.mean()) / (sub_data.std() + 1e-9)

                epsilon = RecurrencePlotGenerator.find_epsilon_for_target_rr(sub_norm, p['tau'], p['m'], target_rr=0.1)

                # Signal
                ax_sig = fig.add_subplot(gs[idx, 0])
                ax_sig.plot(sub_data, color='#1f77b4')
                ax_sig.set_title(f'Subband {band} (N={len(sub_data)})', fontweight='bold')

                # UTRP
                ax_ut = fig.add_subplot(gs[idx, 1])
                ut_mat = self.get_unthresholded_matrix(sub_norm, p['tau'], p['m'])
                im_ut = ax_ut.imshow(ut_mat, cmap='magma', origin='lower')
                ax_ut.set_title(f'UTRP ($\\tau$={p["tau"]}, $m$={p["m"]})')
                plt.colorbar(im_ut, ax=ax_ut, shrink=0.7)

                # Binary RP
                ax_rp = fig.add_subplot(gs[idx, 2])
                rp_mat, _ = self.compute_rp(sub_norm, p['tau'], p['m'], epsilon)
                if rp_mat is not None:
                    ax_rp.imshow(rp_mat, cmap='binary', origin='lower')
                    ax_rp.set_title(f'Binary RP ($\\epsilon$={epsilon:.2f})')

            plt.show()


def main():
    visualizer = DecompositionVisualizer()

    print("--- System Selection ---")
    print("  1. Composite Sine\n  2. Lorenz System\n  3. Rössler System")
    sys_choice = input("Select system (1-3) [1]: ") or "1"
    system_type = {"1": "sine", "2": "lorenz", "3": "rossler"}.get(sys_choice, "sine")

    visualizer.prepare_reference_parameters(system_type)

    print("\n--- Pipeline Selection ---")
    print("  1. swt_recon\n  2. swt_direct\n  3. dwt_recon\n  4. dwt_direct\n  5. fir")
    pipe_choice = input("Select option (1-5) [1]: ") or "1"
    pipeline_name = {
        '1': 'swt_recon', '2': 'swt_direct',
        '3': 'dwt_recon', '4': 'dwt_direct', '5': 'fir'
    }.get(pipe_choice, 'swt_recon')

    snr_levels = [np.inf]
    visualizer.visualize_decomposition_with_rps(system_type, pipeline_name, snr_levels)


if __name__ == '__main__':
    main()