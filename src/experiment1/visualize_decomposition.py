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
from src.pipelines.pipeline_wavelets import Pipeline
from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

os.environ['PYOPENCL_NO_CACHE'] = '1'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'


class DecompositionVisualizer:
    def __init__(self, seed=None):
        self.FS = 128
        self.N = 1024
        self.TRIM = 0
        self.LEVEL = 4
        self.WAVELET = 'db4'
        self.transient = 2000
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # self.SUBBAND_FREQS = {
        #     'A4': 2, 'D4': 6, 'D3': 12, 'D2': 24, 'D1': 48
        # }
        self.SUBBAND_FREQS = {
            'A4': 1,
            'D4': 5,  # 4-8 Hz
            'D3': 13,  # 8-16 Hz
            'D2': 23,  # 16-32 Hz
            'D1': 47  # 32-64 Hz
        }
        self.BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']
        self.OUTPUT_DIR = '/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_new/exp1'
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        self.reference_params = {}

    def _trim_signal(self, signal):
        if self.TRIM == 0:
            return signal
        if len(signal) <= 2 * self.TRIM:
            print(f"Warning: Signal length ({len(signal)}) is too short to trim {self.TRIM} from both sides.")
            return signal
        return signal[self.TRIM: -self.TRIM]

    def _get_selected_signal(self, system_type):
        return self.generate_composite_sine()

    def generate_composite_sine(self):
        t = np.arange(self.N) / self.FS
        signal = np.zeros(self.N)
        for band, freq in self.SUBBAND_FREQS.items():
            signal += np.sin(2 * np.pi * freq * t)

        # signal += np.sin(2 * np.pi *  47* t)
        return signal

    def add_awgn(self, signal, snr_db):
        if snr_db is None or np.isinf(snr_db): return signal
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        return signal + np.sqrt(noise_power) * np.random.randn(len(signal))

    def decompose_signal(self, signal, pipeline_name):
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
        # 1. Handle extremely short signals (common in DWT direct)
        N = len(signal)
        if N < 10:  # Minimal threshold
            return 1, 1

        tau_candidate = AMI(signal=signal).get_first_minimum()

        if tau_candidate is None:
            tau = max(1, N // 100)
        else:
            tau = int(tau_candidate)

        try:
            max_m = max(2, N // (tau if tau > 0 else 1))
            m_obj = FNN(tau=tau, signal=signal)
            m = m_obj.get_optimal_dimension()

            if m >= max_m:
                m = max(1, max_m - 1)
        except Exception as e:
            print(f"  Warning: FNN failed ({e}), using fallback m=2")
            m = 2

        return tau, m

    def get_unthresholded_matrix(self, signal, tau, m):
        from pyrqa.settings import Settings
        from pyrqa.analysis_type import Cross
        from pyrqa.neighbourhood import Unthresholded
        from pyrqa.metric import EuclideanMetric
        from pyrqa.computation import RPComputation
        from pyrqa.time_series import TimeSeries

        ts = TimeSeries(signal, embedding_dimension=m, time_delay=tau)

        settings = Settings(ts, analysis_type=Cross, neighbourhood=Unthresholded(),similarity_measure=EuclideanMetric, theiler_corrector=1)

        return RPComputation.create(settings, verbose=False).run().recurrence_matrix_reverse_normalized

    def visualize_decomposition_with_rps(self, system_type, pipeline_name, snr_levels=[np.inf, 20, 10, 5]):
        signal_clean = self._get_selected_signal(system_type)

        for snr in snr_levels:
            n_bands = len(self.BAND_NAMES)
            fig = plt.figure(figsize=(22, 4 * n_bands), constrained_layout=True)
            gs = GridSpec(n_bands, 3, figure=fig, width_ratios=[1.2, 1, 1])

            name_mapping = {
                "swt_recon": "SWT-R",
                "swt_direct": "SWT-D",
                "dwt_recon": "DWT-R",
                "dwt_direct": "DWT-D",
            }

            fig.suptitle(f'{system_type.upper()} test | {name_mapping[pipeline_name]}', fontsize=20, fontweight='bold')

            signal_noisy = self.add_awgn(signal_clean, snr)
            subbands = self.decompose_signal(signal_noisy, pipeline_name)

            for idx, band in enumerate(self.BAND_NAMES):
                sub_data = subbands[band]

                time_vector = np.arange(len(sub_data)) / self.FS

                sub_norm = (sub_data - sub_data.mean()) / (sub_data.std() + 1e-9)
                tau, m = self.get_optimal_parameters(sub_norm)
                epsilon = RecurrencePlotGenerator.find_epsilon_for_target_rr(sub_norm, tau, m, target_rr=0.1)

                ax_sig = fig.add_subplot(gs[idx, 0])
                if pipeline_name == "swt_recon" or pipeline_name == "swt_recon":
                    ax_sig.plot(time_vector, sub_data, color='#1f77b4')  # Pass time_vector here
                    ax_sig.set_xlabel('Time (s)')
                    ax_sig.set_ylabel('Amplitude')
                else :
                    ax_sig.plot(sub_data, color='#1f77b4')  # Pass time_vector here
                    ax_sig.set_xlabel('Sample')
                    ax_sig.set_ylabel('Magnitude')

                ax_sig.set_title(f'Subband {band}', fontweight='bold')
                # UTRP
                ax_ut = fig.add_subplot(gs[idx, 1])
                ut_mat = self.get_unthresholded_matrix(sub_norm, tau, m)
                im_ut = ax_ut.imshow(ut_mat, cmap='magma', origin='upper')
                ax_ut.set_title(f'UTRP ($\\tau$={tau}, $m$={m})')
                plt.colorbar(im_ut, ax=ax_ut, shrink=0.7)

                # Binary RP
                ax_rp = fig.add_subplot(gs[idx, 2])
                rp_mat, _ = self.compute_rp(sub_norm, tau, m, epsilon)
                if rp_mat is not None:
                    ax_rp.imshow(rp_mat, cmap='binary', origin='upper')
                    ax_rp.set_title(f'Binary RP ($\\epsilon$={epsilon:.2f})')

            plt.show()

    def visualize_single_band(self, system_type, pipeline_name, band_name, snr=np.inf):
        signal_clean = self._get_selected_signal(system_type)
        signal_noisy = self.add_awgn(signal_clean, snr)
        subbands = self.decompose_signal(signal_noisy, pipeline_name)

        if band_name not in subbands:
            print(f"Error: Band {band_name} not found in decomposition. Available: {list(subbands.keys())}")
            return

        sub_data = subbands[band_name]
        N_sub = len(sub_data)

        if pipeline_name == 'dwt_direct':
            level = int(band_name[1])
            effective_fs = self.FS / (2 ** level)
        else:
            effective_fs = self.FS

        time_vector = np.arange(N_sub) / effective_fs
        sub_norm = (sub_data - sub_data.mean()) / (sub_data.std() + 1e-9)

        # Get embedding parameters with previous safety checks
        tau, m = self.get_optimal_parameters(sub_norm)

        try:
            epsilon = RecurrencePlotGenerator.find_epsilon_for_target_rr(sub_norm, tau, m, target_rr=0.1)
        except Exception:
            epsilon = 0.5

            # Create the 1x3 plot
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
        fig.suptitle(f'{system_type.upper()} | {pipeline_name.upper()} | Band: {band_name} (SNR: {snr}dB)',
                     fontsize=16, fontweight='bold')

        # 1. Signal Plot
        axes[0].plot(time_vector, sub_data, color='#1f77b4')
        axes[0].set_title(f'Time Domain Signal (N={N_sub})', fontweight='bold')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')

        # 2. UTRP Plot
        ut_mat = self.get_unthresholded_matrix(sub_norm, tau, m)
        im_ut = axes[1].imshow(ut_mat, cmap='magma', origin='lower')
        axes[1].set_title(f'UTRP ($\\tau$={tau}, $m$={m})')
        plt.colorbar(im_ut, ax=axes[1], shrink=0.8)

        # 3. Binary RP Plot
        rp_mat, _ = self.compute_rp(sub_norm, tau, m, epsilon)
        if rp_mat is not None:
            axes[2].imshow(rp_mat, cmap='binary', origin='lower')
            axes[2].set_title(f'Binary RP ($\\epsilon$={epsilon:.2f})')

        plt.show()

    def visualize_raw_signal(self, system_type, snr=np.inf):
        signal_clean = self._get_selected_signal(system_type)

        signal_noisy = self.add_awgn(signal_clean, snr)

        sub_norm = (signal_noisy - signal_noisy.mean()) / (signal_noisy.std() + 1e-9)

        tau, m = self.get_optimal_parameters(sub_norm)
        time_vector = np.arange(len(signal_noisy)) / self.FS

        try:
            epsilon = RecurrencePlotGenerator.find_epsilon_for_target_rr(sub_norm, tau, m, target_rr=0.1)
        except Exception:
            epsilon = 0.5

        fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
        fig.suptitle(f'Single scale {system_type}',fontsize=18, fontweight='bold')

        # --- Time Domain Plot ---
        axes[0].plot(time_vector, signal_noisy, color='#1f77b4')
        axes[0].set_title(f'Full Signal', fontweight='bold')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')

        # --- UTRP Plot ---
        ut_mat = self.get_unthresholded_matrix(sub_norm, tau, m)
        im_ut = axes[1].imshow(ut_mat, cmap='magma', origin='lower')
        axes[1].set_title(f'UTRP ($\\tau$={tau}, $m$={m})')
        plt.colorbar(im_ut, ax=axes[1], shrink=0.8)

        # --- Binary RP Plot ---
        rp_mat, _ = self.compute_rp(sub_norm, tau, m, epsilon)
        if rp_mat is not None:
            axes[2].imshow(rp_mat, cmap='binary', origin='lower')
            axes[2].set_title(f'Binary RP ($\\epsilon$={epsilon:.2f})')

        plt.show()

def main():
    visualizer = DecompositionVisualizer(seed=43)

    print("--- System Selection ---")
    print("  1. Composite Sine\n  2. Lorenz System\n  3. Rössler System")
    sys_choice = input("Select system (1-3) [1]: ") or "1"
    system_type = {"1": "sine", "2": "lorenz", "3": "rossler"}.get(sys_choice, "sine")

    print("\n--- Visualization Mode ---")
    print("  1. Raw Signal (No Decomposition)")
    print("  2. Full Decomposition (All Bands)")
    print("  3. Single Band Focus (From Decomposition)")
    mode_choice = input("Select mode (1-3) [1]: ") or "1"

    if mode_choice == "1":
        snr_val = input("Enter SNR (dB) [inf]: ")
        snr = float(snr_val) if snr_val and snr_val != "inf" else np.inf
        visualizer.visualize_raw_signal(system_type, snr)

    elif mode_choice == "2":
        print("\nSelect Pipeline: 1.swt_recon, 2.swt_direct, 3.dwt_recon, 4.dwt_direct")
        pipe_choice = input("Choice [1]: ") or "1"
        pipeline_name = {'1':'swt_recon','2':'swt_direct','3':'dwt_recon','4':'dwt_direct'}.get(pipe_choice, 'swt_recon')
        visualizer.visualize_decomposition_with_rps(system_type, pipeline_name, [np.inf])

    elif mode_choice == "3":
        print("\nSelect Pipeline: 1.swt_recon, 2.swt_direct, 3.dwt_recon, 4.dwt_direct")
        pipe_choice = input("Choice [1]: ") or "1"
        pipeline_name = {'1':'swt_recon','2':'swt_direct','3':'dwt_recon','4':'dwt_direct'}.get(pipe_choice, 'swt_recon')
        band_choice = input(f"Enter band (e.g., D3) [D1]: ") or "D1"
        visualizer.visualize_single_band(system_type, pipeline_name, band_choice)

if __name__ == '__main__':
    main()
