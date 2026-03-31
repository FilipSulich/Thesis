import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.recurrence_plot.AMI import AMI
from src.recurrence_plot.FNN import FNN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.pipeline_wavelets import Pipeline
from recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

os.environ['PYOPENCL_NO_CACHE'] = '1'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'


class DecompositionVisualizer:
    def __init__(self):
        self.FS = 128  # Sampling frequency (Hz)
        self.N = 512  # Signal length (samples)
        self.LEVEL = 4  # Wavelet decomposition level
        self.WAVELET = 'db4'

        self.SUBBAND_FREQS = {
            'A4': 2,   # 0-4 Hz
            'D4': 6,   # 4-8 Hz
            'D3': 12,  # 8-16 Hz
            'D2': 24,  # 16-32 Hz
            'D1': 48   # 32-64 Hz
        }

        self.BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']

        self.OUTPUT_DIR = '/Users/filipsulich/Developer/uni/Thesis/Thesis/outputs/experiment1_synthetic'
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def generate_composite_sine(self):
        t = np.arange(self.N) / self.FS
        signal = np.zeros(self.N)

        for band, freq in self.SUBBAND_FREQS.items():
            signal += np.sin(2 * np.pi * freq * t)
        #
        # signal += np.sin(2 * np.pi * t + np.pi/2)

        return signal

    def generate_lorenz_system(self, sigma=10, rho=28, beta=8/3, n_points=25000, discard=5000):
        """
        Generate Lorenz x-variable using 4th-order Runge-Kutta with proper resampling.

        Args:
            sigma, rho, beta: Lorenz parameters
            n_points: Total points to generate (including transient)
            discard: Points to discard as transient

        Returns:
            signal: Z-score normalized, resampled to 128 Hz
        """
        from scipy.signal import resample
        from scipy.integrate import solve_ivp

        def lorenz_rk4(t, state):
            x, y, z = state
            dx_dt = sigma * (y - x)
            dy_dt = x * (rho - z) - y
            dz_dt = x * y - beta * z
            return [dx_dt, dy_dt, dz_dt]

        dt = 0.01
        t_span = (0, n_points * dt)
        t_eval = np.arange(0, n_points * dt, dt)

        initial_state = [1.0, 1.0, 1.0]
        sol = solve_ivp(lorenz_rk4, t_span, initial_state, t_eval=t_eval, method='RK45')

        x = sol.y[0]

        # Discard transient
        x = x[discard:]

        # Resample to 128 Hz
        # Original: 100 samples/sec (dt=0.01), Target: 128 samples/sec
        n_samples_target = int(len(x) * self.FS / 100)
        x_resampled = resample(x, n_samples_target)

        # Z-score normalize
        x_resampled = (x_resampled - x_resampled.mean()) / x_resampled.std()

        return x_resampled

    def add_awgn(self, signal, snr_db):
        if snr_db is None or np.isinf(snr_db):
            return signal

        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        return signal + noise

    def decompose_signal(self, signal, pipeline_name):
        pipeline_obj = Pipeline(pipeline_name='all', level=self.LEVEL, wavelet=self.WAVELET)
        func = getattr(pipeline_obj, f"pipeline_{pipeline_name}")
        return func(signal)

    def get_analytic_embedding_params(self, freq):
        # tau = RecurrencePlotGenerator.estimate_time_delay_analytic(freq, self.FS)
        # m = RecurrencePlotGenerator.estimate_embedding_dim_analytic()
        tau = 2
        m = 2
        return tau, m

    def compute_rp(self, signal, tau, m, epsilon):
        """Compute recurrence plot for a signal."""
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
            rqa_result, rp_result = rp_gen.compute_rqa_binary()
            rp_matrix = rp_result.recurrence_matrix_reverse_normalized

            return rp_matrix, rqa_result
        except Exception as e:
            print(f"Warning: RP computation failed: {e}")
            return None, None

    def visualize_composite_with_noise(self, snr_levels=[np.inf, 20, 10, 5]):
        signal_clean = self.generate_composite_sine()
        t = np.arange(self.N) / self.FS

        n_plots = len(snr_levels)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3*n_plots))

        if n_plots == 1:
            axes = [axes]

        for idx, snr in enumerate(snr_levels):
            ax = axes[idx]

            signal_noisy = self.add_awgn(signal_clean, snr)

            ax.plot(t, signal_noisy, 'k-', linewidth=0.8, alpha=0.8)

            if np.isinf(snr):
                title = 'Composite Sine: Clean'
            else:
                title = f'Composite Sine: SNR = {snr} dB'

            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.grid(alpha=0.3)

            if idx == 0:
                ax.text(0.02, 0.98, 'Components: 2, 6, 12, 24, 48 Hz',
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        filename = 'composite_sine_noise_levels.png'
        plt.savefig(os.path.join(self.OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.show()

        return fig


    def get_optimal_parameters(self, signal):
        ami = AMI(signal=signal)
        tau = int(ami.get_first_minimum())
        fnn = FNN(tau=tau, signal=signal)
        m = fnn.get_optimal_dimension()
        return tau, m

    def visualize_single_scale_rp(self, snr_levels=[np.inf, 20, 10, 5]):
        from matplotlib.gridspec import GridSpec
        import matplotlib.pyplot as plt
        #
        signal_clean = self.generate_composite_sine()
        # signal_clean = self.generate_lorenz_system()
        #
        signal_norm = signal_clean.copy()
        if signal_norm.std() > 0:
            signal_norm = (signal_norm - signal_norm.mean()) / signal_norm.std()

        tau, m = self.get_optimal_parameters(signal_norm)
        ref_epsilon = RecurrencePlotGenerator.find_epsilon_for_target_rr(signal_norm, tau, m, target_rr=0.1)

        for snr in snr_levels:
            # 1. Increase height and use constrained_layout to prevent title/label cutoff
            fig = plt.figure(figsize=(20, 7), constrained_layout=True)
            gs = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1, 1])

            snr_label = 'Clean' if np.isinf(snr) else f'SNR {snr}dB'
            fig.suptitle(f'Single-Scale Analysis: {snr_label}', fontsize=18, fontweight='bold')

            signal_noisy = self.add_awgn(signal_clean, snr)

            signal_rp = signal_noisy.copy()
            if signal_rp.std() > 0:
                signal_rp = (signal_rp - signal_rp.mean()) / signal_rp.std()

            # --- Subplot 1: Signal Plot ---
            ax_signal = fig.add_subplot(gs[0, 0])
            t = np.arange(len(signal_noisy)) / self.FS
            ax_signal.plot(t, signal_noisy, linewidth=1.0, color='#1f77b4')
            ax_signal.set_title('Composite Time Series', fontsize=14, fontweight='bold')
            ax_signal.set_xlabel('Time (s)', fontsize=12)
            ax_signal.set_ylabel('Amplitude', fontsize=12)
            ax_signal.set_aspect('auto')  # Ensure signal is rectangular/readable
            ax_signal.grid(alpha=0.3)


            rp_matrix, rqa_result = self.compute_rp(signal_rp, tau, m, ref_epsilon)


            ax_rp = fig.add_subplot(gs[0, 1])

            if rp_matrix is not None:
                ax_rp.imshow(rp_matrix, cmap='binary', origin='upper', aspect='equal')
                det_val = rqa_result.determinism if rqa_result else 0.0
                ax_rp.set_title(f'Binary RP \n tau={tau} \n m={m}', fontsize=12,fontweight='bold')
                ax_rp.set_xlabel('Time index $i$', fontsize=12)
                ax_rp.set_ylabel('Time index $j$', fontsize=12)

            ax_utrp = fig.add_subplot(gs[0, 2])
            utrp_matrix = self.get_unthresholded_matrix(signal_rp, tau, m)

            if utrp_matrix is not None:
                im_utrp = ax_utrp.imshow(utrp_matrix, cmap='magma', origin='upper', aspect='equal')
                ax_utrp.set_title(f'UTRP \n tau={tau} \n m={m}', fontsize=12, fontweight='bold')
                ax_utrp.set_xlabel('Time index $i$', fontsize=12)
                ax_utrp.set_ylabel('Time index $j$', fontsize=12)
                ax_utrp.sharey(ax_rp)
                plt.colorbar(im_utrp, ax=ax_utrp, shrink=0.8, pad=0.05).set_label('Euclidean Distance')


            clean_snr_str = "clean" if np.isinf(snr) else f"snr_{int(snr)}"
            filename = f'single_scale_rp_viz_{clean_snr_str}.png'
            plt.savefig(os.path.join(self.OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
            plt.show()

    def visualize_decomposition_with_rps(self, pipeline_name, snr_levels=[np.inf, 20, 10, 5]):
        from matplotlib.gridspec import GridSpec
        import matplotlib.pyplot as plt

        signal_clean = self.generate_composite_sine()
        # signal_clean = self.generate_lorenz_system()

        subbands_clean = self.decompose_signal(signal_clean, pipeline_name)
        ref_params = {}
        for band in self.BAND_NAMES:
            sub_sig = subbands_clean[band]
            if sub_sig.std() > 0:
                sub_sig = (sub_sig - sub_sig.mean()) / sub_sig.std()

            tau, m = self.get_optimal_parameters(sub_sig)
            epsilon = RecurrencePlotGenerator.find_epsilon_for_target_rr(sub_sig, tau, m, target_rr=0.1)
            ref_params[band] = {'tau': tau, 'm': m, 'epsilon': epsilon}

        name_mapping = {
            "swt_recon": "SWT Reconstruction (Time Domain)",
            "dwt_recon": "DWT Reconstruction (Time Domain)",
            "dwt_direct": "DWT Coefficients (Decimated)",
            "swt_direct": "SWT Coefficients (Non-Decimated)",
        }

        for snr in snr_levels:
            n_bands = len(self.BAND_NAMES)
            # Increase figure height to accommodate all subbands (approx 4 units per band)
            fig = plt.figure(figsize=(22, 4 * n_bands), constrained_layout=True)
            # 3 Columns: Signal, UTRP, Binary RP
            gs = GridSpec(n_bands, 3, figure=fig, width_ratios=[1.2, 1, 1])

            snr_label = 'Clean' if np.isinf(snr) else f'SNR {snr}dB'
            fig.suptitle(f'{name_mapping[pipeline_name]} | Noise: {snr_label}',
                         fontsize=20, fontweight='bold')

            signal_noisy = self.add_awgn(signal_clean, snr)
            subbands = self.decompose_signal(signal_noisy, pipeline_name)

            for band_idx, band in enumerate(self.BAND_NAMES):
                params = ref_params[band]
                subband_signal = subbands[band]

                sub_norm = subband_signal.copy()
                if sub_norm.std() > 0:
                    sub_norm = (sub_norm - sub_norm.mean()) / sub_norm.std()

                ax_sig = fig.add_subplot(gs[band_idx, 0])
                t_sub = np.arange(len(subband_signal)) / self.FS
                ax_sig.plot(t_sub, subband_signal, linewidth=1.0, color='#1f77b4')
                ax_sig.set_title(f'Subband {band} ({self.SUBBAND_FREQS[band]} Hz)', fontsize=14, fontweight='bold')
                if pipeline_name == 'swt_direct' or pipeline_name == 'dwt_direct':
                    ax_sig.set_ylabel('Magnitude', fontsize=12)
                else :
                    ax_sig.set_xlabel('Amplitude', fontsize=12)
                ax_sig.grid(alpha=0.3)
                ax_sig.set_aspect('auto')
                if band_idx == n_bands - 1:
                    ax_sig.set_xlabel("Time (s)" if "recon" in pipeline_name else "Samples")

                ax_utrp = fig.add_subplot(gs[band_idx, 1])
                utrp_matrix = self.get_unthresholded_matrix(sub_norm, params['tau'], params['m'])

                if utrp_matrix is not None:
                    im_utrp = ax_utrp.imshow(utrp_matrix, cmap='magma', origin='lower', aspect='equal')
                    ax_utrp.set_title(f'UTRP \n tau={params['tau']:.1f} \n m = {params['m']:.1f}', fontsize=12, fontweight='bold')
                    ax_utrp.set_xlabel('Time index $i$', fontsize=10)
                    ax_utrp.set_ylabel('Time index $j$', fontsize=10)
                    plt.colorbar(im_utrp, ax=ax_utrp, shrink=0.7, pad=0.04)

                ax_rp = fig.add_subplot(gs[band_idx, 2])
                rp_matrix, rqa_result = self.compute_rp(sub_norm, params['tau'], params['m'], params['epsilon'])

                if rp_matrix is not None:
                    ax_rp.imshow(rp_matrix, cmap='binary', origin='lower', aspect='equal')
                    det = rqa_result.determinism if rqa_result else 0.0
                    ax_rp.set_title(f'Binary RP \n tau={params['tau']:.1f} \n m = {params['m']:.1f}', fontsize=12, fontweight='bold')
                    ax_rp.set_xlabel('Time index $i$', fontsize=10)
                    ax_rp.set_ylabel('Time index $j$', fontsize=10)
                else:
                    ax_rp.text(0.5, 0.5, 'Failed', ha='center', va='center')

            clean_snr_str = "clean" if np.isinf(snr) else f"snr_{int(snr)}"
            filename = f'decomp_full_viz_{pipeline_name}_{clean_snr_str}.png'
            plt.savefig(os.path.join(self.OUTPUT_DIR, filename), dpi=250, bbox_inches='tight')
            print(f"Generated decomposition visualization: {filename}")
            plt.show()
    def get_unthresholded_matrix(self, signal, tau, m):
        """Explicitly computes the UTRP for Frobenius comparison."""
        from pyrqa.settings import Settings
        from pyrqa.analysis_type import Cross
        from pyrqa.neighbourhood import Unthresholded
        from pyrqa.metric import EuclideanMetric
        from pyrqa.computation import RPComputation
        from pyrqa.time_series import TimeSeries

        # Ensure signal is a TimeSeries object
        ts = TimeSeries(signal, embedding_dimension=m, time_delay=tau)
        settings = Settings(ts, analysis_type=Cross,
                            neighbourhood=Unthresholded(),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=1)

        computation = RPComputation.create(settings, verbose=False)
        result = computation.run()
        # This returns the distance matrix
        return result.recurrence_matrix_reverse_normalized

def main():
    visualizer = DecompositionVisualizer()
    print("  0. single_scale - Single-Scale (No Decomposition)")
    print("  1. swt_recon    - SWT Reconstruction")
    print("  2. swt_direct   - SWT Direct Coefficients")
    print("  3. dwt_recon    - DWT Reconstruction")
    print("  4. dwt_direct   - DWT Direct Coefficients")

    pipeline_choice = input("\nSelect option (0-4) [default: 1]: ").strip() or "1"
    pipeline_map = {
        '0': 'single_scale',
        '1': 'swt_recon',
        '2': 'swt_direct',
        '3': 'dwt_recon',
        '4': 'dwt_direct'
    }
    pipeline_name = pipeline_map.get(pipeline_choice, 'swt_recon')

    print(f"\nUsing: {pipeline_name}")

    snr_input = input("\nEnter SNR levels (comma-separated, use 'inf' for clean) [default: inf,20,10,5]: ").strip()

    if snr_input:
        snr_levels = []
        for val in snr_input.split(','):
            val = val.strip()
            if val.lower() == 'inf':
                snr_levels.append(np.inf)
            else:
                try:
                    snr_levels.append(float(val))
                except:
                    print(f"Warning: Invalid SNR value '{val}', skipping")
    else:
        snr_levels = [np.inf, 20, 10, 5]


    visualizer.visualize_composite_with_noise(snr_levels)

    if pipeline_name == 'single_scale':
        visualizer.visualize_single_scale_rp(snr_levels)
    else:
        visualizer.visualize_decomposition_with_rps(pipeline_name, snr_levels)


if __name__ == '__main__':
    main()


