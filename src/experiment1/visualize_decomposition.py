import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
        #
        for band, freq in self.SUBBAND_FREQS.items():
            signal += np.sin(2 * np.pi * freq * t)
        #
        # signal += np.sin(2 * np.pi * 2 * t)

        return signal

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

    def visualize_single_scale_rp(self, snr_levels=[np.inf, 20, 10, 5]):
        signal_clean = self.generate_composite_sine()

        tau = RecurrencePlotGenerator.estimate_time_delay_analytic(12, self.FS)
        m = RecurrencePlotGenerator.estimate_embedding_dim_analytic()

        # Find reference epsilon from clean signal
        signal_norm = signal_clean.copy()
        if signal_norm.std() > 0:
            signal_norm = (signal_norm - signal_norm.mean()) / signal_norm.std()
        ref_epsilon = RecurrencePlotGenerator.find_epsilon_for_target_rr(signal_norm, tau, m, target_rr=0.1)
        print(ref_epsilon)

        for snr in snr_levels:
            fig = plt.figure(figsize=(16, 5))
            gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3, width_ratios=[1, 1.2])

            snr_label = 'Clean' if np.isinf(snr) else f'SNR {snr}dB'
            fig.suptitle(f'Single-Scale: {snr_label}',
                        fontsize=16, fontweight='bold', y=0.98)

            # Add noise
            signal_noisy = self.add_awgn(signal_clean, snr)

            # Left: Signal
            ax_signal = fig.add_subplot(gs[0, 0])
            t = np.arange(len(signal_noisy)) / self.FS
            ax_signal.plot(t, signal_noisy, linewidth=0.8, color='#1f77b4')
            ax_signal.set_title('Composite Signal', fontsize=12, fontweight='bold')
            ax_signal.set_xlabel('Time (s)', fontsize=10)
            ax_signal.set_ylabel('Amplitude', fontsize=10)
            ax_signal.grid(alpha=0.3)

            # Right: RP
            ax_rp = fig.add_subplot(gs[0, 1])

            # Normalize for RP
            signal_rp = signal_noisy.copy()
            if signal_rp.std() > 0:
                signal_rp = (signal_rp - signal_rp.mean()) / signal_rp.std()

            # Compute RP
            rp_matrix, rqa_result = self.compute_rp(signal_rp, tau, m, ref_epsilon)

            if rp_matrix is not None:
                ax_rp.imshow(rp_matrix, cmap='binary', origin='upper', aspect='equal')
                if rqa_result is not None:
                    ax_rp.set_title(f'RP: τ={tau}, m={m}, DET={rqa_result.determinism:.3f}', fontsize=12)
                else:
                    ax_rp.set_title(f'RP: τ={tau}, m={m}', fontsize=12)
                ax_rp.set_xlabel('Time i', fontsize=10)
                ax_rp.set_ylabel('Time j', fontsize=10)
            else:
                ax_rp.text(0.5, 0.5, 'RP Failed', ha='center', va='center')
                ax_rp.set_title(f'RP: τ={tau}, m={m}', fontsize=12)

            ax_rp.set_xticks([])
            ax_rp.set_yticks([])

            # Save
            clean_snr_str = "clean" if np.isinf(snr) else f"snr_{int(snr)}"
            filename = f'single_scale_rp_{clean_snr_str}.png'
            plt.savefig(os.path.join(self.OUTPUT_DIR, filename), dpi=200, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.show()

    def visualize_decomposition_with_rps(self, pipeline_name, snr_levels=[np.inf, 20, 10, 5]):
        signal_clean = self.generate_composite_sine()

        # 1. Pre-calculate reference epsilons
        subbands_clean = self.decompose_signal(signal_clean, pipeline_name)
        ref_epsilons = {}
        for band in self.BAND_NAMES:
            freq = self.SUBBAND_FREQS[band]
            tau, m = self.get_analytic_embedding_params(freq)
            subband_signal = subbands_clean[band]
            if subband_signal.std() > 0:
                subband_signal = (subband_signal - subband_signal.mean()) / subband_signal.std()
            ref_epsilons[band] = RecurrencePlotGenerator.find_epsilon_for_target_rr(subband_signal, tau, m,
                                                                                    target_rr=0.1)

        name_mapping = {
            "swt_recon": "SWT Time Domain Reconstruction",
            "dwt_recon": "DWT Time Domain Reconstruction",
            "dwt_direct": "DWT Wavelet Coefficients",
            "swt_direct": "SWT Wavelet Coefficients",
        }

        for snr in snr_levels:
            n_bands = len(self.BAND_NAMES)

            # --- Layout Tuning ---
            # We use a wider figure and specific width_ratios to give the RP square space
            fig = plt.figure(figsize=(16, 4 * n_bands))
            gs = GridSpec(n_bands, 2, figure=fig, hspace=0.3, wspace=0.2, width_ratios=[1, 1])

            snr_label = 'Clean' if np.isinf(snr) else f'SNR {snr}dB'
            # fig.suptitle(f'{name_mapping[pipeline_name]} | Noise: {snr_label}',fontsize=18, fontweight='bold', y=0.94)
            fig.suptitle(f'{name_mapping[pipeline_name]}',fontsize=18, fontweight='bold', y=0.94)

            signal_noisy = self.add_awgn(signal_clean, snr)
            subbands = self.decompose_signal(signal_noisy, pipeline_name)

            for band_idx, band in enumerate(self.BAND_NAMES):
                # --- Left: Decomposition Signal ---
                ax_decomp = fig.add_subplot(gs[band_idx, 0])
                subband_signal = subbands[band]
                t_sub = np.arange(len(subband_signal)) / self.FS

                ax_decomp.plot(t_sub, subband_signal, linewidth=0.8, color='#1f77b4')
                ax_decomp.set_title(f'Subband {band} ({self.SUBBAND_FREQS[band]} Hz)', fontsize=12, fontweight='bold')
                ax_decomp.set_ylabel('Amplitude', fontsize=10)
                ax_decomp.grid(alpha=0.3)

                if band_idx == n_bands - 1:
                    ax_decomp.set_xlabel("Time (s)" if "recon" in pipeline_name else "Samples", fontsize=10)

                # --- Right: Recurrence Plot (Matching PAFVisualizer style) ---
                ax_rp = fig.add_subplot(gs[band_idx, 1])
                freq = self.SUBBAND_FREQS[band]
                tau, m = self.get_analytic_embedding_params(freq)
                epsilon = ref_epsilons[band]

                subband_rp = subband_signal.copy()
                if subband_rp.std() > 0:
                    subband_rp = (subband_rp - subband_rp.mean()) / subband_rp.std()

                rp_matrix, rqa_result = self.compute_rp(subband_rp, tau, m, epsilon)

                if rp_matrix is not None:
                    # Using origin='lower' and aspect='equal' as requested
                    im = ax_rp.imshow(rp_matrix, cmap='binary', origin='upper', aspect='equal', interpolation='nearest')

                    det_str = f" | DET={rqa_result.determinism:.3f}" if rqa_result else ""
                    ax_rp.set_title(f"RP (τ={tau}, m={m}){det_str}", fontsize=12)

                    # Restore Axis Labels
                    ax_rp.set_xlabel("Time index $i$", fontsize=10)
                    ax_rp.set_ylabel("Time index $j$", fontsize=10)
                    ax_rp.tick_params(labelsize=9)
                else:
                    ax_rp.text(0.5, 0.5, 'RP Computation Failed', ha='center', va='center')

            # Tighten the overall figure margins
            plt.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.08)

            clean_snr_str = "clean" if np.isinf(snr) else f"snr_{int(snr)}"
            filename = f'decomp_rp_{pipeline_name}_{clean_snr_str}.png'
            plt.savefig(os.path.join(self.OUTPUT_DIR, filename), dpi=200, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.show()


def main():
    """Interactive demonstration."""

    visualizer = DecompositionVisualizer()

    print("="*60)
    print("DECOMPOSITION & RP VISUALIZER")
    print("="*60)

    # Select pipeline
    print("\nAvailable options:")
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

    # SNR levels
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
