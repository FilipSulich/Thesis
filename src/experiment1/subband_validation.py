import os
import sys
import numpy as np
import pandas as pd
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.pipeline_wavelets import Pipeline
from recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

class SubbandValidator:
    def __init__(self):
        self.FS = 128
        self.N = 512
        self.LEVEL = 4
        self.WAVELET = 'db4'

        self.SUBBAND_FREQS = {
            'A4': 2,   #D1 0-4 Hz
            'D4': 6,   #D2 4-8 Hz
            'D3': 12,  #D3 8-16 Hz
            'D2': 24,  #D4 16-32 Hz
            'D1': 48   #A4 32-64 Hz
        }

        self.BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']
        self.PIPELINES = ['swt_recon', 'swt_direct', 'dwt_recon', 'dwt_direct']
        self.OUTPUT_DIR = '/Users/filipsulich/Developer/uni/Thesis/Thesis/outputs/experiment1_synthetic'
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

            tau, m = self.get_analytic_embedding_params(freq)

            epsilon = RecurrencePlotGenerator.find_epsilon_for_target_rr(
                pure_sine, tau, m, target_rr=0.1
            )

            metrics = self.compute_rp_metrics(pure_sine, tau, m, epsilon)

            reference_values[band] = {
                'freq_hz': freq,
                'tau': tau,
                'm': m,
                'epsilon': epsilon,
                'DET': float(metrics['DET']),
                'ENTR': float(metrics['ENTR']),
                'LAM': float(metrics['LAM']),
                'RR': float(metrics['RR']),
                'L_mean_diagonal': float(metrics['L_mean_diagonal']),
                'L_longest_diagonal': float(metrics['L_longest_diagonal']),
                'L_longest_vertical': float(metrics['L_longest_vertical']),
                'TT': float(metrics['TT']),
            }

            reference_data[band] = {
                'rp_matrix': metrics['rp_matrix'],
                'pure_sine': pure_sine
            }

            print(f"    DET: {metrics['DET']:.4f}, ENTR: {metrics['ENTR']:.4f}, "
                  f"LAM: {metrics['LAM']:.4f}, TT: {metrics['TT']:.4f}")

        with open(save_path, 'w') as f:
            json.dump(reference_values, f, indent=2)

        ref_data_path = save_path.replace('.json', '_data.npz')
        np.savez(ref_data_path, **{
            f'{band}_rp': reference_data[band]['rp_matrix']
            for band in self.BAND_NAMES
        }, **{
            f'{band}_signal': reference_data[band]['pure_sine']
            for band in self.BAND_NAMES
        })

        return reference_values

    def load_reference_rqa_values(self, load_path=None):
        if load_path is None:
            load_path = os.path.join(self.OUTPUT_DIR, 'reference_rqa_values.json')

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Reference values file not found: {load_path}")

        with open(load_path, 'r') as f:
            reference_values = json.load(f)

        return reference_values

    def load_reference_data(self, load_path=None):
        """
        Load reference RP matrices and pure sine signals.

        Args:
            load_path: Path to reference data NPZ file. If None, uses OUTPUT_DIR.

        Returns:
            Dictionary with reference RP matrices and pure sine signals for each band.
        """
        if load_path is None:
            load_path = os.path.join(self.OUTPUT_DIR, 'reference_rqa_values_data.npz')

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Reference data file not found: {load_path}")

        data = np.load(load_path)
        reference_data = {}

        for band in self.BAND_NAMES:
            reference_data[band] = {
                'rp_matrix': data[f'{band}_rp'],
                'pure_sine': data[f'{band}_signal']
            }

        return reference_data

    def get_analytic_embedding_params(self, freq):
        tau = RecurrencePlotGenerator.estimate_time_delay_analytic(freq, self.FS)
        m = RecurrencePlotGenerator.estimate_embedding_dim_analytic()
        return tau, m

    def decompose_signal(self, signal, pipeline_name):
        pipeline_obj = Pipeline(pipeline_name='all', level=self.LEVEL, wavelet=self.WAVELET)
        func = getattr(pipeline_obj, f"pipeline_{pipeline_name}")
        return func(signal)

    def compute_rp_metrics(self, signal, tau, m, epsilon):
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

            longest_vertical_line  = rqa_result.longest_vertical_line
            if np.isnan(longest_vertical_line): longest_vertical_line = 0

            det = rqa_result.determinism
            if np.isnan(det): det = 0.0

            lam = rqa_result.laminarity
            if np.isnan(lam): lam = 0.0

            return {
                'DET': det,
                'ENTR': rqa_result.entropy_diagonal_lines,
                'LAM': lam,
                'L_mean_diagonal': l_mean_diag,
                'L_longest_diagonal': longest_diagonal,
                'L_longest_vertical': longest_vertical_line,
                'TT': tt,
                'RR': rqa_result.recurrence_rate,
                'rp_matrix': rp_matrix
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
                'RR': 0.0,
                'rp_matrix': None
            }

    def validate_pipeline(self, signal, pipeline_name, reference_epsilons=None, reference_rqa=None, reference_data=None):

        subbands = self.decompose_signal(signal, pipeline_name)

        results = []
        epsilons_used = {}


        for band in self.BAND_NAMES:
            freq = self.SUBBAND_FREQS[band]
            tau, m = self.get_analytic_embedding_params(freq)

            expected_period = self.FS / freq

            if len(subbands) == 1:
                subband_signal = subbands[0]
            else:
             subband_signal = subbands[band]

            if subband_signal.std() > 0:
                subband_signal = (subband_signal - subband_signal.mean()) / subband_signal.std()

            if reference_epsilons is not None and band in reference_epsilons:
                epsilon = reference_epsilons[band]
            else:
                epsilon = RecurrencePlotGenerator.find_epsilon_for_target_rr(subband_signal, tau, m, target_rr=0.1)
                epsilons_used[band] = epsilon

            metrics = self.compute_rp_metrics(subband_signal, tau, m, epsilon)

            if reference_data is not None and band in reference_data:
                ref_data = reference_data[band]

            if reference_rqa is not None and band in reference_rqa:
                ref = reference_rqa[band]
                det_deviation = abs(ref['DET'] - metrics['DET'])
                entr_deviation = abs(ref['ENTR'] - metrics['ENTR'])
                lam_deviation = abs(ref['LAM'] - metrics['LAM'])
                tt_deviation = abs(ref['TT'] - metrics['TT'])
                longest_diagonal_line_deviation = abs(ref['L_longest_diagonal'] - metrics['L_longest_diagonal']) / len(subband_signal)
                longest_vertical_line_deviation = abs(ref['L_longest_vertical'] - metrics['L_longest_vertical']) / len(subband_signal)
            else:
                det_deviation = abs(1.0 - metrics['DET'])
                entr_deviation = metrics['ENTR']
                lam_deviation = metrics['LAM']
                tt_deviation = metrics['TT']
                longest_diagonal_line_deviation = abs(1 -  metrics['L_longest_diagonal'] / len(subband_signal)) #should approx the length of the signal
                longest_vertical_line_deviation = metrics['L_longest_vertical'] / len(subband_signal)#should be 0

            results.append({
                'pipeline': pipeline_name,
                'band': band,
                'freq_hz': freq,
                'tau': tau,
                'm': m,
                'epsilon': epsilon,

                'DET': metrics['DET'],
                'ENTR': metrics['ENTR'],
                'LAM': metrics['LAM'],
                'RR': metrics['RR'],

                'L_mean_diagonal': metrics['L_mean_diagonal'],
                'L_longest_diagonal': metrics['L_longest_diagonal'],
                'L_longest_vertical': metrics['L_longest_diagonal'],
                'TT': metrics['TT'],

                'det_deviation': det_deviation,
                'entr_deviation': entr_deviation,
                'lam_deviation': lam_deviation,
                'tt_deviation' : tt_deviation,
                'longest_diagonal_deviation': longest_diagonal_line_deviation,
                'longest_vertical_deviation': longest_vertical_line_deviation,
            })

        if reference_epsilons is None:
            return pd.DataFrame(results), epsilons_used
        else:
            return pd.DataFrame(results), None

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

        metric_cols = ['det_deviation', 'entr_deviation', 'lam_deviation', 'tt_deviation','longest_diagonal_deviation', 'longest_vertical_deviation']

        results_mean = combined.groupby('band')[metric_cols].mean().reset_index()
        results_std = combined.groupby('band')[metric_cols].std().reset_index()

        results_mean['pipeline'] = pipeline_name
        results_mean['snr_db'] = snr_db
        results_std['pipeline'] = pipeline_name
        results_std['snr_db'] = snr_db

        return results_mean, results_std

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

        for snr_db in [20, 10, 5, 1]:
            for pipeline in self.PIPELINES:
                mean_df, std_df = self.validate_with_noise(
                    signal, pipeline, reference_epsilons, snr_db, n_realizations=50, reference_rqa=reference_rqa, reference_data=reference_data
                )

                noise_results[f"{pipeline}_snr{snr_db}_mean"] = mean_df
                noise_results[f"{pipeline}_snr{snr_db}_std"] = std_df

        return clean_results, noise_results, reference_epsilons


def main():
    validator = SubbandValidator()

    clean_results, noise_results, ref_eps = validator.run_full_validation()

    results_file = os.path.join(validator.OUTPUT_DIR, 'subband_validation_results.csv')
    all_results = pd.concat(clean_results.values(), ignore_index=True)
    all_results.to_csv(results_file, index=False)
    print("done")

def run_statistical_comparison(n_trials=100, use_reference_rqa=True):
    from scipy.stats import ttest_ind

    validator = SubbandValidator()

    # Generate or load reference RQA values and data
    reference_rqa = None
    reference_data = None

    if use_reference_rqa:
        ref_path = os.path.join(validator.OUTPUT_DIR, 'reference_rqa_values.json')
        ref_data_path = os.path.join(validator.OUTPUT_DIR, 'reference_rqa_values_data.npz')

        if os.path.exists(ref_path):
            print("Loading existing reference RQA values...")
            reference_rqa = validator.load_reference_rqa_values(ref_path)
            if os.path.exists(ref_data_path):
                reference_data = validator.load_reference_data(ref_data_path)
        else:
            print("Generating reference RQA values from pure sinusoids...")
            reference_rqa = validator.generate_reference_rqa_values(ref_path)
            reference_data = validator.load_reference_data(ref_data_path)
        print()

    results = {
        'swt_recon': {'DET': [], 'LAM': [], 'ENTR': [], "TT": []},
        'swt_direct':  {'DET': [], 'LAM': [], 'ENTR': [], "TT": []},
        'dwt_recon':  {'DET': [], 'LAM': [], 'ENTR': [], "TT": []},
        'dwt_direct':  {'DET': [], 'LAM': [], 'ENTR': [], "TT": []},
    }

    pipelines = ['swt_recon', 'swt_direct', 'dwt_recon', 'dwt_direct']

    for trial in range(n_trials):
        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{n_trials}...")

        signal = validator.generate_composite_sine(random_phase=True)

        for pipeline in pipelines:
            df, _ = validator.validate_pipeline(signal, pipeline, reference_rqa=reference_rqa, reference_data=reference_data)

            avg_det = df['det_deviation'].mean()
            avg_lam = df['lam_deviation'].mean()
            avg_entr = df['entr_deviation'].mean()
            avg_tt = df['tt_deviation'].mean()
            # avg_hamming = df['hamming_distance'].mean()
            # avg_crp_det = df['crp_det_deviation'].mean()

            results[pipeline]['DET'].append(avg_det)
            results[pipeline]['LAM'].append(avg_lam)
            results[pipeline]['ENTR'].append(avg_entr)
            results[pipeline]['TT'].append(avg_tt)
            # results[pipeline]['HAMMING'].append(avg_hamming)
            # results[pipeline]['CRP_DET'].append(avg_crp_det)

    for pipeline in pipelines:
        for metric in ['DET', 'LAM', 'ENTR', 'TT']:
            results[pipeline][metric] = np.array(results[pipeline][metric])

    comparisons = [
        ('swt_recon', 'swt_direct', 'SWT Recon vs SWT Direct'),
        ('swt_recon', 'dwt_recon', 'SWT Recon vs DWT Recon'),
        ('swt_direct', 'dwt_recon', 'SWT Direct vs DWT Recon'),
        ('dwt_direct', 'dwt_recon', 'DWT Direct vs DWT Recon'),
        ('dwt_direct', 'swt_recon', 'DWT Direct vs SWT Recon'),
        ('dwt_direct', 'swt_direct', 'DWT Direct vs SWT Direct'),

    ]

    stats_results = []

    for pipe1, pipe2, label in comparisons:
        for metric in ['DET', 'LAM', 'ENTR', 'TT']:
            data1 = results[pipe1][metric]
            data2 = results[pipe2][metric]

            if data1.std() < 1e-10 and data2.std() < 1e-10:
                continue

            t_stat, p_value = ttest_ind(data1, data2, equal_var=False)

            pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
            if pooled_std < 1e-10:
                cohens_d = 0.0
            else:
                cohens_d = (data1.mean() - data2.mean()) / pooled_std

            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = 'ns'

            print(f"  {metric}:")
            print(f"    {pipe1}: {data1.mean():.6f} ± {data1.std():.6f}")
            print(f"    {pipe2}: {data2.mean():.6f} ± {data2.std():.6f}")
            print(f"    t = {t_stat:.4f}, p = {p_value:.6f} {sig}")
            print(f"    Cohen's d = {cohens_d:.4f}")

            stats_results.append({
                'comparison': label,
                'metric': metric,
                'pipeline1': pipe1,
                'pipeline2': pipe2,
                'mean1': data1.mean(),
                'std1': data1.std(),
                'mean2': data2.mean(),
                'std2': data2.std(),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significance': sig
            })

    stats_df = pd.DataFrame(stats_results)
    stats_file = os.path.join(validator.OUTPUT_DIR, 'welch_ttest_results.csv')
    stats_df.to_csv(stats_file, index=False)
    # visualize_statistical_distributions(results, validator.OUTPUT_DIR)
    visualize_bar_comparison(results, validator.OUTPUT_DIR)
    visualize_snr_degradation(validator, validator.OUTPUT_DIR, n_realizations=30, use_reference_rqa=use_reference_rqa)

    return results, stats_df


def visualize_bar_comparison(results, output_dir):
    """Create bar charts comparing pipelines across metrics with error bars."""
    import matplotlib.pyplot as plt

    pipelines = ['swt_recon', 'swt_direct', 'dwt_recon', 'dwt_direct']
    metrics = ['DET', 'LAM', 'ENTR', 'TT']

    colors = {
        'swt_recon': '#2ecc71',
        'swt_direct': '#3498db',
        'dwt_recon': '#e74c3c',
        'dwt_direct': '#34495e',
    }

    labels = {
        'swt_recon': 'SWT Recon',
        'swt_direct': 'SWT Direct',
        'dwt_recon': 'DWT Recon',
        'dwt_direct': 'DWT Direct'
    }

    metrics_mapping = {
        'DET': 'Determinism',
        'ENTR': 'Entropy',
        'TT': 'Trapping Time',
        'LAM': 'Laminarity'
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        means = [results[p][metric].mean() for p in pipelines]
        stds = [results[p][metric].std() for p in pipelines]

        x_pos = np.arange(len(pipelines))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=[colors[p] for p in pipelines],
                     edgecolor='black', linewidth=1.5, alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([labels[p] for p in pipelines], fontsize=11)
        ax.set_ylabel(f'{metric} Deviation', fontsize=12, fontweight='bold')
        ax.set_title(f'{metrics_mapping[metric]} Deviation Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.005, f'{mean:.4f}±{std:.4f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    filename = os.path.join(output_dir, 'bar_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved bar comparison: {filename}")
    plt.show()


def visualize_statistical_distributions(results, output_dir):
    """Create histogram distributions for each metric."""
    import matplotlib.pyplot as plt

    pipelines = ['swt_recon', 'swt_direct', 'dwt_recon', 'dwt_direct']
    metrics = ['DET', 'LAM', 'ENTR', 'TT']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = {
        'swt_recon': '#2ecc71',
        'swt_direct': '#3498db',
        'dwt_recon': '#e74c3c',
        'dwt_direct': '#34495e',
    }

    labels = {
        'swt_recon': 'SWT Recon',
        'swt_direct': 'SWT Direct',
        'dwt_recon': 'DWT Recon',
        'dwt_direct': 'DWT Direct'
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for pipeline in pipelines:
            data = results[pipeline][metric]
            ax.hist(data, bins=15, alpha=0.5, color=colors[pipeline],
                   label=labels[pipeline], edgecolor='black')

        ax.set_xlabel(f'{metric} Deviation', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{metric} Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    filename = os.path.join(output_dir, 'statistical_distributions.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved distribution plot: {filename}")
    plt.show()


def visualize_snr_degradation(validator, output_dir, n_realizations=50, use_reference_rqa=True):
    import matplotlib.pyplot as plt

    # Load reference RQA values
    reference_rqa = None
    if use_reference_rqa:
        ref_path = os.path.join(validator.OUTPUT_DIR, 'reference_rqa_values.json')
        if os.path.exists(ref_path):
            reference_rqa = validator.load_reference_rqa_values(ref_path)

    signal_clean = validator.generate_composite_sine()
    snr_levels = [20, 15, 10, 5, 1, 0]
    metrics = ['det_deviation', 'lam_deviation', 'entr_deviation', 'tt_deviation']

    pipelines = ['swt_recon', 'swt_direct', 'dwt_recon', 'dwt_direct']

    snr_results = {p: {m: {'mean': [], 'std': []} for m in metrics} for p in pipelines}

    ref_epsilons = {}
    for pipeline in pipelines:
        subbands_clean = validator.decompose_signal(signal_clean, pipeline)
        ref_epsilons[pipeline] = {}

        for band in validator.BAND_NAMES:
            freq = validator.SUBBAND_FREQS[band]
            tau, m = validator.get_analytic_embedding_params(freq)
            subband_signal = subbands_clean[band]

            if subband_signal.std() > 0:
                subband_signal = (subband_signal - subband_signal.mean()) / subband_signal.std()

            ref_epsilons[pipeline][band] = RecurrencePlotGenerator.find_epsilon_for_target_rr(
                subband_signal, tau, m, target_rr=0.1)

    for snr_db in snr_levels:
        for pipeline in pipelines:
            mean_df, std_df = validator.validate_with_noise(
                signal_clean, pipeline, ref_epsilons[pipeline], snr_db, n_realizations, reference_rqa)

            for metric in metrics:
                snr_results[pipeline][metric]['mean'].append(mean_df[metric].mean())
                snr_results[pipeline][metric]['std'].append(std_df[metric].mean())

    colors = {
        'swt_recon': '#2ecc71',
        'swt_direct': '#3498db',
        'dwt_recon': '#e74c3c',
        'dwt_direct': '#34495e',
    }

    labels = {
        'swt_recon': 'SWT Recon',
        'swt_direct': 'SWT Direct',
        'dwt_recon': 'DWT Recon',
        'dwt_direct': 'DWT Direct'
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for pipeline in pipelines:
            means = np.array(snr_results[pipeline][metric]['mean'])
            stds = np.array(snr_results[pipeline][metric]['std'])

            ax.plot(snr_levels, means, marker='o', linewidth=2, markersize=8,color=colors[pipeline], label=labels[pipeline])

            ax.fill_between(snr_levels, means - stds, means + stds,color=colors[pipeline], alpha=0.2)

        ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} vs SNR', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.invert_xaxis()  # Higher SNR on left

    plt.tight_layout()
    filename = os.path.join(output_dir, 'snr_degradation_curves.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved SNR degradation curves: {filename}")
    plt.show()


if __name__ == '__main__':
    n_trials = 100
    run_statistical_comparison(n_trials=n_trials)
    # main()
