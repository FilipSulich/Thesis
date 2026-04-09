import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from scipy.stats import gmean, mannwhitneyu
from matplotlib.gridspec import GridSpec

from src.experiment1.subband_calculator import SubbandCalculator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.pipeline_wavelets import Pipeline
from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

class SineNoiseBenchmark:
    # BAND_MAP = {
    #     'D1': {'f0': 48, 'f_low': 32, 'f_high': 64},
    #     'D2': {'f0': 24, 'f_low': 16, 'f_high': 32},
    #     'D3': {'f0': 12, 'f_low':  8, 'f_high': 16},
    #     'D4': {'f0':  6, 'f_low':  4, 'f_high':  8},
    #     'A4': {'f0':  2, 'f_low':  0, 'f_high': 4},
    # }
    BAND_MAP = {
        'D1': {'f0': 47, 'f_low': 32, 'f_high': 64},
        'D2': {'f0': 23, 'f_low': 16, 'f_high': 32},
        'D3': {'f0': 13, 'f_low':  8, 'f_high': 16},
        'D4': {'f0':  5, 'f_low':  4, 'f_high':  8},
        'A4': {'f0':  1, 'f_low':  0, 'f_high': 4},
    }
    BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']

    STYLE_CONFIG = {
        'colors': {'swt_recon': '#2ecc71', 'swt_direct': '#3498db', 'dwt_recon': '#e74c3c'},
        'labels': {'swt_recon': 'SWT-R', 'swt_direct': 'SWT-D', 'dwt_recon': 'DWT-R'},
        'median_color': '#2c3e50',
        'mean_color': 'red',
        'ref_color': '#34495e',
        'grid_alpha': 0.3
    }
    def __init__(self, n_trials: int = 100, fs: int = 128, n_samples: int = 1280):
        self.M = n_trials
        self.FS = fs
        self.N = n_samples
        self.SNR_LEVELS = [20, 10, 5, 0]
        self.PIPELINES = ['swt_recon', 'swt_direct', 'dwt_recon']
        self.OUTPUT_DIR = '../../outputs/sine_noise_benchmark'
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self._results: list[dict] = []
        self.calculator = SubbandCalculator(is_sine=True)
        self._reference: dict = {}

    def _build_clean_references(self, pipeline_name):
        clean_composite = self._composite_noisy(snr_db=np.inf)

        clean_subbands = self._decompose(clean_composite, pipeline_name)
        params = {}
        for band in self.BAND_NAMES:
            sig = clean_subbands[band]

            sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-9)

            tau, m = self.calculator.get_optimal_params(sig)

            eps = self.calculator.get_epsilon(sig, tau, m, target_rr=0.1)

            rqa_dict, rp_obj = self._compute_rqa_with_params(sig, tau, m, eps)

            params[band] =  {
                'tau': tau,
                'm': m,
                'epsilon': eps,
                'rqa': rqa_dict,
                'rp': rp_obj,
                'signal': sig,
            }

        return params
    def _composite_noisy(self, snr_db: float) -> np.ndarray:
        t     = np.arange(self.N) / self.FS
        clean = sum(np.sin(2 * np.pi * info['f0'] * t)
                    for info in self.BAND_MAP.values())

        if np.isinf(snr_db):
            return clean.copy()

        sig_power   = np.mean(clean ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise       = np.random.normal(0.0, np.sqrt(noise_power), self.N)
        return clean + noise

    def _compute_rqa_with_params(self, signal: np.ndarray,tau: int, m: int, eps: float) -> tuple:
        std = np.std(signal)
        if std < 1e-9:
            return {'det': np.nan, 'lam': np.nan, 'tt': np.nan, 'rr': np.nan}, None

        sn = (signal - np.mean(signal)) / std

        try:
            rp_gen = RecurrencePlotGenerator(
                signal=sn,
                time_delay=tau,
                embedding_dim=m,
                epsilon=eps,
                path=self.OUTPUT_DIR,
                unthresholded=False,
            )
            rqa, rp = rp_gen.compute_rqa_binary()
            return {
                'det': rqa.determinism,
                'lam': rqa.laminarity,
                'tt':  rqa.trapping_time,
                'rr':  rqa.recurrence_rate,
            }, rp
        except Exception as e:
            return {'det': np.nan, 'lam': np.nan, 'tt': np.nan, 'rr': np.nan}, None

    @staticmethod
    def extract_matrix(rp_obj):
        if rp_obj is None:
            return None
        if hasattr(rp_obj, 'recurrence_matrix_reverse_normalized'):
            m = rp_obj.recurrence_matrix_reverse_normalized
        else:
            m = rp_obj
        if m is None or (isinstance(m, np.ndarray) and m.size == 0):
            return None
        return np.array(m, dtype=np.float32)

    def inspect_clusters(self, snr_target=10, threshold=0.25):
        """
        Identifies which bands and trials are responsible for the ΔDET outlier clusters.
        """
        print(f"\n--- Cluster Inspection (SNR={snr_target} dB, ΔDET > {threshold}) ---")

        # Filter for the specific outliers seen in your boxplot
        outliers = self.df[(self.df['snr_in'] == snr_target) & (self.df['delta_det'] > threshold)]

        if outliers.empty:
            print("No outliers found for the given threshold.")
            return

        # Count occurrences by band and pipeline
        summary = outliers.groupby(['pipeline', 'band']).size().reset_index(name='outlier_count')
        print(summary)

        # Pick the 'worst' trial to visualize
        worst_case = outliers.sort_values('delta_det', ascending=False).iloc[0]
        print(
            f"\nWorst outlier: Pipeline {worst_case['pipeline']}, Band {worst_case['band']}, Trial {worst_case['trial']}")
        print(f"ΔDET: {worst_case['delta_det']:.4f}")

        print(f"Ref DET for {worst_case['band']}: {self._reference[worst_case['band']]['rqa']['det']:.4f}")
        print(f"Trial DET for {worst_case['band']}: {worst_case['det']:.4f}")

        return worst_case

    def _rp_similarity(self, rp_noisy, rp_ref) -> dict:
        m1 = self.extract_matrix(rp_noisy)
        m2 = self.extract_matrix(rp_ref)

        if m1 is None or m2 is None:
            return {'hamming': np.nan, 'frobenius': np.nan, 'pearson': np.nan}

        n = min(m1.shape[0], m2.shape[0])
        m1, m2 = m1[:n, :n], m2[:n, :n]

        hamming_d  = float(np.mean(m1 != m2))
        frobenius_d = float(np.linalg.norm(m1 - m2, ord='fro'))
        f1, f2     = m1.flatten(), m2.flatten()
        if np.std(f1) > 1e-9 and np.std(f2) > 1e-9:
            pearson = float(np.corrcoef(f1, f2)[0, 1])
        else:
            pearson = np.nan

        return {'hamming': hamming_d, 'frobenius': frobenius_d, 'pearson': pearson}

    @staticmethod
    def _decompose(signal: np.ndarray, pipeline_name: str) -> dict:
        p = Pipeline(pipeline_name='all', level=4, wavelet='db4')
        return getattr(p, f"pipeline_{pipeline_name}")(signal)

    def run_experiment(self):
        for snr in self.SNR_LEVELS:
            print(f"\n--- SNR = {snr} dB ---")
            for trial in range(self.M):
                noisy_sig = self._composite_noisy(snr)

                for p_name in self.PIPELINES:
                    reference_pipeline = self._build_clean_references(p_name)

                    subbands = self._decompose(noisy_sig, p_name)

                    for band in self.BAND_NAMES:
                        ref  = reference_pipeline[band]
                        tau  = ref['tau']
                        m    = ref['m']
                        eps  = ref['epsilon']
                        sub_sig = subbands[band]
                        std     = np.std(sub_sig)
                        if std < 1e-9:
                            continue
                        sn = (sub_sig - np.mean(sub_sig)) / std

                        # eps_noisy = self.calculator.get_epsilon(sn, tau, m, target_rr=0.1)

                        rqa_noisy, rp_noisy = self._compute_rqa_with_params(sn, tau, m, eps)

                        ref_rqa = ref['rqa']
                        delta   = {
                            f'delta_{k}': (
                                abs(rqa_noisy[k] - ref_rqa[k])
                                if (not np.isnan(rqa_noisy.get(k, np.nan))
                                    and not np.isnan(ref_rqa.get(k, np.nan)))
                                else np.nan
                            )
                            for k in ('det', 'lam', 'tt', 'rr')
                        }

                        rp_sim = self._rp_similarity(rp_noisy, ref['rp'])

                        self._results.append({
                            'snr_in':   snr,
                            'trial':    trial,
                            'pipeline': p_name,
                            'band':     band,
                            'det':      rqa_noisy['det'],
                            'lam':      rqa_noisy['lam'],
                            'tt':       rqa_noisy['tt'],
                            'rr':       rqa_noisy['rr'],
                            **delta,
                            **rp_sim,
                        })

                if (trial + 1) % 20 == 0:
                    print(f"  Trials: {trial + 1}/{self.M}")

        self.df = pd.DataFrame(self._results)
        csv_path = os.path.join(self.OUTPUT_DIR, 'sine_bench_results.csv')
        self.df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

        # self._statistical_comparison()
        # self.plot_violinplots()
        self.plot_degradation_curves()
        # self.plot_violinplots()
        self.plot_boxplots()
        snr_targets = [20,10,5,0]
        for target in snr_targets:
            self.inspect_clusters(snr_target=target)

    def _statistical_comparison(self):
        metrics = ['delta_det', 'delta_lam', 'delta_tt',
                   'hamming', 'frobenius', 'pearson',
                   'snr_sub', 'inf', 'flatness']

        rows_band = []
        rows_agg  = []

        for snr in self.SNR_LEVELS:
            sub = self.df[self.df['snr_in'] == snr]

            # Per-band
            for band in self.BAND_NAMES:
                d = sub[sub['band'] == band]
                for metric in metrics:
                    pairs = self._pairwise_tests(d, metric)
                    for p in pairs:
                        p.update({'snr': snr, 'band': band, 'metric': metric})
                    rows_band.extend(self._holm_correct(pairs))

            # Aggregated (mean over bands per trial)
            agg = sub.groupby(['trial', 'pipeline'])[metrics].mean().reset_index()
            for metric in metrics:
                pairs = self._pairwise_tests(agg, metric)
                for p in pairs:
                    p.update({'snr': snr, 'metric': metric})
                rows_agg.extend(self._holm_correct(pairs))

        pd.DataFrame(rows_band).to_csv(
            os.path.join(self.OUTPUT_DIR, 'stats_per_band.csv'), index=False)
        pd.DataFrame(rows_agg).to_csv(
            os.path.join(self.OUTPUT_DIR, 'stats_aggregated.csv'), index=False)
        print("Statistical comparison saved.")

    def plot_boxplots(self):
        metrics = [
            ('delta_det', 'ΔDET', 'lower'), ('delta_lam', 'ΔLAM', 'lower'),
            ('delta_tt', 'ΔTT', 'lower'), ('hamming', 'Hamming Distance', 'lower'),
            ('frobenius', 'Frobenius Distance', 'lower'), ('pearson', 'RP Correlation', 'higher'),
            ('snr_sub', 'Subband SNR (dB)', 'higher')
        ]

        sns.set_theme(style="whitegrid", font_scale=1.1)

        for metric, label, better in metrics:
            fig, axes = plt.subplots(1, len(self.SNR_LEVELS), figsize=(20, 6), sharey=False)

            for j, snr in enumerate(self.SNR_LEVELS):
                ax = axes[j]
                snr_data = self.df[self.df['snr_in'] == snr]

                # Prepare data list for boxplot
                plot_data = [snr_data[snr_data['pipeline'] == p][metric].dropna().values
                             for p in self.PIPELINES]

                bplot = ax.boxplot(plot_data,positions=range(len(self.PIPELINES)),
                                   patch_artist=True,
                                   widths=0.6,
                                   notch=False,
                                   flierprops={
                                       'marker': 'o',
                                       'markersize': 4,
                                       'markerfacecolor': 'black',
                                       'alpha': 0.4,
                                       'markeredgecolor': 'none'
                                   })

                # Styling the boxes and lines
                for i, box in enumerate(bplot['boxes']):
                    color = self.STYLE_CONFIG['colors'][self.PIPELINES[i]]
                    box.set_facecolor(color)
                    box.set_alpha(0.7)
                    box.set_edgecolor('#2c3e50')
                    box.set_linewidth(1.5)

                # Style medians, whiskers, and caps
                plt.setp(bplot['medians'], color=self.STYLE_CONFIG['median_color'], linewidth=2)
                plt.setp(bplot['whiskers'], color='#2c3e50', linewidth=1.2, linestyle='-')
                plt.setp(bplot['caps'], color='#2c3e50', linewidth=1.2)

                ax.set_title(f"Input SNR: {snr} dB", fontweight='bold')
                ax.set_xticks(range(len(self.PIPELINES)))
                ax.set_xticklabels([self.STYLE_CONFIG['labels'][p] for p in self.PIPELINES])
                ax.grid(axis='y', alpha=self.STYLE_CONFIG['grid_alpha'])

                if j == 0:
                    ax.set_ylabel(label, fontweight='bold', fontsize=12)

                sns.despine(ax=ax, trim=True)

            fig.suptitle(f"{label} [{better} is better]",fontsize=16, fontweight='bold', y=1.02)

            plt.tight_layout()
            out = os.path.join(self.OUTPUT_DIR, f'boxplot_{metric}.png')
            plt.savefig(out, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved boxplot: {out}")

    @staticmethod
    def _pairwise_tests(df: pd.DataFrame, metric: str) -> list:
        pairs = []
        pipelines = df['pipeline'].unique().tolist()
        for i, p1 in enumerate(pipelines):
            for p2 in pipelines[i + 1:]:
                x = df[df['pipeline'] == p1][metric].dropna().values
                y = df[df['pipeline'] == p2][metric].dropna().values
                if len(x) < 3 or len(y) < 3:
                    continue
                u, pv = mannwhitneyu(x, y, alternative='two-sided')
                n1, n2 = len(x), len(y)
                s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
                pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
                d = (np.mean(x) - np.mean(y)) / pooled if pooled > 1e-9 else np.nan
                pairs.append({
                    'pipeline_A': p1, 'pipeline_B': p2,
                    'u_stat': u, 'p_value': pv, 'cohens_d': d,
                    'mean_A': np.mean(x), 'mean_B': np.mean(y),
                    'std_A': np.std(x), 'std_B': np.std(y),
                })
        return pairs

    @staticmethod
    def _holm_correct(pairs: list) -> list:
        if not pairs:
            return pairs
        n  = len(pairs)
        idx = np.argsort([p['p_value'] for p in pairs])
        for rank, i in enumerate(idx):
            pairs[i]['alpha_holm']      = 0.05 / (n - rank)
            pairs[i]['significant_holm'] = pairs[i]['p_value'] < pairs[i]['alpha_holm']
        return pairs

    def plot_violinplots(self):
        metrics = [
            ('delta_det', 'ΔDET', 'lower'), ('delta_lam', 'ΔLAM', 'lower'),
            ('delta_tt', 'ΔTT', 'lower'), ('hamming', 'Hamming Distance', 'lower'),
            ('frobenius', 'Frobenius Distance', 'lower'), ('pearson', 'RP Correlation', 'higher'),
            ('snr_sub', 'Subband SNR (dB)', 'higher'), ('inf', 'Noise Fraction', 'lower'),
            ('flatness', 'Noise Flatness', 'higher')
        ]

        sns.set_theme(style="whitegrid", font_scale=1.1)

        for metric, label, better in metrics:
            fig, axes = plt.subplots(1, len(self.SNR_LEVELS), figsize=(20, 6), sharey=False)

            for j, snr in enumerate(self.SNR_LEVELS):
                ax = axes[j]
                snr_data = self.df[self.df['snr_in'] == snr]

                plot_data = [snr_data[snr_data['pipeline'] == p][metric].dropna().values for p in self.PIPELINES]

                parts = ax.violinplot(plot_data, positions=range(len(self.PIPELINES)),
                                      showmeans=True, showmedians=True, widths=0.7)

                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(self.STYLE_CONFIG['colors'][self.PIPELINES[i]])
                    pc.set_alpha(0.65)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1.2)

                parts['cmedians'].set_edgecolor(self.STYLE_CONFIG['median_color'])
                parts['cmedians'].set_linewidth(2.5)
                parts['cmeans'].set_edgecolor(self.STYLE_CONFIG['mean_color'])
                parts['cmeans'].set_linewidth(2)

                for line in ['cmins', 'cmaxes', 'cbars']:
                    parts[line].set_edgecolor(self.STYLE_CONFIG['median_color'])
                    parts[line].set_linewidth(1)

                ax.set_title(f"Input SNR: {snr} dB", fontweight='bold')
                ax.set_xticks(range(len(self.PIPELINES)))
                ax.set_xticklabels([self.STYLE_CONFIG['labels'][p] for p in self.PIPELINES])
                ax.grid(axis='y', alpha=self.STYLE_CONFIG['grid_alpha'])

                if j == 0:
                    ax.set_ylabel(label, fontweight='bold', fontsize=12)

                sns.despine(ax=ax, trim=True)

            fig.suptitle(f"{label} Distribution [{better} is better]",
                         fontsize=16, fontweight='bold', y=1.02)

            plt.tight_layout()
            out = os.path.join(self.OUTPUT_DIR, f'violin_pro_{metric}.png')
            plt.savefig(out, dpi=300, bbox_inches='tight')
            plt.close()

    def plot_degradation_curves(self):
        sns.set_theme(style="whitegrid", font_scale=1.1)

        metrics_to_plot = [
            ('delta_det', 'Avg. Determinism Deviation', 'lower'),
            ('delta_lam', 'Avg. Laminarity Deviation', 'lower'),
            ('delta_tt', 'Avg. Trapping Time Deviation', 'lower'),
            ('hamming', 'Avg. Hamming Distance', 'lower'),
            ('pearson', 'Avg. RP Correlation', 'higher'),
        ]

        metric_cols = [m[0] for m in metrics_to_plot]
        agg_df = self.df.groupby(['snr_in', 'trial', 'pipeline'])[metric_cols].mean().reset_index()

        for metric, label, better in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(9, 6))

            for pipe in self.PIPELINES:
                pipe_data = agg_df[agg_df['pipeline'] == pipe]
                stats = pipe_data.groupby('snr_in')[metric].agg(['mean', 'std']).reset_index()

                stats = stats.sort_values('snr_in')

                color = self.STYLE_CONFIG['colors'][pipe]
                label_name = self.STYLE_CONFIG['labels'][pipe]

                ax.plot(stats['snr_in'], stats['mean'],
                        marker='o', markersize=7, linewidth=2.5,
                        color=color, label=label_name, zorder=3)

                ax.fill_between(stats['snr_in'],
                                stats['mean'] - stats['std'],
                                stats['mean'] + stats['std'],
                                color=color, alpha=0.15, zorder=2)

            # 3. Standardized Formatting
            ax.set_title(f"{label} Degradation\n({better} = better)",
                         fontweight='bold', fontsize=14, pad=20)
            ax.set_xlabel("Input SNR (dB)", fontweight='bold', labelpad=10)
            ax.set_ylabel(label, fontweight='bold', labelpad=10)

            # Ensure the X-axis is inverted (High SNR -> Low SNR) consistently
            ax.set_xlim(max(self.SNR_LEVELS) + 2, min(self.SNR_LEVELS) - 2)
            ax.set_xticks(self.SNR_LEVELS)

            # Style the legend
            ax.legend(frameon=True, facecolor='white', framealpha=0.9,
                      edgecolor='#d1d1d1', fontsize=10, loc='best')

            # Grid and Spines
            ax.grid(True, linestyle='--', alpha=self.STYLE_CONFIG['grid_alpha'])
            sns.despine(trim=True)

            # 4. Save with consistent DPI and bounding box
            plt.tight_layout()
            out_path = os.path.join(self.OUTPUT_DIR, f'curve_pro_{metric}.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved standardized degradation curve: {out_path}")

    def plot_trial_psd(self, snr_target: int = 10):
        np.random.seed(0)
        noisy = self._composite_noisy(snr_target)

        t     = np.arange(self.N) / self.FS
        alias_fold = self.FS / (2 ** 4)  # 8 Hz for level-4 DWT

        fig = plt.figure(figsize=(14, 18))
        gs  = GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.3)
        for row, band in enumerate(self.BAND_NAMES):
            info   = self.BAND_MAP[band]
            f0, f_low, f_high = info['f0'], info['f_low'], info['f_high']

            ref_sig = np.sin(2 * np.pi * f0 * t)

            for col, (p_name, p_title) in enumerate([
                ('swt_recon', 'SWT reconstruction'),
                ('dwt_recon', 'DWT reconstruction'),
                ('swt_direct', 'SWT direct')
                ,
            ]):
                ax = fig.add_subplot(gs[row, col])

                noisy_sub = self._decompose(noisy,   p_name)[band]
                nperseg   = min(256, len(noisy_sub) // 4)

                f_r, pxx_r = welch(ref_sig,   self.FS, nperseg=nperseg)
                f_n, pxx_n = welch(noisy_sub, self.FS, nperseg=nperseg)

                ax.semilogy(f_r, pxx_r, color='steelblue', lw=1.2, label='Clean ref sine')
                ax.semilogy(f_n, pxx_n, color='tomato',    lw=1.2, alpha=0.85, label='Noisy pipeline')
                ax.axvline(f0, color='black', ls='--', lw=1.0, alpha=0.7)
                ax.axvspan(f_low, f_high, alpha=0.07, color='green')

                if p_name == 'dwt_recon' and f_low <= alias_fold <= f_high:
                    ax.axvline(alias_fold, color='orange', ls=':', lw=1.2,
                               label=f'Alias fold {alias_fold:.0f} Hz')

                ax.set_xlim([0, self.FS / 2])
                ax.set_ylim([1e-12, None])
                ax.set_title(f"{band} | {p_title}", fontsize=9, fontweight='bold')
                if row == len(self.BAND_NAMES) - 1:
                    ax.set_xlabel("Frequency (Hz)", fontsize=8)
                if col == 0:
                    ax.set_ylabel("PSD (V²/Hz)", fontsize=8)
                # if row == 0:
                ax.legend(fontsize=7, loc='upper right')

        fig.suptitle(
            f"PSD: SWT recon vs DWT recon | SNR_in = {snr_target} dB\n",
            fontsize=11, fontweight='bold',
        )
        out = os.path.join(self.OUTPUT_DIR, f'psd_snr{snr_target}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out}")

if __name__ == '__main__':
    bench = SineNoiseBenchmark(n_trials=50, fs=128, n_samples=1280)
    bench.run_experiment()