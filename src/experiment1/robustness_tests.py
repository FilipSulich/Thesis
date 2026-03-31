import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch, firwin, filtfilt
from scipy.stats import gmean, mannwhitneyu
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.pipeline_wavelets import Pipeline
from recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator
from recurrence_plot.AMI import AMI
from recurrence_plot.FNN import FNN


class SineNoiseBenchmark:
    """
    Noisy sine wave robustness benchmark for wavelet decomposition pipelines.

    Core design principle
    ---------------------
    The clean reference RP is computed from an ISOLATED single-frequency sine
    wave at each subband centre frequency — NOT from the composite signal
    processed through a pipeline. This is the true ground truth because:

      1. It is pipeline-independent: all pipelines are compared against the
         same reference, so any difference in ΔRQA reflects pipeline quality.
      2. It is noise-free by construction: there is no residual leakage or
         pipeline artifact in the reference.
      3. Embedding parameters (tau, m) are estimated ONCE from this clean
         reference sine via AMI and FNN, then fixed for all comparisons.
         This ensures differences in RQA reflect pipeline distortion, not
         parameter sensitivity.

    For each (SNR, pipeline, band, trial):
      - Decompose the noisy composite signal using the pipeline.
      - Compute RQA of the noisy subband output using the fixed reference
        embedding parameters.
      - ΔRQA = |RQA_noisy_pipeline - RQA_clean_reference|

    Literature basis
    ----------------
    - Thiel et al. (2002, Physica D 171): RQA sensitivity to noise.
    - Kraemer et al. (2018, Chaos 28): fixed RR epsilon selection.
    - Welch (1967, IEEE Trans. Audio): PSD estimation.
    - Vaidyanathan (1993): alias cancellation and noise in filter banks.
    """

    BAND_MAP = {
        'D1': {'f0': 48, 'f_low': 32, 'f_high': 64},
        'D2': {'f0': 24, 'f_low': 16, 'f_high': 32},
        'D3': {'f0': 12, 'f_low':  8, 'f_high': 16},
        'D4': {'f0':  6, 'f_low':  4, 'f_high':  8},
        'A4': {'f0':  2, 'f_low':  0.5, 'f_high': 4},
    }
    BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']

    def __init__(self, n_trials: int = 100, fs: int = 128, n_samples: int = 1280):
        self.M  = n_trials
        self.FS = fs
        self.N  = n_samples
        self.SNR_LEVELS = [20, 10, 5, 0]
        self.PIPELINES  = ['swt_recon', 'swt_direct', 'dwt_recon']

        self.OUTPUT_DIR = 'sine_noise_benchmark'
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        self._results: list[dict] = []

        # Populated in _build_clean_references()
        # key: band -> {'tau', 'm', 'epsilon', 'rqa', 'rp', 'signal'}
        self._reference: dict = {}

    # ------------------------------------------------------------------ #
    #  1. Clean reference: isolated sine per subband                       #
    # ------------------------------------------------------------------ #

    def _build_clean_references(self):
        """
        For each subband, generate an isolated clean sine at the subband
        centre frequency, estimate tau and m via AMI and FNN, find epsilon
        for RR=0.05, and compute the reference RQA and RP.

        These values are fixed for the entire experiment.
        """
        print("Building clean reference RPs from isolated single-frequency sines ...")
        t = np.arange(self.N) / self.FS

        for band, info in self.BAND_MAP.items():
            f0  = info['f0']
            sig = np.sin(2 * np.pi * f0 * t)
            sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-9)

            # Estimate tau via AMI
            ami  = AMI(sig, max_lag=min(50, self.N // 4))
            tau  = ami.get_first_minimum()
            if tau is None or tau < 1:
                tau = max(1, int(round(self.FS / f0 / 4)))

            # Estimate m via FNN
            fnn = FNN(sig, tau=tau, max_dim=8)
            m   = fnn.get_optimal_dimension(threshold=5.0)
            if m is None or m < 2:
                m = 2

            # Find epsilon for RR = 0.05
            eps = self._find_epsilon(sig, tau, m, target_rr=0.1)

            # Compute reference RQA and RP
            rqa_dict, rp_obj = self._compute_rqa_with_params(sig, tau, m, eps)

            self._reference[band] = {
                'tau':     tau,
                'm':       m,
                'epsilon': eps,
                'rqa':     rqa_dict,
                'rp':      rp_obj,
                'signal':  sig,
            }
            print(f"  {band}: tau={tau}, m={m}, eps={eps:.4f}  |  "
                  f"RQA: DET={rqa_dict['det']:.4f}, LAM={rqa_dict['lam']:.4f}")

    # ------------------------------------------------------------------ #
    #  2. Signal generation                                                #
    # ------------------------------------------------------------------ #

    def _composite_noisy(self, snr_db: float) -> np.ndarray:
        """
        Return composite sine + Gaussian noise at the given SNR.
        The composite contains one sine per subband at unit amplitude.
        """
        t     = np.arange(self.N) / self.FS
        clean = sum(np.sin(2 * np.pi * info['f0'] * t)
                    for info in self.BAND_MAP.values())

        if np.isinf(snr_db):
            return clean.copy()

        sig_power   = np.mean(clean ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise       = np.random.normal(0.0, np.sqrt(noise_power), self.N)
        return clean + noise

    # ------------------------------------------------------------------ #
    #  3. Epsilon selection                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_epsilon(signal: np.ndarray, tau: int, m: int,
                      target_rr: float = 0.05,
                      max_iters: int = 30, tol: float = 0.001) -> float:
        n = len(signal) - (m - 1) * tau
        if n < 10:
            return 0.1

        embedded = np.array([signal[i:i + m * tau:tau] for i in range(n)])
        dists    = squareform(pdist(embedded, metric='euclidean'))
        flat     = dists[np.triu_indices_from(dists, k=1)]

        eps_lo = np.percentile(flat, 1)
        eps_hi = np.percentile(flat, 30)

        for _ in range(max_iters):
            eps_mid = (eps_lo + eps_hi) / 2
            rp      = (dists <= eps_mid).astype(np.float32)
            np.fill_diagonal(rp, 0)
            rr = rp.sum() / (rp.shape[0] * (rp.shape[0] - 1))

            if abs(rr - target_rr) < tol:
                return eps_mid
            if rr < target_rr:
                eps_lo = eps_mid
            else:
                eps_hi = eps_mid

        return eps_mid

    # ------------------------------------------------------------------ #
    #  4. RQA computation with explicit parameters                         #
    # ------------------------------------------------------------------ #

    def _compute_rqa_with_params(self, signal: np.ndarray,
                                  tau: int, m: int, eps: float) -> tuple:
        """
        Compute RQA using explicitly provided tau, m, eps.
        Returns (rqa_dict, rp_object).
        """
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

    # ------------------------------------------------------------------ #
    #  5. PSD metrics                                                      #
    # ------------------------------------------------------------------ #

    def _psd_metrics(self, signal: np.ndarray, band: str):
        info   = self.BAND_MAP[band]
        f0, f_low, f_high = info['f0'], info['f_low'], info['f_high']

        nperseg = min(256, len(signal) // 4)
        f, pxx  = welch(signal, self.FS, nperseg=nperseg, noverlap=nperseg // 2)

        in_band    = (f >= f_low) & (f <= f_high)
        peak_idx   = np.argmin(np.abs(f - f0))
        peak_val   = pxx[peak_idx]
        noise_mask = in_band & (np.abs(f - f0) > 2.0)

        if noise_mask.sum() == 0:
            noise_floor = np.mean(pxx[in_band]) + 1e-12
            flatness    = 1.0
        else:
            nf_vals     = pxx[noise_mask]
            noise_floor = np.mean(nf_vals) + 1e-12
            flatness    = float(gmean(nf_vals + 1e-12) / noise_floor)

        snr_sub = 10 * np.log10(peak_val / noise_floor)

        in_band_power = np.trapezoid(pxx[in_band], f[in_band]) + 1e-12
        df_bin        = f[1] - f[0]
        signal_power  = peak_val * df_bin
        inf_val       = max(0.0, (in_band_power - signal_power) / in_band_power)

        return snr_sub, inf_val, flatness, f, pxx

    # ------------------------------------------------------------------ #
    #  6. RP similarity                                                    #
    # ------------------------------------------------------------------ #

    def _rp_similarity(self, rp_noisy, rp_ref) -> dict:
        """
        Compare noisy pipeline RP against the clean reference RP.
        Returns Hamming distance, Frobenius distance, Pearson correlation.
        """
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

        m1 = extract_matrix(rp_noisy)
        m2 = extract_matrix(rp_ref)

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

    # ------------------------------------------------------------------ #
    #  7. Pipeline decomposition                                           #
    # ------------------------------------------------------------------ #

    def _decompose(self, signal: np.ndarray, pipeline_name: str) -> dict:
        if pipeline_name == 'fir_filter':
            return self._fir_bank(signal)
        p = Pipeline(pipeline_name='all', level=4, wavelet='db4')
        return getattr(p, f"pipeline_{pipeline_name}")(signal)

    def _fir_bank(self, signal: np.ndarray) -> dict:
        edges = {
            'D1': (32.0, 63.5),
            'D2': (16.0, 32.0),
            'D3': ( 8.0, 16.0),
            'D4': ( 4.0,  8.0),
            'A4': ( 0.5,  4.0),
        }
        subs = {}
        for band, (lo, hi) in edges.items():
            taps      = firwin(501, [lo, hi], pass_zero=False,
                               window=('kaiser', 8.6), fs=self.FS)
            subs[band] = filtfilt(taps, 1.0, signal)
        return subs

    # ------------------------------------------------------------------ #
    #  8. Main experiment loop                                             #
    # ------------------------------------------------------------------ #

    def run_experiment(self):
        # Step 1: build clean isolated-sine references
        self._build_clean_references()

        # Step 2: noise sweep
        for snr in self.SNR_LEVELS:
            print(f"\n--- SNR = {snr} dB ---")
            for trial in range(self.M):
                noisy_sig = self._composite_noisy(snr)

                for p_name in self.PIPELINES:
                    subbands = self._decompose(noisy_sig, p_name)

                    for band in self.BAND_NAMES:
                        ref  = self._reference[band]
                        tau  = ref['tau']
                        m    = ref['m']
                        eps  = ref['epsilon']
                        # IMPORTANT: epsilon is fixed from the clean reference.
                        # We do NOT re-estimate epsilon from the noisy output.
                        # This ensures RR is the same as in the reference and
                        # differences in RQA reflect structural distortion only.

                        sub_sig = subbands[band]
                        std     = np.std(sub_sig)
                        if std < 1e-9:
                            continue
                        sn = (sub_sig - np.mean(sub_sig)) / std

                        # Spectral metrics
                        snr_sub, inf_val, flatness, _, _ = self._psd_metrics(sn, band)

                        # RQA with reference parameters
                        rqa_noisy, rp_noisy = self._compute_rqa_with_params(
                            sn, tau, m, eps
                        )

                        # ΔRQA: deviation from isolated clean sine reference
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

                        # RP similarity vs clean reference RP
                        rp_sim = self._rp_similarity(rp_noisy, ref['rp'])

                        self._results.append({
                            'snr_in':   snr,
                            'trial':    trial,
                            'pipeline': p_name,
                            'band':     band,
                            'snr_sub':  snr_sub,
                            'inf':      inf_val,
                            'flatness': flatness,
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

        self._print_summary()
        self._statistical_comparison()
        self.plot_degradation_curves()
        self.plot_boxplots()
        self.plot_trial_psd(snr_target=10)

    # ------------------------------------------------------------------ #
    #  9. Statistical comparison                                           #
    # ------------------------------------------------------------------ #

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

    def _pairwise_tests(self, df: pd.DataFrame, metric: str) -> list:
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

    # ------------------------------------------------------------------ #
    #  10. Summary                                                         #
    # ------------------------------------------------------------------ #

    def _print_summary(self):
        print("\n" + "=" * 70)
        print("ΔRQA SUMMARY  (mean ± std, all bands, all trials)")
        print("=" * 70)
        for snr in self.SNR_LEVELS:
            print(f"\nSNR = {snr} dB")
            sub = self.df[self.df['snr_in'] == snr]
            for p in self.PIPELINES:
                d = sub[sub['pipeline'] == p]
                print(f"  {p:15s}  "
                      f"ΔDET={d['delta_det'].mean():.4f}±{d['delta_det'].std():.4f}  "
                      f"ΔLAM={d['delta_lam'].mean():.4f}±{d['delta_lam'].std():.4f}  "
                      f"ΔTT={d['delta_tt'].mean():.4f}±{d['delta_tt'].std():.4f}  "
                      f"Hamming={d['hamming'].mean():.4f}±{d['hamming'].std():.4f}")

    # ------------------------------------------------------------------ #
    #  11. Plots                                                           #
    # ------------------------------------------------------------------ #

    def plot_degradation_curves(self):
        sns.set_style("whitegrid")
        for metric, ylabel in [
            ('delta_det', 'ΔDET'),
            ('delta_lam', 'ΔLAM'),
            ('delta_tt',  'ΔTT'),
            ('hamming',   'Hamming distance'),
            ('pearson',   'Pearson RP correlation'),
        ]:
            fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=False)
            for i, band in enumerate(self.BAND_NAMES):
                ax   = axes[i]
                data = self.df[self.df['band'] == band]
                sns.lineplot(data=data, x='snr_in', y=metric,
                             hue='pipeline', ax=ax, marker='o',
                             errorbar=('ci', 95), palette='colorblind')
                ax.set_title(f"Subband {band}", fontweight='bold')
                ax.invert_xaxis()
                ax.set_xlabel("Input SNR (dB)")
                ax.set_ylabel(ylabel if i == 0 else "")
                ax.legend(fontsize=8, title='Pipeline')
                ax.grid(alpha=0.3)

            fig.suptitle(
                f"{ylabel} degradation vs SNR "
                f"(reference = isolated clean sine, params from AMI/FNN)",
                fontsize=13, fontweight='bold', y=1.02,
            )
            plt.tight_layout()
            out = os.path.join(self.OUTPUT_DIR, f'curve_{metric}.png')
            plt.savefig(out, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {out}")

    def plot_boxplots(self):
        metrics = [
            ('delta_det', 'ΔDET',                  'lower'),
            ('delta_lam', 'ΔLAM',                  'lower'),
            ('delta_tt',  'ΔTT',                   'lower'),
            ('hamming',   'Hamming distance',       'lower'),
            ('frobenius', 'Frobenius distance',     'lower'),
            ('pearson',   'Pearson RP correlation', 'higher'),
            ('snr_sub',   'In-band SNR (dB)',       'higher'),
            ('inf',       'In-band noise fraction', 'lower'),
            ('flatness',  'Noise-floor flatness',   'higher'),
        ]
        for metric, label, better in metrics:
            fig, axes = plt.subplots(1, len(self.SNR_LEVELS), figsize=(18, 5), sharey=True)
            for j, snr in enumerate(self.SNR_LEVELS):
                ax   = axes[j]
                data = self.df[self.df['snr_in'] == snr]
                sns.boxplot(data=data, x='pipeline', y=metric, ax=ax,
                            palette='colorblind', order=self.PIPELINES)
                ax.set_title(f"SNR = {snr} dB")
                ax.set_xticklabels(ax.get_xticklabels(),
                                   rotation=30, ha='right', fontsize=8)
                ax.set_ylabel(label if j == 0 else "")
                ax.grid(axis='y', alpha=0.3)

            fig.suptitle(f"{label}  [{better} is better]",
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            out = os.path.join(self.OUTPUT_DIR, f'box_{metric}.png')
            plt.savefig(out, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {out}")

    def plot_trial_psd(self, snr_target: int = 10):
        np.random.seed(0)
        noisy = self._composite_noisy(snr_target)
        t     = np.arange(self.N) / self.FS
        alias_fold = self.FS / (2 ** 4)  # 8 Hz for level-4 DWT

        fig = plt.figure(figsize=(14, 18))
        gs  = GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.3)

        for row, band in enumerate(self.BAND_NAMES):
            info   = self.BAND_MAP[band]
            f0, f_low, f_high = info['f0'], info['f_low'], info['f_high']

            # Clean isolated reference sine
            ref_sig = np.sin(2 * np.pi * f0 * t)

            for col, (p_name, p_title) in enumerate([
                ('swt_recon', 'SWT reconstruction'),
                ('dwt_recon', 'DWT reconstruction'),
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
                if row == 0:
                    ax.legend(fontsize=7, loc='upper right')

        fig.suptitle(
            f"PSD: SWT recon vs DWT recon | SNR_in = {snr_target} dB\n"
            "Blue = clean isolated reference sine. Red = noisy pipeline output.\n"
            "Green shading = passband. Orange dotted = DWT alias fold (8 Hz).",
            fontsize=11, fontweight='bold',
        )
        out = os.path.join(self.OUTPUT_DIR, f'psd_snr{snr_target}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out}")


if __name__ == '__main__':
    bench = SineNoiseBenchmark(n_trials=50, fs=128, n_samples=1280)
    bench.run_experiment()
    bench.plot_trial_psd(snr_target=20)
    bench.plot_trial_psd(snr_target=0)