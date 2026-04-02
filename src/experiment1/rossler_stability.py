import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import levene, mannwhitneyu
import os

from src.experiment1.visualize_decomposition import DecompositionVisualizer


class StabilityTester:
    BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']
    METRICS    = ['determinism', 'laminarity', 'trapping_time']
    METRIC_LABELS = {
        'determinism':   'DET',
        'laminarity':    'LAM',
        'trapping_time': 'TT',
    }
    PIPELINES = ['swt_recon', 'dwt_recon', 'swt_direct', 'dwt_direct']

    def __init__(self, visualizer, n_trials: int = 20,
                 output_dir: str = 'rossler_stability_output'):
        self.viz       = visualizer
        self.n_trials  = n_trials
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.results = {
            pipe: {band: {met: [] for met in self.METRICS}
                   for band in self.BAND_NAMES}
            for pipe in self.PIPELINES
        }

    @staticmethod
    def _find_epsilon(signal: np.ndarray, tau: int, m: int,
                      target_rr: float = 0.1,
                      max_iters: int = 30, tol: float = 0.005) -> float:
        """
        Binary search for epsilon achieving target recurrence rate.

        Fixed RR ensures that DET/LAM/TT differences across trials and
        pipelines reflect structural RP changes only, not threshold variation.
        Reference: Kraemer et al. (2018, Chaos 28, 085720).
        """
        from scipy.spatial.distance import pdist, squareform

        n = len(signal) - (m - 1) * tau
        if n < 10:
            return 0.1

        embedded = np.array([signal[i:i + m * tau:tau] for i in range(n)])
        D        = squareform(pdist(embedded, metric='euclidean'))
        flat     = D[np.triu_indices_from(D, k=1)]

        lo, hi = np.percentile(flat, 1), np.percentile(flat, 20)

        for _ in range(max_iters):
            mid = (lo + hi) / 2
            rp  = (D <= mid).astype(np.float32)
            np.fill_diagonal(rp, 0)
            rr  = rp.sum() / (rp.shape[0] * (rp.shape[0] - 1))

            if abs(rr - target_rr) < tol:
                return mid
            lo, hi = (mid, hi) if rr < target_rr else (lo, mid)

        return mid

    def run_stability_analysis(self, system_type: str = 'rossler'):
        print(f"\n{'='*70}")
        print(f"STABILITY TEST: {self.n_trials} realisations — {system_type.upper()}")
        print(f"{'='*70}")

        ref_signal = []
        if system_type == 'rossler':
            ref_signal   = self.viz._generate_rossler(seed=0)
        elif system_type == 'lorenz':
            ref_signal = self.viz.generate_lorenz_system(seed=0)

        fixed_params = self._estimate_params_per_pipeline(ref_signal)

        for trial in range(self.n_trials):
            signal = []
            if system_type == 'rossler':
                signal = self.viz._generate_rossler(seed=trial + 100)
            elif system_type == 'lorenz':
                signal = self.viz.generate_lorenz_system(seed=trial + 100)
            else :
                print("wrong syustem")

            for pipe in self.PIPELINES:
                subbands = self.viz.decompose_signal(signal, pipe)
                params   = fixed_params[pipe]

                for band in self.BAND_NAMES:
                    tau, m = params[band]['tau'], params[band]['m']

                    trimmed = self.viz._trim_signal(subbands[band])

                    # Check if signal is valid
                    if len(trimmed) < 10:
                        print(f"  Warning: {pipe} {band} signal too short ({len(trimmed)}), skipping")
                        continue

                    std = trimmed.std()
                    if std < 1e-9:
                        print(f"  Warning: {pipe} {band} has zero variance, skipping")
                        continue

                    normed  = (trimmed - trimmed.mean()) / std

                    # Find epsilon for fixed RR=0.1 (binary search)
                    # This ensures DET/LAM/TT differences reflect RP structure, not threshold
                    eps = self._find_epsilon(normed, tau, m, target_rr=0.1)

                    _, rqa = self.viz.compute_rp(normed, tau, m, eps)

                    if rqa:
                        self.results[pipe][band]['determinism'].append(rqa.determinism)
                        self.results[pipe][band]['laminarity'].append(rqa.laminarity)
                        self.results[pipe][band]['trapping_time'].append(rqa.trapping_time)

            if (trial + 1) % 5 == 0 or trial == 0:
                print(f"  Trial {trial + 1}/{self.n_trials} done")

        self._save_raw_results()
        stats_df = self._statistical_comparison()
        self._plot_distributions(stats_df)


    def _estimate_params_per_pipeline(self, ref_signal: np.ndarray) -> dict:
        """
        Estimate tau and m from reference realization, separately per pipeline.

        IMPORTANT: Applies same trimming as trial loop for consistency.
        """
        fixed_params = {}
        for pipe in self.PIPELINES:
            fixed_params[pipe] = {}
            subbands = self.viz.decompose_signal(ref_signal, pipe)
            for band in self.BAND_NAMES:
                # Apply same trimming as trial loop
                trimmed = self.viz._trim_signal(subbands[band])

                # Check validity
                if len(trimmed) < 10:
                    print(f"  Warning: [{pipe}] {band} signal too short after trim, using default params")
                    fixed_params[pipe][band] = {'tau': 1, 'm': 2}
                    continue

                std = trimmed.std()
                if std < 1e-9:
                    print(f"  Warning: [{pipe}] {band} has zero variance, using default params")
                    fixed_params[pipe][band] = {'tau': 1, 'm': 2}
                    continue

                normed  = (trimmed - trimmed.mean()) / std
                tau, m  = self.viz.get_optimal_parameters(normed)
                fixed_params[pipe][band] = {'tau': tau, 'm': m}
                print(f"  [{pipe}] {band}: tau={tau}, m={m}")

        return fixed_params

    def _save_raw_results(self):
        rows = []
        for pipe in self.PIPELINES:
            for band in self.BAND_NAMES:
                for met in self.METRICS:
                    for trial_idx, val in enumerate(
                            self.results[pipe][band][met]):
                        rows.append({
                            'pipeline': pipe,
                            'band':     band,
                            'metric':   met,
                            'trial':    trial_idx,
                            'value':    val,
                        })
        path = os.path.join(self.output_dir, 'stability_raw.csv')
        pd.DataFrame(rows).to_csv(path, index=False)

    def _statistical_comparison(self) -> pd.DataFrame:
        """
        For each (band, metric):
          - CV per pipeline
          - Levene test on raw value distributions (equality of variances)
          - Mann-Whitney U on absolute deviation distributions
          - Cohen's d on absolute deviations
        """
        rows = []

        for band in self.BAND_NAMES:
            for met in self.METRICS:
                swt_vals = np.array(self.results['swt_recon'][band][met])
                swt_direct_vals = np.array(self.results['swt_direct'][band][met])
                dwt_direct_vals = np.array(self.results['dwt_direct'][band][met])
                dwt_vals = np.array(self.results['dwt_recon'][band][met])

                if len(swt_vals) == 0 or len(dwt_vals) == 0 or len(swt_direct_vals) == 0:
                    continue

                # Robust statistics: Median ± IQR, MAD
                def robust_stats(vals):
                    median = np.median(vals)
                    q1, q3 = np.percentile(vals, [25, 75])
                    iqr = q3 - q1
                    # MAD: Median Absolute Deviation (scaled by 1.4826 for consistency with std under normality)
                    mad = 1.4826 * np.median(np.abs(vals - median))
                    return median, iqr, mad

                swt_med, swt_iqr, swt_mad = robust_stats(swt_vals)
                dwt_med, dwt_iqr, dwt_mad = robust_stats(dwt_vals)
                swt_direct_med, swt_direct_iqr, swt_direct_mad = robust_stats(swt_direct_vals)
                dwt_direct_med, dwt_direct_iqr, dwt_direct_mad = robust_stats(dwt_direct_vals)

                # MAD ratios (robust variability measure, relative to swt_recon baseline)
                mad_ratio_dwt = dwt_mad / swt_mad if swt_mad > 1e-9 else np.nan
                mad_ratio_swt_direct = swt_direct_mad / swt_mad if swt_mad > 1e-9 else np.nan
                mad_ratio_dwt_direct = dwt_direct_mad / swt_mad if swt_mad > 1e-9 else np.nan

                # Levene test: equality of variance
                lev_stat, lev_p = levene(swt_vals, dwt_vals, swt_direct_vals)

                # Mann-Whitney U tests with probability of superiority
                # P(X > Y) = U / (n1 * n2): prob that random value from group 1 > random value from group 2
                def prob_superiority(u_stat, n1, n2):
                    return u_stat / (n1 * n2)

                u_dwt, p_dwt = mannwhitneyu(swt_vals, dwt_vals, alternative='two-sided')
                ps_dwt = prob_superiority(u_dwt, len(swt_vals), len(dwt_vals))

                u_swt_direct, p_swt_direct = mannwhitneyu(swt_vals, swt_direct_vals, alternative='two-sided')
                ps_swt_direct = prob_superiority(u_swt_direct, len(swt_vals), len(swt_direct_vals))

                u_dwt_direct, p_dwt_direct = mannwhitneyu(swt_vals, dwt_direct_vals, alternative='two-sided')
                ps_dwt_direct = prob_superiority(u_dwt_direct, len(swt_vals), len(dwt_direct_vals))

                # Absolute deviations from median (for plotting)
                swt_devs = np.abs(swt_vals - swt_med)
                dwt_devs = np.abs(dwt_vals - dwt_med)
                swt_direct_devs = np.abs(swt_direct_vals - swt_direct_med)
                dwt_direct_devs = np.abs(dwt_direct_vals - dwt_direct_med)

                rows.append({
                    'band':       band,
                    'metric':     met,
                    # Robust statistics
                    'swt_median': swt_med,
                    'swt_iqr':    swt_iqr,
                    'swt_mad':    swt_mad,
                    'dwt_median': dwt_med,
                    'dwt_iqr':    dwt_iqr,
                    'dwt_mad':    dwt_mad,
                    'swt_direct_median': swt_direct_med,
                    'swt_direct_iqr':    swt_direct_iqr,
                    'swt_direct_mad':    swt_direct_mad,
                    'dwt_direct_median': dwt_direct_med,
                    'dwt_direct_iqr':    dwt_direct_iqr,
                    'dwt_direct_mad':    dwt_direct_mad,
                    # MAD ratios (robust variability measure)
                    'mad_ratio_dwt':        mad_ratio_dwt,
                    'mad_ratio_swt_direct': mad_ratio_swt_direct,
                    'mad_ratio_dwt_direct': mad_ratio_dwt_direct,
                    # Levene test
                    'levene_stat': lev_stat,
                    'levene_p':    lev_p,
                    # Mann-Whitney U with probability of superiority
                    'u_dwt':           u_dwt,
                    'p_dwt':           p_dwt,
                    'ps_dwt':          ps_dwt,
                    'u_swt_direct':    u_swt_direct,
                    'p_swt_direct':    p_swt_direct,
                    'ps_swt_direct':   ps_swt_direct,
                    'u_dwt_direct':    u_dwt_direct,
                    'p_dwt_direct':    p_dwt_direct,
                    'ps_dwt_direct':   ps_dwt_direct,
                    # Arrays for plotting
                    '_swt_devs':        swt_devs,
                    '_dwt_devs':        dwt_devs,
                    '_swt_direct_devs': swt_direct_devs,
                    '_dwt_direct_devs': dwt_direct_devs,
                    '_swt_vals':        swt_vals,
                    '_dwt_vals':        dwt_vals,
                    '_swt_direct_vals': swt_direct_vals,
                    '_dwt_direct_vals': dwt_direct_vals,
                })

        df = pd.DataFrame(rows)

        # Holm correction across (band × metric) comparisons
        # applied to Levene p-values (primary test)
        p_vals = df['levene_p'].values
        n      = len(p_vals)
        order  = np.argsort(p_vals)
        alphas = np.zeros(n)
        for rank, idx in enumerate(order):
            alphas[idx] = 0.05 / (n - rank)
        df['alpha_holm']      = alphas
        df['significant_holm'] = df['levene_p'] < df['alpha_holm']

        # Save statistics (drop array columns)
        stat_cols = [c for c in df.columns if not c.startswith('_')]
        path = os.path.join(self.output_dir, 'stability_stats.csv')
        df[stat_cols].to_csv(path, index=False)
        print(f"Statistics saved to {path}")

        return df

    def _plot_distributions(self, df: pd.DataFrame):
        self._plot_raw_violins(df)
        self._plot_deviation_violins(df)
        self._plot_cv_bars(df)
        self._plot_probability_superiority_heatmap(df)

    def _plot_raw_violins(self, df: pd.DataFrame):
        plot_bands = ['D1', 'D2', 'D3', 'D4']
        fig, axes  = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "Raw RQA distributions across Rössler realisations (All Pipelines)\n"
            "D1–D4 pooled across trials. Each observation = one trial × one band.",
            fontsize=12, fontweight='bold'
        )

        colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']  # SWT-R, DWT-R, SWT-D, DWT-D
        pipeline_labels = ['SWT-R', 'DWT-R', 'SWT-D', 'DWT-D']

        for ax, met in zip(axes, self.METRICS):
            pools = [[], [], [], []]  # One for each pipeline
            for band in plot_bands:
                row = df[(df['band'] == band) & (df['metric'] == met)]
                if row.empty:
                    continue
                pools[0].extend(row.iloc[0]['_swt_vals'].tolist())
                pools[1].extend(row.iloc[0]['_dwt_vals'].tolist())
                pools[2].extend(row.iloc[0]['_swt_direct_vals'].tolist())
                pools[3].extend(row.iloc[0]['_dwt_direct_vals'].tolist())

            parts = ax.violinplot(
                pools,
                positions=[1, 2, 3, 4],
                showmedians=True,
                showextrema=True,
            )
            for body, color in zip(parts['bodies'], colors):
                body.set_facecolor(color)
                body.set_alpha(0.7)
            for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
                if key in parts:
                    parts[key].set_color('black')
                    parts[key].set_linewidth(1.2)

            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(pipeline_labels, fontsize=9)
            ax.set_title(self.METRIC_LABELS[met], fontweight='bold')
            ax.set_ylabel('RQA value')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'stability_raw_violins.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    def _plot_deviation_violins(self, df: pd.DataFrame):
        """
        Absolute deviation distributions per pipeline, D1-D4 pooled.

        Each observation = |value_i − pipeline_mean| for one trial × one band.
        This directly visualises trial-to-trial instability: a pipeline
        whose output is consistent across realisations will have a tight
        distribution near zero; one whose output is phase-dependent will
        have a broad distribution with a heavy tail.
        """
        plot_bands = ['D1', 'D2', 'D3', 'D4']
        fig, axes  = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "Absolute deviation distributions across Rössler realisations (All Pipelines)\n"
            "D1–D4 pooled. Each observation = |value − pipeline mean|.\n"
            "Narrower = more consistent = better for clinical feature extraction.",
            fontsize=11, fontweight='bold'
        )

        colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
        pipeline_labels = ['SWT-R', 'DWT-R', 'SWT-D', 'DWT-D']

        for ax, met in zip(axes, self.METRICS):
            devs_pools = [[], [], [], []]  # One for each pipeline
            sig_flags = []

            for band in plot_bands:
                row = df[(df['band'] == band) & (df['metric'] == met)]
                if row.empty:
                    continue
                devs_pools[0].extend(row.iloc[0]['_swt_devs'].tolist())
                devs_pools[1].extend(row.iloc[0]['_dwt_devs'].tolist())
                devs_pools[2].extend(row.iloc[0]['_swt_direct_devs'].tolist())
                devs_pools[3].extend(row.iloc[0]['_dwt_direct_devs'].tolist())
                if row.iloc[0]['significant_holm']:
                    sig_flags.append(True)

            parts = ax.violinplot(
                devs_pools,
                positions=[1, 2, 3, 4],
                showmedians=True,
                showextrema=True,
            )
            for body, color in zip(parts['bodies'], colors):
                body.set_facecolor(color)
                body.set_alpha(0.7)
            for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
                if key in parts:
                    parts[key].set_color('black')
                    parts[key].set_linewidth(1.2)

            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(pipeline_labels, fontsize=9)
            ax.set_title(self.METRIC_LABELS[met], fontweight='bold')
            ax.set_ylabel('|value − pipeline mean|')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'stability_deviation_violins.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    def _plot_cv_bars(self, df: pd.DataFrame):
        """
        MAD (Median Absolute Deviation) per band per metric, grouped bars for all 4 pipelines.
        Holm-corrected significance stars from Levene test.
        """
        plot_bands   = ['D1', 'D2', 'D3', 'D4']
        n_bands      = len(plot_bands)
        n_metrics    = len(self.METRICS)

        fig, axes = plt.subplots(1, n_metrics, figsize=(18, 6), sharey=False)
        fig.suptitle(
            "MAD (Median Absolute Deviation) across Rössler realisations (All Pipelines)\n"
            "Lower MAD = more consistent RQA estimates across different signal phases.",
            fontsize=12, fontweight='bold'
        )

        colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
        pipeline_labels = ['SWT-R', 'DWT-R', 'SWT-D', 'DWT-D']
        x = np.arange(n_bands)
        width = 0.2
        sig_map = lambda p, sig: '***' if p < 0.001 else '**' if p < 0.01 \
                                 else '*' if p < 0.05 else ''

        for ax, met in zip(axes, self.METRICS):
            mads = [[], [], [], []]  # One list for each pipeline
            sigs = []

            for band in plot_bands:
                row = df[(df['band'] == band) & (df['metric'] == met)]
                if row.empty:
                    for mad_list in mads:
                        mad_list.append(0)
                    sigs.append('')
                    continue
                r = row.iloc[0]
                mads[0].append(r['swt_mad'])
                mads[1].append(r['dwt_mad'])
                mads[2].append(r['swt_direct_mad'])
                mads[3].append(r['dwt_direct_mad'])
                sigs.append(sig_map(r['levene_p'], r['significant_holm']))

            # Plot bars for each pipeline
            for i, (mad_list, color, label) in enumerate(zip(mads, colors, pipeline_labels)):
                offset = (i - 1.5) * width
                ax.bar(x + offset, mad_list, width,
                       label=label, color=color,
                       alpha=0.85, edgecolor='white')

            ax.set_xticks(x)
            ax.set_xticklabels(plot_bands)
            ax.set_title(self.METRIC_LABELS[met], fontweight='bold')
            ax.set_ylabel('MAD (scaled)')
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'stability_mad_bars.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    def _plot_probability_superiority_heatmap(self, df: pd.DataFrame):
        """
        Heatmap of probability of superiority across (band × metric) pairs.
        P(X > Y) from Mann-Whitney U: interpretable effect size.
        Shows where pipelines differ in magnitude.
        """
        plot_bands = ['D1', 'D2', 'D3', 'D4']
        n_bands = len(plot_bands)
        n_metrics = len(self.METRICS)

        # Create 3 subplots for 3 comparisons (DWT-R, SWT-D, DWT-D vs SWT-R)
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(
            "Probability of Superiority: P(SWT-R > Other) from Mann-Whitney U\n"
            "Values > 0.5 indicate SWT-R tends to have higher values; < 0.5 indicates lower values.",
            fontsize=12, fontweight='bold'
        )

        comparisons = [
            ('ps_dwt', 'DWT-R vs SWT-R'),
            ('ps_swt_direct', 'SWT-D vs SWT-R'),
            ('ps_dwt_direct', 'DWT-D vs SWT-R')
        ]

        for ax, (ps_col, title) in zip(axes, comparisons):
            # Build matrix: rows = bands, cols = metrics
            matrix = np.zeros((n_bands, n_metrics))

            for i, band in enumerate(plot_bands):
                for j, met in enumerate(self.METRICS):
                    row = df[(df['band'] == band) & (df['metric'] == met)]
                    if not row.empty:
                        matrix[i, j] = row.iloc[0][ps_col]
                    else:
                        matrix[i, j] = np.nan

            # Plot heatmap
            im = ax.imshow(matrix, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')

            # Add text annotations
            for i in range(n_bands):
                for j in range(n_metrics):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if (val < 0.3 or val > 0.7) else 'black'
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                               color=text_color, fontsize=10, fontweight='bold')

            ax.set_xticks(np.arange(n_metrics))
            ax.set_yticks(np.arange(n_bands))
            ax.set_xticklabels([self.METRIC_LABELS[m] for m in self.METRICS])
            ax.set_yticklabels(plot_bands)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('RQA Metric')
            ax.set_ylabel('Subband')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('P(SWT-R > Other)', rotation=270, labelpad=20)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'stability_prob_superiority_heatmap.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

if __name__ == '__main__':
    system = 'lorenz'
    # system = 'rossler'

    viz = DecompositionVisualizer(seed=42)

    # Run stability test
    stab = StabilityTester(viz, n_trials=20, output_dir=f'{system}_stability_output')
    stab.run_stability_analysis(system_type=system)
