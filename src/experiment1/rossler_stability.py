from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scikit_posthocs import posthoc_dunn
from scipy.stats import levene, mannwhitneyu, kruskal
import os

from src.experiment1.visualize_decomposition import DecompositionVisualizer


class StabilityTester:
    BAND_NAMES    = ['D1', 'D2', 'D3', 'D4', 'A4']
    METRICS       = ['determinism', 'laminarity', 'trapping_time']
    METRIC_LABELS = {
        'determinism':   'DET',
        'laminarity':    'LAM',
        'trapping_time': 'TT',
    }
    PIPELINES = ['swt_recon', 'dwt_recon', 'swt_direct', 'dwt_direct']
    PIPE_LABELS = {
        'swt_recon':  'SWT-R',
        'dwt_recon':  'DWT-R',
        'swt_direct': 'SWT-D',
        'dwt_direct': 'DWT-D',
    }
    PIPE_COLORS = {
        'swt_recon':  '#2196F3',
        'dwt_recon':  '#F44336',
        'swt_direct': '#4CAF50',
        'dwt_direct': '#FF9800',
    }

    def __init__(self, visualizer, n_trials: int = 20,
                 output_dir: str = 'rossler_stability_output'):
        self.viz        = visualizer
        self.n_trials   = n_trials
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.results = {
            pipe: {band: {met: [] for met in self.METRICS}
                   for band in self.BAND_NAMES}
            for pipe in self.PIPELINES
        }

    # ------------------------------------------------------------------ #
    #  Epsilon estimation                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_epsilon(signal: np.ndarray, tau: int, m: int,
                      target_rr: float = 0.1,
                      max_iters: int = 30, tol: float = 0.005) -> float:
        """
        Binary search for epsilon achieving target recurrence rate.

        Fixed RR ensures DET/LAM/TT differences across trials reflect
        structural RP changes only, not threshold variation.
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

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def _plot_critical_difference(self, df_kw: pd.DataFrame):
        """
        Critical difference diagram (Demšar 2006) for each metric,
        aggregated across D1-D4.

        Pipelines are ranked by mean RQA value (lower rank = lower value).
        A horizontal bar connects pipelines that are NOT significantly
        different after Dunn/Holm correction — anything not connected is
        significantly different.

        Separate panel per metric so the reader sees DET, LAM, TT structure
        at a glance without parsing a table.
        """
        plot_bands = ['D1', 'D2', 'D3', 'D4']
        pipe_labels = [self.PIPE_LABELS[p] for p in self.PIPELINES]
        colors = [self.PIPE_COLORS[p] for p in self.PIPELINES]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            "Critical Difference Diagram — Kruskal-Wallis + Dunn post-hoc (Holm)\n"
            "Pipelines connected by a bar are NOT significantly different  "
            "(α = 0.05, Holm-corrected). D1–D4 pooled.",
            fontsize=12, fontweight='bold'
        )

        for ax, met in zip(axes, self.METRICS):
            # Pool all trial values across D1-D4 per pipeline
            pool = {pipe: [] for pipe in self.PIPELINES}
            for band in plot_bands:
                row = df_kw[(df_kw['band'] == band) & (df_kw['metric'] == met)]
                if row.empty:
                    continue
                for pipe, vals in row.iloc[0]['_groups'].items():
                    pool[pipe].extend(vals.tolist())

            pool_arrays = {pipe: np.array(v) for pipe, v in pool.items()}

            # Mean rank across all observations (used for CD diagram x-position)
            all_vals = np.concatenate(list(pool_arrays.values()))
            all_ranks = len(all_vals) - np.argsort(np.argsort(all_vals))  # descending ranks
            offset = 0
            mean_ranks = {}
            for pipe in self.PIPELINES:
                n = len(pool_arrays[pipe])
                mean_ranks[pipe] = np.mean(all_ranks[offset: offset + n])
                offset += n

            # Sort pipelines by mean rank (ascending = best)
            sorted_pipes = sorted(self.PIPELINES, key=lambda p: mean_ranks[p])
            sorted_labels = [self.PIPE_LABELS[p] for p in sorted_pipes]
            sorted_ranks = [mean_ranks[p] for p in sorted_pipes]
            sorted_colors = [self.PIPE_COLORS[p] for p in sorted_pipes]

            # Run pooled Dunn test on pooled values
            all_pool_vals = np.concatenate([pool_arrays[p] for p in self.PIPELINES])
            group_ids = np.concatenate([
                np.full(len(pool_arrays[p]), self.PIPE_LABELS[p])
                for p in self.PIPELINES
            ])
            from scikit_posthocs import posthoc_dunn
            dunn = posthoc_dunn(
                pd.DataFrame({'val': all_pool_vals, 'group': group_ids}),
                val_col='val', group_col='group', p_adjust='holm'
            )

            # Determine non-significant pairs (connected by bar)
            ns_pairs = []
            pipe_label_list = [self.PIPE_LABELS[p] for p in sorted_pipes]
            for i in range(len(sorted_pipes)):
                for j in range(i + 1, len(sorted_pipes)):
                    la, lb = pipe_label_list[i], pipe_label_list[j]
                    if dunn.loc[la, lb] > 0.05:
                        ns_pairs.append((i, j))

            # ── Draw diagram ──────────────────────────────────────────────
            y_pipe = 0.5  # y-level for pipeline dots and labels
            y_bar = 0.75  # y-level for NS bars
            bar_step = 0.08  # vertical separation between stacked bars

            # Normalise ranks to [0, 1] for x-axis
            r_min, r_max = min(sorted_ranks), max(sorted_ranks)

            def norm(r):
                return (r - r_min) / (r_max - r_min + 1e-9)

            # Pipeline markers and labels
            for k, (pipe, label, rank, color) in enumerate(
                    zip(sorted_pipes, sorted_labels, sorted_ranks, sorted_colors)):
                xp = norm(rank)
                ax.scatter(xp, y_pipe, s=120, color=color,
                           zorder=5, edgecolors='black', linewidths=0.8)
                va = 'top' if k % 2 == 0 else 'bottom'
                ax.text(xp, y_pipe - 0.06 if va == 'top' else y_pipe + 0.06,
                        label, ha='center', va=va, fontsize=10, fontweight='bold',
                        color=color)
                # Thin vertical tick from dot to rank axis
                ax.plot([xp, xp], [y_pipe - 0.03, y_pipe + 0.03],
                        color=color, linewidth=1.5, zorder=4)

            # Rank axis line
            ax.axhline(y_pipe, color='black', linewidth=1.2, zorder=3)

            # NS connection bars
            used_levels: List[Tuple[float, float]] = []
            for pair_idx, (i, j) in enumerate(ns_pairs):
                x1 = norm(sorted_ranks[i])
                x2 = norm(sorted_ranks[j])

                # Find lowest y-level that doesn't overlap existing bars
                level = 0
                while any(not (x2 < lo or x1 > hi)
                          for lo, hi in used_levels[level:level + 1]):
                    level += 1
                y_b = y_bar + level * bar_step
                used_levels.insert(level, (x1, x2))

                ax.plot([x1, x2], [y_b, y_b],
                        color='#555', linewidth=2.5, solid_capstyle='round')
                ax.plot([x1, x1], [y_pipe + 0.03, y_b],
                        color='#555', linewidth=1.2, linestyle=':')
                ax.plot([x2, x2], [y_pipe + 0.03, y_b],
                        color='#555', linewidth=1.2, linestyle=':')

            # Kruskal-Wallis omnibus annotation
            kw_rows = df_kw[df_kw['metric'] == met]
            kw_p_pooled = kw_rows['kw_p'].min()  # smallest p across bands
            sig_str = ('***' if kw_p_pooled < 0.001 else
                       '**' if kw_p_pooled < 0.01 else
                       '*' if kw_p_pooled < 0.05 else 'ns')
            ax.text(0.5, 0.18,
                    f'KW min p = {kw_p_pooled:.2e}  {sig_str}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='#333',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5',
                              edgecolor='#ccc'))

            # Rank axis label
            ax.text(-0.02, y_pipe, 'best', ha='right', va='center',
                    fontsize=8, color='grey')
            ax.text(1.02, y_pipe, 'worst', ha='left', va='center',
                    fontsize=8, color='grey')

            ax.set_title(self.METRIC_LABELS[met], fontweight='bold', fontsize=13)
            ax.set_xlim(-0.15, 1.15)
            ax.set_ylim(0, 1.1)
            ax.axis('off')

        plt.tight_layout()
        self._save_fig('stability_critical_difference.png')

    from scikit_posthocs import posthoc_dunn

    def _kruskal_wallis_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Kruskal-Wallis omnibus test across all four pipelines per (band, metric).

        Tests H0: all pipelines produce identically distributed RQA values.
        Significant result indicates at least one pipeline differs.

        Followed by Dunn post-hoc test with Holm correction for pairwise
        identification. The omnibus test controls the family-wise error rate
        for the global comparison without requiring 1-to-1 pre-specification.
        """


        rows = []
        for band in self.BAND_NAMES:
            for met in self.METRICS:
                groups = {
                    pipe: np.array(self.results[pipe][band][met])
                    for pipe in self.PIPELINES
                }
                if any(len(v) == 0 for v in groups.values()):
                    continue

                # Omnibus test
                stat, p = kruskal(*groups.values())

                # Dunn post-hoc (Holm correction within each band-metric)
                all_vals = np.concatenate(list(groups.values()))
                group_ids = np.concatenate([
                    np.full(len(v), pipe)
                    for pipe, v in groups.items()
                ])
                dunn = posthoc_dunn(
                    pd.DataFrame({'val': all_vals, 'group': group_ids}),
                    val_col='val', group_col='group', p_adjust='holm'
                )

                rows.append({
                    'band': band,
                    'metric': met,
                    'kw_stat': stat,
                    'kw_p': p,
                    'dunn': dunn,  # stored for plotting
                    '_groups': groups,  # raw arrays for CD diagram
                })

        df_kw = pd.DataFrame(rows)

        # Holm correction on Kruskal-Wallis p-values across all (band, metric) pairs
        p_vals = df_kw['kw_p'].values
        n = len(p_vals)
        order = np.argsort(p_vals)
        alphas = np.zeros(n)
        for rank, idx in enumerate(order):
            alphas[idx] = 0.05 / (n - rank)
        df_kw['alpha_holm'] = alphas
        df_kw['significant_holm'] = df_kw['kw_p'] < df_kw['alpha_holm']

        # Save CSV (drop non-serialisable columns)
        save_cols = [c for c in df_kw.columns if not c.startswith('_') and c != 'dunn']
        df_kw[save_cols].to_csv(
            os.path.join(self.output_dir, 'kruskal_wallis.csv'), index=False
        )
        return df_kw

    def run_stability_analysis(self, system_type: str = 'rossler'):
        print(f"\n{'='*70}")
        print(f"STABILITY TEST: {self.n_trials} realisations — {system_type.upper()}")
        print(f"{'='*70}")

        if system_type == 'rossler':
            ref_signal = self.viz._generate_rossler(seed=0)
        elif system_type == 'lorenz':
            ref_signal = self.viz.generate_lorenz_system(seed=0)
        else:
            raise ValueError(f"Unknown system: {system_type}")

        fixed_params = self._estimate_params_per_pipeline(ref_signal)

        for trial in range(self.n_trials):
            if system_type == 'rossler':
                signal = self.viz._generate_rossler(seed=trial + 100)
            else:
                signal = self.viz.generate_lorenz_system(seed=trial + 100)

            for pipe in self.PIPELINES:
                subbands = self.viz.decompose_signal(signal, pipe)
                params   = fixed_params[pipe]

                for band in self.BAND_NAMES:
                    tau, m  = params[band]['tau'], params[band]['m']
                    trimmed = self.viz._trim_signal(subbands[band])

                    if len(trimmed) < 10:
                        continue
                    std = trimmed.std()
                    if std < 1e-9:
                        continue

                    normed = (trimmed - trimmed.mean()) / std
                    eps    = self._find_epsilon(normed, tau, m, target_rr=0.1)
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

    # ------------------------------------------------------------------ #
    #  Embedding parameter estimation                                      #
    # ------------------------------------------------------------------ #

    def _estimate_params_per_pipeline(self, ref_signal: np.ndarray) -> dict:
        fixed_params = {}
        for pipe in self.PIPELINES:
            fixed_params[pipe] = {}
            subbands = self.viz.decompose_signal(ref_signal, pipe)
            for band in self.BAND_NAMES:
                trimmed = self.viz._trim_signal(subbands[band])
                if len(trimmed) < 10:
                    fixed_params[pipe][band] = {'tau': 1, 'm': 2}
                    continue
                std = trimmed.std()
                if std < 1e-9:
                    fixed_params[pipe][band] = {'tau': 1, 'm': 2}
                    continue
                normed = (trimmed - trimmed.mean()) / std
                tau, m = self.viz.get_optimal_parameters(normed)
                fixed_params[pipe][band] = {'tau': tau, 'm': m}
                print(f"  [{pipe}] {band}: tau={tau}, m={m}")
        return fixed_params

    # ------------------------------------------------------------------ #
    #  Data persistence                                                    #
    # ------------------------------------------------------------------ #

    def _save_raw_results(self):
        rows = []
        for pipe in self.PIPELINES:
            for band in self.BAND_NAMES:
                for met in self.METRICS:
                    for trial_idx, val in enumerate(self.results[pipe][band][met]):
                        rows.append({
                            'pipeline': pipe, 'band': band,
                            'metric': met, 'trial': trial_idx, 'value': val,
                        })
        path = os.path.join(self.output_dir, 'stability_raw.csv')
        pd.DataFrame(rows).to_csv(path, index=False)

    # ------------------------------------------------------------------ #
    #  Statistics                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _robust_stats(vals: np.ndarray):
        """
        Return (median, IQR, MAD, CV%).

        MAD is scaled by 1.4826 to approximate the standard deviation
        under normality, making it comparable across metrics.

        CV = MAD / |median| × 100 — expresses instability as a percentage
        of the central value, allowing comparison across metrics with very
        different scales (DET ≈ 0.99 vs TT ≈ 20).
        """
        median  = np.median(vals)
        q1, q3  = np.percentile(vals, [25, 75])
        iqr     = q3 - q1
        mad     = 1.4826 * np.median(np.abs(vals - median))
        cv      = (mad / abs(median) * 100) if abs(median) > 1e-9 else np.nan
        return median, iqr, mad, cv

    def _statistical_comparison(self) -> pd.DataFrame:
        rows = []

        for band in self.BAND_NAMES:
            for met in self.METRICS:
                swt_vals        = np.array(self.results['swt_recon'][band][met])
                dwt_vals        = np.array(self.results['dwt_recon'][band][met])
                swt_direct_vals = np.array(self.results['swt_direct'][band][met])
                dwt_direct_vals = np.array(self.results['dwt_direct'][band][met])

                if any(len(v) == 0 for v in
                       [swt_vals, dwt_vals, swt_direct_vals, dwt_direct_vals]):
                    continue

                swt_med,  swt_iqr,  swt_mad,  swt_cv  = self._robust_stats(swt_vals)
                dwt_med,  dwt_iqr,  dwt_mad,  dwt_cv  = self._robust_stats(dwt_vals)
                swtd_med, swtd_iqr, swtd_mad, swtd_cv = self._robust_stats(swt_direct_vals)
                dwtd_med, dwtd_iqr, dwtd_mad, dwtd_cv = self._robust_stats(dwt_direct_vals)

                # Ratios relative to SWT-Recon baseline
                def _ratio(num, denom):
                    return num / denom if (denom and denom > 1e-9) else np.nan

                mad_ratio_dwt        = _ratio(dwt_mad,  swt_mad)
                mad_ratio_swt_direct = _ratio(swtd_mad, swt_mad)
                mad_ratio_dwt_direct = _ratio(dwtd_mad, swt_mad)

                cv_ratio_dwt         = _ratio(dwt_cv,  swt_cv)
                cv_ratio_swt_direct  = _ratio(swtd_cv, swt_cv)
                cv_ratio_dwt_direct  = _ratio(dwtd_cv, swt_cv)

                # Levene test: equality of variances across all pipelines
                lev_stat, lev_p = levene(swt_vals, dwt_vals, swt_direct_vals)

                def ps(u, n1, n2):
                    return u / (n1 * n2)

                u_dwt,   p_dwt   = mannwhitneyu(swt_vals, dwt_vals,        alternative='two-sided')
                u_swt_d, p_swt_d = mannwhitneyu(swt_vals, swt_direct_vals, alternative='two-sided')
                u_dwt_d, p_dwt_d = mannwhitneyu(swt_vals, dwt_direct_vals, alternative='two-sided')

                rows.append({
                    'band': band, 'metric': met,
                    # Median ± IQR
                    'swt_median': swt_med,   'swt_iqr': swt_iqr,
                    'dwt_median': dwt_med,   'dwt_iqr': dwt_iqr,
                    'swt_direct_median': swtd_med, 'swt_direct_iqr': swtd_iqr,
                    'dwt_direct_median': dwtd_med, 'dwt_direct_iqr': dwtd_iqr,
                    # MAD
                    'swt_mad': swt_mad,   'dwt_mad': dwt_mad,
                    'swt_direct_mad': swtd_mad, 'dwt_direct_mad': dwtd_mad,
                    # CV (%)
                    'swt_cv': swt_cv,    'dwt_cv': dwt_cv,
                    'swt_direct_cv': swtd_cv, 'dwt_direct_cv': dwtd_cv,
                    # MAD ratios
                    'mad_ratio_dwt': mad_ratio_dwt,
                    'mad_ratio_swt_direct': mad_ratio_swt_direct,
                    'mad_ratio_dwt_direct': mad_ratio_dwt_direct,
                    # CV ratios
                    'cv_ratio_dwt': cv_ratio_dwt,
                    'cv_ratio_swt_direct': cv_ratio_swt_direct,
                    'cv_ratio_dwt_direct': cv_ratio_dwt_direct,
                    # Levene
                    'levene_stat': lev_stat, 'levene_p': lev_p,
                    # Mann-Whitney + probability of superiority
                    'u_dwt': u_dwt, 'p_dwt': p_dwt,
                    'ps_dwt': ps(u_dwt, len(swt_vals), len(dwt_vals)),
                    'u_swt_direct': u_swt_d, 'p_swt_direct': p_swt_d,
                    'ps_swt_direct': ps(u_swt_d, len(swt_vals), len(swt_direct_vals)),
                    'u_dwt_direct': u_dwt_d, 'p_dwt_direct': p_dwt_d,
                    'ps_dwt_direct': ps(u_dwt_d, len(swt_vals), len(dwt_direct_vals)),
                    # Raw arrays for plots
                    '_swt_vals': swt_vals,   '_dwt_vals': dwt_vals,
                    '_swt_direct_vals': swt_direct_vals,
                    '_dwt_direct_vals': dwt_direct_vals,
                    '_swt_devs':  np.abs(swt_vals  - swt_med),
                    '_dwt_devs':  np.abs(dwt_vals  - dwt_med),
                    '_swt_direct_devs': np.abs(swt_direct_vals - swtd_med),
                    '_dwt_direct_devs': np.abs(dwt_direct_vals - dwtd_med),
                })

        df = pd.DataFrame(rows)

        # Holm correction applied to Levene p-values
        p_vals = df['levene_p'].values
        n      = len(p_vals)
        order  = np.argsort(p_vals)
        alphas = np.zeros(n)
        for rank, idx in enumerate(order):
            alphas[idx] = 0.05 / (n - rank)
        df['alpha_holm']       = alphas
        df['significant_holm'] = df['levene_p'] < df['alpha_holm']

        stat_cols = [c for c in df.columns if not c.startswith('_')]
        df[stat_cols].to_csv(
            os.path.join(self.output_dir, 'stability_stats.csv'), index=False
        )
        print(f"Statistics saved → {self.output_dir}/stability_stats.csv")
        return df

    # ------------------------------------------------------------------ #
    #  Plot dispatcher                                                     #
    # ------------------------------------------------------------------ #

    def _plot_distributions(self, df: pd.DataFrame):
        self._plot_raw_violins(df)
        self._plot_deviation_violins(df)
        self._plot_mad_bars(df)
        self._plot_cv_summary(df)
        self._plot_cv_ratio_heatmap(df)
        self._plot_probability_superiority_heatmap(df)
        # Kruskal-Wallis + CD diagram
        df_kw = self._kruskal_wallis_comparison(df)
        self._plot_critical_difference(df_kw)

    # ------------------------------------------------------------------ #
    #  Existing plots                                                      #
    # ------------------------------------------------------------------ #

    def _plot_raw_violins(self, df: pd.DataFrame):
        plot_bands = ['D1', 'D2', 'D3', 'D4']
        fig, axes  = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "Raw RQA distributions across realisations — D1–D4 pooled\n"
            "Each observation = one trial × one subband.",
            fontsize=12, fontweight='bold'
        )
        colors = [self.PIPE_COLORS[p] for p in self.PIPELINES]
        labels = [self.PIPE_LABELS[p] for p in self.PIPELINES]

        for ax, met in zip(axes, self.METRICS):
            pools = [[] for _ in self.PIPELINES]
            for band in plot_bands:
                row = df[(df['band'] == band) & (df['metric'] == met)]
                if row.empty:
                    continue
                r = row.iloc[0]
                for i, key in enumerate(['_swt_vals', '_dwt_vals',
                                         '_swt_direct_vals', '_dwt_direct_vals']):
                    pools[i].extend(r[key].tolist())

            parts = ax.violinplot(pools, positions=range(1, 5),
                                  showmedians=True, showextrema=True)
            for body, color in zip(parts['bodies'], colors):
                body.set_facecolor(color); body.set_alpha(0.7)
            for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
                if key in parts:
                    parts[key].set_color('black'); parts[key].set_linewidth(1.2)

            ax.set_xticks(range(1, 5)); ax.set_xticklabels(labels, fontsize=9)
            ax.set_title(self.METRIC_LABELS[met], fontweight='bold')
            ax.set_ylabel('RQA value'); ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        self._save_fig('stability_raw_violins.png')

    def _plot_deviation_violins(self, df: pd.DataFrame):
        plot_bands = ['D1', 'D2', 'D3', 'D4']
        fig, axes  = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "Absolute deviation from pipeline median — D1–D4 pooled\n"
            "Narrower = more consistent across realisations = better pipeline.",
            fontsize=12, fontweight='bold'
        )
        colors = [self.PIPE_COLORS[p] for p in self.PIPELINES]
        labels = [self.PIPE_LABELS[p] for p in self.PIPELINES]

        for ax, met in zip(axes, self.METRICS):
            pools = [[] for _ in self.PIPELINES]
            for band in plot_bands:
                row = df[(df['band'] == band) & (df['metric'] == met)]
                if row.empty:
                    continue
                r = row.iloc[0]
                for i, key in enumerate(['_swt_devs', '_dwt_devs',
                                         '_swt_direct_devs', '_dwt_direct_devs']):
                    pools[i].extend(r[key].tolist())

            parts = ax.violinplot(pools, positions=range(1, 5),
                                  showmedians=True, showextrema=True)
            for body, color in zip(parts['bodies'], colors):
                body.set_facecolor(color); body.set_alpha(0.7)
            for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
                if key in parts:
                    parts[key].set_color('black'); parts[key].set_linewidth(1.2)

            ax.set_xticks(range(1, 5)); ax.set_xticklabels(labels, fontsize=9)
            ax.set_title(self.METRIC_LABELS[met], fontweight='bold')
            ax.set_ylabel('|value − pipeline median|'); ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        self._save_fig('stability_deviation_violins.png')

    def _plot_mad_bars(self, df: pd.DataFrame):
        plot_bands = ['D1', 'D2', 'D3', 'D4']
        n_bands    = len(plot_bands)
        fig, axes  = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
        fig.suptitle(
            "MAD per subband — all pipelines\n"
            "Lower = more consistent RQA estimates across realisations.",
            fontsize=12, fontweight='bold'
        )
        colors = [self.PIPE_COLORS[p] for p in self.PIPELINES]
        labels = [self.PIPE_LABELS[p] for p in self.PIPELINES]
        x      = np.arange(n_bands)
        width  = 0.2

        for ax, met in zip(axes, self.METRICS):
            mads = [[] for _ in self.PIPELINES]
            for band in plot_bands:
                row = df[(df['band'] == band) & (df['metric'] == met)]
                if row.empty:
                    for m in mads: m.append(0)
                    continue
                r = row.iloc[0]
                for i, key in enumerate(
                        ['swt_mad', 'dwt_mad', 'swt_direct_mad', 'dwt_direct_mad']):
                    mads[i].append(r[key])

            for i, (mad_list, color, label) in enumerate(zip(mads, colors, labels)):
                ax.bar(x + (i - 1.5) * width, mad_list, width,
                       label=label, color=color, alpha=0.85, edgecolor='white')

            ax.set_xticks(x); ax.set_xticklabels(plot_bands)
            ax.set_title(self.METRIC_LABELS[met], fontweight='bold')
            ax.set_ylabel('MAD'); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        self._save_fig('stability_mad_bars.png')

    # ------------------------------------------------------------------ #
    #  NEW: Aggregate CV summary                                           #
    # ------------------------------------------------------------------ #

    def _plot_cv_summary(self, df: pd.DataFrame):
        """
        Grouped bar chart: mean CV (%) per pipeline per metric,
        averaged across D1–D4.

        CV = MAD / |median| × 100 is scale-free, making DET, LAM, and TT
        directly comparable on the same axis despite their very different
        absolute magnitudes. A single bar per pipeline per metric summarises
        the instability picture without requiring the reader to parse five
        separate subband panels.

        Error bars show the standard deviation of CV across subbands,
        indicating whether instability is uniform across frequency bands
        or concentrated in specific ones.
        """
        plot_bands  = ['D1', 'D2', 'D3', 'D4']
        cv_keys     = ['swt_cv', 'dwt_cv', 'swt_direct_cv', 'dwt_direct_cv']
        pipe_labels = [self.PIPE_LABELS[p] for p in self.PIPELINES]
        colors      = [self.PIPE_COLORS[p]  for p in self.PIPELINES]

        # mean and std of CV across subbands — shape (n_pipelines, n_metrics)
        mean_cv = np.full((len(self.PIPELINES), len(self.METRICS)), np.nan)
        std_cv  = np.full_like(mean_cv, np.nan)

        for j, met in enumerate(self.METRICS):
            for i, key in enumerate(cv_keys):
                vals = []
                for band in plot_bands:
                    row = df[(df['band'] == band) & (df['metric'] == met)]
                    if not row.empty:
                        v = row.iloc[0][key]
                        if not np.isnan(v):
                            vals.append(v)
                if vals:
                    mean_cv[i, j] = np.mean(vals)
                    std_cv[i, j]  = np.std(vals)

        fig, ax = plt.subplots(figsize=(11, 6))
        x     = np.arange(len(self.METRICS))
        width = 0.18

        for i, (label, color) in enumerate(zip(pipe_labels, colors)):
            offset = (i - 1.5) * width
            bars   = ax.bar(
                x + offset, mean_cv[i], width,
                yerr=std_cv[i], capsize=4,
                label=label, color=color,
                alpha=0.88, edgecolor='white',
                error_kw={'elinewidth': 1.2, 'ecolor': '#444'},
            )
            # Annotate bar tops with the numeric value
            for bar, val, err in zip(bars, mean_cv[i], std_cv[i]):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (err if not np.isnan(err) else 0) + 0.3,
                        f'{val:.1f}%',
                        ha='center', va='bottom', fontsize=7.5, color='#333',
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([self.METRIC_LABELS[m] for m in self.METRICS], fontsize=12)
        ax.set_ylabel('Mean CV (%) across D1–D4   [lower = more stable]', fontsize=11)
        ax.set_title(
            'Coefficient of Variation per pipeline — D1–D4 averaged\n'
            'CV = MAD / |median| × 100   (robust, scale-free instability measure)\n'
            'Error bars = std of CV across subbands',
            fontsize=11, fontweight='bold',
        )
        ax.legend(title='Pipeline', fontsize=10, title_fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        plt.tight_layout()
        self._save_fig('stability_cv_summary.png')

    # ------------------------------------------------------------------ #
    #  NEW: CV-ratio heatmap                                               #
    # ------------------------------------------------------------------ #

    def _plot_cv_ratio_heatmap(self, df: pd.DataFrame):
        """
        Heatmap of CV ratio (pipeline / SWT-Recon) per subband × metric.

        Each cell shows how many times more variable a pipeline is than
        SWT-Recon. A ratio of 1 = identical stability; 10 = ten times
        worse. Log colour scale prevents large outlier ratios from washing
        out the structure for the more stable comparisons.

        This figure is the key result for Experiment 1: it shows where
        (which subbands and metrics) DWT aliasing causes instability,
        making the case for SWT reconstruction spatial and metric-specific
        rather than a single aggregate number.
        """
        plot_bands  = ['D1', 'D2', 'D3', 'D4', 'A4']
        comparisons = [
            ('cv_ratio_dwt',        'DWT-R / SWT-R'),
            ('cv_ratio_swt_direct', 'SWT-D / SWT-R'),
            ('cv_ratio_dwt_direct', 'DWT-D / SWT-R'),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            'CV ratio relative to SWT-Recon baseline  (log colour scale)\n'
            'Ratio annotation = linear scale.  '
            'Ratio = 1 → identical stability to SWT-Recon.',
            fontsize=12, fontweight='bold',
        )

        for ax, (col, title) in zip(axes, comparisons):
            matrix = np.full((len(plot_bands), len(self.METRICS)), np.nan)
            for i, band in enumerate(plot_bands):
                for j, met in enumerate(self.METRICS):
                    row = df[(df['band'] == band) & (df['metric'] == met)]
                    if not row.empty:
                        matrix[i, j] = row.iloc[0][col]

            # Log10 transform for colour mapping
            log_matrix = np.log10(np.clip(matrix, 1e-3, None))
            vmax = max(np.nanmax(np.abs(log_matrix)), 0.5)

            im = ax.imshow(log_matrix, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, aspect='auto')

            # Annotate with raw (linear) ratio values
            for i in range(len(plot_bands)):
                for j in range(len(self.METRICS)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        lv  = log_matrix[i, j]
                        txt = 'white' if abs(lv) > vmax * 0.6 else 'black'
                        ax.text(j, i, f'{val:.1f}×',
                                ha='center', va='center',
                                color=txt, fontsize=10, fontweight='bold')

            ax.set_xticks(range(len(self.METRICS)))
            ax.set_yticks(range(len(plot_bands)))
            ax.set_xticklabels([self.METRIC_LABELS[m] for m in self.METRICS])
            ax.set_yticklabels(plot_bands)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('RQA Metric')
            ax.set_ylabel('Subband')

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('log₁₀(CV ratio)', rotation=270, labelpad=18)
            tick_vals = np.linspace(-vmax, vmax, 5)
            cbar.set_ticks(tick_vals)
            cbar.set_ticklabels([f'{10**v:.1f}×' for v in tick_vals])

        plt.tight_layout()
        self._save_fig('stability_cv_ratio_heatmap.png')

    # ------------------------------------------------------------------ #
    #  Existing: probability of superiority heatmap                        #
    # ------------------------------------------------------------------ #

    def _plot_probability_superiority_heatmap(self, df: pd.DataFrame):
        plot_bands  = ['D1', 'D2', 'D3', 'D4']
        fig, axes   = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(
            "Probability of Superiority: P(SWT-R > Other) from Mann-Whitney U\n"
            "Values > 0.5 → SWT-R tends higher; < 0.5 → SWT-R tends lower.",
            fontsize=12, fontweight='bold',
        )
        comparisons = [
            ('ps_dwt',        'DWT-R vs SWT-R'),
            ('ps_swt_direct', 'SWT-D vs SWT-R'),
            ('ps_dwt_direct', 'DWT-D vs SWT-R'),
        ]

        for ax, (ps_col, title) in zip(axes, comparisons):
            matrix = np.full((len(plot_bands), len(self.METRICS)), np.nan)
            for i, band in enumerate(plot_bands):
                for j, met in enumerate(self.METRICS):
                    row = df[(df['band'] == band) & (df['metric'] == met)]
                    if not row.empty:
                        matrix[i, j] = row.iloc[0][ps_col]

            im = ax.imshow(matrix, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
            for i in range(len(plot_bands)):
                for j in range(len(self.METRICS)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        txt = 'white' if (val < 0.3 or val > 0.7) else 'black'
                        ax.text(j, i, f'{val:.2f}',
                                ha='center', va='center',
                                color=txt, fontsize=10, fontweight='bold')

            ax.set_xticks(range(len(self.METRICS)))
            ax.set_yticks(range(len(plot_bands)))
            ax.set_xticklabels([self.METRIC_LABELS[m] for m in self.METRICS])
            ax.set_yticklabels(plot_bands)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('RQA Metric')
            ax.set_ylabel('Subband')

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('P(SWT-R > Other)', rotation=270, labelpad=20)

        plt.tight_layout()
        self._save_fig('stability_prob_superiority_heatmap.png')

    # ------------------------------------------------------------------ #
    #  Helper                                                              #
    # ------------------------------------------------------------------ #

    def _save_fig(self, filename: str):
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    system = 'lorenz'
    # system = 'rossler'

    viz  = DecompositionVisualizer(seed=42)
    stab = StabilityTester(viz, n_trials=100,
                           output_dir=f'{system}_stability_output')
    stab.run_stability_analysis(system_type=system)