import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

from src.experiment1.subband_calculator import SubbandCalculator
from src.experiment1.subband_plotter import SubbandPlotter


class ExperimentRunner:
    def __init__(self, is_sine=True):
        self.calculator = SubbandCalculator(is_sine=is_sine)
        self.plotter = SubbandPlotter(
            self.calculator.OUTPUT_DIR,
            self.calculator.BAND_NAMES,
            self.calculator.SUBBAND_FREQS,
            self.calculator.FS
        )

    def run_full_experiment(self, n_trials=100, include_leakage=True, include_aliasing=True):
        ref_path = os.path.join(self.calculator.OUTPUT_DIR, 'reference_rqa_values.json')
        if not os.path.exists(ref_path):
            ref_rqa = self.calculator.generate_reference_rqa_values(ref_path)
        else:
            ref_rqa = self.calculator.load_reference_rqa_values(ref_path)

        ref_data = self.calculator.load_reference_data(ref_path.replace('.json', '_data.npz'))
        results = {
            p: {
                m: [] for m in ['det_deviation', 'lam_deviation', 'tt_deviation',
                               'hamming_distance', 'ibf', 'rp_pattern_corr', 'frobenius_distance']
            } for p in self.calculator.PIPELINES
        }
        all_trial_data = []

        for trial in range(n_trials):
            if (trial + 1) % 10 == 0:
                print(f'  Trial {trial + 1}/{n_trials}')

            sig = self.calculator.generate_composite_sine(random_phase=True)

            for pipe in self.calculator.PIPELINES:
                df, _ = self.calculator.validate_pipeline(sig, pipe, ref_rqa, ref_data)

                all_trial_data.append(df)

                results[pipe]['det_deviation'].append(df['det_deviation'].mean())
                results[pipe]['lam_deviation'].append(df['lam_deviation'].mean())
                results[pipe]['tt_deviation'].append(df['tt_deviation'].mean())
                results[pipe]['hamming_distance'].append(df['hamming_distance'].mean())
                results[pipe]['ibf'].append(df['ibf'].mean())
                results[pipe]['rp_pattern_corr'].append(df['rp_pattern_corr'].mean())
                results[pipe]['frobenius_distance'].append(df['frobenius_distance'].mean())

        for pipeline in self.calculator.PIPELINES:
            for metric in results[pipeline].keys():
                results[pipeline][metric] = np.array(results[pipeline][metric])

        master_df = pd.concat(all_trial_data, ignore_index=True)


        df_inv = None
        if include_aliasing:
            df_inv = self.calculator.run_shift_invariance_analysis(n_trials=n_trials)

        df_leakage = None
        if include_leakage:
            df_leakage = self.calculator.run_leakage_analysis(n_trials=n_trials)
            print("Leakage analysis complete.\n")
        df_der, df_alias_summary, df_stats = None, None, None
        if include_aliasing and include_leakage:
            df_der, df_alias_summary, df_stats = self.calculator.analyze_aliasing_and_leakage(df_inv, df_leakage)
        elif include_leakage and df_leakage is not None:
            df_der_list = []
            for (pipe, trial), sub in df_leakage.groupby(['pipeline', 'trial']):
                diag = sub[sub['input_sine'] == sub['output_subband']]['energy_fraction'].sum()
                total = sub['energy_fraction'].sum()
                df_der_list.append({
                    'pipeline': pipe,
                    'trial': trial,
                    'DER': diag / total
                })
            df_der = pd.DataFrame(df_der_list)
        master_csv_path = os.path.join(self.calculator.OUTPUT_DIR, 'master_results.csv')
        master_df.to_csv(master_csv_path, index=False)

        if include_leakage and df_leakage is not None:
            self.plotter.plot_leakage_matrices(df_leakage)
            if df_der is not None:
                self.plotter.plot_deviation_leakage(df_der, self.plotter.output_dir)

        if include_aliasing and df_inv is not None:
            self.plotter.visualize_shift_invariance(df_inv)

        self.run_statistical_comparison(n_trials)

        self.plotter.visualize_absolute_metrics(master_df)

        self.plotter.visualize_aggregated_structural_metrics_violin(master_df)

        self.plotter.visualize_bar_comparison_pro(results, master_df, ref_rqa)

        self.plotter.visualize_frobenius_distances(master_df)

        summary_df = self.summarize_results(results)
        summary_df.to_csv(os.path.join(self.calculator.OUTPUT_DIR, 'summary_metrics.csv'), index=False)

        disc_df = self.compute_metric_discriminative_power(results)
        disc_df.to_csv(os.path.join(self.calculator.OUTPUT_DIR, 'metric_discriminative_power.csv'), index=False)


        return results, master_df, ref_rqa

    def run_statistical_comparison(self, n_trials=100):
        reference_rqa = None
        reference_data = None

        ref_path = os.path.join(self.calculator.OUTPUT_DIR, 'reference_rqa_values.json')
        ref_data_path = os.path.join(self.calculator.OUTPUT_DIR, 'reference_rqa_values_data.npz')

        if os.path.exists(ref_path):
            print("Loading saved reference rqa values")
            reference_rqa = self.calculator.load_reference_rqa_values(ref_path)
            if os.path.exists(ref_data_path):
                reference_data = self.calculator.load_reference_data(ref_data_path)
        else:
            print("No reference rqa data_old found, generating...")
            reference_rqa = self.calculator.generate_reference_rqa_values(ref_path)
            reference_data = self.calculator.load_reference_data(ref_data_path)
        print()

        results = {
            'swt_recon': {'det_deviation': [], 'lam_deviation': [], 'tt_deviation': [],
                         'hamming_distance': [], 'ibf': [], 'rp_pattern_corr': [], 'frobenius_distance': []},
            'swt_direct': {'det_deviation': [], 'lam_deviation': [], 'tt_deviation': [],
                          'hamming_distance': [], 'ibf': [], 'rp_pattern_corr': [], 'frobenius_distance': []},
            'dwt_recon': {'det_deviation': [], 'lam_deviation': [], 'tt_deviation': [],
                         'hamming_distance': [], 'ibf': [], 'rp_pattern_corr': [], 'frobenius_distance': []},
            'dwt_direct': {'det_deviation': [], 'lam_deviation': [], 'tt_deviation': [],
                          'hamming_distance': [], 'ibf': [], 'rp_pattern_corr': [], 'frobenius_distance': []},
        }

        pipelines = ['swt_recon', 'swt_direct', 'dwt_recon', 'dwt_direct']

        all_trial_data = []

        METRICS = ['det_deviation', 'lam_deviation', 'tt_deviation',
                   'hamming_distance', 'ibf', 'rp_pattern_corr', 'frobenius_distance']

        # One row per (trial, pipeline) — no band-level rows here
        aggregated_rows = []

        for trial in range(n_trials):
            if (trial + 1) % 10 == 0:
                print(f"  Trial {trial + 1}/{n_trials}...")

            signal = self.calculator.generate_composite_sine(random_phase=True)

            for pipeline in self.calculator.PIPELINES:
                # Single call per pipeline per trial
                df, _ = self.calculator.validate_pipeline(
                    signal, pipeline,
                    reference_rqa=reference_rqa,
                    reference_data=reference_data
                )

                # Aggregate across bands — one observation per (trial, pipeline)
                row = {'trial': trial, 'pipeline': pipeline}
                for metric in METRICS:
                    if metric in df.columns:
                        row[metric] = df[metric].mean()
                aggregated_rows.append(row)

        agg_df = pd.DataFrame(aggregated_rows)

        # Statistical comparison: across n_trials observations per pipeline
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
            for metric in METRICS:
                x = agg_df[agg_df['pipeline'] == pipe1][metric].dropna()
                y = agg_df[agg_df['pipeline'] == pipe2][metric].dropna()
                if len(x) < 3 or len(y) < 3:
                    continue

                u, p = mannwhitneyu(x, y, alternative='two-sided')
                pooled = np.sqrt(((len(x) - 1) * x.var(ddof=1) +
                                  (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
                d = (x.mean() - y.mean()) / pooled if pooled > 1e-9 else np.nan

                stats_results.append({
                    'comparison': label, 'metric': metric,
                    'pipeline1': pipe1, 'pipeline2': pipe2,
                    'mean1': x.mean(), 'std1': x.std(),
                    'mean2': y.mean(), 'std2': y.std(),
                    'u_stat': u, 'p_value': p, 'cohens_d': d,
                    'significance': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                })

        stats_df = pd.DataFrame(stats_results)
        agg_df.to_csv(os.path.join(self.calculator.OUTPUT_DIR, 'aggregated_trial_metrics.csv'), index=False)
        stats_df.to_csv(os.path.join(self.calculator.OUTPUT_DIR, 'stats_final.csv'), index=False)
        return agg_df, stats_df

    def summarize_results(self, results):
        summary = []

        for pipe, metrics in results.items():
            for metric, values in metrics.items():
                values = np.array(values)

                summary.append({
                    'pipeline': pipe,
                    'metric': metric,
                    'mean': values.mean(),
                    'std': values.std()
                })

        return pd.DataFrame(summary)

    def compute_metric_discriminative_power(self, results):
        from scipy.stats import kruskal

        metrics = list(next(iter(results.values())).keys())
        pipelines = list(results.keys())

        rows = []

        for metric in metrics:
            data = [results[p][metric] for p in pipelines]

            stat, p = kruskal(*data)

            rows.append({
                'metric': metric,
                'kruskal_stat': stat,
                'p_value': p
            })

        return pd.DataFrame(rows).sort_values('p_value')

def main():
    runner = ExperimentRunner(is_sine=True)

    results, master_df, ref_rqa = runner.run_full_experiment(
        n_trials=100,
        include_leakage=True,
        include_aliasing=True
    )

if __name__ == '__main__':
    main()
