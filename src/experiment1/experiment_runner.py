import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

from src.experiment1.subband_calculator import SubbandCalculator
from src.experiment1.subband_plotter import SubbandPlotter


class ExperimentRunner:
    """
    Runner class that orchestrates full experiments using SubbandCalculator and SubbandPlotter.
    Combines computation and visualization to run complete validation experiments.
    """

    def __init__(self, is_sine=True, is_lorenz=False):
        """
        Initialize the experiment runner.

        Args:
            is_sine: Whether to use sine wave composite signal (True) or Lorenz system (False)
            is_lorenz: Whether to use Lorenz system (opposite of is_sine typically)
        """
        self.calculator = SubbandCalculator(is_sine=is_sine, is_lorenz=is_lorenz)
        self.plotter = SubbandPlotter(
            self.calculator.OUTPUT_DIR,
            self.calculator.BAND_NAMES,
            self.calculator.SUBBAND_FREQS,
            self.calculator.FS
        )

    def run_full_experiment(self, n_trials=100, include_leakage=True, include_aliasing=True):
        """
        Run the full experiment with all tests and visualizations.

        Args:
            n_trials: Number of trials to run for statistical validation
            include_leakage: Whether to run leakage analysis
            include_aliasing: Whether to run shift invariance (aliasing) analysis

        Returns:
            Tuple of (results dict, master_df, reference_rqa)
        """
        print("="*80)
        print("STARTING FULL EXPERIMENT")
        print("="*80)

        # 1. Generate or load reference RQA values
        ref_path = os.path.join(self.calculator.OUTPUT_DIR, 'reference_rqa_values.json')
        if not os.path.exists(ref_path):
            print("\n[1/7] Generating reference RQA values...")
            ref_rqa = self.calculator.generate_reference_rqa_values(ref_path)
        else:
            print("\n[1/7] Loading reference RQA values...")
            ref_rqa = self.calculator.load_reference_rqa_values(ref_path)

        ref_data = self.calculator.load_reference_data(ref_path.replace('.json', '_data.npz'))
        print("Reference data loaded.\n")

        # 2. Initialize results storage with proper naming (deviations named as deviations)
        results = {
            p: {
                m: [] for m in ['det_deviation', 'lam_deviation', 'tt_deviation',
                               'hamming_distance', 'ibf', 'rp_pattern_corr', 'frobenius_distance']
            } for p in self.calculator.PIPELINES
        }
        all_trial_data = []

        # 3. Run validation trials
        print(f"[2/7] Running {n_trials} validation trials...")
        for trial in range(n_trials):
            if (trial + 1) % 10 == 0:
                print(f'  Trial {trial + 1}/{n_trials}')

            sig = self.calculator.generate_composite_sine(random_phase=True)

            for pipe in self.calculator.PIPELINES:
                df, _ = self.calculator.validate_pipeline(sig, pipe, ref_rqa, ref_data)
                all_trial_data.append(df)

                # Collect AVERAGED metrics per trial (average across all 5 bands)
                # Store with proper descriptive names
                results[pipe]['det_deviation'].append(df['det_deviation'].mean())
                results[pipe]['lam_deviation'].append(df['lam_deviation'].mean())
                results[pipe]['tt_deviation'].append(df['tt_deviation'].mean())
                results[pipe]['hamming_distance'].append(df['hamming_distance'].mean())
                results[pipe]['ibf'].append(df['ibf'].mean())
                results[pipe]['rp_pattern_corr'].append(df['rp_pattern_corr'].mean())
                results[pipe]['frobenius_distance'].append(df['frobenius_distance'].mean())

        # Convert lists to numpy arrays
        for pipeline in self.calculator.PIPELINES:
            for metric in results[pipeline].keys():
                results[pipeline][metric] = np.array(results[pipeline][metric])

        master_df = pd.concat(all_trial_data, ignore_index=True)

        print("Validation trials complete.\n")

        # 4. Run shift invariance analysis (aliasing tests)
        df_inv = None
        if include_aliasing:
            print(f"[3/7] Running shift invariance analysis ({n_trials} trials)...")
            df_inv = self.calculator.run_shift_invariance_analysis(n_trials=n_trials)
            print("Shift invariance analysis complete.\n")
        else:
            print("[3/7] Skipping shift invariance analysis (disabled).\n")

        # 5. Run leakage analysis
        df_leakage = None
        if include_leakage:
            print(f"[4/7] Running leakage analysis ({n_trials} trials)...")
            df_leakage = self.calculator.run_leakage_analysis(n_trials=n_trials)
            print("Leakage analysis complete.\n")
        else:
            print("[4/7] Skipping leakage analysis (disabled).\n")

        # 6. Analyze aliasing and leakage
        df_der, df_alias_summary, df_stats = None, None, None
        if include_aliasing and include_leakage:
            print("[5/7] Analyzing aliasing and leakage...")
            df_der, df_alias_summary, df_stats = self.calculator.analyze_aliasing_and_leakage(df_inv, df_leakage)
            print("Analysis complete.\n")
        elif include_leakage and df_leakage is not None:
            # Compute DER for leakage visualization even if aliasing is disabled
            print("[5/7] Computing diagonal energy ratio for leakage analysis...")
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
            print("DER computation complete.\n")
        else:
            print("[5/7] Skipping aliasing/leakage analysis (tests disabled).\n")

        # 7. Save results
        print("[6/7] Saving results...")
        master_csv_path = os.path.join(self.calculator.OUTPUT_DIR, 'master_results.csv')
        master_df.to_csv(master_csv_path, index=False)
        print(f"Master results saved to: {master_csv_path}\n")

        # 8. Generate all visualizations
        print("[7/7] Generating visualizations...")

        if include_leakage and df_leakage is not None:
            print("  - Leakage matrices...")
            self.plotter.plot_leakage_matrices(df_leakage)
            if df_der is not None:
                print("  - Diagonal energy ratio plot...")
                self.plotter.plot_deviation_leakage(df_der, self.plotter.output_dir)

        if include_aliasing and df_inv is not None:
            print("  - Shift invariance plots...")
            self.plotter.visualize_shift_invariance(df_inv)

        print("  - Absolute metrics (Hamming & Correlation)...")
        self.plotter.visualize_absolute_metrics(master_df)

        print("  - RQA deviation and absolute value plots...")
        self.plotter.visualize_bar_comparison_pro(results, master_df, ref_rqa)

        print("  - Frobenius distance plots...")
        self.plotter.visualize_frobenius_distances(master_df)

        print("  - Cross-band correlation plot...")
        self.plotter.plot_cross_band_summary(master_df)

        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE!")
        print("="*80)
        print(f"\nAll results saved to: {self.calculator.OUTPUT_DIR}")

        return results, master_df, ref_rqa

    def run_statistical_comparison(self, n_trials=100):
        """
        Run statistical comparison with t-tests and generate visualizations.
        This is the full statistical analysis from the original code.

        Args:
            n_trials: Number of trials for statistical validation

        Returns:
            Tuple of (results dict, stats_df)
        """
        print("="*80)
        print("RUNNING STATISTICAL COMPARISON")
        print("="*80)

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
            print("No reference rqa data found, generating...")
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

        for trial in range(n_trials):
            if (trial + 1) % 10 == 0:
                print(f"  Trial {trial + 1}/{n_trials}...")

            signal = self.calculator.generate_composite_sine(random_phase=True)

            for pipeline in pipelines:
                df, _ = self.calculator.validate_pipeline(signal, pipeline, reference_rqa=reference_rqa, reference_data=reference_data)
                all_trial_data.append(df)

            for pipeline in pipelines:
                df, _ = self.calculator.validate_pipeline(signal, pipeline, reference_rqa=reference_rqa, reference_data=reference_data)

                results[pipeline]['det_deviation'].append(df['det_deviation'].mean())
                results[pipeline]['lam_deviation'].append(df['lam_deviation'].mean())
                results[pipeline]['tt_deviation'].append(df['tt_deviation'].mean())
                results[pipeline]['hamming_distance'].append(df['hamming_distance'].mean())
                results[pipeline]['ibf'].append(df['ibf'].mean())
                results[pipeline]['rp_pattern_corr'].append(df['rp_pattern_corr'].mean())
                results[pipeline]['frobenius_distance'].append(df['frobenius_distance'].mean())

        for pipeline in pipelines:
            for metric in results[pipeline].keys():
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
            for metric in ['det_deviation', 'lam_deviation', 'tt_deviation',
                          'hamming_distance', 'ibf', 'rp_pattern_corr', 'frobenius_distance']:
                data1 = results[pipe1][metric]
                data2 = results[pipe2][metric]

                if data1.std() < 1e-10 and data2.std() < 1e-10:
                    continue

                mann_U, p_value = mannwhitneyu(data1, data2)

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
                print(f"    Mann-Whitney U = {mann_U:.4f}, p = {p_value:.6f} {sig}")
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
                    'mann_whitney_u': mann_U,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significance': sig
                })

        stats_df = pd.DataFrame(stats_results)
        stats_file = os.path.join(self.calculator.OUTPUT_DIR, 'stats_final.csv')
        stats_df.to_csv(stats_file, index=False)

        master_df = pd.concat(all_trial_data, ignore_index=True)
        self.plotter.visualize_absolute_metrics(master_df)
        self.plotter.visualize_bar_comparison_pro(results, master_df, reference_rqa)
        self.plotter.visualize_frobenius_distances(master_df)

        print("\n" + "="*80)
        print("STATISTICAL COMPARISON COMPLETE!")
        print("="*80)

        return results, stats_df


def main():
    """
    Main entry point for running experiments.
    """
    # Run full experiment with leakage and aliasing tests
    runner = ExperimentRunner(is_sine=True, is_lorenz=False)

    # Full experiment with all tests
    results, master_df, ref_rqa = runner.run_full_experiment(
        n_trials=10,
        include_leakage=True,
        include_aliasing=True
    )
    print(results)
    print(master_df)
    print("\nExperiment completed successfully!")


if __name__ == '__main__':
    main()
