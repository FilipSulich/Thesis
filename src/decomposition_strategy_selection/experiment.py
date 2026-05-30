import os
import pandas as pd

from src.decomposition_strategy_selection.config import (
    ExperimentConfig,
    DEFAULT_METRICS,
)
from src.decomposition_strategy_selection.signals import generate_composite_sine
from src.decomposition_strategy_selection.ground_truth import (
    get_or_generate_ground_truth,
)
from src.decomposition_strategy_selection.pipeline_analyzer import analyze_pipeline
from src.decomposition_strategy_selection.statistical_tests import (
    run_statistical_analysis,
    save_statistical_results,
)
from src.decomposition_strategy_selection.plotting import (
    plot_boxplot_comparisons,
    plot_violinplot_comparisons,
)


def run_trials(config, band_names, pipeline_names, ground_truth, ground_truth_data, n_trials=100, metrics=None):
    """
    run multiple trials with random phase variations.

    args:
        config: ExperimentConfig instance
        band_names: list of band names
        pipeline_names: list of pipeline names
        ground_truth: dict with ground truth metrics
        ground_truth_data: dict with ground truth RPs
        n_trials: number of trials to run
        metrics: list of metrics to compute (default: uses DEFAULT_METRICS)

    returns:
        DataFrame with aggregated results across all trials
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    trial_data = []

    for trial in range(n_trials):
        if (trial + 1) % 10 == 0:
            print(f'  trial {trial + 1}/{n_trials}')

        # generate signal with random phase
        sig = generate_composite_sine(
            config.n, config.fs, config.subband_freqs, random_phase=True
        )

        # analyze each pipeline
        for pipeline in pipeline_names:
            df = analyze_pipeline(
                sig, pipeline, config, band_names,
                ground_truth, ground_truth_data
            )

            # aggregate metrics across bands
            row = {'trial': trial, 'pipeline': pipeline}
            for metric in metrics:
                if metric in df.columns:
                    row[metric] = df[metric].mean()

            trial_data.append(row)

    return pd.DataFrame(trial_data)


def run_experiment(config, band_names, pipeline_names, n_trials=100, metrics=None):
    """
    run complete pipeline comparison experiment.

    args:
        config: ExperimentConfig instance
        band_names: list of band names (e.g., ['D1', 'D2', 'D3', 'D4', 'A4'])
        pipeline_names: list of pipeline names (e.g., ['swt_recon', 'swt_direct', 'dwt_recon'])
        n_trials: number of trials with random phases
        metrics: list of metrics to analyze (default: uses DEFAULT_METRICS)

    returns:
        tuple of (agg_df, stats_df):
            - agg_df: aggregated results DataFrame
            - stats_df: statistical analysis results
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    os.makedirs(config.output_dir, exist_ok=True)

    # load or generate ground truth
    ground_truth, ground_truth_data = get_or_generate_ground_truth(
        config, band_names, force_regenerate=False
    )

    # run trials
    agg_df = run_trials(
        config, band_names, pipeline_names,
        ground_truth, ground_truth_data,
        n_trials, metrics
    )

    # save results
    results_path = os.path.join(config.output_dir, 'pipeline_comparison_results.csv')
    agg_df.to_csv(results_path, index=False)

    stats_df = run_statistical_analysis(agg_df, metrics, pipeline_names)
    stats_path = os.path.join(config.output_dir, 'statistical_analysis.csv')
    save_statistical_results(stats_df, stats_path)

    plot_boxplot_comparisons(agg_df, metrics, config.output_dir)
    plot_violinplot_comparisons(agg_df, metrics, config.output_dir)

    return agg_df, stats_df
