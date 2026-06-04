import os
import pandas as pd

from src.decomposition_strategy_validation.config import (
    DEFAULT_METRICS,
)
from src.decomposition_strategy_validation.signals import generate_composite_sine
from src.decomposition_strategy_validation.ground_truth import (
    get_or_generate_ground_truth,
)
from src.decomposition_strategy_validation.pipeline_analyzer import analyze_pipeline
from src.decomposition_strategy_validation.statistical_tests import (
    run_statistical_analysis,
    save_statistical_results,
)
from src.decomposition_strategy_validation.plotting import (
    plot_boxplot_comparisons,
    plot_violinplot_comparisons,
)


def run_trials(config, band_names, pipeline_names, ground_truth, ground_truth_data, n_trials=100, metrics=None):
    """
    run multiple trials of pipeline comparison using composite signals with random phases

    Args:
        config (ExperimentConfig): experiment configuration instance with signal and decomposition parameters
        band_names (list): list of wavelet subband names to analyze
        pipeline_names (list): list of pipeline identifiers to compare (e.g., ['swt_recon', 'swt_direct', 'dwt_recon'])
        ground_truth (dict): dictionary containing reference RQA metrics for each subband
        ground_truth_data (dict): dictionary containing reference recurrence plots for each subband
        n_trials (int): number of random phase trials to execute (default: 100)
        metrics (list): list of metric names to aggregate, uses DEFAULT_METRICS if None (default: None)

    Returns:
        pandas.DataFrame: dataframe with aggregated metric values across all trials, pipelines, and subbands
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    trial_data = []

    for trial in range(n_trials):
        if (trial + 1) % 10 == 0:
            print(f' trial {trial + 1}/{n_trials}')

        sig = generate_composite_sine(
            config.n, config.fs, config.subband_freqs
        )

        for pipeline in pipeline_names:
            df = analyze_pipeline(
                sig, pipeline, config, band_names,
                ground_truth, ground_truth_data
            )

            row = {'trial': trial, 'pipeline': pipeline}
            for metric in metrics:
                if metric in df.columns:
                    row[metric] = df[metric].mean()

            trial_data.append(row)

    return pd.DataFrame(trial_data)


def run_experiment(config, band_names, pipeline_names, n_trials=100, metrics=None):
    """
    run complete pipeline comparison experiment including ground truth generation, trial execution, statistical analysis, and visualization

    Args:
        config (ExperimentConfig): experiment configuration instance specifying all experimental parameters
        band_names (list): list of wavelet subband names to analyze (e.g., ['D1', 'D2', 'D3', 'D4', 'A4'])
        pipeline_names (list): list of pipeline identifiers to compare (e.g., ['swt_recon', 'swt_direct', 'dwt_recon'])
        n_trials (int): number of random phase trials for statistical robustness (default: 100)
        metrics (list): list of metric names to analyze, uses DEFAULT_METRICS if None (default: None)

    Returns:
        tuple: tuple containing (agg_df, stats_df) where agg_df is aggregated results and stats_df is statistical test results
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    os.makedirs(config.output_dir, exist_ok=True)

    print("running experiment....")

    ground_truth, ground_truth_data = get_or_generate_ground_truth(
        config, band_names, force_regenerate=False
    )

    agg_df = run_trials(
        config, band_names, pipeline_names,
        ground_truth, ground_truth_data,
        n_trials, metrics
    )

    results_path = os.path.join(config.output_dir, 'pipeline_comparison_results.csv')
    agg_df.to_csv(results_path, index=False)

    stats_df = run_statistical_analysis(agg_df, metrics, pipeline_names)
    stats_path = os.path.join(config.output_dir, 'statistical_analysis.csv')
    save_statistical_results(stats_df, stats_path)

    plot_boxplot_comparisons(agg_df, metrics, config.output_dir)
    plot_violinplot_comparisons(agg_df, metrics, config.output_dir)

    return agg_df, stats_df
