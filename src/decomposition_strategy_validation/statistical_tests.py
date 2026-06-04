import numpy as np
import pandas as pd
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
import pingouin as pg



def run_kruskal_wallis(agg_df, metric, pipeline_names):
    """
    run kruskal-wallis test to detect differences in metric values across multiple pipelines

    Args:
        agg_df (pandas.DataFrame): aggregated results dataframe containing pipeline and metric columns
        metric (str): name of the metric column to test
        pipeline_names (list): list of pipeline names to include in the test

    Returns:
        dict: dictionary containing test name, statistic, p-value
    """
    groups = [
        agg_df[agg_df['pipeline'] == p][metric].dropna().values
        for p in pipeline_names
    ]

    h_stat, kw_pvalue = kruskal(*groups)

    return {
        'metric': metric,
        'test': 'Kruskal-Wallis',
        'pipeline1': 'all',
        'pipeline2': 'all',
        'statistic': h_stat,
        'p_value': kw_pvalue,
        'rank_biserial': np.nan,
    }


def run_dunn_posthoc(agg_df, metric, pipeline_names):
    """
    run dunn pairwise post-hoc comparisons with holm-bonferroni correction and compute rank-biserial effect sizes

    Args:
        agg_df (pandas.DataFrame): aggregated results dataframe containing pipeline and metric columns
        metric (str): name of the metric column to test
        pipeline_names (list): list of pipeline names to compare pairwise

    Returns:
        list: list of dictionaries containing pairwise comparison results with p-values, rank-biserial correlations
    """
    dunn_df = posthoc_dunn(
        agg_df,
        val_col=metric,
        group_col='pipeline',
        p_adjust='holm'
    )

    results = []

    for i, pipe1 in enumerate(pipeline_names):
        for pipe2 in pipeline_names[i + 1:]:
            x = agg_df[agg_df['pipeline'] == pipe1][metric].dropna().values
            y = agg_df[agg_df['pipeline'] == pipe2][metric].dropna().values

            if len(x) < 3 or len(y) < 3:
                continue

            mwu_result = pg.mwu(x, y)
            rank_biserial = mwu_result['RBC'].values[0]

            dunn_pvalue = dunn_df.loc[pipe1, pipe2]

            results.append({
                'metric': metric,
                'test': 'Dunn',
                'pipeline1': pipe1,
                'pipeline2': pipe2,
                'statistic': np.nan,
                'p_value': dunn_pvalue,
                'rank_biserial': rank_biserial,
            })

    return results


def run_statistical_analysis(agg_df, metrics, pipeline_names):
    """
    perform comprehensive statistical analysis including kruskal-wallis tests and dunn post-hoc comparisons for multiple metrics

    Args:
        agg_df (pandas.DataFrame): aggregated results dataframe containing pipeline and metric columns
        metrics (list): list of metric names to analyze
        pipeline_names (list): list of pipeline names to compare

    Returns:
        pandas.DataFrame: dataframe containing all statistical test results including overall tests and pairwise comparisons
    """
    stats_results = []

    for metric in metrics:
        kw_result = run_kruskal_wallis(agg_df, metric, pipeline_names)
        stats_results.append(kw_result)

        dunn_results = run_dunn_posthoc(agg_df, metric, pipeline_names)
        stats_results.extend(dunn_results)

    return pd.DataFrame(stats_results)


def save_statistical_results(stats_df, output_path):
    """
    save statistical analysis results dataframe to a CSV file

    Args:
        stats_df (pandas.DataFrame): dataframe containing statistical test results
        output_path (str): file path where the CSV will be saved
    """
    stats_df.to_csv(output_path, index=False)
