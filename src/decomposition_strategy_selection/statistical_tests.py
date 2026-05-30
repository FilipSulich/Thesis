"""statistical tests for pipeline comparison."""

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
import pingouin as pg


def get_significance_notation(p_value):
    """
    convert p-value to significance notation.

    args:
        p_value: p-value from statistical test

    returns:
        string: '***' (p<0.001), '**' (p<0.01), '*' (p<0.05), 'ns' (not significant)
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def run_kruskal_wallis(agg_df, metric, pipeline_names):
    """
    run kruskal-wallis test for overall differences across pipelines.

    args:
        agg_df: aggregated results DataFrame
        metric: metric name to test
        pipeline_names: list of pipeline names

    returns:
        dict with test results (statistic, p_value, significance)
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
        'significance': get_significance_notation(kw_pvalue)
    }


def run_dunn_posthoc(agg_df, metric, pipeline_names):
    """
    run dunn pairwise comparisons with holm-bonferroni correction.

    args:
        agg_df: aggregated results DataFrame
        metric: metric name to test
        pipeline_names: list of pipeline names

    returns:
        list of dicts with pairwise comparison results
    """
    # dunn post-hoc test
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

            # rank-biserial correlation (non-parametric effect size)
            # computed from mann-whitney U test
            mwu_result = pg.mwu(x, y)
            rank_biserial = mwu_result['RBC'].values[0]  # RBC = rank-biserial correlation

            dunn_pvalue = dunn_df.loc[pipe1, pipe2]

            results.append({
                'metric': metric,
                'test': 'Dunn',
                'pipeline1': pipe1,
                'pipeline2': pipe2,
                'statistic': np.nan,
                'p_value': dunn_pvalue,
                'rank_biserial': rank_biserial,
                'significance': get_significance_notation(dunn_pvalue)
            })

    return results


def run_statistical_analysis(agg_df, metrics, pipeline_names):
    """
    perform statistical tests on pipeline comparison results.

    tests:
    - kruskal-wallis: overall differences across pipelines
    - dunn: pairwise comparisons with holm correction
    - rank-biserial correlation: non-parametric effect sizes

    args:
        agg_df: aggregated results DataFrame
        metrics: list of metric names to test
        pipeline_names: list of pipeline names

    returns:
        DataFrame with statistical test results
    """
    print("\nrunning statistical analysis...")

    stats_results = []

    for metric in metrics:
        # kruskal-wallis test
        kw_result = run_kruskal_wallis(agg_df, metric, pipeline_names)
        stats_results.append(kw_result)

        # dunn pairwise comparisons
        dunn_results = run_dunn_posthoc(agg_df, metric, pipeline_names)
        stats_results.extend(dunn_results)

    return pd.DataFrame(stats_results)


def save_statistical_results(stats_df, output_path):
    """
    save statistical analysis results to CSV.

    args:
        stats_df: DataFrame with statistical test results
        output_path: path to save CSV file
    """
    stats_df.to_csv(output_path, index=False)
