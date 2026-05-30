"""
statistical testing utilities for model comparison.
"""

import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare

def compute_rosenthal_r(x, y):
    """
    compute rosenthal's r effect size for wilcoxon signed-rank test.
    """
    n = len(x)
    result = wilcoxon(x, y, alternative="two-sided", method="approx")
    z_score = result.zstatistic
    r = z_score / np.sqrt(n)
    return float(r)


def friedman_test(data_dict, metric='auc'):
    """
    perform friedman test across multiple groups.

    args:
        data_dict: dict mapping group names to list of fold results
        metric: metric name to extract from fold results

    returns:
        dict with statistic, p_value, and significance
    """
    # extract metric values for each group
    groups = []
    group_names = []

    for name, fold_results in data_dict.items():
        values = [fold[metric] for fold in fold_results]
        groups.append(values)
        group_names.append(name)

    # perform test
    stat, p_value = friedmanchisquare(*groups)

    return {
        'test': 'friedman',
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'groups': group_names
    }


def wilcoxon_pairwise(group1, group2, metric='auc'):
    """
    perform wilcoxon signed-rank test between two groups.

    args:
        group1: list of fold results for group 1
        group2: list of fold results for group 2
        metric: metric name to extract

    returns:
        dict with statistic, p_value, effect_size, and significance
    """
    values1 = [fold[metric] for fold in group1]
    values2 = [fold[metric] for fold in group2]

    result = wilcoxon(values1, values2, alternative="two-sided")
    effect_size = compute_rosenthal_r(values1, values2)

    return {
        'test': 'wilcoxon',
        'statistic': float(result.statistic),
        'p_value': float(result.pvalue),
        'effect_size': effect_size,
        'significant': bool(result.pvalue < 0.05)
    }


def holm_bonferroni_correction(comparisons, alpha=0.05):
    """
    apply holm-bonferroni correction to multiple comparisons.

    args:
        comparisons: list of dicts with 'p_value' key
        alpha: family-wise error rate (default 0.05)

    returns:
        updated comparisons with corrected significance
    """
    # sort by p-value
    sorted_comps = sorted(comparisons, key=lambda x: x['p_value'])
    n = len(sorted_comps)

    # apply holm correction
    for idx, comp in enumerate(sorted_comps):
        adj_alpha = alpha / (n - idx)
        comp['adj_alpha'] = adj_alpha
        comp['significant_corrected'] = comp['p_value'] < adj_alpha

        # if this comparison is not significant, all remaining are also not significant
        if not comp['significant_corrected']:
            for j in range(idx + 1, len(sorted_comps)):
                sorted_comps[j]['adj_alpha'] = alpha / (n - j)
                sorted_comps[j]['significant_corrected'] = False
            break

    return sorted_comps


def pairwise_wilcoxon_with_correction(data_dict, metric='auc', alpha=0.05):
    """
    perform all pairwise wilcoxon tests with holm-bonferroni correction.

    args:
        data_dict: dict mapping group names to list of fold results
        metric: metric name to test
        alpha: family-wise error rate

    returns:
        list of comparison results with corrected significance
    """
    group_names = list(data_dict.keys())
    comparisons = []

    # perform all pairwise comparisons
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            name1, name2 = group_names[i], group_names[j]
            result = wilcoxon_pairwise(data_dict[name1], data_dict[name2], metric)
            result['comparison'] = f'{name1}_vs_{name2}'
            comparisons.append(result)

    # apply correction
    return holm_bonferroni_correction(comparisons, alpha)


# ============================================================================
# summary statistics
# ============================================================================

def compute_summary_stats(fold_results, metric='auc'):
    """
    compute summary statistics for a metric across folds.

    returns:
        dict with mean, std, median, min, max, values
    """
    values = [fold[metric] for fold in fold_results]

    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'values': values
    }


def compare_models(data_dict, metrics=['auc', 'auprc'], alpha=0.05):
    """
    comprehensive statistical comparison of multiple models.

    args:
        data_dict: dict mapping model names to list of fold results
        metrics: list of metrics to test
        alpha: significance level

    returns:
        dict with summary stats, friedman results, and pairwise comparisons
    """
    results = {
        'summary': {},
        'friedman': {},
        'pairwise': {}
    }

    # summary statistics for each model and metric
    for model_name, fold_results in data_dict.items():
        results['summary'][model_name] = {}
        for metric in metrics:
            results['summary'][model_name][metric] = compute_summary_stats(fold_results, metric)

    if len(data_dict) >= 3:
        for metric in metrics:
            results['friedman'][metric] = friedman_test(data_dict, metric)

    # pairwise wilcoxon tests with correction
    for metric in metrics:
        results['pairwise'][metric] = pairwise_wilcoxon_with_correction(data_dict, metric, alpha)

    return results
