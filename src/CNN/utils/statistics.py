import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare

def compute_rosenthal_r(x, y):
    """
    computes rosenthal's r effect size for wilcoxon signed-rank test

    Args:
        x (list): first group of values
        y (list): second group of values

    Returns:
        float: rosenthal's r effect size value
    """
    n = len(x)
    result = wilcoxon(x, y, alternative="two-sided", method="approx")
    z_score = result.zstatistic
    r = z_score / np.sqrt(n)
    return float(r)


def friedman_test(data_dict, metric='auc'):
    """
    performs friedman test to compare metric values across multiple model groups

    Args:
        data_dict (dict): mapping of group names to list of fold results
        metric (str): metric name to extract from fold results

    Returns:
        dict: test results with statistic, p_value, and significance flag
    """
    groups = []
    group_names = []

    for name, fold_results in data_dict.items():
        values = [fold[metric] for fold in fold_results]
        groups.append(values)
        group_names.append(name)

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
    performs wilcoxon signed-rank test between two model groups

    Args:
        group1 (list): list of fold results for first group
        group2 (list): list of fold results for second group
        metric (str): metric name to extract and compare

    Returns:
        dict: test results with statistic, p_value, effect_size, and significance flag
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
    applies holm-bonferroni correction for multiple testing to control family-wise error rate

    Args:
        comparisons (list): list of comparison dictionaries each with 'p_value' key
        alpha (float): significance level threshold

    Returns:
        list: updated comparisons with corrected significance flags
    """
    sorted_comps = sorted(comparisons, key=lambda x: x['p_value'])
    n = len(sorted_comps)

    for idx, comp in enumerate(sorted_comps):
        adj_alpha = alpha / (n - idx)
        comp['adj_alpha'] = adj_alpha
        comp['significant_corrected'] = comp['p_value'] < adj_alpha

        if not comp['significant_corrected']:
            for j in range(idx + 1, len(sorted_comps)):
                sorted_comps[j]['adj_alpha'] = alpha / (n - j)
                sorted_comps[j]['significant_corrected'] = False
            break

    return sorted_comps


def pairwise_wilcoxon_with_correction(data_dict, metric='auc', alpha=0.05):
    """
    performs all pairwise wilcoxon tests between model groups with holm-bonferroni correction

    Args:
        data_dict (dict): mapping of group names to list of fold results
        metric (str): metric name to test
        alpha (float): significance level threshold

    Returns:
        list: pairwise comparison results with corrected significance flags
    """
    group_names = list(data_dict.keys())
    comparisons = []

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            name1, name2 = group_names[i], group_names[j]
            result = wilcoxon_pairwise(data_dict[name1], data_dict[name2], metric)
            result['comparison'] = f'{name1}_vs_{name2}'
            comparisons.append(result)

    return holm_bonferroni_correction(comparisons, alpha)

def compute_summary_stats(fold_results, metric='auc'):
    """
    computes summary statistics for a single metric across all folds

    Args:
        fold_results (list): list of fold result dictionaries
        metric (str): metric name to extract

    Returns:
        dict: summary with mean, std, median, min, max, and raw values
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
    performs comprehensive statistical comparison of multiple models across specified metrics

    Args:
        data_dict (dict): mapping of model names to list of fold results
        metrics (list): list of metric names to test
        alpha (float): significance level threshold

    Returns:
        dict: complete statistical analysis with summary stats, friedman results, and pairwise comparisons
    """
    for model_name, fold_results in data_dict.items():
        if not fold_results:
            raise ValueError(f"model {model_name} has no fold results")

    results = {
        'summary': {},
        'friedman': {},
        'pairwise': {}
    }

    for model_name, fold_results in data_dict.items():
        results['summary'][model_name] = {}
        for metric in metrics:
            results['summary'][model_name][metric] = compute_summary_stats(fold_results, metric)

    if len(data_dict) >= 3:
        for metric in metrics:
            results['friedman'][metric] = friedman_test(data_dict, metric)

    for metric in metrics:
        results['pairwise'][metric] = pairwise_wilcoxon_with_correction(data_dict, metric, alpha)

    return results
