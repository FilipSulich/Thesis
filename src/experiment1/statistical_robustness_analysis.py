
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
from sklearn.linear_model import LinearRegression


def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def two_way_anova(df, metric='det_deviation'):
    """
    Perform Two-Way ANOVA on the specified metric.

    Args:
        df: DataFrame with columns [pipeline, snr_db, metric]
        metric: RQA metric deviation column name

    Returns:
        ANOVA table
    """
    formula = f'{metric} ~ C(pipeline) + C(snr_db) + C(pipeline):C(snr_db)'
    model = ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=2)
    return anova_table


def tukey_posthoc(df, metric='det_deviation', snr_level=5):
    """
    Perform Tukey HSD test for pipeline comparisons at a specific SNR level.

    Args:
        df: DataFrame with columns [pipeline, snr_db, metric]
        metric: RQA metric deviation column name
        snr_level: SNR level to analyze

    Returns:
        Tukey HSD results
    """
    df_snr = df[df['snr_db'] == snr_level]
    tukey = pairwise_tukeyhsd(endog=df_snr[metric], groups=df_snr['pipeline'], alpha=0.05)
    return tukey


def calculate_cv(df, metric='det_deviation'):
    """
    Calculate Coefficient of Variation for each pipeline at each SNR level.

    Args:
        df: DataFrame with columns [pipeline, snr_db, metric]
        metric: RQA metric deviation column name

    Returns:
        DataFrame with CV values
    """
    cv_results = []

    for pipeline in df['pipeline'].unique():
        for snr in sorted(df['snr_db'].unique()):
            subset = df[(df['pipeline'] == pipeline) & (df['snr_db'] == snr)]
            mean_val = subset[metric].mean()
            std_val = subset[metric].std()
            cv = (std_val / mean_val * 100) if mean_val != 0 else np.nan

            cv_results.append({
                'pipeline': pipeline,
                'snr_db': snr,
                'mean': mean_val,
                'std': std_val,
                'cv_percent': cv
            })

    return pd.DataFrame(cv_results)


def calculate_regression_slopes(df, metric='det_deviation'):
    """
    Calculate linear regression slope of metric vs SNR for each pipeline.

    Args:
        df: DataFrame with columns [pipeline, snr_db, metric]
        metric: RQA metric deviation column name

    Returns:
        DataFrame with regression slopes
    """
    slope_results = []

    for pipeline in df['pipeline'].unique():
        subset = df[df['pipeline'] == pipeline]
        X = subset['snr_db'].values.reshape(-1, 1)
        y = subset[metric].values

        model = LinearRegression()
        model.fit(X, y)

        slope = model.coef_[0]
        intercept = model.intercept_
        r_squared = model.score(X, y)

        slope_results.append({
            'pipeline': pipeline,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared
        })

    return pd.DataFrame(slope_results)


def generate_robustness_report(df, metric='det_deviation', snr_levels=[5, 1]):
    """
    Generate comprehensive robustness report.

    Args:
        df: DataFrame with columns [pipeline, snr_db, metric]
        metric: RQA metric deviation column name
        snr_levels: SNR levels for post-hoc analysis

    Returns:
        Dictionary containing all analysis results
    """
    print("=" * 80)
    print(f"ROBUSTNESS ANALYSIS REPORT: {metric}")
    print("=" * 80)

    # Task 1: Two-Way ANOVA
    print("\n[1] TWO-WAY ANOVA")
    print("-" * 80)
    anova_table = two_way_anova(df, metric)
    print(anova_table)

    # Check interaction significance
    interaction_p = anova_table.loc['C(pipeline):C(snr_db)', 'PR(>F)']
    print(f"\nInteraction effect p-value: {interaction_p:.4e}")
    print(f"Interaction significant: {interaction_p < 0.05}")

    # Task 2: Post-Hoc Analysis
    if interaction_p < 0.05:
        print("\n[2] POST-HOC TUKEY HSD TESTS")
        print("-" * 80)
        for snr in snr_levels:
            print(f"\nAt SNR = {snr} dB:")
            tukey_result = tukey_posthoc(df, metric, snr)
            print(tukey_result)

            # Calculate Cohen's d for pairwise comparisons
            pipelines = df['pipeline'].unique()
            print(f"\nCohen's d effect sizes at SNR = {snr} dB:")
            for i, p1 in enumerate(pipelines):
                for p2 in pipelines[i+1:]:
                    data1 = df[(df['pipeline'] == p1) & (df['snr_db'] == snr)][metric]
                    data2 = df[(df['pipeline'] == p2) & (df['snr_db'] == snr)][metric]
                    d = calculate_cohens_d(data1, data2)
                    print(f"  {p1} vs {p2}: d = {d:.3f}")
    else:
        print("\n[2] POST-HOC ANALYSIS: Skipped (interaction not significant)")

    # Task 3: Precision Metrics
    print("\n[3] COEFFICIENT OF VARIATION (CV)")
    print("-" * 80)
    cv_table = calculate_cv(df, metric)
    print(cv_table.to_string(index=False))

    # Task 4: Regression Slopes
    print("\n[4] RATE OF FAILURE (Regression Slopes)")
    print("-" * 80)
    slope_table = calculate_regression_slopes(df, metric)
    print(slope_table.to_string(index=False))
    print("\nNote: Negative slope indicates improvement with increasing SNR")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)

    return {
        'anova': anova_table,
        'cv': cv_table,
        'slopes': slope_table,
        'interaction_p': interaction_p
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)

    pipelines = ['swt_recon', 'dwt_direct', 'swt_direct', 'dwt_recon']
    snr_levels = [20, 10, 5, 1]
    n_realizations = 10

    data = []
    for pipeline in pipelines:
        for snr in snr_levels:
            for _ in range(n_realizations):
                # Simulate data: higher deviation at lower SNR
                base_deviation = 100 / snr + np.random.normal(0, 5)
                data.append({
                    'pipeline': pipeline,
                    'snr_db': snr,
                    'det_deviation': abs(base_deviation),
                    'lam_deviation': abs(base_deviation * 0.8 + np.random.normal(0, 3))
                })

    df = pd.DataFrame(data)

    # Run analysis
    print("Running analysis on DET deviation...")
    results_det = generate_robustness_report(df, metric='det_deviation', snr_levels=[5, 1])

    print("\n\n")
    print("Running analysis on LAM deviation...")
    results_lam = generate_robustness_report(df, metric='lam_deviation', snr_levels=[5, 1])