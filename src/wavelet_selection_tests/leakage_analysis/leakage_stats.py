import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from ..utils import _compute_rank_biserial


class LeakageStats:
    """performs statistical analysis on wavelet leakage test results"""

    def __init__(self, output_dir):
        """
        initializes statistical analyzer with output directory

        Args:
            output_dir (str): directory path for saving statistical test results
        """
        self.output_dir = Path(output_dir)

    def save_all_stats(self, df: pd.DataFrame) -> None:
        """
        runs all statistical tests and saves results to files for both conditions

        Args:
            df (pd.DataFrame): results dataframe with columns [wavelet, condition, DER]

        Returns:
            None
        """
        for condition in ['Harmonic', 'Non-harmonic']:
            df_cond = df[df['condition'] == condition]

            self._save_summary(df_cond, condition)
            self._run_tests(df_cond, condition)

    def _save_summary(self, df_cond: pd.DataFrame, condition: str) -> None:
        """
        computes and saves descriptive statistics for one condition

        Args:
            df_cond (pd.DataFrame): dataframe filtered for single condition
            condition (str): condition name for file naming

        Returns:
            None
        """
        summary_stats = []
        for wavelet in sorted(df_cond['wavelet'].unique()):
            ders = df_cond[df_cond['wavelet'] == wavelet]['DER'].values
            summary_stats.append({
                'wavelet': wavelet,
                'mean': float(np.mean(ders)),
                'median': float(np.median(ders)),
                'std': float(np.std(ders)),
                'min': float(np.min(ders)),
                'max': float(np.max(ders)),
                'n_trials': len(ders),
            })

        summary_df = pd.DataFrame(summary_stats)
        summary_df = summary_df.sort_values('mean', ascending=False)

        summary_path = self.output_dir / f'summary_statistics_{condition}.csv'
        summary_df.to_csv(summary_path, index=False)

    def _run_tests(self, df_cond: pd.DataFrame, condition: str) -> None:
        """
        performs kruskal-wallis test and dunn post-hoc pairwise comparisons

        Args:
            df_cond (pd.DataFrame): dataframe filtered for single condition
            condition (str): condition name for file naming

        Returns:
            None
        """
        wavelets = sorted(df_cond['wavelet'].unique())
        groups = [df_cond[df_cond['wavelet'] == w]['DER'].values for w in wavelets]

        h_stat, p_value_kw = kruskal(*groups)
        sig = '***' if p_value_kw < 0.001 else '**' if p_value_kw < 0.01 else '*' if p_value_kw < 0.05 else 'ns'

        kw_results = {
            'test': 'Kruskal-Wallis',
            'condition': condition,
            'H_statistic': float(h_stat),
            'p_value': float(p_value_kw),
            'significance': sig
        }
        kw_path = self.output_dir / f'kruskal_wallis_{condition}.json'
        with open(kw_path, 'w') as f:
            json.dump(kw_results, f, indent=2)

        try:
            dunn_results = posthoc_dunn(df_cond, val_col='DER', group_col='wavelet', p_adjust='fdr_bh')

            dunn_path = self.output_dir / f'dunn_posthoc_{condition}.csv'
            dunn_results.to_csv(dunn_path)

            comparisons = []
            for i, wav1 in enumerate(wavelets):
                for j, wav2 in enumerate(wavelets):
                    if i < j:
                        p_val = dunn_results.loc[wav1, wav2]
                        mean1 = df_cond[df_cond['wavelet'] == wav1]['DER'].mean()
                        mean2 = df_cond[df_cond['wavelet'] == wav2]['DER'].mean()
                        mean_diff = abs(mean1 - mean2)

                        if p_val < 0.001:
                            sig = '***'
                        elif p_val < 0.01:
                            sig = '**'
                        elif p_val < 0.05:
                            sig = '*'
                        else:
                            sig = 'ns'

                        comparisons.append({
                            'wavelet_1': wav1,
                            'wavelet_2': wav2,
                            'p_value': float(p_val),
                            'significance': sig,
                            'mean_diff': float(mean_diff)
                        })

            comp_df = pd.DataFrame(comparisons)
            comp_path = self.output_dir / f'pairwise_comparisons_{condition}.csv'
            comp_df.to_csv(comp_path, index=False)

        except Exception:
            pass

    def create_db8_comparison_table(self, df: pd.DataFrame) -> None:
        """
        creates pairwise comparison table of all wavelets against db8 baseline

        Args:
            df (pd.DataFrame): complete results dataframe with all wavelets and conditions

        Returns:
            None
        """
        all_wavelets = sorted([w for w in df['wavelet'].unique() if w != 'db8'])

        db8_harm = df[(df['wavelet'] == 'db8') & (df['condition'] == 'Harmonic')]['DER'].values
        db8_nonharm = df[(df['wavelet'] == 'db8') & (df['condition'] == 'Non-harmonic')]['DER'].values

        dunn_harm_path = self.output_dir / 'dunn_posthoc_Harmonic.csv'
        dunn_nonharm_path = self.output_dir / 'dunn_posthoc_Non-harmonic.csv'

        if not dunn_harm_path.exists() or not dunn_nonharm_path.exists():
            return

        dunn_harm = pd.read_csv(dunn_harm_path, index_col=0)
        dunn_nonharm = pd.read_csv(dunn_nonharm_path, index_col=0)

        comparison_data = []

        for wavelet in all_wavelets:
            try:
                p_harm = dunn_harm.loc['db8', wavelet]
            except:
                p_harm = np.nan

            try:
                p_nonharm = dunn_nonharm.loc['db8', wavelet]
            except:
                p_nonharm = np.nan

            wav_harm = df[(df['wavelet'] == wavelet) & (df['condition'] == 'Harmonic')]['DER'].values
            wav_nonharm = df[(df['wavelet'] == wavelet) & (df['condition'] == 'Non-harmonic')]['DER'].values

            rb_harm = _compute_rank_biserial(db8_harm, wav_harm)
            rb_nonharm = _compute_rank_biserial(db8_nonharm, wav_nonharm)

            comparison_data.append({
                'wavelet': wavelet,
                'p_value_non_harmonic': p_nonharm,
                'p_value_harmonic': p_harm,
                'rank_biserial_non_harmonic': rb_nonharm,
                'rank_biserial_harmonic': rb_harm
            })

        comparison_df = pd.DataFrame(comparison_data)
        output_path = self.output_dir / 'db8_comparison_table.csv'
        comparison_df.to_csv(output_path, index=False)
