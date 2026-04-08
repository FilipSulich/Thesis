import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu


class RQAMetricsAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.meta_cols = ["segment_id", "patient_id", "label", "single_scale", "method"]
        self.metric_cols = [c for c in df.columns if c not in self.meta_cols]

    @classmethod
    def from_csv(cls, path: str):
        return cls(pd.read_csv(path))

    @classmethod
    def from_two_csv(cls, path_label0: str, path_label1: str):
        df0 = pd.read_csv(path_label0)
        df1 = pd.read_csv(path_label1)

        df0["label"] = 0
        df1["label"] = 1

        return cls(pd.concat([df0, df1], ignore_index=True))

    # --------------------------------------------------
    # Patient-level aggregation (median per patient)
    # --------------------------------------------------
    def aggregate_patient_medians(self):
        grouped = self.df.groupby(["patient_id", "label"])
        return grouped[self.metric_cols].median().reset_index()

    # --------------------------------------------------
    # Cohen's d
    # --------------------------------------------------
    @staticmethod
    def cohens_d(x, y):
        x, y = np.array(x), np.array(y)
        nx, ny = len(x), len(y)

        if nx < 2 or ny < 2:
            return np.nan

        std_x = np.std(x, ddof=1)
        std_y = np.std(y, ddof=1)

        pooled_std = np.sqrt(
            ((nx - 1) * std_x**2 + (ny - 1) * std_y**2) / (nx + ny - 2)
        )

        if pooled_std == 0:
            return 0.0

        return (np.mean(x) - np.mean(y)) / pooled_std

    # --------------------------------------------------
    # Holm-Bonferroni correction
    # --------------------------------------------------
    @staticmethod
    def holm_correction(p_values, alpha=0.05):
        p_values = np.array(p_values)
        m = len(p_values)

        order = np.argsort(p_values)
        sorted_p = p_values[order]

        adjusted = np.empty(m)
        reject = np.zeros(m, dtype=bool)

        for i, p in enumerate(sorted_p):
            adjusted_p = (m - i) * p
            adjusted[i] = min(adjusted_p, 1.0)

        # Enforce monotonicity
        for i in range(1, m):
            adjusted[i] = max(adjusted[i], adjusted[i - 1])

        # Map back to original order
        adjusted_final = np.empty(m)
        adjusted_final[order] = adjusted

        # Reject decisions
        for i, p in enumerate(sorted_p):
            if p <= alpha / (m - i):
                reject[i] = True
            else:
                break

        reject_final = np.zeros(m, dtype=bool)
        reject_final[order] = reject

        return adjusted_final, reject_final

    # --------------------------------------------------
    # Comparison
    # --------------------------------------------------
    def compare_labels(self, alpha=0.05):
        patient_df = self.aggregate_patient_medians()

        group0 = patient_df[patient_df["label"] == 0]
        group1 = patient_df[patient_df["label"] == 1]

        results = []
        p_values = []

        # First pass: compute stats
        for col in self.metric_cols:
            x = group0[col].dropna()
            y = group1[col].dropna()

            if len(x) == 0 or len(y) == 0:
                continue

            stat, p = mannwhitneyu(x, y, alternative='two-sided')
            d = self.cohens_d(x, y)

            results.append({
                "metric": col,
                "mean_label0": np.mean(x),
                "mean_label1": np.mean(y),
                "median_label0": np.median(x),
                "median_label1": np.median(y),
                "mw_u_stat": stat,
                "p_value": p,
                "cohens_d": d,
                "n_label0": len(x),
                "n_label1": len(y)
            })

            p_values.append(p)

        results_df = pd.DataFrame(results)

        # Apply Holm correction
        if len(p_values) > 0:
            p_corr, reject = self.holm_correction(p_values, alpha)

            results_df["p_value_corrected"] = p_corr
            results_df["reject_null"] = reject

        return results_df

if __name__ == "__main__":
    label0 = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data_new/exp2/results_parallel/nsr_single_rqa.csv"
    label1 = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data_new/exp2/results_parallel/preaf_single_rqa.csv"
    label0_m = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data_new/exp2/results_parallel/preaf_swt_rqa.csv"
    label1_m = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data_new/exp2/results_parallel/nsr_swt_rqa.csv"
    # analyzer = RQAMetricsAnalyzer.from_two_csv(label0_m, label1_m)
    analyzer = RQAMetricsAnalyzer.from_two_csv(label0, label1)

    comparison_df = analyzer.compare_labels()
    pd.DataFrame(comparison_df).to_csv("comparison_df_single.csv")
    # pd.DataFrame(comparison_df).to_csv("comparison_df_swt.csv")
    print(comparison_df)