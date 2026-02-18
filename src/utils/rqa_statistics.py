import csv
import numpy as np
from typing import Dict, List, Optional
import scipy.stats._wilcoxon as _wilcoxon
from scipy.stats import wilcoxon


class RQAStatistics:
    RQA_METRICS = [
        "recurrence_rate",
        "determinism",
        "average_diagonal_line",
        "longest_diagonal_line",
        "divergence",
        "entropy_diagonal_lines",
        "laminarity",
        "trapping_time",
        "longest_vertical_line",
        "entropy_vertical_lines",
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        with open(self.csv_path, "r") as f:
            return list(csv.DictReader(f))

    def _get_metric_values(self, metric: str, is_normal: Optional[bool] = None, prefix: str = "signal") -> np.ndarray:
        values = []
        col_name = f"{prefix}_{metric}"

        for row in self.data:
            if is_normal is not None:
                row_is_normal = row["is_normal"] == "True"
                if row_is_normal != is_normal:
                    continue
            if col_name in row and row[col_name]:
                values.append(float(row[col_name]))

        return np.array(values)

    def descriptive_stats(self, metric: str, is_normal: Optional[bool] = None, prefix: str = "signal") -> Dict[str, float]:
        values = self._get_metric_values(metric, is_normal, prefix)

        if len(values) == 0:
            return {}

        return {
            "n": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "q25": float(np.percentile(values, 25)),
            "median": float(np.median(values)),
            "q75": float(np.percentile(values, 75)),
            "max": float(np.max(values)),
        }

    def compare_groups(self, metric: str, prefix: str = "signal") -> Dict[str, Dict]:
        preaf_values = self._get_metric_values(metric, is_normal=False, prefix=prefix)
        normal_values = self._get_metric_values(metric, is_normal=True, prefix=prefix)

        result = {
            "pre_af": self.descriptive_stats(metric, is_normal=False, prefix=prefix),
            "normal": self.descriptive_stats(metric, is_normal=True, prefix=prefix),
        }
        if len(preaf_values) > 0 and len(normal_values) > 0:
            pooled_std = np.sqrt((np.var(preaf_values) + np.var(normal_values)) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(preaf_values) - np.mean(normal_values)) / pooled_std
                result["cohens_d"] = float(cohens_d)
                test = wilcoxon(preaf_values, normal_values)
                result["p-value"] = float(test.pvalue)

        return result

    def summary(self, prefix: str = "signal") -> Dict[str, Dict]:
        return {metric: self.compare_groups(metric, prefix) for metric in self.RQA_METRICS}

    def print_summary(self, prefix: str = "signal"):
        summary = self.summary(prefix)

        print(f"{'Metric':<25} {'Pre-AF Mean':>12} {"Pre-AF STD" : >12}{'Normal Mean':>12} {"Normal STD": >12} {'Cohen d':>10} {"P value" : >10} {"Significant" : >10}")
        print("-" * 120)

        for metric, data in summary.items():
            preaf_mean = data["pre_af"].get("mean", float("nan"))
            preaf_std = data["pre_af"].get("std", float("nan"))
            normal_mean = data["normal"].get("mean", float("nan"))
            normal_std = data["normal"].get("std", float("nan"))
            cohens_d = data.get("cohens_d", float("nan"))
            pvalue = data.get("p-value", float("nan"))
            significant = False
            if pvalue <=0.05:
                significant = True
            print(f"{metric:<25} {preaf_mean:>12.4f} {preaf_std: >12.4f} {normal_mean:>12.4f} {normal_std: >12.4f} {cohens_d:>10.3f} {pvalue:>10.3f} {significant}")

    def to_csv(self, output_path: str, prefix: str = "signal"):
        """Export summary statistics to CSV."""
        summary = self.summary(prefix)

        with open(output_path, "w", newline="") as f:
            fieldnames = ["metric", "preaf_n", "preaf_mean", "preaf_std", "preaf_median",
                          "normal_n", "normal_mean", "normal_std", "normal_median", "cohens_d"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for metric, data in summary.items():
                row = {
                    "metric": metric,
                    "preaf_n": data["pre_af"].get("n", 0),
                    "preaf_mean": data["pre_af"].get("mean", ""),
                    "preaf_std": data["pre_af"].get("std", ""),
                    "preaf_median": data["pre_af"].get("median", ""),
                    "normal_n": data["normal"].get("n", 0),
                    "normal_mean": data["normal"].get("mean", ""),
                    "normal_std": data["normal"].get("std", ""),
                    "normal_median": data["normal"].get("median", ""),
                    "cohens_d": data.get("cohens_d", ""),
                }
                writer.writerow(row)

        print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    stats = RQAStatistics("/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/rqa_results.csv")
    stats.print_summary()
    stats.to_csv("/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/rqa_summary.csv")