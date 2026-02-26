import csv
import numpy as np
from typing import Dict, List
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

    def _aggregate_per_patient(self, metric: str, prefix: str = "signal"):
        """
        Returns:
            dict:
                patient_id -> {"pre_af": mean_value, "normal": mean_value}
        """
        col_name = f"{prefix}_{metric}"
        patient_dict = {}

        for row in self.data:
            if col_name not in row or not row[col_name]:
                continue

            # Skip NaN values
            if row[col_name].lower() == "nan":
                continue

            patient = row["patient_id"]
            is_normal = row["is_normal"] == "True"
            value = float(row[col_name])

            if patient not in patient_dict:
                patient_dict[patient] = {"pre_af": [], "normal": []}

            if is_normal:
                patient_dict[patient]["normal"].append(value)
            else:
                patient_dict[patient]["pre_af"].append(value)

        # convert to means per patient
        for patient in patient_dict:
            patient_dict[patient]["pre_af"] = (
                np.mean(patient_dict[patient]["pre_af"])
                if patient_dict[patient]["pre_af"]
                else None
            )
            patient_dict[patient]["normal"] = (
                np.mean(patient_dict[patient]["normal"])
                if patient_dict[patient]["normal"]
                else None
            )

        return patient_dict

    def compare_groups(self, metric: str, prefix: str = "signal") -> Dict:
        """
        Compare pre-AF vs normal for patients with both segments
        """
        patient_data = self._aggregate_per_patient(metric, prefix)

        preaf = []
        normal = []
        patient_pairs = {}

        # keep only patients with both conditions
        for patient, values in patient_data.items():
            if values["pre_af"] is not None and values["normal"] is not None:
                preaf.append(values["pre_af"])
                normal.append(values["normal"])
                patient_pairs[patient] = {"pre_af": values["pre_af"], "normal": values["normal"]}

        preaf = np.array(preaf)
        normal = np.array(normal)

        result = {}
        if len(preaf) == 0:
            return result

        # Descriptive stats per group (between patients)
        result["n_patients"] = len(preaf)
        result["preaf_mean"] = float(np.mean(preaf))
        result["preaf_std"] = float(np.std(preaf, ddof=1))
        result["normal_mean"] = float(np.mean(normal))
        result["normal_std"] = float(np.std(normal, ddof=1))

        diff = preaf - normal
        if np.allclose(diff, 0):
            p_value = 1.0
        else:
            test = wilcoxon(preaf, normal, zero_method="wilcox")
            p_value = test.pvalue

        result["p_value"] = float(p_value)

        # Cohen's d for paired samples
        diff = preaf - normal
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        result["cohens_d"] = float(cohens_d)

        # store patient-level data
        result["patient_level"] = patient_pairs

        return result

    def save_results_csv(self, output_path: str):
        """
        Save patient-level and between-patient RQA statistics to CSV
        """
        fieldnames = [
            "metric",
            "n_patients",
            "preaf_mean",
            "preaf_std",
            "normal_mean",
            "normal_std",
            "cohens_d",
            "p_value"
        ]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for metric in self.RQA_METRICS:
                data = self.compare_groups(metric)
                if not data:
                    continue

                row = {
                    "metric": metric,
                    "n_patients": data["n_patients"],
                    "preaf_mean": data["preaf_mean"],
                    "preaf_std": data["preaf_std"],
                    "normal_mean": data["normal_mean"],
                    "normal_std": data["normal_std"],
                    "cohens_d": data["cohens_d"],
                    "p_value": data["p_value"]
                }
                writer.writerow(row)

    def plot_metrics(self, output_dir: str):
        """
        Make boxplots of RQA metrics per metric (pre-AF vs normal)
        """
        os.makedirs(output_dir, exist_ok=True)
        for metric in self.RQA_METRICS:
            data = self.compare_groups(metric)
            if not data:
                continue

            patients = data["patient_level"]
            preaf = [v["pre_af"] for v in patients.values()]
            normal = [v["normal"] for v in patients.values()]

            plt.figure(figsize=(6, 4))
            sns.boxplot(data=[normal, preaf], palette=["skyblue", "salmon"])
            plt.xticks([0, 1], ["Normal", "Pre-AF"])
            plt.title(f"RQA Metric: {metric}")
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{metric}_boxplot.png"))
            plt.close()

    def print_summary(self, prefix: str = "signal"):
        print(f"{'Metric':<30} {'N':>3} {'Pre-AF (mean±std)':>20} {'Normal (mean±std)':>20} {'Cohen d':>9} {'P':>8}")
        print("-" * 95)
        for metric in self.RQA_METRICS:
            data = self.compare_groups(metric, prefix)
            if not data:
                continue
            preaf_str = f"{data['preaf_mean']:.4f}±{data['preaf_std']:.4f}"
            normal_str = f"{data['normal_mean']:.4f}±{data['normal_std']:.4f}"
            print(
                f"{metric:<30} "
                f"{data['n_patients']:>3} "
                f"{preaf_str:>20} "
                f"{normal_str:>20} "
                f"{data['cohens_d']:>9.3f} "
                f"{data['p_value']:>8.4f}"
            )

    def print_multiscale_summary(self):
        """Print summary for multi-scale results (D1, D2, D3, D4, A4 subbands)."""
        subbands = ["D1", "D2", "D3", "D4", "A4"]

        for subband in subbands:
            print(f"\n{'='*80}")
            print(f"SUBBAND: {subband}")
            print(f"{'='*80}")
            self.print_summary(prefix=subband)

    def get_significant_results(self, alpha: float = 0.05, prefixes: List[str] = None) -> List[Dict]:
        """Get all metrics with p < alpha."""
        if prefixes is None:
            prefixes = ["signal"]

        significant = []
        for prefix in prefixes:
            for metric in self.RQA_METRICS:
                data = self.compare_groups(metric, prefix)
                if data and data.get("p_value", 1.0) < alpha:
                    significant.append({
                        "prefix": prefix,
                        "metric": metric,
                        "p_value": data["p_value"],
                        "cohens_d": data["cohens_d"],
                        "preaf_mean": data["preaf_mean"],
                        "normal_mean": data["normal_mean"],
                    })

        return sorted(significant, key=lambda x: x["p_value"])


if __name__ == "__main__":
    import sys

    # Check if multi-scale results exist
    multiscale_path = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/old/rqa_results_multiscale.csv"
    single_scale_path = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/rqa_results_single_scale.csv"

    multi_scale = True

    if multi_scale:
        print("="*80)
        print("MULTI-SCALE RQA RESULTS")
        print("="*80)
        stats = RQAStatistics(multiscale_path)
        stats.print_multiscale_summary()

        print("\n" + "="*80)
        print("SIGNIFICANT RESULTS (p < 0.05)")
        print("="*80)
        sig = stats.get_significant_results(alpha=0.05, prefixes=["D1", "D2", "D3", "D4", "A4"])
        if sig:
            for r in sig:
                print(f"{r['prefix']}_{r['metric']}: p={r['p_value']:.4f}, d={r['cohens_d']:.3f}")
        else:
            print("No significant results found.")
    else:
        print("SINGLE-SCALE RQA RESULTS")
        print("="*80)
        stats = RQAStatistics(single_scale_path)
        stats.print_summary()