import csv
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import os


class RQAStatistics:

    RQA_METRICS = [
        "epsilon",
        # "recurrence_rate",
        "determinism",
        "average_diagonal_line",
        "longest_diagonal_line",
        "entropy_diagonal_lines",
        "laminarity",
        "trapping_time",
        "longest_vertical_line",
        "entropy_vertical_lines",
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data     = self._load_data()

    def _load_data(self) -> List[Dict]:
        with open(self.csv_path, "r") as f:
            return list(csv.DictReader(f))

    def _aggregate_per_patient(self, metric: str, prefix: str = "signal") -> Dict:
        col_name     = f"{prefix}_{metric}"
        patient_dict = {}

        for row in self.data:
            if col_name not in row or not row[col_name]:
                continue
            val_str = row[col_name].strip().lower()
            if val_str in ("nan", "inf", "-inf", ""):
                continue

            patient_id = row["patient_id"]
            label      = int(row["label"])
            value      = float(row[col_name])

            key = (patient_id, label)
            if key not in patient_dict:
                patient_dict[key] = []
            patient_dict[key].append(value)

        result = {}
        for (patient_id, label), values in patient_dict.items():
            result[(patient_id, label)] = {
                "label":  label,
                "mean":   float(np.mean(values)),
                "n_segs": len(values),
            }

        return result

    def _split_groups(
        self, metric: str, prefix: str = "signal"
    ) -> Tuple[Dict, Dict]:
        """
        Split patient data into Pre-AF and Normal (n* patients only) groups.
        Returns independent groups for Mann-Whitney U test.
        """
        patient_data = self._aggregate_per_patient(metric, prefix)

        preaf_patients  = {}
        normal_patients = {}

        for (patient_id, label), info in patient_data.items():
            # Check if patient ID contains '_n' (normal patients like paf_n01)
            is_n_patient = '_n' in patient_id.lower()

            if label == 1:  # Pre-AF
                preaf_patients[patient_id] = info["mean"]
            elif label == 0 and is_n_patient:  # Normal from n* patients
                normal_patients[patient_id] = info["mean"]
            # Ignore normal segments from pre-AF patients

        return preaf_patients, normal_patients


    def compare_groups(self, metric: str, prefix: str = "signal") -> Dict:
        """
        Compare Pre-AF vs Normal (n* patients only) using Mann-Whitney U test.
        Groups are independent (not paired).
        """
        preaf_patients, normal_patients = self._split_groups(metric, prefix)

        if not preaf_patients or not normal_patients:
            return {}

        preaf_vals  = np.array(list(preaf_patients.values()))
        normal_vals = np.array(list(normal_patients.values()))

        # Mann-Whitney U test (independent samples)
        if len(preaf_vals) >= 2 and len(normal_vals) >= 2:
            _, p_value = mannwhitneyu(preaf_vals, normal_vals, alternative='two-sided')

            # Cohen's d for independent samples
            pooled_std = np.sqrt((np.std(preaf_vals, ddof=1) ** 2 +
                                 np.std(normal_vals, ddof=1) ** 2) / 2)
            cohens_d = (float((np.mean(preaf_vals) - np.mean(normal_vals)) / pooled_std)
                       if pooled_std > 0 else 0.0)
        else:
            p_value = float('nan')
            cohens_d = float('nan')

        return {
            "n_preaf":   len(preaf_vals),
            "n_normal":  len(normal_vals),
            "p_value":   float(p_value),
            "cohens_d":  float(cohens_d),

            "preaf_mean":    float(np.mean(preaf_vals)),
            "preaf_std":     float(np.std(preaf_vals, ddof=1)) if len(preaf_vals) > 1 else 0.0,
            "preaf_median":  float(np.median(preaf_vals)),

            "normal_mean":   float(np.mean(normal_vals)),
            "normal_std":    float(np.std(normal_vals, ddof=1)) if len(normal_vals) > 1 else 0.0,
            "normal_median": float(np.median(normal_vals)),

            "preaf_values":  preaf_vals.tolist(),
            "normal_values": normal_vals.tolist(),
        }

    def print_summary(self, prefix: str = "signal"):
        """Print summary using Mann-Whitney U test (independent samples)."""
        header = (f"\n{'Metric':<30} {'N_preAF':>8} {'N_normal':>9} "
                  f"{'Pre-AF (mean±std)':>22} {'Normal (mean±std)':>22} "
                  f"{'Cohen d':>8} {'p-value':>9}")
        print(header)
        print("-" * 120)

        for metric in self.RQA_METRICS:
            data = self.compare_groups(metric, prefix)
            if not data:
                continue

            preaf_str  = f"{data['preaf_mean']:.4f}±{data['preaf_std']:.4f}"
            normal_str = f"{data['normal_mean']:.4f}±{data['normal_std']:.4f}"
            p_str = f"{data['p_value']:.4f}" if not np.isnan(data['p_value']) else "   nan"
            d_str = f"{data['cohens_d']:.3f}" if not np.isnan(data['cohens_d']) else "  nan"

            print(
                f"{metric:<30} "
                f"{data['n_preaf']:>8} "
                f"{data['n_normal']:>9} "
                f"{preaf_str:>22} "
                f"{normal_str:>22} "
                f"{d_str:>8} "
                f"{p_str:>9}"
            )

    def print_multiscale_summary(self):
        for subband in ["D1", "D2", "D3", "D4", "A4"]:
            print(f"\n{'='*80}\nSUBBAND: {subband}\n{'='*80}")
            self.print_summary(prefix=subband)

    def get_significant_results(
        self,
        alpha: float = 0.05,
        prefixes: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Get statistically significant results using Mann-Whitney U test."""
        if prefixes is None:
            prefixes = ["signal"]

        significant = []
        for prefix in prefixes:
            for metric in self.RQA_METRICS:
                data = self.compare_groups(metric, prefix)
                if not data:
                    continue
                p = data.get("p_value", float('nan'))
                if not np.isnan(p) and p < alpha:
                    significant.append({
                        "prefix":    prefix,
                        "metric":    metric,
                        "p_value":   p,
                        "cohens_d":  data["cohens_d"],
                        "n_preaf":   data["n_preaf"],
                        "n_normal":  data["n_normal"],
                    })

        return sorted(significant, key=lambda x: x["p_value"])

    def save_results_csv(self, output_path: str,
                         prefixes: Optional[List[str]] = None):
        """Save statistical results using Mann-Whitney U test."""
        if prefixes is None:
            prefixes = ["signal"]

        fieldnames = [
            "prefix", "metric",
            "n_preaf", "n_normal",
            "preaf_mean", "preaf_std", "preaf_median",
            "normal_mean", "normal_std", "normal_median",
            "cohens_d", "p_value",
        ]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for prefix in prefixes:
                for metric in self.RQA_METRICS:
                    data = self.compare_groups(metric, prefix)
                    if not data:
                        continue
                    writer.writerow({
                        "prefix":        prefix,
                        "metric":        metric,
                        "n_preaf":       data["n_preaf"],
                        "n_normal":      data["n_normal"],
                        "preaf_mean":    data["preaf_mean"],
                        "preaf_std":     data["preaf_std"],
                        "preaf_median":  data["preaf_median"],
                        "normal_mean":   data["normal_mean"],
                        "normal_std":    data["normal_std"],
                        "normal_median": data["normal_median"],
                        "cohens_d":      data["cohens_d"],
                        "p_value":       data["p_value"],
                    })

        print(f"Saved results to {output_path}")

    def plot_metrics(self, output_dir: str, prefix: str = "signal"):
        """Plot metrics with Mann-Whitney U test results."""
        os.makedirs(output_dir, exist_ok=True)
        for metric in self.RQA_METRICS:
            data = self.compare_groups(metric, prefix)
            if not data:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                data=[data["normal_values"], data["preaf_values"]],
                palette=["skyblue", "salmon"], ax=ax
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels([
                f"Normal\n(n={data['n_normal']})",
                f"Pre-AF\n(n={data['n_preaf']})"
            ])
            ax.set_title(
                f"{prefix}_{metric}\n"
                f"Mann-Whitney U: p={data['p_value']:.4f}, d={data['cohens_d']:.3f}"
            )
            ax.set_ylabel(metric)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{prefix}_{metric}_boxplot.png"),
                dpi=150
            )
            plt.close()

if __name__ == "__main__":
    PAF_OUTPUT_DIR = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf_updated/results_filtered"

    # Choose which method to analyze
    method = "dwt_recon"  # Options: "single_scale", "swt_recon", "swt_direct", "dwt_recon"

    method_files = {
        "single_scale": "single_scale_rqa.csv",
        "swt_recon": "multi_scale_swt_recon_rqa.csv",
        "swt_direct": "multi_scale_swt_direct_rqa.csv",
        "dwt_recon": "multi_scale_dwt_recon_rqa.csv",
    }

    csv_path = f"{PAF_OUTPUT_DIR}/{method_files[method]}"
    is_multiscale = method != "single_scale"

    print(f"\n{'='*80}")
    print(f"Statistical Analysis: {method.upper()}")
    print(f"Using Mann-Whitney U Test (Independent Samples)")
    print(f"Pre-AF vs Normal (n* patients only)")
    print(f"{'='*80}")

    stats = RQAStatistics(csv_path)

    if is_multiscale:
        stats.print_multiscale_summary()

        sig = stats.get_significant_results(
            alpha=0.05, prefixes=["D1", "D2", "D3", "D4", "A4"]
        )
        print(f"\n{'='*80}")
        print(f"Statistically Significant Results (p < 0.05):")
        print(f"{'='*80}")
        for r in sig:
            print(f"  {r['prefix']}_{r['metric']}: "
                  f"p={r['p_value']:.4f}, d={r['cohens_d']:.3f}, "
                  f"n_preaf={r['n_preaf']}, n_normal={r['n_normal']}")

        # Save results
        stats.save_results_csv(
            f"{PAF_OUTPUT_DIR}/stats_{method}.csv",
            prefixes=["D1", "D2", "D3", "D4", "A4"]
        )

    else:
        stats.print_summary(prefix="signal")

        sig = stats.get_significant_results(alpha=0.05, prefixes=["signal"])
        print(f"\n{'='*80}")
        print(f"Statistically Significant Results (p < 0.05):")
        print(f"{'='*80}")
        for r in sig:
            print(f"  {r['prefix']}_{r['metric']}: "
                  f"p={r['p_value']:.4f}, d={r['cohens_d']:.3f}, "
                  f"n_preaf={r['n_preaf']}, n_normal={r['n_normal']}")

        # Save results
        stats.save_results_csv(
            f"{PAF_OUTPUT_DIR}/stats_{method}.csv",
            prefixes=["signal"]
        )