import csv
import numpy as np
from typing import Dict, List, Optional
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import os


class RQAStatistics:

    RQA_METRICS = [
        "recurrence_rate",
        "determinism",
        "average_diagonal_line",
        "longest_diagonal_line",
        # "divergence",
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

    def _get_record_id(self, segment_id: str) -> str:
        """
        Extract record_id from segment_id.
        e.g. "paf_n01_s0_l0" -> "n01"
             "paf_p02_s5_l0" -> "p02"
        """
        parts = segment_id.split('_')
        return parts[1]

    def _aggregate_per_record(self, metric: str, prefix: str = "signal") -> Dict:
        """
        Aggregate RQA metric values per record by averaging all segments
        belonging to that record.

        Returns:
            dict: record_id -> {"label": int, "mean": float}
        """
        col_name = f"{prefix}_{metric}"
        record_dict = {}

        for row in self.data:
            if col_name not in row or not row[col_name]:
                continue
            if row[col_name].strip().lower() == "nan":
                continue

            record_id = self._get_record_id(row["segment_id"])
            label = int(row["label"])
            value = float(row[col_name])

            if record_id not in record_dict:
                record_dict[record_id] = {"label": label, "values": []}

            record_dict[record_id]["values"].append(value)

        # Average segments within each record
        result = {}
        for record_id, data in record_dict.items():
            if data["values"]:
                result[record_id] = {
                    "label": data["label"],
                    "mean": float(np.mean(data["values"]))
                }

        return result

    def compare_groups(self, metric: str, prefix: str = "signal") -> Dict:
        """
        Compare pre-AF vs normal at the record level using Mann-Whitney U test.
        Groups are unpaired (different patients in each group).
        """
        record_data = self._aggregate_per_record(metric, prefix)

        preaf = []
        normal = []

        for record_id, data in record_data.items():
            if data["label"] == 1:
                preaf.append(data["mean"])
            else:
                normal.append(data["mean"])

        preaf = np.array(preaf)
        normal = np.array(normal)

        if len(preaf) == 0 or len(normal) == 0:
            return {}

        result = {}
        result["n_preaf"] = len(preaf)
        result["n_normal"] = len(normal)
        result["preaf_mean"] = float(np.mean(preaf))
        result["preaf_std"] = float(np.std(preaf, ddof=1))
        result["preaf_median"] = float(np.median(preaf))
        result["normal_mean"] = float(np.mean(normal))
        result["normal_std"] = float(np.std(normal, ddof=1))
        result["normal_median"] = float(np.median(normal))

        # Mann-Whitney U — unpaired, non-parametric
        stat, p_value = mannwhitneyu(preaf, normal, alternative='two-sided')
        result["p_value"] = float(p_value)

        # Unpaired Cohen's d using pooled standard deviation
        pooled_std = np.sqrt(
            (np.std(preaf, ddof=1) ** 2 + np.std(normal, ddof=1) ** 2) / 2
        )
        result["cohens_d"] = (
            float((np.mean(preaf) - np.mean(normal)) / pooled_std)
            if pooled_std > 0 else 0.0
        )

        # Store record-level values for plotting
        result["preaf_values"] = preaf.tolist()
        result["normal_values"] = normal.tolist()

        return result

    def print_summary(self, prefix: str = "signal"):
        print(f"\n{'Metric':<30} {'N_pre':>6} {'N_norm':>7} "
              f"{'Pre-AF (mean±std)':>22} {'Normal (mean±std)':>22} "
              f"{'Pre-AF (median)':>16} {'Normal (median)':>16} "
              f"{'Cohen d':>9} {'P':>8}")
        print("-" * 155)
        for metric in self.RQA_METRICS:
            data = self.compare_groups(metric, prefix)
            if not data:
                continue
            preaf_str = f"{data['preaf_mean']:.4f}±{data['preaf_std']:.4f}"
            normal_str = f"{data['normal_mean']:.4f}±{data['normal_std']:.4f}"
            preaf_median_str = f"{data['preaf_median']:.4f}"
            normal_median_str = f"{data['normal_median']:.4f}"
            print(
                f"{metric:<30} "
                f"{data['n_preaf']:>6} "
                f"{data['n_normal']:>7} "
                f"{preaf_str:>22} "
                f"{normal_str:>22} "
                f"{preaf_median_str:>16} "
                f"{normal_median_str:>16} "
                f"{data['cohens_d']:>9.3f} "
                f"{data['p_value']:>8.4f}"
            )

    def print_multiscale_summary(self):
        subbands = ["D1", "D2", "D3", "D4", "A4"]
        for subband in subbands:
            print(f"\n{'='*80}")
            print(f"SUBBAND: {subband}")
            print(f"{'='*80}")
            self.print_summary(prefix=subband)

    def get_significant_results(
        self,
        alpha: float = 0.05,
        prefixes: Optional[List[str]] = None
    ) -> List[Dict]:
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
                        "n_preaf": data["n_preaf"],
                        "n_normal": data["n_normal"],
                        "preaf_mean": data["preaf_mean"],
                        "normal_mean": data["normal_mean"],
                    })

        return sorted(significant, key=lambda x: x["p_value"])

    def save_results_csv(self, output_path: str, prefixes: Optional[List[str]] = None):
        if prefixes is None:
            prefixes = ["signal"]

        fieldnames = [
            "prefix", "metric", "n_preaf", "n_normal",
            "preaf_mean", "preaf_std", "preaf_median",
            "normal_mean", "normal_std", "normal_median",
            "cohens_d", "p_value"
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
                        "prefix": prefix,
                        "metric": metric,
                        "n_preaf": data["n_preaf"],
                        "n_normal": data["n_normal"],
                        "preaf_mean": data["preaf_mean"],
                        "preaf_std": data["preaf_std"],
                        "preaf_median": data["preaf_median"],
                        "normal_mean": data["normal_mean"],
                        "normal_std": data["normal_std"],
                        "normal_median": data["normal_median"],
                        "cohens_d": data["cohens_d"],
                        "p_value": data["p_value"],
                    })

        print(f"Saved results to {output_path}")

    def plot_metrics(self, output_dir: str, prefix: str = "signal"):
        os.makedirs(output_dir, exist_ok=True)

        for metric in self.RQA_METRICS:
            data = self.compare_groups(metric, prefix)
            if not data:
                continue

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                data=[data["normal_values"], data["preaf_values"]],
                palette=["skyblue", "salmon"],
                ax=ax
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels([
                f"Normal\n(n={data['n_normal']})",
                f"Pre-AF\n(n={data['n_preaf']})"
            ])
            ax.set_title(
                f"{prefix}_{metric}\np={data['p_value']:.4f}, d={data['cohens_d']:.3f}"
            )
            ax.set_ylabel(metric)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_{metric}_boxplot.png"), dpi=150)
            plt.close()


if __name__ == "__main__":
    # PAF dataset paths
    PAF_OUTPUT_DIR = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/output"
    paf_multiscale_path = f"{PAF_OUTPUT_DIR}/results_multi_scale_all_clean_new_20_std.csv"
    paf_singlescale_path = f"{PAF_OUTPUT_DIR}/single_scale_10_dist.csv"

    # Configuration
    multi_scale = True
    csv_path = paf_multiscale_path if multi_scale else paf_singlescale_path

    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"ERROR: Results file not found: {csv_path}")
        print("\nTo generate results, run the PAF pipeline:")
        print("  from pipeline_paf.pipeline_paf import PAFPipeline")
        print("  pipeline = PAFPipeline()")
        print("  results = pipeline.run_batch(split='all', single_scale=False)")
        print("  pipeline.results_to_csv(results, Path(csv_path))")
        exit(1)

    stats = RQAStatistics(csv_path)

    if multi_scale:
        print("=" * 80)
        print("PAF MULTI-SCALE RQA RESULTS")
        print("=" * 80)
        stats.print_multiscale_summary()

        print("\n" + "=" * 80)
        print("SIGNIFICANT RESULTS (p < 0.05)")
        print("=" * 80)
        sig = stats.get_significant_results(
            alpha=0.05,
            prefixes=["D1", "D2", "D3", "D4", "A4"]
        )
        if sig:
            for r in sig:
                print(f"{r['prefix']}_{r['metric']}: p={r['p_value']:.4f}, d={r['cohens_d']:.3f}")
        else:
            print("No significant results found.")

        # Save detailed results
        stats.save_results_csv(f"{PAF_OUTPUT_DIR}/rqa_stats_multiscale.csv",
                               prefixes=["D1", "D2", "D3", "D4", "A4"])
    else:
        print("=" * 80)
        print("PAF SINGLE-SCALE RQA RESULTS")
        print("=" * 80)
        stats.print_summary(prefix="signal")

        print("\n" + "=" * 80)
        print("SIGNIFICANT RESULTS (p < 0.05)")
        print("=" * 80)
        sig = stats.get_significant_results(alpha=0.05, prefixes=["signal"])
        if sig:
            for r in sig:
                print(f"{r['prefix']}_{r['metric']}: p={r['p_value']:.4f}, d={r['cohens_d']:.3f}")
        else:
            print("No significant results found.")

        # Save detailed results
        stats.save_results_csv(f"{PAF_OUTPUT_DIR}/rqa_stats_singlescale.csv",
                               prefixes=["signal"])