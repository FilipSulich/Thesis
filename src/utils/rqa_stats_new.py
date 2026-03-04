import csv
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import os


class RQAStatistics:

    RQA_METRICS = [
        "epsilon",
        "recurrence_rate",
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
    ) -> Tuple[Dict, Dict, Dict]:
        patient_data = self._aggregate_per_patient(metric, prefix)

        paired_preaf  = {}
        paired_normal = {}
        indep_normal  = {}

        for (patient_id, label), info in patient_data.items():
            is_n_record = patient_id.startswith("paf_n")
            if label == 1:
                paired_preaf[patient_id]  = info["mean"]
            elif is_n_record:
                indep_normal[patient_id]  = info["mean"]
            else:
                paired_normal[patient_id] = info["mean"]

        return paired_preaf, paired_normal, indep_normal


    def compare_groups(self, metric: str, prefix: str = "signal") -> Dict:
        paired_preaf, paired_normal, indep_normal = self._split_groups(metric, prefix)

        if not paired_preaf:
            return {}

        common = sorted(set(paired_preaf) & set(paired_normal))
        paired_pre_vals  = np.array([paired_preaf[p]  for p in common])
        paired_norm_vals = np.array([paired_normal[p] for p in common])

        if len(common) >= 5:
            try:
                _, p_paired = wilcoxon(paired_pre_vals, paired_norm_vals, alternative='two-sided')
            except Exception:
                p_paired = float('nan')
            diffs         = paired_pre_vals - paired_norm_vals
            cohens_d_paired = (float(np.mean(diffs) / np.std(diffs, ddof=1))
                               if np.std(diffs, ddof=1) > 0 else 0.0)
        else:
            p_paired        = float('nan')
            cohens_d_paired = float('nan')

        all_preaf  = np.array(list(paired_preaf.values()))
        all_normal = np.array(list(paired_normal.values()) +
                              list(indep_normal.values()))

        if len(all_preaf) >= 2 and len(all_normal) >= 2:
            _, p_unpaired = mannwhitneyu(all_preaf, all_normal,
                                         alternative='two-sided')
            pooled_std    = np.sqrt((np.std(all_preaf,  ddof=1) ** 2 +
                                     np.std(all_normal, ddof=1) ** 2) / 2)
            cohens_d_unpaired = (float((np.mean(all_preaf) - np.mean(all_normal))
                                       / pooled_std)
                                 if pooled_std > 0 else 0.0)
        else:
            p_unpaired        = float('nan')
            cohens_d_unpaired = float('nan')

        return {
            "n_pairs":          len(common),
            "p_paired":         float(p_paired),
            "cohens_d_paired":  float(cohens_d_paired),
            "paired_pre_vals":  paired_pre_vals.tolist(),
            "paired_norm_vals": paired_norm_vals.tolist(),

            "n_preaf":            len(all_preaf),
            "n_normal":           len(all_normal),
            "n_indep_normal":     len(indep_normal),
            "p_unpaired":         float(p_unpaired),
            "cohens_d_unpaired":  float(cohens_d_unpaired),

            "preaf_mean":   float(np.mean(all_preaf)),
            "preaf_std":    float(np.std(all_preaf,  ddof=1)),
            "preaf_median": float(np.median(all_preaf)),
            "normal_mean":  float(np.mean(all_normal)),
            "normal_std":   float(np.std(all_normal, ddof=1)),
            "normal_median":float(np.median(all_normal)),

            "preaf_values":  all_preaf.tolist(),
            "normal_values": all_normal.tolist(),
        }

    def print_summary(self, prefix: str = "signal"):
        header = (f"\n{'Metric':<30} {'N_pairs':>8} {'N_pre':>6} {'N_norm':>7} "
                  f"{'Pre-AF (mean±std)':>22} {'Normal (mean±std)':>22} "
                  f"{'d_paired':>9} {'p_paired':>9} "
                  f"{'d_all':>7} {'p_all':>8}")
        print(header)
        print("-" * 155)

        for metric in self.RQA_METRICS:
            data = self.compare_groups(metric, prefix)
            if not data:
                continue

            preaf_str  = f"{data['preaf_mean']:.4f}±{data['preaf_std']:.4f}"
            normal_str = f"{data['normal_mean']:.4f}±{data['normal_std']:.4f}"
            p_pair_str = f"{data['p_paired']:.4f}" if not np.isnan(data['p_paired']) else "  nan"
            p_all_str  = f"{data['p_unpaired']:.4f}" if not np.isnan(data['p_unpaired']) else "  nan"
            d_pair_str = f"{data['cohens_d_paired']:.3f}" if not np.isnan(data['cohens_d_paired']) else " nan"
            d_all_str  = f"{data['cohens_d_unpaired']:.3f}" if not np.isnan(data['cohens_d_unpaired']) else "nan"

            print(
                f"{metric:<30} "
                f"{data['n_pairs']:>8} "
                f"{data['n_preaf']:>6} "
                f"{data['n_normal']:>7} "
                f"{preaf_str:>22} "
                f"{normal_str:>22} "
                f"{d_pair_str:>9} "
                f"{p_pair_str:>9} "
                f"{d_all_str:>7} "
                f"{p_all_str:>8}"
            )

    def print_multiscale_summary(self):
        for subband in ["D1", "D2", "D3", "D4", "A4"]:
            print(f"\n{'='*80}\nSUBBAND: {subband}\n{'='*80}")
            self.print_summary(prefix=subband)

    def get_significant_results(
        self,
        alpha: float = 0.05,
        prefixes: Optional[List[str]] = None,
        use_paired: bool = True,
    ) -> List[Dict]:
        if prefixes is None:
            prefixes = ["signal"]

        p_key = "p_paired" if use_paired else "p_unpaired"
        d_key = "cohens_d_paired" if use_paired else "cohens_d_unpaired"

        significant = []
        for prefix in prefixes:
            for metric in self.RQA_METRICS:
                data = self.compare_groups(metric, prefix)
                if not data:
                    continue
                p = data.get(p_key, float('nan'))
                if not np.isnan(p) and p < alpha:
                    significant.append({
                        "prefix":    prefix,
                        "metric":    metric,
                        "p_value":   p,
                        "cohens_d":  data.get(d_key, float('nan')),
                        "n_pairs":   data["n_pairs"],
                        "n_preaf":   data["n_preaf"],
                        "n_normal":  data["n_normal"],
                    })

        return sorted(significant, key=lambda x: x["p_value"])

    def save_results_csv(self, output_path: str,
                         prefixes: Optional[List[str]] = None):
        if prefixes is None:
            prefixes = ["signal"]

        fieldnames = [
            "prefix", "metric",
            "n_pairs", "n_preaf", "n_normal", "n_indep_normal",
            "preaf_mean", "preaf_std", "preaf_median",
            "normal_mean", "normal_std", "normal_median",
            "cohens_d_paired", "p_paired",
            "cohens_d_unpaired", "p_unpaired",
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
                        "prefix":           prefix,
                        "metric":           metric,
                        "n_pairs":          data["n_pairs"],
                        "n_preaf":          data["n_preaf"],
                        "n_normal":         data["n_normal"],
                        "n_indep_normal":   data["n_indep_normal"],
                        "preaf_mean":       data["preaf_mean"],
                        "preaf_std":        data["preaf_std"],
                        "preaf_median":     data["preaf_median"],
                        "normal_mean":      data["normal_mean"],
                        "normal_std":       data["normal_std"],
                        "normal_median":    data["normal_median"],
                        "cohens_d_paired":  data["cohens_d_paired"],
                        "p_paired":         data["p_paired"],
                        "cohens_d_unpaired":data["cohens_d_unpaired"],
                        "p_unpaired":       data["p_unpaired"],
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
                palette=["skyblue", "salmon"], ax=ax
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels([
                f"Normal\n(n={data['n_normal']})",
                f"Pre-AF\n(n={data['n_preaf']})"
            ])
            ax.set_title(
                f"{prefix}_{metric}\n"
                f"paired p={data['p_paired']:.4f}, d={data['cohens_d_paired']:.3f}"
            )
            ax.set_ylabel(metric)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{prefix}_{metric}_boxplot.png"),
                dpi=150
            )
            plt.close()

if __name__ == "__main__":
    PAF_OUTPUT_DIR     = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf_updated/results"
    paf_multiscale_path  = f"{PAF_OUTPUT_DIR}/multi_scale_rqa.csv"
    paf_singlescale_path = f"{PAF_OUTPUT_DIR}/single_scale_rqa.csv"

    multi_scale = True
    csv_path    = paf_multiscale_path if multi_scale else paf_singlescale_path

    stats = RQAStatistics(csv_path)

    if multi_scale:
        stats.print_multiscale_summary()

        sig = stats.get_significant_results(
            alpha=0.05, prefixes=["D1", "D2", "D3", "D4", "A4"],
            use_paired=True
        )
        print("statistically significant: ")
        for r in sig:
            print(f"  {r['prefix']}_{r['metric']}: "f"p={r['p_value']:.4f}, d={r['cohens_d']:.3f}, "f"n_pairs={r['n_pairs']}")

        # stats.save_results_csv(f"{PAF_OUTPUT_DIR}/rqa_stats_multiscale.csv",prefixes=["D1", "D2", "D3", "D4", "A4"])

    else:
        stats.print_summary(prefix="signal")

        sig = stats.get_significant_results(
            alpha=0.05, prefixes=["signal"], use_paired=True
        )
        print("statistically significant: ")
        for r in sig:
            print(f"  {r['prefix']}_{r['metric']}: "f"p={r['p_value']:.4f}, d={r['cohens_d']:.3f}")

        # stats.save_results_csv(f"{PAF_OUTPUT_DIR}/rqa_stats_singlescale.csv",prefixes=["signal"])