"""
Validation for RQA findings:
  1. Bootstrap stability — subsample segments per patient, check p/d consistency
  2. Permutation test — shuffle labels within pairs, verify null distribution

Usage:
    validator = RQAValidator(csv_path, metric="D3_longest_vertical_line")
    validator.run_all()
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# =============================================================================
# Data loading (mirrors RQAStatistics logic)
# =============================================================================

def load_patient_segments(csv_path: str, metric_col: str) -> Dict:
    """
    Load all segments from CSV and group by (patient_id, label).

    Returns:
        dict: {(patient_id, label): [segment_values]}
    """
    patient_segments = defaultdict(list)

    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            if metric_col not in row or not row[metric_col]:
                continue
            val_str = row[metric_col].strip().lower()
            if val_str in ("nan", "inf", "-inf", ""):
                continue
            patient_id = row["patient_id"]
            label      = int(row["label"])
            value      = float(row[metric_col])
            patient_segments[(patient_id, label)].append(value)

    return dict(patient_segments)


def get_paired_means(
    patient_segments: Dict,
    rng: Optional[np.random.Generator] = None,
    frac: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute per-patient means for p-pair patients only.

    Parameters
    ----------
    patient_segments : dict
        {(patient_id, label): [values]}
    rng : np.random.Generator or None
        If provided, subsample frac of segments before averaging.
    frac : float
        Fraction of segments to subsample per patient per label.

    Returns
    -------
    paired_pre  : np.ndarray  — pre-AF means, one per pair
    paired_norm : np.ndarray  — normal means, one per pair
    pair_ids    : List[str]   — patient IDs for reference
    """
    # Collect patient IDs that have both label=0 and label=1 entries
    # (these are the p-pair patients)
    patients_with_preaf  = {pid for (pid, lbl) in patient_segments if lbl == 1}
    patients_with_normal = {pid for (pid, lbl) in patient_segments if lbl == 0
                            and not pid.startswith("paf_n")}
    paired_patients = sorted(patients_with_preaf & patients_with_normal)

    paired_pre, paired_norm = [], []

    for pid in paired_patients:
        pre_vals  = patient_segments.get((pid, 1), [])
        norm_vals = patient_segments.get((pid, 0), [])

        if not pre_vals or not norm_vals:
            continue

        if rng is not None and frac < 1.0:
            n_pre  = max(1, int(len(pre_vals)  * frac))
            n_norm = max(1, int(len(norm_vals) * frac))
            idx_pre  = rng.choice(len(pre_vals),  size=n_pre,  replace=False)
            idx_norm = rng.choice(len(norm_vals), size=n_norm, replace=False)
            pre_vals  = [pre_vals[i]  for i in idx_pre]
            norm_vals = [norm_vals[i] for i in idx_norm]

        paired_pre.append(np.mean(pre_vals))
        paired_norm.append(np.mean(norm_vals))

    return np.array(paired_pre), np.array(paired_norm), paired_patients


def cohens_d_paired(pre: np.ndarray, norm: np.ndarray) -> float:
    diffs = pre - norm
    std   = np.std(diffs, ddof=1)
    return float(np.mean(diffs) / std) if std > 0 else 0.0


# =============================================================================
# Validator
# =============================================================================

class RQAValidator:

    def __init__(self, csv_path: str, metric_col: str, output_dir: str = None):
        """
        Parameters
        ----------
        csv_path   : str   Path to RQA results CSV (single or multi-scale).
        metric_col : str   Column to validate, e.g. "D3_longest_vertical_line".
        output_dir : str   Directory to save plots. Defaults to CSV directory.
        """
        self.csv_path   = csv_path
        self.metric_col = metric_col
        self.output_dir = Path(output_dir or Path(csv_path).parent)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading {metric_col} from {csv_path}...")
        self.patient_segments = load_patient_segments(csv_path, metric_col)

        # Full-data paired means (baseline)
        self.pre_full, self.norm_full, self.pair_ids = get_paired_means(
            self.patient_segments
        )
        _, self.p_observed = wilcoxon(self.pre_full, self.norm_full,
                                       alternative='two-sided')
        self.d_observed    = cohens_d_paired(self.pre_full, self.norm_full)

        print(f"Observed: n_pairs={len(self.pre_full)}, "
              f"p={self.p_observed:.4f}, d={self.d_observed:.3f}")

    # ------------------------------------------------------------------
    # 1. Bootstrap stability
    # ------------------------------------------------------------------

    def bootstrap_stability(
        self,
        n_iterations: int = 1000,
        frac: float = 0.5,
        seed: int = 42,
        plot: bool = True,
    ) -> Dict:
        """
        Repeatedly subsample frac of segments per patient and recompute
        the paired Wilcoxon test and Cohen's d.

        A stable finding produces:
          - p < 0.05 in >70% of iterations
          - Tight distribution of d centered near observed value
          - Consistent sign of d across all iterations

        Parameters
        ----------
        n_iterations : int    Number of bootstrap iterations.
        frac         : float  Fraction of segments to keep per patient (0-1).
        seed         : int    Random seed.
        plot         : bool   Save distribution plots.

        Returns
        -------
        dict with stability metrics.
        """
        rng       = np.random.default_rng(seed)
        p_values  = []
        d_values  = []
        failed    = 0

        print(f"\n--- Bootstrap stability ({n_iterations} iterations, frac={frac}) ---")

        for i in range(n_iterations):
            pre, norm, _ = get_paired_means(
                self.patient_segments, rng=rng, frac=frac
            )
            if len(pre) < 5:
                failed += 1
                continue
            try:
                _, p = wilcoxon(pre, norm, alternative='two-sided')
                d    = cohens_d_paired(pre, norm)
                p_values.append(p)
                d_values.append(d)
            except Exception:
                failed += 1

        p_values = np.array(p_values)
        d_values = np.array(d_values)

        pct_significant = 100 * np.mean(p_values < 0.05)
        pct_same_sign   = 100 * np.mean(np.sign(d_values) == np.sign(self.d_observed))

        print(f"  Significant (p<0.05): {pct_significant:.1f}% of iterations")
        print(f"  Consistent sign of d: {pct_same_sign:.1f}% of iterations")
        print(f"  Median d  = {np.median(d_values):.3f}  "
              f"(95% CI: [{np.percentile(d_values, 2.5):.3f}, "
              f"{np.percentile(d_values, 97.5):.3f}])")
        print(f"  Median p  = {np.median(p_values):.4f}")
        print(f"  Failed iterations: {failed}/{n_iterations}")

        if pct_significant >= 70:
            print("  ✓ Finding is STABLE across segment subsamples")
        elif pct_significant >= 40:
            print("  ~ Finding is MARGINAL — moderate segment dependence")
        else:
            print("  ✗ Finding is UNSTABLE — likely driven by specific segments")

        if plot:
            self._plot_bootstrap(p_values, d_values)

        return {
            "n_iterations":      n_iterations,
            "frac":              frac,
            "failed":            failed,
            "pct_significant":   pct_significant,
            "pct_same_sign":     pct_same_sign,
            "median_d":          float(np.median(d_values)),
            "d_ci_low":          float(np.percentile(d_values, 2.5)),
            "d_ci_high":         float(np.percentile(d_values, 97.5)),
            "median_p":          float(np.median(p_values)),
            "d_values":          d_values.tolist(),
            "p_values":          p_values.tolist(),
        }

    def _plot_bootstrap(self, p_values: np.ndarray, d_values: np.ndarray):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"Bootstrap stability: {self.metric_col}", fontsize=12)

        # p-value distribution
        axes[0].hist(p_values, bins=30, color='steelblue', edgecolor='white')
        axes[0].axvline(0.05, color='red', linestyle='--', label='p=0.05')
        axes[0].axvline(self.p_observed, color='orange', linestyle='-',
                        label=f'observed p={self.p_observed:.4f}')
        axes[0].set_xlabel("p-value")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"p<0.05 in {100*np.mean(p_values<0.05):.1f}% of iterations")
        axes[0].legend(fontsize=8)

        # d distribution
        axes[1].hist(d_values, bins=30, color='salmon', edgecolor='white')
        axes[1].axvline(0, color='black', linestyle='--', alpha=0.5)
        axes[1].axvline(self.d_observed, color='orange', linestyle='-',
                        label=f'observed d={self.d_observed:.3f}')
        axes[1].axvline(np.percentile(d_values, 2.5),  color='gray',
                        linestyle=':', label='95% CI')
        axes[1].axvline(np.percentile(d_values, 97.5), color='gray', linestyle=':')
        axes[1].set_xlabel("Cohen's d")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Median d = {np.median(d_values):.3f}")
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        out = self.output_dir / f"bootstrap_{self.metric_col}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Plot saved: {out}")

    # ------------------------------------------------------------------
    # 2. Permutation test
    # ------------------------------------------------------------------

    def permutation_test(
        self,
        n_permutations: int = 10000,
        seed: int = 42,
        plot: bool = True,
    ) -> Dict:
        """
        Permutation test: randomly flip the sign of within-pair differences
        and recompute the Wilcoxon statistic. The observed statistic should
        sit in the extreme tail of the null distribution.

        Parameters
        ----------
        n_permutations : int   Number of permutations.
        seed           : int   Random seed.
        plot           : bool  Save null distribution plot.

        Returns
        -------
        dict with permutation p-value and null distribution.
        """
        rng = np.random.default_rng(seed)

        diffs         = self.pre_full - self.norm_full
        observed_stat, _ = wilcoxon(self.pre_full, self.norm_full,
                                     alternative='two-sided')

        print(f"\n--- Permutation test ({n_permutations} permutations) ---")
        print(f"  Observed Wilcoxon statistic = {observed_stat:.4f}")

        null_stats = []
        for _ in range(n_permutations):
            # Randomly flip signs within pairs — this is the null:
            # pre-AF and normal are exchangeable within each pair
            signs          = rng.choice([-1, 1], size=len(diffs))
            permuted_diffs = diffs * signs
            # Reconstruct permuted arrays for wilcoxon
            permuted_pre  = self.norm_full + permuted_diffs
            permuted_norm = self.norm_full
            try:
                stat, _ = wilcoxon(permuted_pre, permuted_norm,
                                    alternative='two-sided')
                null_stats.append(stat)
            except Exception:
                pass

        null_stats   = np.array(null_stats)

        # Two-sided: count how often null stat is as extreme as observed
        # Wilcoxon statistic: smaller = more extreme for two-sided
        p_permutation = float(np.mean(null_stats <= observed_stat))

        print(f"  Permutation p = {p_permutation:.4f}")
        print(f"  Parametric  p = {self.p_observed:.4f}")

        if abs(p_permutation - self.p_observed) < 0.01:
            print("  ✓ Parametric and permutation p-values agree — "
                  "test assumptions are valid")
        else:
            print("  ! Parametric and permutation p-values diverge — "
                  "interpret parametric p with caution")

        if p_permutation < 0.05:
            print("  ✓ Finding survives permutation test")
        else:
            print("  ✗ Finding does not survive permutation test")

        if plot:
            self._plot_permutation(null_stats, observed_stat, p_permutation)

        return {
            "n_permutations":  n_permutations,
            "observed_stat":   float(observed_stat),
            "p_permutation":   p_permutation,
            "p_parametric":    self.p_observed,
            "null_mean":       float(np.mean(null_stats)),
            "null_std":        float(np.std(null_stats)),
            "null_stats":      null_stats.tolist(),
        }

    def _plot_permutation(
        self, null_stats: np.ndarray,
        observed_stat: float,
        p_permutation: float,
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(null_stats, bins=50, color='steelblue', edgecolor='white',
                label='Null distribution')
        ax.axvline(observed_stat, color='red', linewidth=2,
                   label=f'Observed stat={observed_stat:.2f}')
        ax.set_xlabel("Wilcoxon statistic")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Permutation test: {self.metric_col}\n"
            f"Permutation p={p_permutation:.4f}, "
            f"Parametric p={self.p_observed:.4f}"
        )
        ax.legend()
        plt.tight_layout()
        out = self.output_dir / f"permutation_{self.metric_col}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Plot saved: {out}")

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------

    def run_all(
        self,
        bootstrap_iterations: int = 1000,
        bootstrap_frac: float = 0.5,
        permutation_iterations: int = 10000,
        seed: int = 42,
    ) -> Dict:
        print(f"\n{'='*60}")
        print(f"VALIDATION: {self.metric_col}")
        print(f"{'='*60}")

        bootstrap_results   = self.bootstrap_stability(
            n_iterations=bootstrap_iterations,
            frac=bootstrap_frac,
            seed=seed,
        )
        permutation_results = self.permutation_test(
            n_permutations=permutation_iterations,
            seed=seed,
        )

        print(f"\n--- Summary ---")
        print(f"  Observed:         p={self.p_observed:.4f}, d={self.d_observed:.3f}")
        print(f"  Bootstrap stable: {bootstrap_results['pct_significant']:.1f}% significant")
        print(f"  Permutation p:    {permutation_results['p_permutation']:.4f}")

        return {
            "metric":      self.metric_col,
            "observed_p":  self.p_observed,
            "observed_d":  self.d_observed,
            "bootstrap":   bootstrap_results,
            "permutation": permutation_results,
        }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    PAF_OUTPUT_DIR = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf_updated/results"

    # Validate the two significant findings
    metrics_to_validate = [
        ("multi_scale_rqa.csv",  "D3_longest_vertical_line"),
        ("single_scale_rqa.csv", "signal_longest_vertical_line"),
    ]

    all_results = {}
    for csv_file, metric in metrics_to_validate:
        csv_path  = f"{PAF_OUTPUT_DIR}/{csv_file}"
        validator = RQAValidator(
            csv_path=csv_path,
            metric_col=metric,
            output_dir=f"{PAF_OUTPUT_DIR}/validation",
        )
        results = validator.run_all(
            bootstrap_iterations=1000,
            bootstrap_frac=0.5,
            permutation_iterations=10000,
            seed=42,
        )
        all_results[metric] = results