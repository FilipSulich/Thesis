import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pywt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import friedmanchisquare, wilcoxon

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.wavelet_selection_tests.urp_discrminability.urp_features import URPFeatures
from src.wavelet_selection_tests.utils import get_subject_group
from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator
CANDIDATE_WAVELETS = (
    [f'db{i}' for i in range(2, 9)] +
    [f'sym{i}' for i in range(2, 9)] +
    [f'coif{i}' for i in range(1, 4)]
)
SUBBAND_LEVELS = list(range(2, 9))  # D2-D8
N_SWT_LEVELS = 10 #decomposition levels
FS = 500 #target sampling frequency of the ECG data (after resampling)

class WaveletEvaluator:
    """evaluates wavelets for URP discriminability using logistic regression on texture features"""

    def __init__(self, signals, labels, record_ids, embedding_params, urp_size=256, n_cv_splits=5, results_dir=None):
        """
        initializes wavelet evaluator with ECG data and embedding parameters

        Args:
            signals (np.array): ECG segments array of shape (N, signal_length)
            labels (np.array): binary labels (0=NSR, 1=Pre-AF) of shape (N,)
            record_ids (np.array): patient record IDs for grouping of shape (N,)
            embedding_params (dict): dictionary with tau/m parameters per subband
            urp_size (int): target URP image size in pixels
            n_cv_splits (int): number of cross-validation folds
            results_dir (Path): directory path for saving results

        Returns:
            None
        """
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.int32)
        self.record_ids = record_ids
        self.embedding_params = embedding_params
        self.urp_size = urp_size
        self.n_cv_splits = n_cv_splits
        self.results_dir = Path(results_dir) if results_dir else Path('wavelet_selection')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.pad_samples = math.ceil(signals.shape[1] / (2 ** N_SWT_LEVELS)) * (2 ** N_SWT_LEVELS)

        self.results = {}

    def _pad_signal(self, signal):
        """
        pads signal to dyadic length required for SWT decomposition

        Args:
            signal (np.array): input signal array

        Returns:
            np.array: padded signal with reflection mode
        """
        pad = self.pad_samples - len(signal)
        if pad <= 0:
            return signal[:self.pad_samples]
        return np.pad(signal, (0, pad), mode='reflect').astype(np.float32)

    def _extract_subband(self, signal_padded, wavelet, level):
        """
        extracts single frequency subband using SWT decomposition and reconstruction

        Args:
            signal_padded (np.array): zero-padded signal ready for SWT
            wavelet (str): wavelet function name
            level (int): decomposition level to extract (2-8 for D2-D8)

        Returns:
            np.array: reconstructed subband signal
        """
        coeffs = pywt.swt(signal_padded, wavelet, level=N_SWT_LEVELS)
        n = len(coeffs)
        target_idx = n - level

        zeroed = []
        for j, (cA, cD) in enumerate(coeffs):
            if j == target_idx:
                zeroed.append((np.zeros_like(cA), cD))
            else:
                zeroed.append((np.zeros_like(cA), np.zeros_like(cD)))

        sub = pywt.iswt(zeroed, wavelet)[:self.signals.shape[1]]
        return sub.astype(np.float32)

    def _signal_to_urp(self, signal, tau, m):
        """
        converts signal to URP image using phase-space embedding and JET colormap

        Args:
            signal (np.array): input signal to convert
            tau (int): time delay for phase space reconstruction
            m (int): embedding dimension for phase space reconstruction

        Returns:
            np.array: RGB URP image of shape (urp_size, urp_size, 3) normalized to [0,1]
        """
        gen = RecurrencePlotGenerator(
            signal=signal,
            time_delay=tau,
            embedding_dim=m,
            epsilon=0
        )
        rp = gen.compute_rp_unthresholded()

        #min-max normalize
        d_max = rp.max()
        if d_max > 1e-9:
            rp = rp / d_max

        #resize to 256 x 256 and fill the diagonal with 0s
        rp_r = cv2.resize(rp.astype(np.float32),(self.urp_size, self.urp_size),interpolation=cv2.INTER_LINEAR)
        np.fill_diagonal(rp_r, 0.0)


        # color map JET encoding
        u8 = (rp_r * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        return rgb

    def _build_features(self, wavelet, level):
        """
        builds feature matrix by extracting texture features from URPs for one wavelet-subband combination

        Args:
            wavelet (str): wavelet function name
            level (int): subband decomposition level (2-8)

        Returns:
            tuple: (features array, labels array, record IDs list)
        """
        name = f'D{level}'
        tau = self.embedding_params[name]['tau']
        m = self.embedding_params[name]['m']

        X_list, y_list, rec_list = [], [], []
        for sig, label, rec_id in zip(self.signals, self.labels, self.record_ids):
            try:
                pad = self._pad_signal(sig)
                sub = self._extract_subband(pad, wavelet, level)
                urp = self._signal_to_urp(sub, tau, m)
                feat = URPFeatures.extract(urp)

                if not np.all(np.isfinite(feat)):
                    continue

                X_list.append(feat)
                y_list.append(label)
                rec_list.append(rec_id)

            except Exception as e:
                continue

        return np.array(X_list, dtype=np.float32), np.array(y_list), rec_list

    def _cross_validate(self, X, y, record_ids):
        """
        performs grouped cross-validation using logistic regression with balanced class weights

        Args:
            X (np.array): feature matrix of shape (N, n_features)
            y (np.array): binary labels of shape (N,)
            record_ids (list): patient record IDs for grouping

        Returns:
            dict: cross-validation results with mean, std, and fold-wise AUC scores
        """
        clf = make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
        groups = np.array([get_subject_group(r) for r in record_ids])
        cv = GroupKFold(n_splits=self.n_cv_splits)
        scores = []

        for train_idx, test_idx in cv.split(X, y, groups):
            clf.fit(X[train_idx], y[train_idx])
            prob = clf.predict_proba(X[test_idx])[:, 1]
            scores.append(float(roc_auc_score(y[test_idx], prob)))

        scores = np.array(scores)
        return {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'values': scores.tolist(),
        }

    def evaluate_wavelet(self, wavelet):
        """
        evaluates single wavelet across all subbands (D2-D8) using cross-validated AUC

        Args:
            wavelet (str): wavelet function name to evaluate

        Returns:
            dict: summary results with median, mean, min, max AUC and per-subband details
        """
        print(f"wavelet: {wavelet}")
        subband_results = {}

        for level in SUBBAND_LEVELS:
            name = f'D{level}'
            X, y, rec_ids = self._build_features(wavelet, level)

            # Need enough samples for CV
            if len(X) < 20 or y.sum() < 5 or (len(y) - y.sum()) < 5:
                subband_results[name] = None
                continue

            scores = self._cross_validate(X, y, rec_ids)
            print(f" AUC={scores['mean']:.4f}")
            subband_results[name] = scores

        valid_means = [v['mean'] for v in subband_results.values() if v is not None]
        if not valid_means:
            summary = {
                'median_auc': 0.0,
                'mean_auc': 0.0,
                'min_auc': 0.0,
                'max_auc': 0.0,
                'per_subband': subband_results,
            }
        else:
            summary = {
                'median_auc': float(np.median(valid_means)),
                'mean_auc': float(np.mean(valid_means)),
                'min_auc': float(np.min(valid_means)),
                'max_auc': float(np.max(valid_means)),
                'per_subband': subband_results,
            }
        self.results[wavelet] = summary
        return summary

    def run_all(self):
        """
        evaluates all candidate wavelets and returns ranked results

        Returns:
            pd.DataFrame: ranked wavelets by mean AUC score
        """
        for wavelet in CANDIDATE_WAVELETS:
            self.evaluate_wavelet(wavelet)
        return self.rank_wavelets()

    def rank_wavelets(self):
        """
        ranks all evaluated wavelets by mean AUC in descending order

        Returns:
            pd.DataFrame: sorted dataframe with columns [wavelet, median_auc, mean_auc, min_auc, max_auc]
        """
        rows = []
        for wavelet, res in self.results.items():
            rows.append({
                'wavelet': wavelet,
                'median_auc': res['median_auc'],
                'mean_auc': res['mean_auc'],
                'min_auc': res['min_auc'],
                'max_auc': res['max_auc'],
            })

        df = pd.DataFrame(rows).sort_values('mean_auc', ascending=False).reset_index(drop=True)
        return df

    def perform_statistical_tests(self, alpha=0.05):
        """
        perform friedman test and pairwise wilcoxon tests for each subband.

        args:
            alpha: significance level (default 0.05)

        returns:
            dict with friedman and pairwise test results per subband
        """

        stats_results = {}

        for level in SUBBAND_LEVELS:
            name = f'D{level}'

            wavelet_scores = {}
            for wavelet, res in self.results.items():
                if res['per_subband'][name] is not None:
                    wavelet_scores[wavelet] = res['per_subband'][name]['values']

            if len(wavelet_scores) < 3:
                stats_results[name] = None
                continue

            groups = list(wavelet_scores.values())
            stat, p_value = friedmanchisquare(*groups)

            friedman_result = {
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': bool(p_value < alpha)
            }

            comparisons = []

            wavelet_names = list(wavelet_scores.keys())
            for i in range(len(wavelet_names)):
                for j in range(i + 1, len(wavelet_names)):
                    w1, w2 = wavelet_names[i], wavelet_names[j]
                    scores1 = np.array(wavelet_scores[w1])
                    scores2 = np.array(wavelet_scores[w2])

                    # wilcoxon signed-rank test
                    result = wilcoxon(scores1, scores2, alternative='two-sided')

                    # rosenthal's r effect size
                    n = len(scores1)
                    r = result.zstatistic / np.sqrt(n) if hasattr(result, 'zstatistic') else 0.0

                    comparisons.append({
                        'comparison': f'{w1}_vs_{w2}',
                        'p_value': float(result.pvalue),
                        'effect_size': float(r)
                    })

            # holm-bonferroni correction
            comparisons.sort(key=lambda x: x['p_value'])
            n_comparisons = len(comparisons)

            for idx, comp in enumerate(comparisons):
                adj_alpha = alpha / (n_comparisons - idx)
                comp['adj_alpha'] = adj_alpha
                comp['significant_corrected'] = comp['p_value'] < adj_alpha

                if not comp['significant_corrected']:
                    for j in range(idx + 1, len(comparisons)):
                        comparisons[j]['adj_alpha'] = alpha / (n_comparisons - j)
                        comparisons[j]['significant_corrected'] = False
                    break

            stats_results[name] = {
                'friedman': friedman_result,
                'pairwise': comparisons,
                'n_wavelets': len(wavelet_scores),
                'wavelets': wavelet_names
            }

        return stats_results

    def save_results(self):
        """
        saves all wavelet evaluation results to JSON file

        Returns:
            None
        """
        path = self.results_dir / 'wavelet_selection_results.json'
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)

    def save_statistical_tests(self, stats_results):
        """
        saves statistical test results to JSON file

        Args:
            stats_results (dict): dictionary containing statistical test results

        Returns:
            None
        """
        path = self.results_dir / 'wavelet_statistical_tests.json'
        with open(path, 'w') as f:
            json.dump(stats_results, f, indent=2)

    @staticmethod
    def perform_statistical_tests_from_json(json_path, alpha=0.05, output_path=None):
        """
        perform friedman test comparing wavelets based on median AUC across all subbands.

        for each wavelet, computes fold-wise median AUC across all subbands (D2-D8), then compares these medians using friedman test and pairwise wilcoxon tests.

        args:
            json_path: path to wavelet_selection_results.json
            alpha: significance level (default 0.05)
            output_path: where to save statistical test results (default: same dir as json_path)

        returns:
            dict with statistical test results
        """
        json_path = Path(json_path)

        with open(json_path) as f:
            results = json.load(f)


        wavelet_median_scores = {}
        wavelet_summary = {}

        for wavelet, res in results.items():
            n_folds = None
            subband_fold_aucs = {}

            for level in SUBBAND_LEVELS:
                name = f'D{level}'
                if res['per_subband'][name] is not None:
                    fold_aucs = res['per_subband'][name]['values']
                    subband_fold_aucs[name] = fold_aucs
                    if n_folds is None:
                        n_folds = len(fold_aucs)

            if not subband_fold_aucs:
                continue

            median_aucs = []
            for fold_idx in range(n_folds):
                fold_values = [subband_fold_aucs[name][fold_idx] for name in subband_fold_aucs.keys()]
                median_aucs.append(np.median(fold_values))

            median_aucs = np.array(median_aucs)
            wavelet_median_scores[wavelet] = median_aucs

            wavelet_summary[wavelet] = {
                'median': float(np.median(median_aucs)),
                'mean': float(np.mean(median_aucs)),
                'std': float(np.std(median_aucs)),
                'min': float(np.min(median_aucs)),
                'max': float(np.max(median_aucs)),
                'values': median_aucs.tolist()
            }

        ranked = sorted(wavelet_summary.items(), key=lambda x: x[1]['median'], reverse=True)

        if len(wavelet_median_scores) < 3:
            return None

        groups = list(wavelet_median_scores.values())
        stat, p_value = friedmanchisquare(*groups)

        friedman_result = {
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant': bool(p_value < alpha),
            'n_wavelets': len(wavelet_median_scores),
            'n_folds': len(groups[0])
        }

        comparisons = []
        wavelet_names = list(wavelet_median_scores.keys())

        for i in range(len(wavelet_names)):
            for j in range(i + 1, len(wavelet_names)):
                w1, w2 = wavelet_names[i], wavelet_names[j]
                scores1 = wavelet_median_scores[w1]
                scores2 = wavelet_median_scores[w2]

                # wilcoxon signed-rank test
                result = wilcoxon(scores1, scores2, alternative='two-sided')

                # rosenthal's r effect size
                n = len(scores1)
                r = result.zstatistic / np.sqrt(n) if hasattr(result, 'zstatistic') else 0.0

                comparisons.append({
                    'comparison': f'{w1}_vs_{w2}',
                    'wavelet_1': w1,
                    'wavelet_2': w2,
                    'p_value': float(result.pvalue),
                    'effect_size': float(r),
                    'median_diff': float(np.median(scores1) - np.median(scores2)),
                    'mean_diff': float(np.mean(scores1) - np.mean(scores2))
                })

        # holm-bonferroni correction
        comparisons.sort(key=lambda x: x['p_value'])
        n_comparisons = len(comparisons)

        for idx, comp in enumerate(comparisons):
            adj_alpha = alpha / (n_comparisons - idx)
            comp['adj_alpha'] = adj_alpha
            comp['significant_corrected'] = comp['p_value'] < adj_alpha

            if not comp['significant_corrected']:
                for j in range(idx + 1, len(comparisons)):
                    comparisons[j]['adj_alpha'] = alpha / (n_comparisons - j)
                    comparisons[j]['significant_corrected'] = False
                break

        sig_comparisons = [c for c in comparisons if c['significant_corrected']]

        stats_results = {
            'summary_statistics': wavelet_summary,
            'ranking': [{'rank': i+1, 'wavelet': w, 'median_auc': s['median']} for i, (w, s) in enumerate(ranked)],
            'friedman_test': friedman_result,
            'pairwise_comparisons': comparisons,
            'n_significant': len(sig_comparisons)
        }
        if output_path is None:
            output_path = json_path.parent / 'wavelet_statistical_tests.json'
        else:
            output_path = Path(output_path)

        with open(output_path, 'w') as f:
            json.dump(stats_results, f, indent=2)

        return stats_results
