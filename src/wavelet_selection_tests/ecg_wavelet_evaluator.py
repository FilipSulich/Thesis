import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from scipy.stats import skew, kurtosis as scipy_kurtosis


# ── Constants ──────────────────────────────────────────────────────────────
CANDIDATE_WAVELETS = (
    [f'db{i}'   for i in range(2, 9)] +
    [f'sym{i}'  for i in range(2, 9)] +
    [f'coif{i}' for i in range(1, 4)]
)
SUBBAND_LEVELS = list(range(2, 9))   # D2-D8
SUBBAND_NAMES  = [f'D{i}' for i in SUBBAND_LEVELS]
N_LEVELS_SWT   = 10
UTRP_SIZE      = 256
FS_TARGET      = 500


class UTRPFeatureExtractor:
    """
    Extracts texture and structural features directly from UTRP images.
    These are the features that reflect what a CNN would learn from
    the recurrence representation — not RQA from the time-domain signal.
    """

    @staticmethod
    def extract(utrp_rgb: np.ndarray) -> np.ndarray:
        """
        Extract features from a UTRP image (H, W, 3) float32 in [0,1].

        Features:
        - Per-channel statistics: mean, std, skewness, kurtosis (12 features)
        - Grayscale entropy (1 feature)
        - GLCM texture: contrast, correlation, energy, homogeneity
          at 2 distances x 4 angles → mean + std per property (8 features)

        Total: 21 features per UTRP

        Note: NaN values (from constant channels) are replaced with 0.0
        """
        features = []

        # Per-channel statistics (R, G, B)
        for c in range(3):
            ch = utrp_rgb[:, :, c].ravel()
            # For constant channels, skew and kurtosis are undefined (NaN)
            # Replace with 0 (no skew, no excess kurtosis)
            ch_skew = skew(ch)
            ch_kurt = scipy_kurtosis(ch)
            features.extend([
                float(ch.mean()),
                float(ch.std()),
                0.0 if not np.isfinite(ch_skew) else float(ch_skew),
                0.0 if not np.isfinite(ch_kurt) else float(ch_kurt),
            ])

        # Grayscale entropy
        gray = rgb2gray(utrp_rgb)
        hist, _ = np.histogram(gray, bins=64, range=(0, 1), density=True)
        hist    = hist + 1e-10
        entropy = float(-np.sum(hist * np.log(hist)))
        features.append(entropy)

        # GLCM texture on quantised grayscale
        # Captures diagonal line structure and recurrence density patterns
        gray_uint8 = (gray * 255).astype(np.uint8)
        glcm = graycomatrix(
            gray_uint8,
            distances=[1, 3],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256, symmetric=True, normed=True
        )
        for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
            val = graycoprops(glcm, prop)   # (n_dist, n_angle)
            mean_val = val.mean()
            std_val = val.std()
            # Replace NaN with sensible defaults
            features.append(0.0 if not np.isfinite(mean_val) else float(mean_val))
            features.append(0.0 if not np.isfinite(std_val) else float(std_val))

        return np.array(features, dtype=np.float32)


class ECGWaveletEvaluator:
    """
    Wavelet function selection via UTRP-based discriminability analysis.

    For each candidate wavelet and each subband D2-D8:
    1. Decompose raw ECG via 10-level SWT
    2. Reconstruct each subband via iSWT
    3. Min-max normalise
    4. Generate UTRP with per-subband embedding parameters
    5. Extract texture features from UTRP image
    6. Train logistic regression under stratified 5-fold CV
    7. Report mean AUC ± std per (wavelet, subband)

    Selection criterion: wavelet with highest median AUC across D2-D8,
    with minimum AUC as a consistency tiebreaker.

    This directly measures discriminability of the UTRP representation
    that the downstream CNN will receive, using a linear probe that
    reflects information content without introducing architectural bias.
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        embedding_params: Dict,
        fs: int = FS_TARGET,
        n_cv_splits: int = 5,
        utrp_size: int = UTRP_SIZE,
        results_dir: Optional[Path] = None,
    ):
        """
        Args:
            signals:           (N, SEGMENT_SAMPLES) float32, min-max normalised
            labels:            (N,) int, 0=NSR 1=Pre-AF
            embedding_params:  {'D2': {'tau': int, 'm': int}, ...}
            fs:                sampling frequency
            n_cv_splits:       number of stratified CV folds
            utrp_size:         UTRP image size (pixels)
            results_dir:       directory for saving outputs
        """
        self.signals          = signals.astype(np.float32)
        self.labels           = labels.astype(np.int32)
        self.embedding_params = embedding_params
        self.fs               = fs
        self.n_cv_splits      = n_cv_splits
        self.utrp_size        = utrp_size
        self.results_dir      = Path(results_dir) if results_dir else Path('wavelet_selection')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.pad_samples = math.ceil(
            self.signals.shape[1] / (2 ** N_LEVELS_SWT)
        ) * (2 ** N_LEVELS_SWT)

        # Results storage
        # {wavelet: {subband: {'mean': float, 'std': float, 'values': list}}}
        self.results: Dict[str, Dict] = {}

    # ── Signal processing ──────────────────────────────────────────────────

    def _pad(self, signal: np.ndarray) -> np.ndarray:
        pad = self.pad_samples - len(signal)
        if pad <= 0:
            return signal[:self.pad_samples]
        return np.pad(signal, (0, pad), mode='reflect').astype(np.float32)

    def _minmax(self, x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-9)

    def _extract_subband(
        self, signal_padded: np.ndarray, wavelet: str, level: int
    ) -> np.ndarray:
        """
        SWT decomposition + iSWT reconstruction for a single subband.
        Correct index: SWT returns finest level first, so D_level is
        at index n_lvls - level (not level - 1).
        """
        coeffs = pywt.swt(signal_padded, wavelet, level=N_LEVELS_SWT)
        n_lvls = len(coeffs)
        target = n_lvls - level   # D2 → index 8, D8 → index 2

        zeroed = [
            ((np.zeros_like(cA), cD) if j == target
             else (np.zeros_like(cA), np.zeros_like(cD)))
            for j, (cA, cD) in enumerate(coeffs)
        ]
        sub = pywt.iswt(zeroed, wavelet)[:self.signals.shape[1]]
        return self._minmax(sub.astype(np.float32))

    def _signal_to_utrp(
        self, signal: np.ndarray, tau: int, m: int
    ) -> np.ndarray:
        """
        Phase-space embedding → distance matrix → UTRP RGB image.
        Returns (H, W, 3) float32 in [0, 1].
        """
        from src.recurrence_plot.recurrence_plot_generation import (
            RecurrencePlotGenerator
        )
        gen = RecurrencePlotGenerator(
            signal=signal, time_delay=tau,
            embedding_dim=m, epsilon=0
        )
        rp    = gen.compute_rp_unthresholded()
        d_max = rp.max()
        if d_max > 1e-9:
            rp = rp / d_max

        rp_r = cv2.resize(
            rp.astype(np.float32),
            (self.utrp_size, self.utrp_size),
            interpolation=cv2.INTER_LINEAR
        )
        np.fill_diagonal(rp_r, 0.0)

        u8  = (rp_r * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb

    # ── Feature extraction ─────────────────────────────────────────────────

    def _build_feature_matrix(
        self, wavelet: str, level: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate UTRP for every segment using the given wavelet and subband,
        then extract texture features. Returns (X, y) ready for classification.
        """
        name = f'D{level}'
        tau  = self.embedding_params[name]['tau']
        m    = self.embedding_params[name]['m']

        X_list, y_list = [], []
        n_failed = 0

        for idx, (sig, label) in enumerate(zip(self.signals, self.labels)):
            try:
                pad  = self._pad(sig)
                sub  = self._extract_subband(pad, wavelet, level)

                # Check subband validity
                if np.std(sub) < 1e-9:
                    n_failed += 1
                    continue  # Skip degenerate subbands

                utrp = self._signal_to_utrp(sub, tau, m)
                feat = UTRPFeatureExtractor.extract(utrp)

                # Features should now be NaN-free due to handling in extractor
                # but double-check just in case
                if not np.all(np.isfinite(feat)):
                    n_failed += 1
                    continue

                X_list.append(feat)
                y_list.append(label)

            except Exception as e:
                n_failed += 1
                continue

        if n_failed > 0:
            print(f" [{n_failed} failed]", end="")

        return np.array(X_list, dtype=np.float32), np.array(y_list)

    # ── Cross-validation ───────────────────────────────────────────────────

    def _cv_auc(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict:
        """
        Stratified k-fold CV with logistic regression.
        Returns mean, std, 95% CI, and per-fold AUC values.
        """
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42,
                               class_weight='balanced')
        )
        cv     = StratifiedKFold(n_splits=self.n_cv_splits,
                                 shuffle=True, random_state=42)
        scores = []

        for train_idx, test_idx in cv.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            prob = clf.predict_proba(X[test_idx])[:, 1]
            scores.append(float(roc_auc_score(y[test_idx], prob)))

        scores = np.array(scores)
        return {
            'mean':    float(scores.mean()),
            'std':     float(scores.std()),
            'ci_low':  float(scores.mean() - 1.96 * scores.std()),
            'ci_high': float(scores.mean() + 1.96 * scores.std()),
            'values':  scores.tolist(),
        }

    # ── Wavelet evaluation ─────────────────────────────────────────────────

    def evaluate_wavelet(self, wavelet: str) -> Dict:
        """
        Evaluate one wavelet across all subbands D2-D8.
        Returns per-subband CV AUC results and summary statistics.
        """
        print(f"\n  Wavelet: {wavelet}")
        subband_results = {}

        for level in SUBBAND_LEVELS:
            name = f'D{level}'
            lo   = FS_TARGET / (2 ** (level + 1))
            hi   = FS_TARGET / (2 ** level)

            X, y = self._build_feature_matrix(wavelet, level)

            if len(X) < 20 or y.sum() < 5 or (len(y) - y.sum()) < 5:
                print(f"    {name}: insufficient data")
                subband_results[name] = None
                continue

            scores = self._cv_auc(X, y)
            subband_results[name] = scores
            print(f"    {name} ({lo:.1f}-{hi:.1f} Hz): "
                  f"AUC={scores['mean']:.4f} ± {scores['std']:.4f} "
                  f"[{scores['ci_low']:.4f}, {scores['ci_high']:.4f}]")

        # Aggregate across subbands
        valid_means = [
            v['mean'] for v in subband_results.values()
            if v is not None
        ]
        summary = {
            'median_auc': float(np.median(valid_means)),
            'mean_auc':   float(np.mean(valid_means)),
            'min_auc':    float(np.min(valid_means)),
            'max_auc':    float(np.max(valid_means)),
            'per_subband': subband_results,
        }
        print(f"  → median={summary['median_auc']:.4f}  "
              f"min={summary['min_auc']:.4f}  "
              f"max={summary['max_auc']:.4f}")

        self.results[wavelet] = summary
        return summary

    def run_all(self) -> pd.DataFrame:
        """Evaluate all candidate wavelets and return ranked DataFrame."""
        print(f"\n{'='*65}")
        print("WAVELET SELECTION — UTRP FEATURE DISCRIMINABILITY")
        print(f"{'='*65}")
        print(f"Probe classifier: Logistic Regression, "
              f"{self.n_cv_splits}-fold stratified CV")
        print(f"Features: UTRP texture (29 per subband)")
        print(f"Metric: AUC-ROC per subband, median across D2-D8\n")

        for wavelet in CANDIDATE_WAVELETS:
            self.evaluate_wavelet(wavelet)

        return self.rank_wavelets()

    # ── Results and visualisation ──────────────────────────────────────────

    def rank_wavelets(self) -> pd.DataFrame:
        """Return DataFrame ranked by median AUC across subbands."""
        rows = []
        for wavelet, res in self.results.items():
            rows.append({
                'wavelet':    wavelet,
                'median_auc': res['median_auc'],
                'mean_auc':   res['mean_auc'],
                'min_auc':    res['min_auc'],
                'max_auc':    res['max_auc'],
            })
        df = pd.DataFrame(rows).sort_values(
            'median_auc', ascending=False
        ).reset_index(drop=True)

        print(f"\n{'='*65}")
        print("RANKING BY MEDIAN AUC ACROSS D2-D8")
        print(f"{'='*65}")
        print(df.to_string(index=False))
        return df

    def save_results(self) -> None:
        """Save results to JSON."""
        path = self.results_dir / 'wavelet_selection_results.json'
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved → {path}")

    def plot_results(self) -> None:
        """
        Two plots:
        1. Bar chart: median AUC per wavelet with min AUC dot
        2. Heatmap: AUC per wavelet × subband
        """
        ranked = sorted(
            self.results.items(),
            key=lambda x: x[1]['median_auc'], reverse=True
        )
        wavelets_ordered = [w for w, _ in ranked]

        # ── Bar chart ──────────────────────────────────────────────────
        medians = [self.results[w]['median_auc'] for w in wavelets_ordered]
        mins    = [self.results[w]['min_auc']    for w in wavelets_ordered]
        colours = [
            '#1f77b4' if w.startswith('db')
            else '#d62728' if w.startswith('sym')
            else '#2ca02c'
            for w in wavelets_ordered
        ]

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(wavelets_ordered))
        ax.bar(x, medians, color=colours, alpha=0.85,
               edgecolor='black', lw=0.5)
        ax.scatter(x, mins, color='black', s=50, zorder=5,
                   label='Min AUC (worst subband)')
        ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.6,
                   label='Chance level')
        ax.set_xticks(x)
        ax.set_xticklabels(wavelets_ordered, rotation=45,
                           ha='right', fontsize=10)
        ax.set_ylabel('Median AUC across D2–D8', fontsize=12)
        ax.set_title(
            'Wavelet selection: UTRP feature discriminability\n'
            'Logistic regression probe, 5-fold CV  |  '
            'Pre-AF vs NSR',
            fontsize=12
        )
        ax.legend(handles=[
            Patch(facecolor='#1f77b4', label='Daubechies'),
            Patch(facecolor='#d62728', label='Symlets'),
            Patch(facecolor='#2ca02c', label='Coiflets'),
            plt.Line2D([0], [0], color='black', marker='o',
                       ls='', label='Min AUC'),
        ], fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0.4, 1.0)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'wavelet_barchart.png', dpi=150)
        plt.close()
        print(f"✓ Bar chart saved")

        # ── Heatmap ────────────────────────────────────────────────────
        auc_matrix = np.array([
            [
                self.results[w]['per_subband'].get(
                    f'D{l}', {'mean': 0.0}
                )['mean']
                if self.results[w]['per_subband'].get(f'D{l}') else 0.0
                for l in SUBBAND_LEVELS
            ]
            for w in wavelets_ordered
        ])

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(auc_matrix, aspect='auto',
                       cmap='YlOrRd', vmin=0.4, vmax=0.8)
        ax.set_xticks(range(len(SUBBAND_LEVELS)))
        ax.set_xticklabels([
            f'D{l}\n{FS_TARGET/2**(l+1):.0f}–{FS_TARGET/2**l:.0f}Hz'
            for l in SUBBAND_LEVELS
        ], fontsize=9)
        ax.set_yticks(range(len(wavelets_ordered)))
        ax.set_yticklabels(wavelets_ordered, fontsize=10)
        ax.set_xlabel('Subband', fontsize=12)
        ax.set_ylabel('Wavelet (ranked by median AUC)', fontsize=12)
        ax.set_title(
            'AUC per wavelet × subband\n'
            '(UTRP texture features, logistic regression probe)',
            fontsize=12
        )
        plt.colorbar(im, ax=ax, label='AUC')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'wavelet_heatmap.png', dpi=150)
        plt.close()
        print(f"✓ Heatmap saved")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import wfdb
    from scipy.signal import resample as sp_resample

    DATA_RAW   = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_old/paf')
    SPLIT_JSON = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_final/utrp_dataset_db8/split.json')
    PARAMS_JSON = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_old/data_new/tau_m_params.json')
    RESULTS_DIR = Path('/results_final/results/wavelet_selection_utrp_new')

    FS_ORIG         = 128
    SEGMENT_SAMPLES = 5000
    PAD_SAMPLES     = 5120
    N_SEGS_PER_REC  = 5

    def _minmax(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-9)

    def load_segments():
        with open(SPLIT_JSON) as f:
            split = json.load(f)
        train_records = split['train']
        nsr_recs   = [r for r in train_records if r.startswith('n')]
        preaf_recs = [r for r in train_records
                      if r.startswith('p') and int(r[1:3]) % 2 == 1]

        def extract(record_id, label):
            try:
                rec = wfdb.rdrecord(str(DATA_RAW / record_id))
                raw = rec.p_signal[:, 0].astype(np.float64)
                n   = int(len(raw) * FS_TARGET / FS_ORIG)
                sig = sp_resample(raw, n).astype(np.float32)
                max_start = len(sig) - SEGMENT_SAMPLES
                if max_start <= 0: return [], []
                starts = np.linspace(0, max_start, 15, dtype=int)
                segs, labs = [], []
                for s in starts[:N_SEGS_PER_REC]:
                    seg = sig[s:s + SEGMENT_SAMPLES]
                    if len(seg) == SEGMENT_SAMPLES:
                        segs.append(_minmax(seg))
                        labs.append(label)
                return segs, labs
            except Exception as e:
                print(f"  Skip {record_id}: {e}")
                return [], []

        signals, labels = [], []
        for rid in preaf_recs:
            s, l = extract(rid, 1)
            signals.extend(s); labels.extend(l)
        for rid in nsr_recs:
            s, l = extract(rid, 0)
            signals.extend(s); labels.extend(l)

        return np.array(signals), np.array(labels)

    print("Loading raw ECG segments...")
    signals, labels = load_segments()
    print(f"Loaded {len(signals)} segments  "
          f"(Pre-AF={(labels==1).sum()}  NSR={(labels==0).sum()})")

    print("\nLoading embedding parameters...")
    with open(PARAMS_JSON) as f:
        raw_params = json.load(f)

    embedding_params = {}
    if 'subband_tau' in raw_params and 'subband_m' in raw_params:
        for k in SUBBAND_NAMES:
            if k in raw_params['subband_tau'] and k in raw_params['subband_m']:
                embedding_params[k] = {
                    'tau': raw_params['subband_tau'][k],
                    'm': raw_params['subband_m'][k]
                }
    else:
        embedding_params = raw_params

    evaluator = ECGWaveletEvaluator(
        signals=signals,
        labels=labels,
        embedding_params=embedding_params,
        n_cv_splits=5,
        results_dir=RESULTS_DIR,
    )

    df = evaluator.run_all()
    evaluator.save_results()
    evaluator.plot_results()

    best = df.iloc[0]
    print(f"\n{'='*65}")
    print(f"SELECTED WAVELET: {best['wavelet']}")
    print(f"Median AUC: {best['median_auc']:.4f}  "
          f"Min AUC: {best['min_auc']:.4f}")
    print(f"{'='*65}")