# """
# UTRP Group Separability Analysis.
#
# GSR metrics:
#   1. Frobenius   — element-wise distance after amplitude normalisation
#   2. Correlation — structural similarity (scale-invariant)
#   3. RQA         — dynamical invariants after heart-rate detrending
#   4. GLCM        — Haralick texture features from the UTRP treated as an image
#
# Why GLCM is better than direct RP distance metrics:
#   Distance metrics between UTRPs (Frobenius, Wasserstein, Spectral) all
#   compare patients *to each other*. Two NSR patients with different heart
#   rates produce UTRPs at different scales, making within-group distances
#   as large as between-group distances and collapsing GSR toward 1.
#
#   GLCM extracts scalar texture features *from each patient's UTRP
#   independently*, then compares the feature distributions across groups.
#   Each patient's UTRP is quantised to grey levels after per-patient
#   normalisation, making texture features scale-invariant. The resulting
#   feature vector captures spatial structure — contrast, homogeneity,
#   energy, correlation, dissimilarity — that reflects attractor geometry
#   without sensitivity to traversal speed or amplitude scale.
#
#   Reference: Haralick et al. (1973, IEEE Trans. SMC 3, 610-621).
#
# Statistical unit: patient (median features / mean UTRP across segments).
# Embedding: fixed across all patients from median over 10 NSR patients.
# Epsilon: fixed recurrence rate per segment (Kraemer et al. 2018).
# """
#
# import numpy as np
# import pandas as pd
# import pywt
# from pathlib import Path
# from typing import Callable, Dict, List, Optional, Tuple
# from scipy.spatial.distance import pdist, squareform
# from scipy.stats import pearsonr, linregress
# from itertools import combinations
# from skimage.feature import graycomatrix, graycoprops
# from tqdm import tqdm
# import sys
# import os
#
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# from src.recurrence_plot.AMI import AMI
# from src.recurrence_plot.FNN import FNN
# from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator
#
#
# # ──────────────────────────────────────────────────────────────────────────────
# #  UTRP normalisation
# # ──────────────────────────────────────────────────────────────────────────────
#
# def normalise_utrp(dm: np.ndarray, method: str = 'p95') -> np.ndarray:
#     """
#     Remove amplitude-scale confounds from a distance matrix.
#
#     method : 'p95'  — 95th percentile of upper triangle (default)
#              'max'  — maximum distance
#              'mean' — mean distance
#     """
#     vals = dm[np.triu_indices_from(dm, k=1)]
#     if method == 'p95':
#         scale = np.percentile(vals, 95)
#     elif method == 'max':
#         scale = vals.max()
#     elif method == 'mean':
#         scale = vals.mean()
#     else:
#         raise ValueError(f"Unknown normalisation method: {method}")
#     return dm / scale if scale > 1e-9 else dm
#
#
# # ──────────────────────────────────────────────────────────────────────────────
# #  GLCM texture features
# # ──────────────────────────────────────────────────────────────────────────────
#
# def utrp_glcm_features(utrp: np.ndarray, levels: int = 64) -> np.ndarray:
#     """
#     Extract Haralick texture features from a normalised UTRP image.
#
#     The UTRP is quantised to `levels` grey levels after per-patient
#     normalisation to [0, 1], making features scale-invariant across patients
#     with different heart rates or QRS amplitudes.
#
#     GLCM is computed at distances [1, 2, 5] and angles [0, 45, 90, 135]
#     degrees and averaged, giving rotation-robust texture descriptors.
#
#     Features returned:
#       0 — contrast     : local intensity variation (high in chaotic UTRPs)
#       1 — homogeneity  : inverse difference moment (high in periodic UTRPs)
#       2 — energy       : angular second moment / textural uniformity
#       3 — correlation  : linear grey-level dependency between pixel pairs
#       4 — dissimilarity: mean absolute grey-level difference between pairs
#
#     Parameters
#     ----------
#     utrp   : normalised distance matrix (values in [0, 1] after normalise_utrp)
#     levels : number of grey levels for quantisation (default 64)
#
#     Returns
#     -------
#     np.ndarray of shape (5,)
#     """
#     # Quantise to [0, levels-1] integer grey levels
#     u = utrp - utrp.min()
#     u = (u / (u.max() + 1e-9) * (levels - 1)).astype(np.uint8)
#
#     glcm = graycomatrix(
#         u,
#         distances=[1, 2, 5],
#         angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
#         levels=levels,
#         symmetric=True,
#         normed=True,
#     )
#
#     return np.array([
#         graycoprops(glcm, 'contrast').mean(),
#         graycoprops(glcm, 'homogeneity').mean(),
#         graycoprops(glcm, 'energy').mean(),
#         graycoprops(glcm, 'correlation').mean(),
#         graycoprops(glcm, 'dissimilarity').mean(),
#     ])
#
#
# # ──────────────────────────────────────────────────────────────────────────────
# #  Pairwise distance functions (kept for Frobenius / Correlation GSR)
# # ──────────────────────────────────────────────────────────────────────────────
#
# def frobenius_distance(a: np.ndarray, b: np.ndarray) -> float:
#     n = min(a.shape[0], b.shape[0])
#     return float(np.linalg.norm(a[:n, :n] - b[:n, :n], ord='fro'))
#
#
# def correlation_distance(a: np.ndarray, b: np.ndarray) -> float:
#     """1 − Pearson correlation of flattened upper triangles (scale-invariant)."""
#     n = min(a.shape[0], b.shape[0])
#     idx = np.triu_indices(n, k=1)
#     f1, f2 = a[:n, :n][idx], b[:n, :n][idx]
#     if np.std(f1) < 1e-9 or np.std(f2) < 1e-9:
#         return 1.0
#     return float(1.0 - pearsonr(f1, f2)[0])
#
#
# def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
#     return float(np.linalg.norm(a - b))
#
#
# # ──────────────────────────────────────────────────────────────────────────────
# #  GSR core
# # ──────────────────────────────────────────────────────────────────────────────
#
# def compute_gsr(
#     pre_af: List[np.ndarray],
#     nsr: List[np.ndarray],
#     dist_fn: Callable,
# ) -> Dict[str, float]:
#     """
#     Group Separability Ratio from patient-level representations.
#
#     GSR = between_group_mean_distance / within_group_mean_distance
#     GSR > 1 → groups more different from each other than within themselves.
#     """
#     within_pre_af = [dist_fn(a, b) for a, b in combinations(pre_af, 2)]
#     within_nsr    = [dist_fn(a, b) for a, b in combinations(nsr,    2)]
#     between       = [dist_fn(a, b) for a in pre_af for b in nsr]
#
#     w_pre  = float(np.mean(within_pre_af)) if within_pre_af else 0.0
#     w_nsr  = float(np.mean(within_nsr))    if within_nsr    else 0.0
#     w_mean = (w_pre + w_nsr) / 2.0
#     b_mean = float(np.mean(between))       if between       else 0.0
#
#     return {
#         'within_pre_af':  w_pre,
#         'within_nsr':     w_nsr,
#         'within_group':   w_mean,
#         'between_group':  b_mean,
#         'gsr':            b_mean / (w_mean + 1e-12),
#     }
#
#
# # ──────────────────────────────────────────────────────────────────────────────
# #  Analyser
# # ──────────────────────────────────────────────────────────────────────────────
#
# class UTRPSeparabilityAnalyzer:
#     """
#     Analyse UTRP separability between pre-AF and NSR patient groups.
#
#     Parameters
#     ----------
#     wavelets           : wavelet names to compare
#     level              : SWT decomposition level
#     fs                 : sampling frequency (Hz)
#     target_subbands    : subbands to analyse
#     normalise_method   : UTRP normalisation ('p95', 'max', 'mean')
#     rqa_target_rr      : target recurrence rate for epsilon selection
#     glcm_levels        : number of grey levels for GLCM quantisation
#     n_sample_patients  : NSR patients used for embedding estimation
#     random_seed        : reproducibility
#     """
#
#     def __init__(
#         self,
#         wavelets: List[str],
#         level: int = 4,
#         fs: int = 128,
#         target_subbands: Optional[List[str]] = None,
#         normalise_method: str = 'p95',
#         rqa_target_rr: float = 0.1,
#         glcm_levels: int = 64,
#         n_sample_patients: int = 10,
#         random_seed: int = 42,
#     ):
#         self.wavelets          = wavelets
#         self.level             = level
#         self.fs                = fs
#         self.target_subbands   = target_subbands or ['D1', 'D2', 'D3', 'D4']
#         self.normalise_method  = normalise_method
#         self.rqa_target_rr     = rqa_target_rr
#         self.glcm_levels       = glcm_levels
#         self.n_sample_patients = n_sample_patients
#         self.rng               = np.random.default_rng(random_seed)
#         self.embedding_params: Dict[str, Dict[str, int]] = {}
#
#     # ──────────────────────────────────────────────────────────────────
#     #  SWT decomposition
#     # ──────────────────────────────────────────────────────────────────
#
#     def _decompose(self, signal: np.ndarray, wavelet: str) -> Dict[str, np.ndarray]:
#         """Z-scored SWT subband reconstruction. Returns D1–D4 and A4."""
#         coeffs = pywt.swt(signal, wavelet, level=self.level)
#         n      = len(signal)
#         result: Dict[str, np.ndarray] = {}
#
#         for i in range(self.level):
#             band   = f'D{i + 1}'
#             target = [(np.zeros_like(cA), np.zeros_like(cD)) for cA, cD in coeffs]
#             target[self.level - 1 - i] = (
#                 np.zeros_like(coeffs[self.level - 1 - i][0]),
#                 coeffs[self.level - 1 - i][1],
#             )
#             rec = pywt.iswt(target, wavelet)[:n]
#             result[band] = (rec - np.mean(rec)) / (np.std(rec) + 1e-9)
#
#         target = [(cA, np.zeros_like(cD)) for cA, cD in coeffs]
#         for j in range(1, self.level):
#             target[j] = (np.zeros_like(target[j][0]), np.zeros_like(target[j][1]))
#         rec = pywt.iswt(target, wavelet)[:n]
#         result['A4'] = (rec - np.mean(rec)) / (np.std(rec) + 1e-9)
#
#         return result
#
#     # ──────────────────────────────────────────────────────────────────
#     #  Embedding parameter estimation
#     # ──────────────────────────────────────────────────────────────────
#
#     def estimate_embedding_params(
#         self,
#         sample_segments: List[np.ndarray],
#         reference_wavelet: str = 'db4',
#     ):
#         """
#         Estimate tau and m via AMI/FNN per subband.
#         Median across all sample segments for stability.
#         Fixed for the entire experiment.
#         """
#         print(f"\nEstimating embedding parameters ({reference_wavelet})...")
#
#         for subband in self.target_subbands:
#             taus: List[int] = []
#             ms:   List[int] = []
#
#             for seg in sample_segments:
#                 sig = self._decompose(seg, reference_wavelet)[subband]
#                 try:
#                     tau = AMI(sig, max_lag=50).get_first_minimum()
#                     tau = max(1, min(int(tau or 1), 20))
#                     m   = FNN(signal=sig, tau=tau, max_dim=10).get_optimal_dimension()
#                     m   = max(2, min(int(m   or 3), 10))
#                 except Exception:
#                     tau, m = 1, 3
#                 taus.append(tau)
#                 ms.append(m)
#
#             final_tau = int(np.median(taus))
#             final_m   = int(np.median(ms))
#             self.embedding_params[subband] = {'tau': final_tau, 'm': final_m}
#             print(f"  {subband}: tau={final_tau}, m={final_m}")
#
#         print()
#
#     # ──────────────────────────────────────────────────────────────────
#     #  Per-segment: single decomposition pass → UTRP + RQA
#     # ──────────────────────────────────────────────────────────────────
#
#     def _segment_features(
#         self,
#         signal:  np.ndarray,
#         wavelet: str,
#         subband: str,
#     ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
#         """
#         Decompose once; return (normalised_utrp, rqa_feature_vector).
#         GLCM is computed from the UTRP at patient-aggregation time, not here,
#         to avoid redundant computation when averaging UTRPs across segments.
#         Returns (None, None) on failure.
#         """
#         try:
#             sub = self._decompose(signal, wavelet)[subband]
#             tau = self.embedding_params[subband]['tau']
#             m   = self.embedding_params[subband]['m']
#
#             n_pts = len(sub) - (m - 1) * tau
#             if n_pts <= 10:
#                 return None, None
#
#             emb  = np.stack([sub[i * tau: i * tau + n_pts] for i in range(m)], axis=1)
#             dm   = squareform(pdist(emb, metric='euclidean'))
#             utrp = normalise_utrp(dm, self.normalise_method)
#
#             eps = RecurrencePlotGenerator.find_epsilon_for_target_rr(
#                 sub, tau, m, target_rr=self.rqa_target_rr
#             )
#             rp_gen = RecurrencePlotGenerator(
#                 signal=sub, time_delay=tau, embedding_dim=m,
#                 epsilon=eps, unthresholded=False,
#             )
#             rqa, _ = rp_gen.compute_rqa_binary()
#
#             def _safe(v) -> float:
#                 try:
#                     f = float(v)
#                     return f if np.isfinite(f) else 0.0
#                 except Exception:
#                     return 0.0
#
#             rqa_vec = np.array([
#                 _safe(rqa.determinism),
#                 _safe(rqa.laminarity),
#                 _safe(rqa.trapping_time),
#                 _safe(rqa.average_diagonal_line),
#                 _safe(rqa.longest_diagonal_line),
#             ])
#
#             return utrp, rqa_vec
#
#         except Exception:
#             return None, None
#
#     # ──────────────────────────────────────────────────────────────────
#     #  Patient-level aggregation → UTRP + RQA + GLCM
#     # ──────────────────────────────────────────────────────────────────
#
#     def _patient_representations(
#         self,
#         segments: List[np.ndarray],
#         wavelet:  str,
#         subband:  str,
#     ) -> Tuple[
#         Optional[np.ndarray],   # patient UTRP  (mean across segments)
#         Optional[np.ndarray],   # patient RQA   (median across segments)
#         Optional[np.ndarray],   # patient GLCM  (from mean UTRP)
#     ]:
#         """
#         Compute one UTRP, one RQA vector, and one GLCM feature vector per patient.
#
#         UTRPs   : mean across segments after normalisation.
#                   Scale is already removed by normalise_utrp so averaging is safe.
#         RQA     : median across segments (robust to ectopic beat outliers).
#         GLCM    : computed from the mean UTRP so it represents the patient's
#                   average attractor texture rather than a single noisy segment.
#                   This is correct: GLCM on the mean UTRP ≠ mean of per-segment
#                   GLCMs, but the mean UTRP is the most stable representation of
#                   the patient's phase-space geometry.
#         """
#         utrps:    List[np.ndarray] = []
#         rqa_vecs: List[np.ndarray] = []
#
#         for seg in segments:
#             u, r = self._segment_features(seg, wavelet, subband)
#             if u is not None:
#                 utrps.append(u)
#             if r is not None:
#                 rqa_vecs.append(r)
#
#         patient_utrp = None
#         patient_rqa  = None
#         patient_glcm = None
#
#         if utrps:
#             min_n        = min(u.shape[0] for u in utrps)
#             patient_utrp = np.mean([u[:min_n, :min_n] for u in utrps], axis=0)
#             # GLCM computed from the mean UTRP (not per-segment)
#             try:
#                 patient_glcm = utrp_glcm_features(
#                     patient_utrp, levels=self.glcm_levels
#                 )
#             except Exception:
#                 patient_glcm = None
#
#         if rqa_vecs:
#             patient_rqa = np.median(rqa_vecs, axis=0)
#
#         return patient_utrp, patient_rqa, patient_glcm
#
#     # ──────────────────────────────────────────────────────────────────
#     #  Heart-rate detrending
#     # ──────────────────────────────────────────────────────────────────
#
#     @staticmethod
#     def _detrend_heart_rate(
#         pre_af_vecs: List[np.ndarray],
#         nsr_vecs:    List[np.ndarray],
#         pre_af_hr:   List[float],
#         nsr_hr:      List[float],
#     ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
#         """
#         Remove the linear effect of heart rate from each feature dimension.
#
#         Used for both RQA vectors and GLCM vectors before computing GSR.
#         Residuals retain the pre-AF vs NSR signal while removing HR-driven
#         inter-patient variance that would otherwise collapse GSR toward 1.
#         """
#         all_vecs = np.array(pre_af_vecs + nsr_vecs)
#         all_hr   = np.array(pre_af_hr   + nsr_hr)
#         n_feats  = all_vecs.shape[1]
#
#         residuals = np.zeros_like(all_vecs)
#         for f in range(n_feats):
#             slope, intercept, *_ = linregress(all_hr, all_vecs[:, f])
#             residuals[:, f] = all_vecs[:, f] - (slope * all_hr + intercept)
#
#         n_pre = len(pre_af_vecs)
#         return (
#             [residuals[i] for i in range(n_pre)],
#             [residuals[i] for i in range(n_pre, len(residuals))],
#         )
#
#     def _mean_hr(self, segments: List[np.ndarray]) -> float:
#         """Estimate mean heart rate (bpm) from R-peaks across all segments."""
#         from ecgdetectors import Detectors
#         det    = Detectors(self.fs)
#         all_rr: List[float] = []
#         for seg in segments:
#             try:
#                 peaks = det.pan_tompkins_detector(seg)
#                 if len(peaks) >= 2:
#                     all_rr.extend(np.diff(peaks).tolist())
#             except Exception:
#                 continue
#         return float(60.0 * self.fs / np.median(all_rr)) if all_rr else 70.0
#
#     # ──────────────────────────────────────────────────────────────────
#     #  Robust-scaled Euclidean GSR for feature vectors
#     # ──────────────────────────────────────────────────────────────────
#
#     @staticmethod
#     def _robustly_scaled_gsr(
#         pre_af_vecs: List[np.ndarray],
#         nsr_vecs:    List[np.ndarray],
#     ) -> Dict[str, float]:
#         """
#         Per-feature robust scaling (median/IQR) then Euclidean GSR.
#
#         Prevents large-magnitude features (L_max, contrast) from dominating
#         small-magnitude features (DET, energy) without using cosine distance,
#         which destroys the geometric meaning of direction in feature space.
#         """
#         all_vecs = np.array(pre_af_vecs + nsr_vecs)
#         median   = np.median(all_vecs, axis=0)
#         iqr      = (
#             np.percentile(all_vecs, 75, axis=0)
#             - np.percentile(all_vecs, 25, axis=0)
#         ) + 1e-9
#
#         pre_scaled = [(v - median) / iqr for v in pre_af_vecs]
#         nsr_scaled  = [(v - median) / iqr for v in nsr_vecs]
#
#         return compute_gsr(pre_scaled, nsr_scaled, euclidean_distance)
#
#     # ──────────────────────────────────────────────────────────────────
#     #  Per-wavelet analysis
#     # ──────────────────────────────────────────────────────────────────
#
#     def _analyze_wavelet(
#         self,
#         wavelet:         str,
#         segments_pre_af: Dict[str, List[np.ndarray]],
#         segments_nsr:    Dict[str, List[np.ndarray]],
#         pre_af_hr_map:   Dict[str, float],
#         nsr_hr_map:      Dict[str, float],
#     ) -> pd.DataFrame:
#
#         rows = []
#
#         for subband in self.target_subbands:
#             print(f"  {wavelet} — {subband}", flush=True)
#
#             # ── collect patient representations ──────────────────────
#             pre_af_utrps: List[np.ndarray] = []
#             pre_af_rqa:   List[np.ndarray] = []
#             pre_af_glcm:  List[np.ndarray] = []
#             pre_af_hr:    List[float]       = []
#
#             for pid, segs in tqdm(
#                 segments_pre_af.items(), desc="    pre-AF", leave=False
#             ):
#                 u, r, g = self._patient_representations(segs, wavelet, subband)
#                 if u is not None:
#                     pre_af_utrps.append(u)
#                 if r is not None:
#                     pre_af_rqa.append(r)
#                     pre_af_hr.append(pre_af_hr_map.get(pid, 70.0))
#                 if g is not None:
#                     pre_af_glcm.append(g)
#
#             nsr_utrps: List[np.ndarray] = []
#             nsr_rqa:   List[np.ndarray] = []
#             nsr_glcm:  List[np.ndarray] = []
#             nsr_hr:    List[float]       = []
#
#             for pid, segs in tqdm(
#                 segments_nsr.items(), desc="    NSR", leave=False
#             ):
#                 u, r, g = self._patient_representations(segs, wavelet, subband)
#                 if u is not None:
#                     nsr_utrps.append(u)
#                 if r is not None:
#                     nsr_rqa.append(r)
#                     nsr_hr.append(nsr_hr_map.get(pid, 70.0))
#                 if g is not None:
#                     nsr_glcm.append(g)
#
#             n_pre = len(pre_af_utrps)
#             n_nsr = len(nsr_utrps)
#             print(f"    patients: pre-AF={n_pre}, NSR={n_nsr}")
#
#             # ── UTRP-based GSR ────────────────────────────────────────
#             frob = compute_gsr(pre_af_utrps, nsr_utrps, frobenius_distance)
#             corr = compute_gsr(pre_af_utrps, nsr_utrps, correlation_distance)
#
#             # ── RQA GSR: detrend HR, then robust-scale ────────────────
#             if len(pre_af_rqa) >= 2 and len(nsr_rqa) >= 2:
#                 pre_rqa_dt, nsr_rqa_dt = self._detrend_heart_rate(
#                     pre_af_rqa, nsr_rqa, pre_af_hr, nsr_hr
#                 )
#                 rqa = self._robustly_scaled_gsr(pre_rqa_dt, nsr_rqa_dt)
#             else:
#                 rqa = {'gsr': float('nan'),
#                        'within_group': float('nan'),
#                        'between_group': float('nan')}
#
#             # ── GLCM GSR: detrend HR, then robust-scale ───────────────
#             # GLCM features share the same HR list as RQA since both are
#             # derived from the same patient segments.
#             if len(pre_af_glcm) >= 2 and len(nsr_glcm) >= 2:
#                 # HR list aligned to GLCM patients (same order as UTRP loop)
#                 pre_glcm_hr = pre_af_hr[:len(pre_af_glcm)]
#                 nsr_glcm_hr = nsr_hr[:len(nsr_glcm)]
#                 pre_glcm_dt, nsr_glcm_dt = self._detrend_heart_rate(
#                     pre_af_glcm, nsr_glcm,
#                     pre_glcm_hr, nsr_glcm_hr,
#                 )
#                 glcm_gsr = self._robustly_scaled_gsr(pre_glcm_dt, nsr_glcm_dt)
#             else:
#                 glcm_gsr = {'gsr': float('nan'),
#                             'within_group': float('nan'),
#                             'between_group': float('nan')}
#
#             print(
#                 f"    GSR  frob={frob['gsr']:.4f}  "
#                 f"corr={corr['gsr']:.4f}  "
#                 f"rqa={rqa['gsr']:.4f}  "
#                 f"glcm={glcm_gsr['gsr']:.4f}\n"
#             )
#
#             rows.append({
#                 'wavelet':        wavelet,
#                 'subband':        subband,
#                 'gsr_frob':       frob['gsr'],
#                 'gsr_corr':       corr['gsr'],
#                 'gsr_rqa':        rqa['gsr'],
#                 'gsr_glcm':       glcm_gsr['gsr'],
#                 'within_frob':    frob['within_group'],
#                 'between_frob':   frob['between_group'],
#                 'within_rqa':     rqa['within_group'],
#                 'between_rqa':    rqa['between_group'],
#                 'within_glcm':    glcm_gsr['within_group'],
#                 'between_glcm':   glcm_gsr['between_group'],
#                 'n_pre_af':       n_pre,
#                 'n_nsr':          n_nsr,
#             })
#
#         return pd.DataFrame(rows)
#
#     # ──────────────────────────────────────────────────────────────────
#     #  Weighted composite score
#     # ──────────────────────────────────────────────────────────────────
#
#     @staticmethod
#     def _weighted_score(results_df: pd.DataFrame) -> pd.DataFrame:
#         """
#         score = mean(GSR_D3, GSR_D4) - 0.5 * mean(GSR_D1, GSR_D2)
#
#         Rewards separability in clinically relevant subbands (D3, D4).
#         Penalises separability in control subbands (D1, D2).
#         """
#         rows = []
#         for wavelet, grp in results_df.groupby('wavelet'):
#             for metric in ('frob', 'corr', 'rqa', 'glcm'):
#                 col = f'gsr_{metric}'
#                 if col not in grp.columns:
#                     continue
#
#                 def g(band: str) -> float:
#                     v = grp.loc[grp['subband'] == band, col]
#                     return float(v.values[0]) if len(v) else 1.0
#
#                 d3, d4 = g('D3'), g('D4')
#                 d1, d2 = g('D1'), g('D2')
#                 score  = (d3 + d4) / 2.0 - 0.5 * (d1 + d2) / 2.0
#
#                 rows.append({
#                     'wavelet':        wavelet,
#                     'metric':         metric,
#                     'gsr_D1':         d1,
#                     'gsr_D2':         d2,
#                     'gsr_D3':         d3,
#                     'gsr_D4':         d4,
#                     'weighted_score': score,
#                 })
#
#         return (
#             pd.DataFrame(rows)
#             .sort_values('weighted_score', ascending=False)
#             .reset_index(drop=True)
#         )
#
#     # ──────────────────────────────────────────────────────────────────
#     #  Public entry point
#     # ──────────────────────────────────────────────────────────────────
#
#     def run_full_analysis(
#         self,
#         segments_pre_af: Dict[str, List[np.ndarray]],
#         segments_nsr:    Dict[str, List[np.ndarray]],
#         output_path:     Optional[Path] = None,
#     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         """
#         Run complete separability analysis for all wavelets.
#
#         Parameters
#         ----------
#         segments_pre_af : {patient_id: [segment_array, ...]}
#         segments_nsr    : {patient_id: [segment_array, ...]}
#         output_path     : optional CSV save path
#
#         Returns
#         -------
#         results_df : per-wavelet × per-subband metrics
#         summary_df : weighted composite scores per wavelet
#         """
#         print("\n" + "=" * 80)
#         print("UTRP GROUP SEPARABILITY ANALYSIS")
#         print(f"  normalisation  : {self.normalise_method}")
#         print(f"  subbands       : {self.target_subbands}")
#         print(f"  target RR      : {self.rqa_target_rr}")
#         print(f"  GLCM levels    : {self.glcm_levels}")
#         print(f"  wavelets       : {self.wavelets}")
#         print("=" * 80)
#
#         # ── embedding estimation from randomly selected NSR patients ──
#         nsr_pids = list(segments_nsr.keys())
#         n_sample = min(self.n_sample_patients, len(nsr_pids))
#         selected = self.rng.choice(nsr_pids, size=n_sample, replace=False)
#         print(f"\nEstimation patients (NSR, n={n_sample}): {list(selected)}")
#
#         sample_segs: List[np.ndarray] = []
#         for pid in selected:
#             sample_segs.extend(segments_nsr[pid])
#         if not sample_segs:
#             raise ValueError("No segments found for selected NSR patients.")
#
#         self.estimate_embedding_params(sample_segs)
#
#         # ── heart rate map (once, outside all loops) ──────────────────
#         print("Computing mean heart rate per patient...")
#         pre_af_hr_map: Dict[str, float] = {
#             pid: self._mean_hr(segs)
#             for pid, segs in tqdm(segments_pre_af.items(), desc="  pre-AF HR")
#         }
#         nsr_hr_map: Dict[str, float] = {
#             pid: self._mean_hr(segs)
#             for pid, segs in tqdm(segments_nsr.items(), desc="  NSR HR")
#         }
#         pre_hrs = list(pre_af_hr_map.values())
#         nsr_hrs = list(nsr_hr_map.values())
#         print(
#             f"  pre-AF: {np.mean(pre_hrs):.1f} ± {np.std(pre_hrs):.1f} bpm  "
#             f"  NSR: {np.mean(nsr_hrs):.1f} ± {np.std(nsr_hrs):.1f} bpm\n"
#         )
#
#         # ── main loop ─────────────────────────────────────────────────
#         all_dfs: List[pd.DataFrame] = []
#         for wavelet in self.wavelets:
#             print(f"\n{'=' * 80}\nWavelet: {wavelet}\n{'=' * 80}")
#             all_dfs.append(
#                 self._analyze_wavelet(
#                     wavelet,
#                     segments_pre_af,
#                     segments_nsr,
#                     pre_af_hr_map,
#                     nsr_hr_map,
#                 )
#             )
#
#         results_df = pd.concat(all_dfs, ignore_index=True)
#         summary_df = self._weighted_score(results_df)
#
#         if output_path is not None:
#             p = Path(output_path)
#             p.parent.mkdir(parents=True, exist_ok=True)
#             results_df.to_csv(p, index=False)
#             summary_df.to_csv(p.with_stem(p.stem + '_summary'), index=False)
#             print(f"\nSaved: {p}")
#
#         self._print_summary(results_df, summary_df)
#         return results_df, summary_df
#
#     # ──────────────────────────────────────────────────────────────────
#     #  Summary printing
#     # ──────────────────────────────────────────────────────────────────
#
#     def _print_summary(
#         self,
#         results_df: pd.DataFrame,
#         summary_df: pd.DataFrame,
#     ):
#         labels = {
#             'frob': 'Normalised Frobenius',
#             'corr': 'Correlation',
#             'rqa':  'RQA (HR-detrended)',
#             'glcm': 'GLCM texture (HR-detrended)',
#         }
#
#         print("\n" + "=" * 80)
#         print("RESULTS SUMMARY")
#         print("=" * 80)
#
#         for metric, label in labels.items():
#             col = f'gsr_{metric}'
#             if col not in results_df.columns:
#                 continue
#             print(f"\n── {label} GSR ──")
#             for subband in self.target_subbands:
#                 sub = (
#                     results_df[results_df['subband'] == subband][['wavelet', col]]
#                     .sort_values(col, ascending=False)
#                 )
#                 print(
#                     f"  {subband}: "
#                     + "  ".join(
#                         f"{row['wavelet']}={row[col]:.4f}"
#                         for _, row in sub.iterrows()
#                     )
#                 )
#
#             top3 = (
#                 summary_df[summary_df['metric'] == metric]
#                 [['wavelet', 'gsr_D3', 'gsr_D4', 'weighted_score']]
#                 .head(3)
#             )
#             print(f"\n  Top-3 weighted score:\n{top3.to_string(index=False)}")
#             best = summary_df[summary_df['metric'] == metric].iloc[0]
#             print(
#                 f"\n  >>> SELECTED: {best['wavelet']}  "
#                 f"(score={best['weighted_score']:.4f}) <<<"
#             )

import numpy as np
import pandas as pd
import pywt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.stats import pearsonr, wasserstein_distance, mannwhitneyu
from numpy.linalg import eigvalsh
from itertools import combinations
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recurrence_plot.AMI import AMI
from src.recurrence_plot.FNN import FNN
from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator


# ------------------------------------------------------------------ #
#  Distance & Normalisation Helpers                                   #
# ------------------------------------------------------------------ #

def normalise_utrp(dm: np.ndarray, method: str = 'p95') -> np.ndarray:
    vals = dm[np.triu_indices_from(dm, k=1)]
    if method == 'p95':
        scale = np.percentile(vals, 95)
    elif method == 'max':
        scale = vals.max()
    elif method == 'mean':
        scale = vals.mean()
    else:
        raise ValueError(f"Unknown normalisation method: {method}")
    return dm / scale if scale > 1e-9 else dm


def wasserstein_utrp_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[0], b.shape[0])
    idx = np.triu_indices(n, k=1)
    dist_a = a[:n, :n][idx]
    dist_b = b[:n, :n][idx]
    return float(wasserstein_distance(dist_a, dist_b))


def spectral_utrp_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[0], b.shape[0])
    ev_a = eigvalsh(a[:n, :n])
    ev_b = eigvalsh(b[:n, :n])
    return float(np.linalg.norm(ev_a - ev_b))


def frobenius_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[0], b.shape[0])
    return float(np.linalg.norm(a[:n, :n] - b[:n, :n], ord='fro'))


def correlation_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[0], b.shape[0])
    idx = np.triu_indices(n, k=1)
    f1, f2 = a[:n, :n][idx], b[:n, :n][idx]
    if np.std(f1) < 1e-9 or np.std(f2) < 1e-9:
        return 1.0
    return float(1.0 - pearsonr(f1, f2)[0])


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ------------------------------------------------------------------ #
#  Entropy Implementations                                            #
# ------------------------------------------------------------------ #

def compute_sample_entropy(sig: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Calculates Sample Entropy."""
    n = len(sig)
    if n <= m: return 0.0
    r_tol = r * np.std(sig)

    def _count_matches(dim):
        # Build template matrix
        X = np.array([sig[i:i + dim] for i in range(n - dim + 1)])
        # Compute all-to-all distances
        dists = pdist(X, metric='chebyshev')
        return np.sum(dists <= r_tol)

    A = _count_matches(m + 1)
    B = _count_matches(m)
    if A == 0 or B == 0: return 0.0
    return -np.log(A / B)


def compute_permutation_entropy(sig: np.ndarray, m: int = 3, delay: int = 1) -> float:
    """Calculates Permutation Entropy (Bandt-Pompe)."""
    n = len(sig)
    if n < m: return 0.0
    # Generate ordinal patterns
    indices = np.array([sig[i: i + m * delay: delay] for i in range(n - (m - 1) * delay)])
    patterns = np.argsort(indices, axis=1)
    # Convert patterns to unique strings for counting
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))


def compute_gsr(pre_af: List[np.ndarray], nsr: List[np.ndarray], dist_fn) -> Dict[str, float]:
    within_pre_af = [dist_fn(a, b) for a, b in combinations(pre_af, 2)]
    within_nsr = [dist_fn(a, b) for a, b in combinations(nsr, 2)]
    between = [dist_fn(a, b) for a in pre_af for b in nsr]
    w_pre = float(np.mean(within_pre_af)) if within_pre_af else 0.0
    w_nsr = float(np.mean(within_nsr)) if within_nsr else 0.0
    w_mean = (w_pre + w_nsr) / 2.0
    b_mean = float(np.mean(between)) if between else 0.0
    return {
        'within_group': w_mean,
        'between_group': b_mean,
        'gsr': b_mean / (w_mean + 1e-12),
    }


# ------------------------------------------------------------------ #
#  Analyser Class                                                      #
# ------------------------------------------------------------------ #

class UTRPSeparabilityAnalyzer:
    def __init__(
            self,
            wavelets: List[str],
            level: int = 4,
            fs: int = 128,
            target_subbands: Optional[List[str]] = None,
            normalise_method: str = 'p95',
            rqa_target_rr: float = 0.1,
            random_seed: int = 42,
    ):
        self.wavelets = wavelets
        self.level = level
        self.fs = fs
        self.target_subbands = target_subbands or ['D1', 'D2', 'D3', 'D4']
        self.normalise_method = normalise_method
        self.rqa_target_rr = rqa_target_rr
        self.rng = np.random.default_rng(random_seed)
        self.embedding_params: Dict[str, Dict[str, int]] = {}

    def _decompose(self, signal: np.ndarray, wavelet: str) -> Dict[str, np.ndarray]:
        coeffs = pywt.swt(signal, wavelet, level=self.level)
        n = len(signal)
        result: Dict[str, np.ndarray] = {}
        for i in range(self.level):
            band = f'D{i + 1}'
            target = [(np.zeros_like(cA), np.zeros_like(cD)) for cA, cD in coeffs]
            target[self.level - 1 - i] = (np.zeros_like(coeffs[self.level - 1 - i][0]), coeffs[self.level - 1 - i][1])
            rec = pywt.iswt(target, wavelet)[:n]
            result[band] = (rec - np.mean(rec)) / (np.std(rec) + 1e-9)
        # A4
        target = [(cA, np.zeros_like(cD)) for cA, cD in coeffs]
        for j in range(1, self.level): target[j] = (np.zeros_like(target[j][0]), np.zeros_like(target[j][1]))
        rec = pywt.iswt(target, wavelet)[:n]
        result['A4'] = (rec - np.mean(rec)) / (np.std(rec) + 1e-9)
        return result

    def _wavelet_entropy(self, decomposition: Dict[str, np.ndarray]) -> float:
        """Calculates global Wavelet Entropy based on energy distribution."""
        energies = np.array([np.sum(np.square(sig)) for sig in decomposition.values()])
        ps = energies / (np.sum(energies) + 1e-12)
        return -np.sum(ps * np.log2(ps + 1e-12))

    def _segment_features(self, signal, wavelet, subband, full_decomp=None):
        try:
            sub = self._decompose(signal, wavelet)[subband] if full_decomp is None else full_decomp[subband]
            tau = self.embedding_params[subband]['tau']
            m = self.embedding_params[subband]['m']

            # 1. UTRP Generation
            n_pts = len(sub) - (m - 1) * tau
            if n_pts <= 10: return None, None, None
            emb = np.stack([sub[i * tau: i * tau + n_pts] for i in range(m)], axis=1)
            dm = squareform(pdist(emb, metric='euclidean'))
            utrp = normalise_utrp(dm, self.normalise_method)

            # 2. RQA Generation
            eps = RecurrencePlotGenerator.find_epsilon_for_target_rr(sub, tau, m, target_rr=self.rqa_target_rr)
            rp_gen = RecurrencePlotGenerator(signal=sub, time_delay=tau, embedding_dim=m, epsilon=eps,
                                             unthresholded=False)
            rqa, _ = rp_gen.compute_rqa_binary()

            def _safe(v):
                return float(v) if np.isfinite(v) else 0.0

            rqa_vec = np.array([_safe(rqa.determinism), _safe(rqa.laminarity), _safe(rqa.trapping_time),
                                _safe(rqa.average_diagonal_line), _safe(rqa.longest_diagonal_line)])

            # 3. Entropy Features
            entropy_vec = {
                'samp_en': compute_sample_entropy(sub, m=2, r=0.2),
                'perm_en': compute_permutation_entropy(sub, m=3, delay=tau),
                'wave_en': self._wavelet_entropy(full_decomp if full_decomp else self._decompose(signal, wavelet))
            }

            return utrp, rqa_vec, entropy_vec
        except Exception:
            return None, None, None

    def _patient_representations(self, segments: List[np.ndarray], wavelet: str, subband: str):
        utrps, rqa_vecs, entropy_list = [], [], []
        for seg in segments:
            decomp = self._decompose(seg, wavelet)
            u, r, e = self._segment_features(seg, wavelet, subband, full_decomp=decomp)
            if u is not None:
                utrps.append(u);
                rqa_vecs.append(r);
                entropy_list.append(e)

        p_utrp = np.mean([u[:min(x.shape[0] for x in utrps), :min(x.shape[0] for x in utrps)] for u in utrps],
                         axis=0) if utrps else None
        p_rqa = np.median(rqa_vecs, axis=0) if rqa_vecs else None
        p_ent = {k: np.median([e[k] for e in entropy_list]) for k in entropy_list[0].keys()} if entropy_list else None
        return p_utrp, p_rqa, p_ent

    def _analyze_wavelet(self, wavelet, segments_pre_af, segments_nsr) -> pd.DataFrame:
        rows = []
        for subband in self.target_subbands:
            print(f"  Processing {wavelet} - {subband}")
            pre_af_data = [self._patient_representations(segs, wavelet, subband) for segs in segments_pre_af.values()]
            nsr_data = [self._patient_representations(segs, wavelet, subband) for segs in segments_nsr.values()]

            pre_af_utrps = [d[0] for d in pre_af_data if d[0] is not None]
            nsr_utrps = [d[0] for d in nsr_data if d[0] is not None]

            # Distance GSR
            frob = compute_gsr(pre_af_utrps, nsr_utrps, frobenius_distance)
            # wass = compute_gsr(pre_af_utrps, nsr_utrps, wasserstein_utrp_distance)
            # spec = compute_gsr(pre_af_utrps, nsr_utrps, spectral_utrp_distance)

            # Entropy Stats Collection
            ent_rows = []
            for m_name in ['samp_en', 'perm_en', 'wave_en']:
                pre_v = [d[2][m_name] for d in pre_af_data if d[2] is not None]
                nsr_v = [d[2][m_name] for d in nsr_data if d[2] is not None]
                res = self._perform_mann_whitney(pre_v, nsr_v)
                ent_rows.append({f'p_{m_name}': res['p'], f'd_{m_name}': res['cohen_d']})

            row = {
                'wavelet': wavelet, 'subband': subband,
                'gsr_frob': frob['gsr'],
                # 'gsr_wass': wass['gsr'],
                # 'gsr_spec': spec['gsr'],
                **{k: v for d in ent_rows for k, v in d.items()}
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        # Apply Holm Correction per wavelet across subbands/measures
        p_cols = [c for c in df.columns if c.startswith('p_')]
        for col in p_cols:
            df[f'{col}_adj'] = self._holm_correction(df[col].values)
        return df

    def _perform_mann_whitney(self, group1, group2):
        u_stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
        n1, n2 = len(group1), len(group2)
        # Cohen's d calculation
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1) + (n2 - 1) * np.var(group2)) / (n1 + n2 - 2))
        d = mean_diff / (pooled_std + 1e-12)
        return {'p': p, 'cohen_d': d}

    def _holm_correction(self, p_values):
        n = len(p_values)
        indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
        adj_p = [0] * n
        for i, (original_idx, p) in enumerate(indexed_p):
            adj_p[original_idx] = min(1.0, p * (n - i))
        # Ensure monotonicity
        for i in range(1, n):
            idx_curr = indexed_p[i][0]
            idx_prev = indexed_p[i - 1][0]
            adj_p[idx_curr] = max(adj_p[idx_curr], adj_p[idx_prev])
        return adj_p

    def estimate_embedding_params(self, sample_segments, reference_wavelet='db2'):
        for subband in self.target_subbands:
            taus, ms = [], []
            for seg in sample_segments:
                sig = self._decompose(seg, reference_wavelet)[subband]
                try:
                    tau = AMI(sig, max_lag=50).get_first_minimum() or 1
                    m = FNN(signal=sig, tau=tau, max_dim=10).get_optimal_dimension() or 3
                    taus.append(max(1, min(int(tau), 20)));
                    ms.append(max(2, min(int(m), 10)))
                except:
                    taus.append(1); ms.append(3)
            self.embedding_params[subband] = {'tau': int(np.median(taus)), 'm': int(np.median(ms))}

    def rank_wavelets_by_power(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ranks wavelets using a composite of Effect Size, GSR, and Significance.
        """
        ranking_rows = []

        for wavelet, grp in results_df.groupby('wavelet'):
            # Target subbands (Clinical Signal)
            clinical = grp[grp['subband'].isin(['D3', 'D4'])]
            # Noise subbands (Control)
            noise = grp[grp['subband'].isin(['D1', 'D2'])]

            def get_score(sub_df, prefix='samp_en'):
                # Combine |Cohen's d| * GSR * Significance
                scores = np.abs(sub_df[f'd_{prefix}']) * sub_df['gsr_frob'] * (-np.log10(sub_df[f'p_{prefix}'] + 1e-12))
                return scores.mean()

            d3_d4_signal = get_score(clinical)
            d1_d2_noise = get_score(noise)

            # Final Score: Signal / (Noise + 1)
            # This rewards wavelets that isolate pre-AF dynamics specifically
            # in the lower frequencies.
            composite_score = d3_d4_signal / (d1_d2_noise + 1.0)

            ranking_rows.append({
                'wavelet': wavelet,
                'composite_score': composite_score,
                'avg_d_clinical': clinical['d_samp_en'].abs().mean(),
                'max_gsr': clinical['gsr_frob'].max()
            })

        return pd.DataFrame(ranking_rows).sort_values('composite_score', ascending=False)

    def run_full_analysis(self, segments_pre_af, segments_nsr, output_path=None):
        nsr_pids = list(segments_nsr.keys())
        selected_pids = self.rng.choice(nsr_pids, size=min(10, len(nsr_pids)), replace=False)
        sample_segs = [s for pid in selected_pids for s in segments_nsr[pid]]
        self.estimate_embedding_params(sample_segs)

        all_dfs = [self._analyze_wavelet(w, segments_pre_af, segments_nsr) for w in self.wavelets]
        results_df = pd.concat(all_dfs, ignore_index=True)

        if output_path: results_df.to_csv(output_path, index=False)

        scores = self.rank_wavelets_by_power(results_df)
        return results_df, scores