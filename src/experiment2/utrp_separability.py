import numpy as np
import pandas as pd
import pywt
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.stats import pearsonr, wasserstein_distance, mannwhitneyu
from numpy.linalg import eigvalsh
from itertools import combinations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recurrence_plot.AMI import AMI
from src.recurrence_plot.FNN import FNN
from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

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


def compute_sample_entropy(sig: np.ndarray, m: int = 2, r: float = 0.2) -> float:
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

            frob = compute_gsr(pre_af_utrps, nsr_utrps, frobenius_distance)

            ent_rows = []
            for m_name in ['samp_en', 'perm_en', 'wave_en']:
                pre_v = [d[2][m_name] for d in pre_af_data if d[2] is not None]
                nsr_v = [d[2][m_name] for d in nsr_data if d[2] is not None]
                res = self._perform_mann_whitney(pre_v, nsr_v)
                ent_rows.append({f'p_{m_name}': res['p'], f'd_{m_name}': res['cohen_d']})

            row = {
                'wavelet': wavelet, 'subband': subband,
                'gsr_frob': frob['gsr'],
                **{k: v for d in ent_rows for k, v in d.items()}
            }
            rows.append(row)

        df = pd.DataFrame(rows)
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