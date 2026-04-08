import numpy as np
import pandas as pd
import pywt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.stats import pearsonr
from itertools import combinations
from tqdm import tqdm

# Assuming internal project imports are available
from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

class UTRPSeparabilityAnalyzer:
    def __init__(
        self,
        wavelets: List[str],
        level: int = 4,
        fs: int = 128,
        target_subbands: Optional[List[str]] = None,
        rqa_target_rr: float = 0.1,
    ):
        self.wavelets = wavelets
        self.level = level
        self.fs = fs
        self.target_subbands = target_subbands or ['D1', 'D2', 'D3', 'D4', 'A4']
        self.rqa_target_rr = rqa_target_rr
        self.embedding_params: Dict[str, Dict[str, int]] = {}

    # ---------------------------------------------------------------- #
    #  RQA Feature Extraction (The Core Fix)                           #
    # ---------------------------------------------------------------- #

    def _extract_rqa_robust(self, signal: np.ndarray, subband: str) -> np.ndarray:
        """
        Extracts structural features while protecting against 'fake' 
        determinism using a Theiler Window.
        """
        p = self.embedding_params[subband]
        tau, m = p['tau'], p['m']
        
        # 1. Phase-space embedding
        n_pts = len(signal) - (m - 1) * tau
        emb = np.stack([signal[i*tau : i*tau + n_pts] for i in range(m)], axis=1)
        
        # 2. Distance Matrix and Thresholding
        dm = squareform(pdist(emb, 'euclidean'))
        
        # Use percentile to allow Recurrence Rate to vary between groups (it's a signal!)
        eps = np.percentile(dm, self.rqa_target_rr * 100)
        rp = (dm <= eps).astype(int)

        # 3. Theiler Window Calculation
        # We must skip diagonals near the main diagonal (LOI) to see the 'real' structure
        n = rp.shape[0]
        w = max(3, tau * 2) # Theiler Window
        
        diag_lengths = []
        for k in range(w, n - 2):
            diag = np.diag(rp, k=k)
            # Find continuous diagonal lines (length >= 2)
            # We use diff to find runs of 1s
            runs = np.diff(np.where(np.concatenate(([0], diag, [0])) == 0)[0]) - 1
            diag_lengths.extend(runs[runs >= 2])

        # 4. Feature Vector Construction
        # We exclude L_max because it is too volatile. L_mean and DET are more robust.
        total_rec = np.sum(rp) - n
        det = np.sum(diag_lengths) / (total_rec + 1e-12)
        l_mean = np.mean(diag_lengths) if diag_lengths else 0
        
        # Log-transform time-based features to normalize NSR variance
        return np.array([
            det,
            np.log10(l_mean + 1e-12),
            np.log10(np.var(dm) + 1e-12) # Global variance captures chaos
        ])

    # ---------------------------------------------------------------- #
    #  Main Analysis Logic                                             #
    # ---------------------------------------------------------------- #

    def analyze_wavelet(
        self,
        wavelet: str,
        segments_pre_af: Dict[str, List[np.ndarray]],
        segments_nsr: Dict[str, List[np.ndarray]],
    ) -> pd.DataFrame:
        
        rows = []
        for subband in self.target_subbands:
            print(f"  {wavelet} — {subband}", flush=True)

            def collect_patient_vectors(data_dict):
                p_vectors = []
                for pid, segs in data_dict.items():
                    s_vectors = []
                    for s in segs:
                        # Decompose and extract RQA
                        sub_sig = self._decompose_single(s, wavelet, subband)
                        rqa = self._extract_rqa_robust(sub_sig, subband)
                        s_vectors.append(rqa)
                    
                    # Aggregate segments to patient level using MEDIAN (ignores outliers)
                    if s_vectors:
                        p_vectors.append(np.median(s_vectors, axis=0))
                return p_vectors

            pre_rqas = collect_patient_vectors(segments_pre_af)
            nsr_rqas = collect_patient_vectors(segments_nsr)

            # --- Robust GSR Calculation ---
            # Use Robust Scaling (Median/IQR) to prevent NSR outliers from killing the ratio
            all_vecs = np.array(pre_rqas + nsr_rqas)
            med = np.median(all_vecs, axis=0)
            iqr = np.percentile(all_vecs, 75, axis=0) - np.percentile(all_vecs, 25, axis=0) + 1e-12
            
            pre_norm = [(v - med) / iqr for v in pre_rqas]
            nsr_norm = [(v - med) / iqr for v in nsr_rqas]

            # Use Cosine distance: measures the 'shape' of the RQA profile
            # rather than absolute magnitude, which is robust to heart-rate shifts.
            def gsr_calc(g1, g2):
                within = (np.mean([cosine(i, j) for i, j in combinations(g1, 2)]) + 
                          np.mean([cosine(i, j) for i, j in combinations(g2, 2)])) / 2
                bet = np.mean([cosine(i, j) for i in g1 for j in g2])
                return bet / (within + 1e-12)

            gsr_val = gsr_calc(pre_norm, nsr_norm)
            print(f"    GSR rqa={gsr_val:.4f}")

            rows.append({
                'wavelet': wavelet,
                'subband': subband,
                'gsr_rqa': gsr_val,
                'n_pre': len(pre_rqas),
                'n_nsr': len(nsr_rqas)
            })

        return pd.DataFrame(rows)

    def _decompose_single(self, signal, wavelet, subband):
        """Standard SWT decomposition and subband extraction."""
        coeffs = pywt.swt(signal, wavelet, level=4)
        idx = {'D1': 3, 'D2': 2, 'D3': 1, 'D4': 0, 'A4': 0}.get(subband)
        is_approx = (subband == 'A4')
        
        target = [(np.zeros_like(cA), np.zeros_like(cD)) for cA, cD in coeffs]
        if is_approx:
            target[0] = (coeffs[0][0], np.zeros_like(coeffs[0][1]))
        else:
            target[idx] = (np.zeros_like(coeffs[idx][0]), coeffs[idx][1])
            
        rec = pywt.iswt(target, wavelet)[:len(signal)]
        return (rec - np.mean(rec)) / (np.std(rec) + 1e-9)