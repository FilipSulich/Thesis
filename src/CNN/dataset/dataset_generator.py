import json
import multiprocessing
import random
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import h5py
import numpy as np
import pywt
import wfdb
from scipy import signal as sp_signal
from tqdm import tqdm

from src.recurrence_plot.AMI import AMI
from src.recurrence_plot.FNN import FNN
from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

warnings.filterwarnings('ignore', category=FutureWarning, module='pywt')

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path('/data/paf')

# ── Constants ──────────────────────────────────────────────────────────────────
FS_ORIG          = 128
FS_TARGET        = 256
SEGMENT_SAMPLES  = FS_TARGET * 10   # 5000 samples = 10 seconds
N_SEGMENTS       = 20
UTRP_SIZE        = 256
PAD_SAMPLES      = 5120             # next multiple of 2^10 above 5000

WAVELET          = 'db4'
N_LEVELS         = 8               # covers down to 0.25 Hz at 500 Hz

# Embedding parameters
# Single-scale: τ=3 at 128 Hz = 23ms lag. At 500 Hz equivalent = τ=12
# m=3 throughout — sufficient for ECG attractor dimensionality
TAU_SINGLE = 3
M_SINGLE   = 3

# Subband decomposition — single level-10 SWT, select detail levels D1-D9
USEFUL_LEVELS = [1, 2, 3, 4, 5, 6, 7]

SUBBAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']

# Tau and M will be estimated once using AMI/FNN and fixed across all subbands
# Will be loaded from JSON lookup file if it exists, otherwise estimated
SUBBAND_TAU = {}
SUBBAND_M = {}

N_SUBSET_PER_CLASS = 30  # Number of records per class for tau/m estimation


# ── Tau/M Parameter Estimation and Lookup ──────────────────────────────────────

def load_tau_m_params(params_file: Path) -> Optional[Dict]:
    """
    Load tau/m parameters from JSON file if it exists.

    Returns:
        Dict with 'tau_single', 'm_single', 'subband_tau', 'subband_m'
        where subband_tau/m are dicts mapping subband names to values
        or None if file doesn't exist
    """
    if not params_file.exists():
        return None

    with open(params_file, 'r') as f:
        params = json.load(f)

    print(f"Loaded tau/m params from {params_file}")
    print(f"  Single-scale: τ={params['tau_single']}, m={params['m_single']}")
    print(f"  Subbands:")
    for name in SUBBAND_NAMES:
        if name in params['subband_tau']:
            print(f"    {name}: τ={params['subband_tau'][name]}, m={params['subband_m'][name]}")

    return params


def save_tau_m_params(params: Dict, params_file: Path):
    """Save tau/m parameters to JSON file."""
    params_file.parent.mkdir(parents=True, exist_ok=True)

    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"Saved tau/m params to {params_file}")


def estimate_global_tau_m(nsr_records: List[str],
                          preaf_records: List[str],
                          n_per_class: int = N_SUBSET_PER_CLASS,
                          seed: int = 42) -> Dict:
    """
    Estimate tau/m for single-scale and for EACH subband (D1-D9) separately.

    Strategy:
      - Sample n_per_class from each class
      - For each record, take middle segment
      - Preprocess identically to full pipeline
      - Run AMI/FNN on single-scale signal → tau_single, m_single
      - Run AMI/FNN on EACH subband D1-D9 → tau/m per subband

    Returns:
        Dict with 'tau_single', 'm_single', 'subband_tau', 'subband_m'
        where subband_tau/m are dicts mapping subband names to values
    """
    import random
    rng = random.Random(seed)

    nsr_sample = rng.sample(nsr_records, min(n_per_class, len(nsr_records)))
    preaf_sample = rng.sample(preaf_records, min(n_per_class, len(preaf_records)))
    all_sample = nsr_sample + preaf_sample

    print(f"\nEstimating tau/m parameters on {len(all_sample)} records "
          f"({len(nsr_sample)} NSR + {len(preaf_sample)} Pre-AF)...")

    single_signals = []
    subband_signals = {name: [] for name in SUBBAND_NAMES}

    for rec_id in tqdm(all_sample, desc='Loading subset for tau/m estimation'):
        try:
            rec = wfdb.rdrecord(str(DATA_DIR / rec_id))
            raw = rec.p_signal[:, 0].astype(np.float64)
            n_tgt = int(len(raw) * FS_TARGET / FS_ORIG)
            res = sp_signal.resample(raw, n_tgt)

            # Middle segment
            mid_start = max(0, (len(res) - SEGMENT_SAMPLES) // 2)
            seg = res[mid_start:mid_start + SEGMENT_SAMPLES]
            if len(seg) < SEGMENT_SAMPLES:
                continue

            # Preprocess
            seg_norm = _minmax_norm(seg.astype(np.float32))
            seg_pad = _pad(seg_norm)
            # seg_den = np.clip(adaptive_wavelet_denoise(seg_pad)[:SEGMENT_SAMPLES],
            #                  0.0, 1.0).astype(np.float32)

            single_signals.append(seg_norm)

            # Extract ALL subbands
            subbands = extract_subbands(seg_pad)
            for name in SUBBAND_NAMES:
                if name in subbands:
                    subband_signals[name].append(subbands[name][:SEGMENT_SAMPLES])

        except Exception as e:
            print(f"  [WARN] {rec_id}: {e}")
            continue

    # Estimate single-scale tau/m
    print(f"\n→ Estimating single-scale tau/m ({len(single_signals)} signals):")
    tau_s, m_s = _estimate_tau_m_for_signals(single_signals)

    # Estimate tau/m for each subband separately
    subband_tau = {}
    subband_m = {}

    print(f"\n→ Estimating tau/m for each subband:")
    for name in SUBBAND_NAMES:
        signals = subband_signals[name]
        if len(signals) > 0:
            print(f"  {name} ({len(signals)} signals):")
            tau, m = _estimate_tau_m_for_signals(signals)
            subband_tau[name] = tau
            subband_m[name] = m
        else:
            print(f"  {name}: No signals, using default τ=12, m=3")
            subband_tau[name] = 12
            subband_m[name] = 3

    result = {
        'tau_single': tau_s,
        'm_single': m_s,
        'subband_tau': subband_tau,
        'subband_m': subband_m,
    }

    print(f"\nFinal tau/m estimates:")
    print(f"  Single-scale: τ={tau_s}, m={m_s}")
    print(f"  Subbands:")
    for name in SUBBAND_NAMES:
        print(f"    {name}: τ={subband_tau[name]}, m={subband_m[name]}")

    return result


def _estimate_tau_m_for_signals(signals: List[np.ndarray]) -> Tuple[int, int]:
    """Estimate tau and m for a list of signals using AMI/FNN."""
    taus, dims = [], []

    for sig in tqdm(signals, desc='  AMI/FNN', leave=False):
        tau, m = _get_optimal_parameters(sig)
        taus.append(tau)
        dims.append(m)

    tau_med = int(np.median(taus))
    m_med = int(np.median(dims))

    print(f"    τ values: min={min(taus)}, median={tau_med}, max={max(taus)}")
    print(f"    m values: min={min(dims)}, median={m_med}, max={max(dims)}")

    return tau_med, m_med


def _get_optimal_parameters(signal: np.ndarray) -> Tuple[int, int]:
    """Estimate tau and m for a single signal using AMI and FNN."""
    N = len(signal)
    if N < 10:
        return 1, 1

    tau_candidate = AMI(signal=signal).get_first_minimum()

    if tau_candidate is None:
        tau = max(1, N // 100)
    else:
        tau = int(tau_candidate)

    try:
        max_m = max(2, N // (tau if tau > 0 else 1))
        m_obj = FNN(tau=tau, signal=signal)
        m = m_obj.get_optimal_dimension()

        if m >= max_m:
            m = max(1, max_m - 1)
    except Exception as e:
        print(f"  Warning: FNN failed ({e}), using fallback m=2")
        m = 2

    return tau, m


def _minmax_norm(seg: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0,1]."""
    s_min, s_max = seg.min(), seg.max()
    return (seg - s_min) / (s_max - s_min + 1e-9)


def _pad(seg: np.ndarray) -> np.ndarray:
    """Reflect-pad to PAD_SAMPLES for level-10 SWT."""
    pad = PAD_SAMPLES - len(seg)
    if pad <= 0:
        return seg[:PAD_SAMPLES]
    return np.pad(seg, (0, pad), mode='reflect')


def pad_to_dyadic_length(signal: np.ndarray, min_length: int = None) -> Tuple[np.ndarray, int]:
    """
    Pad signal to dyadic length (power of 2).
    
    Args:
        signal: Input signal (1D)
        min_length: Minimum length after padding (default: use signal length)
    
    Returns:
        (padded_signal, pad_length) tuple
    """
    if min_length is None:
        min_length = len(signal)
    
    # Find next power of 2 >= max(len(signal), min_length)
    target_length = max(len(signal), min_length)
    dyadic_length = 2 ** int(np.ceil(np.log2(target_length)))
    
    pad_amount = dyadic_length - len(signal)
    if pad_amount <= 0:
        return signal[:dyadic_length], 0
    
    padded = np.pad(signal, (0, pad_amount), mode='reflect')
    return padded, pad_amount


def extract_subbands_swt(signal: np.ndarray, wavelet: str, level: int, 
                        target_length: int = None, pad_mode: str = 'reflect') -> Dict[str, np.ndarray]:
    """
    Extract subbands from signal using SWT (Stationary Wavelet Transform) reconstruction.
    
    Procedure:
      1. Pad signal to dyadic length (or target_length if specified)
      2. Perform SWT decomposition at specified level
      3. Reconstruct each subband by zeroing all levels except target
      4. Return subband dict
    
    Args:
        signal: Input signal (1D numpy array, float32)
        wavelet: Wavelet name (e.g., 'db4', 'db5', 'sym4')
        level: Decomposition level (1-10 typical)
        target_length: Length after padding. If None, pads to dyadic length of signal
        pad_mode: Padding mode ('reflect', 'edge', 'wrap', etc.)
    
    Returns:
        Dict mapping subband names ('D1', 'D2', ..., f'D{level}') to (target_length,) float32 arrays
    
    Example:
        >>> signal = np.random.randn(5000)
        >>> subbands = extract_subbands_swt(signal, wavelet='db4', level=7)
        >>> print(subbands.keys())  # dict_keys(['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'])
    """
    # Step 1: Pad to dyadic length
    if target_length is None:
        padded_signal, _ = pad_to_dyadic_length(signal)
        target_length = len(padded_signal)
    else:
        # Pad to specified length
        pad_amount = target_length - len(signal)
        if pad_amount <= 0:
            padded_signal = signal[:target_length]
        else:
            padded_signal = np.pad(signal, (0, pad_amount), mode=pad_mode)
    
    padded_signal = padded_signal.astype(np.float32)
    
    # Step 2: SWT decomposition
    coeffs = pywt.swt(padded_signal, wavelet, level=level)
    n_levels = len(coeffs)
    
    subbands = {}
    
    # Step 3: Reconstruct each subband
    for target_level in range(1, level + 1):
        # SWT returns coeffs from coarse to fine: (cA_n, cD_n), ..., (cA_1, cD_1)
        # So target_level=1 is the finest (last in list)
        target_idx = n_levels - target_level
        
        # Zero all coefficients except target
        zeroed_coeffs = []
        for j, (cA, cD) in enumerate(coeffs):
            if j == target_idx:
                # Keep this level's detail coefficients
                zeroed_coeffs.append((np.zeros_like(cA), cD))
            else:
                # Zero out all others
                zeroed_coeffs.append((np.zeros_like(cA), np.zeros_like(cD)))
        
        # Reconstruct
        subband = pywt.iswt(zeroed_coeffs, wavelet)[:target_length].astype(np.float32)
        subbands[f'D{target_level}'] = subband
    
    return subbands



def adaptive_wavelet_denoise(signal: np.ndarray) -> np.ndarray:
    """
    Adaptive wavelet denoising (Donoho & Johnstone 1994).
    Soft thresholding with MAD noise estimator per level.
    """
    coeffs   = pywt.swt(signal, WAVELET, level=N_LEVELS)
    denoised = []
    for cA, cD in coeffs:
        delta  = np.median(np.abs(cD)) / 0.6745
        T      = delta * np.sqrt(2.0 * np.log(N_LEVELS))
        cD_den = np.sign(cD) * np.maximum(np.abs(cD) - T, 0.0)
        denoised.append((cA, cD_den))
    return pywt.iswt(denoised, WAVELET)


def extract_subbands(signal_padded: np.ndarray) -> Dict[str, np.ndarray]:
    coeffs   = pywt.swt(signal_padded, WAVELET, level=N_LEVELS)
    n_levels = len(coeffs)
    subbands = {}

    for target_level in USEFUL_LEVELS:
        target_idx = n_levels - target_level
        zeroed = []
        for j, (cA, cD) in enumerate(coeffs):
            if j == target_idx:
                zeroed.append((np.zeros_like(cA), cD))
            else:
                zeroed.append((np.zeros_like(cA), np.zeros_like(cD)))

        sub = pywt.iswt(zeroed, WAVELET)[:SEGMENT_SAMPLES].astype(np.float32)

        # Min-max normalise the subband — same as single-scale signal preprocessing
        # This makes each subband's distance matrix comparable in scale to single-scale
        sub = _minmax_norm(sub)

        subbands[f'D{target_level}'] = sub

    return subbands

def compute_distance_matrix(signal: np.ndarray, tau: int, m: int) -> np.ndarray:
    rp = RecurrencePlotGenerator(signal[:SEGMENT_SAMPLES], tau, m, epsilon=0)
    utrp = rp.compute_rp_unthresholded()
    return utrp


def create_binary_rp(signal, tau, m, output_size=256):
    sig = signal[:SEGMENT_SAMPLES]
    n = len(sig) - (m - 1) * tau
    if n <= 0:
        return np.zeros((3, output_size, output_size), dtype=np.float32)

    embedded = np.array([sig[i:i + m * tau:tau] for i in range(n)])
    diff = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2)).astype(np.float32)

    # Binary threshold: ε = 10% of max distance
    epsilon = 0.1 * distances.max()
    binary = (distances < epsilon).astype(np.float32)

    binary_r = cv2.resize(binary, (output_size, output_size),
                          interpolation=cv2.INTER_LINEAR).astype(np.float32)
    np.fill_diagonal(binary_r, 0.0)

    # JET colourmap on binary image
    u8 = (binary_r * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)

def create_utrp(signal: np.ndarray, tau: int, m: int, output_size: int = UTRP_SIZE):
    """
    Raw distance matrix → RGB image.
    Normalises by this segment's own max distance so the full
    colour range is used. No clipping, no global constants.
    """
    sig = signal[:SEGMENT_SAMPLES]
    n   = len(sig) - (m - 1) * tau
    if n <= 0:
        return np.zeros((3, output_size, output_size), dtype=np.float32)

    embedded  = np.array([sig[i:i + m * tau:tau] for i in range(n)])
    diff      = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2)).astype(np.float32)

    # Normalise by max so full JET range is used — no clipping
    d_max = distances.max()
    if d_max > 1e-9:
        distances = distances / d_max

    # Resize
    dist_r = cv2.resize(distances, (output_size, output_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    np.fill_diagonal(dist_r, 0.0)

    # Convert to RGB
    u8 = (dist_r * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Return both the RGB image and the raw resized distance matrix
    return rgb.transpose(2, 0, 1), dist_r  # (3, H, W), (H, W)


@dataclass
class ZhuSample:
    single_scale_utrp: np.ndarray   # (3, 256, 256) float32 RGB
    single_scale_dist: np.ndarray   # (256, 256) float32 raw distance matrix (resized)
    multi_scale_utrp:  np.ndarray   # (9, 3, 256, 256) float32 RGB — D1-D9
    label:             int
    record_id:         str
    segment_idx:       int
    time_to_onset:     float


def get_time_to_onset(record_id: str, segment_idx: int) -> float:
    import re
    match = re.match(r'p(\d+)', record_id)
    if match:
        num = int(match.group(1))
        if num % 2 == 1:
            max_start     = 30 * 60 * FS_TARGET - SEGMENT_SAMPLES
            starts        = np.linspace(0, max_start, N_SEGMENTS, dtype=int)
            seg_start_sec = starts[segment_idx] / FS_TARGET
            return float(30 * 60 - seg_start_sec)
        else:
            return float(45 * 60)
    return float('nan')


def denoise_and_extract_subbands(signal_padded: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Single SWT decomposition that simultaneously denoises and extracts subbands.

    Steps:
      1. SWT decomposition of normalised+padded signal
      2. Soft threshold ALL detail coefficients (denoising)
      3. Reconstruct full denoised signal from all thresholded coefficients
      4. Reconstruct each subband by zeroing all levels except the target
         (using already-thresholded coefficients — no separate denoising pass)

    Returns:
      denoised: (SEGMENT_SAMPLES,) float32 — full denoised signal
      subbands: dict {name: (SEGMENT_SAMPLES,) float32} — denoised subband signals
    """
    coeffs = pywt.swt(signal_padded, WAVELET, level=N_LEVELS)

    # Step 1: Soft threshold all detail coefficients
    fs = 500
    fl = 0.5
    denoised_coeffs = []
    for cA, cD in coeffs:
        delta = np.median(np.abs(cD)) / 0.6745

        N = np.log2(fs/fl)

        T = delta * np.sqrt(2.0 * np.log(N))

        # T_old = delta * np.sqrt(2.0 * np.log(N_LEVELS))


        cD_den = np.sign(cD) * np.maximum(np.abs(cD) - T, 0.0)

        # cD_den_old = np.sign(cD) * np.maximum(np.abs(cD) - T_old, 0.0)

        # print("new")
        # print(T, cD_den)
        # print("")
        # print("old")
        # print(T_old, cD_den_old)

        denoised_coeffs.append((cA, cD_den))

    # Step 2: Reconstruct full denoised signal
    # denoised = np.clip(
    #     pywt.iswt(denoised_coeffs, WAVELET)[:SEGMENT_SAMPLES],
    #     0.0, 1.0
    # ).astype(np.float32)

    denoised = pywt.iswt(denoised_coeffs, WAVELET)[:SEGMENT_SAMPLES].astype(np.float32)

    # Step 3: Reconstruct each subband from denoised coefficients
    # Zero all levels except target — uses already-thresholded cD
    n_levels = len(coeffs)
    # n_levels = len(denoised_coeffs)
    subbands = {}

    for target_level in USEFUL_LEVELS:
        target_idx = n_levels - target_level
        zeroed = []
        # for j, (cA, cD) in enumerate(denoised_coeffs):
        for j, (cA, cD) in enumerate(coeffs):
            if j == target_idx:
                zeroed.append((np.zeros_like(cA), cD))  # keep denoised cD
            else:
                zeroed.append((np.zeros_like(cA), np.zeros_like(cD)))

        sub = pywt.iswt(zeroed, WAVELET)[:SEGMENT_SAMPLES].astype(np.float32)
        # sub = _minmax_norm(sub)
        subbands[f'D{target_level}'] = sub

    return denoised, subbands
    # return signal_padded, subbands

def process_segment(args: Tuple) -> Optional[ZhuSample]:
    (record_id, label, segment_idx, tau_single, m_single, subband_tau, subband_m) = args
    try:
        rec = wfdb.rdrecord(str(DATA_DIR / record_id))
        raw = rec.p_signal[:, 0].astype(np.float64)

        n_tgt = int(len(raw) * FS_TARGET / FS_ORIG)
        res   = sp_signal.resample(raw, n_tgt)

        max_start = len(res) - SEGMENT_SAMPLES
        if max_start <= 0:
            return None
        starts = np.linspace(0, max_start, N_SEGMENTS, dtype=int)
        seg    = res[starts[segment_idx]:starts[segment_idx] + SEGMENT_SAMPLES]

        if len(seg) < SEGMENT_SAMPLES:
            return None

        seg_norm = _minmax_norm(seg.astype(np.float32))
        seg_pad = _pad(seg_norm)

        seg_den, subbands = denoise_and_extract_subbands(seg_pad)

        # create_utrp now returns (rgb, dist_r)
        ss_utrp, ss_dist = create_utrp(seg_den, tau_single, m_single, UTRP_SIZE)
        # ss_utrp, ss_dist = create_utrp(seg_norm, tau_single, m_single, UTRP_SIZE)

        # Subband RPs
        ms_channels = []
        for name in SUBBAND_NAMES:
            # For subbands we only need the RGB images for the model; discard raw dist
            rgb, _ = create_utrp(subbands[name], subband_tau[name], subband_m[name], UTRP_SIZE)
            ms_channels.append(rgb)

        ms_utrp = np.stack(ms_channels, axis=0).reshape(-1, UTRP_SIZE, UTRP_SIZE)

        return ZhuSample(
            single_scale_utrp=ss_utrp,
            single_scale_dist=ss_dist,
            multi_scale_utrp=ms_utrp,
            label=label,
            record_id=record_id,
            segment_idx=segment_idx,
            time_to_onset=get_time_to_onset(record_id, segment_idx),
        )

    except Exception as e:
        print(f"[WARN] {record_id} seg {segment_idx}: {e}")
        return None

def process_segment_paper(args: Tuple) -> Optional[np.ndarray]:
    """Process segment with NO pre-UTRP normalization (paper pipeline)."""
    (record_id, label, segment_idx, tau, m) = args
    try:
        rec = wfdb.rdrecord(str(DATA_DIR / record_id))
        raw = rec.p_signal[:, 0].astype(np.float64)
        
        # Resample to target FS
        n_tgt = int(len(raw) * FS_TARGET / FS_ORIG)
        res = sp_signal.resample(raw, n_tgt)
        
        # Extract segment
        max_start = len(res) - SEGMENT_SAMPLES
        if max_start <= 0:
            return None
        starts = np.linspace(0, max_start, N_SEGMENTS, dtype=int)
        seg = res[starts[segment_idx]:starts[segment_idx] + SEGMENT_SAMPLES]
        
        if len(seg) < SEGMENT_SAMPLES:
            return None
        
        # NO normalization — use raw resampled signal
        seg_raw = seg.astype(np.float32)
        
        # Generate UTRP from raw signal (helper)
        sig = seg_raw[:SEGMENT_SAMPLES]
        n = len(sig) - (m - 1) * tau
        if n <= 0:
            return None
        
        # Embed signal
        embedded = np.array([sig[i:i + m * tau:tau] for i in range(n)])
        diff = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2)).astype(np.float32)
        
        # Normalize to 0-1 range
        d_max = distances.max()
        if d_max > 1e-9:
            distances = distances / d_max
        
        # Resize to 224x224
        dist_r = cv2.resize(distances, (224, 224), 
                           interpolation=cv2.INTER_LINEAR).astype(np.float32)
        
        # Zero diagonal
        np.fill_diagonal(dist_r, 0.0)
        
        # Convert to RGB (JET colormap)
        u8 = (dist_r * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        return rgb.transpose(2, 0, 1)  # (3, H, W)
    except Exception as e:
        print(f"[WARN] {record_id} seg {segment_idx}: {e}")
        return None

def build_subject_groups(preaf_records, nsr_records):
    subjects  = []
    preaf_set = set(preaf_records)
    nsr_set   = set(nsr_records)

    for i in range(1, 51):
        recs = [r for r in [f'n{i:02d}', f'n{i:02d}c'] if r in nsr_set]
        if recs:
            subjects.append((f'nsr_{i:02d}', recs, 0))

    for i in range(1, 50, 2):
        recs = [r for r in [f'p{i:02d}', f'p{i+1:02d}'] if r in preaf_set]
        if recs:
            subjects.append((f'paf_{i:02d}', recs, 1))

    return subjects

def split_subjects_721(subjects, seed=42):
    rng = random.Random(seed)

    preaf_subj = [s for s in subjects if s[2] == 1]
    nsr_subj   = [s for s in subjects if s[2] == 0]
    rng.shuffle(preaf_subj); rng.shuffle(nsr_subj)

    def split(lst):
        n      = len(lst)
        n_test = max(1, round(n * 0.1))
        n_val  = max(1, round(n * 0.2))
        return lst[n_test+n_val:], lst[n_test:n_test+n_val], lst[:n_test]

    pa_tr, pa_va, pa_te = split(preaf_subj)
    ns_tr, ns_va, ns_te = split(nsr_subj)

    def extract(subj_list):
        recs = []
        for _, rs, _ in subj_list:
            recs.extend(rs)
        return sorted(recs)

    return {
        'train': extract(pa_tr + ns_tr),
        'val':   extract(pa_va + ns_va),
        'test':  extract(pa_te + ns_te),
    }

class ZhuDatasetGenerator:
    def generate(self, output_path: Path, seed: int = 42) -> None:
        global TAU_SINGLE, M_SINGLE, SUBBAND_TAU, SUBBAND_M

        nsr_records   = ([f'n{i:02d}'  for i in range(1, 51)] + [f'n{i:02d}c' for i in range(1, 51)])

        preaf_records = [f'p{i:02d}' for i in range(1, 51)]

        all_records = ([(r, 0) for r in nsr_records] + [(r, 1) for r in preaf_records])

        print(f"Records: {len(all_records)} "f"(Learning: NSR={len(nsr_records)}, Pre-AF={len(preaf_records)}) ")

        # ── Load or estimate tau/m parameters ──────────────────────────────
        params_file = output_path.parent / 'tau_m_params.json'
        params = load_tau_m_params(params_file)

        if params is None:
            print("\nNo tau/m params found, estimating from data...")
            params = estimate_global_tau_m(nsr_records, preaf_records, seed=seed)
            save_tau_m_params(params, params_file)

        # Set global parameters
        TAU_SINGLE = params['tau_single']
        M_SINGLE = params['m_single']

        # Use per-subband tau/m values
        SUBBAND_TAU = params['subband_tau']
        SUBBAND_M = params['subband_m']

        print(f"\nUsing parameters:")
        print(f"  Single-scale: τ={TAU_SINGLE}, m={M_SINGLE}")
        print(f"  Subbands:")
        for name in SUBBAND_NAMES:
            print(f"    {name}: τ={SUBBAND_TAU[name]}, m={SUBBAND_M[name]}")

        #
        # Split
        subjects = build_subject_groups(preaf_records, nsr_records)
        split    = split_subjects_721(subjects, seed=seed)

        nsr_set   = set(nsr_records)
        preaf_set = set(preaf_records)
        for name in ('train', 'val', 'test'):
            recs    = split[name]
            n_nsr   = sum(1 for r in recs if r in nsr_set)
            n_preaf = sum(1 for r in recs if r in preaf_set)
            print(f"  {name:5s}: {len(recs)} records "
                  f"(NSR={n_nsr} Pre-AF={n_preaf}) "
                  f"→ ~{len(recs)*N_SEGMENTS} segments")

        split_out = {
            **split,
            'seed':               seed,
            'segments_per_record': N_SEGMENTS,
            'tau_single':         TAU_SINGLE,
            'm_single':           M_SINGLE,
            'subband_tau':        SUBBAND_TAU,
            'subband_m':          SUBBAND_M,
            'subband_names':      SUBBAND_NAMES,
            # 'global_scales':      scales,
            'note': ('Subject-level 7:2:1 split. Global distance scales '
                     'estimated from median p95 across 60 sampled records.'),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        split_path = output_path.with_suffix('.split.json')
        with open(split_path, 'w') as f:
            json.dump(split_out, f, indent=2)
        print(f"\nSplit + scales saved → {split_path}")

        # Build tasks (include tau/m params for multiprocessing)
        tasks = [
            (rec_id, label, seg_idx, TAU_SINGLE, M_SINGLE, SUBBAND_TAU, SUBBAND_M)
            for rec_id, label in all_records
            for seg_idx in range(N_SEGMENTS)
        ]
        print(f"Total tasks: {len(tasks)}")

        # Process
        n_workers = min(multiprocessing.cpu_count(), 8)
        samples: List[ZhuSample] = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(process_segment, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(tasks),
                            desc='Generating UTRPs'):
                r = fut.result()
                if r is not None:
                    samples.append(r)

        print(f"Processed {len(samples)}/{len(tasks)} segments")

        self._save_hdf5(samples, output_path,  split_out)

    def _save_hdf5(self, samples: List[ZhuSample], path: Path, split_info: dict) -> None:
        print(f"\nSaving → {path}")

        ss    = np.stack([s.single_scale_utrp for s in samples])
        sd    = np.stack([s.single_scale_dist for s in samples])
        ms    = np.stack([s.multi_scale_utrp  for s in samples])
        lbls  = np.array([s.label             for s in samples], dtype=np.int8)
        rids  = np.array([s.record_id.encode() for s in samples], dtype='S16')
        sidxs = np.array([s.segment_idx       for s in samples], dtype=np.int16)
        t2os  = np.array([s.time_to_onset     for s in samples], dtype=np.float32)

        with h5py.File(path, 'w') as f:
            grp = f.create_group('single_scale')
            grp.create_dataset('utrps', data=ss, dtype=np.float32, compression='gzip', compression_opts=4)
            grp.create_dataset('distances', data=sd, dtype=np.float32, compression='gzip', compression_opts=4)

            f.create_group('multi_scale').create_dataset('utrps', data=ms, dtype=np.float32, compression='gzip', compression_opts=4)

            meta = f.create_group('metadata')
            meta.create_dataset('labels',          data=lbls)
            meta.create_dataset('record_ids',      data=rids)
            meta.create_dataset('segment_indices', data=sidxs)
            meta.create_dataset('time_to_onset',   data=t2os)

            f.attrs.update({
                'n_samples':             len(samples),
                'n_preaf':               int((lbls == 1).sum()),
                'n_nsr':                 int((lbls == 0).sum()),
                'utrp_size':             UTRP_SIZE,
                'tau_single':            TAU_SINGLE,
                'm_single':              M_SINGLE,
                'subband_names':         ','.join(SUBBAND_NAMES),
                'fs_target':             FS_TARGET,
                'segment_samples':       SEGMENT_SAMPLES,
                'n_segments_per_record': N_SEGMENTS,
                'normalisation':         'global_median_p95',
            })

        print(f"Saved {len(samples)} — "f"Pre-AF={int((lbls==1).sum())}  NSR={int((lbls==0).sum())}")
        print(f"Single-scale: {ss.shape}  Multi-scale: {ms.shape}")

    def generate_utrp_paper_pipeline(self, output_path: Path, seed: int = 42) -> None:
        """
        Generate UTRPs following the paper's exact pipeline:
        1. NO signal normalization (use raw ECG after resampling and optional denoising)
        2. Euclidean distance matrix (unthresholded)
        3. Color mapping to 0-1 range
        4. Resize to 224x224
        5. Compute and subtract training set pixel means
        
        This is distinct from the existing multi-scale UTRP generation.
        """
        global TAU_SINGLE, M_SINGLE, SUBBAND_TAU, SUBBAND_M
        
        nsr_records   = ([f'n{i:02d}'  for i in range(1, 51)] +
                         [f'n{i:02d}c' for i in range(1, 51)])
        preaf_records = [f'p{i:02d}' for i in range(1, 51)]
        
        all_records = ([(r, 0) for r in nsr_records] +
                      [(r, 1) for r in preaf_records])
        
        print(f"Paper Pipeline UTRP Generation")
        print(f"Records: {len(all_records)} (NSR={len(nsr_records)}, Pre-AF={len(preaf_records)})")
        
        # Load or estimate tau/m parameters
        params_file = output_path.parent / 'tau_m_params.json'
        params = load_tau_m_params(params_file)
        
        if params is None:
            print("\nNo tau/m params found, estimating from data...")
            params = estimate_global_tau_m(nsr_records, preaf_records, seed=seed)
            save_tau_m_params(params, params_file)
        
        TAU_SINGLE = params['tau_single']
        M_SINGLE = params['m_single']
        SUBBAND_TAU = params['subband_tau']
        SUBBAND_M = params['subband_m']
        
        print(f"\nUsing parameters:")
        print(f"  Single-scale: τ={TAU_SINGLE}, m={M_SINGLE}")
        
        # Build split
        subjects = build_subject_groups(preaf_records, nsr_records)
        split = split_subjects_721(subjects, seed=seed)
        
        nsr_set = set(nsr_records)
        preaf_set = set(preaf_records)
        for name in ('train', 'val', 'test'):
            recs = split[name]
            n_nsr = sum(1 for r in recs if r in nsr_set)
            n_preaf = sum(1 for r in recs if r in preaf_set)
            print(f"  {name:5s}: {len(recs)} records (NSR={n_nsr} Pre-AF={n_preaf}) → ~{len(recs)*N_SEGMENTS} segments")
        
        # Process all samples (no normalization before UTRP generation)
        tasks = [
            (rec_id, label, seg_idx, TAU_SINGLE, M_SINGLE)
            for rec_id, label in all_records
            for seg_idx in range(N_SEGMENTS)
        ]
        print(f"Total tasks: {len(tasks)}")
        
        # Process segments using multiprocessing
        n_workers = min(multiprocessing.cpu_count(), 8)
        samples_raw = []
        sample_info = []  # (record_id, label, segment_idx)
        
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(process_segment_paper, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(tasks), desc='Generating paper-style UTRPs'):
                result = fut.result()
                args = futures[fut]
                record_id, label, segment_idx, _, _ = args
                if result is not None:
                    samples_raw.append(result)
                    sample_info.append((record_id, label, segment_idx))
        
        print(f"Processed {len(samples_raw)}/{len(tasks)} segments")
        
        # Compute pixel means from training set only
        train_recs = set(split['train'])
        train_indices = [i for i, (rid, _, _) in enumerate(sample_info) if rid in train_recs]
        
        if train_indices:
            train_images = np.stack([samples_raw[i] for i in train_indices], axis=0)  # (N, 3, 224, 224)
            pixel_means = train_images.mean(axis=0)  # (3, 224, 224)
            print(f"Computed pixel means from {len(train_indices)} training samples")
        else:
            pixel_means = np.zeros((3, 224, 224), dtype=np.float32)
            print("Warning: No training samples found; using zero means")
        
        # Subtract means from all samples
        samples_centered = np.stack(samples_raw, axis=0)  # (N, 3, 224, 224)
        samples_centered = samples_centered - pixel_means[np.newaxis, ...]
        
        # Save to HDF5
        self._save_hdf5_paper_pipeline(
            samples_centered, sample_info, split, output_path, pixel_means
        )
    
    def _save_hdf5_paper_pipeline(self, images: np.ndarray, sample_info: list, split: dict, 
                                  path: Path, pixel_means: np.ndarray) -> None:
        """Save paper-pipeline UTRPs to HDF5."""
        print(f"\nSaving paper-pipeline UTRPs → {path}")
        
        labels = np.array([label for _, label, _ in sample_info], dtype=np.int8)
        record_ids = np.array([rid.encode() for rid, _, _ in sample_info], dtype='S16')
        segment_indices = np.array([sidx for _, _, sidx in sample_info], dtype=np.int16)
        
        with h5py.File(path, 'w') as f:
            # Main UTRP data
            f.create_dataset('images', data=images, dtype=np.float32, compression='gzip', compression_opts=4)
            
            # Metadata
            meta = f.create_group('metadata')
            meta.create_dataset('labels', data=labels)
            meta.create_dataset('record_ids', data=record_ids)
            meta.create_dataset('segment_indices', data=segment_indices)
            
            # Pixel means for reproducibility
            f.create_dataset('pixel_means', data=pixel_means, dtype=np.float32)
            
            # Split information
            split_info = {
                'train': split['train'],
                'val': split['val'],
                'test': split['test'],
            }
            f.attrs['split_info'] = json.dumps(split_info)
            
            # Summary attributes
            f.attrs.update({
                'n_samples': len(images),
                'n_preaf': int((labels == 1).sum()),
                'n_nsr': int((labels == 0).sum()),
                'image_size': 224,
                'tau_single': TAU_SINGLE,
                'm_single': M_SINGLE,
                'fs_target': FS_TARGET,
                'segment_samples': SEGMENT_SAMPLES,
                'n_segments_per_record': N_SEGMENTS,
                'normalization': 'no_pre_utrp_norm_with_pixel_mean_subtraction',
                'paper_pipeline': True,
            })
        
        print(f"Saved {len(images)} samples — Pre-AF={int((labels==1).sum())} NSR={int((labels==0).sum())}")
        print(f"Images shape: {images.shape}, Pixel means shape: {pixel_means.shape}")


if __name__ == '__main__':
    OUTPUT_PATH = Path(
        '/data/data_new/attention/utrp_256_level_8.h5'
    )
    gen = ZhuDatasetGenerator()
    gen.generate(output_path=OUTPUT_PATH, seed=42)
    # gen.generate_utrp_paper_pipeline(output_path=OUTPUT_PATH, seed=42)



