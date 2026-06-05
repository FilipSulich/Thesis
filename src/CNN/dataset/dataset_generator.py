import json
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import h5py
import numpy as np
import pywt
import wfdb
from scipy import signal as sp_signal
from tqdm import tqdm

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

DATA_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = DATA_ROOT / 'data_container' / 'raw_data' / 'paf'
OUTPUT_DIR = DATA_ROOT / 'data_container' / 'data' / 'processed_data_new'

#parameters
FS_ORIG = 128
FS_TARGET = 500
SEGMENT_SAMPLES = FS_TARGET * 10
N_SEGMENTS = 15
URP_SIZE = 256
WAVELET = 'db8'
SUBBAND_LEVELS = [2, 3, 4, 5, 6, 7, 8]
N_LEVELS_SWT = 10
PAD_SAMPLES = math.ceil(SEGMENT_SAMPLES / 2**N_LEVELS_SWT) * 2**N_LEVELS_SWT


def _pad(x: np.ndarray):
    """
    pad signal to power-of-2 length for SWT.

    Args:
        x (np.ndarray): input signal

    Returns:
        padded signal of length PAD_SAMPLES
    """
    pad = PAD_SAMPLES - len(x)
    if pad <= 0:
        return x[:PAD_SAMPLES]
    return np.pad(x, (0, pad), mode='reflect')


def _apply_soft_threshold(coeffs: List[Tuple[np.ndarray, np.ndarray]]):
    """
    apply soft thresholding to wavelet coefficients (following the Dohono and Johnstone thresholding approach).

    threshold: T = sigma * sqrt(2 * log(N))
    sigma: noise estimation from MAD (median absolute deviation)

    Args:
        coeffs (List[Tuple[np.ndarray, np.ndarray]]): list of (approximation, detail) coefficient tuples from SWT

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: list of denoised (approximation, detail) coefficient tuples
    """
    N_log = math.log2(FS_TARGET / 0.5)
    denoised = []

    for cA, cD in coeffs:
        sigma = np.median(np.abs(cD)) / 0.6745
        T = sigma * math.sqrt(2.0 * math.log(N_log))
        cD_clean = np.sign(cD) * np.maximum(np.abs(cD) - T, 0.0)
        denoised.append((cA, cD_clean))

    return denoised

def extract_subbands_no_denoise(signal_padded: np.ndarray,wavelet: str = WAVELET,n_levels: int = N_LEVELS_SWT,subband_levels: List[int] = SUBBAND_LEVELS):
    """
    extract subbands without denoising (direct iSWT per subband).

    decomposes signal with SWT, then reconstructs each subband individually by zeroing all other levels

    Args:
        signal_padded (np.ndarray): padded input signal
        wavelet (str): wavelet family to use
        n_levels (int): number of decomposition levels
        subband_levels (List[int]): which subband levels to extract

    Returns:
        Dict: dict of {'D2': ..., 'D3': ..., ...}
    """
    coeffs = pywt.swt(signal_padded, wavelet, level=n_levels)
    n_lvls = len(coeffs)
    subbands = {}

    for level in subband_levels:
        target_idx = n_lvls - level
        zeroed = [((np.zeros_like(cA), cD) if j == target_idx else (np.zeros_like(cA), np.zeros_like(cD))) for j, (cA, cD) in enumerate(coeffs)]
        sub = pywt.iswt(zeroed, wavelet)[:SEGMENT_SAMPLES]
        subbands[f'D{level}'] = sub.astype(np.float32)

    return subbands


def denoise_and_extract_subbands(signal_padded: np.ndarray,wavelet: str = WAVELET,n_levels: int = N_LEVELS_SWT,subband_levels: List[int] = SUBBAND_LEVELS):
    """
    single SWT decomposition with denoising and subband extraction.

    steps:
      1. decompose signal with SWT
      2. apply soft thresholding to all detail coefficients
      3. reconstruct full denoised signal
      4. reconstruct individual subbands by zeroing other levels

    Args:
        signal_padded (np.ndarray): padded input signal
        wavelet (str): wavelet family to use
        n_levels (int): number of decomposition levels
        subband_levels (List[int]): which subband levels to extract

    Returns:
        Tuple: (denoised signal, subbands dict) where denoised is full denoised signal (SEGMENT_SAMPLES,) and subbands is dict of {'D2': ..., 'D3': ..., ...}
    """
    coeffs = pywt.swt(signal_padded, wavelet, level=n_levels)
    denoised_coeffs = _apply_soft_threshold(coeffs)

    # full denoised signal
    denoised = pywt.iswt(denoised_coeffs, wavelet)[:SEGMENT_SAMPLES]

    # extract individual subbands
    n_lvls = len(denoised_coeffs)
    subbands = {}

    for level in subband_levels:
        target_idx = n_lvls - level
        # zero all coefficients except target level
        zeroed = [((np.zeros_like(cA), cD) if j == target_idx else (np.zeros_like(cA), np.zeros_like(cD))) for j, (cA, cD) in enumerate(denoised_coeffs)]
        sub = pywt.iswt(zeroed, wavelet)[:SEGMENT_SAMPLES]
        subbands[f'D{level}'] = (sub.astype(np.float32))

    return denoised, subbands

def compute_urp(signal: np.ndarray, tau: int, m: int, size: int = URP_SIZE):
    """
    compute unthresholded recurrence plot (URP) with JET colormap.

    Args:
        signal (np.ndarray): input signal
        tau (int): time delay for embedding
        m (int): embedding dimension
        size (int): output image size

    Returns:
        np.ndarray or None: RGB image (3, size, size) or None if computation fails
    """
    try:
        sig = signal[:SEGMENT_SAMPLES].astype(np.float32)
        n = len(sig) - (m - 1) * tau
        if n <= 0:
            return np.zeros((3, size, size), dtype=np.float32)

        gen = RecurrencePlotGenerator(
            signal=sig, time_delay=tau, embedding_dim=m, epsilon=0
        )
        rp = gen.compute_rp_unthresholded()

        # normalize to [0, 1]
        d_max = rp.max()
        if d_max > 1e-9:
            rp = rp / d_max

        # resize and apply colormap
        rp_r = cv2.resize(rp.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
        np.fill_diagonal(rp_r, 0.0)

        u8 = (rp_r * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        return rgb.transpose(2, 0, 1)

    except Exception as e:
        return None

def build_subject_groups(nsr_records: List[str], preaf_records: List[str]):
    """
    group paired recordings by subject

    NSR:   (n01, n01c, n02, n02c) → one subject group
    Pre-AF: (p01, p02)  → one subject group (paired odd/even)

    Args:
        nsr_records (List[str]): list of NSR record IDs
        preaf_records (List[str]): list of Pre-AF record IDs

    Returns:
        List[Tuple]: list of (group_id, record_ids, label)
    """
    nsr_set = set(nsr_records)
    preaf_set = set(preaf_records)
    subjects = []

    # NSR subjects: n01, n01c, n02, n02c are from same patient
    for i in range(1, 51, 2):
        recs = [r for r in [f'n{i:02d}', f'n{i:02d}c', f'n{i+1:02d}', f'n{i+1:02d}c'] if r in nsr_set]
        if recs:
            subjects.append((f'nsr_{i:02d}', recs, 0))

    # Pre-AF subjects: odd and even recordings are paired (p01-p02, p03-p04, ...)
    for i in range(1, 50, 2):
        recs = [r for r in [f'p{i:02d}', f'p{i+1:02d}'] if r in preaf_set]
        if recs:
            subjects.append((f'paf_{i:02d}', recs, 1))

    return subjects

def split_subjects_721(subjects: List[Tuple],seed: int = 42):
    """
    split subjects into train/val/test (70/20/10) while keeping pairs together.

    Args:
        subjects (List[Tuple]): list of (group_id, record_ids, label) tuples
        seed (int): random seed for reproducibility

    Returns:
        Dict: dict with keys 'train', 'val', 'test' containing record IDs
    """
    rng = random.Random(seed)
    preaf_s = [s for s in subjects if s[2] == 1]
    nsr_s = [s for s in subjects if s[2] == 0]
    rng.shuffle(preaf_s)
    rng.shuffle(nsr_s)

    def _split(lst):
        n = len(lst)
        n_test = max(1, round(n * 0.1))
        n_val = max(1, round(n * 0.2))
        return lst[n_test+n_val:], lst[n_test:n_test+n_val], lst[:n_test]

    pa_tr, pa_va, pa_te = _split(preaf_s)
    ns_tr, ns_va, ns_te = _split(nsr_s)

    def _extract(subj_list):
        recs = []
        for _, rs, _ in subj_list:
            recs.extend(rs)
        return sorted(recs)

    return {
        'train': _extract(pa_tr + ns_tr),
        'val': _extract(pa_va + ns_va),
        'test': _extract(pa_te + ns_te),
    }

class DatasetGenerator:
    KEYS = ['single'] + [f'D{l}' for l in SUBBAND_LEVELS]

    def __init__(self,output_dir: Path = OUTPUT_DIR,wavelet: str = WAVELET,mode: str = 'both',multi_denoise: bool = True):
        """
        args:
            output_dir: where to save HDF5 files
            wavelet: wavelet family (db8, sym4, etc.)
            mode: 'single', 'multi', or 'both' - what to generate
            multi_denoise: if True, use denoised subbands; if False, direct iSWT per subband
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wavelet = wavelet
        self.mode = mode
        self.multi_denoise = multi_denoise

        if mode not in ('single', 'multi', 'both'):
            raise ValueError(f"invalid mode: {mode}")

    def _load_signal(self, record_id: str):
        """
        load and resample ECG signal from WFDB record.

        Args:
            record_id (str): WFDB record identifier

        Returns:
            np.ndarray or None: resampled signal or None if loading fails
        """
        try:
            rec = wfdb.rdrecord(str(DATA_DIR / record_id))
            raw = rec.p_signal[:, 0].astype(np.float64)
            n = int(len(raw) * FS_TARGET / FS_ORIG)
            return sp_signal.resample(raw, n).astype(np.float32)
        except Exception as e:
            return None

    def _extract_segments(self, signal: np.ndarray):
        """
        extract N evenly-spaced segments from signal.

        Args:
            signal (np.ndarray): input signal

        Returns:
            List[np.ndarray]: list of signal segments
        """
        max_start = len(signal) - SEGMENT_SAMPLES
        if max_start <= 0:
            return []
        starts = np.linspace(0, max_start, N_SEGMENTS, dtype=int)
        segs = []
        for s in starts:
            seg = signal[s:s + SEGMENT_SAMPLES]
            if len(seg) == SEGMENT_SAMPLES:
                segs.append((seg.astype(np.float32)))
        return segs

    def _process_record(self,record_id: str,label: int,tau_single: int,m_single: int,subband_tau: Dict,subband_m: Dict):
        """
        process one record into URP images.

        Args:
            record_id (str): WFDB record identifier
            label (int): class label (0=NSR, 1=Pre-AF)
            tau_single (int): time delay for full-band URP
            m_single (int): embedding dimension for full-band URP
            subband_tau (Dict): dict of time delays per subband
            subband_m (Dict): dict of embedding dimensions per subband

        Returns:
            List[Dict]: list of sample dicts with images and metadata
        """
        signal = self._load_signal(record_id)
        if signal is None:
            return []

        segments = self._extract_segments(signal)
        if not segments:
            return []

        results = []
        for seg_idx, seg in enumerate(segments):
            seg_pad = _pad(seg)
            images = {}

            # full-band URP
            if self.mode in ('single', 'both'):
                # for full-band, always use denoised signal
                denoised, _ = denoise_and_extract_subbands(
                    seg_pad, self.wavelet, N_LEVELS_SWT, SUBBAND_LEVELS
                )
                single_URP = compute_urp(denoised, tau_single, m_single)
                if single_URP is None:
                    continue
                images['single'] = single_URP

            # multi-scale URPs
            if self.mode in ('multi', 'both'):
                if self.multi_denoise:
                    # use denoised subbands
                    _, subbands_multi = denoise_and_extract_subbands(seg_pad, self.wavelet, N_LEVELS_SWT, SUBBAND_LEVELS)
                else:
                    # use non-denoised subbands (direct iSWT)
                    subbands_multi = extract_subbands_no_denoise(seg_pad, self.wavelet, N_LEVELS_SWT, SUBBAND_LEVELS)

                skip_segment = False
                for level in SUBBAND_LEVELS:
                    name = f'D{level}'

                    # 3 used as fallback parameter values here
                    tau = subband_tau.get(name, 3)
                    m = subband_m.get(name, 3)

                    sub_URP = compute_urp(subbands_multi[name], tau, m)
                    if sub_URP is None:
                        skip_segment = True
                        break
                    images[name] = sub_URP

                if skip_segment:
                    continue

            results.append({
                'record_id': record_id,
                'segment_idx': seg_idx,
                'label': label,
                'images': images,
            })

        return results

    def _save_hdf5(self, samples: List[Dict], path: Path):
        """
        save processed samples to HDF5 file.

        Args:
            samples (List[Dict]): list of sample dicts with images and metadata
            path (Path): output HDF5 file path

        Returns:
            None
        """
        if not samples:
            return

        N = len(samples)

        labels = np.array([s['label'] for s in samples], dtype=np.int8)
        rids = np.array([s['record_id'].encode() for s in samples], dtype='S16')
        sidxs = np.array([s['segment_idx'] for s in samples], dtype=np.int16)

        keys_to_save = []
        if self.mode in ('single', 'both'):
            keys_to_save.append('single')
        if self.mode in ('multi', 'both'):
            keys_to_save.extend([f'D{l}' for l in SUBBAND_LEVELS])

        with h5py.File(path, 'w') as f:
            for key in keys_to_save:
                data = np.stack([s['images'][key] for s in samples])
                f.create_dataset(
                    key, data=data, dtype=np.float32,
                    compression='gzip', compression_opts=4
                )

            meta = f.create_group('metadata')
            meta.create_dataset('labels', data=labels)
            meta.create_dataset('record_ids', data=rids)
            meta.create_dataset('segment_indices', data=sidxs)

            denoising_str = 'soft-thresholding' if self.multi_denoise else 'none'
            f.attrs.update({
                'n_samples': N,
                'n_preaf': int((labels == 1).sum()),
                'n_nsr': int((labels == 0).sum()),
                'URP_size': URP_SIZE,
                'wavelet': self.wavelet,
                'mode': self.mode,
                'multi_denoise': self.multi_denoise,
                'subband_levels': ','.join(f'D{l}' for l in SUBBAND_LEVELS),
                'fs_target': FS_TARGET,
                'segment_samples': SEGMENT_SAMPLES,
                'colourmap': 'JET',
                'denoising': denoising_str,
            })

    def generate(self, params_file: Path, seed: int = 42) -> None:
        """
        generate complete URP dataset from ECG recordings.

        creates train/val/test splits with:
          - full-band URPs from denoised full-band signal
          - multi-scale URPs from denoised wavelet subbands (D2-D8)

        Args:
            params_file (Path): path to JSON file with tau and m parameters
            seed (int): random seed for train/val/test split

        Returns:
            None
        """
        with open(params_file) as f:
            params = json.load(f)
        tau_single = params['tau_single']
        m_single = params['m_single']
        subband_tau = params['subband_tau']
        subband_m = params['subband_m']

        nsr_records = ([f'n{i:02d}' for i in range(1, 51)] + [f'n{i:02d}c' for i in range(1, 51)])
        preaf_records = [f'p{i:02d}' for i in range(1, 51)]

        nsr_set = set(nsr_records)

        subjects = build_subject_groups(nsr_records, preaf_records)
        split = split_subjects_721(subjects, seed=seed)

        split_out = {**split, 'seed': seed, 'wavelet': self.wavelet}
        split_path = self.output_dir / 'split.json'
        with open(split_path, 'w') as f:
            json.dump(split_out, f, indent=2)

        for split_name in ('train', 'val', 'test'):
            samples = []
            records = split[split_name]
            for record_id in tqdm(records, desc=f'{split_name.capitalize()}'):
                label = 0 if record_id in nsr_set else 1
                samples.extend(self._process_record(
                    record_id, label,
                    tau_single, m_single,
                    subband_tau, subband_m
                ))
            self._save_hdf5(samples, self.output_dir / f'{split_name}.h5')