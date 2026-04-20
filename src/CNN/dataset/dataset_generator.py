import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import pywt
import wfdb
from scipy import signal as sp_signal
import os
import sys

from src.CNN.dataset.visualizer import UTRPVisualiser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

warnings.filterwarnings('ignore')

DATA_DIR        = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/paf')
OUTPUT_DIR      = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data_final/utrp_dataset_db8_no_denoise')

FS_ORIG         = 128
FS_TARGET       = 500
SEGMENT_SAMPLES = FS_TARGET * 10
N_SEGMENTS      = 15
UTRP_SIZE       = 256
PAD_SAMPLES     = 8192

WAVELET         = 'db8'
SUBBAND_LEVELS  = [2, 3, 4, 5, 6, 7, 8]
N_LEVELS_SWT    = 10
EPSILON_FRAC    = 0.10



def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


def _pad(x: np.ndarray) -> np.ndarray:
    pad = PAD_SAMPLES - len(x)
    if pad <= 0:
        return x[:PAD_SAMPLES]
    return np.pad(x, (0, pad), mode='reflect')


def extract_subbands_no_denoise(
        signal_padded: np.ndarray,
        wavelet: str = WAVELET,
        n_levels: int = N_LEVELS_SWT,
        subband_levels: List[int] = SUBBAND_LEVELS,
) -> Dict[str, np.ndarray]:
    coeffs = pywt.swt(signal_padded, wavelet, level=n_levels)
    n_lvls = len(coeffs)
    subbands: Dict[str, np.ndarray] = {}

    for level in subband_levels:
        target_idx = n_lvls - level
        zeroed = [
            ((np.zeros_like(cA), cD) if j == target_idx
             else (np.zeros_like(cA), np.zeros_like(cD)))
            for j, (cA, cD) in enumerate(coeffs)
        ]
        sub = pywt.iswt(zeroed, wavelet)[:SEGMENT_SAMPLES].astype(np.float32)
        subbands[f'D{level}'] = _minmax(sub)

    return subbands


def denoise_and_extract_subbands(
        signal_padded: np.ndarray,
        wavelet: str = WAVELET,
        n_levels: int = N_LEVELS_SWT,
        subband_levels: List[int] = SUBBAND_LEVELS,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Single SWT call: denoise signal and reconstruct per-level subbands.

    Steps:
      1. SWT decomposition
      2. Soft-threshold all detail coefficients (Donoho & Johnstone 1994)
         threshold T = MAD/0.6745 * sqrt(2 * log(N))
         where N = log2(fs / f_low) = log2(500/0.5) ≈ 10
      3. Reconstruct full denoised signal from thresholded coefficients
      4. Reconstruct each subband by zeroing all levels except target
         (uses already-thresholded coefficients — no second denoising pass)

    Returns:
        denoised:  (SEGMENT_SAMPLES,) float32
        subbands:  {'D2': ..., 'D3': ..., ...} each (SEGMENT_SAMPLES,) float32,
                   min-max normalised
    """
    import math
    coeffs = pywt.swt(signal_padded, wavelet, level=n_levels)

    N_log = math.log2(FS_TARGET / 0.5)          # Donoho level count
    denoised_coeffs = []
    for cA, cD in coeffs:
        sigma = np.median(np.abs(cD)) / 0.6745
        T     = sigma * math.sqrt(2.0 * math.log(N_log))
        cD_d  = np.sign(cD) * np.maximum(np.abs(cD) - T, 0.0)
        denoised_coeffs.append((cA, cD_d))

    denoised = pywt.iswt(denoised_coeffs, wavelet)[:SEGMENT_SAMPLES].astype(np.float32)

    n_lvls = len(denoised_coeffs)
    subbands: Dict[str, np.ndarray] = {}

    for level in subband_levels:
        target_idx = n_lvls - level
        zeroed = [
            ((np.zeros_like(cA), cD) if j == target_idx
             else (np.zeros_like(cA), np.zeros_like(cD)))
            for j, (cA, cD) in enumerate(denoised_coeffs)
        ]
        sub = pywt.iswt(zeroed, wavelet)[:SEGMENT_SAMPLES].astype(np.float32)
        subbands[f'D{level}'] = _minmax(sub)

    return denoised, subbands

def compute_utrp(signal: np.ndarray, tau: int, m: int,
                 size: int = UTRP_SIZE) -> Optional[np.ndarray]:
    try:
        sig = signal[:SEGMENT_SAMPLES].astype(np.float32)
        n   = len(sig) - (m - 1) * tau
        if n <= 0:
            return np.zeros((3, size, size), dtype=np.float32)

        gen = RecurrencePlotGenerator(
            signal=sig, time_delay=tau, embedding_dim=m, epsilon=0
        )
        rp = gen.compute_rp_unthresholded()

        d_max = rp.max()
        if d_max > 1e-9:
            rp = rp / d_max

        rp_r = cv2.resize(rp.astype(np.float32), (size, size),
                          interpolation=cv2.INTER_LINEAR)
        np.fill_diagonal(rp_r, 0.0)

        u8  = (rp_r * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb.transpose(2, 0, 1)

    except Exception as e:
        print(f"    [ERROR] compute_utrp: {e}")
        return None


def build_subject_groups(nsr_records: List[str],
                          preaf_records: List[str]) -> List[Tuple]:
    """
    Group paired records by patient before splitting.

    NSR:   (n01, n01c) → one subject group
    PreAF: (p01, p02)  → one subject group (paired odd/even)
    """
    nsr_set   = set(nsr_records)
    preaf_set = set(preaf_records)
    subjects  = []

    for i in range(1, 51):
        recs = [r for r in [f'n{i:02d}', f'n{i:02d}c'] if r in nsr_set]
        if recs:
            subjects.append((f'nsr_{i:02d}', recs, 0))

    for i in range(1, 50, 2):
        recs = [r for r in [f'p{i:02d}', f'p{i+1:02d}'] if r in preaf_set]
        if recs:
            subjects.append((f'paf_{i:02d}', recs, 1))

    return subjects


def split_subjects_721(subjects: List[Tuple],
                        seed: int = 42) -> Dict[str, List[str]]:
    rng     = random.Random(seed)
    preaf_s = [s for s in subjects if s[2] == 1]
    nsr_s   = [s for s in subjects if s[2] == 0]
    rng.shuffle(preaf_s); rng.shuffle(nsr_s)

    def _split(lst):
        n      = len(lst)
        n_test = max(1, round(n * 0.1))
        n_val  = max(1, round(n * 0.2))
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
        'val':   _extract(pa_va + ns_va),
        'test':  _extract(pa_te + ns_te),
    }

class DatasetGenerator:
    KEYS = ['single'] + [f'D{l}' for l in SUBBAND_LEVELS]

    def __init__(self, output_dir: Path = OUTPUT_DIR,
                 wavelet: str = WAVELET):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wavelet    = wavelet

    def _load_signal(self, record_id: str) -> Optional[np.ndarray]:
        try:
            rec = wfdb.rdrecord(str(DATA_DIR / record_id))
            raw = rec.p_signal[:, 0].astype(np.float64)
            n   = int(len(raw) * FS_TARGET / FS_ORIG)
            return sp_signal.resample(raw, n).astype(np.float32)
        except Exception as e:
            print(f"  [ERROR] {record_id}: {e}")
            return None

    def _extract_segments(self, signal: np.ndarray) -> List[np.ndarray]:
        max_start = len(signal) - SEGMENT_SAMPLES
        if max_start <= 0:
            return []
        starts = np.linspace(0, max_start, N_SEGMENTS, dtype=int)
        segs   = []
        for s in starts:
            seg = signal[s:s + SEGMENT_SAMPLES]
            if len(seg) < SEGMENT_SAMPLES:
                continue
            segs.append(_minmax(seg.astype(np.float32)))
        return segs

    def _get_time_to_onset(self, record_id: str, segment_idx: int) -> float:
        if not record_id.startswith('p'):
            return -1.0

        import re
        match = re.match(r'p(\d+)', record_id)
        if not match:
            return -1.0

        num = int(match.group(1))

        if num % 2 == 0:
            return 45

        # p-odd: 30-min pre-AF recording ending at onset
        max_start = FS_TARGET * 30 * 60 - SEGMENT_SAMPLES
        starts = np.linspace(0, max_start, N_SEGMENTS, dtype=int)
        seg_start_sec = starts[segment_idx] / FS_TARGET
        return float(30 * 60 - seg_start_sec)

    def _process_record(self, record_id: str, label: int,tau_single: int, m_single: int,subband_tau: Dict, subband_m: Dict) -> List[Dict]:
        signal = self._load_signal(record_id)
        if signal is None:
            return []

        segments = self._extract_segments(signal)
        if not segments:
            print(f"  [WARN] No segments: {record_id}")
            return []

        results = []
        for seg_idx, seg in enumerate(segments):
            seg_pad = _pad(seg)

            # Multi-scale: extract subbands WITHOUT denoising
            subbands_multi = extract_subbands_no_denoise(
                seg_pad, self.wavelet, N_LEVELS_SWT, SUBBAND_LEVELS
            )
            
            # Single-scale: denoise and get full reconstruction
            # seg_den, _ = denoise_and_extract_subbands(
            #     seg_pad, self.wavelet, N_LEVELS_SWT, SUBBAND_LEVELS
            # )

            images: Dict[str, np.ndarray] = {}
            min_max_seg = _minmax(seg)

            # Single-scale UTRP from denoised signal
            # single_utrp = compute_utrp(seg_den, tau_single, m_single)
            single_utrp = compute_utrp(min_max_seg, tau_single, m_single)

            if single_utrp is None:
                continue  # Skip this segment if UTRP computation failed
            images['single'] = single_utrp

            # Multi-scale UTRPs (D2-D8) from undenoised subbands
            skip_segment = False
            for level in SUBBAND_LEVELS:
                name = f'D{level}'
                tau  = subband_tau.get(name, 3)
                m    = subband_m.get(name, 3)
                sub_utrp = compute_utrp(subbands_multi[name], tau, m)
                if sub_utrp is None:
                    skip_segment = True
                    break
                images[name] = sub_utrp
            
            if skip_segment:
                continue  # Skip segment if any subband UTRP failed

            results.append({
                'record_id':   record_id,
                'segment_idx': seg_idx,
                'label':       label,
                'time_to_onset_min': self._get_time_to_onset(record_id, seg_idx),
                'images':      images,
            })

        print(f"    {record_id}  label={label}  {len(results)} segments")
        return results

    def _save_hdf5(self, samples: List[Dict], path: Path) -> None:
        if not samples:
            print(f"  ✗ No samples to save!")
            return
            
        N = len(samples)
        print(f"  Saving {N} samples → {path}")

        labels  = np.array([s['label']       for s in samples], dtype=np.int8)
        rids    = np.array([s['record_id'].encode() for s in samples], dtype='S16')
        sidxs   = np.array([s['segment_idx'] for s in samples], dtype=np.int16)
        t2os = np.array([s['time_to_onset_min'] for s in samples], dtype=np.float32)

        with h5py.File(path, 'w') as f:
            # One dataset per image type
            for key in self.KEYS:
                # Filter samples that have this key
                data = np.stack([s['images'][key] for s in samples])
                f.create_dataset(key, data=data, dtype=np.float32,
                                 compression='gzip', compression_opts=4)

            meta = f.create_group('metadata')
            meta.create_dataset('labels',          data=labels)
            meta.create_dataset('record_ids',      data=rids)
            meta.create_dataset('segment_indices', data=sidxs)
            meta.create_dataset('time_to_onset_min', data=t2os)

            f.attrs.update({
                'n_samples':       N,
                'n_preaf':         int((labels == 1).sum()),
                'n_nsr':           int((labels == 0).sum()),
                'utrp_size':       UTRP_SIZE,
                'wavelet':         self.wavelet,
                'subband_levels':  ','.join(f'D{l}' for l in SUBBAND_LEVELS),
                'fs_target':       FS_TARGET,
                'segment_samples': SEGMENT_SAMPLES,
                'epsilon_frac':    EPSILON_FRAC,
                'colourmap':       'JET',
                'single_preprocessing': 'raw_minmax_no_denoise',
            })

        n_preaf = int((labels == 1).sum())
        n_nsr   = int((labels == 0).sum())
        print(f"  Saved — NSR={n_nsr}  Pre-AF={n_preaf}")

    def generate(self, params_file: Path, seed: int = 42) -> None:
        print('=' * 70)
        print('GENERATING UTRP DATASET')
        print(f'  Wavelet:  {self.wavelet}')
        print(f'  Subbands: D{min(SUBBAND_LEVELS)}–D{max(SUBBAND_LEVELS)}')
        print(f'  Output:   {self.output_dir}')
        print('=' * 70)

        # Load tau/m
        with open(params_file) as f:
            params = json.load(f)
        tau_single   = params['tau_single']
        m_single     = params['m_single']
        subband_tau  = params['subband_tau']
        subband_m    = params['subband_m']
        print(f"\nEmbedding params: τ_single={tau_single}  m_single={m_single}")

        # Build records lists
        nsr_records   = ([f'n{i:02d}'  for i in range(1, 51)] +
                         [f'n{i:02d}c' for i in range(1, 51)])
        preaf_records = [f'p{i:02d}' for i in range(1, 51)]
        nsr_set       = set(nsr_records)

        # Patient-level split (pairs grouped)
        subjects = build_subject_groups(nsr_records, preaf_records)
        split    = split_subjects_721(subjects, seed=seed)

        print('\nSplit summary:')
        for name in ('train', 'val', 'test'):
            recs  = split[name]
            n_nsr = sum(1 for r in recs if r in nsr_set)
            n_paf = sum(1 for r in recs if r not in nsr_set)
            print(f'  {name:5s}: {len(recs):3d} records '
                  f'(NSR={n_nsr:2d} PreAF={n_paf:2d}) '
                  f'→ ~{len(recs)*N_SEGMENTS} segments')

        # Save split JSON
        split_out = {**split, 'seed': seed, 'wavelet': self.wavelet}
        split_path = self.output_dir / 'split.json'
        with open(split_path, 'w') as f:
            json.dump(split_out, f, indent=2)
        print(f'\nSplit saved → {split_path}')

        # Process and save each split
        for split_name in ('train', 'val', 'test'):
            print(f'\n── {split_name.upper()} ──')
            samples = []
            for record_id in split[split_name]:
                label = 0 if record_id in nsr_set else 1
                samples.extend(self._process_record(
                    record_id, label,
                    tau_single, m_single,
                    subband_tau, subband_m
                ))
            self._save_hdf5(samples, self.output_dir / f'{split_name}.h5')

        print(f'\n✓ Done → {self.output_dir}')


if __name__ == '__main__':
    mode = 'generate'
    params_file = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_new/attention/tau_m_params.json')
    output_dir = OUTPUT_DIR
    seed = 42
    
    split = 'train'
    n_samples = 6
    
    if mode == 'generate':
        gen = DatasetGenerator(output_dir=output_dir, wavelet=WAVELET)
        gen.generate(params_file=params_file, seed=seed)

    else:  # visualise
        vis = UTRPVisualiser(dataset_dir=output_dir)
        vis.class_distribution()
        vis.show_grid(
            split=split,
            n_samples=n_samples,
            seed=seed,
            save_path=output_dir / f'sample_grid_{split}.png',
        )