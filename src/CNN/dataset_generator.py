"""
UTRP Dataset Generator — simplified, p-even excluded.

Class composition:
  - Pre-AF (label=1): p01, p03, p05, ..., p49  (25 patients)
  - NSR    (label=0): n01, n02, ..., n50        (50 patients)

Total: 75 patients × 30 segments = 2,250 segments
Class imbalance: 2:1 (NSR:Pre-AF) — handled by pos_weight in loss.

Segments per patient:
  6 non-overlapping 5-min windows × 5 non-overlapping 10-sec segments = 30
"""

import re
import json
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import h5py
import numpy as np
import wfdb
from tqdm import tqdm

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator
from src.extractors.paf_extractor import PAFExtractor, PAFSegment, DATA_DIR, SAMPLING_FREQ
from src.data_cleaning.data_cleaner import DataCleaner
from src.wavelets.stationary_wavelet_transform import SWT

# ── Constants ────────────────────────────────────────────────────────────────
OUTPUT_SIZE        = 128          # UTRP pixel size (128×128 is sufficient, saves memory)
WAVELET            = 'db4'        # Fixed from Experiment 2
SWT_LEVEL          = 4
SEGMENT_SAMPLES    = 1280         # 10 s × 128 Hz
WINDOW_SAMPLES     = 5 * 60 * SAMPLING_FREQ   # 5-min window = 38,400 samples
N_WINDOWS          = 6
N_SEG_PER_WINDOW   = 5
N_SEGMENTS         = N_WINDOWS * N_SEG_PER_WINDOW   # 30 per patient

SUBBAND_NAMES      = ['D1', 'D2', 'D3', 'D4', 'A4']

# Fixed embedding params (τ=3, m=3 validated in Experiment 1)
TAU, M = 3, 3

# ── Normalisation ─────────────────────────────────────────────────────────────
def normalize_utrp(distance_matrix: np.ndarray) -> np.ndarray:
    """Normalise by 95th percentile of upper-triangle, clip to [0,1]."""
    idx = np.triu_indices_from(distance_matrix, k=1)
    p95 = np.percentile(distance_matrix[idx], 95)
    if p95 < 1e-9:
        return np.zeros_like(distance_matrix)
    return np.clip(distance_matrix / p95, 0.0, 1.0)


# ── UTRP creation ─────────────────────────────────────────────────────────────
def create_utrp(signal: np.ndarray, output_size: int = OUTPUT_SIZE) -> np.ndarray:
    gen = RecurrencePlotGenerator(signal, TAU, M, 0, unthresholded=True)
    dist = gen.compute_rp_unthresholded()
    utrp = normalize_utrp(dist)
    return cv2.resize(utrp, (output_size, output_size),
                      interpolation=cv2.INTER_LINEAR).astype(np.float32)


# ── Dataclass ─────────────────────────────────────────────────────────────────
@dataclass
class UTRPSample:
    single_scale_utrp: np.ndarray   # (H, W)  float32
    multi_scale_utrp:  np.ndarray   # (5, H, W) float32
    label:      int
    patient_id: str
    record_id:  str
    segment_idx: int


# ── Top-level worker (must be picklable) ──────────────────────────────────────
def _process_segment(args) -> Optional[UTRPSample]:
    """
    Worker function.  Receives a tuple so it can be used with
    ProcessPoolExecutor.map without lambda (not picklable on macOS).
    """
    record_id, label, seg_idx, wavelet, swt_level = args
    try:
        record = wfdb.rdrecord(str(DATA_DIR / record_id))
    except Exception as e:
        print(f"[WARN] Cannot load {record_id}: {e}")
        return None

    raw = record.p_signal[:, 0].astype(np.float32)
    swt = SWT(wavelet=wavelet)

    # --- pick segment window ---
    window_idx  = seg_idx // N_SEG_PER_WINDOW
    local_idx   = seg_idx %  N_SEG_PER_WINDOW
    start = window_idx * WINDOW_SAMPLES + local_idx * SEGMENT_SAMPLES
    end   = start + SEGMENT_SAMPLES

    if end > len(raw):
        return None

    signal = raw[start:end]
    signal = (signal - signal.mean()) / (signal.std() + 1e-9)

    try:
        # Single-scale UTRP
        ss_utrp = create_utrp(signal)

        # Multi-scale: decompose → reconstruct each subband → UTRP
        coeffs = swt.decompose(signal, level=swt_level)
        ms_channels = []

        for lvl in range(swt_level):          # detail levels D1…D4
            zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            zeroed[lvl] = (np.zeros_like(coeffs[lvl][0]), coeffs[lvl][1])
            sub = swt.reconstruct(zeroed)
            sub = (sub - sub.mean()) / (sub.std() + 1e-9)
            ms_channels.append(create_utrp(sub))

        # Approximation A4
        zeroed_a = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        sub_a = swt.reconstruct(zeroed_a)
        sub_a = (sub_a - sub_a.mean()) / (sub_a.std() + 1e-9)
        ms_channels.append(create_utrp(sub_a))

        ms_utrp = np.stack(ms_channels, axis=0)   # (5, H, W)

        return UTRPSample(
            single_scale_utrp=ss_utrp,
            multi_scale_utrp=ms_utrp,
            label=label,
            patient_id=record_id,
            record_id=record_id,
            segment_idx=seg_idx,
        )
    except Exception as e:
        print(f"[WARN] Processing failed {record_id} seg {seg_idx}: {e}")
        return None


# ── Dataset Generator ──────────────────────────────────────────────────────────
class UTRPDatasetGenerator:
    """
    Generates balanced UTRP dataset:
      Pre-AF: p01, p03, ..., p49  (25 patients × 30 segs = 750)
      NSR:    n01, ..., n50       (50 patients × 30 segs = 1500)
      Total:  2250 segments
    """

    def __init__(self, wavelet: str = WAVELET, swt_level: int = SWT_LEVEL):
        self.wavelet   = wavelet
        self.swt_level = swt_level

    def generate(self, output_path: Path,
                 n_test_per_class: int = 5,
                 seed: int = 42) -> None:
        """
        Full pipeline: extract → UTRP → HDF5 + split JSON.

        Args:
            output_path:      Where to save the HDF5 file.
            n_test_per_class: Patients per class held out for test.
            seed:             RNG seed for reproducible splits.
        """
        rng = random.Random(seed)

        preaf_patients = [f'p{i:02d}' for i in range(1, 50, 2)]   # p01…p49
        nsr_patients   = [f'n{i:02d}' for i in range(1,  51)]      # n01…n50

        rng.shuffle(preaf_patients)
        rng.shuffle(nsr_patients)

        # ── Patient-level split ───────────────────────────────────────────────
        test_patients      = (preaf_patients[:n_test_per_class] +
                              nsr_patients  [:n_test_per_class])
        train_val_patients = (preaf_patients[n_test_per_class:] +
                              nsr_patients  [n_test_per_class:])

        split = {
            'test':      sorted(test_patients),
            'train_val': sorted(train_val_patients),
            'seed': seed,
            'n_test_per_class': n_test_per_class,
            'note': 'p-odd=Pre-AF(1), n*=NSR(0). p-even excluded.'
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        split_path = output_path.with_suffix('.split.json')
        with open(split_path, 'w') as f:
            json.dump(split, f, indent=2)
        print(f"Split saved → {split_path}")
        print(f"  Test:      {len(test_patients)} patients "
              f"({n_test_per_class} Pre-AF, {n_test_per_class} NSR)")
        print(f"  Train+Val: {len(train_val_patients)} patients")

        # ── Build task list ───────────────────────────────────────────────────
        all_patients = (
            [(pid, 1) for pid in preaf_patients] +
            [(pid, 0) for pid in nsr_patients]
        )

        tasks = [
            (record_id, label, seg_idx, self.wavelet, self.swt_level)
            for record_id, label in all_patients
            for seg_idx in range(N_SEGMENTS)
        ]
        print(f"\nTotal tasks: {len(tasks)} "
              f"({len(preaf_patients)} Pre-AF + {len(nsr_patients)} NSR) × {N_SEGMENTS} segs")

        # ── Parallel processing ───────────────────────────────────────────────
        n_workers = min(multiprocessing.cpu_count(), 8)
        print(f"Processing with {n_workers} workers …")

        samples: List[UTRPSample] = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_process_segment, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(tasks),
                            desc='UTRP generation'):
                s = fut.result()
                if s is not None:
                    samples.append(s)

        print(f"Processed {len(samples)}/{len(tasks)} segments successfully")

        # ── Save HDF5 ─────────────────────────────────────────────────────────
        self._save_hdf5(samples, output_path, split)

    # ── Internal helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _save_hdf5(samples: List[UTRPSample], path: Path, split: dict) -> None:
        print(f"\nSaving HDF5 → {path}")
        labels = np.array([s.label for s in samples], dtype=np.int8)

        with h5py.File(path, 'w') as f:
            ss = f.create_group('single_scale')
            ss.create_dataset(
                'utrps',
                data=np.stack([s.single_scale_utrp for s in samples]),
                dtype=np.float32, compression='gzip', compression_opts=4
            )

            ms = f.create_group('multi_scale')
            ms.create_dataset(
                'utrps',
                data=np.stack([s.multi_scale_utrp for s in samples]),
                dtype=np.float32, compression='gzip', compression_opts=4
            )

            meta = f.create_group('metadata')
            meta.create_dataset('labels',      data=labels)
            meta.create_dataset('patient_ids', data=np.array(
                [s.patient_id.encode() for s in samples], dtype='S16'))
            meta.create_dataset('record_ids',  data=np.array(
                [s.record_id.encode() for s in samples], dtype='S16'))
            meta.create_dataset('segment_idx', data=np.array(
                [s.segment_idx for s in samples], dtype=np.int16))

            f.attrs.update({
                'n_samples':     len(samples),
                'n_preaf':       int((labels == 1).sum()),
                'n_nsr':         int((labels == 0).sum()),
                'output_size':   OUTPUT_SIZE,
                'tau':           TAU,
                'm':             M,
                'subband_names': ','.join(SUBBAND_NAMES),
                'split_json':    str(path.with_suffix('.split.json')),
            })

        print(f"Saved {len(samples)} samples  "
              f"(Pre-AF={int((labels==1).sum())}, NSR={int((labels==0).sum())})")


if __name__ == '__main__':
    OUTPUT_PATH = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_new/exp3/'
                       'utrp_dataset_128.h5')
    gen = UTRPDatasetGenerator(wavelet=WAVELET, swt_level=SWT_LEVEL)
    gen.generate(output_path=OUTPUT_PATH, n_test_per_class=5, seed=42)