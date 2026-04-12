"""
Simplified UTRP dataset generator.

Dataset structure:
- Class 1 (Pre-AF): 25 p-odd records (p01, p03, ..., p49) with label=1
- Class 0 (NSR): 50 n-records (n01-n50) with label=0
- Total: 75 patients × 30 segments = 2,250 segments

Segmentation:
- Each 30-min record → 6 non-overlapping 5-min windows
- Each window → 5 non-overlapping 10-sec segments
- Total: 6 × 5 = 30 segments per patient
"""

import numpy as np
import h5py
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wfdb

# Import from parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from extractors.paf_extractor import DATA_DIR, SAMPLING_FREQ, LEAD_INDEX
from data_cleaning.data_cleaner import DataCleaner
from wavelets.stationary_wavelet_transform import SWT
from recurrence_plot.AMI import AMI
from recurrence_plot.FNN import FNN

# Constants
OUTPUT_SIZE = 128
WAVELET = 'coif2'
SWT_LEVEL = 4
SEGMENT_DURATION = 10  # seconds
WINDOW_DURATION = 5  # minutes
RECORD_DURATION = 30  # minutes
SAMPLING_FREQ = 128  # Hz

# Derived constants
SEGMENT_SIZE = SEGMENT_DURATION * SAMPLING_FREQ  # 1,280 samples
WINDOW_SIZE = WINDOW_DURATION * 60 * SAMPLING_FREQ  # 38,400 samples
RECORD_SIZE = RECORD_DURATION * 60 * SAMPLING_FREQ  # 230,400 samples
N_WINDOWS = 6
N_SEGMENTS_PER_WINDOW = 5
N_SEGMENTS_PER_PATIENT = N_WINDOWS * N_SEGMENTS_PER_WINDOW  # 30

# Output
OUTPUT_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_new/exp2')
OUTPUT_PATH = OUTPUT_DIR / 'utrp_dataset_simple.h5'
SPLIT_PATH = OUTPUT_DIR / 'split.json'


def normalize_utrp(distance_matrix: np.ndarray) -> np.ndarray:
    """Normalize UTRP by 95th percentile of upper triangle."""
    upper_tri_vals = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    p95 = np.percentile(upper_tri_vals, 95)
    if p95 < 1e-9:
        return distance_matrix
    normalized = distance_matrix / p95
    return np.clip(normalized, 0, 1)


def estimate_embedding_params(signal: np.ndarray, max_lag: int = 50, max_dim: int = 15) -> Tuple[int, int]:
    """Estimate time delay (tau) using AMI and embedding dimension (m) using FNN."""
    ami_estimator = AMI(signal, max_lag=max_lag)
    tau = ami_estimator.get_first_minimum()

    fnn_estimator = FNN(max_dim=max_dim, tau=tau, signal=signal)
    m = fnn_estimator.get_optimal_dimension()

    return tau, m


def embed_signal(signal: np.ndarray, tau: int, m: int) -> np.ndarray:
    """Time-delay embedding."""
    n = len(signal)
    embedded_length = n - (m - 1) * tau

    if embedded_length <= 0:
        raise ValueError(f"Signal too short for embedding: len={n}, tau={tau}, m={m}")

    embedded = np.zeros((embedded_length, m))
    for i in range(m):
        embedded[:, i] = signal[i * tau : i * tau + embedded_length]

    return embedded


def create_utrp(signal: np.ndarray, tau: int, m: int) -> np.ndarray:
    """
    Create Unthresholded Recurrence Plot.
    Returns normalized distance matrix resized to OUTPUT_SIZE × OUTPUT_SIZE.
    """
    embedded = embed_signal(signal, tau, m)
    distance_matrix = squareform(pdist(embedded, metric='euclidean'))
    normalized = normalize_utrp(distance_matrix)
    resized = cv2.resize(normalized, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


def preprocess_segment(signal: np.ndarray) -> np.ndarray:
    """Preprocess: demean → highpass 0.5Hz → z-score."""
    cleaner = DataCleaner()

    signal = cleaner.preprocess(signal, check_quality=False, max_rr_cv=0.5)

    return signal


def extract_segments_from_record(signal: np.ndarray, record_id: str) -> List[Tuple[np.ndarray, int, int, float]]:
    """
    Extract 30 segments from a 30-minute record.

    Returns:
        List of (segment_signal, window_idx, segment_idx, time_to_onset)
    """
    if len(signal) < RECORD_SIZE:
        raise ValueError(f"Record {record_id} too short: {len(signal)} < {RECORD_SIZE}")

    # Determine if Pre-AF (p-odd)
    is_preaf = record_id.startswith('p') and int(record_id[1:]) % 2 == 1

    segments = []
    for window_idx in range(N_WINDOWS):
        window_start = window_idx * WINDOW_SIZE

        for segment_idx in range(N_SEGMENTS_PER_WINDOW):
            segment_start = window_start + segment_idx * SEGMENT_SIZE
            segment_end = segment_start + SEGMENT_SIZE

            segment_signal = signal[segment_start:segment_end]

            # Compute time to onset (minutes before end of record)
            if is_preaf:
                samples_to_end = RECORD_SIZE - segment_start
                time_to_onset = samples_to_end / SAMPLING_FREQ / 60  # convert to minutes
            else:
                time_to_onset = np.nan

            segments.append((segment_signal, window_idx, segment_idx, time_to_onset))

    return segments


@dataclass
class UTRPSample:
    """Container for a processed UTRP sample."""
    utrp_single: np.ndarray  # (128, 128)
    utrp_multi: np.ndarray   # (5, 128, 128)
    label: int
    patient_id: str
    window_idx: int
    segment_idx: int
    time_to_onset: float
    tau_single: int
    m_single: int
    tau_multi: np.ndarray  # (5,) for each subband
    m_multi: np.ndarray    # (5,)


def process_segment(args) -> UTRPSample:
    """Process a single segment to create single-scale and multi-scale UTRPs."""
    segment_signal, window_idx, segment_idx, time_to_onset, label, patient_id = args

    # Preprocess
    preprocessed = preprocess_segment(segment_signal)

    # Single-scale UTRP
    tau_single, m_single = estimate_embedding_params(preprocessed)
    utrp_single = create_utrp(preprocessed, tau_single, m_single)

    # Multi-scale UTRP (SWT decomposition)
    swt = SWT(wavelet=WAVELET, level=SWT_LEVEL)
    coeffs = swt.decompose(preprocessed, level=SWT_LEVEL)

    # Reconstruct subbands by zeroing out other coefficients
    subbands = {}

    # Detail subbands: D4, D3, D2, D1 (from coeffs[0] to coeffs[3])
    for i in range(SWT_LEVEL):
        label = f"D{SWT_LEVEL - i}"
        zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
        zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
        raw_subband = swt.reconstruct(zeroed)
        std = np.std(raw_subband)
        if std > 1e-9:
            normalized = (raw_subband - np.mean(raw_subband)) / std
        else:
            normalized = raw_subband
        subbands[label] = normalized

    # Approximation subband: A4
    zeroed_a = [(c[0], np.zeros_like(c[1])) for c in coeffs]
    raw_subband_a = swt.reconstruct(zeroed_a)
    std_a = np.std(raw_subband_a)
    if std_a > 1e-9:
        normalized_a = (raw_subband_a - np.mean(raw_subband_a)) / std_a
    else:
        normalized_a = raw_subband_a
    subbands['A4'] = normalized_a

    # Create UTRPs for each subband
    utrp_multi = np.zeros((5, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
    tau_multi = np.zeros(5, dtype=np.int8)
    m_multi = np.zeros(5, dtype=np.int8)

    for i, subband_name in enumerate(['D1', 'D2', 'D3', 'D4', 'A4']):
        subband = subbands[subband_name]

        # Estimate params and create UTRP
        tau, m = estimate_embedding_params(subband)
        utrp_multi[i] = create_utrp(subband, tau, m)
        tau_multi[i] = tau
        m_multi[i] = m

    return UTRPSample(
        utrp_single=utrp_single,
        utrp_multi=utrp_multi,
        label=label,
        patient_id=patient_id,
        window_idx=window_idx,
        segment_idx=segment_idx,
        time_to_onset=time_to_onset,
        tau_single=tau_single,
        m_single=m_single,
        tau_multi=tau_multi,
        m_multi=m_multi
    )


def create_patient_split(n_test_preaf: int = 3, n_test_nsr: int = 5, seed: int = 42) -> Dict:
    """
    Create patient-level train/val/test split.

    Args:
        n_test_preaf: Number of Pre-AF patients for test set
        n_test_nsr: Number of NSR patients for test set
        seed: Random seed

    Returns:
        Dict with 'train', 'val', 'test' lists of patient IDs
    """
    random.seed(seed)
    np.random.seed(seed)

    # All patient IDs
    preaf_patients = [f'p{i:02d}' for i in range(1, 50, 2)]  # p01, p03, ..., p49
    nsr_patients = [f'n{i:02d}' for i in range(1, 51)]  # n01-n50

    # Shuffle
    random.shuffle(preaf_patients)
    random.shuffle(nsr_patients)

    # Split
    test_preaf = preaf_patients[:n_test_preaf]
    test_nsr = nsr_patients[:n_test_nsr]

    remaining_preaf = preaf_patients[n_test_preaf:]
    remaining_nsr = nsr_patients[n_test_nsr:]

    # Combine train+val (we'll do CV on this)
    train_val = remaining_preaf + remaining_nsr
    random.shuffle(train_val)

    split = {
        'test': sorted(test_preaf + test_nsr),
        'train_val': sorted(train_val),
        'seed': seed
    }

    print(f"\nDataset split:")
    print(f"  Test: {len(split['test'])} patients ({n_test_preaf} Pre-AF + {n_test_nsr} NSR)")
    print(f"  Train+Val: {len(split['train_val'])} patients")
    print(f"  Total segments: test={len(split['test'])*30}, train_val={len(split['train_val'])*30}")

    return split


def main():
    """Generate the UTRP dataset."""
    print("="*80)
    print("UTRP Dataset Generator (Simplified)")
    print("="*80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create split
    split = create_patient_split(n_test_preaf=3, n_test_nsr=5, seed=42)

    # Save split
    with open(SPLIT_PATH, 'w') as f:
        json.dump(split, f, indent=2)
    print(f"\nSaved split to: {SPLIT_PATH}")

    # Collect all patient IDs with labels
    all_patients = []

    # Pre-AF patients (p-odd)
    for i in range(1, 50, 2):
        patient_id = f'p{i:02d}'
        all_patients.append((patient_id, 1))

    # NSR patients (n-records)
    for i in range(1, 51):
        patient_id = f'n{i:02d}'
        all_patients.append((patient_id, 0))

    print(f"\nTotal patients: {len(all_patients)}")
    print(f"  Pre-AF: {sum(1 for _, label in all_patients if label == 1)}")
    print(f"  NSR: {sum(1 for _, label in all_patients if label == 0)}")
    print(f"  Expected segments: {len(all_patients) * N_SEGMENTS_PER_PATIENT}")

    # Extract and process segments
    print("\nExtracting segments from records...")
    all_tasks = []

    for patient_id, label in tqdm(all_patients, desc="Loading records"):
        # Load record using wfdb
        try:
            record = wfdb.rdrecord(str(DATA_DIR / patient_id))
            if record.fs != SAMPLING_FREQ:
                print(f"Warning: {patient_id} has unexpected fs={record.fs}, skipping")
                continue

            record_data = record.p_signal[:, LEAD_INDEX].astype(np.float32)
        except Exception as e:
            print(f"Warning: Could not load {patient_id}: {e}")
            continue

        # Extract 30 segments
        try:
            segments = extract_segments_from_record(record_data, patient_id)

            # Create tasks for parallel processing
            for segment_signal, window_idx, segment_idx, time_to_onset in segments:
                all_tasks.append((
                    segment_signal, window_idx, segment_idx, time_to_onset, label, patient_id
                ))
        except Exception as e:
            print(f"Error extracting segments from {patient_id}: {e}")
            continue

    print(f"\nProcessing {len(all_tasks)} segments...")

    # Process in parallel
    samples = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_segment, task) for task in all_tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Creating UTRPs"):
            try:
                sample = future.result()
                samples.append(sample)
            except Exception as e:
                print(f"Error processing segment: {e}")
                continue

    print(f"\nSuccessfully processed {len(samples)} segments")

    # Create HDF5 file
    print(f"\nCreating HDF5 file: {OUTPUT_PATH}")

    n_samples = len(samples)

    with h5py.File(OUTPUT_PATH, 'w') as f:
        # Create datasets
        single_scale = f.create_dataset(
            'single_scale/utrps',
            shape=(n_samples, OUTPUT_SIZE, OUTPUT_SIZE),
            dtype=np.float32,
            compression='gzip'
        )

        multi_scale = f.create_dataset(
            'multi_scale/utrps',
            shape=(n_samples, 5, OUTPUT_SIZE, OUTPUT_SIZE),
            dtype=np.float32,
            compression='gzip'
        )

        labels = f.create_dataset('metadata/labels', shape=(n_samples,), dtype=np.int8)
        patient_ids = f.create_dataset('metadata/patient_ids', shape=(n_samples,), dtype=h5py.string_dtype())
        window_idx = f.create_dataset('metadata/window_idx', shape=(n_samples,), dtype=np.int8)
        segment_idx = f.create_dataset('metadata/segment_idx', shape=(n_samples,), dtype=np.int8)
        time_to_onset = f.create_dataset('metadata/time_to_onset', shape=(n_samples,), dtype=np.float32)

        embedding_tau = f.create_dataset('metadata/embedding_tau', shape=(n_samples, 6), dtype=np.int8)
        embedding_m = f.create_dataset('metadata/embedding_m', shape=(n_samples, 6), dtype=np.int8)

        # Fill datasets
        print("Writing data to HDF5...")
        for i, sample in enumerate(tqdm(samples, desc="Saving")):
            single_scale[i] = sample.utrp_single
            multi_scale[i] = sample.utrp_multi
            labels[i] = sample.label
            patient_ids[i] = sample.patient_id
            window_idx[i] = sample.window_idx
            segment_idx[i] = sample.segment_idx
            time_to_onset[i] = sample.time_to_onset

            # Store embedding params: [single, D1, D2, D3, D4, A4]
            embedding_tau[i, 0] = sample.tau_single
            embedding_tau[i, 1:] = sample.tau_multi

            embedding_m[i, 0] = sample.m_single
            embedding_m[i, 1:] = sample.m_multi

        # Store metadata
        f.attrs['n_samples'] = n_samples
        f.attrs['output_size'] = OUTPUT_SIZE
        f.attrs['sampling_freq'] = SAMPLING_FREQ
        f.attrs['segment_duration'] = SEGMENT_DURATION
        f.attrs['n_segments_per_patient'] = N_SEGMENTS_PER_PATIENT

    print(f"\n{'='*80}")
    print("Dataset generation complete!")
    print(f"{'='*80}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Split: {SPLIT_PATH}")
    print(f"Total samples: {n_samples}")
    print(f"Label distribution:")
    print(f"  Pre-AF: {sum(s.label for s in samples)}")
    print(f"  NSR: {n_samples - sum(s.label for s in samples)}")


if __name__ == '__main__':
    main()
