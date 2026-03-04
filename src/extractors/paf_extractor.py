"""
PAF (Paroxysmal Atrial Fibrillation) Dataset Extractor

Dataset structure (PhysioNet PAF Prediction Challenge):
  n* records : normal subjects (no AF)
  p* records : PAF subjects in pairs
    odd  (p01, p03, ...): normal rhythm, >45 min from any AF  -> label=0
    even (p02, p04, ...): immediately precede PAF onset        -> label=1

SWT coefficient ordering (verified by PSD analysis):
  coeffs[0] -> 4-8 Hz  -> D4
  coeffs[1] -> 8-16 Hz -> D3
  coeffs[2] -> 16-32 Hz -> D2
  coeffs[3] -> 32-64 Hz -> D1
  A4        -> 0-4 Hz

Quality check thresholds:
  Normal segments : max_rr_cv=0.5  (strict — normal rhythm should be regular)
  Pre-AF segments : max_rr_cv=0.7  (loose — legitimate RR variability before AF onset)
"""

import wfdb
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import defaultdict
from tqdm import tqdm

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_cleaning.data_cleaner import DataCleaner
from wavelets.stationary_wavelet_transform import SWT

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR   = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/paf')
OUTPUT_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf_updated')
HDF5_PATH  = OUTPUT_DIR / 'paf_segments_all_clean_new_cleaning.h5'

SEGMENT_DURATION_SEC = 12
SAMPLING_FREQ        = 128
LEAD_INDEX           = 0

# Temporal window parameters for pre-AF progression analysis
WINDOW_DURATION_SEC = 150  # 2.5 minutes
WINDOWS_PER_PATIENT = 12   # 12 windows = 30 minutes total
PREAF_RECORD_MIN_DURATION_SEC = 1800  # 30 minutes minimum

# Per-label RR-CV thresholds
# Pre-AF recordings immediately precede AF onset — increased RR variability
# is physiologically expected and should not cause rejection.
MAX_RR_CV_NORMAL = 0.6
MAX_RR_CV_PREAF  = 0.7


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class PAFSegment:
    segment_id: str
    patient_id: str
    record_id:  str
    signal:     np.ndarray
    label:      int
    subbands:   Optional[Dict[str, np.ndarray]] = None


@dataclass
class PAFTemporalWindow:
    """
    Temporal window for pre-AF progression analysis.

    Each pre-AF patient contributes 12 matched pairs:
      - 12 consecutive 2.5-min pre-AF windows from p-even record
      - 12 random 2.5-min normal windows from p-odd record
    """
    window_id: str              # e.g., "paf_p02_w0_l1"
    patient_id: str             # e.g., "paf_p01"
    preaf_record_id: str        # p-even record (only set if label=1)
    normal_record_id: str       # p-odd record (only set if label=0)
    signal: np.ndarray          # 2.5 min @ 128 Hz = 19,200 samples
    label: int                  # 0=normal, 1=pre-AF
    window_index: int           # 0-11 (temporal position in 30-min window)
    subbands: Optional[Dict[str, np.ndarray]] = None


# =============================================================================
# Extractor
# =============================================================================

class PAFExtractor:
    def __init__(self, random_seed: int = 42, enable_subband: bool = False,
                 swt_level: int = 4, wavelet: str = 'db4'):
        self.rng            = np.random.default_rng(random_seed)
        self.cleaner        = DataCleaner(fs=SAMPLING_FREQ)
        self.enable_subband = enable_subband
        self.swt_level      = swt_level
        self.wavelet        = wavelet
        if enable_subband:
            self.swt = SWT(wavelet=wavelet)

        # Rejection tracking — populated during extraction, printed in summary
        self._rejection_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # ------------------------------------------------------------------
    # Record metadata
    # ------------------------------------------------------------------

    def _label(self, record_id: str) -> Optional[int]:
        if record_id.endswith('c') or record_id.startswith('t'):
            return None
        if record_id.startswith('n'):
            return 0
        if record_id.startswith('p'):
            try:
                return 1 if int(record_id[1:]) % 2 == 0 else 0
            except ValueError:
                return None
        return None

    def _patient_id(self, record_id: str) -> str:
        if record_id.startswith('p'):
            pair_base = ((int(record_id[1:]) - 1) // 2) + 1
            return f"paf_p{pair_base:02d}"
        return f"paf_{record_id}"

    def _get_all_records(self) -> Dict[str, List[str]]:
        records = {'normal': [], 'preaf': []}
        for hea in sorted(DATA_DIR.glob('*.hea')):
            rid   = hea.stem
            label = self._label(rid)
            if label == 0:
                records['normal'].append(rid)
            elif label == 1:
                records['preaf'].append(rid)
        return records

    # ------------------------------------------------------------------
    # Subband decomposition — verified coefficient ordering
    # ------------------------------------------------------------------

    def _decompose_to_subbands(self, signal: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """
        SWT decomposition with correct frequency labeling.

        Verified by PSD analysis:
          coeffs[0] -> 4-8 Hz  -> D4  (coarsest detail)
          coeffs[1] -> 8-16 Hz -> D3
          coeffs[2] -> 16-32 Hz -> D2
          coeffs[3] -> 32-64 Hz -> D1  (finest detail)

        The loop uses label = D{SWT_LEVEL - i} to map correctly:
          i=0 -> D4, i=1 -> D3, i=2 -> D2, i=3 -> D1
        """
        coeffs = self.swt.decompose(signal, level=self.swt_level)

        subbands = {}

        # Detail subbands — reversed labeling
        for i in range(self.swt_level):
            label  = f"D{self.swt_level - i}"   # i=0->D4, i=1->D3, i=2->D2, i=3->D1
            zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
            raw = self.swt.reconstruct(zeroed)
            subbands[label] = self.cleaner.preprocess_subband(raw)

        # Approximation subband A4
        zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        raw    = self.swt.reconstruct(zeroed)
        subbands[f'A{self.swt_level}'] = self.cleaner.preprocess_subband(raw)

        return subbands

    # ------------------------------------------------------------------
    # Segment extraction from one record
    # ------------------------------------------------------------------

    def _extract_from_record(
        self,
        record_id: str,
        label: int,
        n_segments: int,
    ) -> List[PAFSegment]:
        """
        Extract up to n_segments cleaned segments from one record.

        Quality check threshold depends on label:
          label=0 (normal) : max_rr_cv=0.5
          label=1 (pre-AF) : max_rr_cv=0.7

        Rejection reasons are tracked in self._rejection_counts for
        post-extraction diagnostics.
        """
        try:
            record = wfdb.rdrecord(str(DATA_DIR / record_id))
        except Exception as e:
            print(f"  Cannot load {record_id}: {e}")
            return []

        if record.fs != SAMPLING_FREQ:
            print(f"  Skipping {record_id}: unexpected fs={record.fs}")
            return []

        raw_signal  = record.p_signal[:, LEAD_INDEX].astype(np.float32)
        seg_samples = int(SEGMENT_DURATION_SEC * SAMPLING_FREQ)
        n_possible  = len(raw_signal) // seg_samples

        if n_possible == 0:
            return []

        n_extract  = min(n_segments, n_possible)
        starts     = np.linspace(0, len(raw_signal) - seg_samples, n_extract, dtype=int)
        patient_id = self._patient_id(record_id)
        label_name = 'preaf' if label == 1 else 'normal'
        max_rr_cv  = MAX_RR_CV_PREAF if label == 1 else MAX_RR_CV_NORMAL

        segments = []
        for idx, start in enumerate(starts):
            raw     = raw_signal[start: start + seg_samples]
            cleaned, info = self.cleaner.preprocess(raw, check_quality=True,
                                                     max_rr_cv=max_rr_cv)
            if cleaned is None:
                reason = info.get('reason', 'unknown')
                self._rejection_counts[label_name][reason] += 1
                continue

            subbands = None
            if self.enable_subband:
                try:
                    subbands = self._decompose_to_subbands(cleaned)
                except Exception as e:
                    print(f"  Subband decomposition failed {record_id}_s{idx}: {e}")
                    self._rejection_counts[label_name]['subband_error'] += 1
                    continue

            segments.append(PAFSegment(
                segment_id=f"paf_{record_id}_s{idx}_l{label}",
                patient_id=patient_id,
                record_id=record_id,
                signal=cleaned,
                label=label,
                subbands=subbands,
            ))

        return segments

    # ------------------------------------------------------------------
    # Temporal window extraction (for pre-AF progression analysis)
    # ------------------------------------------------------------------

    def _extract_windows_from_patient_pair(
        self,
        preaf_record_id: str,
        normal_record_id: str,
        patient_id: str
    ) -> List[PAFTemporalWindow]:
        """
        Extract 12 matched window pairs from one patient.

        Returns up to 12 matched pairs:
          - 12 consecutive 2.5-min pre-AF windows from preaf_record_id
          - 12 random 2.5-min normal windows from normal_record_id

        Only returns windows where BOTH pre-AF and normal passed quality check.
        """
        # Load pre-AF record (p-even)
        try:
            preaf_rec = wfdb.rdrecord(str(DATA_DIR / preaf_record_id))
        except Exception as e:
            print(f"  Cannot load pre-AF record {preaf_record_id}: {e}")
            return []

        # Load normal record (p-odd)
        try:
            normal_rec = wfdb.rdrecord(str(DATA_DIR / normal_record_id))
        except Exception as e:
            print(f"  Cannot load normal record {normal_record_id}: {e}")
            return []

        if preaf_rec.fs != SAMPLING_FREQ or normal_rec.fs != SAMPLING_FREQ:
            print(f"  Skipping {patient_id}: unexpected sampling frequency")
            return []

        preaf_signal = preaf_rec.p_signal[:, LEAD_INDEX].astype(np.float32)
        normal_signal = normal_rec.p_signal[:, LEAD_INDEX].astype(np.float32)

        window_samples = int(WINDOW_DURATION_SEC * SAMPLING_FREQ)

        # Validate pre-AF record length (must have 30 min = 12 windows)
        preaf_duration_sec = len(preaf_signal) / SAMPLING_FREQ
        if preaf_duration_sec < PREAF_RECORD_MIN_DURATION_SEC:
            print(f"  Skipping {patient_id}: pre-AF record too short "
                  f"({preaf_duration_sec:.1f}s < {PREAF_RECORD_MIN_DURATION_SEC}s)")
            return []

        # Validate normal record length (must fit 12 non-overlapping windows)
        normal_min_samples = window_samples * WINDOWS_PER_PATIENT
        if len(normal_signal) < normal_min_samples:
            print(f"  Skipping {patient_id}: normal record too short for 12 windows")
            return []

        # Extract 12 consecutive pre-AF windows
        preaf_windows = []
        for i in range(WINDOWS_PER_PATIENT):
            start = i * window_samples
            end = start + window_samples
            raw = preaf_signal[start:end]

            cleaned, info = self.cleaner.preprocess(raw, check_quality=True,
                                                     max_rr_cv=MAX_RR_CV_PREAF)
            if cleaned is None:
                reason = info.get('reason', 'unknown')
                self._rejection_counts['preaf'][f'window_{reason}'] += 1
                preaf_windows.append(None)  # Placeholder to maintain indexing
            else:
                preaf_windows.append(cleaned)

        # Extract 12 random non-overlapping normal windows
        # Generate random start positions for 12 segments
        max_start = len(normal_signal) - window_samples
        possible_starts = np.arange(0, max_start - window_samples * (WINDOWS_PER_PATIENT - 1), window_samples)

        if len(possible_starts) < 1:
            # Fallback: allow overlapping if record is shorter
            possible_starts = np.arange(0, max_start + 1, window_samples // 2)

        # Randomly select starting points ensuring non-overlap
        selected_starts = []
        attempts = 0
        max_attempts = 100

        while len(selected_starts) < WINDOWS_PER_PATIENT and attempts < max_attempts:
            candidate = self.rng.integers(0, max_start + 1)
            # Check for overlap with already selected windows
            overlaps = False
            for existing_start in selected_starts:
                if abs(candidate - existing_start) < window_samples:
                    overlaps = True
                    break
            if not overlaps:
                selected_starts.append(candidate)
            attempts += 1

        if len(selected_starts) < WINDOWS_PER_PATIENT:
            print(f"  Warning {patient_id}: could only find {len(selected_starts)} "
                  f"non-overlapping normal windows")

        normal_windows = []
        for start in selected_starts[:WINDOWS_PER_PATIENT]:
            raw = normal_signal[start:start + window_samples]
            cleaned, info = self.cleaner.preprocess(raw, check_quality=True,
                                                     max_rr_cv=MAX_RR_CV_NORMAL)
            if cleaned is None:
                reason = info.get('reason', 'unknown')
                self._rejection_counts['normal'][f'window_{reason}'] += 1
                # Try one replacement attempt
                replacement_start = self.rng.integers(0, max_start + 1)
                raw_replacement = normal_signal[replacement_start:replacement_start + window_samples]
                cleaned, info = self.cleaner.preprocess(raw_replacement, check_quality=True,
                                                         max_rr_cv=MAX_RR_CV_NORMAL)
                if cleaned is None:
                    normal_windows.append(None)
                else:
                    normal_windows.append(cleaned)
            else:
                normal_windows.append(cleaned)

        # Pad normal_windows if we couldn't get enough
        while len(normal_windows) < WINDOWS_PER_PATIENT:
            normal_windows.append(None)

        # Create matched pairs — only where both pre-AF and normal passed
        windows = []
        for i in range(WINDOWS_PER_PATIENT):
            if preaf_windows[i] is None or normal_windows[i] is None:
                continue

            # Process subbands if enabled
            preaf_subbands = None
            normal_subbands = None
            if self.enable_subband:
                try:
                    preaf_subbands = self._decompose_to_subbands(preaf_windows[i])
                    normal_subbands = self._decompose_to_subbands(normal_windows[i])
                except Exception as e:
                    print(f"  Subband decomposition failed {patient_id}_w{i}: {e}")
                    self._rejection_counts['preaf']['subband_error'] += 1
                    continue

            # Add pre-AF window
            windows.append(PAFTemporalWindow(
                window_id=f"paf_{preaf_record_id}_w{i}_l1",
                patient_id=patient_id,
                preaf_record_id=preaf_record_id,
                normal_record_id="",
                signal=preaf_windows[i],
                label=1,
                window_index=i,
                subbands=preaf_subbands
            ))

            # Add matched normal window
            windows.append(PAFTemporalWindow(
                window_id=f"paf_{normal_record_id}_w{i}_l0",
                patient_id=patient_id,
                preaf_record_id="",
                normal_record_id=normal_record_id,
                signal=normal_windows[i],
                label=0,
                window_index=i,
                subbands=normal_subbands
            ))

        return windows

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _print_rejection_summary(self):
        """Print per-label rejection counts broken down by reason."""
        print("\n--- Rejection summary ---")
        for label_name in ['preaf', 'normal']:
            counts = self._rejection_counts[label_name]
            if not counts:
                print(f"  {label_name}: no rejections")
                continue
            total = sum(counts.values())
            print(f"  {label_name}: {total} rejected")
            for reason, count in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")
        print()

    def print_per_record_yield(self, segments_per_record: int = 50):
        """
        Dry run — print how many segments each record produces without saving.
        Run this first to diagnose which records are causing imbalance.
        """
        records = self._get_all_records()
        self._rejection_counts.clear()

        print("=== Per-record yield (dry run) ===")
        for label_name, label_int in [('preaf', 1), ('normal', 0)]:
            print(f"\n{label_name.upper()} records:")
            record_list = records['preaf'] if label_int == 1 else records['normal']
            for rid in record_list:
                segs = self._extract_from_record(rid, label_int, segments_per_record)
                print(f"  {rid}: {len(segs)} / {segments_per_record} accepted")

        self._print_rejection_summary()

    # ------------------------------------------------------------------
    # Temporal window extraction - main method
    # ------------------------------------------------------------------

    def extract_temporal_windows(self) -> List[PAFTemporalWindow]:
        """
        Extract temporal windows for pre-AF progression analysis.

        For each pre-AF patient (p-pairs):
          - Extract 12 consecutive 2.5-min windows from p-even (pre-AF)
          - Extract 12 random 2.5-min windows from p-odd (normal)
          - Return matched pairs only (both must pass quality check)

        Returns
        -------
        List[PAFTemporalWindow]
            All temporal windows from all patients, alternating pre-AF/normal pairs.
        """
        records = self._get_all_records()
        self._rejection_counts.clear()

        # Build patient pairs (p-even → p-odd mapping)
        preaf_records = [r for r in records['preaf'] if r.startswith('p')]
        patient_pairs = []

        for preaf_rec in preaf_records:
            # p02 → p01, p04 → p03, etc.
            preaf_num = int(preaf_rec[1:])
            normal_num = preaf_num - 1
            normal_rec = f"p{normal_num:02d}"

            # Verify the normal record exists
            if normal_rec in records['normal']:
                patient_id = self._patient_id(preaf_rec)
                patient_pairs.append((preaf_rec, normal_rec, patient_id))

        print(f"Found {len(patient_pairs)} pre-AF patient pairs")
        print(f"Extracting {WINDOWS_PER_PATIENT} × 2.5-min windows per patient...")
        print(f"  Pre-AF  max_rr_cv={MAX_RR_CV_PREAF}")
        print(f"  Normal  max_rr_cv={MAX_RR_CV_NORMAL}")

        all_windows = []
        successful_patients = 0

        for preaf_rec, normal_rec, patient_id in tqdm(patient_pairs, desc="Extracting temporal windows"):
            windows = self._extract_windows_from_patient_pair(preaf_rec, normal_rec, patient_id)
            if windows:
                all_windows.extend(windows)
                # Count successful if at least one matched pair
                if len(windows) >= 2:
                    successful_patients += 1

        self._print_rejection_summary()

        n_preaf = sum(w.label == 1 for w in all_windows)
        n_normal = sum(w.label == 0 for w in all_windows)

        print(f"\nTemporal window extraction complete:")
        print(f"  Patients with windows: {successful_patients} / {len(patient_pairs)}")
        print(f"  Pre-AF windows  : {n_preaf}")
        print(f"  Normal windows  : {n_normal}")
        print(f"  Matched pairs   : {min(n_preaf, n_normal)}")

        return all_windows

    # ------------------------------------------------------------------
    # Balanced extraction
    # ------------------------------------------------------------------

    def extract_balanced(self, segments_per_record: int = 50) -> List[PAFSegment]:
        """
        Extract segments ensuring total pre-AF == total normal.

        Strategy:
          1. Extract from every pre-AF record (p-even).
          2. For each paired p-odd record, extract same count as its
             paired p-even produced (within-patient balance).
          3. Fill remaining normal quota from n* records.
          4. Cap normal to match pre-AF total exactly.
        """
        records = self._get_all_records()
        self._rejection_counts.clear()

        print(f"Found {len(records['preaf'])} pre-AF records, "
              f"{len(records['normal'])} normal records")
        print(f"Extracting up to {segments_per_record} segments per record...")
        print(f"  Pre-AF  max_rr_cv={MAX_RR_CV_PREAF}")
        print(f"  Normal  max_rr_cv={MAX_RR_CV_NORMAL}")

        # Step 1: pre-AF segments
        preaf_segments: List[PAFSegment] = []
        for rid in tqdm(records['preaf'], desc="Pre-AF records"):
            segs = self._extract_from_record(rid, label=1,
                                              n_segments=segments_per_record)
            preaf_segments.extend(segs)

        total_preaf = len(preaf_segments)
        print(f"Pre-AF segments accepted: {total_preaf}")

        if total_preaf == 0:
            self._print_rejection_summary()
            raise RuntimeError("No pre-AF segments accepted — check data path and thresholds.")

        # Step 2: paired p-odd records
        preaf_by_patient: Dict[str, int] = {}
        for seg in preaf_segments:
            preaf_by_patient[seg.patient_id] = preaf_by_patient.get(seg.patient_id, 0) + 1

        paired_normal: List[PAFSegment] = []
        p_odd_records = [rid for rid in records['normal'] if rid.startswith('p')]
        for rid in tqdm(p_odd_records, desc="Paired normal (p-odd)"):
            patient_id = self._patient_id(rid)
            n_match    = preaf_by_patient.get(patient_id, segments_per_record)
            segs       = self._extract_from_record(rid, label=0, n_segments=n_match)
            paired_normal.extend(segs)

        # Step 3: n* normal records to fill quota
        n_needed  = total_preaf - len(paired_normal)
        n_records = [rid for rid in records['normal'] if rid.startswith('n')]

        independent_normal: List[PAFSegment] = []
        if n_needed > 0 and n_records:
            per_n_record = max(1, int(np.ceil(n_needed / len(n_records))))
            for rid in tqdm(n_records, desc="Independent normal (n*)"):
                if len(independent_normal) >= n_needed:
                    break
                segs = self._extract_from_record(rid, label=0, n_segments=per_n_record)
                independent_normal.extend(segs)
            independent_normal = independent_normal[:n_needed]

        all_normal = paired_normal + independent_normal

        # Step 4: cap normal to pre-AF total
        if len(all_normal) > total_preaf:
            idx        = self.rng.choice(len(all_normal), size=total_preaf, replace=False)
            all_normal = [all_normal[i] for i in sorted(idx)]

        self._print_rejection_summary()

        print(f"Extraction complete:")
        print(f"  Pre-AF : {len(preaf_segments)}")
        print(f"  Normal : {len(all_normal)}")
        print(f"    paired p-odd  : {len(paired_normal)}")
        print(f"    n* records    : {len(independent_normal)}")

        return preaf_segments + all_normal

    # ------------------------------------------------------------------
    # HDF5 export - temporal windows
    # ------------------------------------------------------------------

    def export_temporal_windows_to_hdf5(self, export_csv: bool = True):
        """
        Extract and export temporal windows to HDF5 and CSV.

        Creates:
          - HDF5: paf_temporal_windows.h5
          - CSV: temporal_single_scale.csv + subband CSVs if enabled
        """
        windows = self.extract_temporal_windows()

        if not windows:
            print("No temporal windows extracted")
            return

        n_preaf = sum(w.label == 1 for w in windows)
        n_normal = sum(w.label == 0 for w in windows)

        hdf5_path = OUTPUT_DIR / 'paf_temporal_windows.h5'
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving {len(windows)} temporal windows to {hdf5_path}...")

        with h5py.File(hdf5_path, 'w') as f:
            grp = f.create_group('temporal_windows')
            grp.create_dataset('signals',
                               data=np.stack([w.signal for w in windows]).astype(np.float32),
                               compression='gzip', compression_opts=4)
            grp.create_dataset('labels',
                               data=np.array([w.label for w in windows], dtype=np.int8))
            grp.create_dataset('window_indices',
                               data=np.array([w.window_index for w in windows], dtype=np.int8))
            grp.create_dataset('window_ids',
                               data=np.array([w.window_id for w in windows], dtype='S64'))
            grp.create_dataset('patient_ids',
                               data=np.array([w.patient_id for w in windows], dtype='S16'))
            grp.create_dataset('preaf_record_ids',
                               data=np.array([w.preaf_record_id for w in windows], dtype='S16'))
            grp.create_dataset('normal_record_ids',
                               data=np.array([w.normal_record_id for w in windows], dtype='S16'))

            grp.attrs['n_preaf'] = n_preaf
            grp.attrs['n_normal'] = n_normal
            grp.attrs['sampling_freq'] = SAMPLING_FREQ
            grp.attrs['window_duration_sec'] = WINDOW_DURATION_SEC
            grp.attrs['windows_per_patient'] = WINDOWS_PER_PATIENT

            # Save subbands if enabled
            if self.enable_subband and any(w.subbands for w in windows):
                subband_names = [k for k in windows[0].subbands.keys()
                                 if windows[0].subbands[k] is not None]
                for sb_name in subband_names:
                    data = np.stack([w.subbands[sb_name] for w in windows
                                     if w.subbands and w.subbands.get(sb_name) is not None]
                                    ).astype(np.float32)
                    grp.create_dataset(f'subband_{sb_name}', data=data,
                                       compression='gzip', compression_opts=4)

        print(f"Saved — Pre-AF: {n_preaf}, Normal: {n_normal}")

        if export_csv:
            self._export_temporal_windows_csv(windows)

    # ------------------------------------------------------------------
    # HDF5 export - standard segments
    # ------------------------------------------------------------------

    def export_to_hdf5(self, segments_per_record: int = 50, export_csv: bool = True):
        segments = self.extract_balanced(segments_per_record=segments_per_record)

        n_preaf  = sum(s.label == 1 for s in segments)
        n_normal = sum(s.label == 0 for s in segments)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving {len(segments)} segments to {HDF5_PATH}...")

        with h5py.File(HDF5_PATH, 'w') as f:
            grp = f.create_group('all')
            grp.create_dataset('signals',
                               data=np.stack([s.signal for s in segments]).astype(np.float32),
                               compression='gzip', compression_opts=4)
            grp.create_dataset('labels',
                               data=np.array([s.label for s in segments], dtype=np.int8))
            grp.create_dataset('segment_ids',
                               data=np.array([s.segment_id for s in segments], dtype='S64'))
            grp.create_dataset('patient_ids',
                               data=np.array([s.patient_id for s in segments], dtype='S16'))
            grp.create_dataset('record_ids',
                               data=np.array([s.record_id for s in segments], dtype='S16'))
            grp.attrs['n_preaf']       = n_preaf
            grp.attrs['n_normal']      = n_normal
            grp.attrs['sampling_freq'] = SAMPLING_FREQ
            grp.attrs['segment_sec']   = SEGMENT_DURATION_SEC

            if self.enable_subband and any(s.subbands for s in segments):
                subband_names = [k for k in segments[0].subbands.keys()
                                 if segments[0].subbands[k] is not None]
                for sb_name in subband_names:
                    data = np.stack([s.subbands[sb_name] for s in segments
                                     if s.subbands and s.subbands.get(sb_name) is not None]
                                    ).astype(np.float32)
                    grp.create_dataset(f'subband_{sb_name}', data=data,
                                       compression='gzip', compression_opts=4)

        print(f"Saved — Pre-AF: {n_preaf}, Normal: {n_normal}")

        if export_csv:
            self._export_single_scale_csv(segments)
            if self.enable_subband:
                self._export_subbands_csv(segments)

    # ------------------------------------------------------------------
    # CSV helpers - temporal windows
    # ------------------------------------------------------------------

    def _export_temporal_windows_csv(self, windows: List[PAFTemporalWindow]):
        """Export temporal windows to CSV files."""
        output_dir = OUTPUT_DIR / 'temporal_windows'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Single-scale CSV
        rows = []
        for w in tqdm(windows, desc="Exporting temporal single-scale CSV"):
            row = {
                'window_id': w.window_id,
                'patient_id': w.patient_id,
                'label': w.label,
                'window_index': w.window_index,
                'preaf_record_id': w.preaf_record_id,
                'normal_record_id': w.normal_record_id
            }
            for i, v in enumerate(w.signal):
                row[f's{i}'] = v
            rows.append(row)

        pd.DataFrame(rows).to_csv(output_dir / 'temporal_single_scale.csv', index=False)
        print(f"Saved temporal_single_scale.csv ({len(windows)} windows)")

        # Subband CSVs if enabled
        if self.enable_subband:
            valid = [w for w in windows if w.subbands]
            if valid:
                subband_names = sorted(valid[0].subbands.keys())
                for sb_name in tqdm(subband_names, desc="Exporting temporal subband CSVs"):
                    rows = []
                    for w in valid:
                        sb = w.subbands.get(sb_name)
                        if sb is None:
                            continue
                        row = {
                            'window_id': w.window_id,
                            'patient_id': w.patient_id,
                            'label': w.label,
                            'window_index': w.window_index,
                            'preaf_record_id': w.preaf_record_id,
                            'normal_record_id': w.normal_record_id
                        }
                        for i, v in enumerate(sb):
                            row[f's{i}'] = v
                        rows.append(row)
                    pd.DataFrame(rows).to_csv(output_dir / f'temporal_{sb_name}.csv', index=False)

                print(f"Saved {len(subband_names)} temporal subband CSVs ({len(valid)} windows each)")

    # ------------------------------------------------------------------
    # CSV helpers - standard segments
    # ------------------------------------------------------------------

    def _export_single_scale_csv(self, segments: List[PAFSegment],
                                  output_dir: Optional[Path] = None):
        if output_dir is None:
            output_dir = OUTPUT_DIR / 'subbands'
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for seg in tqdm(segments, desc="Exporting single-scale CSV"):
            row = {'segment_id': seg.segment_id,
                   'patient_id': seg.patient_id,
                   'label': seg.label}
            for i, v in enumerate(seg.signal):
                row[f's{i}'] = v
            rows.append(row)

        pd.DataFrame(rows).to_csv(output_dir / 'single_scale.csv', index=False)
        print(f"Saved single_scale.csv ({len(segments)} segments)")

    def _export_subbands_csv(self, segments: List[PAFSegment],
                              output_dir: Optional[Path] = None):
        if output_dir is None:
            output_dir = OUTPUT_DIR / 'subbands'
        output_dir.mkdir(parents=True, exist_ok=True)

        valid = [s for s in segments if s.subbands]
        if not valid:
            return

        subband_names = sorted(valid[0].subbands.keys())
        for sb_name in tqdm(subband_names, desc="Exporting subband CSVs"):
            rows = []
            for seg in valid:
                sb = seg.subbands.get(sb_name)
                if sb is None:
                    continue
                row = {'segment_id': seg.segment_id,
                       'patient_id': seg.patient_id,
                       'label': seg.label}
                for i, v in enumerate(sb):
                    row[f's{i}'] = v
                rows.append(row)
            pd.DataFrame(rows).to_csv(output_dir / f'{sb_name}.csv', index=False)

        print(f"Saved {len(subband_names)} subband CSVs ({len(valid)} segments each)")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    extractor = PAFExtractor(random_seed=42, enable_subband=True)

    # Option 1: Standard segment extraction (random segments from records)
    # extractor.print_per_record_yield(segments_per_record=50)
    # extractor.export_to_hdf5(segments_per_record=50, export_csv=True)

    # Option 2: Temporal window extraction (12 × 2.5-min windows per patient)
    # For pre-AF progression analysis with temporal information preserved
    extractor.export_temporal_windows_to_hdf5(export_csv=True)