import re

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


DATA_DIR   = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/paf')
OUTPUT_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_new/exp2')
HDF5_PATH  = OUTPUT_DIR / 'paf_new.h5'

SEGMENT_DURATION_SEC = 10
SAMPLING_FREQ        = 128
LEAD_INDEX           = 0

MAX_RR_CV_NORMAL = 0.6
MAX_RR_CV_PREAF  = 0.7


@dataclass
class PAFSegment:
    segment_id: str
    patient_id: str
    record_id:  str
    signal:     np.ndarray
    label:      int
    subbands:   Optional[Dict[str, np.ndarray]] = None


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
        self._rejection_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _label(self, record_id: str) -> Optional[int]:
        """
        CORRECTED LABELING (Sensors 2024 / AFPDB Ground Truth):
        - p-odd (p01, p03...): Pre-AF (Class 1)
        - p-even (p02, p04...): Distant / Non-AF (Class 0)
        - n* : Healthy / Non-AF (Class 0)
        """
        if record_id.endswith('c') or record_id.startswith('t'):
            return None

        # Extract the numeric part regardless of prefix
        match = re.search(r'\d+', record_id)
        if not match: return None
        num = int(match.group())

        if 'n' in record_id:
            return 0
        if 'p' in record_id:
            return 1 if num % 2 != 0 else 0
        return None

    def _patient_id(self, record_id: str) -> str:
        """
        Robustly groups records into unique human subjects.
        p01 and p02 -> paf_subject_p01
        n01 and n02 -> paf_subject_n01
        """
        match = re.search(r'([pn])(\d+)', record_id)
        if match:
            prefix = match.group(1)
            num = int(match.group(2))
            # Group pairs: (01,02)->1, (03,04)->2
            pair_base = (num + 1) // 2
            return f"paf_subject_{prefix}{pair_base:02d}"
        return f"paf_{record_id}"

    def _get_all_records(self) -> Dict[str, List[str]]:
        """Scans directory and categorizes records based on corrected labels."""
        records = {'normal': [], 'preaf': []}
        for hea in sorted(DATA_DIR.glob('*.hea')):
            rid = hea.stem
            label = self._label(rid)
            if label == 0:
                records['normal'].append(rid)
            elif label == 1:
                records['preaf'].append(rid)
        return records


    def _decompose_to_subbands(self, signal: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        coeffs = self.swt.decompose(signal, level=self.swt_level)

        subbands = {}

        for i in range(self.swt_level):
            label  = f"D{self.swt_level - i}"   # i=0->D4, i=1->D3, i=2->D2, i=3->D1
            zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
            raw = self.swt.reconstruct(zeroed)
            subbands[label] = self.cleaner.preprocess_subband(raw)

        zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        raw    = self.swt.reconstruct(zeroed)
        subbands[f'A{self.swt_level}'] = self.cleaner.preprocess_subband(raw)

        return subbands

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
            cleaned, info = self.cleaner.preprocess(raw, check_quality=False,max_rr_cv=max_rr_cv)
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


    def extract_all(self, segments_per_record: int = 170) -> List[PAFSegment]:
        records = self._get_all_records()
        self._rejection_counts.clear()

        print(f"Detected: {len(records['preaf'])} Pre-AF (p-odd) | {len(records['normal'])} Normal (n* & p-even)")

        all_segments: List[PAFSegment] = []

        for rid in tqdm(records['preaf'], desc="Extracting Pre-AF (p-odd)"):
            segs = self._extract_from_record(rid, label=1, n_segments=segments_per_record)
            all_segments.extend(segs)

        # 2. Normal / Distant (p-even & n*)
        for rid in tqdm(records['normal'], desc="Extracting Non-AF (p-even & n*)"):
            segs = self._extract_from_record(rid, label=0, n_segments=segments_per_record)
            all_segments.extend(segs)

        self._print_rejection_summary()
        return all_segments


    def export_to_hdf5(self, segments_per_record: int = 50, export_csv: bool = True):

        segments = self.extract_all(
            segments_per_record=segments_per_record
        )

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

    def _export_single_scale_csv(self, segments: List[PAFSegment], output_dir: Optional[Path] = None):
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

    def _export_subbands_csv(self, segments: List[PAFSegment], output_dir: Optional[Path] = None):
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


if __name__ == '__main__':
    extractor = PAFExtractor(random_seed=42, enable_subband=False)

    # extractor.print_per_record_yield(segments_per_record=50)

    extractor.export_to_hdf5(
        segments_per_record=1000,
        export_csv=True,
    )
