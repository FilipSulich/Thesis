"""
PAF (Paroxysmal Atrial Fibrillation) Dataset Extractor

Extracts ECG segments from the PAF Prediction Challenge Database.

Dataset structure (from PhysioNet docs):
- Learning set: 100 records from 48 subjects
  - n* records: Normal subjects (no documented AF)
  - p* records: PAF subjects in pairs:
    - Odd-numbered (p01, p03, ...): Distant from PAF (>45 min from any AF)
    - Even-numbered (p02, p04, ...): Immediately precede PAF onset

Labeling scheme:
- label=1 (pre-AF): Even-numbered p* records (p02, p04, p06, ...)
- label=0 (normal): All n* records + odd-numbered p* records (p01, p03, ...)

Processing:
- Extract 12s segments @ 128 Hz
- Patient-wise split into train/val/test
- Cleaning (filter, normalize, trim to 10s) applied ONLY to train set
- Upsample to 250 Hz

Sampling frequency: 128 Hz -> 250 Hz
"""

import wfdb
import numpy as np
import h5py
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from tqdm import tqdm
from scipy.signal import resample

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_cleaning.data_cleaner import DataCleaner

DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/paf')
OUTPUT_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf')

SEGMENT_DURATION_SEC = 12
TRIMMED_DURATION_SEC = 10
LEAD_INDEX = 0
ORIGINAL_FS = 128
TARGET_FS = 128

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


@dataclass
class PAFSegment:
    """Represents an extracted PAF segment."""
    segment_id: str
    patient_id: str
    record_id: str
    signal: np.ndarray
    sampling_freq: int
    segment_start_sec: float
    segment_duration_sec: float
    lead_index: int
    lead_name: str
    label: int  # 1 = pre-AF, 0 = normal


class PAFExtractor:
    """Extracts labeled segments from the PAF database."""

    def __init__(self):
        self.data_dir = DATA_DIR
        self.segment_duration_sec = SEGMENT_DURATION_SEC
        self.lead_index = LEAD_INDEX

    def _get_record_label(self, record_id: str) -> Optional[int]:
        """Determine label for a record based on its ID."""
        if record_id.endswith('c') or record_id.startswith('t'):
            return None

        if record_id.startswith('n'):
            return 0

        if record_id.startswith('p'):
            try:
                num = int(record_id[1:])
                return 1 if num % 2 == 0 else 0
            except ValueError:
                return None

        return None

    def _get_patient_id(self, record_id: str) -> str:
        """Get patient ID from record ID."""
        if record_id.startswith('p'):
            num = int(record_id[1:])
            pair_base = ((num - 1) // 2) + 1
            return f"paf_p{pair_base:02d}"
        else:
            return f"paf_{record_id}"

    def get_record_ids(self) -> Dict[str, List[str]]:
        """Get record IDs organized by category."""
        all_headers = list(self.data_dir.glob('*.hea'))
        records = {'normal': [], 'preaf': []}

        for hea_file in all_headers:
            record_id = hea_file.stem
            label = self._get_record_label(record_id)

            if label is None:
                continue
            elif label == 0:
                records['normal'].append(record_id)
            elif label == 1:
                records['preaf'].append(record_id)

        for key in records:
            records[key] = sorted(records[key])

        return records

    def get_patient_record_mapping(self) -> Dict[str, List[str]]:
        """Get mapping of patient_id -> list of record_ids."""
        records = self.get_record_ids()
        all_records = records['normal'] + records['preaf']

        patient_records = {}
        for record_id in all_records:
            patient_id = self._get_patient_id(record_id)
            if patient_id not in patient_records:
                patient_records[patient_id] = []
            patient_records[patient_id].append(record_id)

        return patient_records

    def create_patient_split(self, seed: int = 42) -> Dict[str, Set[str]]:
        patient_records = self.get_patient_record_mapping()

        # Separate patients by whether they have pre-AF records
        preaf_patients = []
        normal_patients = []

        records = self.get_record_ids()
        preaf_record_ids = set(records['preaf'])

        for pid, rids in patient_records.items():
            if any(r in preaf_record_ids for r in rids):
                preaf_patients.append(pid)
            else:
                normal_patients.append(pid)

        np.random.seed(seed)
        np.random.shuffle(preaf_patients)
        np.random.shuffle(normal_patients)

        def split_list(lst):
            n = len(lst)
            n_train = int(n * TRAIN_RATIO)
            n_val = int(n * VAL_RATIO)
            return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

        preaf_train, preaf_val, preaf_test = split_list(preaf_patients)
        norm_train, norm_val, norm_test = split_list(normal_patients)

        split = {
            'train': set(preaf_train + norm_train),
            'val': set(preaf_val + norm_val),
            'test': set(preaf_test + norm_test),
        }

        print(f"Patient split: {len(split['train'])} train ({len(preaf_train)} preaf, {len(norm_train)} normal), "
              f"{len(split['val'])} val, {len(split['test'])} test")
        return split

    def extract_segments_from_record(
        self,
        record_id: str,
        label: int,
        num_segments: Optional[int] = None
    ) -> List[PAFSegment]:
        """Extract segments from a single record."""
        record_path = str(self.data_dir / record_id)
        record = wfdb.rdrecord(record_path)

        fs = record.fs
        signal_length = record.sig_len
        segment_samples = int(self.segment_duration_sec * fs)

        max_segments = signal_length // segment_samples
        if num_segments is not None:
            max_segments = min(max_segments, num_segments)

        segments = []
        lead_name = record.sig_name[self.lead_index] if self.lead_index < len(record.sig_name) else f"lead_{self.lead_index}"
        patient_id = self._get_patient_id(record_id)

        for seg_idx in range(max_segments):
            start_sample = seg_idx * segment_samples
            end_sample = start_sample + segment_samples

            signal = record.p_signal[start_sample:end_sample, self.lead_index]
            segment_id = f"paf_{record_id}_s{seg_idx}_l{self.lead_index}"

            segments.append(PAFSegment(
                segment_id=segment_id,
                patient_id=patient_id,
                record_id=record_id,
                signal=signal.astype(np.float32),
                sampling_freq=fs,
                segment_start_sec=start_sample / fs,
                segment_duration_sec=self.segment_duration_sec,
                lead_index=self.lead_index,
                lead_name=lead_name,
                label=label
            ))

        return segments

    def extract_all(self) -> List[PAFSegment]:
        """Extract segments from all records."""
        records = self.get_record_ids()
        print(f"Found {len(records['normal'])} normal, {len(records['preaf'])} pre-AF records")

        all_segments = []

        for record_id in tqdm(records['normal'], desc="Normal"):
            try:
                segments = self.extract_segments_from_record(record_id, label=0)
                all_segments.extend(segments)
            except Exception as e:
                print(f"Error processing {record_id}: {e}")

        for record_id in tqdm(records['preaf'], desc="Pre-AF"):
            try:
                segments = self.extract_segments_from_record(record_id, label=1)
                all_segments.extend(segments)
            except Exception as e:
                print(f"Error processing {record_id}: {e}")

        return all_segments


    def _clean_segments(self, segments: List[PAFSegment], cleaner: DataCleaner, desc: str) -> List[PAFSegment]:
        cleaned = []
        rejected = 0
        for seg in tqdm(segments, desc=desc):
            sig, info = cleaner.preprocess(seg.signal, check_quality=True)
            if sig is not None:
                seg.signal = sig
                cleaned.append(seg)
            else:
                rejected += 1
        print(f"{desc}: {len(cleaned)} passed, {rejected} rejected")
        return cleaned

    def export_to_hdf5(self) -> Dict:
        segments = self.extract_all()
        patient_split = self.create_patient_split()

        train_segs = [s for s in segments if s.patient_id in patient_split['train']]
        val_segs = [s for s in segments if s.patient_id in patient_split['val']]
        test_segs = [s for s in segments if s.patient_id in patient_split['test']]

        print(f"Segments: {len(train_segs)} train, {len(val_segs)} val, {len(test_segs)} test")

        cleaner = DataCleaner(fs=ORIGINAL_FS, notch_freq=60.0)
        train_segs = self._clean_segments(train_segs, cleaner, "Cleaning train")
        val_segs = self._clean_segments(val_segs, cleaner, "Cleaning val")
        test_segs = self._clean_segments(test_segs, cleaner, "Cleaning test")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / 'paf_segments.h5'

        def save_split(f: h5py.File, name: str, segs: List[PAFSegment]):
            if not segs:
                return

            grp = f.create_group(name)
            signals = np.stack([s.signal for s in segs])
            grp.create_dataset('signals', data=signals, dtype='float32',
                               compression='gzip', compression_opts=4)
            grp.create_dataset('labels', data=np.array([s.label for s in segs], dtype=np.int8))

            dt_str = h5py.special_dtype(vlen=str)
            grp.create_dataset('segment_ids', data=[s.segment_id for s in segs], dtype=dt_str)
            grp.create_dataset('patient_ids', data=[s.patient_id for s in segs], dtype=dt_str)
            grp.create_dataset('record_ids', data=np.array([s.record_id for s in segs], dtype=h5py.special_dtype(vlen=str)))

            grp.attrs['n_segments'] = len(segs)
            grp.attrs['n_normal'] = sum(1 for s in segs if s.label == 0)
            grp.attrs['n_preaf'] = sum(1 for s in segs if s.label == 1)
            grp.attrs['n_patients'] = len(set(s.patient_id for s in segs))


        with h5py.File(output_path, 'w') as f:
            save_split(f, 'train', train_segs)
            save_split(f, 'val', val_segs)
            save_split(f, 'test', test_segs)

            f.attrs['segment_duration_sec'] = TRIMMED_DURATION_SEC
            f.attrs['sampling_freq'] = TARGET_FS
            f.attrs['lead_index'] = LEAD_INDEX
            f.attrs['train_patients'] = list(patient_split['train'])
            f.attrs['val_patients'] = list(patient_split['val'])
            f.attrs['test_patients'] = list(patient_split['test'])

        summary = {
            'train_segments': len(train_segs),
            'val_segments': len(val_segs),
            'test_segments': len(test_segs),
            'train_normal': sum(1 for s in train_segs if s.label == 0),
            'train_preaf': sum(1 for s in train_segs if s.label == 1),
            'train_patients': len(patient_split['train']),
            'val_patients': len(patient_split['val']),
            'test_patients': len(patient_split['test']),
        }

        print(f"\nSaved to {output_path}")
        print(f"  Train: {summary['train_segments']} segments ({summary['train_normal']} normal, {summary['train_preaf']} pre-AF) from {summary['train_patients']} patients")
        print(f"  Val: {summary['val_segments']} segments from {summary['val_patients']} patients")
        print(f"  Test: {summary['test_segments']} segments from {summary['test_patients']} patients")
        print(f"  Signal: {TRIMMED_DURATION_SEC}s @ {TARGET_FS} Hz = {TRIMMED_DURATION_SEC * TARGET_FS} samples")

        return summary

    def export_all_for_rqa(self, path) -> Dict:
        """Export all records without train/val/test split, for RQA statistical analysis."""
        segments = self.extract_all()

        cleaner = DataCleaner(fs=ORIGINAL_FS, notch_freq=60.0)
        cleaned = self._clean_segments(segments, cleaner, "Cleaning all segments") #TODO removed cleaning
        # cleaned = segments

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / path

        n_normal = sum(1 for s in cleaned if s.label == 0)
        n_preaf = sum(1 for s in cleaned if s.label == 1)

        with h5py.File(output_path, 'w') as f:
            grp = f.create_group('all')
            signals = np.stack([s.signal for s in cleaned])
            grp.create_dataset('signals', data=signals, dtype='float32',
                               compression='gzip', compression_opts=4)
            grp.create_dataset('labels',
                               data=np.array([s.label for s in cleaned], dtype=np.int8))
            dt_str = h5py.special_dtype(vlen=str)
            grp.create_dataset('segment_ids',
                               data=[s.segment_id for s in cleaned], dtype=dt_str)
            grp.create_dataset('patient_ids',
                               data=[s.patient_id for s in cleaned], dtype=dt_str)
            grp.create_dataset('record_ids',
                               data=[s.record_id for s in cleaned], dtype=dt_str)
            grp.attrs['n_normal'] = n_normal
            grp.attrs['n_preaf'] = n_preaf

        print(f"Saved {len(cleaned)} segments: {n_normal} normal, {n_preaf} pre-AF to {output_path}" )
        return output_path

    def visualize_segments(
        self,
        segments: Optional[List[PAFSegment]] = None,
        n_normal: int = 3,
        n_preaf: int = 3,
        save_path: Optional[Path] = None
    ):
        """Visualize sample segments from normal and pre-AF classes."""
        import matplotlib.pyplot as plt

        if segments is None:
            records = self.get_record_ids()
            segments = []

            for record_id in records['normal'][:n_normal]:
                segs = self.extract_segments_from_record(record_id, label=0, num_segments=1)
                segments.extend(segs)

            for record_id in records['preaf'][:n_preaf]:
                segs = self.extract_segments_from_record(record_id, label=1, num_segments=1)
                segments.extend(segs)

        normal_segs = [s for s in segments if s.label == 0][:n_normal]
        preaf_segs = [s for s in segments if s.label == 1][:n_preaf]

        n_rows = max(len(normal_segs), len(preaf_segs))
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 10))

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('PAF Dataset: Normal vs Pre-AF Segments', fontsize=14, fontweight='bold')

        for i, seg in enumerate(normal_segs):
            ax = axes[i, 0]
            time = np.arange(len(seg.signal)) / seg.sampling_freq
            ax.plot(time, seg.signal, 'b-', linewidth=0.5)
            ax.set_ylabel('mV')
            ax.grid(True, alpha=0.3)
            title = f'NORMAL (label=0)\n' if i == 0 else ''
            ax.set_title(f'{title}{seg.record_id} (start: {seg.segment_start_sec:.0f}s)', fontsize=10)

        for i, seg in enumerate(preaf_segs):
            ax = axes[i, 1]
            time = np.arange(len(seg.signal)) / seg.sampling_freq
            ax.plot(time, seg.signal, 'r-', linewidth=0.5)
            ax.grid(True, alpha=0.3)
            title = f'PRE-AF (label=1)\n' if i == 0 else ''
            ax.set_title(f'{title}{seg.record_id} (start: {seg.segment_start_sec:.0f}s)', fontsize=10)

        for i in range(len(normal_segs), n_rows):
            axes[i, 0].set_visible(False)
        for i in range(len(preaf_segs), n_rows):
            axes[i, 1].set_visible(False)

        for ax in axes[-1, :]:
            if ax.get_visible():
                ax.set_xlabel('Time (s)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        else:
            plt.show()

        return fig


def load_paf_hdf5(split: str = 'train', hdf5_path: Optional[Path] = None):
    # Use provided path or default based on split
    if hdf5_path is not None:
        path = Path(hdf5_path)
    elif split == 'all':
        path = OUTPUT_DIR / 'paf_segments_all.h5'
    else:
        path = OUTPUT_DIR / 'paf_segments.h5'

    with h5py.File(path, 'r') as f:
        grp = f[split]
        signals = grp['signals'][:]
        labels = grp['labels'][:]
        segment_ids = [s.decode() if isinstance(s, bytes) else s for s in grp['segment_ids'][:]]

        # patient_ids may not exist in 'all' file
        if 'patient_ids' in grp:
            patient_ids = [s.decode() if isinstance(s, bytes) else s for s in grp['patient_ids'][:]]
        else:
            patient_ids = [''] * len(signals)

        record_ids = [s.decode() if isinstance(s, bytes) else s for s in grp['record_ids'][:]]

    return signals, labels, segment_ids, patient_ids, record_ids




def export_hdf5_to_csv(split: str = 'train', output_path: Optional[Path] = None, hdf5_path: Optional[Path] = None):
    """
    Export HDF5 data to CSV format for parameter analysis.

    CSV columns: segment_id, patient_id, record_id, label, s0, s1, ..., sN
    """
    import pandas as pd

    signals, labels, segment_ids, patient_ids, record_ids = load_paf_hdf5(split, hdf5_path=hdf5_path)

    if output_path is None:
        output_path = OUTPUT_DIR / f'paf_{split}.csv'

    rows = []
    for i in range(len(signals)):
        row = {
            'segment_id': segment_ids[i],
            'patient_id': patient_ids[i],
            'record_id': record_ids[i],
            'label': labels[i],
        }
        for j, val in enumerate(signals[i]):
            row[f's{j}'] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} segments to {output_path}")
    return df


if __name__ == '__main__':
    hdf5_path = OUTPUT_DIR / "paf_segments_all_clean_new_cleaning_min_max_with_upper.h5"
    csv_path = OUTPUT_DIR / "paf_segments_all_clean_new_cleaning_min_max_with_upper.csv"

    extractor = PAFExtractor()
    # export_all_for_rqa saves to HDF5, not CSV
    extractor.export_all_for_rqa(path=hdf5_path)

    # Now export the HDF5 to CSV
    export_hdf5_to_csv(split='all', output_path=csv_path, hdf5_path=hdf5_path)
