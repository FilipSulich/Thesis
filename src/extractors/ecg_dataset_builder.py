"""
Unified ECG Dataset Builder for AF Prediction

Extracts segments from:
- MIT-BIH AFDB (250Hz) - pre-AF and normal segments
- LTAFDB (128Hz -> 250Hz) - pre-AF and normal segments
- PTB-XL (500Hz -> 250Hz) - normal segments only

Output: Single HDF5 file with patient-wise train/val/test splits
"""

import wfdb
import numpy as np
import h5py
from pathlib import Path
from scipy.signal import resample
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import ast
import pandas as pd

# Configuration
SEGMENT_DURATION_SEC = 12
TARGET_FS = 250
SEGMENT_SAMPLES = SEGMENT_DURATION_SEC * TARGET_FS  # 2500 samples
LEAD_TIMES_MIN = (5, 10, 15, 30, 45)
MIN_DISTANCE_FROM_AF_MIN = 60  # Normal segments must be 5+ min away from any AF
RANDOM_SEED = 42


@dataclass
class Segment:
    """A single ECG segment."""
    signal: np.ndarray
    label: int  # 0=normal, 1=pre-AF
    lead_time: int  # minutes before AF onset (0 for normal)
    patient_id: str
    source_dataset: str
    segment_id: str


def resample_signal(signal: np.ndarray, original_fs: int, target_fs: int = TARGET_FS) -> np.ndarray:
    """Resample signal to target frequency."""
    if original_fs == target_fs:
        return signal
    target_samples = int(len(signal) * target_fs / original_fs)
    return resample(signal, target_samples).astype(np.float32)


# =============================================================================
# AFDB Extraction
# =============================================================================

def get_afdb_records(data_dir: Path) -> List[str]:
    """Get list of AFDB record IDs."""
    return sorted([f.stem for f in data_dir.glob('*.hea')])


def parse_afdb_rhythm_annotations(data_dir: Path, record_id: str) -> List[Dict]:
    """Parse rhythm annotations from AFDB record."""
    ann = wfdb.rdann(str(data_dir / record_id), 'atr')
    fs = 250  # AFDB is 250Hz

    changes = []
    for sample, aux in zip(ann.sample, ann.aux_note):
        if aux and aux.startswith('('):
            rhythm = aux.strip('\x00')
            changes.append({
                'sample': sample,
                'time_sec': sample / fs,
                'rhythm': rhythm
            })
    return changes


def find_af_onsets_afdb(rhythm_changes: List[Dict], fs: int = 250) -> List[Dict]:
    """Find AF episode onsets with preceding rhythm info."""
    af_onsets = []

    for i, change in enumerate(rhythm_changes):
        if change['rhythm'] == '(AFIB':
            # Get preceding rhythm duration
            if i > 0:
                prev = rhythm_changes[i-1]
                preceding_duration = (change['sample'] - prev['sample']) / fs
                preceding_rhythm = prev['rhythm']
            else:
                preceding_duration = change['sample'] / fs
                preceding_rhythm = None

            af_onsets.append({
                'onset_sample': change['sample'],
                'onset_time_sec': change['time_sec'],
                'preceding_duration_sec': preceding_duration,
                'preceding_rhythm': preceding_rhythm,
                'is_from_normal': preceding_rhythm == '(N'
            })

    return af_onsets


def find_normal_periods_afdb(rhythm_changes: List[Dict], af_onsets: List[Dict],
                              record_duration_sec: float, fs: int = 250) -> List[Tuple[float, float]]:
    """Find normal sinus rhythm periods that are at least 5 min away from any AF."""
    min_distance_sec = MIN_DISTANCE_FROM_AF_MIN * 60

    # Build list of AF episode time ranges (with buffer)
    af_ranges = []
    for onset in af_onsets:
        # Find when this AF episode ends
        onset_time = onset['onset_time_sec']
        # Add buffer before and after
        af_ranges.append((onset_time - min_distance_sec, onset_time + min_distance_sec))

    # Find normal periods
    normal_periods = []
    for i, change in enumerate(rhythm_changes):
        if change['rhythm'] == '(N':
            start_time = change['time_sec']
            # Find end of this normal period
            if i + 1 < len(rhythm_changes):
                end_time = rhythm_changes[i + 1]['time_sec']
            else:
                end_time = record_duration_sec

            # Check if this period is far enough from any AF
            safe_start = start_time
            safe_end = end_time

            for af_start, af_end in af_ranges:
                if safe_start < af_end and safe_end > af_start:
                    # Overlap - adjust the safe region
                    if safe_start >= af_start:
                        safe_start = max(safe_start, af_end)
                    if safe_end <= af_end:
                        safe_end = min(safe_end, af_start)

            if safe_end - safe_start >= SEGMENT_DURATION_SEC:
                normal_periods.append((safe_start, safe_end))

    return normal_periods


def extract_afdb_segments(data_dir: Path, verbose: bool = True) -> Tuple[List[Segment], List[Segment]]:
    """Extract pre-AF and normal segments from AFDB."""
    records = get_afdb_records(data_dir)
    pre_af_segments = []
    normal_segments = []
    patients_with_preaf = set()

    for record_id in records:
        try:
            record = wfdb.rdrecord(str(data_dir / record_id))
            fs = record.fs
            duration_sec = record.sig_len / fs
            signal = record.p_signal[:, 0].astype(np.float32)  # Lead 0

            patient_id = f"afdb_{record_id}"
            rhythm_changes = parse_afdb_rhythm_annotations(data_dir, record_id)
            af_onsets = find_af_onsets_afdb(rhythm_changes, fs)

            # Extract pre-AF segments
            for ep_idx, onset in enumerate(af_onsets):
                if not onset['is_from_normal']:
                    continue

                for lead_time in LEAD_TIMES_MIN:
                    lead_time_sec = lead_time * 60

                    # Segment ends at lead_time before AF onset
                    seg_end_sec = onset['onset_time_sec'] - lead_time_sec
                    seg_start_sec = seg_end_sec - SEGMENT_DURATION_SEC

                    # Validate: segment must be within bounds and preceding rhythm
                    if seg_start_sec < 0:
                        continue
                    if onset['preceding_duration_sec'] < (lead_time_sec + SEGMENT_DURATION_SEC):
                        continue

                    # Check for overlap with previous AF episodes
                    overlaps = False
                    for prev_onset in af_onsets[:ep_idx]:
                        if seg_start_sec < prev_onset['onset_time_sec']:
                            overlaps = True
                            break
                    if overlaps:
                        continue

                    # Extract segment
                    start_sample = int(seg_start_sec * fs)
                    end_sample = int(seg_end_sec * fs)
                    seg_signal = signal[start_sample:end_sample]

                    # Resample if needed
                    if fs != TARGET_FS:
                        seg_signal = resample_signal(seg_signal, fs)

                    # Ensure exact length
                    if len(seg_signal) > SEGMENT_SAMPLES:
                        seg_signal = seg_signal[:SEGMENT_SAMPLES]
                    elif len(seg_signal) < SEGMENT_SAMPLES:
                        seg_signal = np.pad(seg_signal, (0, SEGMENT_SAMPLES - len(seg_signal)), mode='edge')

                    pre_af_segments.append(Segment(
                        signal=seg_signal,
                        label=1,
                        lead_time=lead_time,
                        patient_id=patient_id,
                        source_dataset='afdb',
                        segment_id=f"afdb_{record_id}_ep{ep_idx}_lt{lead_time}"
                    ))
                    patients_with_preaf.add(patient_id)

            # Extract normal segments (only from patients with pre-AF segments)
            if patient_id in patients_with_preaf:
                normal_periods = find_normal_periods_afdb(rhythm_changes, af_onsets, duration_sec, fs)

                for period_idx, (start_sec, end_sec) in enumerate(normal_periods):
                    # Extract as many non-overlapping segments as possible
                    current_start = start_sec
                    seg_count = 0
                    while current_start + SEGMENT_DURATION_SEC <= end_sec:
                        start_sample = int(current_start * fs)
                        end_sample = int((current_start + SEGMENT_DURATION_SEC) * fs)
                        seg_signal = signal[start_sample:end_sample]

                        if fs != TARGET_FS:
                            seg_signal = resample_signal(seg_signal, fs)

                        if len(seg_signal) > SEGMENT_SAMPLES:
                            seg_signal = seg_signal[:SEGMENT_SAMPLES]
                        elif len(seg_signal) < SEGMENT_SAMPLES:
                            seg_signal = np.pad(seg_signal, (0, SEGMENT_SAMPLES - len(seg_signal)), mode='edge')

                        normal_segments.append(Segment(
                            signal=seg_signal,
                            label=0,
                            lead_time=0,
                            patient_id=patient_id,
                            source_dataset='afdb',
                            segment_id=f"afdb_{record_id}_normal_p{period_idx}_s{seg_count}"
                        ))
                        current_start += SEGMENT_DURATION_SEC
                        seg_count += 1

        except Exception as e:
            if verbose:
                print(f"Error processing AFDB {record_id}: {e}")

    if verbose:
        print(f"AFDB: {len(pre_af_segments)} pre-AF, {len(normal_segments)} normal segments")
        print(f"      {len(patients_with_preaf)} patients with pre-AF data")

    return pre_af_segments, normal_segments


# =============================================================================
# LTAFDB Extraction
# =============================================================================

def get_ltafdb_records(data_dir: Path) -> List[str]:
    """Get list of LTAFDB record IDs."""
    records_file = data_dir / 'RECORDS'
    if records_file.exists():
        return [line.strip() for line in records_file.read_text().splitlines() if line.strip()]
    return sorted([f.stem for f in data_dir.glob('*.hea')])


def parse_ltafdb_rhythm_annotations(data_dir: Path, record_id: str, fs: int = 128) -> List[Dict]:
    """Parse rhythm annotations from LTAFDB record."""
    ann = wfdb.rdann(str(data_dir / record_id), 'atr')

    changes = []
    for sample, aux in zip(ann.sample, ann.aux_note):
        if aux and aux.startswith('('):
            rhythm = aux.strip('\x00')
            changes.append({
                'sample': sample,
                'time_sec': sample / fs,
                'rhythm': rhythm
            })
    return changes


def find_af_onsets_ltafdb(rhythm_changes: List[Dict], fs: int = 128) -> List[Dict]:
    """Find AF episode onsets in LTAFDB."""
    af_onsets = []

    for i, change in enumerate(rhythm_changes):
        if change['rhythm'] == '(AFIB':
            if i > 0:
                prev = rhythm_changes[i-1]
                preceding_duration = (change['sample'] - prev['sample']) / fs
                preceding_rhythm = prev['rhythm']
            else:
                preceding_duration = change['sample'] / fs
                preceding_rhythm = None

            af_onsets.append({
                'onset_sample': change['sample'],
                'onset_time_sec': change['time_sec'],
                'preceding_duration_sec': preceding_duration,
                'preceding_rhythm': preceding_rhythm,
                'is_from_normal': preceding_rhythm == '(N'
            })

    return af_onsets


def find_normal_periods_ltafdb(rhythm_changes: List[Dict], af_onsets: List[Dict],
                                record_duration_sec: float, fs: int = 128) -> List[Tuple[float, float]]:
    """Find normal periods in LTAFDB at least 5 min from any AF."""
    min_distance_sec = MIN_DISTANCE_FROM_AF_MIN * 60

    af_ranges = []
    for onset in af_onsets:
        onset_time = onset['onset_time_sec']
        af_ranges.append((onset_time - min_distance_sec, onset_time + min_distance_sec))

    normal_periods = []
    for i, change in enumerate(rhythm_changes):
        if change['rhythm'] == '(N':
            start_time = change['time_sec']
            if i + 1 < len(rhythm_changes):
                end_time = rhythm_changes[i + 1]['time_sec']
            else:
                end_time = record_duration_sec

            safe_start = start_time
            safe_end = end_time

            for af_start, af_end in af_ranges:
                if safe_start < af_end and safe_end > af_start:
                    if safe_start >= af_start:
                        safe_start = max(safe_start, af_end)
                    if safe_end <= af_end:
                        safe_end = min(safe_end, af_start)

            if safe_end - safe_start >= SEGMENT_DURATION_SEC:
                normal_periods.append((safe_start, safe_end))

    return normal_periods


def extract_ltafdb_segments(data_dir: Path, verbose: bool = True) -> Tuple[List[Segment], List[Segment]]:
    """Extract pre-AF and normal segments from LTAFDB."""
    records = get_ltafdb_records(data_dir)
    pre_af_segments = []
    normal_segments = []
    patients_with_preaf = set()

    for record_id in records:
        try:
            record = wfdb.rdrecord(str(data_dir / record_id))
            fs = record.fs  # 128Hz
            duration_sec = record.sig_len / fs
            signal = record.p_signal[:, 0].astype(np.float32)

            patient_id = f"ltafdb_{record_id}"
            rhythm_changes = parse_ltafdb_rhythm_annotations(data_dir, record_id, fs)
            af_onsets = find_af_onsets_ltafdb(rhythm_changes, fs)

            # Extract pre-AF segments
            for ep_idx, onset in enumerate(af_onsets):
                if not onset['is_from_normal']:
                    continue

                for lead_time in LEAD_TIMES_MIN:
                    lead_time_sec = lead_time * 60

                    seg_end_sec = onset['onset_time_sec'] - lead_time_sec
                    seg_start_sec = seg_end_sec - SEGMENT_DURATION_SEC

                    if seg_start_sec < 0:
                        continue
                    if onset['preceding_duration_sec'] < (lead_time_sec + SEGMENT_DURATION_SEC):
                        continue

                    # Check overlap with previous AF
                    overlaps = False
                    for prev_onset in af_onsets[:ep_idx]:
                        if seg_start_sec < prev_onset['onset_time_sec']:
                            overlaps = True
                            break
                    if overlaps:
                        continue

                    start_sample = int(seg_start_sec * fs)
                    end_sample = int(seg_end_sec * fs)
                    seg_signal = signal[start_sample:end_sample]

                    # Resample 128Hz -> 250Hz
                    seg_signal = resample_signal(seg_signal, fs, TARGET_FS)

                    if len(seg_signal) > SEGMENT_SAMPLES:
                        seg_signal = seg_signal[:SEGMENT_SAMPLES]
                    elif len(seg_signal) < SEGMENT_SAMPLES:
                        seg_signal = np.pad(seg_signal, (0, SEGMENT_SAMPLES - len(seg_signal)), mode='edge')

                    pre_af_segments.append(Segment(
                        signal=seg_signal,
                        label=1,
                        lead_time=lead_time,
                        patient_id=patient_id,
                        source_dataset='ltafdb',
                        segment_id=f"ltafdb_{record_id}_ep{ep_idx}_lt{lead_time}"
                    ))
                    patients_with_preaf.add(patient_id)

            # Extract normal segments from patients with pre-AF
            if patient_id in patients_with_preaf:
                normal_periods = find_normal_periods_ltafdb(rhythm_changes, af_onsets, duration_sec, fs)

                for period_idx, (start_sec, end_sec) in enumerate(normal_periods):
                    current_start = start_sec
                    seg_count = 0
                    while current_start + SEGMENT_DURATION_SEC <= end_sec:
                        start_sample = int(current_start * fs)
                        end_sample = int((current_start + SEGMENT_DURATION_SEC) * fs)
                        seg_signal = signal[start_sample:end_sample]

                        seg_signal = resample_signal(seg_signal, fs, TARGET_FS)

                        if len(seg_signal) > SEGMENT_SAMPLES:
                            seg_signal = seg_signal[:SEGMENT_SAMPLES]
                        elif len(seg_signal) < SEGMENT_SAMPLES:
                            seg_signal = np.pad(seg_signal, (0, SEGMENT_SAMPLES - len(seg_signal)), mode='edge')

                        normal_segments.append(Segment(
                            signal=seg_signal,
                            label=0,
                            lead_time=0,
                            patient_id=patient_id,
                            source_dataset='ltafdb',
                            segment_id=f"ltafdb_{record_id}_normal_p{period_idx}_s{seg_count}"
                        ))
                        current_start += SEGMENT_DURATION_SEC
                        seg_count += 1

        except Exception as e:
            if verbose:
                print(f"Error processing LTAFDB {record_id}: {e}")

    if verbose:
        print(f"LTAFDB: {len(pre_af_segments)} pre-AF, {len(normal_segments)} normal segments")
        print(f"        {len(patients_with_preaf)} patients with pre-AF data")

    return pre_af_segments, normal_segments


# =============================================================================
# PTB-XL Extraction
# =============================================================================

def extract_ptbxl_segments(data_dir: Path, verbose: bool = True) -> List[Segment]:
    """Extract normal segments from PTB-XL."""
    csv_path = data_dir / 'ptbxl_database.csv'
    df = pd.read_csv(csv_path, index_col='ecg_id')
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

    # Filter for NORM with high confidence
    def has_norm(scp_codes):
        return 'NORM' in scp_codes and scp_codes['NORM'] >= 80.0

    norm_df = df[df['scp_codes'].apply(has_norm)]

    segments = []
    for ecg_id, row in norm_df.iterrows():
        try:
            # Use 500Hz version
            filename = row['filename_hr']
            record = wfdb.rdrecord(str(data_dir / filename))
            fs = record.fs  # 500Hz
            signal = record.p_signal[:, 1].astype(np.float32)  # Lead II (index 1)

            # Resample 500Hz -> 250Hz
            signal = resample_signal(signal, fs, TARGET_FS)

            if len(signal) > SEGMENT_SAMPLES:
                signal = signal[:SEGMENT_SAMPLES]
            elif len(signal) < SEGMENT_SAMPLES:
                signal = np.pad(signal, (0, SEGMENT_SAMPLES - len(signal)), mode='edge')

            patient_id = f"ptbxl_{int(row['patient_id']):05d}"

            segments.append(Segment(
                signal=signal,
                label=0,
                lead_time=0,
                patient_id=patient_id,
                source_dataset='ptbxl',
                segment_id=f"ptbxl_{ecg_id}"
            ))

        except Exception as e:
            if verbose:
                print(f"Error processing PTB-XL {ecg_id}: {e}")

    if verbose:
        print(f"PTB-XL: {len(segments)} normal segments from {len(set(s.patient_id for s in segments))} patients")

    return segments


# =============================================================================
# Patient-wise Splitting
# =============================================================================

def patient_wise_split(segments: List[Segment], train_ratio: float = 0.7,
                       val_ratio: float = 0.15, seed: int = RANDOM_SEED) -> Dict[str, List[Segment]]:
    """Split segments by patient into train/val/test."""
    np.random.seed(seed)

    patients = list(set(s.patient_id for s in segments))
    np.random.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])

    return {
        'train': [s for s in segments if s.patient_id in train_patients],
        'val': [s for s in segments if s.patient_id in val_patients],
        'test': [s for s in segments if s.patient_id in test_patients]
    }


def balance_normal_segments(pre_af_segments: List[Segment], normal_segments: List[Segment],
                            seed: int = RANDOM_SEED) -> List[Segment]:
    """Sample normal segments to match pre-AF count per lead time."""
    np.random.seed(seed)

    # Count pre-AF per lead time
    preaf_by_leadtime = {}
    for lt in LEAD_TIMES_MIN:
        preaf_by_leadtime[lt] = sum(1 for s in pre_af_segments if s.lead_time == lt)

    total_preaf = sum(preaf_by_leadtime.values())

    # Sample normal segments
    if len(normal_segments) >= total_preaf:
        indices = np.random.choice(len(normal_segments), total_preaf, replace=False)
    else:
        indices = np.random.choice(len(normal_segments), total_preaf, replace=True)

    return [normal_segments[i] for i in indices]


# =============================================================================
# Main Pipeline
# =============================================================================

def build_dataset(afdb_dir: Path, ltafdb_dir: Path, ptbxl_dir: Path,
                  output_path: Path, verbose: bool = True):
    """Build the unified dataset."""

    print("=" * 70)
    print("ECG Dataset Builder for AF Prediction")
    print("=" * 70)

    # Extract from each dataset
    print("\n[1/5] Extracting from AFDB...")
    afdb_preaf, afdb_normal = extract_afdb_segments(afdb_dir, verbose)

    print("\n[2/5] Extracting from LTAFDB...")
    ltafdb_preaf, ltafdb_normal = extract_ltafdb_segments(ltafdb_dir, verbose)

    print("\n[3/5] Extracting from PTB-XL...")
    ptbxl_normal = extract_ptbxl_segments(ptbxl_dir, verbose)

    # Verify no patient overlap between AFDB and LTAFDB
    afdb_patients = set(s.patient_id for s in afdb_preaf + afdb_normal)
    ltafdb_patients = set(s.patient_id for s in ltafdb_preaf + ltafdb_normal)
    overlap = afdb_patients & ltafdb_patients
    if overlap:
        print(f"WARNING: Patient overlap between AFDB and LTAFDB: {overlap}")
    else:
        print("\nNo patient overlap between AFDB and LTAFDB")

    # Combine all pre-AF and normal segments
    all_preaf = afdb_preaf + ltafdb_preaf
    all_normal = afdb_normal + ltafdb_normal + ptbxl_normal

    print(f"\nTotal pre-AF segments: {len(all_preaf)}")
    print(f"Total normal segments before balancing: {len(all_normal)}")

    # Split each dataset independently, then combine
    print("\n[4/5] Patient-wise splitting...")

    # Split AFDB
    afdb_all = afdb_preaf + afdb_normal
    afdb_splits = patient_wise_split(afdb_all)

    # Split LTAFDB
    ltafdb_all = ltafdb_preaf + ltafdb_normal
    ltafdb_splits = patient_wise_split(ltafdb_all)

    # Split PTB-XL
    ptbxl_splits = patient_wise_split(ptbxl_normal)

    # Combine splits
    combined_splits = {
        'train': afdb_splits['train'] + ltafdb_splits['train'] + ptbxl_splits['train'],
        'val': afdb_splits['val'] + ltafdb_splits['val'] + ptbxl_splits['val'],
        'test': afdb_splits['test'] + ltafdb_splits['test'] + ptbxl_splits['test']
    }

    # Balance normal segments per split per lead time
    print("\n[5/5] Balancing normal segments per lead time...")

    for split_name in ['train', 'val', 'test']:
        split_segs = combined_splits[split_name]
        preaf_segs = [s for s in split_segs if s.label == 1]
        normal_segs = [s for s in split_segs if s.label == 0]

        if len(preaf_segs) > 0 and len(normal_segs) > 0:
            balanced_normal = balance_normal_segments(preaf_segs, normal_segs)
            combined_splits[split_name] = preaf_segs + balanced_normal

    # Save to HDF5
    print(f"\nSaving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        for split_name, segments in combined_splits.items():
            if not segments:
                continue

            grp = f.create_group(split_name)

            signals = np.stack([s.signal for s in segments])
            grp.create_dataset('signals', data=signals, dtype='float32',
                             compression='gzip', compression_opts=4)
            grp.create_dataset('labels', data=np.array([s.label for s in segments], dtype=np.int8))
            grp.create_dataset('lead_times', data=np.array([s.lead_time for s in segments], dtype=np.int16))

            dt_str = h5py.special_dtype(vlen=str)
            grp.create_dataset('patient_ids', data=[s.patient_id for s in segments], dtype=dt_str)
            grp.create_dataset('source_datasets', data=[s.source_dataset for s in segments], dtype=dt_str)
            grp.create_dataset('segment_ids', data=[s.segment_id for s in segments], dtype=dt_str)

            grp.attrs['n_segments'] = len(segments)
            grp.attrs['n_patients'] = len(set(s.patient_id for s in segments))

        f.attrs['sampling_rate'] = TARGET_FS
        f.attrs['segment_duration_sec'] = SEGMENT_DURATION_SEC
        f.attrs['segment_samples'] = SEGMENT_SAMPLES
        f.attrs['lead_times_min'] = list(LEAD_TIMES_MIN)

    # Print summary
    print_summary(combined_splits)


def print_summary(splits: Dict[str, List[Segment]]):
    """Print dataset summary."""
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    for split_name in ['train', 'val', 'test']:
        segments = splits.get(split_name, [])
        if not segments:
            continue

        preaf = [s for s in segments if s.label == 1]
        normal = [s for s in segments if s.label == 0]

        print(f"\n{split_name.upper()}:")
        print(f"  Total segments: {len(segments)}")
        print(f"  Pre-AF: {len(preaf)}, Normal: {len(normal)}")
        print(f"  Patients: {len(set(s.patient_id for s in segments))}")

        # By dataset
        by_dataset = {}
        for s in segments:
            by_dataset.setdefault(s.source_dataset, {'preaf': 0, 'normal': 0, 'patients': set()})
            if s.label == 1:
                by_dataset[s.source_dataset]['preaf'] += 1
            else:
                by_dataset[s.source_dataset]['normal'] += 1
            by_dataset[s.source_dataset]['patients'].add(s.patient_id)

        print(f"  By dataset:")
        for ds, counts in sorted(by_dataset.items()):
            print(f"    {ds}: {counts['preaf']} pre-AF, {counts['normal']} normal, {len(counts['patients'])} patients")

        # By lead time (pre-AF only)
        print(f"  Pre-AF by lead time:")
        for lt in LEAD_TIMES_MIN:
            count = sum(1 for s in preaf if s.lead_time == lt)
            print(f"    {lt} min: {count}")

    # Total
    all_segs = splits['train'] + splits['val'] + splits['test']
    print(f"\nTOTAL:")
    print(f"  Segments: {len(all_segs)}")
    print(f"  Patients: {len(set(s.patient_id for s in all_segs))}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build unified ECG dataset for AF prediction')
    parser.add_argument('--afdb_dir', default='/Users/filipsulich/Developer/uni/Thesis/Thesis/data/MIT')
    parser.add_argument('--ltafdb_dir', default='/Users/filipsulich/Developer/uni/Thesis/Thesis/data/ltafdb')
    parser.add_argument('--ptbxl_dir', default='/Users/filipsulich/Developer/uni/Thesis/Thesis/data/ptb-xl')
    parser.add_argument('--output', default='/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/unified_af_dataset.h5')

    args = parser.parse_args()

    build_dataset(
        afdb_dir=Path(args.afdb_dir),
        ltafdb_dir=Path(args.ltafdb_dir),
        ptbxl_dir=Path(args.ptbxl_dir),
        output_path=Path(args.output)
    )