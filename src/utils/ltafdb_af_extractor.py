"""
LTAFDB (Long Term AF Database) Pre-AF Segment Extractor

Extracts 10-second ECG segments at specific lead times (5, 10, 30, 45 min)
before AF onset, filtering for direct Normal→AF transitions.

LTAFDB characteristics:
- 84 records of long-term ECG (20-26 hours each)
- Sampling frequency: 128 Hz
- 2 ECG leads per record
- Each record = one unique patient

Compatible with MIT-BIH AFDB extractor for dataset merging.

Usage:
    python ltafdb_af_extractor.py --data_dir /path/to/ltafdb --output_dir /path/to/output
"""

import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
import h5py
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

class RhythmType(Enum):
    """Rhythm types found in LTAFDB annotations."""
    NORMAL = "(N"
    AFIB = "(AFIB"
    AFL = "(AFL"
    VT = "(VT"           # Ventricular tachycardia
    JUNCTIONAL = "(J"
    BIGEMINY = "(B"
    TRIGEMINY = "(T"
    SVTA = "(SVTA"       # Supraventricular tachyarrhythmia
    SBR = "(SBR"         # Sinus bradycardia
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_annotation(cls, aux_note: str) -> "RhythmType":
        """Parse rhythm type from annotation aux_note."""
        if not aux_note:
            return cls.UNKNOWN
        cleaned = aux_note.strip('\x00')
        for rhythm in cls:
            if rhythm.value == cleaned:
                return rhythm
        return cls.UNKNOWN


@dataclass
class SegmentConfig:
    """Configuration for segment extraction."""
    segment_duration_sec: int = 10  # 10-second segments
    lead_times_min: Tuple[int, ...] = (5, 10, 30, 45)  # Minutes before AF onset
    target_fs: int = 128  # LTAFDB native frequency
    lead_indices: Tuple[int, ...] = (0,)  # Which leads to extract
    require_normal_preceding: bool = True  # Only Normal→AF transitions


@dataclass
class PatientInfo:
    """Patient information from LTAFDB.

    In LTAFDB, each record corresponds to a unique patient.
    """
    patient_id: str
    source_database: str
    record_id: str
    sampling_freq: int
    duration_sec: float
    num_leads: int
    lead_names: List[str]


@dataclass
class AFEpisode:
    """Information about a single AF episode."""
    episode_index: int
    onset_sample: int
    onset_time_sec: float
    end_sample: Optional[int]
    end_time_sec: Optional[float]
    duration_sec: Optional[float]
    preceding_rhythm: RhythmType
    preceding_duration_sec: float
    following_rhythm: RhythmType
    is_valid_transition: bool  # True if Normal→AF direct transition


@dataclass
class ExtractableSegment:
    """Metadata for a segment that can be extracted."""
    segment_id: str
    patient_id: str
    source_database: str
    record_id: str
    episode_index: int
    lead_time_min: int
    af_onset_sample: int
    af_onset_time_sec: float
    segment_start_sample: int
    segment_end_sample: int
    segment_start_sec: float
    segment_end_sec: float
    segment_duration_sec: float
    original_fs: int
    lead_index: int
    lead_name: str
    preceding_rhythm: str
    is_normal_to_af: bool
    label: int = 1  # Pre-AF


@dataclass
class ExtractedSegment:
    """A fully extracted segment with signal data."""
    signal: np.ndarray
    metadata: ExtractableSegment


# =============================================================================
# Core Functions
# =============================================================================

def get_patient_id(record_id: str) -> str:
    """Generate unique patient ID for LTAFDB record."""
    return f"ltafdb_{record_id.zfill(3)}"


def load_record_info(data_dir: Path, record_id: str) -> PatientInfo:
    """Load patient/record information from LTAFDB."""
    record_path = str(data_dir / record_id)
    record = wfdb.rdrecord(record_path)

    return PatientInfo(
        patient_id=get_patient_id(record_id),
        source_database="LTAFDB",
        record_id=record_id,
        sampling_freq=record.fs,
        duration_sec=record.sig_len / record.fs,
        num_leads=record.n_sig,
        lead_names=record.sig_name
    )


def parse_rhythm_annotations(annotation: wfdb.Annotation) -> List[Dict]:
    """Extract rhythm change annotations from annotation object."""
    rhythm_changes = []
    fs = annotation.fs if hasattr(annotation, 'fs') else 128

    for sample, symbol, aux_note in zip(
        annotation.sample,
        annotation.symbol,
        annotation.aux_note
    ):
        if aux_note and aux_note.startswith('('):
            rhythm = RhythmType.from_annotation(aux_note)
            rhythm_changes.append({
                'sample': sample,
                'time_sec': sample / fs,
                'time_min': sample / fs / 60,
                'rhythm': rhythm,
                'rhythm_str': aux_note.strip('\x00'),
                'symbol': symbol,
            })

    return rhythm_changes


def identify_af_episodes(
    rhythm_changes: List[Dict],
    fs: int = 128
) -> List[AFEpisode]:
    """
    Identify AF episodes from rhythm changes.

    Only captures transitions TO AF (not AFL or other arrhythmias).
    Marks valid Normal→AF direct transitions.
    """
    af_episodes = []
    episode_count = 0

    for i, change in enumerate(rhythm_changes):
        if change['rhythm'] != RhythmType.AFIB:
            continue

        onset_sample = change['sample']
        onset_time = change['time_sec']

        # Get preceding rhythm info
        if i > 0:
            prev = rhythm_changes[i - 1]
            preceding_rhythm = prev['rhythm']
            preceding_start = prev['sample']
            preceding_duration = (onset_sample - preceding_start) / fs
        else:
            preceding_rhythm = RhythmType.UNKNOWN
            preceding_start = 0
            preceding_duration = onset_sample / fs

        # Get following rhythm (AF end)
        if i < len(rhythm_changes) - 1:
            next_change = rhythm_changes[i + 1]
            end_sample = next_change['sample']
            end_time = next_change['time_sec']
            following_rhythm = next_change['rhythm']
            duration = (end_sample - onset_sample) / fs
        else:
            end_sample = None
            end_time = None
            following_rhythm = RhythmType.UNKNOWN
            duration = None

        # Valid if directly preceded by Normal sinus rhythm
        is_valid = preceding_rhythm == RhythmType.NORMAL

        af_episodes.append(AFEpisode(
            episode_index=episode_count,
            onset_sample=onset_sample,
            onset_time_sec=onset_time,
            end_sample=end_sample,
            end_time_sec=end_time,
            duration_sec=duration,
            preceding_rhythm=preceding_rhythm,
            preceding_duration_sec=preceding_duration,
            following_rhythm=following_rhythm,
            is_valid_transition=is_valid
        ))
        episode_count += 1

    return af_episodes


def find_extractable_segments(
    patient_info: PatientInfo,
    af_episodes: List[AFEpisode],
    config: SegmentConfig
) -> List[ExtractableSegment]:
    """Find all segments that can be extracted from AF episodes."""
    extractable = []
    fs = patient_info.sampling_freq

    for episode in af_episodes:
        if config.require_normal_preceding and not episode.is_valid_transition:
            continue

        for lead_time_min in config.lead_times_min:
            lead_time_sec = lead_time_min * 60

            # Segment ends at lead_time before AF onset
            segment_end_sec = episode.onset_time_sec - lead_time_sec
            segment_start_sec = segment_end_sec - config.segment_duration_sec

            # Validation: segment must start after recording begins
            if segment_start_sec < 0:
                continue

            # Validation: segment must be within preceding rhythm
            min_required = lead_time_sec + config.segment_duration_sec
            if episode.preceding_duration_sec < min_required:
                continue

            # Calculate exact sample indices
            expected_samples = config.segment_duration_sec * fs
            segment_end_sample = int(segment_end_sec * fs)
            segment_start_sample = segment_end_sample - expected_samples

            for lead_idx in config.lead_indices:
                segment_id = (
                    f"ltafdb_{patient_info.record_id}_"
                    f"ep{episode.episode_index}_lt{lead_time_min}_l{lead_idx}"
                )

                lead_name = (
                    patient_info.lead_names[lead_idx]
                    if lead_idx < len(patient_info.lead_names)
                    else f"lead_{lead_idx}"
                )

                extractable.append(ExtractableSegment(
                    segment_id=segment_id,
                    patient_id=patient_info.patient_id,
                    source_database=patient_info.source_database,
                    record_id=patient_info.record_id,
                    episode_index=episode.episode_index,
                    lead_time_min=lead_time_min,
                    af_onset_sample=episode.onset_sample,
                    af_onset_time_sec=episode.onset_time_sec,
                    segment_start_sample=segment_start_sample,
                    segment_end_sample=segment_end_sample,
                    segment_start_sec=segment_start_sec,
                    segment_end_sec=segment_end_sec,
                    segment_duration_sec=config.segment_duration_sec,
                    original_fs=fs,
                    lead_index=lead_idx,
                    lead_name=lead_name,
                    preceding_rhythm=episode.preceding_rhythm.value,
                    is_normal_to_af=episode.is_valid_transition,
                    label=1
                ))

    return extractable


def extract_signal(data_dir: Path, segment: ExtractableSegment) -> np.ndarray:
    """Extract actual signal data for a segment."""
    record_path = str(data_dir / segment.record_id)
    record = wfdb.rdrecord(record_path)

    signal = record.p_signal[
        segment.segment_start_sample:segment.segment_end_sample,
        segment.lead_index
    ]

    return signal.astype(np.float32)


# =============================================================================
# Analysis and Extraction Pipeline
# =============================================================================

def analyze_record(
    data_dir: Path,
    record_id: str,
    config: SegmentConfig
) -> Tuple[PatientInfo, List[AFEpisode], List[ExtractableSegment]]:
    """Fully analyze a single record."""
    patient_info = load_record_info(data_dir, record_id)

    record_path = str(data_dir / record_id)
    annotation = wfdb.rdann(record_path, 'atr')
    rhythm_changes = parse_rhythm_annotations(annotation)

    af_episodes = identify_af_episodes(rhythm_changes, patient_info.sampling_freq)
    extractable = find_extractable_segments(patient_info, af_episodes, config)

    return patient_info, af_episodes, extractable


def analyze_all_records(
    data_dir: Path,
    config: SegmentConfig,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """Analyze all records in LTAFDB."""
    # Read record IDs from RECORDS file
    records_file = data_dir / 'RECORDS'
    if records_file.exists():
        records = [line.strip() for line in records_file.read_text().splitlines() if line.strip()]
    else:
        records = sorted([f.stem for f in data_dir.glob('*.hea')])

    if verbose:
        print(f"Found {len(records)} records in {data_dir}")

    all_segments = []
    summary = {
        'total_records': len(records),
        'records_with_af': 0,
        'total_af_episodes': 0,
        'normal_to_af_episodes': 0,
        'total_extractable_segments': 0,
        'by_lead_time': {lt: 0 for lt in config.lead_times_min},
        'by_patient': {}
    }

    iterator = tqdm(records, desc="Analyzing records") if verbose else records

    for record_id in iterator:
        try:
            patient_info, af_episodes, extractable = analyze_record(
                data_dir, record_id, config
            )

            if len(af_episodes) > 0:
                summary['records_with_af'] += 1

            all_segments.extend(extractable)

            summary['total_af_episodes'] += len(af_episodes)
            summary['normal_to_af_episodes'] += sum(
                1 for ep in af_episodes if ep.is_valid_transition
            )
            summary['total_extractable_segments'] += len(extractable)

            for seg in extractable:
                summary['by_lead_time'][seg.lead_time_min] += 1

            if len(extractable) > 0:
                summary['by_patient'][patient_info.patient_id] = {
                    'record_id': record_id,
                    'af_episodes': len(af_episodes),
                    'valid_transitions': sum(1 for ep in af_episodes if ep.is_valid_transition),
                    'extractable_segments': len(extractable)
                }

        except Exception as e:
            if verbose:
                print(f"\nError processing {record_id}: {e}")

    segments_df = pd.DataFrame([asdict(seg) for seg in all_segments])

    return segments_df, summary


def extract_all_signals(
    data_dir: Path,
    segments_df: pd.DataFrame,
    verbose: bool = True
) -> List[ExtractedSegment]:
    """Extract signals for all segments in DataFrame."""
    extracted = []
    errors = []

    grouped = segments_df.groupby('record_id')
    iterator = tqdm(grouped, desc="Extracting signals") if verbose else grouped

    for record_id, group in iterator:
        try:
            record_path = str(data_dir / record_id)
            record = wfdb.rdrecord(record_path)

            for _, row in group.iterrows():
                try:
                    signal = record.p_signal[
                        row['segment_start_sample']:row['segment_end_sample'],
                        row['lead_index']
                    ].astype(np.float32)

                    metadata = ExtractableSegment(**row.to_dict())
                    extracted.append(ExtractedSegment(signal=signal, metadata=metadata))

                except Exception as e:
                    errors.append({
                        'segment_id': row['segment_id'],
                        'error': str(e)
                    })

        except Exception as e:
            errors.append({
                'record_id': record_id,
                'error': f"Failed to load record: {e}"
            })

    if verbose and errors:
        print(f"\nEncountered {len(errors)} errors during extraction")
        for err in errors[:5]:
            print(f"  {err}")

    return extracted


# =============================================================================
# Output Functions
# =============================================================================

def save_to_hdf5(
    extracted: List[ExtractedSegment],
    output_path: Path,
    config: SegmentConfig
) -> None:
    """Save extracted segments to HDF5 format."""
    if not extracted:
        print("No segments to save!")
        return

    n_segments = len(extracted)
    expected_length = config.segment_duration_sec * config.target_fs

    signals = np.zeros((n_segments, expected_length), dtype=np.float32)

    segment_ids = []
    patient_ids = []
    record_ids = []
    source_databases = []
    episode_indices = []
    lead_times = []
    lead_indices = []
    lead_names = []
    labels = []
    original_fs = []
    preceding_rhythms = []
    is_normal_to_af = []

    for i, seg in enumerate(extracted):
        sig = seg.signal
        if len(sig) > expected_length:
            sig = sig[:expected_length]
        elif len(sig) < expected_length:
            sig = np.pad(sig, (0, expected_length - len(sig)), mode='edge')
        signals[i, :] = sig

        m = seg.metadata
        segment_ids.append(m.segment_id)
        patient_ids.append(m.patient_id)
        record_ids.append(m.record_id)
        source_databases.append(m.source_database)
        episode_indices.append(m.episode_index)
        lead_times.append(m.lead_time_min)
        lead_indices.append(m.lead_index)
        lead_names.append(m.lead_name)
        labels.append(m.label)
        original_fs.append(m.original_fs)
        preceding_rhythms.append(m.preceding_rhythm)
        is_normal_to_af.append(m.is_normal_to_af)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset(
            'signals', data=signals, dtype='float32',
            compression='gzip', compression_opts=4
        )

        f.create_dataset('labels', data=np.array(labels, dtype=np.int8))
        f.create_dataset('lead_times', data=np.array(lead_times, dtype=np.int16))
        f.create_dataset('episode_indices', data=np.array(episode_indices, dtype=np.int32))
        f.create_dataset('lead_indices', data=np.array(lead_indices, dtype=np.int8))
        f.create_dataset('original_fs', data=np.array(original_fs, dtype=np.int16))
        f.create_dataset('is_normal_to_af', data=np.array(is_normal_to_af, dtype=bool))

        dt_str = h5py.special_dtype(vlen=str)
        f.create_dataset('segment_ids', data=segment_ids, dtype=dt_str)
        f.create_dataset('patient_ids', data=patient_ids, dtype=dt_str)
        f.create_dataset('record_ids', data=record_ids, dtype=dt_str)
        f.create_dataset('source_databases', data=source_databases, dtype=dt_str)
        f.create_dataset('lead_names', data=lead_names, dtype=dt_str)
        f.create_dataset('preceding_rhythms', data=preceding_rhythms, dtype=dt_str)

        f.attrs['n_segments'] = n_segments
        f.attrs['segment_length'] = expected_length
        f.attrs['segment_duration_sec'] = config.segment_duration_sec
        f.attrs['lead_times_min'] = list(config.lead_times_min)
        f.attrs['target_fs'] = config.target_fs
        f.attrs['unique_patients'] = len(set(patient_ids))

    print(f"Saved {n_segments} segments to {output_path}")
    print(f"  Shape: ({n_segments}, {expected_length})")
    print(f"  Unique patients: {len(set(patient_ids))}")


def save_metadata_csv(segments_df: pd.DataFrame, output_path: Path) -> None:
    """Save segment metadata to CSV."""
    segments_df.to_csv(output_path, index=False)
    print(f"Saved metadata to {output_path}")


def print_summary(summary: Dict, segments_df: pd.DataFrame) -> None:
    """Print extraction summary."""
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)

    print(f"\nRecords analyzed: {summary['total_records']}")
    print(f"Records with AF: {summary['records_with_af']}")
    print(f"Total AF episodes: {summary['total_af_episodes']}")
    print(f"Normal→AF episodes: {summary['normal_to_af_episodes']}")
    print(f"Extractable segments: {summary['total_extractable_segments']}")

    print(f"\nSegments by lead time:")
    for lt, count in sorted(summary['by_lead_time'].items()):
        print(f"  {lt:2d} min: {count:4d} segments")

    if len(segments_df) > 0:
        print(f"\nUnique patients: {segments_df['patient_id'].nunique()}")

    print(f"\nPatients with extractable segments:")
    for pid, info in sorted(summary['by_patient'].items()):
        if info['extractable_segments'] > 0:
            print(f"  {pid}: {info['extractable_segments']} segments "
                  f"({info['valid_transitions']}/{info['af_episodes']} valid AF episodes)")


# =============================================================================
# Main Entry Point
# =============================================================================

def main(
    data_dir: str,
    output_dir: str,
    segment_duration: int = 10,
    lead_times: Tuple[int, ...] = (5, 10, 30, 45),
    lead_indices: Tuple[int, ...] = (0,),
    require_normal_preceding: bool = True
) -> pd.DataFrame:
    """
    Main extraction pipeline for LTAFDB.

    Parameters:
    -----------
    data_dir : str
        Path to LTAFDB directory
    output_dir : str
        Path to output directory
    segment_duration : int
        Segment duration in seconds (default: 10)
    lead_times : tuple
        Lead times in minutes before AF onset
    lead_indices : tuple
        Which leads to extract (0-indexed)
    require_normal_preceding : bool
        Only extract Normal→AF transitions

    Returns:
    --------
    pd.DataFrame : Segment metadata
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = SegmentConfig(
        segment_duration_sec=segment_duration,
        lead_times_min=lead_times,
        target_fs=128,  # LTAFDB native frequency
        lead_indices=lead_indices,
        require_normal_preceding=require_normal_preceding
    )

    print("LTAFDB Pre-AF Segment Extractor")
    print("=" * 70)
    print(f"Data directory: {data_path}")
    print(f"Output directory: {output_path}")
    print(f"Segment duration: {config.segment_duration_sec}s")
    print(f"Lead times: {config.lead_times_min} min")
    print(f"Leads: {config.lead_indices}")
    print(f"Sampling frequency: {config.target_fs} Hz")
    print(f"Normal→AF only: {config.require_normal_preceding}")
    print("=" * 70)

    # Step 1: Analyze all records
    print("\n[1/3] Analyzing records...")
    segments_df, summary = analyze_all_records(data_path, config)

    if len(segments_df) == 0:
        print("No extractable segments found!")
        return segments_df

    # Step 2: Extract signals
    print("\n[2/3] Extracting signals...")
    extracted = extract_all_signals(data_path, segments_df)

    # Step 3: Save outputs
    print("\n[3/3] Saving outputs...")
    save_to_hdf5(extracted, output_path / 'ltafdb_af_segments.h5', config)
    save_metadata_csv(segments_df, output_path / 'ltafdb_af_segments_metadata.csv')

    with open(output_path / 'ltafdb_af_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print_summary(summary, segments_df)

    return segments_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract pre-AF segments from LTAFDB'
    )
    parser.add_argument(
        '--data_dir',
        default='/Users/filipsulich/Developer/uni/Thesis/Thesis/data/ltafdb',
        help='Path to LTAFDB directory'
    )
    parser.add_argument(
        '--output_dir',
        default='/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/ltafdb',
        help='Path to output directory'
    )
    parser.add_argument(
        '--segment_duration',
        type=int,
        default=10,
        help='Segment duration in seconds'
    )
    parser.add_argument(
        '--lead_times',
        type=int,
        nargs='+',
        default=[5, 10, 15, 30, 45],
        help='Lead times in minutes before AF onset'
    )
    parser.add_argument(
        '--leads',
        type=int,
        nargs='+',
        default=[0],
        help='Lead indices to extract (0-indexed)'
    )
    parser.add_argument(
        '--include_non_normal',
        action='store_true',
        help='Include non-Normal→AF transitions'
    )

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        segment_duration=args.segment_duration,
        lead_times=tuple(args.lead_times),
        lead_indices=tuple(args.leads),
        require_normal_preceding=not args.include_non_normal
    )